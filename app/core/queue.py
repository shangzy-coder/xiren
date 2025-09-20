"""
并发处理队列系统

实现基于asyncio.Queue和ThreadPoolExecutor的异步任务队列管理
支持不同优先级、任务类型和并发控制
"""

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from threading import RLock
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """任务类型枚举"""
    ASR = "asr"                    # 语音识别
    SPEAKER_ID = "speaker_id"      # 声纹识别
    DIARIZATION = "diarization"    # 说话人分离
    COMPREHENSIVE = "comprehensive" # 综合处理
    BATCH = "batch"                # 批量处理
    
    # 流水线任务类型
    PREPROCESSING = "preprocessing"  # 音频预处理
    VAD = "vad"                     # 语音活动检测
    SPEAKER_EMBEDDING = "speaker_embedding"      # 声纹特征提取
    SPEAKER_IDENTIFICATION = "speaker_identification"  # 声纹识别
    SPEAKER_DIARIZATION = "speaker_diarization"  # 说话人分离
    POSTPROCESSING = "postprocessing"  # 后处理


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class QueueTask:
    """队列任务"""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    callback: Optional[Callable] = None
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.created_at is None:
            self.created_at = time.time()
    
    def __lt__(self, other):
        """优先级排序：高优先级任务在前"""
        if isinstance(other, QueueTask):
            return self.priority.value > other.priority.value
        return False


class QueueManager:
    """队列管理器"""
    
    def __init__(self, 
                 max_workers: int = None,
                 max_queue_size: int = 1000,
                 enable_metrics: bool = True):
        """
        初始化队列管理器
        
        Args:
            max_workers: 线程池最大工作线程数
            max_queue_size: 队列最大容量
            enable_metrics: 是否启用指标统计
        """
        self.max_workers = max_workers or settings.THREAD_POOL_SIZE
        self.max_queue_size = max_queue_size
        self.enable_metrics = enable_metrics
        
        # 队列和执行器
        self._queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # 任务状态管理
        self._tasks: Dict[str, QueueTask] = {}
        self._results: Dict[str, TaskResult] = {}
        self._active_tasks: Dict[str, asyncio.Task] = {}
        
        # 同步锁
        self._lock = RLock()
        
        # 统计指标
        self._metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "cancelled_tasks": 0,
            "queue_size": 0,
            "active_tasks": 0,
            "average_execution_time": 0.0,
            "task_type_stats": {task_type.value: 0 for task_type in TaskType}
        }
        
        # 运行状态
        self._running = False
        self._worker_tasks = []
        
    async def initialize(self) -> bool:
        """初始化队列管理器"""
        try:
            self._running = True
            
            # 启动工作协程
            for i in range(min(self.max_workers, 4)):  # 限制异步工作协程数量
                worker_task = asyncio.create_task(
                    self._worker(f"worker-{i}")
                )
                self._worker_tasks.append(worker_task)
            
            logger.info(f"队列管理器初始化成功，工作线程: {self.max_workers}, 工作协程: {len(self._worker_tasks)}")
            return True
            
        except Exception as e:
            logger.error(f"队列管理器初始化失败: {e}")
            return False
    
    async def shutdown(self):
        """关闭队列管理器"""
        self._running = False
        
        # 取消所有活动任务
        for task_id, task in self._active_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"取消任务: {task_id}")
        
        # 等待工作协程完成
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # 关闭线程池
        self._thread_pool.shutdown(wait=True)
        
        logger.info("队列管理器已关闭")
    
    async def submit_task(self,
                         task_type: TaskType,
                         func: Callable,
                         args: tuple = (),
                         kwargs: dict = None,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         callback: Optional[Callable] = None,
                         timeout: Optional[float] = None,
                         max_retries: int = 3) -> str:
        """
        提交任务到队列
        
        Args:
            task_type: 任务类型
            func: 要执行的函数
            args: 函数参数
            kwargs: 函数关键字参数
            priority: 任务优先级
            callback: 完成回调函数
            timeout: 超时时间
            max_retries: 最大重试次数
            
        Returns:
            任务ID
        """
        if not self._running:
            raise RuntimeError("队列管理器未运行")
        
        if self._queue.qsize() >= self.max_queue_size:
            raise RuntimeError("队列已满，无法提交新任务")
        
        # 创建任务
        task_id = str(uuid.uuid4())
        task = QueueTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            func=func,
            args=args,
            kwargs=kwargs or {},
            callback=callback,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # 添加到队列
        with self._lock:
            self._tasks[task_id] = task
            self._metrics["total_tasks"] += 1
            self._metrics["task_type_stats"][task_type.value] += 1
        
        # 使用优先级元组：(负优先级值, 创建时间, 任务)
        # 负优先级值确保高优先级(大数值)排在前面
        await self._queue.put((-priority.value, task.created_at, task))
        
        logger.info(f"任务已提交: {task_id} ({task_type.value}, 优先级: {priority.value})")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """获取任务状态"""
        with self._lock:
            return self._results.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            # 检查任务是否在活动任务中
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                if not task.done():
                    task.cancel()
                    self._results[task_id] = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.CANCELLED,
                        completed_at=time.time()
                    )
                    self._metrics["cancelled_tasks"] += 1
                    logger.info(f"任务已取消: {task_id}")
                    return True
            
            # 检查任务是否在待处理队列中
            if task_id in self._tasks:
                self._results[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.CANCELLED,
                    completed_at=time.time()
                )
                self._metrics["cancelled_tasks"] += 1
                logger.info(f"待处理任务已标记为取消: {task_id}")
                return True
            
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取队列指标"""
        with self._lock:
            metrics = self._metrics.copy()
            metrics["queue_size"] = self._queue.qsize()
            metrics["active_tasks"] = len(self._active_tasks)
            return metrics
    
    async def _worker(self, worker_name: str):
        """工作协程"""
        logger.info(f"工作协程启动: {worker_name}")
        
        while self._running:
            try:
                # 获取任务（带超时避免永久阻塞）
                try:
                    priority, created_at, task = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 检查任务是否已被取消
                if task.task_id in self._results:
                    result = self._results[task.task_id]
                    if result.status == TaskStatus.CANCELLED:
                        continue
                
                # 执行任务
                await self._execute_task(task)
                
            except Exception as e:
                logger.error(f"工作协程 {worker_name} 错误: {e}")
                await asyncio.sleep(0.1)
        
        logger.info(f"工作协程关闭: {worker_name}")
    
    async def _execute_task(self, task: QueueTask):
        """执行任务"""
        task_id = task.task_id
        start_time = time.time()
        
        # 记录任务开始
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            created_at=task.created_at,
            started_at=start_time
        )
        
        with self._lock:
            self._results[task_id] = result
        
        try:
            # 创建执行任务
            if asyncio.iscoroutinefunction(task.func):
                # 异步函数
                exec_task = asyncio.create_task(
                    task.func(*task.args, **task.kwargs)
                )
            else:
                # 同步函数，使用线程池
                loop = asyncio.get_event_loop()
                exec_task = loop.run_in_executor(
                    self._thread_pool,
                    lambda: task.func(*task.args, **task.kwargs)
                )
            
            # 记录活动任务
            with self._lock:
                self._active_tasks[task_id] = exec_task
            
            # 执行任务（带超时）
            if task.timeout:
                task_result = await asyncio.wait_for(exec_task, timeout=task.timeout)
            else:
                task_result = await exec_task
            
            # 记录成功结果
            execution_time = time.time() - start_time
            result.status = TaskStatus.COMPLETED
            result.result = task_result
            result.execution_time = execution_time
            result.completed_at = time.time()
            
            with self._lock:
                self._results[task_id] = result
                self._metrics["completed_tasks"] += 1
                
                # 更新平均执行时间
                total_completed = self._metrics["completed_tasks"]
                current_avg = self._metrics["average_execution_time"]
                self._metrics["average_execution_time"] = (
                    (current_avg * (total_completed - 1) + execution_time) / total_completed
                )
            
            # 调用回调
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(result)
                    else:
                        task.callback(result)
                except Exception as e:
                    logger.error(f"任务回调失败 {task_id}: {e}")
            
            logger.info(f"任务完成: {task_id}, 耗时: {execution_time:.2f}秒")
            
        except asyncio.CancelledError:
            # 任务被取消
            result.status = TaskStatus.CANCELLED
            result.completed_at = time.time()
            with self._lock:
                self._results[task_id] = result
                self._metrics["cancelled_tasks"] += 1
            logger.info(f"任务被取消: {task_id}")
            
        except Exception as e:
            # 任务执行失败
            execution_time = time.time() - start_time
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.execution_time = execution_time
            result.completed_at = time.time()
            
            with self._lock:
                self._results[task_id] = result
                self._metrics["failed_tasks"] += 1
            
            # 重试逻辑
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.warning(f"任务失败，准备重试 {task.retry_count}/{task.max_retries}: {task_id}")
                
                # 重新提交任务
                await asyncio.sleep(min(2 ** task.retry_count, 10))  # 指数退避
                await self._queue.put((-task.priority.value, time.time(), task))
            else:
                logger.error(f"任务执行失败: {task_id}, 错误: {e}")
        
        finally:
            # 清理活动任务记录
            with self._lock:
                self._active_tasks.pop(task_id, None)


# 全局队列管理器实例
_queue_manager: Optional[QueueManager] = None


async def get_queue_manager() -> QueueManager:
    """获取队列管理器实例"""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager()
        await _queue_manager.initialize()
    return _queue_manager


async def shutdown_queue_manager():
    """关闭队列管理器"""
    global _queue_manager
    if _queue_manager:
        await _queue_manager.shutdown()
        _queue_manager = None
