"""
并发请求管理器

管理和协调来自API的请求，使用队列系统进行负载均衡和并发控制
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Callable
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import RLock

from app.config import settings
from app.core.queue import QueueManager, TaskType, TaskPriority, TaskResult, get_queue_manager

logger = logging.getLogger(__name__)


class RequestLimiter:
    """请求限制器"""
    
    def __init__(self, max_concurrent: int = None):
        self.max_concurrent = max_concurrent or settings.MAX_CONCURRENT_REQUESTS
        self._active_requests = 0
        self._lock = RLock()
        self._condition = asyncio.Condition()
    
    async def acquire(self):
        """获取请求许可"""
        async with self._condition:
            while self._active_requests >= self.max_concurrent:
                await self._condition.wait()
            
            with self._lock:
                self._active_requests += 1
    
    async def release(self):
        """释放请求许可"""
        async with self._condition:
            with self._lock:
                self._active_requests = max(0, self._active_requests - 1)
            self._condition.notify()
    
    def current_load(self) -> float:
        """当前负载百分比"""
        with self._lock:
            return (self._active_requests / self.max_concurrent) * 100


class RequestManager:
    """请求管理器"""
    
    def __init__(self):
        self._queue_manager: Optional[QueueManager] = None
        self._limiter = RequestLimiter()
        self._request_cache: Dict[str, Any] = {}
        self._cache_lock = RLock()
        
        # 请求统计
        self._stats = {
            "total_requests": 0,
            "active_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "request_types": {task_type.value: 0 for task_type in TaskType}
        }
        self._stats_lock = RLock()
    
    async def initialize(self):
        """初始化请求管理器"""
        try:
            self._queue_manager = await get_queue_manager()
            logger.info("请求管理器初始化成功")
            return True
        except Exception as e:
            logger.error(f"请求管理器初始化失败: {e}")
            return False
    
    async def submit_asr_request(self,
                               func: Callable,
                               args: tuple = (),
                               kwargs: dict = None,
                               priority: TaskPriority = TaskPriority.NORMAL,
                               timeout: Optional[float] = None) -> str:
        """
        提交ASR请求
        
        Args:
            func: 执行函数
            args: 函数参数
            kwargs: 函数关键字参数
            priority: 优先级
            timeout: 超时时间
            
        Returns:
            任务ID
        """
        return await self._submit_request(
            TaskType.ASR, func, args, kwargs, priority, timeout
        )
    
    async def submit_speaker_request(self,
                                   func: Callable,
                                   args: tuple = (),
                                   kwargs: dict = None,
                                   priority: TaskPriority = TaskPriority.NORMAL,
                                   timeout: Optional[float] = None) -> str:
        """提交声纹识别请求"""
        return await self._submit_request(
            TaskType.SPEAKER_ID, func, args, kwargs, priority, timeout
        )
    
    async def submit_comprehensive_request(self,
                                         func: Callable,
                                         args: tuple = (),
                                         kwargs: dict = None,
                                         priority: TaskPriority = TaskPriority.NORMAL,
                                         timeout: Optional[float] = None) -> str:
        """提交综合处理请求"""
        return await self._submit_request(
            TaskType.COMPREHENSIVE, func, args, kwargs, priority, timeout
        )
    
    async def submit_batch_request(self,
                                 func: Callable,
                                 args: tuple = (),
                                 kwargs: dict = None,
                                 priority: TaskPriority = TaskPriority.LOW,
                                 timeout: Optional[float] = None) -> str:
        """提交批量处理请求"""
        return await self._submit_request(
            TaskType.BATCH, func, args, kwargs, priority, timeout
        )
    
    async def _submit_request(self,
                            task_type: TaskType,
                            func: Callable,
                            args: tuple,
                            kwargs: dict,
                            priority: TaskPriority,
                            timeout: Optional[float]) -> str:
        """内部请求提交方法"""
        # 检查并发限制
        await self._limiter.acquire()
        
        try:
            # 更新统计
            with self._stats_lock:
                self._stats["total_requests"] += 1
                self._stats["active_requests"] += 1
                self._stats["request_types"][task_type.value] += 1
            
            # 提交到队列
            task_id = await self._queue_manager.submit_task(
                task_type=task_type,
                func=func,
                args=args,
                kwargs=kwargs or {},
                priority=priority,
                timeout=timeout or settings.TASK_TIMEOUT,
                callback=self._task_completed_callback
            )
            
            logger.info(f"请求已提交到队列: {task_id} ({task_type.value})")
            return task_id
            
        except Exception as e:
            # 释放并发限制
            await self._limiter.release()
            
            # 更新统计
            with self._stats_lock:
                self._stats["active_requests"] -= 1
                self._stats["failed_requests"] += 1
            
            logger.error(f"提交请求失败: {e}")
            raise
    
    async def get_request_result(self, task_id: str, timeout: float = None) -> TaskResult:
        """
        获取请求结果（阻塞等待）
        
        Args:
            task_id: 任务ID
            timeout: 等待超时时间
            
        Returns:
            任务结果
        """
        start_time = time.time()
        timeout = timeout or settings.TASK_TIMEOUT
        
        while True:
            result = await self._queue_manager.get_task_status(task_id)
            
            if result and result.status.value in ['completed', 'failed', 'cancelled']:
                return result
            
            # 检查超时
            if time.time() - start_time > timeout:
                # 取消任务
                await self._queue_manager.cancel_task(task_id)
                raise asyncio.TimeoutError(f"等待任务结果超时: {task_id}")
            
            # 短暂等待
            await asyncio.sleep(0.1)
    
    async def get_request_status(self, task_id: str) -> Optional[TaskResult]:
        """获取请求状态（非阻塞）"""
        return await self._queue_manager.get_task_status(task_id)
    
    async def cancel_request(self, task_id: str) -> bool:
        """取消请求"""
        return await self._queue_manager.cancel_task(task_id)
    
    def _task_completed_callback(self, result: TaskResult):
        """任务完成回调"""
        try:
            # 更新统计
            with self._stats_lock:
                self._stats["active_requests"] -= 1
                if result.status.value == 'completed':
                    self._stats["completed_requests"] += 1
                    
                    # 更新平均响应时间
                    if result.execution_time:
                        total_completed = self._stats["completed_requests"]
                        current_avg = self._stats["average_response_time"]
                        self._stats["average_response_time"] = (
                            (current_avg * (total_completed - 1) + result.execution_time) / total_completed
                        )
                else:
                    self._stats["failed_requests"] += 1
            
            # 释放并发限制
            asyncio.create_task(self._limiter.release())
            
        except Exception as e:
            logger.error(f"任务完成回调错误: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取请求统计信息"""
        with self._stats_lock:
            stats = self._stats.copy()
            stats["current_load_percent"] = self._limiter.current_load()
            stats["queue_metrics"] = self._queue_manager.get_metrics() if self._queue_manager else {}
            return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        stats = self.get_stats()
        
        # 计算健康分数
        load_percent = stats["current_load_percent"]
        success_rate = 0
        if stats["total_requests"] > 0:
            success_rate = stats["completed_requests"] / stats["total_requests"] * 100
        
        health_score = 100
        
        # 根据负载降低健康分数
        if load_percent > 90:
            health_score -= 30
        elif load_percent > 70:
            health_score -= 15
        elif load_percent > 50:
            health_score -= 5
        
        # 根据成功率降低健康分数
        if success_rate < 50:
            health_score -= 40
        elif success_rate < 70:
            health_score -= 20
        elif success_rate < 90:
            health_score -= 10
        
        # 判断健康状态
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "health_score": max(0, health_score),
            "load_percent": load_percent,
            "success_rate": success_rate,
            "active_requests": stats["active_requests"],
            "queue_size": stats["queue_metrics"].get("queue_size", 0),
            "average_response_time": stats["average_response_time"]
        }


# 全局请求管理器实例
_request_manager: Optional[RequestManager] = None


async def get_request_manager() -> RequestManager:
    """获取请求管理器实例"""
    global _request_manager
    if _request_manager is None:
        _request_manager = RequestManager()
        await _request_manager.initialize()
    return _request_manager


# 装饰器：自动使用请求管理器
def with_request_manager(task_type: TaskType, priority: TaskPriority = TaskPriority.NORMAL):
    """
    装饰器：自动将函数提交到请求管理器
    
    Args:
        task_type: 任务类型
        priority: 优先级
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = await get_request_manager()
            
            # 提交任务
            task_id = await manager._submit_request(
                task_type=task_type,
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                timeout=settings.TASK_TIMEOUT
            )
            
            # 等待结果
            result = await manager.get_request_result(task_id)
            
            if result.status.value == 'completed':
                return result.result
            elif result.status.value == 'failed':
                raise Exception(result.error)
            else:
                raise Exception(f"任务未完成: {result.status.value}")
        
        return wrapper
    return decorator
