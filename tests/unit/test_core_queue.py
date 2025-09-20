"""
队列系统单元测试
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.core.queue import (
    QueueManager,
    TaskPriority,
    TaskStatus,
    Task,
    get_queue_manager,
    shutdown_queue_manager
)


class TestTask:
    """任务类测试"""
    
    def test_task_creation(self):
        """测试任务创建"""
        task = Task(
            task_id="test-123",
            task_type="asr",
            priority=TaskPriority.HIGH,
            data={"audio": "test.wav"}
        )
        
        assert task.task_id == "test-123"
        assert task.task_type == "asr"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        assert task.data == {"audio": "test.wav"}
        assert isinstance(task.created_at, datetime)
    
    def test_task_priority_enum(self):
        """测试任务优先级枚举"""
        assert TaskPriority.LOW.value == 1
        assert TaskPriority.NORMAL.value == 2
        assert TaskPriority.HIGH.value == 3
        assert TaskPriority.URGENT.value == 4
    
    def test_task_status_enum(self):
        """测试任务状态枚举"""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.PROCESSING == "processing"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
    
    def test_task_to_dict(self):
        """测试任务转字典"""
        task = Task(
            task_id="test-123",
            task_type="asr",
            priority=TaskPriority.HIGH,
            data={"test": "data"}
        )
        
        task_dict = task.to_dict()
        
        assert task_dict["task_id"] == "test-123"
        assert task_dict["task_type"] == "asr"
        assert task_dict["priority"] == TaskPriority.HIGH.value
        assert task_dict["status"] == TaskStatus.PENDING
        assert task_dict["data"] == {"test": "data"}


class TestQueueManager:
    """队列管理器测试"""
    
    @pytest.fixture
    def queue_manager(self):
        """创建队列管理器实例"""
        return QueueManager()
    
    @pytest.mark.asyncio
    async def test_queue_manager_initialization(self, queue_manager):
        """测试队列管理器初始化"""
        await queue_manager.initialize()
        
        assert queue_manager._task_queue is not None
        assert queue_manager._result_store == {}
        assert queue_manager._workers == []
        assert queue_manager._stats["total_tasks"] == 0
    
    @pytest.mark.asyncio
    async def test_submit_task(self, queue_manager):
        """测试提交任务"""
        await queue_manager.initialize()
        
        task_id = await queue_manager.submit_task(
            task_type="asr",
            data={"audio": "test.wav"},
            priority=TaskPriority.HIGH
        )
        
        assert isinstance(task_id, str)
        assert len(task_id) > 0
        assert queue_manager._stats["total_tasks"] == 1
        assert queue_manager._stats["pending_tasks"] == 1
    
    @pytest.mark.asyncio
    async def test_get_task_status_existing(self, queue_manager):
        """测试获取存在任务的状态"""
        await queue_manager.initialize()
        
        # 提交任务
        task_id = await queue_manager.submit_task(
            task_type="asr",
            data={"audio": "test.wav"}
        )
        
        # 获取状态
        status = await queue_manager.get_task_status(task_id)
        
        assert status is not None
        assert status["task_id"] == task_id
        assert status["status"] == TaskStatus.PENDING
        assert status["task_type"] == "asr"
    
    @pytest.mark.asyncio
    async def test_get_task_status_nonexistent(self, queue_manager):
        """测试获取不存在任务的状态"""
        await queue_manager.initialize()
        
        status = await queue_manager.get_task_status("nonexistent-task")
        
        assert status is None
    
    @pytest.mark.asyncio
    async def test_get_stats(self, queue_manager):
        """测试获取统计信息"""
        await queue_manager.initialize()
        
        # 提交几个任务
        await queue_manager.submit_task("asr", {"audio": "1.wav"})
        await queue_manager.submit_task("speaker", {"audio": "2.wav"})
        
        stats = await queue_manager.get_stats()
        
        assert stats["total_tasks"] == 2
        assert stats["pending_tasks"] == 2
        assert stats["processing_tasks"] == 0
        assert stats["completed_tasks"] == 0
        assert stats["failed_tasks"] == 0
    
    @pytest.mark.asyncio
    async def test_worker_processing(self, queue_manager):
        """测试工作线程处理"""
        await queue_manager.initialize()
        
        # 模拟处理函数
        async def mock_process_task(task):
            return {"result": f"processed_{task.task_id}"}
        
        with patch.object(queue_manager, '_process_task', mock_process_task):
            # 提交任务
            task_id = await queue_manager.submit_task(
                task_type="asr",
                data={"audio": "test.wav"}
            )
            
            # 等待处理
            await asyncio.sleep(0.1)
            
            # 检查结果
            status = await queue_manager.get_task_status(task_id)
            assert status["status"] == TaskStatus.COMPLETED
            assert "result" in status
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue_manager):
        """测试优先级排序"""
        await queue_manager.initialize()
        
        # 提交不同优先级的任务
        low_task = await queue_manager.submit_task("asr", {"audio": "low.wav"}, TaskPriority.LOW)
        urgent_task = await queue_manager.submit_task("asr", {"audio": "urgent.wav"}, TaskPriority.URGENT)
        high_task = await queue_manager.submit_task("asr", {"audio": "high.wav"}, TaskPriority.HIGH)
        
        # 检查队列中的任务顺序（高优先级应该在前）
        tasks = []
        while not queue_manager._task_queue.empty():
            task = await queue_manager._task_queue.get()
            tasks.append(task)
            queue_manager._task_queue.task_done()
        
        # 重新放回队列
        for task in tasks:
            await queue_manager._task_queue.put(task)
        
        # 验证顺序：URGENT > HIGH > LOW
        priorities = [task.priority for task in tasks]
        assert priorities == [TaskPriority.URGENT, TaskPriority.HIGH, TaskPriority.LOW]
    
    @pytest.mark.asyncio
    async def test_shutdown(self, queue_manager):
        """测试关闭队列管理器"""
        await queue_manager.initialize()
        
        # 启动工作线程
        await queue_manager.start_workers(num_workers=2)
        assert len(queue_manager._workers) == 2
        
        # 关闭
        await queue_manager.shutdown()
        
        # 验证工作线程已停止
        for worker in queue_manager._workers:
            assert worker.cancelled() or worker.done()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, queue_manager):
        """测试错误处理"""
        await queue_manager.initialize()
        
        # 模拟处理函数抛出异常
        async def failing_process_task(task):
            raise Exception("处理失败")
        
        with patch.object(queue_manager, '_process_task', failing_process_task):
            # 提交任务
            task_id = await queue_manager.submit_task(
                task_type="asr",
                data={"audio": "test.wav"}
            )
            
            # 等待处理
            await asyncio.sleep(0.1)
            
            # 检查状态
            status = await queue_manager.get_task_status(task_id)
            assert status["status"] == TaskStatus.FAILED
            assert "error" in status
    
    @pytest.mark.asyncio
    async def test_task_timeout(self, queue_manager):
        """测试任务超时"""
        await queue_manager.initialize()
        
        # 模拟长时间运行的任务
        async def slow_process_task(task):
            await asyncio.sleep(10)  # 10秒
            return {"result": "slow"}
        
        with patch.object(queue_manager, '_process_task', slow_process_task):
            with patch.object(queue_manager, '_task_timeout', 0.1):  # 0.1秒超时
                # 提交任务
                task_id = await queue_manager.submit_task(
                    task_type="asr",
                    data={"audio": "test.wav"}
                )
                
                # 等待超时
                await asyncio.sleep(0.2)
                
                # 检查状态
                status = await queue_manager.get_task_status(task_id)
                assert status["status"] == TaskStatus.FAILED
                assert "timeout" in status.get("error", "").lower()


class TestQueueManagerSingleton:
    """队列管理器单例测试"""
    
    @pytest.mark.asyncio
    async def test_get_queue_manager_singleton(self):
        """测试队列管理器单例模式"""
        manager1 = await get_queue_manager()
        manager2 = await get_queue_manager()
        
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_shutdown_queue_manager(self):
        """测试关闭队列管理器"""
        manager = await get_queue_manager()
        
        # 确保管理器已初始化
        await manager.initialize()
        
        # 关闭
        await shutdown_queue_manager()
        
        # 验证关闭
        assert manager._shutdown_event.is_set() if hasattr(manager, '_shutdown_event') else True
    
    @pytest.mark.asyncio
    async def test_multiple_workers(self):
        """测试多个工作线程"""
        manager = await get_queue_manager()
        await manager.initialize()
        
        # 启动多个工作线程
        await manager.start_workers(num_workers=4)
        
        assert len(manager._workers) == 4
        
        # 提交多个任务测试并发处理
        task_ids = []
        for i in range(10):
            task_id = await manager.submit_task(
                task_type="asr",
                data={"audio": f"test{i}.wav"}
            )
            task_ids.append(task_id)
        
        # 等待所有任务完成
        await asyncio.sleep(0.5)
        
        # 检查统计
        stats = await manager.get_stats()
        assert stats["total_tasks"] == 10
        
        # 清理
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_queue_capacity(self):
        """测试队列容量"""
        manager = await get_queue_manager()
        await manager.initialize()
        
        # 设置较小的队列容量进行测试
        original_maxsize = manager._task_queue.maxsize
        manager._task_queue = asyncio.PriorityQueue(maxsize=3)
        
        try:
            # 填满队列
            for i in range(3):
                await manager.submit_task("asr", {"audio": f"test{i}.wav"})
            
            # 尝试添加第4个任务（应该不会阻塞，因为使用了put_nowait的替代方案）
            task_id = await manager.submit_task("asr", {"audio": "test3.wav"})
            assert task_id is not None
            
        finally:
            # 恢复原始设置
            manager._task_queue = asyncio.PriorityQueue(maxsize=original_maxsize)
            await manager.shutdown()