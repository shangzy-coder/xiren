"""
队列系统集成示例API

展示如何在API端点中使用并发队列系统
"""

from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from pydantic import BaseModel
from typing import Optional, Any, Dict
import logging
import time
import asyncio

from app.core.request_manager import get_request_manager, TaskPriority
from app.core.queue import TaskType

logger = logging.getLogger(__name__)

router = APIRouter()


class TaskSubmissionResponse(BaseModel):
    """任务提交响应"""
    success: bool
    task_id: str
    message: str
    estimated_completion_time: Optional[float] = None


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


# 示例：CPU密集型任务函数
def cpu_intensive_task(data: str, iterations: int = 1000000) -> dict:
    """模拟CPU密集型任务"""
    start_time = time.time()
    
    # 模拟复杂计算
    result = 0
    for i in range(iterations):
        result += len(data) * i
    
    processing_time = time.time() - start_time
    
    return {
        "processed_data": data[:100] + "..." if len(data) > 100 else data,
        "result": result,
        "iterations": iterations,
        "processing_time": processing_time
    }


# 示例：IO密集型任务函数  
async def io_intensive_task(url: str, delay: float = 1.0) -> dict:
    """模拟IO密集型任务"""
    start_time = time.time()
    
    # 模拟网络请求或文件IO
    await asyncio.sleep(delay)
    
    processing_time = time.time() - start_time
    
    return {
        "url": url,
        "delay": delay,
        "processing_time": processing_time,
        "timestamp": start_time
    }


@router.post("/submit/cpu-task", response_model=TaskSubmissionResponse)
async def submit_cpu_task(
    data: str = Form(..., description="要处理的数据"),
    iterations: int = Form(default=1000000, description="计算迭代次数"),
    priority: str = Form(default="normal", description="任务优先级: low, normal, high, urgent")
):
    """
    提交CPU密集型任务到队列
    
    这个端点展示了如何将CPU密集型任务提交到队列系统，
    避免阻塞API响应。
    """
    try:
        # 解析优先级
        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL, 
            "high": TaskPriority.HIGH,
            "urgent": TaskPriority.URGENT
        }
        task_priority = priority_map.get(priority.lower(), TaskPriority.NORMAL)
        
        # 获取请求管理器
        request_manager = await get_request_manager()
        
        # 提交任务
        task_id = await request_manager.submit_asr_request(
            func=cpu_intensive_task,
            args=(data, iterations),
            priority=task_priority,
            timeout=300  # 5分钟超时
        )
        
        # 估算完成时间（基于迭代次数）
        estimated_time = iterations / 100000  # 简单估算
        
        return TaskSubmissionResponse(
            success=True,
            task_id=task_id,
            message=f"CPU任务已提交到队列，优先级: {priority}",
            estimated_completion_time=estimated_time
        )
        
    except Exception as e:
        logger.error(f"提交CPU任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交任务失败: {str(e)}")


@router.post("/submit/io-task", response_model=TaskSubmissionResponse)
async def submit_io_task(
    url: str = Form(..., description="要处理的URL"),
    delay: float = Form(default=1.0, description="模拟延迟时间(秒)"),
    priority: str = Form(default="normal", description="任务优先级")
):
    """
    提交IO密集型任务到队列
    
    展示如何处理异步IO任务
    """
    try:
        # 解析优先级
        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH, 
            "urgent": TaskPriority.URGENT
        }
        task_priority = priority_map.get(priority.lower(), TaskPriority.NORMAL)
        
        # 获取请求管理器
        request_manager = await get_request_manager()
        
        # 提交任务
        task_id = await request_manager.submit_speaker_request(
            func=io_intensive_task,
            args=(url, delay),
            priority=task_priority,
            timeout=60  # 1分钟超时
        )
        
        return TaskSubmissionResponse(
            success=True,
            task_id=task_id,
            message=f"IO任务已提交到队列，优先级: {priority}",
            estimated_completion_time=delay + 1.0  # 延迟时间加上处理时间
        )
        
    except Exception as e:
        logger.error(f"提交IO任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交任务失败: {str(e)}")


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    获取任务状态
    
    客户端可以轮询此端点来检查任务状态和获取结果
    """
    try:
        # 获取请求管理器
        request_manager = await get_request_manager()
        
        # 查询任务状态
        result = await request_manager.get_request_status(task_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return TaskStatusResponse(
            task_id=task_id,
            status=result.status.value,
            result=result.result,
            error=result.error,
            execution_time=result.execution_time,
            created_at=result.created_at,
            started_at=result.started_at,
            completed_at=result.completed_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.get("/result/{task_id}")
async def get_task_result(task_id: str, timeout: float = 30.0):
    """
    等待并获取任务结果
    
    这个端点会阻塞等待任务完成，适合需要同步响应的场景
    """
    try:
        # 获取请求管理器
        request_manager = await get_request_manager()
        
        # 等待任务完成
        result = await request_manager.get_request_result(task_id, timeout)
        
        if result.status.value == 'completed':
            return {
                "success": True,
                "task_id": task_id,
                "result": result.result,
                "execution_time": result.execution_time
            }
        elif result.status.value == 'failed':
            raise HTTPException(
                status_code=500, 
                detail=f"任务执行失败: {result.error}"
            )
        else:
            raise HTTPException(
                status_code=408,
                detail=f"任务未完成: {result.status.value}"
            )
            
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="等待任务结果超时")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取结果失败: {str(e)}")


@router.delete("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """取消任务"""
    try:
        # 获取请求管理器
        request_manager = await get_request_manager()
        
        # 取消任务
        success = await request_manager.cancel_request(task_id)
        
        if success:
            return {
                "success": True,
                "message": f"任务 {task_id} 已取消"
            }
        else:
            raise HTTPException(status_code=404, detail="任务不存在或无法取消")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"取消失败: {str(e)}")


@router.get("/queue/stats")
async def get_queue_stats():
    """获取队列统计信息"""
    try:
        # 获取请求管理器
        request_manager = await get_request_manager()
        
        # 获取统计信息
        stats = request_manager.get_stats()
        
        return {
            "success": True,
            "stats": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"获取队列统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@router.get("/queue/health")
async def get_queue_health():
    """获取队列健康状态"""
    try:
        # 获取请求管理器  
        request_manager = await get_request_manager()
        
        # 获取健康状态
        health = request_manager.get_health_status()
        
        return {
            "success": True,
            "health": health,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"获取队列健康状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取健康状态失败: {str(e)}")
