"""
语音处理流水线API

提供基于队列的分阶段语音处理接口
每个处理步骤（VAD、ASR、Speaker识别）都是独立的队列任务
"""

from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from pydantic import BaseModel
from typing import Optional, Any, Dict, List
import logging
import time
import asyncio

from app.core.pipeline import get_pipeline_orchestrator, PipelineStage
from app.core.queue import TaskPriority
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class PipelineSubmissionResponse(BaseModel):
    """流水线提交响应"""
    success: bool
    pipeline_id: str
    session_id: str
    message: str
    enabled_stages: List[str]
    estimated_completion_time: Optional[float] = None


class PipelineStatusResponse(BaseModel):
    """流水线状态响应"""
    pipeline_id: str
    session_id: str
    filename: str
    stages_completed: List[str]
    stages_failed: List[str]
    total_processing_time: float
    is_completed: bool
    created_at: float


@router.post("/submit", response_model=PipelineSubmissionResponse)
async def submit_audio_pipeline(
    audio_file: UploadFile = File(..., description="音频文件"),
    enable_vad: bool = Form(default=True, description="启用语音活动检测"),
    enable_asr: bool = Form(default=True, description="启用语音识别"),
    enable_speaker_id: bool = Form(default=True, description="启用声纹识别"),
    enable_diarization: bool = Form(default=False, description="启用说话人分离"),
    sample_rate: Optional[int] = Form(default=None, description="音频采样率"),
    priority: str = Form(default="normal", description="任务优先级: low, normal, high, urgent")
):
    """
    提交音频处理流水线
    
    将音频处理分解为多个独立的队列任务：
    1. 音频预处理（必须）
    2. VAD语音活动检测（可选）
    3. ASR语音识别（可选）
    4. 声纹识别（可选）
    5. 说话人分离（可选）
    
    每个阶段都是独立的队列任务，可以并行处理和监控
    """
    try:
        # 验证音频文件
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="上传的文件不是音频格式"
            )
        
        # 至少启用一个处理功能
        if not any([enable_vad, enable_asr, enable_speaker_id, enable_diarization]):
            raise HTTPException(
                status_code=400,
                detail="必须至少启用一个处理功能"
            )
        
        # 解析优先级
        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH,
            "urgent": TaskPriority.URGENT
        }
        task_priority = priority_map.get(priority.lower(), TaskPriority.NORMAL)
        
        # 读取音频数据
        audio_data = await audio_file.read()
        
        # 确定启用的阶段
        enabled_stages = ["preprocessing"]  # 预处理总是启用
        if enable_vad:
            enabled_stages.append("vad")
        if enable_asr:
            enabled_stages.append("asr")
        if enable_speaker_id:
            enabled_stages.append("speaker_identification")
        if enable_diarization:
            enabled_stages.append("speaker_diarization")
        
        # 估算处理时间（基于文件大小和启用的阶段）
        base_time = len(audio_data) / (1024 * 1024) * 2  # 2秒/MB基础时间
        stage_multiplier = len(enabled_stages) * 0.5  # 每个阶段增加0.5倍时间
        estimated_time = base_time * (1 + stage_multiplier)
        
        # 获取流水线编排器
        orchestrator = await get_pipeline_orchestrator()
        
        # 提交流水线
        pipeline_id = await orchestrator.submit_pipeline(
            audio_data=audio_data,
            filename=audio_file.filename,
            enable_vad=enable_vad,
            enable_asr=enable_asr,
            enable_speaker_id=enable_speaker_id,
            enable_diarization=enable_diarization,
            sample_rate=sample_rate or settings.SAMPLE_RATE,
            priority=task_priority
        )
        
        # 获取会话ID
        pipeline_status = await orchestrator.get_pipeline_status(pipeline_id)
        session_id = pipeline_status["session_id"] if pipeline_status else "unknown"
        
        logger.info(f"语音处理流水线已提交: {pipeline_id}, 文件: {audio_file.filename}, 阶段: {enabled_stages}")
        
        return PipelineSubmissionResponse(
            success=True,
            pipeline_id=pipeline_id,
            session_id=session_id,
            message=f"语音处理流水线已提交，优先级: {priority}",
            enabled_stages=enabled_stages,
            estimated_completion_time=estimated_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交语音处理流水线失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"提交流水线失败: {str(e)}"
        )


@router.get("/status/{pipeline_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(pipeline_id: str):
    """获取流水线状态"""
    try:
        orchestrator = await get_pipeline_orchestrator()
        status = await orchestrator.get_pipeline_status(pipeline_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="流水线不存在")
        
        return PipelineStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取流水线状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.get("/result/{pipeline_id}")
async def get_pipeline_result(pipeline_id: str, timeout: float = 300.0):
    """等待并获取流水线结果"""
    try:
        orchestrator = await get_pipeline_orchestrator()
        result = await orchestrator.get_pipeline_result(pipeline_id, timeout)
        
        return result
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="等待流水线结果超时")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取流水线结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取结果失败: {str(e)}")


@router.get("/stages")
async def get_available_stages():
    """获取可用的处理阶段"""
    return {
        "available_stages": [stage.value for stage in PipelineStage],
        "stage_descriptions": {
            "preprocessing": "音频预处理和格式转换",
            "vad": "语音活动检测，分离语音和静音段",
            "asr": "语音识别，将语音转换为文字",
            "speaker_embedding": "声纹特征提取",
            "speaker_identification": "声纹识别，识别说话人身份",
            "speaker_diarization": "说话人分离，区分不同说话人",
            "postprocessing": "后处理和结果整合",
            "storage": "结果存储到数据库"
        },
        "typical_workflows": {
            "transcription_only": ["preprocessing", "vad", "asr"],
            "speaker_identification": ["preprocessing", "vad", "asr", "speaker_identification"],
            "full_analysis": ["preprocessing", "vad", "asr", "speaker_identification", "speaker_diarization"],
            "speaker_diarization_only": ["preprocessing", "speaker_diarization"]
        }
    }


@router.get("/stats")
async def get_pipeline_stats():
    """获取流水线统计信息"""
    try:
        orchestrator = await get_pipeline_orchestrator()
        
        # 获取活动流水线数量
        active_count = len(orchestrator._active_pipelines)
        
        # 统计各阶段的完成情况
        stage_stats = {}
        completed_pipelines = 0
        failed_pipelines = 0
        
        for pipeline_data in orchestrator._active_pipelines.values():
            if pipeline_data.final_results is not None:
                completed_pipelines += 1
            elif pipeline_data.stages_failed:
                failed_pipelines += 1
            
            for stage in pipeline_data.stages_completed:
                stage_stats[stage] = stage_stats.get(stage, 0) + 1
        
        return {
            "active_pipelines": active_count,
            "completed_pipelines": completed_pipelines,
            "failed_pipelines": failed_pipelines,
            "stage_completion_stats": stage_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"获取流水线统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@router.post("/test")
async def test_pipeline_system():
    """测试流水线系统"""
    try:
        orchestrator = await get_pipeline_orchestrator()
        
        # 创建测试音频数据
        import numpy as np
        test_audio = np.random.randn(16000).astype(np.float32)  # 1秒测试音频
        test_audio_bytes = test_audio.tobytes()
        
        # 提交测试流水线
        pipeline_id = await orchestrator.submit_pipeline(
            audio_data=test_audio_bytes,
            filename="test_audio.wav",
            enable_vad=True,
            enable_asr=False,  # 跳过ASR避免模型依赖
            enable_speaker_id=False,
            enable_diarization=False,
            sample_rate=16000,
            priority=TaskPriority.LOW
        )
        
        return {
            "success": True,
            "message": "测试流水线已提交",
            "test_pipeline_id": pipeline_id,
            "note": "这是一个最小化测试，只启用了预处理和VAD"
        }
        
    except Exception as e:
        logger.error(f"测试流水线系统失败: {e}")
        raise HTTPException(status_code=500, detail=f"测试失败: {str(e)}")
