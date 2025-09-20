"""
综合语音处理API
整合语音识别、声纹识别和多模态功能
"""
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import logging
import numpy as np
import asyncio
import time
import uuid
import json

from app.core.model import recognize_audio, get_model_info
from app.core.speaker_pool import get_speaker_pool
from app.services.db import get_database_service
from app.utils.audio import AudioProcessor
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class ComprehensiveResponse(BaseModel):
    """综合处理响应模型"""
    success: bool
    session_id: str
    message: str = ""
    transcription: Dict[str, Any] = {}
    speaker_analysis: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class ProcessingOptions(BaseModel):
    """处理选项模型"""
    enable_asr: bool = True
    enable_speaker_id: bool = True
    enable_diarization: bool = False
    enable_vad: bool = True
    speaker_threshold: Optional[float] = None
    language: str = "auto"
    save_to_database: bool = True


@router.post("/process", response_model=ComprehensiveResponse)
async def comprehensive_audio_processing(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    enable_asr: bool = Form(default=True, description="启用语音转文字"),
    enable_speaker_id: bool = Form(default=True, description="启用声纹识别"),
    enable_diarization: bool = Form(default=False, description="启用说话人分离"),
    enable_vad: bool = Form(default=True, description="启用语音活动检测"),
    speaker_threshold: Optional[float] = Form(default=None, description="声纹相似度阈值"),
    language: str = Form(default="auto", description="语言设置"),
    save_to_database: bool = Form(default=True, description="保存结果到数据库")
):
    """
    综合音频处理接口
    
    支持同时进行语音识别、声纹识别、说话人分离等多项功能
    
    Args:
        audio_file: 音频文件
        enable_asr: 是否启用语音转文字
        enable_speaker_id: 是否启用声纹识别
        enable_diarization: 是否启用说话人分离
        enable_vad: 是否启用语音活动检测
        speaker_threshold: 声纹相似度阈值
        language: 语言设置
        save_to_database: 是否保存结果到数据库
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # 验证音频文件
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="上传的文件不是音频格式"
            )
        
        logger.info(f"开始综合音频处理，会话ID: {session_id}")
        
        # 读取音频数据
        audio_data = await audio_file.read()
        
        # 音频预处理
        audio_processor = AudioProcessor()
        processed_audio = await audio_processor.convert_and_resample(
            audio_data, 
            output_sample_rate=settings.SAMPLE_RATE
        )
        
        # 转换为numpy数组
        audio_array = np.frombuffer(processed_audio, dtype=np.float32)
        
        # 计算音频时长
        duration = len(audio_array) / settings.SAMPLE_RATE
        
        # 结果容器
        transcription_result = {}
        speaker_analysis_result = {}
        metadata = {
            "session_id": session_id,
            "filename": audio_file.filename,
            "file_size": len(audio_data),
            "duration": duration,
            "sample_rate": settings.SAMPLE_RATE,
            "processing_options": {
                "enable_asr": enable_asr,
                "enable_speaker_id": enable_speaker_id,
                "enable_diarization": enable_diarization,
                "enable_vad": enable_vad,
                "language": language
            }
        }
        
        # 并行处理任务
        tasks = []
        
        # 语音识别任务
        if enable_asr:
            tasks.append(_process_asr(audio_array, enable_vad, language))
        
        # 声纹识别任务
        if enable_speaker_id or enable_diarization:
            tasks.append(_process_speaker_analysis(
                audio_array, 
                enable_speaker_id, 
                enable_diarization,
                speaker_threshold
            ))
        
        # 并行执行所有任务
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            result_index = 0
            if enable_asr:
                asr_result = results[result_index]
                if isinstance(asr_result, Exception):
                    logger.error(f"ASR处理失败: {asr_result}")
                    transcription_result = {"error": str(asr_result)}
                else:
                    transcription_result = asr_result
                result_index += 1
            
            if enable_speaker_id or enable_diarization:
                speaker_result = results[result_index]
                if isinstance(speaker_result, Exception):
                    logger.error(f"声纹分析失败: {speaker_result}")
                    speaker_analysis_result = {"error": str(speaker_result)}
                else:
                    speaker_analysis_result = speaker_result
        
        # 计算处理时间
        processing_time = time.time() - start_time
        metadata["processing_time"] = processing_time
        
        # 后台任务：保存到数据库
        if save_to_database:
            background_tasks.add_task(
                _save_processing_results,
                session_id,
                audio_file.filename,
                transcription_result,
                speaker_analysis_result,
                metadata
            )
        
        response = ComprehensiveResponse(
            success=True,
            session_id=session_id,
            message="音频综合处理完成",
            transcription=transcription_result,
            speaker_analysis=speaker_analysis_result,
            metadata=metadata
        )
        
        logger.info(f"综合音频处理完成，会话ID: {session_id}, 耗时: {processing_time:.2f}秒")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"综合音频处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"音频处理失败: {str(e)}"
        )


async def _process_asr(audio_array: np.ndarray, enable_vad: bool, language: str) -> Dict[str, Any]:
    """处理语音识别"""
    try:
        result = await recognize_audio(
            audio_data=audio_array,
            sample_rate=settings.SAMPLE_RATE,
            enable_vad=enable_vad,
            enable_speaker_id=False  # 在声纹分析中单独处理
        )
        
        if result["success"]:
            return {
                "success": True,
                "text": " ".join([segment["text"] for segment in result["results"]]),
                "segments": result["results"],
                "statistics": result["statistics"],
                "language": language
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "语音识别失败")
            }
    except Exception as e:
        logger.error(f"ASR处理异常: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def _process_speaker_analysis(
    audio_array: np.ndarray, 
    enable_speaker_id: bool, 
    enable_diarization: bool,
    threshold: Optional[float]
) -> Dict[str, Any]:
    """处理声纹分析"""
    try:
        speaker_pool = await get_speaker_pool()
        result = {}
        
        # 声纹识别
        if enable_speaker_id:
            identification = await speaker_pool.identify_speaker(
                audio_data=audio_array,
                sample_rate=settings.SAMPLE_RATE,
                threshold=threshold
            )
            
            if identification:
                speaker_name, similarity = identification
                result["identification"] = {
                    "speaker_name": speaker_name,
                    "similarity": float(similarity),
                    "confidence": "high" if similarity > 0.8 else "medium" if similarity > 0.6 else "low"
                }
            else:
                result["identification"] = {
                    "speaker_name": None,
                    "similarity": 0.0,
                    "confidence": "none"
                }
        
        # 说话人分离
        if enable_diarization:
            segments = await speaker_pool.diarize_speakers(
                audio_data=audio_array,
                sample_rate=settings.SAMPLE_RATE,
                num_speakers=-1  # 自动检测
            )
            
            if segments:
                result["diarization"] = {
                    "segments": segments,
                    "total_speakers": len(set(seg["speaker"] for seg in segments)),
                    "total_duration": sum(seg["duration"] for seg in segments)
                }
            else:
                result["diarization"] = {
                    "segments": [],
                    "total_speakers": 0,
                    "total_duration": 0.0
                }
        
        result["success"] = True
        return result
        
    except Exception as e:
        logger.error(f"声纹分析异常: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/sessions/{session_id}")
async def get_processing_session(session_id: str):
    """
    获取处理会话详情
    
    Args:
        session_id: 会话ID
    """
    try:
        db_service = await get_database_service()
        
        # 从数据库查询会话信息
        # 这里需要实现具体的数据库查询逻辑
        session_info = {
            "session_id": session_id,
            "status": "completed",  # TODO: 从数据库获取实际状态
            "message": "会话信息获取功能待实现"
        }
        
        return {
            "success": True,
            "data": session_info
        }
        
    except Exception as e:
        logger.error(f"获取会话信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取会话信息失败: {str(e)}"
        )


@router.get("/sessions")
async def list_processing_sessions(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None
):
    """
    列出处理会话
    
    Args:
        limit: 返回数量限制
        offset: 分页偏移
        status_filter: 状态过滤器
    """
    try:
        # TODO: 实现从数据库获取会话列表
        sessions = []
        
        return {
            "success": True,
            "data": {
                "sessions": sessions,
                "total": len(sessions),
                "limit": limit,
                "offset": offset,
                "status_filter": status_filter
            }
        }
        
    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取会话列表失败: {str(e)}"
        )


@router.post("/quick-transcribe")
async def quick_transcribe(
    audio_file: UploadFile = File(...),
    language: str = Form(default="auto")
):
    """
    快速语音转文字接口
    
    简化版的语音识别接口，只返回文字结果
    
    Args:
        audio_file: 音频文件
        language: 语言设置
    """
    try:
        # 验证音频文件
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="上传的文件不是音频格式"
            )
        
        # 读取音频数据
        audio_data = await audio_file.read()
        
        # 音频预处理
        audio_processor = AudioProcessor()
        processed_audio = await audio_processor.convert_and_resample(
            audio_data, 
            output_sample_rate=settings.SAMPLE_RATE
        )
        
        # 转换为numpy数组
        audio_array = np.frombuffer(processed_audio, dtype=np.float32)
        
        # 语音识别
        result = await recognize_audio(
            audio_data=audio_array,
            sample_rate=settings.SAMPLE_RATE,
            enable_vad=True,
            enable_speaker_id=False
        )
        
        if result["success"]:
            # 合并所有文字片段
            full_text = " ".join([segment["text"] for segment in result["results"]])
            
            return {
                "success": True,
                "text": full_text,
                "language": language,
                "duration": result["statistics"].get("total_duration", 0),
                "confidence": result["statistics"].get("average_confidence", 0)
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "语音识别失败")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"快速转录失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"语音转录失败: {str(e)}"
        )


@router.post("/quick-speaker-id")
async def quick_speaker_identification(
    audio_file: UploadFile = File(...),
    threshold: Optional[float] = Form(default=None)
):
    """
    快速声纹识别接口
    
    简化版的声纹识别接口，只返回说话人信息
    
    Args:
        audio_file: 音频文件
        threshold: 相似度阈值
    """
    try:
        # 验证音频文件
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="上传的文件不是音频格式"
            )
        
        # 读取音频数据
        audio_data = await audio_file.read()
        
        # 音频预处理
        audio_processor = AudioProcessor()
        processed_audio = await audio_processor.convert_and_resample(
            audio_data, 
            output_sample_rate=settings.SAMPLE_RATE
        )
        
        # 转换为numpy数组
        audio_array = np.frombuffer(processed_audio, dtype=np.float32)
        
        # 声纹识别
        speaker_pool = await get_speaker_pool()
        result = await speaker_pool.identify_speaker(
            audio_data=audio_array,
            sample_rate=settings.SAMPLE_RATE,
            threshold=threshold
        )
        
        if result:
            speaker_name, similarity = result
            return {
                "success": True,
                "speaker_name": speaker_name,
                "similarity": float(similarity),
                "confidence": "high" if similarity > 0.8 else "medium" if similarity > 0.6 else "low",
                "threshold": threshold or settings.SPEAKER_SIMILARITY_THRESHOLD
            }
        else:
            return {
                "success": True,
                "speaker_name": None,
                "similarity": 0.0,
                "confidence": "none",
                "threshold": threshold or settings.SPEAKER_SIMILARITY_THRESHOLD,
                "message": "未识别到已注册的说话人"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"快速声纹识别失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"声纹识别失败: {str(e)}"
        )


async def _save_processing_results(
    session_id: str,
    filename: str,
    transcription_result: Dict[str, Any],
    speaker_analysis_result: Dict[str, Any],
    metadata: Dict[str, Any]
):
    """后台任务：保存处理结果到数据库"""
    try:
        db_service = await get_database_service()
        
        # 构建结果数据
        results = {
            "transcription": transcription_result,
            "speaker_analysis": speaker_analysis_result,
            "metadata": metadata
        }
        
        # 插入识别会话记录
        await db_service.insert_recognition_session(
            session_id=session_id,
            recognition_type="comprehensive",
            status="completed"
        )
        
        # 更新会话结果
        await db_service.update_recognition_session(
            session_id=session_id,
            results=results,
            status="completed",
            processing_time=metadata.get("processing_time", 0)
        )
        
        logger.info(f"处理结果已保存到数据库，会话ID: {session_id}")
        
    except Exception as e:
        logger.error(f"保存处理结果失败: {e}")


@router.get("/health")
async def health_check():
    """
    综合服务健康检查
    """
    try:
        # 检查ASR服务
        asr_info = get_model_info()
        
        # 检查声纹服务
        speaker_pool = await get_speaker_pool()
        speaker_stats = speaker_pool.get_stats()
        
        # 检查数据库服务
        try:
            db_service = await get_database_service()
            db_stats = await db_service.get_speaker_stats()
            db_healthy = True
        except:
            db_stats = {}
            db_healthy = False
        
        health_status = {
            "service": "comprehensive-speech-processing",
            "status": "healthy",
            "components": {
                "asr_service": {
                    "status": "ok" if asr_info["initialized"] else "degraded",
                    "details": asr_info
                },
                "speaker_service": {
                    "status": "ok" if speaker_stats["initialized"] else "degraded",
                    "details": speaker_stats
                },
                "database_service": {
                    "status": "ok" if db_healthy else "unavailable",
                    "details": db_stats
                }
            },
            "version": "1.0.0"
        }
        
        # 确定整体状态
        if not asr_info["initialized"] or not speaker_stats["initialized"]:
            health_status["status"] = "degraded"
            
        return health_status
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "comprehensive-speech-processing",
                "status": "unhealthy",
                "error": str(e)
            }
        )
