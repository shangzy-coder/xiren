"""
声纹识别接口
基于Sherpa-ONNX实现声纹注册、识别和管理
"""
from fastapi import APIRouter, HTTPException, status, Query, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import numpy as np
import io
import uuid
import time

from app.core.speaker_pool import get_speaker_pool
from app.services.db import get_database_service
from app.utils.audio import AudioProcessor
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class SpeakerResponse(BaseModel):
    """声纹识别响应模型"""
    success: bool
    message: str = ""
    data: Dict[str, Any] = {}


class SpeakerRegisterRequest(BaseModel):
    """说话人注册请求模型"""
    name: str
    metadata: Optional[Dict[str, Any]] = None


class SpeakerIdentifyRequest(BaseModel):
    """说话人识别请求模型"""
    threshold: Optional[float] = None
    
    
@router.get("/status")
async def get_speaker_status():
    """
    获取声纹识别服务状态
    """
    try:
        speaker_pool = await get_speaker_pool()
        db_service = await get_database_service()
        
        speaker_stats = speaker_pool.get_stats()
        db_stats = await db_service.get_speaker_stats()
        
        return {
            "status": "running",
            "speaker_pool": speaker_stats,
            "database": db_stats,
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"获取声纹识别状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法获取声纹识别服务状态"
        )


@router.get("/speakers")
async def list_speakers(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    列出已注册的说话人
    
    Args:
        limit: 返回结果数量限制
        offset: 分页偏移量
    """
    try:
        db_service = await get_database_service()
        speakers = await db_service.get_all_speakers(limit=limit, offset=offset)
        total = len(speakers)  # 这里可以优化，单独查询总数
        
        return {
            "status": "success",
            "total": total,
            "speakers": speakers,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"列出说话人失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法列出说话人"
        )


@router.post("/register")
async def register_speaker(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    audio_file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    注册新的说话人
    
    Args:
        name: 说话人姓名
        audio_file: 音频文件
        metadata: 附加元数据 (JSON字符串)
    """
    try:
        # 验证音频文件格式
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="上传的文件不是音频格式"
            )
        
        # 读取音频数据
        audio_data = await audio_file.read()
        
        # 处理音频
        audio_processor = AudioProcessor()
        processed_audio = await audio_processor.convert_and_resample(
            audio_data, 
            output_sample_rate=settings.SAMPLE_RATE
        )
        
        # 转换为numpy数组
        audio_array = np.frombuffer(processed_audio, dtype=np.float32)
        
        # 解析元数据
        speaker_metadata = {}
        if metadata:
            import json
            try:
                speaker_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"无效的元数据格式: {metadata}")
        
        # 添加文件信息到元数据
        speaker_metadata.update({
            'original_filename': audio_file.filename,
            'file_size': len(audio_data),
            'registration_source': 'api_upload'
        })
        
        # 注册说话人
        speaker_pool = await get_speaker_pool()
        success = await speaker_pool.register_speaker(
            speaker_name=name,
            audio_data=audio_array,
            sample_rate=settings.SAMPLE_RATE,
            metadata=speaker_metadata
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="说话人注册失败"
            )
        
        # 后台任务：保存到数据库
        background_tasks.add_task(
            _save_speaker_to_database,
            name,
            audio_array,
            speaker_metadata
        )
        
        return {
            "status": "success",
            "message": f"说话人 {name} 注册成功",
            "data": {
                "name": name,
                "metadata": speaker_metadata
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注册说话人失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"注册说话人失败: {str(e)}"
        )


@router.post("/identify")
async def identify_speaker(
    audio_file: UploadFile = File(...),
    threshold: Optional[float] = Form(None)
):
    """
    识别说话人
    
    Args:
        audio_file: 音频文件
        threshold: 相似度阈值
    """
    try:
        # 验证音频文件格式
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="上传的文件不是音频格式"
            )
        
        # 读取音频数据
        audio_data = await audio_file.read()
        
        # 处理音频
        audio_processor = AudioProcessor()
        processed_audio = await audio_processor.convert_and_resample(
            audio_data, 
            output_sample_rate=settings.SAMPLE_RATE
        )
        
        # 转换为numpy数组
        audio_array = np.frombuffer(processed_audio, dtype=np.float32)
        
        # 识别说话人
        speaker_pool = await get_speaker_pool()
        result = await speaker_pool.identify_speaker(
            audio_data=audio_array,
            sample_rate=settings.SAMPLE_RATE,
            threshold=threshold
        )
        
        if result:
            speaker_name, similarity = result
            return {
                "status": "success",
                "message": "说话人识别成功",
                "data": {
                    "speaker_name": speaker_name,
                    "similarity": float(similarity),
                    "threshold": threshold or settings.SPEAKER_SIMILARITY_THRESHOLD,
                    "confidence": "high" if similarity > 0.8 else "medium" if similarity > 0.6 else "low"
                }
            }
        else:
            return {
                "status": "success",
                "message": "未识别到已注册的说话人",
                "data": {
                    "speaker_name": None,
                    "similarity": 0.0,
                    "threshold": threshold or settings.SPEAKER_SIMILARITY_THRESHOLD,
                    "confidence": "none"
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"识别说话人失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"识别说话人失败: {str(e)}"
        )


@router.post("/diarization")
async def speaker_diarization(
    audio_file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None)
):
    """
    说话人分离
    
    Args:
        audio_file: 音频文件
        num_speakers: 预期说话人数量 (-1为自动检测)
    """
    try:
        # 验证音频文件格式
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="上传的文件不是音频格式"
            )
        
        # 读取音频数据
        audio_data = await audio_file.read()
        
        # 处理音频
        audio_processor = AudioProcessor()
        processed_audio = await audio_processor.convert_and_resample(
            audio_data, 
            output_sample_rate=settings.SAMPLE_RATE
        )
        
        # 转换为numpy数组
        audio_array = np.frombuffer(processed_audio, dtype=np.float32)
        
        # 说话人分离
        speaker_pool = await get_speaker_pool()
        segments = await speaker_pool.diarize_speakers(
            audio_data=audio_array,
            sample_rate=settings.SAMPLE_RATE,
            num_speakers=num_speakers or -1
        )
        
        if not segments:
            return {
                "status": "success",
                "message": "未检测到说话人片段",
                "data": {
                    "segments": [],
                    "total_speakers": 0,
                    "total_duration": 0.0
                }
            }
        
        # 计算统计信息
        total_duration = sum(seg['duration'] for seg in segments)
        unique_speakers = len(set(seg['speaker'] for seg in segments))
        
        return {
            "status": "success",
            "message": f"检测到 {unique_speakers} 个说话人",
            "data": {
                "segments": segments,
                "total_speakers": unique_speakers,
                "total_duration": total_duration,
                "total_segments": len(segments)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"说话人分离失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"说话人分离失败: {str(e)}"
        )


@router.get("/speakers/{speaker_name}")
async def get_speaker_info(speaker_name: str):
    """
    获取特定说话人信息
    
    Args:
        speaker_name: 说话人姓名
    """
    try:
        db_service = await get_database_service()
        speaker_info = await db_service.get_speaker_by_name(speaker_name)
        
        if not speaker_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"说话人 {speaker_name} 不存在"
            )
        
        # 不返回声纹特征数据，只返回元信息
        response_data = {
            'id': speaker_info['id'],
            'name': speaker_info['name'],
            'metadata': speaker_info['metadata'],
            'created_at': speaker_info['created_at'].isoformat(),
            'updated_at': speaker_info['updated_at'].isoformat(),
            'has_embedding': speaker_info['embedding'] is not None
        }
        
        return {
            "status": "success",
            "data": response_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取说话人信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取说话人信息失败: {str(e)}"
        )


@router.delete("/speakers/{speaker_name}")
async def delete_speaker(speaker_name: str):
    """
    删除说话人记录
    
    Args:
        speaker_name: 说话人姓名
    """
    try:
        # 从内存池中删除
        speaker_pool = await get_speaker_pool()
        pool_success = speaker_pool.remove_speaker(speaker_name)
        
        # 从数据库中删除
        db_service = await get_database_service()
        db_success = await db_service.delete_speaker(speaker_name)
        
        if not db_success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"说话人 {speaker_name} 不存在"
            )
        
        return {
            "status": "success",
            "message": f"说话人 {speaker_name} 已删除",
            "data": {
                "removed_from_pool": pool_success,
                "removed_from_database": db_success
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除说话人失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除说话人失败: {str(e)}"
        )


@router.post("/search")
async def search_speakers_by_voice(
    audio_file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    limit: Optional[int] = Form(5)
):
    """
    通过语音搜索相似的说话人
    
    Args:
        audio_file: 音频文件
        threshold: 相似度阈值
        limit: 返回结果数量限制
    """
    try:
        # 验证音频文件格式
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="上传的文件不是音频格式"
            )
        
        # 读取音频数据
        audio_data = await audio_file.read()
        
        # 处理音频
        audio_processor = AudioProcessor()
        processed_audio = await audio_processor.convert_and_resample(
            audio_data, 
            output_sample_rate=settings.SAMPLE_RATE
        )
        
        # 转换为numpy数组
        audio_array = np.frombuffer(processed_audio, dtype=np.float32)
        
        # 提取声纹特征
        speaker_pool = await get_speaker_pool()
        embedding = await speaker_pool.extract_embedding(
            audio_data=audio_array,
            sample_rate=settings.SAMPLE_RATE
        )
        
        if embedding is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="无法提取声纹特征"
            )
        
        # 在数据库中搜索相似说话人
        db_service = await get_database_service()
        similar_speakers = await db_service.search_speakers_by_embedding(
            embedding=embedding,
            threshold=threshold or settings.SPEAKER_SIMILARITY_THRESHOLD,
            limit=limit or 5
        )
        
        return {
            "status": "success",
            "message": f"找到 {len(similar_speakers)} 个相似说话人",
            "data": {
                "speakers": similar_speakers,
                "threshold": threshold or settings.SPEAKER_SIMILARITY_THRESHOLD,
                "total": len(similar_speakers)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"声纹搜索失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"声纹搜索失败: {str(e)}"
        )


async def _save_speaker_to_database(
    name: str, 
    embedding: np.ndarray, 
    metadata: Dict[str, Any]
):
    """后台任务：保存说话人到数据库"""
    try:
        db_service = await get_database_service()
        await db_service.insert_speaker(
            name=name,
            embedding=embedding,
            metadata=metadata
        )
        logger.info(f"说话人 {name} 已保存到数据库")
    except Exception as e:
        logger.error(f"保存说话人到数据库失败: {e}")


# TODO: 添加声纹模型管理、批量注册、导入导出等功能