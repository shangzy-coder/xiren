"""
语音识别接口
基于Sherpa-ONNX实现ASR服务API
"""
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import numpy as np
import asyncio
import json

from app.config import settings
from app.core.model import (
    model_manager, 
    initialize_models, 
    recognize_audio, 
    register_speaker,
    get_model_info
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ASRResponse(BaseModel):
    """ASR识别响应模型"""
    success: bool
    message: str = ""
    results: list = []
    statistics: Dict[str, Any] = {}


class ASRRequest(BaseModel):
    """ASR识别请求模型"""
    enable_vad: bool = True
    enable_speaker_id: bool = False
    language: str = "auto"


@router.get("/status")
async def get_asr_status():
    """
    获取语音识别服务的状态
    """
    try:
        model_info = get_model_info()
        
        return {
            "status": "running" if model_info["initialized"] else "initializing",
            "model_loaded": model_info["offline_recognizer"],
            "vad_enabled": model_info["vad"],
            "speaker_id_enabled": model_info["speaker_extractor"],
            "sample_rate": model_info["sample_rate"],
            "max_workers": model_info["max_workers"],
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"获取ASR状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法获取ASR服务状态"
        )


@router.post("/initialize")
async def initialize_asr_models(
    model_type: str = Form(default="sense_voice"),
    use_gpu: bool = Form(default=False),
    enable_vad: bool = Form(default=True),
    enable_speaker_id: bool = Form(default=False),
    enable_punctuation: bool = Form(default=False)
):
    """
    初始化ASR模型
    
    Args:
        model_type: 模型类型 (sense_voice, paraformer, whisper)
        use_gpu: 是否使用GPU加速
        enable_vad: 是否启用语音活动检测
        enable_speaker_id: 是否启用声纹识别
        enable_punctuation: 是否启用标点处理
    """
    try:
        logger.info(f"开始初始化ASR模型: {model_type}")
        
        await initialize_models(
            model_type=model_type,
            use_gpu=use_gpu,
            enable_vad=enable_vad,
            enable_speaker_id=enable_speaker_id,
            enable_punctuation=enable_punctuation
        )
        
        return ASRResponse(
            success=True,
            message=f"ASR模型 {model_type} 初始化成功",
            results=[],
            statistics=get_model_info()
        )
        
    except Exception as e:
        logger.error(f"初始化ASR模型失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型初始化失败: {str(e)}"
        )


@router.post("/transcribe", response_model=ASRResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="音频文件"),
    enable_vad: bool = Form(default=True, description="是否启用语音活动检测"),
    enable_speaker_id: bool = Form(default=False, description="是否启用声纹识别"),
    sample_rate: Optional[int] = Form(default=None, description="音频采样率")
):
    """
    离线音频文件识别
    
    Args:
        file: 上传的音频文件
        enable_vad: 是否启用VAD语音段落分割
        enable_speaker_id: 是否启用声纹识别
        sample_rate: 音频采样率，不指定则使用默认值
    """
    try:
        # 检查模型状态
        if not get_model_info()["initialized"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ASR模型未初始化，请先调用 /initialize 接口"
            )
        
        # 验证文件类型
        if not file.content_type.startswith('audio/'):
            logger.warning(f"上传文件类型可能不正确: {file.content_type}")
        
        # 读取音频文件
        audio_bytes = await file.read()
        logger.info(f"接收到音频文件: {file.filename}, 大小: {len(audio_bytes)} bytes")
        
        # 将音频数据转换为numpy数组
        # 注意：这里假设上传的是WAV格式的32位浮点音频
        # 实际应用中可能需要使用librosa或其他库来处理各种格式
        try:
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        except Exception:
            # 如果直接转换失败，尝试其他方式
            logger.warning("直接转换音频数据失败，需要音频格式转换")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不支持的音频格式，请上传WAV格式的音频文件"
            )
        
        # 调用识别接口
        result = await recognize_audio(
            audio_data=audio_data,
            sample_rate=sample_rate or settings.SAMPLE_RATE,
            enable_vad=enable_vad,
            enable_speaker_id=enable_speaker_id
        )
        
        if result["success"]:
            logger.info(f"音频识别成功，识别出 {len(result['results'])} 个语音段落")
            return ASRResponse(
                success=True,
                message="音频识别成功",
                results=result["results"],
                statistics=result["statistics"]
            )
        else:
            logger.error(f"音频识别失败: {result.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"音频识别失败: {result.get('error', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理音频识别请求失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理音频识别请求失败: {str(e)}"
        )


@router.post("/batch-transcribe", response_model=ASRResponse)
async def batch_transcribe_audio(
    files: list[UploadFile] = File(..., description="音频文件列表"),
    enable_vad: bool = Form(default=True, description="是否启用语音活动检测"),
    enable_speaker_id: bool = Form(default=False, description="是否启用声纹识别")
):
    """
    批量音频文件识别
    
    Args:
        files: 上传的音频文件列表
        enable_vad: 是否启用VAD语音段落分割
        enable_speaker_id: 是否启用声纹识别
    """
    try:
        # 检查模型状态
        if not get_model_info()["initialized"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ASR模型未初始化，请先调用 /initialize 接口"
            )
        
        if len(files) > 10:  # 限制批量处理数量
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="批量处理文件数量不能超过10个"
            )
        
        all_results = []
        total_statistics = {
            "total_files": len(files),
            "successful_files": 0,
            "failed_files": 0,
            "total_duration": 0.0,
            "total_processing_time": 0.0
        }
        
        for i, file in enumerate(files):
            try:
                # 读取音频文件
                audio_bytes = await file.read()
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # 识别音频
                result = await recognize_audio(
                    audio_data=audio_data,
                    sample_rate=settings.SAMPLE_RATE,
                    enable_vad=enable_vad,
                    enable_speaker_id=enable_speaker_id
                )
                
                if result["success"]:
                    total_statistics["successful_files"] += 1
                    total_statistics["total_duration"] += result["statistics"].get("total_duration", 0)
                    total_statistics["total_processing_time"] += result["statistics"].get("processing_time", 0)
                    
                    # 为每个文件的结果添加文件信息
                    file_result = {
                        "file_index": i,
                        "filename": file.filename,
                        "success": True,
                        "results": result["results"],
                        "statistics": result["statistics"]
                    }
                else:
                    total_statistics["failed_files"] += 1
                    file_result = {
                        "file_index": i,
                        "filename": file.filename,
                        "success": False,
                        "error": result.get("error", "Unknown error")
                    }
                
                all_results.append(file_result)
                
            except Exception as e:
                total_statistics["failed_files"] += 1
                all_results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"处理文件 {file.filename} 失败: {e}")
        
        return ASRResponse(
            success=True,
            message=f"批量处理完成，成功: {total_statistics['successful_files']}, 失败: {total_statistics['failed_files']}",
            results=all_results,
            statistics=total_statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量音频识别失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量音频识别失败: {str(e)}"
        )


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    实时音频流识别 (WebSocket)
    
    协议说明:
    - 客户端发送: {"type": "audio", "data": base64_audio_data}
    - 服务端返回: {"type": "transcription", "text": "识别结果", "timestamp": 1.23, "speaker": "说话人"}
    - 客户端发送: {"type": "end"} 结束会话
    """
    await websocket.accept()
    logger.info(f"WebSocket连接已建立: {websocket.client}")
    
    try:
        # 检查模型状态
        if not get_model_info()["initialized"]:
            await websocket.send_json({
                "type": "error",
                "message": "ASR模型未初始化，请先初始化模型"
            })
            await websocket.close()
            return
        
        # 音频缓冲区
        audio_buffer = []
        buffer_size_seconds = 2.0  # 缓冲区大小(秒)
        buffer_size_samples = int(settings.SAMPLE_RATE * buffer_size_seconds)
        
        while True:
            try:
                # 接收消息
                message = await websocket.receive_json()
                
                if message.get("type") == "audio":
                    # 处理音频数据
                    import base64
                    audio_data = base64.b64decode(message["data"])
                    audio_samples = np.frombuffer(audio_data, dtype=np.float32)
                    
                    # 添加到缓冲区
                    audio_buffer.extend(audio_samples)
                    
                    # 当缓冲区达到指定大小时进行识别
                    if len(audio_buffer) >= buffer_size_samples:
                        # 取出识别所需的样本
                        recognition_samples = np.array(audio_buffer[:buffer_size_samples])
                        
                        # 清理已处理的样本（保留部分重叠）
                        overlap_samples = buffer_size_samples // 4
                        audio_buffer = audio_buffer[buffer_size_samples - overlap_samples:]
                        
                        # 异步识别
                        result = await recognize_audio(
                            audio_data=recognition_samples,
                            sample_rate=settings.SAMPLE_RATE,
                            enable_vad=True,
                            enable_speaker_id=True
                        )
                        
                        if result["success"] and result["results"]:
                            for segment in result["results"]:
                                if segment["text"].strip():  # 只发送非空结果
                                    await websocket.send_json({
                                        "type": "transcription",
                                        "text": segment["text"],
                                        "timestamp": segment["start_time"],
                                        "speaker": segment.get("speaker", "unknown"),
                                        "language": segment.get("language", "unknown"),
                                        "emotion": segment.get("emotion", "unknown")
                                    })
                                    
                elif message.get("type") == "end":
                    # 处理剩余缓冲区中的音频
                    if len(audio_buffer) > settings.SAMPLE_RATE * 0.5:  # 至少0.5秒
                        final_samples = np.array(audio_buffer)
                        result = await recognize_audio(
                            audio_data=final_samples,
                            sample_rate=settings.SAMPLE_RATE,
                            enable_vad=True,
                            enable_speaker_id=True
                        )
                        
                        if result["success"] and result["results"]:
                            for segment in result["results"]:
                                if segment["text"].strip():
                                    await websocket.send_json({
                                        "type": "transcription",
                                        "text": segment["text"],
                                        "timestamp": segment["start_time"],
                                        "speaker": segment.get("speaker", "unknown"),
                                        "language": segment.get("language", "unknown"),
                                        "emotion": segment.get("emotion", "unknown")
                                    })
                    
                    # 发送结束消息
                    await websocket.send_json({"type": "end", "message": "识别会话结束"})
                    break
                    
                else:
                    await websocket.send_json({
                        "type": "error", 
                        "message": f"未知消息类型: {message.get('type')}"
                    })
                    
            except asyncio.TimeoutError:
                logger.warning("WebSocket接收超时")
                continue
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"服务器错误: {str(e)}"
            })
        except:
            pass
        await websocket.close()


@router.get("/health")
async def health_check():
    """
    健康检查接口
    """
    try:
        model_info = get_model_info()
        
        # 检查关键组件状态
        health_status = {
            "service": "speech-recognition-asr",
            "status": "healthy" if model_info["initialized"] else "degraded",
            "components": {
                "asr_model": "ok" if model_info["offline_recognizer"] else "unavailable",
                "vad_model": "ok" if model_info["vad"] else "unavailable",
                "speaker_id": "ok" if model_info["speaker_extractor"] else "unavailable"
            },
            "version": "1.0.0"
        }
        
        status_code = 200 if model_info["initialized"] else 503
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "speech-recognition-asr",
                "status": "unhealthy",
                "error": str(e)
            }
        )