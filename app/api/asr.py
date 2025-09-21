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
from app.core.request_manager import get_request_manager, TaskPriority
from app.core.queue import TaskType
from app.utils.logging_config import get_logger, log_request_response
from app.utils.metrics import metrics_collector

logger = get_logger(__name__)
router = APIRouter()


class ASRResponse(BaseModel):
    """ASR识别响应模型"""
    success: bool
    message: str = ""
    results: list = []
    statistics: Dict[str, Any] = {}


class ASRAsyncResponse(BaseModel):
    """ASR异步处理响应模型"""
    success: bool
    task_id: str
    message: str
    estimated_completion_time: Optional[float] = None


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
        # 获取当前模型状态
        current_info = get_model_info()

        # 检查模型是否已经初始化
        if current_info.get("is_initialized", False):
            logger.info(f"模型已初始化，当前状态: {current_info}")
            return ASRResponse(
                success=True,
                message=f"模型已初始化 (当前: {current_info.get('model_type', 'unknown')})",
                results=[],
                statistics=current_info
            )

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
@log_request_response
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


@router.post("/transcribe-async", response_model=ASRAsyncResponse)
async def transcribe_audio_async(
    file: UploadFile = File(..., description="音频文件"),
    enable_vad: bool = Form(default=True, description="是否启用语音活动检测"),
    enable_speaker_id: bool = Form(default=False, description="是否启用声纹识别"),
    sample_rate: Optional[int] = Form(default=None, description="音频采样率"),
    priority: str = Form(default="normal", description="任务优先级: low, normal, high, urgent")
):
    """
    异步音频识别接口
    
    将语音识别任务提交到队列系统，立即返回任务ID，可通过任务ID查询结果
    """
    try:
        # 检查模型状态
        if not get_model_info()["initialized"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ASR模型未初始化，请先调用 /initialize 接口"
            )
        
        # 验证音频文件
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="上传的文件不是音频格式"
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
        audio_data = await file.read()
        
        # 估算处理时间（基于文件大小）
        estimated_time = len(audio_data) / (1024 * 1024) * 2  # 简单估算：2秒/MB
        
        # 获取请求管理器
        request_manager = await get_request_manager()
        
        # 提交异步ASR任务
        task_id = await request_manager.submit_asr_request(
            func=_process_asr_task,
            args=(audio_data, file.filename, enable_vad, enable_speaker_id, sample_rate),
            priority=task_priority,
            timeout=settings.TASK_TIMEOUT
        )
        
        logger.info(f"异步ASR任务已提交: {task_id}, 文件: {file.filename}")
        
        return ASRAsyncResponse(
            success=True,
            task_id=task_id,
            message=f"ASR任务已提交到队列，优先级: {priority}",
            estimated_completion_time=estimated_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交异步ASR任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"提交任务失败: {str(e)}"
        )


async def _process_asr_task(audio_data: bytes, 
                          filename: str,
                          enable_vad: bool = True,
                          enable_speaker_id: bool = False,
                          sample_rate: Optional[int] = None) -> Dict[str, Any]:
    """
    异步处理ASR任务的内部函数
    
    Args:
        audio_data: 音频数据
        filename: 文件名
        enable_vad: 是否启用VAD
        enable_speaker_id: 是否启用声纹识别
        sample_rate: 采样率
        
    Returns:
        识别结果
    """
    try:
        # 音频预处理
        from app.utils.audio import AudioProcessor
        audio_processor = AudioProcessor()
        processed_audio = await audio_processor.convert_and_resample(
            audio_data, 
            output_sample_rate=sample_rate or settings.SAMPLE_RATE
        )
        
        # 转换为numpy数组
        audio_array = np.frombuffer(processed_audio, dtype=np.float32)
        
        # 调用语音识别
        result = await recognize_audio(
            audio_data=audio_array,
            sample_rate=sample_rate or settings.SAMPLE_RATE,
            enable_vad=enable_vad,
            enable_speaker_id=enable_speaker_id
        )
        
        # 添加文件信息
        result["filename"] = filename
        result["file_size"] = len(audio_data)
        result["duration"] = len(audio_array) / (sample_rate or settings.SAMPLE_RATE)
        
        if result["success"]:
            logger.info(f"异步ASR任务完成: {filename}, 识别出 {len(result.get('results', []))} 个语音段落")
        else:
            logger.error(f"异步ASR任务失败: {filename}, 错误: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"处理异步ASR任务失败: {filename}, 错误: {e}")
        return {
            "success": False,
            "error": str(e),
            "filename": filename,
            "results": [],
            "statistics": {}
        }


@router.get("/task/{task_id}/status")
async def get_asr_task_status(task_id: str):
    """获取ASR任务状态"""
    try:
        request_manager = await get_request_manager()
        result = await request_manager.get_request_status(task_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return {
            "task_id": task_id,
            "status": result.status.value,
            "result": result.result,
            "error": result.error,
            "execution_time": result.execution_time,
            "created_at": result.created_at,
            "started_at": result.started_at,
            "completed_at": result.completed_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取ASR任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.get("/task/{task_id}/result")
async def get_asr_task_result(task_id: str, timeout: float = 30.0):
    """等待并获取ASR任务结果"""
    try:
        request_manager = await get_request_manager()
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
        logger.error(f"获取ASR任务结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取结果失败: {str(e)}")


@router.delete("/task/{task_id}")
async def cancel_asr_task(task_id: str):
    """取消ASR任务"""
    try:
        request_manager = await get_request_manager()
        success = await request_manager.cancel_request(task_id)
        
        if success:
            return {"success": True, "message": f"任务 {task_id} 已取消"}
        else:
            raise HTTPException(status_code=404, detail="任务不存在或无法取消")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消ASR任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"取消失败: {str(e)}")


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


@router.post("/batch-transcribe-async", response_model=ASRAsyncResponse)
async def batch_transcribe_audio_async(
    files: list[UploadFile] = File(..., description="音频文件列表"),
    enable_vad: bool = Form(default=True, description="是否启用语音活动检测"),
    enable_speaker_id: bool = Form(default=False, description="是否启用声纹识别"),
    priority: str = Form(default="low", description="任务优先级，批量任务建议使用low")
):
    """
    异步批量音频识别接口
    
    将批量语音识别任务提交到队列系统
    """
    try:
        # 检查模型状态
        if not get_model_info()["initialized"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ASR模型未初始化，请先调用 /initialize 接口"
            )
        
        if len(files) > settings.BATCH_SIZE_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"批量处理文件数量不能超过{settings.BATCH_SIZE_LIMIT}个"
            )
        
        # 解析优先级
        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH,
            "urgent": TaskPriority.URGENT
        }
        task_priority = priority_map.get(priority.lower(), TaskPriority.LOW)
        
        # 准备批量任务数据
        batch_data = []
        total_size = 0
        
        for file in files:
            if not file.content_type or not file.content_type.startswith('audio/'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"文件 {file.filename} 不是音频格式"
                )
            
            audio_data = await file.read()
            total_size += len(audio_data)
            batch_data.append({
                "audio_data": audio_data,
                "filename": file.filename,
                "enable_vad": enable_vad,
                "enable_speaker_id": enable_speaker_id,
                "sample_rate": None
            })
        
        # 估算处理时间
        estimated_time = total_size / (1024 * 1024) * 3  # 批量处理稍慢：3秒/MB
        
        # 获取请求管理器
        request_manager = await get_request_manager()
        
        # 提交批量任务
        task_id = await request_manager.submit_batch_request(
            func=_process_batch_asr_task,
            args=(batch_data,),
            priority=task_priority,
            timeout=settings.TASK_TIMEOUT * 2  # 批量任务超时时间更长
        )
        
        logger.info(f"异步批量ASR任务已提交: {task_id}, 文件数: {len(files)}")
        
        return ASRAsyncResponse(
            success=True,
            task_id=task_id,
            message=f"批量ASR任务已提交到队列，文件数: {len(files)}，优先级: {priority}",
            estimated_completion_time=estimated_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交异步批量ASR任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"提交批量任务失败: {str(e)}"
        )


async def _process_batch_asr_task(batch_data: list) -> Dict[str, Any]:
    """
    异步处理批量ASR任务
    
    Args:
        batch_data: 批量任务数据列表
        
    Returns:
        批量处理结果
    """
    try:
        from app.utils.audio import AudioProcessor
        audio_processor = AudioProcessor()
        
        all_results = []
        total_statistics = {
            "total_files": len(batch_data),
            "successful_files": 0,
            "failed_files": 0,
            "total_duration": 0.0,
            "total_processing_time": 0.0
        }
        
        for i, file_data in enumerate(batch_data):
            try:
                # 音频预处理
                processed_audio = await audio_processor.convert_and_resample(
                    file_data["audio_data"], 
                    output_sample_rate=file_data["sample_rate"] or settings.SAMPLE_RATE
                )
                
                # 转换为numpy数组
                audio_array = np.frombuffer(processed_audio, dtype=np.float32)
                
                # 语音识别
                result = await recognize_audio(
                    audio_data=audio_array,
                    sample_rate=file_data["sample_rate"] or settings.SAMPLE_RATE,
                    enable_vad=file_data["enable_vad"],
                    enable_speaker_id=file_data["enable_speaker_id"]
                )
                
                # 添加文件信息
                result["filename"] = file_data["filename"]
                result["file_size"] = len(file_data["audio_data"])
                result["duration"] = len(audio_array) / (file_data["sample_rate"] or settings.SAMPLE_RATE)
                result["file_index"] = i
                
                all_results.append(result)
                
                if result["success"]:
                    total_statistics["successful_files"] += 1
                    total_statistics["total_duration"] += result["duration"]
                    if result.get("statistics", {}).get("processing_time"):
                        total_statistics["total_processing_time"] += result["statistics"]["processing_time"]
                else:
                    total_statistics["failed_files"] += 1
                
                logger.info(f"批量ASR任务进度: {i+1}/{len(batch_data)}, 文件: {file_data['filename']}")
                
            except Exception as e:
                logger.error(f"处理文件失败: {file_data['filename']}, 错误: {e}")
                all_results.append({
                    "success": False,
                    "error": str(e),
                    "filename": file_data["filename"],
                    "file_index": i,
                    "results": [],
                    "statistics": {}
                })
                total_statistics["failed_files"] += 1
        
        logger.info(f"批量ASR任务完成，成功: {total_statistics['successful_files']}, 失败: {total_statistics['failed_files']}")
        
        return {
            "success": True,
            "message": "批量音频识别完成",
            "results": all_results,
            "statistics": total_statistics
        }
        
    except Exception as e:
        logger.error(f"批量ASR任务处理失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "statistics": total_statistics
        }


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