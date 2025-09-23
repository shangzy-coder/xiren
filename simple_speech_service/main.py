"""
简单语音识别服务主入口
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import time
import numpy as np

from .config import settings
from .speech_processor import SpeechProcessor
from .audio_utils import audio_processor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="简单语音识别服务",
    description="基于Sherpa-ONNX的轻量级语音识别和说话人识别服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局语音处理器实例
speech_processor = None

# 数据模型
class RecognitionResult(BaseModel):
    """语音识别结果"""
    text: str
    text_with_punct: Optional[str] = None
    emotion: str
    event: str
    language: str
    speaker: str
    start_time: float
    end_time: float

class SpeakerInfo(BaseModel):
    """说话人信息"""
    name: str
    embedding_dim: int
    registered_at: str
    metadata: Dict[str, Any]

class APIResponse(BaseModel):
    """API响应"""
    success: bool
    message: str
    data: Any = None

# API路由
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    global speech_processor
    logger.info("正在初始化语音处理器...")

    try:
        speech_processor = SpeechProcessor(use_gpu=settings.USE_GPU)
        logger.info("语音处理器初始化完成")
    except Exception as e:
        logger.error(f"语音处理器初始化失败: {e}")
        raise

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "简单语音识别服务运行正常",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    if not speech_processor:
        return {
            "status": "unhealthy",
            "message": "语音处理器未初始化"
        }

    stats = speech_processor.get_stats()
    return {
        "status": "healthy",
        "service": "simple-speech-service",
        "version": "1.0.0",
        "processor_stats": stats
    }

@app.post("/api/recognize", response_model=APIResponse)
async def recognize_audio(
    audio_file: UploadFile = File(...),
    enable_vad: bool = Form(default=True, description="启用语音活动检测"),
    language: str = Form(default="auto", description="识别语言")
):
    """
    语音识别接口

    - **audio_file**: 音频文件
    - **enable_vad**: 是否启用语音活动检测
    - **language**: 识别语言 (auto, zh, en, ja, ko, yue)
    """
    try:
        # 验证文件格式
        if not audio_processor.is_supported_format(audio_file.filename, audio_file.content_type):
            supported_formats = ", ".join(settings.SUPPORTED_FORMATS)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件格式。支持的格式: {supported_formats}"
            )

        # 检查文件大小
        file_size = len(await audio_file.read())
        await audio_file.seek(0)  # 重置文件指针

        if file_size > settings.MAX_AUDIO_SIZE:
            max_size_mb = settings.MAX_AUDIO_SIZE / 1024 / 1024
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"文件过大。最大支持 {max_size_mb:.1f}MB"
            )

        logger.info(f"开始处理音频文件: {audio_file.filename}")

        # 读取和处理音频数据
        audio_data = await audio_file.read()
        samples, sample_rate = audio_processor.load_audio_from_bytes(audio_data, audio_file.filename)

        # 语音识别
        start_time = time.time()
        results = speech_processor.recognize_audio(samples, sample_rate, enable_vad)
        processing_time = time.time() - start_time

        # 转换为响应格式
        recognition_results = []
        for result in results:
            recognition_results.append(RecognitionResult(**result))

        return APIResponse(
            success=True,
            message=f"识别完成，处理了 {len(results)} 个语音段落",
            data={
                "results": [result.dict() for result in recognition_results],
                "processing_time": processing_time,
                "total_segments": len(results)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"语音识别失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"语音识别失败: {str(e)}"
        )

@app.post("/api/speakers/register", response_model=APIResponse)
async def register_speaker(
    audio_file: UploadFile = File(...),
    name: str = Form(..., description="说话人姓名"),
    metadata: Optional[str] = Form(default=None, description="附加元数据 (JSON字符串)")
):
    """
    注册说话人

    - **audio_file**: 说话人的音频样本
    - **name**: 说话人姓名
    - **metadata**: 附加元数据
    """
    try:
        # 验证文件格式
        if not audio_processor.is_supported_format(audio_file.filename, audio_file.content_type):
            supported_formats = ", ".join(settings.SUPPORTED_FORMATS)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件格式。支持的格式: {supported_formats}"
            )

        logger.info(f"注册说话人: {name}")

        # 读取和处理音频数据
        audio_data = await audio_file.read()
        samples, sample_rate = audio_processor.load_audio_from_bytes(audio_data, audio_file.filename)

        # 解析元数据
        speaker_metadata = {}
        if metadata:
            try:
                import json
                speaker_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"无效的元数据格式: {metadata}")

        # 添加文件信息
        speaker_metadata.update({
            'original_filename': audio_file.filename,
            'file_size': len(audio_data),
            'registration_source': 'api_upload'
        })

        # 注册说话人
        success = speech_processor.register_speaker(name, samples, sample_rate, speaker_metadata)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="说话人注册失败"
            )

        return APIResponse(
            success=True,
            message=f"说话人 {name} 注册成功",
            data={"name": name, "metadata": speaker_metadata}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注册说话人失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"注册说话人失败: {str(e)}"
        )

@app.get("/api/speakers", response_model=APIResponse)
async def list_speakers():
    """列出所有已注册的说话人"""
    try:
        speaker_manager = speech_processor.get_speaker_manager()
        speakers = speaker_manager.list_speakers()

        return APIResponse(
            success=True,
            message=f"找到 {len(speakers)} 个已注册说话人",
            data={"speakers": speakers, "total": len(speakers)}
        )

    except Exception as e:
        logger.error(f"获取说话人列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取说话人列表失败: {str(e)}"
        )

@app.post("/api/speakers/identify", response_model=APIResponse)
async def identify_speaker(
    audio_file: UploadFile = File(...),
    threshold: Optional[float] = Form(default=None, description="相似度阈值")
):
    """
    识别说话人

    - **audio_file**: 待识别的音频文件
    - **threshold**: 相似度阈值 (可选)
    """
    try:
        # 验证文件格式
        if not audio_processor.is_supported_format(audio_file.filename, audio_file.content_type):
            supported_formats = ", ".join(settings.SUPPORTED_FORMATS)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件格式。支持的格式: {supported_formats}"
            )

        # 读取和处理音频数据
        audio_data = await audio_file.read()
        samples, sample_rate = audio_processor.load_audio_from_bytes(audio_data, audio_file.filename)

        # 说话人识别
        speaker_name, similarity = speech_processor.identify_speaker_from_audio(
            samples, sample_rate, threshold
        )

        if speaker_name:
            return APIResponse(
                success=True,
                message="说话人识别成功",
                data={
                    "speaker_name": speaker_name,
                    "similarity": float(similarity),
                    "threshold": threshold or settings.SPEAKER_THRESHOLD,
                    "confidence": "high" if similarity > 0.8 else "medium" if similarity > 0.6 else "low"
                }
            )
        else:
            return APIResponse(
                success=True,
                message="未识别到已注册的说话人",
                data={
                    "speaker_name": None,
                    "similarity": 0.0,
                    "threshold": threshold or settings.SPEAKER_THRESHOLD,
                    "confidence": "none"
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"说话人识别失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"说话人识别失败: {str(e)}"
        )

@app.get("/api/speakers/{speaker_name}", response_model=APIResponse)
async def get_speaker_info(speaker_name: str):
    """
    获取特定说话人信息

    - **speaker_name**: 说话人姓名
    """
    try:
        speaker_manager = speech_processor.get_speaker_manager()
        speaker_info = speaker_manager.get_speaker_info(speaker_name)

        if not speaker_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"说话人 {speaker_name} 不存在"
            )

        return APIResponse(
            success=True,
            message=f"找到说话人 {speaker_name}",
            data=speaker_info
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取说话人信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取说话人信息失败: {str(e)}"
        )

@app.delete("/api/speakers/{speaker_name}", response_model=APIResponse)
async def delete_speaker(speaker_name: str):
    """
    删除说话人

    - **speaker_name**: 说话人姓名
    """
    try:
        speaker_manager = speech_processor.get_speaker_manager()
        success = speaker_manager.remove_speaker(speaker_name)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"说话人 {speaker_name} 不存在"
            )

        return APIResponse(
            success=True,
            message=f"说话人 {speaker_name} 已删除"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除说话人失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除说话人失败: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )