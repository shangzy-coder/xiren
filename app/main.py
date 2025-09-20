"""
语音识别服务主应用入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from app.api import asr, speaker
from app.config import settings

# 配置日志
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="语音识别服务",
    description="基于Sherpa-ONNX的智能语音识别与声纹识别服务",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应设置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含API路由
app.include_router(asr.router, prefix="/api/v1/asr", tags=["语音识别"])
app.include_router(speaker.router, prefix="/api/v1/speaker", tags=["声纹识别"])

@app.get("/")
async def root():
    """根路径健康检查"""
    return {
        "message": "语音识别服务运行正常",
        "version": "0.1.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """详细健康检查"""
    return {
        "status": "healthy",
        "service": "speech-recognition-service",
        "version": "0.1.0",
        "models_dir": settings.MODELS_DIR
    }

@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("语音识别服务启动中...")
    # TODO: 初始化模型、数据库连接等
    logger.info("语音识别服务启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    logger.info("语音识别服务正在关闭...")
    # TODO: 清理资源
    logger.info("语音识别服务已关闭")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )