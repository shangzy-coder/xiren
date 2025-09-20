"""
语音识别服务主应用入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from app.api import asr, speaker, comprehensive, queue_example, pipeline, websocket_stream
from app.config import settings
from app.core.queue import get_queue_manager, shutdown_queue_manager
from app.core.request_manager import get_request_manager
from app.core.websocket_manager import cleanup_websocket_manager
from app.services.db import initialize_database

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
app.include_router(comprehensive.router, prefix="/api/v1", tags=["综合处理"])
app.include_router(pipeline.router, prefix="/api/v1/pipeline", tags=["语音处理流水线"])
app.include_router(queue_example.router, prefix="/api/v1/queue", tags=["队列系统示例"])
app.include_router(websocket_stream.router, prefix="/api/v1/websocket", tags=["WebSocket实时通信"])

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
    try:
        # 获取请求管理器状态
        request_manager = await get_request_manager()
        health_status = request_manager.get_health_status()
        
        return {
            "status": health_status["status"],
            "service": "speech-recognition-service", 
            "version": "0.1.0",
            "models_dir": settings.MODELS_DIR,
            "queue_health": health_status,
            "concurrent_config": {
                "max_workers": settings.THREAD_POOL_SIZE,
                "max_queue_size": settings.MAX_QUEUE_SIZE,
                "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS
            }
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "service": "speech-recognition-service",
            "version": "0.1.0",
            "error": str(e)
        }

@app.get("/metrics")
async def get_metrics():
    """获取系统指标"""
    try:
        request_manager = await get_request_manager()
        return request_manager.get_stats()
    except Exception as e:
        logger.error(f"获取指标失败: {e}")
        return {"error": str(e)}

@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("语音识别服务启动中...")
    
    try:
        # 初始化数据库
        logger.info("正在初始化数据库...")
        db_service = await initialize_database()
        if db_service:
            logger.info("数据库初始化成功")
        else:
            logger.warning("数据库初始化失败，服务将以降级模式运行")
        
        # 初始化队列管理器
        logger.info("正在初始化队列管理器...")
        queue_manager = await get_queue_manager()
        logger.info("队列管理器初始化成功")
        
        # 初始化请求管理器
        logger.info("正在初始化请求管理器...")
        request_manager = await get_request_manager()
        logger.info("请求管理器初始化成功")
        
        logger.info("语音识别服务启动完成")
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    logger.info("语音识别服务正在关闭...")
    
    try:
        # 关闭队列管理器
        logger.info("正在关闭队列管理器...")
        await shutdown_queue_manager()
        logger.info("队列管理器已关闭")
        
        # 关闭WebSocket管理器
        logger.info("正在关闭WebSocket管理器...")
        await cleanup_websocket_manager()
        logger.info("WebSocket管理器已关闭")
        
        logger.info("语音识别服务已关闭")
        
    except Exception as e:
        logger.error(f"服务关闭过程中出现错误: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )