"""
语音识别服务主应用入口
"""
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import uvicorn
import logging
import asyncio
import time

from app.api import asr, speaker, comprehensive, queue_example, pipeline, websocket_stream
from app.config import settings
from app.core.queue import get_queue_manager, shutdown_queue_manager
from app.core.request_manager import get_request_manager
from app.core.websocket_manager import cleanup_websocket_manager
from app.core.model import initialize_models, get_model_info
from app.services.db import initialize_database

# 导入新的日志和监控模块
from app.utils.logging_config import configure_logging, get_logger
from app.utils.metrics import setup_instrumentator, get_metrics, metrics_collector

# 配置结构化日志
configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="语音识别服务",
    description="基于Sherpa-ONNX的智能语音识别与声纹识别服务",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 设置Prometheus监控
if settings.ENABLE_METRICS:
    instrumentator = setup_instrumentator(app)
    logger.info("Prometheus监控已启用", endpoint="/metrics")

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

# 自定义监控端点
@app.get("/metrics", response_class=PlainTextResponse, tags=["监控"])
async def metrics_endpoint():
    """Prometheus指标端点"""
    return get_metrics()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """请求日志中间件"""
    start_time = time.time()
    
    # 记录请求开始
    logger.info(
        "HTTP请求开始",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown"
    )
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # 记录响应
        logger.info(
            "HTTP请求完成",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration=duration
        )
        
        # 记录监控指标
        if settings.ENABLE_METRICS:
            metrics_collector.record_api_request(
                method=request.method,
                endpoint=str(request.url.path),
                status_code=response.status_code,
                duration=duration
            )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        # 记录错误
        logger.error(
            "HTTP请求失败",
            method=request.method,
            url=str(request.url),
            error=str(e),
            error_type=type(e).__name__,
            duration=duration
        )
        
        # 记录错误指标
        if settings.ENABLE_METRICS:
            metrics_collector.record_error("http_middleware", type(e).__name__)
        
        raise

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

        # 获取模型状态
        model_info = get_model_info()

        return {
            "status": health_status["status"],
            "service": "speech-recognition-service",
            "version": "0.1.0",
            "models_dir": settings.MODELS_DIR,
            "model_status": {
                "is_initialized": model_info.get("is_initialized", False),
                "model_type": model_info.get("model_type", "none"),
                "asr_loaded": model_info.get("asr_loaded", False),
                "vad_loaded": model_info.get("vad_loaded", False),
                "speaker_loaded": model_info.get("speaker_loaded", False),
                "punctuation_loaded": model_info.get("punctuation_loaded", False),
                "use_gpu": model_info.get("use_gpu", False),
                "preload_enabled": settings.ENABLE_MODEL_PRELOAD
            },
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

@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    try:
        request_manager = await get_request_manager()
        return request_manager.get_stats()
    except Exception as e:
        logger.error("获取统计信息失败", error=str(e))
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

        # 预加载模型（如果启用）
        if settings.ENABLE_MODEL_PRELOAD:
            logger.info("正在预加载模型...")
            try:
                await initialize_models(
                    model_type=settings.DEFAULT_MODEL_TYPE,
                    use_gpu=settings.DEFAULT_USE_GPU,
                    enable_vad=settings.DEFAULT_ENABLE_VAD,
                    enable_speaker_id=settings.DEFAULT_ENABLE_SPEAKER_ID,
                    enable_punctuation=settings.DEFAULT_ENABLE_PUNCTUATION
                )
                logger.info("模型预加载完成",
                           model_type=settings.DEFAULT_MODEL_TYPE,
                           use_gpu=settings.DEFAULT_USE_GPU,
                           enable_vad=settings.DEFAULT_ENABLE_VAD,
                           enable_speaker_id=settings.DEFAULT_ENABLE_SPEAKER_ID,
                           enable_punctuation=settings.DEFAULT_ENABLE_PUNCTUATION)
            except Exception as e:
                logger.error("模型预加载失败，服务将退出", error=str(e), error_type=type(e).__name__)
                raise
        else:
            logger.info("模型预加载已禁用，需要手动初始化模型")

        # 启动系统指标更新任务
        if settings.ENABLE_SYSTEM_METRICS:
            asyncio.create_task(update_system_metrics_task())
            logger.info("系统指标更新任务已启动")

        logger.info("语音识别服务启动完成",
                   enable_metrics=settings.ENABLE_METRICS,
                   enable_system_metrics=settings.ENABLE_SYSTEM_METRICS,
                   model_preload=settings.ENABLE_MODEL_PRELOAD)
        
    except Exception as e:
        logger.error("服务启动失败", error=str(e), error_type=type(e).__name__)
        raise

async def update_system_metrics_task():
    """定期更新系统指标的后台任务"""
    while True:
        try:
            if settings.ENABLE_SYSTEM_METRICS:
                metrics_collector.update_system_metrics()
            await asyncio.sleep(settings.METRICS_UPDATE_INTERVAL)
        except Exception as e:
            logger.error("更新系统指标失败", error=str(e))
            await asyncio.sleep(settings.METRICS_UPDATE_INTERVAL)

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
        logger.error("服务关闭过程中出现错误", error=str(e), error_type=type(e).__name__)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )