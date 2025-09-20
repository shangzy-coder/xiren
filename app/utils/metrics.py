"""
Prometheus监控指标模块

提供应用程序的各种性能指标收集和暴露功能。
"""

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from typing import Dict, Any
import time
import psutil
import os

# 创建自定义指标注册表
REGISTRY = CollectorRegistry()

# API请求相关指标
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code'],
    registry=REGISTRY
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

# 语音识别相关指标
speech_recognition_requests = Counter(
    'speech_recognition_requests_total',
    'Total number of speech recognition requests',
    ['model_type', 'status'],
    registry=REGISTRY
)

speech_recognition_duration = Histogram(
    'speech_recognition_duration_seconds',
    'Speech recognition processing time in seconds',
    ['model_type'],
    registry=REGISTRY
)

audio_processing_duration = Histogram(
    'audio_processing_duration_seconds',
    'Audio preprocessing duration in seconds',
    ['operation'],
    registry=REGISTRY
)

# 声纹识别相关指标
speaker_recognition_requests = Counter(
    'speaker_recognition_requests_total',
    'Total number of speaker recognition requests',
    ['status'],
    registry=REGISTRY
)

speaker_recognition_duration = Histogram(
    'speaker_recognition_duration_seconds',
    'Speaker recognition processing time in seconds',
    registry=REGISTRY
)

speaker_database_size = Gauge(
    'speaker_database_size',
    'Number of speakers in the database',
    registry=REGISTRY
)

# WebSocket相关指标
websocket_connections = Gauge(
    'websocket_connections_active',
    'Number of active WebSocket connections',
    registry=REGISTRY
)

websocket_messages = Counter(
    'websocket_messages_total',
    'Total number of WebSocket messages',
    ['direction', 'message_type'],
    registry=REGISTRY
)

websocket_connection_duration = Histogram(
    'websocket_connection_duration_seconds',
    'WebSocket connection duration in seconds',
    registry=REGISTRY
)

# 系统资源指标
system_memory_usage = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes',
    ['type'],
    registry=REGISTRY
)

system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=REGISTRY
)

gpu_memory_usage = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id'],
    registry=REGISTRY
)

# 数据库相关指标
database_connections = Gauge(
    'database_connections_active',
    'Number of active database connections',
    registry=REGISTRY
)

database_query_duration = Histogram(
    'database_query_duration_seconds',
    'Database query duration in seconds',
    ['operation'],
    registry=REGISTRY
)

# 错误指标
error_count = Counter(
    'errors_total',
    'Total number of errors',
    ['component', 'error_type'],
    registry=REGISTRY
)

# 应用信息
app_info = Info(
    'app_info',
    'Application information',
    registry=REGISTRY
)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self._start_time = time.time()
        self.update_app_info()
    
    def update_app_info(self):
        """更新应用信息"""
        app_info.info({
            'version': '0.1.0',
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'pid': str(os.getpid()),
            'start_time': str(self._start_time)
        })
    
    def update_system_metrics(self):
        """更新系统资源指标"""
        try:
            # 内存使用
            memory = psutil.virtual_memory()
            system_memory_usage.labels(type='total').set(memory.total)
            system_memory_usage.labels(type='available').set(memory.available)
            system_memory_usage.labels(type='used').set(memory.used)
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage.set(cpu_percent)
            
            # GPU内存使用（如果有NVIDIA GPU）
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_memory_usage.labels(gpu_id=str(i)).set(gpu.memoryUsed * 1024 * 1024)  # MB to bytes
            except (ImportError, Exception):
                # 如果没有安装GPUtil或获取GPU信息失败，跳过GPU指标
                pass
                
        except Exception as e:
            error_count.labels(component='metrics_collector', error_type='system_metrics').inc()
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """记录API请求指标"""
        api_requests_total.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
        api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_speech_recognition(self, model_type: str, duration: float, status: str = 'success'):
        """记录语音识别指标"""
        speech_recognition_requests.labels(model_type=model_type, status=status).inc()
        if status == 'success':
            speech_recognition_duration.labels(model_type=model_type).observe(duration)
    
    def record_audio_processing(self, operation: str, duration: float):
        """记录音频处理指标"""
        audio_processing_duration.labels(operation=operation).observe(duration)
    
    def record_speaker_recognition(self, duration: float, status: str = 'success'):
        """记录声纹识别指标"""
        speaker_recognition_requests.labels(status=status).inc()
        if status == 'success':
            speaker_recognition_duration.observe(duration)
    
    def update_speaker_database_size(self, size: int):
        """更新声纹数据库大小"""
        speaker_database_size.set(size)
    
    def record_websocket_connection(self, action: str):
        """记录WebSocket连接事件"""
        if action == 'connect':
            websocket_connections.inc()
        elif action == 'disconnect':
            websocket_connections.dec()
    
    def record_websocket_message(self, direction: str, message_type: str):
        """记录WebSocket消息"""
        websocket_messages.labels(direction=direction, message_type=message_type).inc()
    
    def record_websocket_connection_duration(self, duration: float):
        """记录WebSocket连接持续时间"""
        websocket_connection_duration.observe(duration)
    
    def record_database_query(self, operation: str, duration: float):
        """记录数据库查询指标"""
        database_query_duration.labels(operation=operation).observe(duration)
    
    def update_database_connections(self, count: int):
        """更新数据库连接数"""
        database_connections.set(count)
    
    def record_error(self, component: str, error_type: str):
        """记录错误"""
        error_count.labels(component=component, error_type=error_type).inc()


# 全局指标收集器实例
metrics_collector = MetricsCollector()


def setup_instrumentator(app):
    """设置FastAPI Prometheus instrumentator"""
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="inprogress",
        inprogress_labels=True,
    )
    
    instrumentator.instrument(app).expose(app, endpoint="/metrics", tags=["监控"])
    
    # 添加自定义指标
    @instrumentator.add(
        metrics.requests(
            metric_name="http_requests",
            metric_doc="Total number of HTTP requests",
        )
    )
    def custom_http_requests(info: metrics.Info) -> None:
        metrics_collector.record_api_request(
            method=info.method,
            endpoint=info.modified_handler,
            status_code=info.response.status_code,
            duration=info.modified_duration
        )
    
    return instrumentator


def get_metrics():
    """获取所有指标数据"""
    # 更新系统指标
    metrics_collector.update_system_metrics()
    
    # 返回指标数据
    return generate_latest(REGISTRY)


class MetricsMiddleware:
    """自定义指标中间件"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    duration = time.time() - start_time
                    # 这里可以添加更多自定义指标记录
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)