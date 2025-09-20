# 日志和监控使用指南

本文档介绍语音识别服务中集成的日志和监控功能。

## 功能概述

### 日志系统
- **结构化日志**: 使用 `structlog` 提供结构化、可搜索的日志
- **环境适配**: 开发环境使用彩色控制台输出，生产环境使用JSON格式
- **上下文丰富**: 自动添加时间戳、日志级别、模块名等信息
- **装饰器支持**: 提供API请求和模型操作的自动日志记录

### 监控系统
- **Prometheus集成**: 提供标准的Prometheus指标格式
- **全面指标**: 覆盖API请求、语音识别、WebSocket连接、系统资源等
- **自动收集**: 支持自动系统资源监控和定期更新
- **FastAPI集成**: 自动记录HTTP请求指标

## 配置说明

### 环境变量

```bash
# 日志配置
LOG_LEVEL=INFO                    # 日志级别: DEBUG, INFO, WARNING, ERROR
ENVIRONMENT=development           # 环境: development, production

# 监控配置
ENABLE_METRICS=true              # 是否启用Prometheus监控
ENABLE_SYSTEM_METRICS=true       # 是否启用系统资源监控
METRICS_UPDATE_INTERVAL=30       # 系统指标更新间隔（秒）
```

### 配置文件

日志和监控配置存储在 `app/config.py` 中，可通过环境变量覆盖。

## 使用方式

### 结构化日志

```python
from app.utils.logging_config import get_logger

# 获取logger实例
logger = get_logger(__name__)

# 基本日志记录
logger.info("操作完成", user_id=123, action="login")
logger.warning("性能警告", response_time=2.5, threshold=2.0)
logger.error("处理失败", error=str(e), error_type=type(e).__name__)

# 使用装饰器自动记录API请求
from app.utils.logging_config import log_request_response

@log_request_response
async def my_api_endpoint():
    # API逻辑
    return {"status": "success"}
```

### 监控指标

```python
from app.utils.metrics import metrics_collector

# 记录API请求
metrics_collector.record_api_request("POST", "/api/v1/transcribe", 200, 1.23)

# 记录语音识别
metrics_collector.record_speech_recognition("sherpa-onnx", 2.5, "success")

# 记录WebSocket事件
metrics_collector.record_websocket_connection('connect')
metrics_collector.record_websocket_message('inbound', 'audio')

# 记录错误
metrics_collector.record_error("transcription", "ModelError")
```

## 监控端点

### 指标查询

```bash
# 获取Prometheus格式指标
curl http://localhost:8000/metrics

# 获取系统统计信息
curl http://localhost:8000/stats

# 健康检查
curl http://localhost:8000/health
```

### 主要指标

#### API相关指标
- `api_requests_total`: API请求总数（按方法、端点、状态码分组）
- `api_request_duration_seconds`: API请求持续时间分布

#### 语音识别指标
- `speech_recognition_requests_total`: 语音识别请求总数
- `speech_recognition_duration_seconds`: 语音识别处理时间分布
- `audio_processing_duration_seconds`: 音频预处理时间分布

#### WebSocket指标
- `websocket_connections_active`: 当前活跃WebSocket连接数
- `websocket_messages_total`: WebSocket消息总数
- `websocket_connection_duration_seconds`: WebSocket连接持续时间分布

#### 系统资源指标
- `system_memory_usage_bytes`: 系统内存使用量
- `system_cpu_usage_percent`: CPU使用率
- `gpu_memory_usage_bytes`: GPU内存使用量（如果可用）

#### 错误指标
- `errors_total`: 错误总数（按组件和错误类型分组）

## 日志格式

### 开发环境（控制台输出）
```
2025-09-20T15:08:02.618841Z [info] API请求处理 [api_module] method=POST endpoint=/api/v1/test duration=0.123 status_code=200
```

### 生产环境（JSON格式）
```json
{
  "event": "API请求处理",
  "level": "info",
  "logger": "api_module",
  "timestamp": "2025-09-20T15:08:02.618841Z",
  "method": "POST",
  "endpoint": "/api/v1/test",
  "duration": 0.123,
  "status_code": 200
}
```

## 性能优化

### 日志优化
- 生产环境使用适当的日志级别（INFO或WARNING）
- 避免在高频路径中记录过多详细信息
- 使用异步日志处理减少I/O阻塞

### 监控优化
- 根据需要调整系统指标更新频率
- 在高负载环境中可考虑禁用详细的系统监控
- 使用标签过滤减少指标存储开销

## 集成建议

### Prometheus + Grafana
1. 配置Prometheus抓取 `/metrics` 端点
2. 在Grafana中创建仪表板监控关键指标
3. 设置告警规则监控异常情况

### ELK Stack
1. 配置Logstash或Fluentd收集JSON格式日志
2. 在Elasticsearch中建立索引
3. 使用Kibana创建日志分析仪表板

### 示例Prometheus配置
```yaml
scrape_configs:
  - job_name: 'speech-recognition-service'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

## 故障排除

### 常见问题

1. **指标不更新**
   - 检查 `ENABLE_METRICS` 环境变量
   - 确认系统指标更新任务正常运行

2. **日志格式问题**
   - 检查 `ENVIRONMENT` 环境变量设置
   - 确认structlog配置正确

3. **系统指标缺失**
   - 检查psutil依赖是否安装
   - 确认 `ENABLE_SYSTEM_METRICS` 设置

4. **WebSocket指标异常**
   - 检查WebSocket管理器集成
   - 确认连接和断开事件正确记录

## 最佳实践

1. **日志记录**
   - 使用有意义的事件名称
   - 包含足够的上下文信息
   - 避免记录敏感信息

2. **指标命名**
   - 使用标准的Prometheus命名约定
   - 合理使用标签进行分组
   - 避免高基数标签

3. **性能考虑**
   - 在生产环境中适当调整日志级别
   - 监控指标收集的性能影响
   - 定期清理历史数据

## 测试验证

运行测试脚本验证功能：

```bash
cd /workspace
python3 test_logging_monitoring.py
```

该脚本会测试：
- 结构化日志输出
- 各种监控指标收集
- Prometheus格式指标导出
- 系统资源监控

成功运行后会显示详细的测试结果和指标统计。