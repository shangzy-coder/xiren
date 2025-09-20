# API 概览

## 语音识别服务

本服务提供基于Sherpa-ONNX的智能语音识别与声纹识别功能。

### 主要功能

1. **语音识别 (ASR)**
   - 音频文件转文字
   - 实时语音流识别
   - 多语言支持
   - 批量处理

2. **声纹识别**
   - 说话人注册
   - 说话人识别
   - 声纹比较
   - 说话人管理

3. **综合处理**
   - 音频预处理
   - VAD语音活动检测
   - 说话人分离
   - 结果存储

4. **实时通信**
   - WebSocket流式处理
   - 队列系统
   - 并发处理
   - 监控指标

### API端点

#### 语音识别
- `POST /api/v1/asr/initialize` - 初始化ASR模型
- `POST /api/v1/asr/transcribe` - 音频转录
- `POST /api/v1/asr/transcribe-async` - 异步转录
- `GET /api/v1/asr/result/{task_id}` - 获取转录结果

#### 声纹识别
- `POST /api/v1/speaker/register` - 注册说话人
- `POST /api/v1/speaker/identify` - 识别说话人
- `GET /api/v1/speaker/list` - 获取说话人列表
- `GET /api/v1/speaker/info/{speaker_id}` - 获取说话人信息

#### 综合处理
- `POST /api/v1/process` - 综合音频处理

#### 流水线处理
- `POST /api/v1/pipeline/submit` - 提交流水线任务
- `GET /api/v1/pipeline/status/{pipeline_id}` - 获取流水线状态

#### WebSocket
- `WS /api/v1/websocket/stream` - 实时音频流处理

#### 监控
- `GET /health` - 健康检查
- `GET /metrics` - Prometheus指标

### 认证

当前版本不需要认证，生产环境建议添加API密钥或JWT认证。

### 错误处理

API使用标准HTTP状态码：
- 200: 成功
- 400: 请求错误
- 404: 资源不存在
- 500: 服务器错误

### 限制

- 最大文件大小: 100MB
- 支持的音频格式: WAV, MP3, FLAC
- 并发连接限制: 100个WebSocket连接

