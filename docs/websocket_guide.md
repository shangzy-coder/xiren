# WebSocket 实时语音识别指南

本指南介绍如何使用 Xiren 语音识别系统的 WebSocket 实时通信功能。

## 概述

Xiren 提供两个 WebSocket 端点用于实时语音识别：

1. **基础端点** (`/api/v1/asr/stream`) - 简单的实时识别功能
2. **增强端点** (`/api/v1/websocket/stream`) - 包含连接管理、统计监控和心跳检测的完整功能

## 增强 WebSocket API

### 连接地址
```
ws://localhost:8002/api/v1/websocket/stream
```

### 协议说明

#### 客户端发送消息

**音频数据**
```json
{
  "type": "audio",
  "data": "base64_encoded_audio_data"
}
```

**结束会话**
```json
{
  "type": "end"
}
```

**心跳检测**
```json
{
  "type": "ping",
  "timestamp": 1234567890.123
}
```

**请求统计信息**
```json
{
  "type": "get_stats"
}
```

#### 服务端返回消息

**连接确认**
```json
{
  "type": "connected",
  "message": "WebSocket连接已建立",
  "connection_id": "ws_1_192.168.1.100:12345",
  "model_status": {
    "asr_model": "ok",
    "vad_model": "ok", 
    "speaker_id": "ok"
  }
}
```

**识别结果**
```json
{
  "type": "transcription",
  "text": "识别的文字内容",
  "timestamp": 1.23,
  "speaker": "说话人ID",
  "language": "zh-cn",
  "emotion": "neutral",
  "confidence": 0.95
}
```

**心跳响应**
```json
{
  "type": "pong",
  "timestamp": 1234567890.123
}
```

**统计信息**
```json
{
  "type": "stats",
  "connection_id": "ws_1_192.168.1.100:12345",
  "messages_received": 150,
  "messages_sent": 75,
  "audio_chunks_processed": 120,
  "recognition_requests": 45,
  "errors": 2,
  "bytes_received": 1048576,
  "bytes_sent": 32768,
  "connected_duration": 300.5
}
```

**错误消息**
```json
{
  "type": "error",
  "message": "错误描述",
  "timestamp": 1234567890.123
}
```

**会话结束**
```json
{
  "type": "end",
  "message": "识别会话结束",
  "timestamp": 1234567890.123
}
```

## 音频格式要求

- **采样率**: 16kHz (推荐)
- **声道数**: 单声道 (mono)
- **数据格式**: 32位浮点数 (float32)
- **编码**: Base64编码
- **块大小**: 建议每次发送 1-3 秒的音频数据

## 使用示例

### Python 客户端示例

```python
import asyncio
import websockets
import json
import base64
import numpy as np

async def websocket_client():
    uri = "ws://localhost:8002/api/v1/websocket/stream"
    
    async with websockets.connect(uri) as websocket:
        # 等待连接确认
        response = await websocket.recv()
        data = json.loads(response)
        print(f"连接建立: {data}")
        
        # 发送音频数据
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        
        # 生成测试音频 (440Hz 正弦波)
        t = np.linspace(0, duration, samples, False)
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # 编码并发送
        audio_bytes = audio_data.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        message = {
            "type": "audio",
            "data": audio_base64
        }
        
        await websocket.send(json.dumps(message))
        
        # 接收识别结果
        response = await websocket.recv()
        result = json.loads(response)
        print(f"识别结果: {result}")
        
        # 发送结束信号
        await websocket.send(json.dumps({"type": "end"}))
        
        # 等待结束确认
        response = await websocket.recv()
        print(f"会话结束: {json.loads(response)}")

# 运行客户端
asyncio.run(websocket_client())
```

### JavaScript 客户端示例

```javascript
const websocket = new WebSocket('ws://localhost:8002/api/v1/websocket/stream');

websocket.onopen = function(event) {
    console.log('WebSocket连接已建立');
};

websocket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'connected':
            console.log('连接确认:', data);
            break;
        case 'transcription':
            console.log('识别结果:', data.text);
            break;
        case 'error':
            console.error('错误:', data.message);
            break;
        case 'end':
            console.log('会话结束:', data.message);
            break;
    }
};

// 发送音频数据
function sendAudio(audioData) {
    const message = {
        type: 'audio',
        data: btoa(String.fromCharCode.apply(null, new Uint8Array(audioData)))
    };
    websocket.send(JSON.stringify(message));
}

// 结束会话
function endSession() {
    websocket.send(JSON.stringify({type: 'end'}));
}
```

## 测试工具

项目提供了多个测试工具来验证 WebSocket 功能：

### 1. Python 测试客户端

**基础测试客户端**
```bash
python demo/websocket_test_client.py --server ws://localhost:8002/api/v1/asr/stream
```

**增强测试客户端**
```bash
python demo/enhanced_websocket_test.py --server ws://localhost:8002/api/v1/websocket/stream
python demo/enhanced_websocket_test.py --mode interactive
```

### 2. HTML 测试客户端

在浏览器中打开 `demo/websocket_test_client.html` 进行可视化测试。

## 监控和统计

### 获取连接统计信息

```bash
curl http://localhost:8002/api/v1/websocket/stream/stats
```

响应示例：
```json
{
  "service": "websocket-stream",
  "status": "healthy",
  "statistics": {
    "active_connections": 3,
    "total_messages": 1250,
    "total_audio_chunks": 890,
    "total_recognitions": 156,
    "total_errors": 5,
    "connections": {
      "ws_1_192.168.1.100:12345": {
        "client_address": "192.168.1.100:12345",
        "connected_at": "2025-09-20T14:30:00",
        "last_activity": "2025-09-20T14:35:30",
        "messages_sent": 45,
        "messages_received": 120,
        "audio_chunks_processed": 98,
        "recognition_requests": 23,
        "errors": 1,
        "bytes_received": 524288,
        "bytes_sent": 8192
      }
    }
  }
}
```

### 健康检查

```bash
curl http://localhost:8002/api/v1/websocket/stream/health
```

## 性能优化建议

### 1. 音频数据优化
- 使用合适的音频块大小 (1-3秒)
- 预处理音频到标准格式 (16kHz, 单声道)
- 避免发送过小或过大的音频块

### 2. 连接管理
- 实现客户端重连机制
- 使用心跳检测保持连接活跃
- 正确处理连接断开和错误

### 3. 错误处理
- 监听错误消息并适当响应
- 实现指数退避重试策略
- 记录和分析错误模式

## 故障排除

### 常见问题

**连接失败**
- 检查服务器是否运行
- 验证 WebSocket 地址是否正确
- 确认防火墙设置

**识别结果为空**
- 检查音频格式是否正确
- 确认模型已正确初始化
- 验证音频数据是否包含有效语音

**连接断开**
- 实现重连机制
- 检查网络稳定性
- 监控服务器资源使用情况

### 调试技巧

1. **启用详细日志**
   ```bash
   export LOG_LEVEL=DEBUG
   ```

2. **使用测试客户端验证**
   ```bash
   python demo/enhanced_websocket_test.py --mode interactive
   ```

3. **监控连接统计**
   ```bash
   curl http://localhost:8002/api/v1/websocket/stream/stats
   ```

## 限制和注意事项

- WebSocket 连接超时时间为 30 分钟
- 单个连接最大并发识别请求数为 10
- 音频数据大小限制为每块 1MB
- 服务器最大并发 WebSocket 连接数为 100

## 更多信息

- [API 参考文档](api_reference.md)
- [部署指南](deployment.md)
- [性能调优指南](performance_tuning.md)