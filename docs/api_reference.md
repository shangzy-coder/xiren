# 语音识别服务API参考文档

## 概述

本服务基于Sherpa-ONNX提供完整的语音处理能力，包括：
- 语音转文字 (ASR)
- 声纹识别 (Speaker Recognition)
- 说话人分离 (Speaker Diarization)
- 综合音频处理

## 基础信息

- **服务地址**: `http://localhost:8000`
- **API版本**: `v1`
- **支持格式**: WAV, MP3, FLAC, M4A, OGG
- **最大文件**: 100MB
- **采样率**: 16kHz (自动转换)

## API路由结构

```
/api/v1/
├── asr/                    # 语音识别专用接口
├── speaker/                # 声纹识别专用接口
├── process                 # 综合处理接口
├── quick-transcribe        # 快速转录
└── quick-speaker-id        # 快速声纹识别
```

## 1. 综合处理接口

### POST /api/v1/process

**描述**: 一站式音频处理，支持同时进行语音识别、声纹识别、说话人分离

**请求参数**:
```
Content-Type: multipart/form-data

audio_file: File (required)           # 音频文件
enable_asr: bool = true               # 启用语音转文字
enable_speaker_id: bool = true        # 启用声纹识别
enable_diarization: bool = false      # 启用说话人分离
enable_vad: bool = true               # 启用语音活动检测
speaker_threshold: float = null       # 声纹相似度阈值
language: str = "auto"                # 语言设置
save_to_database: bool = true         # 保存到数据库
```

**响应示例**:
```json
{
  "success": true,
  "session_id": "uuid-1234",
  "message": "音频综合处理完成",
  "transcription": {
    "success": true,
    "text": "这是识别出的完整文字内容",
    "segments": [
      {
        "text": "这是第一段",
        "start_time": 0.0,
        "end_time": 2.5,
        "confidence": 0.95
      }
    ],
    "statistics": {
      "total_duration": 10.5,
      "processing_time": 1.2
    }
  },
  "speaker_analysis": {
    "success": true,
    "identification": {
      "speaker_name": "张三",
      "similarity": 0.85,
      "confidence": "high"
    },
    "diarization": {
      "segments": [
        {
          "start": 0.0,
          "end": 5.2,
          "speaker": "Speaker_01",
          "duration": 5.2
        }
      ],
      "total_speakers": 2,
      "total_duration": 10.5
    }
  },
  "metadata": {
    "session_id": "uuid-1234",
    "filename": "audio.wav",
    "duration": 10.5,
    "processing_time": 1.8
  }
}
```

## 2. 快速接口

### POST /api/v1/quick-transcribe

**描述**: 快速语音转文字，只返回文字结果

**请求参数**:
```
Content-Type: multipart/form-data

audio_file: File (required)
language: str = "auto"
```

**响应示例**:
```json
{
  "success": true,
  "text": "识别出的完整文字内容",
  "language": "auto",
  "duration": 10.5,
  "confidence": 0.92
}
```

### POST /api/v1/quick-speaker-id

**描述**: 快速声纹识别，只返回说话人信息

**请求参数**:
```
Content-Type: multipart/form-data

audio_file: File (required)
threshold: float = null
```

**响应示例**:
```json
{
  "success": true,
  "speaker_name": "张三",
  "similarity": 0.85,
  "confidence": "high",
  "threshold": 0.75
}
```

## 3. 语音识别接口 (ASR)

### GET /api/v1/asr/status
获取ASR服务状态

### POST /api/v1/asr/initialize
初始化ASR模型

### POST /api/v1/asr/transcribe
离线音频文件识别

### POST /api/v1/asr/batch-transcribe
批量音频文件识别

### WebSocket /api/v1/asr/stream (传统端点)
实时音频流识别 (基础版本)

### WebSocket /api/v1/websocket/stream (增强端点)
增强的实时音频流识别，包含连接管理、统计监控和心跳检测

**WebSocket协议**:
```javascript
// 发送音频数据
{
  "type": "audio",
  "data": "base64_encoded_audio"
}

// 接收识别结果
{
  "type": "transcription",
  "text": "识别文字",
  "timestamp": 1.23,
  "speaker": "说话人"
}

// 结束会话
{
  "type": "end"
}

// 心跳检测
{
  "type": "ping",
  "timestamp": 1234567890.123
}

// 请求统计信息
{
  "type": "get_stats"
}
```

## 4. 声纹识别接口 (Speaker)

### GET /api/v1/speaker/status
获取声纹服务状态

### GET /api/v1/speaker/speakers
列出已注册说话人

### POST /api/v1/speaker/register
注册新说话人

### POST /api/v1/speaker/identify
识别说话人

### POST /api/v1/speaker/diarization
说话人分离

### POST /api/v1/speaker/search
声纹相似度搜索

### GET /api/v1/speaker/speakers/{name}
获取说话人详情

### DELETE /api/v1/speaker/speakers/{name}
删除说话人

## 5. 会话管理

### GET /api/v1/sessions/{session_id}
获取处理会话详情

### GET /api/v1/sessions
列出处理会话

## 6. WebSocket管理接口

### GET /api/v1/websocket/stream/health
WebSocket服务健康检查

### GET /api/v1/websocket/stream/stats
获取WebSocket连接统计信息

## 状态码说明

- `200` - 成功
- `400` - 请求参数错误
- `404` - 资源不存在
- `500` - 服务器内部错误
- `503` - 服务不可用 (模型未初始化)

## 错误响应格式

```json
{
  "detail": "错误描述信息"
}
```

## 使用示例

### Python示例

```python
import requests

# 综合处理
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/process',
        files={'audio_file': f},
        data={
            'enable_asr': True,
            'enable_speaker_id': True,
            'enable_diarization': False
        }
    )
    result = response.json()
    print(f"识别文字: {result['transcription']['text']}")
    print(f"说话人: {result['speaker_analysis']['identification']['speaker_name']}")
```

### curl示例

```bash
# 快速转录
curl -X POST "http://localhost:8000/api/v1/quick-transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@audio.wav" \
  -F "language=auto"

# 综合处理
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@audio.wav" \
  -F "enable_asr=true" \
  -F "enable_speaker_id=true" \
  -F "enable_diarization=false"
```

### JavaScript示例

```javascript
// 使用 fetch API
const formData = new FormData();
formData.append('audio_file', audioFile);
formData.append('enable_asr', 'true');
formData.append('enable_speaker_id', 'true');

fetch('http://localhost:8000/api/v1/process', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('识别文字:', data.transcription.text);
  console.log('说话人:', data.speaker_analysis.identification.speaker_name);
});
```

## 性能优化建议

1. **音频预处理**: 上传前转换为16kHz WAV格式可提高处理速度
2. **批量处理**: 使用batch-transcribe接口处理多个文件
3. **异步处理**: 对于大文件，建议使用后台任务处理
4. **缓存策略**: 相同音频文件可复用处理结果

## 限制说明

- 单个音频文件最大100MB
- 批量处理最多10个文件
- WebSocket连接超时时间为30分钟
- 并发请求限制为每秒50次

## 模型支持

### ASR模型
- SenseVoice (中文优化)
- Paraformer (通用)
- Whisper (多语言)

### 声纹模型
- 3DSpeaker (中文优化)
- WeSpeaker (通用)
- ECAPA-TDNN (高精度)

## 部署说明

```bash
# 启动服务
docker-compose up -d

# 查看服务状态
curl http://localhost:8000/health

# 查看API文档
访问 http://localhost:8000/docs
```

