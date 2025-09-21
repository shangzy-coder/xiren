# 音频格式支持文档

## 📋 支持的文件格式

### 🎵 音频格式
- **MP3** (.mp3) - MPEG Audio Layer 3
- **WAV** (.wav) - Waveform Audio File Format
- **M4A** (.m4a) - MPEG-4 Audio
- **FLAC** (.flac) - Free Lossless Audio Codec
- **OGG** (.ogg) - Ogg Vorbis
- **AMR** (.amr) - Adaptive Multi-Rate
- **MPGA** (.mpga) - MPEG Audio

### 🎬 视频格式（提取音频）
- **MP4** (.mp4) - MPEG-4 Video
- **MOV** (.mov) - QuickTime Movie
- **WEBM** (.webm) - WebM Video
- **MPEG** (.mpeg) - MPEG Video

## 🔧 技术实现

### 音频预处理流程
1. **格式验证** - 检查文件扩展名和MIME类型
2. **格式转换** - 使用FFmpeg转换为WAV格式
3. **重采样** - 转换为模型所需的采样率（默认16kHz）
4. **声道处理** - 转换为单声道
5. **编码转换** - 转换为32位浮点PCM格式

### 性能优化
- **异步处理** - 使用asyncio进行非阻塞音频转换
- **临时文件管理** - 自动清理临时文件，防止磁盘空间泄漏
- **内存管理** - 大文件处理优化，支持流式处理
- **错误恢复** - 详细的错误分类和处理策略

## 📊 使用限制

### 文件大小限制
- **最大文件大小**: 100MB（可配置）
- **建议文件大小**: 小于50MB以获得最佳性能

### 音频质量要求
- **采样率**: 支持8kHz-48kHz，自动转换为16kHz
- **声道数**: 支持单声道和立体声，自动转换为单声道
- **时长限制**: 建议单个文件不超过30分钟

## 🚀 API使用示例

### 语音识别接口
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/asr/transcribe' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@audio.mp3;type=audio/mpeg' \
  -F 'enable_vad=true' \
  -F 'enable_speaker_id=false'
```

### 声纹识别接口
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/speaker/identify' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'audio_file=@voice.wav;type=audio/wav' \
  -F 'threshold=0.8'
```

### 综合处理接口
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/comprehensive/process' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'audio_file=@recording.m4a;type=audio/mp4' \
  -F 'enable_asr=true' \
  -F 'enable_speaker_id=true' \
  -F 'enable_vad=true'
```

## ⚠️ 错误处理

### 常见错误信息
- **不支持的文件格式**: `不支持的文件格式。支持的格式: wav, mp3, flac, m4a, ogg, mpga, amr, mp4, mov, mpeg, webm`
- **文件过大**: `音频文件过大: {size}字节 > {max_size}字节`
- **文件损坏**: `音频文件处理失败: Invalid data found`
- **转换失败**: `音频转换失败: {error_details}`

### 错误代码
- **400 Bad Request**: 文件格式不支持或文件损坏
- **413 Payload Too Large**: 文件大小超过限制
- **500 Internal Server Error**: 服务器内部错误

## 🔍 故障排除

### 常见问题
1. **文件上传失败**
   - 检查文件格式是否在支持列表中
   - 确认文件大小不超过限制
   - 验证文件是否损坏

2. **转换速度慢**
   - 检查文件大小，建议压缩大文件
   - 确认服务器资源充足
   - 考虑使用更高效的音频格式（如MP3）

3. **识别准确率低**
   - 确保音频质量良好（无噪音、清晰）
   - 建议使用16kHz采样率的音频
   - 避免背景音乐和多人同时说话

## 📈 性能监控

系统提供音频处理统计信息：
- 总处理文件数
- 平均处理时间
- 平均文件大小
- 成功率
- 处理吞吐量

可通过健康检查接口获取详细信息：
```bash
curl http://localhost:8000/health
```

## 🔄 版本更新

### v0.1.0 (当前版本)
- ✅ 支持11种音频/视频格式
- ✅ 异步音频处理
- ✅ 自动临时文件管理
- ✅ 详细错误处理和日志
- ✅ 性能监控和统计

### 计划中的功能
- 🔄 更多视频格式支持（AVI, MKV等）
- 🔄 批量文件处理
- 🔄 音频质量自动优化
- 🔄 实时音频流处理
