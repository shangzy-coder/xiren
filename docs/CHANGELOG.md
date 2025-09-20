# 变更日志

## [0.1.0] - 2024-01-XX

### 新增功能
- ✨ 基于Sherpa-ONNX的语音识别功能
- 👤 声纹识别和说话人管理
- 🔄 WebSocket实时音频流处理
- 📊 Prometheus监控指标集成
- 🚀 异步队列处理系统
- 🗄️ PostgreSQL + pgvector声纹存储
- 📦 Docker容器化支持

### API端点
- `POST /api/v1/asr/transcribe` - 音频转录
- `POST /api/v1/speaker/register` - 声纹注册
- `POST /api/v1/speaker/identify` - 声纹识别
- `POST /api/v1/process` - 综合处理
- `WS /api/v1/websocket/stream` - 实时流处理

### 测试
- 🧪 完整的测试套件（单元测试、API测试、集成测试）
- 📈 80%以上的代码覆盖率
- 🔧 pytest配置和fixture

### 文档
- 📚 完整的API文档
- 📖 部署和监控指南
- 🧪 测试文档

### 技术栈
- FastAPI + Uvicorn
- PostgreSQL + pgvector
- MinIO对象存储
- Prometheus监控
- Docker容器化

---

## 格式说明

本变更日志遵循 [Keep a Changelog](https://keepachangelog.com/) 格式。

### 类型
- `新增功能` - 新功能
- `变更` - 现有功能的变更
- `废弃` - 即将移除的功能
- `移除` - 已移除的功能
- `修复` - Bug修复
- `安全` - 安全相关修复
