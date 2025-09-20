# 🎙️ Xiren - 智能语音识别与声纹识别系统

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://docker.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-orange.svg)](https://github.com/pgvector/pgvector)

Xiren是一个基于**异步队列架构**的现代化语音处理系统，提供语音识别(ASR)、声纹识别、说话人分离等功能。采用微服务化设计，每个处理步骤都是独立的队列任务，支持高并发和水平扩展。

## ✨ 核心特性

### 🚀 异步流水线处理
- **真正的异步架构**：VAD、ASR、Speaker识别分别作为独立队列任务
- **智能流水线编排**：自动协调各处理阶段的执行顺序
- **灵活配置**：用户可选择启用的处理阶段
- **错误隔离**：单个阶段失败不影响整个系统

### 🎯 语音处理功能
- **语音识别(ASR)**：基于Sherpa-ONNX，支持多种模型
- **声纹识别**：注册、识别、搜索说话人身份
- **说话人分离**：多说话人场景的语音分离
- **VAD检测**：智能的语音活动检测
- **批量处理**：支持大规模音频文件处理

### 🏗️ 技术架构
- **异步队列系统**：基于asyncio和ThreadPoolExecutor
- **数据库存储**：PostgreSQL + pgvector向量搜索
- **GPU加速**：CUDA支持，自动CPU/GPU适配
- **Docker部署**：容器化部署，支持GPU和CPU模式
- **RESTful API**：完整的API接口和文档

## 🔧 快速开始

### 环境要求
- Python 3.12+
- Docker & Docker Compose
- NVIDIA GPU (可选，支持CUDA加速)
- PostgreSQL 15+ (自动部署)

### 1. 克隆仓库
```bash
git clone https://github.com/shangzy-coder/xiren.git
cd xiren
```

### 2. 启动服务
```bash
# GPU模式 (推荐)
docker-compose -f docker-compose.gpu.yml up -d

# CPU模式
docker-compose -f docker-compose.cpu.yml up -d
```

### 3. 验证部署
```bash
curl http://localhost:8002/health
```

## 📚 API文档

服务启动后，访问以下地址查看完整API文档：
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

### 核心API端点

#### 🎙️ 异步流水线处理 (推荐)
```bash
# 提交语音处理流水线
POST /api/v1/pipeline/submit
curl -X POST "http://localhost:8002/api/v1/pipeline/submit" \
  -F "audio_file=@speech.wav" \
  -F "enable_vad=true" \
  -F "enable_asr=true" \
  -F "enable_speaker_id=true" \
  -F "priority=normal"

# 查看处理状态
GET /api/v1/pipeline/status/{pipeline_id}

# 获取处理结果
GET /api/v1/pipeline/result/{pipeline_id}
```

#### 🗣️ 语音识别
```bash
# 异步语音识别
POST /api/v1/asr/transcribe-async

# 同步语音识别 (简单场景)
POST /api/v1/asr/transcribe
```

#### 👤 声纹识别
```bash
# 注册说话人
POST /api/v1/speaker/register

# 识别说话人
POST /api/v1/speaker/identify

# 说话人分离
POST /api/v1/speaker/diarize
```

## 🏗️ 系统架构

### 异步流水线架构
```
Client Request
     ↓
流水线编排器 (PipelineOrchestrator)
     ↓
┌─────────────────────────────────────┐
│  Independent Queue Tasks            │
├─────────────────────────────────────┤
│ 🔄 音频预处理任务                    │
│ 🎙️ VAD语音活动检测任务               │
│ 📝 ASR语音识别任务                   │
│ 🔍 声纹识别任务                      │
│ 👥 说话人分离任务                    │
└─────────────────────────────────────┘
     ↓
Final Results
```

### 技术栈
- **后端框架**: FastAPI + uvicorn
- **异步处理**: asyncio + ThreadPoolExecutor
- **语音引擎**: Sherpa-ONNX
- **数据库**: PostgreSQL + pgvector
- **容器化**: Docker + Docker Compose
- **GPU加速**: CUDA + onnxruntime-gpu

## 📊 性能特性

### 并发处理能力
- **异步队列**: 支持数千个并发请求
- **任务优先级**: LOW/NORMAL/HIGH/URGENT四级优先级
- **智能重试**: 指数退避重试机制
- **负载均衡**: 自动任务分发和负载均衡

### 处理性能
- **GPU加速**: CUDA支持，性能提升5-10倍
- **批量处理**: 优化的批量音频处理
- **内存优化**: 流式处理，降低内存占用
- **缓存机制**: 声纹特征缓存，提升识别速度

## 🔧 配置说明

### 环境变量配置
```bash
# AI模型配置
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
PERPLEXITY_API_KEY=your_key

# 数据库配置
DATABASE_URL=postgresql+asyncpg://speech_user:speech_pass@localhost:5432/speech_recognition

# 并发配置
MAX_WORKERS=8
THREAD_POOL_SIZE=8
MAX_QUEUE_SIZE=1000
MAX_CONCURRENT_REQUESTS=50
```

### 模型配置
```bash
# 配置AI模型
python -m app.config models --setup

# 或使用MCP工具
task-master models --set-main claude-3-5-sonnet-20241022
```

## 📁 项目结构

```
xiren/
├── app/
│   ├── api/                    # API路由
│   │   ├── asr.py             # 语音识别API
│   │   ├── speaker.py         # 声纹识别API
│   │   ├── pipeline.py        # 流水线API
│   │   └── comprehensive.py   # 综合处理API
│   ├── core/                   # 核心功能
│   │   ├── model.py           # 模型管理
│   │   ├── pipeline.py        # 流水线编排器
│   │   ├── queue.py           # 异步队列系统
│   │   ├── speaker_pool.py    # 声纹池管理
│   │   └── vad.py             # 语音活动检测
│   ├── services/               # 服务层
│   │   ├── db.py              # 数据库服务
│   │   └── storage.py         # 存储服务
│   └── utils/                  # 工具函数
├── docs/                       # 文档
├── scripts/                    # 部署脚本
├── docker-compose.gpu.yml      # GPU部署配置
├── docker-compose.cpu.yml      # CPU部署配置
└── requirements.txt            # 依赖管理
```

## 🚀 部署指南

### Docker部署 (推荐)
```bash
# 1. 克隆仓库
git clone https://github.com/shangzy-coder/xiren.git
cd xiren

# 2. 选择部署模式
# GPU模式 (需要NVIDIA Docker支持)
docker-compose -f docker-compose.gpu.yml up -d

# CPU模式
docker-compose -f docker-compose.cpu.yml up -d

# 3. 检查服务状态
docker-compose ps
curl http://localhost:8002/health
```

### 本地开发
```bash
# 1. 创建虚拟环境
conda create -n xiren python=3.12
conda activate xiren

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动数据库
docker-compose up -d postgres

# 4. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

## 🤝 贡献指南

我们欢迎社区贡献！请参考以下步骤：

1. Fork 这个仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📝 更新日志

### v0.3.0 (2025-09-20)
- ✨ 重新设计为真正的异步流水线处理系统
- 🚀 每个处理步骤(VAD/ASR/Speaker)都是独立队列任务
- 📊 新增流水线API和实时状态监控
- 🔧 完善的错误处理和重试机制

### v0.2.0 (2025-09-19)
- ✨ 集成异步队列系统到ASR接口
- 🚀 完成并发处理系统构建
- 📊 新增健康检查和指标监控
- 🔧 优化任务优先级和重试机制

### v0.1.0 (2025-09-18)
- 🎉 初始版本发布
- ✨ 基础语音识别和声纹识别功能
- 🗄️ PostgreSQL + pgvector集成
- 🐳 Docker部署支持

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) - 语音识别引擎
- [pgvector](https://github.com/pgvector/pgvector) - PostgreSQL向量扩展
- [FastAPI](https://fastapi.tiangolo.com/) - 现代化API框架

---

⭐ 如果这个项目对你有帮助，请给我们一个Star！
