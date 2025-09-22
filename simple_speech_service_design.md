# 简单语音识别服务设计文档

## 项目概述

基于现有demo中的语音功能，创建一个轻量级的语音识别服务。去掉复杂的数据库、队列、监控等功能，专注于核心的语音识别、说话人识别和标记功能。

## 设计目标

- **简单性**: 去除不必要的复杂性，保持核心功能清晰
- **易迁移**: 向量数据存储在内存或简单文件中，便于迁移
- **快速启动**: 最小依赖，快速部署和运行
- **功能完整**: 提供语音识别、说话人列表管理、说话人标记

## 核心功能

### 1. 语音识别 (Speech Recognition)
- 支持多种音频格式 (wav, mp3, flac, m4a等)
- 使用Sherpa-ONNX SenseVoice模型
- 支持中文、英文、日文、韩文、粤语
- 自动语言检测
- 语音活动检测 (VAD)

### 2. 说话人列表管理 (Speaker List Management)
- 注册说话人: 通过上传音频样本注册说话人
- 列出已注册说话人
- 删除说话人
- 说话人信息查询

### 3. 说话人标记 (Speaker Tagging)
- 自动识别已注册说话人
- 为未注册说话人分配标签 (Speaker1, Speaker2等)
- 说话人相似度计算
- 实时说话人分离

## 技术架构

### 核心组件

```
simple_speech_service/
├── main.py              # 服务主入口
├── config.py            # 简单配置
├── speech_processor.py  # 语音处理核心类
├── speaker_manager.py   # 说话人管理类
├── audio_utils.py       # 音频处理工具
├── models/             # 模型文件目录
└── data/               # 数据存储目录
    ├── speakers.json   # 说话人注册信息
    └── embeddings/     # 说话人向量文件 (可选)
```

### 数据存储策略

1. **内存存储** (推荐):
   - 说话人向量存储在内存中
   - 注册信息存储在JSON文件中
   - 服务重启时重新加载

2. **文件存储** (备选):
   - 向量数据序列化存储在文件中
   - 便于迁移和备份

### API设计

#### REST API接口

```python
# 语音识别
POST /api/recognize
- 上传音频文件
- 返回: 识别文本、说话人标记、时间戳等

# 说话人注册
POST /api/speakers/register
- 上传说话人音频样本
- 参数: 说话人姓名
- 返回: 注册状态

# 说话人列表
GET /api/speakers
- 返回: 已注册说话人列表

# 说话人识别
POST /api/speakers/identify
- 上传音频文件
- 返回: 匹配的说话人及相似度

# 说话人分离
POST /api/speakers/diarize
- 上传音频文件
- 返回: 时间段内不同说话人的标记
```

### 核心类设计

#### SpeechProcessor 类
```python
class SpeechProcessor:
    def __init__(self, use_gpu=False):
        # 初始化ASR模型、VAD、标点模型

    def recognize(self, audio_data, sample_rate):
        # 语音识别主流程
        # 返回识别结果

    def detect_speech_segments(self, audio_data, sample_rate):
        # 语音段落检测
```

#### SpeakerManager 类
```python
class SpeakerManager:
    def __init__(self, storage_type='memory'):
        # 初始化说话人管理器
        # storage_type: 'memory' 或 'file'

    def register_speaker(self, name, audio_data, sample_rate):
        # 注册说话人
        # 提取向量并存储

    def identify_speaker(self, audio_data, sample_rate):
        # 识别说话人
        # 返回最匹配的说话人

    def list_speakers(self):
        # 返回已注册说话人列表

    def save_to_file(self):
        # 保存到文件 (文件存储模式)

    def load_from_file(self):
        # 从文件加载 (文件存储模式)
```

## 依赖管理

### 核心依赖
```
fastapi==0.104.1
uvicorn==0.24.0
sherpa-onnx==1.9.14
soundfile==0.12.1
librosa==0.10.1
numpy==1.24.3
pydantic==2.5.0
python-multipart==0.0.6
```

### 可选依赖
```
# GPU加速支持
onnxruntime-gpu

# 额外音频格式支持
pydub
```

## 配置设计

### 环境变量配置
```bash
# 服务配置
HOST=0.0.0.0
PORT=8000

# 模型配置
MODELS_DIR=./models
USE_GPU=false

# 说话人配置
SPEAKER_THRESHOLD=0.6
STORAGE_TYPE=memory  # memory 或 file

# 音频配置
SAMPLE_RATE=16000
SUPPORTED_FORMATS=wav,mp3,flac,m4a
```

## 部署和运行

### 快速启动
```bash
# 安装依赖
pip install -r requirements-simple.txt

# 下载模型 (可脚本化)
python download_models.py

# 启动服务
python main.py
```

### Docker部署
```dockerfile
FROM python:3.9-slim

COPY . /app
WORKDIR /app

RUN pip install -r requirements-simple.txt
RUN python download_models.py

EXPOSE 8000
CMD ["python", "main.py"]
```

## 优势特点

1. **轻量化**: 去除复杂组件，专注核心功能
2. **易部署**: 最小依赖，快速启动
3. **易迁移**: 数据存储简单，便于备份和迁移
4. **可扩展**: 架构清晰，便于后续功能扩展
5. **高性能**: 直接使用demo中优化过的核心算法

## 后续扩展方向

1. **数据库集成**: 如需持久化，可轻松添加数据库支持
2. **队列系统**: 高并发场景下可添加异步处理队列
3. **监控指标**: 生产环境可添加基础监控
4. **多模型支持**: 可扩展支持其他语音模型
5. **实时流处理**: 添加WebSocket支持实时语音处理

## 实现计划

1. **Phase 1**: 基础框架搭建
   - 创建项目结构
   - 实现基础配置和工具类

2. **Phase 2**: 核心功能实现
   - 实现SpeechProcessor类
   - 实现SpeakerManager类
   - 集成demo中的核心算法

3. **Phase 3**: API接口开发
   - 实现REST API接口
   - 添加错误处理和验证

4. **Phase 4**: 测试和优化
   - 功能测试
   - 性能优化
   - 文档完善

这个设计保持了现有demo的核心功能，同时大幅简化了架构，便于快速开发和部署。