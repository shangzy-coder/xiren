# 简单语音识别服务

基于Sherpa-ONNX的轻量级语音识别和说话人识别服务。

## 功能特点

- ✅ **语音识别**: 支持中英文等多语言识别
- ✅ **说话人识别**: 自动识别和标记说话人
- ✅ **轻量化**: 去除复杂组件，专注核心功能
- ✅ **易部署**: 最小依赖，快速启动
- ✅ **易迁移**: 数据存储简单，便于备份和迁移

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

```bash
python download_models.py
```

### 3. 启动服务

```bash
python main.py
```

服务将在 http://localhost:8000 启动，API文档位于 http://localhost:8000/docs

## API接口

### 语音识别
```bash
POST /api/recognize
```
上传音频文件，获取识别结果，包括文本、情感、说话人等信息。

### 说话人注册
```bash
POST /api/speakers/register
```
上传说话人音频样本，注册到系统中。

### 说话人列表
```bash
GET /api/speakers
```
获取所有已注册的说话人。

### 说话人识别
```bash
POST /api/speakers/identify
```
上传音频，识别其中的说话人。

## 配置说明

通过环境变量配置：

```bash
# 服务配置
HOST=0.0.0.0
PORT=8000

# 模型配置
USE_GPU=false  # 是否使用GPU

# 说话人配置
STORAGE_TYPE=memory  # memory 或 file
SPEAKER_THRESHOLD=0.6  # 相似度阈值
```

## 支持的音频格式

- WAV
- MP3
- FLAC
- M4A
- OGG
- AMR

## 架构说明

- **SpeechProcessor**: 语音处理核心，整合ASR、VAD、说话人识别
- **SpeakerManager**: 说话人管理，支持内存和文件存储
- **AudioProcessor**: 音频预处理工具

## 数据存储

- **内存模式**: 向量数据存储在内存中，重启后重新加载
- **文件模式**: 向量数据持久化到文件中，便于迁移

## 性能优化

- 自动语音活动检测 (VAD)
- 批处理识别
- GPU加速支持
- 并行处理