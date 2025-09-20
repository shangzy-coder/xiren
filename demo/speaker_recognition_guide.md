# 说话人识别功能使用指南

## 功能概述

现在的语音识别系统已经集成了完整的说话人识别功能，能够：

1. **注册已知说话人**：为特定说话人创建语音特征库
2. **自动识别说话人**：
   - 已注册的说话人：显示注册的名字（如"张三"）
   - 未注册的说话人：显示标签（如"Speaker1", "Speaker2"）

## 使用方法

### 1. 注册说话人

```bash
# 注册说话人：张三
python speech_recognition.py audio.wav --register-speaker 张三 sample_audio.wav

# 注册多个说话人
python speech_recognition.py audio.wav --register-speaker 李四 sample_audio2.wav
```

### 2. 启用说话人识别

```bash
# 进行识别并显示说话人信息
python speech_recognition.py audio.wav --enable-speaker-id

# 同时启用所有功能
python speech_recognition.py audio.wav --use-gpu --enable-speaker-id --enable-punctuation
```

### 3. 注册+识别一体化

```bash
# 注册说话人并立即进行识别
python speech_recognition.py audio.wav --enable-speaker-id --register-speaker 张三 sample.wav
```

## 输出示例

### 未注册说话人
```
👤 说话人: Speaker1
👤 说话人: Speaker2
```

### 已注册说话人
```
👤 说话人: 张三
👤 说话人: 李四
```

## 技术实现

- **嵌入提取器**：使用3dspeaker模型提取192维说话人特征向量
- **说话人管理器**：管理注册的说话人数据库
- **相似度匹配**：通过余弦相似度进行说话人匹配
- **自动标签**：未匹配的说话人自动分配SpeakerN标签

## 注意事项

1. **GPU加速**：建议使用`--use-gpu`参数加速嵌入提取
2. **注册音频**：注册时使用的音频应该清晰、包含足够长的语音
3. **相似度阈值**：当前使用0.0阈值，可根据需要调整
4. **内存管理**：系统会为每个语音段落提取嵌入向量

## 命令行参数

```bash
--enable-speaker-id        # 启用说话人识别功能
--register-speaker NAME FILE  # 注册说话人：名字和音频文件
--use-gpu                  # 使用GPU加速
--enable-punctuation       # 启用标点处理
```

## 工作流程

1. **注册阶段**（可选）：
   - 加载说话人音频样本
   - 提取说话人嵌入向量
   - 存储到说话人数据库

2. **识别阶段**：
   - 加载待识别音频
   - 检测语音段落
   - 为每个段落提取嵌入
   - 搜索匹配的注册说话人
   - 返回说话人标签或注册名字
