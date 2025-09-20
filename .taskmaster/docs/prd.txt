# 语音识别服务 PRD — 第一版（无 Triton，使用 Sherpa-ONNX）

## 1. 背景与目标
提供一套企业级中文优先的语音识别+声纹管理服务：
- 支持**实时在线语音识别**（WebSocket/HTTP流）
- 支持**离线批量音频识别**
- 支持**声纹注册、存储与检索**
- 支持**Speaker Diarization（多说话人分离）**
- 初版**不引入 Triton**，在 Python 服务里直接加载 **Sherpa-ONNX** 模块，通过内建队列控制并发

## 2. 架构设计（第一版）

```mermaid
flowchart TD
    Client[客户端/Web/App] -->|HTTP/WebSocket| FastAPI[FastAPI API服务层]
    FastAPI -->|SQL/REST| PG[(PostgreSQL + pgvector)]
    FastAPI -->|MinIO API| MinIO[(MinIO 对象存储)]

    subgraph FastAPI[FastAPI API服务层]
    direction TB
    API[接口层 REST/WS] --> Queue[内建任务队列/批处理]
    Queue --> SherpaONNX[语音识别/声纹提取模型(3dspeaker)]
    SherpaONNX --> SpeakerPool[临时声纹池]
    end
```

## 3. 模块划分

| 模块 | 技术栈/说明 |
|------|-------------|
| **接口层** | FastAPI + Uvicorn，提供 REST/WS 接口 |
| **队列调度** | Python 内部队列（`asyncio.Queue`），控制并发批处理 |
| **模型推理模块** | Sherpa-ONNX，加载语音识别、声纹提取、VAD 模型 |
| **声纹管理** | - 声纹注册：存音频到 MinIO；提取 embedding 存 pgvector；存元数据到 PostgreSQL<br> - 声纹检索：先查 pgvector，若无匹配，放入临时声纹池 |
| **存储** | MinIO 存音频；PostgreSQL+pgvector 存信息与向量 |
| **监控日志** | Prometheus 客户端 / logging |

## 4. 功能流程（简述）

### 4.1 实时识别
1. 客户端通过 WebSocket 发送音频流
2. FastAPI 收流，送入 **内建队列**
3. 队列 worker 调用 Sherpa-ONNX 模型进行识别
4. 识别结果带时间戳返回客户端
5. 并行提取声纹 embedding → pgvector 检索 → 匹配或放入临时池 → 输出 `Speaker1/2` 标签

### 4.2 离线批量识别
1. 客户端上传音频文件
2. FastAPI 存 MinIO
3. 将音频分片（VAD），送入批处理队列
4. 识别完毕后合并结果+时间戳返回

### 4.3 声纹注册
1. 客户端上传音频
2. 存 MinIO → 提取 embedding → 存 pgvector & PostgreSQL

## 5. 项目骨架（建议）

```
project_root/
├── app/
│   ├── main.py            # FastAPI入口
│   ├── api/
│   │   ├── asr.py         # 语音识别接口
│   │   ├── speaker.py     # 声纹注册/检索接口
│   ├── core/
│   │   ├── queue.py       # 任务队列/worker调度
│   │   ├── model.py       # 模型加载与推理封装(Sherpa-ONNX)
│   │   ├── vad.py         # VAD分割工具
│   │   ├── speaker_pool.py# 临时声纹池管理
│   ├── services/
│   │   ├── storage.py     # MinIO上传/下载
│   │   ├── db.py          # PostgreSQL连接/pgvector检索
│   ├── config.py          # 配置文件
│   └── utils/             # 工具函数
├── models/                # 模型文件(.onnx)
├── requirements.txt
└── README.md
```

- `core/model.py`：封装 `load_model()` 和 `infer()` 方法（调用 Sherpa-ONNX）。后续如果换 Triton，这个文件只要改成 TritonClient 调用。
- `core/queue.py`：用 `asyncio.Queue` 或 `ThreadPoolExecutor` 管理并发。
- `core/speaker_pool.py`：在未匹配 pgvector 时维护临时声纹池。

## 6. 未来升级方向
- 替换模型调用模块为 TritonClient（即可切换为 Triton 架构）
- 用 Celery/RabbitMQ 做分布式队列
- 用 Kubernetes 部署多副本，实现水平扩展
