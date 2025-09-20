#!/usr/bin/env python3
"""
文档生成脚本
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加app模块到路径
sys.path.insert(0, '/workspace')

def generate_api_documentation():
    """生成API文档"""
    print("📚 生成API文档...")
    
    try:
        # 尝试导入FastAPI应用
        from app.main import app
        
        # 获取OpenAPI schema
        openapi_schema = app.openapi()
        
        # 保存OpenAPI schema到文件
        docs_dir = Path("/workspace/docs")
        docs_dir.mkdir(exist_ok=True)
        
        with open(docs_dir / "openapi.json", "w", encoding="utf-8") as f:
            json.dump(openapi_schema, f, ensure_ascii=False, indent=2)
        
        print("✅ OpenAPI schema已保存到 docs/openapi.json")
        
        # 生成API参考文档
        generate_api_reference(openapi_schema)
        
        return True
        
    except ImportError as e:
        print(f"❌ 无法导入应用: {e}")
        print("⚠️  在生产环境中需要安装所有依赖才能生成完整文档")
        
        # 生成基础文档结构
        generate_basic_docs()
        return False

def generate_api_reference(openapi_schema):
    """生成API参考文档"""
    print("📖 生成API参考文档...")
    
    doc_content = []
    doc_content.append("# API 参考文档")
    doc_content.append("")
    doc_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc_content.append(f"**API版本**: {openapi_schema.get('info', {}).get('version', 'unknown')}")
    doc_content.append("")
    doc_content.append("## 概述")
    doc_content.append("")
    doc_content.append(openapi_schema.get('info', {}).get('description', '语音识别服务API'))
    doc_content.append("")
    
    # 服务器信息
    if 'servers' in openapi_schema:
        doc_content.append("## 服务器")
        doc_content.append("")
        for server in openapi_schema['servers']:
            doc_content.append(f"- **{server.get('description', '默认服务器')}**: `{server.get('url', 'http://localhost:8000')}`")
        doc_content.append("")
    
    # 标签分组
    paths = openapi_schema.get('paths', {})
    tags_dict = {}
    
    for path, methods in paths.items():
        for method, details in methods.items():
            if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                tags = details.get('tags', ['未分类'])
                for tag in tags:
                    if tag not in tags_dict:
                        tags_dict[tag] = []
                    tags_dict[tag].append({
                        'path': path,
                        'method': method.upper(),
                        'summary': details.get('summary', ''),
                        'description': details.get('description', ''),
                        'parameters': details.get('parameters', []),
                        'requestBody': details.get('requestBody', {}),
                        'responses': details.get('responses', {})
                    })
    
    # 按标签生成文档
    for tag, endpoints in tags_dict.items():
        doc_content.append(f"## {tag}")
        doc_content.append("")
        
        for endpoint in endpoints:
            doc_content.append(f"### {endpoint['method']} {endpoint['path']}")
            doc_content.append("")
            
            if endpoint['summary']:
                doc_content.append(f"**摘要**: {endpoint['summary']}")
                doc_content.append("")
            
            if endpoint['description']:
                doc_content.append(f"**描述**: {endpoint['description']}")
                doc_content.append("")
            
            # 参数
            if endpoint['parameters']:
                doc_content.append("**参数**:")
                doc_content.append("")
                for param in endpoint['parameters']:
                    param_name = param.get('name', '')
                    param_type = param.get('schema', {}).get('type', 'string')
                    param_desc = param.get('description', '')
                    required = "必需" if param.get('required', False) else "可选"
                    doc_content.append(f"- `{param_name}` ({param_type}, {required}): {param_desc}")
                doc_content.append("")
            
            # 请求体
            if endpoint['requestBody']:
                doc_content.append("**请求体**:")
                doc_content.append("")
                content = endpoint['requestBody'].get('content', {})
                for content_type, schema_info in content.items():
                    doc_content.append(f"- **Content-Type**: `{content_type}`")
                    if 'schema' in schema_info:
                        doc_content.append("- **Schema**: 见OpenAPI规范")
                doc_content.append("")
            
            # 响应
            if endpoint['responses']:
                doc_content.append("**响应**:")
                doc_content.append("")
                for status_code, response_info in endpoint['responses'].items():
                    description = response_info.get('description', '')
                    doc_content.append(f"- **{status_code}**: {description}")
                doc_content.append("")
            
            doc_content.append("---")
            doc_content.append("")
    
    # 保存API参考文档
    docs_dir = Path("/workspace/docs")
    with open(docs_dir / "api_reference_generated.md", "w", encoding="utf-8") as f:
        f.write("\n".join(doc_content))
    
    print("✅ API参考文档已生成到 docs/api_reference_generated.md")

def generate_basic_docs():
    """生成基础文档结构"""
    print("📝 生成基础文档结构...")
    
    docs_dir = Path("/workspace/docs")
    docs_dir.mkdir(exist_ok=True)
    
    # 生成基础API概览
    api_overview = """# API 概览

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

"""
    
    with open(docs_dir / "api_overview.md", "w", encoding="utf-8") as f:
        f.write(api_overview)
    
    print("✅ API概览文档已生成到 docs/api_overview.md")

def generate_testing_docs():
    """生成测试文档"""
    print("🧪 生成测试文档...")
    
    testing_doc = """# 测试文档

## 测试概述

本项目使用pytest作为测试框架，包含单元测试、集成测试和API测试。

## 测试结构

```
tests/
├── __init__.py
├── conftest.py              # pytest配置和fixture
├── unit/                    # 单元测试
│   ├── test_config.py       # 配置测试
│   ├── test_utils.py        # 工具函数测试
│   ├── test_core_models.py  # 核心模型测试
│   ├── test_core_queue.py   # 队列系统测试
│   └── test_services.py     # 服务模块测试
├── api/                     # API测试
│   ├── test_main.py         # 主应用测试
│   ├── test_asr.py          # ASR API测试
│   └── test_speaker.py      # 声纹API测试
└── integration/             # 集成测试
    └── test_pipeline_integration.py
```

## 运行测试

### 安装依赖

```bash
pip install pytest pytest-asyncio pytest-cov httpx
```

### 运行所有测试

```bash
pytest tests/ -v
```

### 运行特定类型的测试

```bash
# 单元测试
pytest tests/unit/ -v -m unit

# API测试
pytest tests/api/ -v -m api

# 集成测试
pytest tests/integration/ -v -m integration
```

### 生成覆盖率报告

```bash
pytest tests/ --cov=app --cov-report=html --cov-report=term-missing
```

## 测试配置

### pytest.ini

项目包含pytest配置文件，设置了：
- 测试路径
- 覆盖率目标（80%）
- 异步测试支持
- 测试标记

### Fixtures

`conftest.py`提供了多个测试fixture：
- `client`: FastAPI测试客户端
- `async_client`: 异步测试客户端
- `mock_audio_file`: 模拟音频文件
- `mock_model`: 模拟ML模型
- `mock_database`: 模拟数据库
- 各种管理器的模拟对象

## 测试策略

### 单元测试
- 测试单个函数或类的功能
- 使用模拟对象隔离依赖
- 覆盖正常和异常情况

### API测试
- 测试HTTP端点
- 验证请求/响应格式
- 测试错误处理

### 集成测试
- 测试组件间协作
- 端到端流程验证
- 性能和并发测试

## 测试数据

测试使用模拟数据和临时文件，不依赖外部服务：
- 音频文件通过代码生成
- 数据库使用SQLite内存数据库
- 外部API调用被模拟

## 持续集成

测试可以在CI/CD流水线中运行：
- 自动运行所有测试
- 生成覆盖率报告
- 失败时阻止部署

## 测试最佳实践

1. **命名规范**: 测试函数以`test_`开头
2. **独立性**: 每个测试独立运行
3. **可读性**: 使用描述性的测试名称
4. **覆盖率**: 目标80%以上代码覆盖率
5. **模拟**: 合理使用mock隔离依赖
6. **断言**: 使用明确的断言消息

"""
    
    docs_dir = Path("/workspace/docs")
    with open(docs_dir / "testing_guide.md", "w", encoding="utf-8") as f:
        f.write(testing_doc)
    
    print("✅ 测试文档已生成到 docs/testing_guide.md")

def update_readme():
    """更新README文件"""
    print("📄 更新README文件...")
    
    readme_content = """# 语音识别服务

基于Sherpa-ONNX的智能语音识别与声纹识别服务。

## 功能特性

- 🎤 **语音识别**: 支持多种音频格式的语音转文字
- 👤 **声纹识别**: 说话人识别和注册功能  
- 🔄 **实时处理**: WebSocket支持实时音频流处理
- 📊 **监控指标**: Prometheus集成，完善的监控体系
- 🚀 **高性能**: 异步处理，支持并发请求
- 🐳 **容器化**: Docker支持，易于部署

## 快速开始

### 使用Docker运行

```bash
# GPU版本
docker-compose -f docker-compose.gpu.yml up

# CPU版本  
docker-compose -f docker-compose.cpu.yml up
```

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 生成覆盖率报告
pytest tests/ --cov=app --cov-report=html
```

测试覆盖率目标：80%以上

## 部署

详见 [部署文档](docs/deployment.md)

## 监控

- 健康检查: `GET /health`
- Prometheus指标: `GET /metrics`
- 详见 [监控指南](docs/logging_monitoring_guide.md)

## 项目结构

```
app/
├── api/          # API路由
├── core/         # 核心业务逻辑
├── services/     # 服务层
├── utils/        # 工具函数
└── main.py       # 应用入口

tests/
├── unit/         # 单元测试
├── api/          # API测试
└── integration/  # 集成测试

docs/             # 文档
```

## 开发指南

1. 代码风格：使用black和isort格式化
2. 类型检查：使用mypy
3. 测试：编写测试确保80%+覆盖率
4. 文档：更新相关文档

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
"""

    with open("/workspace/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ README文件已更新")

def generate_changelog():
    """生成变更日志"""
    print("📋 生成变更日志...")
    
    changelog = """# 变更日志

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
"""

    docs_dir = Path("/workspace/docs")
    with open(docs_dir / "CHANGELOG.md", "w", encoding="utf-8") as f:
        f.write(changelog)
    
    print("✅ 变更日志已生成到 docs/CHANGELOG.md")

if __name__ == "__main__":
    print("=" * 60)
    print("📚 语音识别服务 - 文档生成")
    print("=" * 60)
    
    # 生成文档
    api_success = generate_api_documentation()
    generate_testing_docs()
    update_readme()
    generate_changelog()
    
    print("\n" + "=" * 60)
    print("📋 文档生成完成状态:")
    if api_success:
        print("✅ OpenAPI文档生成成功")
        print("✅ API参考文档生成成功")
    else:
        print("⚠️  OpenAPI文档生成需要完整环境")
        print("✅ 基础API文档生成成功")
    print("✅ 测试文档生成成功")
    print("✅ README更新成功")
    print("✅ 变更日志生成成功")
    print("=" * 60)