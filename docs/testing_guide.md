# 测试文档

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

