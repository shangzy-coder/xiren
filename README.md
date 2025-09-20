# 语音识别服务

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
