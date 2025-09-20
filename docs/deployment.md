# 部署指南

## 自适应CPU/GPU部署

本项目支持根据硬件环境自动选择CPU或GPU版本进行部署。

### 快速开始

#### 1. 自动检测部署（推荐）
```bash
# 自动检测硬件环境并部署
make deploy
# 或
./scripts/deploy.sh
```

#### 2. 指定版本部署
```bash
# 强制使用CPU版本
make deploy-cpu
# 或
./scripts/deploy.sh --cpu

# 强制使用GPU版本
make deploy-gpu  
# 或
./scripts/deploy.sh --gpu
```

### 环境检测逻辑

系统会按以下顺序检测硬件环境：

1. **检查NVIDIA GPU**: 运行 `nvidia-smi` 命令
2. **检查CUDA环境**: 运行 `nvcc --version` 命令
3. **选择合适版本**:
   - GPU + CUDA ✅ → 使用GPU版本 (`requirements-gpu.txt`)
   - 其他情况 → 使用CPU版本 (`requirements-cpu.txt`)

### 依赖文件说明

- `requirements-base.txt`: 基础依赖，CPU/GPU通用
- `requirements-cpu.txt`: CPU版本专用依赖
- `requirements-gpu.txt`: GPU版本专用依赖（包含CUDA支持）

### Docker配置文件

- `Dockerfile.cpu`: CPU版本Docker配置
- `Dockerfile.gpu`: GPU版本Docker配置（基于CUDA 12.2.2）
- `docker-compose.cpu.yml`: CPU版本服务编排
- `docker-compose.gpu.yml`: GPU版本服务编排（包含GPU资源分配）

### 管理命令

```bash
# 查看帮助
make help

# 安装依赖（本地开发）
make install

# 启动服务
make start

# 停止服务
make stop

# 重启服务
make restart

# 查看日志
make logs

# 健康检查
make health-check

# 清理资源
make clean

# 运行测试
make test
```

### 环境变量配置

| 变量名 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `DEVICE_TYPE` | 设备类型 | `auto` | `cpu`, `gpu`, `auto` |
| `CUDA_VISIBLE_DEVICES` | GPU设备ID | `0` | `0,1` |
| `NVIDIA_VISIBLE_DEVICES` | NVIDIA可见设备 | `all` | `all`, `none` |

### 性能对比

| 功能 | CPU版本 | GPU版本 |
|------|---------|---------|
| 语音识别 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 声纹识别 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 内存占用 | 低 | 中等 |
| 启动时间 | 快 | 中等 |
| 并发处理 | 中等 | 高 |

### 故障排除

#### GPU版本无法启动
1. 检查NVIDIA驱动是否正确安装
2. 确认Docker支持GPU (nvidia-docker2)
3. 验证CUDA版本兼容性

#### 依赖安装失败
1. 检查网络连接
2. 尝试使用国内镜像源
3. 手动指定依赖版本

#### 服务无法访问
1. 检查端口是否被占用 (8000)
2. 确认防火墙设置
3. 查看容器日志

### 开发环境设置

```bash
# 设置开发环境
make dev-setup

# 手动检测硬件
python scripts/detect_hardware.py

# 本地运行（需要PostgreSQL和MinIO）
python -m app.main
```

### 监控和维护

```bash
# 实时查看日志
make logs

# 检查服务状态
docker-compose -f docker-compose.gpu.yml ps

# 查看资源使用
docker stats

# 备份数据
docker-compose -f docker-compose.gpu.yml exec postgres pg_dump -U speech_user speech_recognition > backup.sql
```
