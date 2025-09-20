# 语音识别服务 Makefile

.PHONY: help install build deploy deploy-cpu deploy-gpu start stop restart logs clean test

# 默认目标
help:
	@echo "语音识别服务管理命令:"
	@echo ""
	@echo "  make install     - 安装依赖(根据硬件自动选择)"
	@echo "  make build       - 构建Docker镜像"
	@echo "  make deploy      - 自动检测硬件并部署"
	@echo "  make deploy-cpu  - 强制使用CPU版本部署"
	@echo "  make deploy-gpu  - 强制使用GPU版本部署"
	@echo "  make start       - 启动服务"
	@echo "  make stop        - 停止服务"
	@echo "  make restart     - 重启服务"
	@echo "  make logs        - 查看日志"
	@echo "  make clean       - 清理Docker资源"
	@echo "  make test        - 运行测试"
	@echo ""

# 安装依赖
install:
	@echo "检测硬件环境并安装依赖..."
	@REQUIREMENTS_FILE=$$(python scripts/detect_hardware.py) && \
	echo "使用依赖文件: $$REQUIREMENTS_FILE" && \
	pip install -r $$REQUIREMENTS_FILE

# 构建镜像
build:
	@./scripts/deploy.sh --help

# 自动部署
deploy:
	@./scripts/deploy.sh

# CPU版本部署
deploy-cpu:
	@./scripts/deploy.sh --cpu

# GPU版本部署
deploy-gpu:
	@./scripts/deploy.sh --gpu

# 启动服务
start: deploy

# 停止服务
stop:
	@./scripts/deploy.sh --stop

# 重启服务
restart:
	@./scripts/deploy.sh --restart

# 查看日志
logs:
	@./scripts/deploy.sh --logs

# 清理Docker资源
clean:
	@echo "清理Docker资源..."
	@docker-compose -f docker-compose.cpu.yml down -v --remove-orphans 2>/dev/null || true
	@docker-compose -f docker-compose.gpu.yml down -v --remove-orphans 2>/dev/null || true
	@docker system prune -f
	@echo "清理完成"

# 运行测试
test:
	@echo "运行测试..."
	@python -m pytest tests/ -v

# 开发环境设置
dev-setup:
	@echo "设置开发环境..."
	@python scripts/detect_hardware.py
	@make install
	@echo "开发环境设置完成"

# 生产环境健康检查
health-check:
	@echo "健康检查..."
	@curl -f http://localhost:8000/health || (echo "服务不可用" && exit 1)
	@echo "服务运行正常"
