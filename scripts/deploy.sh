#!/bin/bash
# 自动部署脚本 - 根据硬件环境选择CPU或GPU版本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== 语音识别服务自动部署脚本 ===${NC}"

# 检测硬件环境
detect_hardware() {
    echo -e "${YELLOW}检测硬件环境...${NC}"
    
    # 检查是否有NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            echo -e "${GREEN}✓ 检测到NVIDIA GPU${NC}"
            
            # 检查CUDA
            if command -v nvcc >/dev/null 2>&1; then
                echo -e "${GREEN}✓ 检测到CUDA环境${NC}"
                echo "gpu"
                return 0
            else
                echo -e "${YELLOW}⚠ 未检测到CUDA，使用CPU版本${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ NVIDIA GPU不可用，使用CPU版本${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ 未检测到NVIDIA GPU，使用CPU版本${NC}"
    fi
    
    echo "cpu"
}

# 部署函数
deploy() {
    local hardware_type=$1
    local compose_file="docker-compose.${hardware_type}.yml"
    
    echo -e "${GREEN}使用 ${hardware_type^^} 版本部署${NC}"
    echo -e "${YELLOW}使用配置文件: ${compose_file}${NC}"
    
    # 检查配置文件是否存在
    if [ ! -f "$compose_file" ]; then
        echo -e "${RED}错误: 配置文件 ${compose_file} 不存在${NC}"
        exit 1
    fi
    
    # 停止现有服务
    echo -e "${YELLOW}停止现有服务...${NC}"
    docker-compose -f "$compose_file" down 2>/dev/null || true
    
    # 构建和启动服务
    echo -e "${YELLOW}构建镜像...${NC}"
    docker-compose -f "$compose_file" build --no-cache
    
    echo -e "${YELLOW}启动服务...${NC}"
    docker-compose -f "$compose_file" up -d
    
    # 等待服务启动
    echo -e "${YELLOW}等待服务启动...${NC}"
    sleep 10
    
    # 检查服务状态
    echo -e "${YELLOW}检查服务状态...${NC}"
    docker-compose -f "$compose_file" ps
    
    # 检查健康状态
    echo -e "${YELLOW}检查服务健康状态...${NC}"
    for i in {1..30}; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            echo -e "${GREEN}✓ 服务启动成功！${NC}"
            echo -e "${GREEN}API地址: http://localhost:8000${NC}"
            echo -e "${GREEN}API文档: http://localhost:8000/docs${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    
    echo -e "${RED}✗ 服务启动失败或健康检查超时${NC}"
    echo -e "${YELLOW}查看日志:${NC}"
    docker-compose -f "$compose_file" logs speech-service
    exit 1
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -c, --cpu     强制使用CPU版本"
    echo "  -g, --gpu     强制使用GPU版本"
    echo "  -h, --help    显示此帮助信息"
    echo "  --logs        查看服务日志"
    echo "  --stop        停止所有服务"
    echo "  --restart     重启服务"
    echo ""
    echo "如果不指定选项，将自动检测硬件环境。"
}

# 查看日志
show_logs() {
    echo -e "${YELLOW}选择要查看的日志:${NC}"
    echo "1) CPU版本日志"
    echo "2) GPU版本日志"
    read -p "请选择 (1/2): " choice
    
    case $choice in
        1)
            docker-compose -f docker-compose.cpu.yml logs -f
            ;;
        2)
            docker-compose -f docker-compose.gpu.yml logs -f
            ;;
        *)
            echo -e "${RED}无效选择${NC}"
            exit 1
            ;;
    esac
}

# 停止服务
stop_services() {
    echo -e "${YELLOW}停止所有服务...${NC}"
    docker-compose -f docker-compose.cpu.yml down 2>/dev/null || true
    docker-compose -f docker-compose.gpu.yml down 2>/dev/null || true
    echo -e "${GREEN}服务已停止${NC}"
}

# 重启服务
restart_services() {
    echo -e "${YELLOW}重启服务...${NC}"
    hardware_type=$(detect_hardware)
    stop_services
    deploy "$hardware_type"
}

# 主逻辑
case "${1:-}" in
    -c|--cpu)
        deploy "cpu"
        ;;
    -g|--gpu)
        deploy "gpu"
        ;;
    -h|--help)
        show_help
        ;;
    --logs)
        show_logs
        ;;
    --stop)
        stop_services
        ;;
    --restart)
        restart_services
        ;;
    "")
        # 自动检测
        hardware_type=$(detect_hardware)
        deploy "$hardware_type"
        ;;
    *)
        echo -e "${RED}未知选项: $1${NC}"
        show_help
        exit 1
        ;;
esac
