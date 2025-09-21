#!/usr/bin/env python3
"""
简化的服务启动脚本
禁用Prometheus监控，只启动基本的语音识别服务
"""

import os
import sys
import uvicorn

# 设置环境变量禁用监控
os.environ["ENABLE_METRICS"] = "false"
os.environ["ENABLE_SYSTEM_METRICS"] = "false"
os.environ["LOG_LEVEL"] = "INFO"
os.environ["ENVIRONMENT"] = "development"

def main():
    """启动服务"""
    print("🚀 启动语音识别服务 (简化模式)")
    print("📊 监控功能已禁用")
    print("🔧 开发模式")
    print("-" * 50)
    
    try:
        # 启动服务
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
