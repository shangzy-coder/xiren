#!/usr/bin/env python3
"""
启动简单语音服务
"""
import sys
import os
from pathlib import Path

# 添加simple_speech_service到Python路径
sys.path.insert(0, str(Path(__file__).parent / "simple_speech_service"))

# 启动服务
from simple_speech_service.main import app
import uvicorn

if __name__ == "__main__":
    # 从环境变量获取配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print("启动简单语音识别服务...")
    print(f"服务地址: http://{host}:{port}")
    print(f"API文档: http://{host}:{port}/docs")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )