"""
API模块初始化
导出所有API路由
"""

from .asr import router as asr_router
from .speaker import router as speaker_router

__all__ = ["asr_router", "speaker_router"]
