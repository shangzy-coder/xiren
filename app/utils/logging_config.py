"""
日志配置模块

使用structlog提供结构化日志记录，支持JSON格式输出和丰富的上下文信息。
"""

import sys
import structlog
from structlog.stdlib import LoggerFactory, add_logger_name, add_log_level
from structlog.dev import ConsoleRenderer
from structlog.processors import JSONRenderer, TimeStamper, StackInfoRenderer, format_exc_info
from typing import Any, Dict
import logging

from app.config import settings


def configure_logging():
    """配置structlog日志系统"""
    
    # 配置标准库logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL),
    )
    
    # 根据环境选择渲染器
    if settings.ENVIRONMENT == "development":
        # 开发环境使用彩色控制台输出
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            TimeStamper(fmt="ISO"),
            StackInfoRenderer(),
            format_exc_info,
            ConsoleRenderer(colors=True)
        ]
    else:
        # 生产环境使用JSON格式
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            TimeStamper(fmt="ISO"),
            StackInfoRenderer(),
            format_exc_info,
            JSONRenderer()
        ]
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """获取结构化logger实例
    
    Args:
        name: logger名称，默认使用调用模块名
        
    Returns:
        配置好的structlog logger实例
    """
    return structlog.get_logger(name)


class LoggingMixin:
    """日志混入类，为其他类提供结构化日志功能"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, **context):
        """记录信息日志"""
        self.logger.info(message, **context)
    
    def log_error(self, message: str, **context):
        """记录错误日志"""
        self.logger.error(message, **context)
    
    def log_warning(self, message: str, **context):
        """记录警告日志"""
        self.logger.warning(message, **context)
    
    def log_debug(self, message: str, **context):
        """记录调试日志"""
        self.logger.debug(message, **context)


def log_request_response(func):
    """装饰器：记录API请求和响应"""
    import functools
    import time
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = get_logger("api_request")
        start_time = time.time()
        
        # 记录请求开始
        logger.info(
            "API request started",
            function=func.__name__,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys())
        )
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            # 记录成功响应
            logger.info(
                "API request completed",
                function=func.__name__,
                duration=duration,
                status="success"
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # 记录错误响应
            logger.error(
                "API request failed",
                function=func.__name__,
                duration=duration,
                error=str(e),
                error_type=type(e).__name__,
                status="error"
            )
            
            raise
    
    return wrapper


def log_model_operation(operation_type: str):
    """装饰器：记录模型操作"""
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger("model_operation")
            start_time = time.time()
            
            # 记录操作开始
            logger.info(
                "Model operation started",
                operation=operation_type,
                function=func.__name__
            )
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 记录成功操作
                logger.info(
                    "Model operation completed",
                    operation=operation_type,
                    function=func.__name__,
                    duration=duration,
                    status="success"
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # 记录错误操作
                logger.error(
                    "Model operation failed",
                    operation=operation_type,
                    function=func.__name__,
                    duration=duration,
                    error=str(e),
                    error_type=type(e).__name__,
                    status="error"
                )
                
                raise
        
        return wrapper
    return decorator