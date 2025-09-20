"""
配置模块单元测试
"""
import pytest
from unittest.mock import patch
from app.config import Settings


class TestSettings:
    """配置设置测试"""
    
    def test_default_settings(self):
        """测试默认设置"""
        settings = Settings()
        
        assert settings.APP_NAME == "语音识别服务"
        assert settings.VERSION == "0.1.0"
        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == "INFO"
        assert settings.MAX_FILE_SIZE == 100 * 1024 * 1024  # 100MB
        assert settings.ENABLE_METRICS is True
    
    @patch.dict('os.environ', {
        'DEBUG': 'true',
        'LOG_LEVEL': 'DEBUG',
        'MAX_FILE_SIZE': '50000000',
        'ENABLE_METRICS': 'false'
    })
    def test_environment_override(self):
        """测试环境变量覆盖"""
        settings = Settings()
        
        assert settings.DEBUG is True
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.MAX_FILE_SIZE == 50000000
        assert settings.ENABLE_METRICS is False
    
    def test_database_url_validation(self):
        """测试数据库URL验证"""
        with patch.dict('os.environ', {'DATABASE_URL': 'invalid-url'}):
            settings = Settings()
            # 应该有默认值或抛出验证错误
            assert isinstance(settings.DATABASE_URL, str)
    
    def test_minio_settings(self):
        """测试MinIO设置"""
        with patch.dict('os.environ', {
            'MINIO_ENDPOINT': 'localhost:9000',
            'MINIO_ACCESS_KEY': 'testkey',
            'MINIO_SECRET_KEY': 'testsecret'
        }):
            settings = Settings()
            assert settings.MINIO_ENDPOINT == "localhost:9000"
            assert settings.MINIO_ACCESS_KEY == "testkey"
            assert settings.MINIO_SECRET_KEY == "testsecret"