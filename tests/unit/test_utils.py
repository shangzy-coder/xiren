"""
工具模块单元测试
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

from app.utils.audio import (
    validate_audio_file,
    convert_audio_format,
    extract_audio_features,
    AudioProcessingError
)
from app.utils.logging_config import configure_logging, get_logger
from app.utils.metrics import metrics_collector, get_metrics


class TestAudioUtils:
    """音频工具函数测试"""
    
    def test_validate_audio_file_valid(self, mock_audio_file):
        """测试有效音频文件验证"""
        result = validate_audio_file(mock_audio_file)
        assert result is True
    
    def test_validate_audio_file_invalid(self):
        """测试无效音频文件验证"""
        # 测试不存在的文件
        with pytest.raises(AudioProcessingError):
            validate_audio_file("nonexistent.wav")
        
        # 测试非音频文件
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"not an audio file")
            tmp.flush()
            
            with pytest.raises(AudioProcessingError):
                validate_audio_file(tmp.name)
            
            os.unlink(tmp.name)
    
    @patch('app.utils.audio.librosa.load')
    def test_convert_audio_format(self, mock_librosa_load):
        """测试音频格式转换"""
        # 模拟librosa.load返回值
        mock_librosa_load.return_value = (np.random.randn(16000), 16000)
        
        with tempfile.NamedTemporaryFile(suffix=".mp3") as input_file:
            input_file.write(b"fake mp3 data")
            input_file.flush()
            
            result = convert_audio_format(
                input_file.name,
                target_format="wav",
                target_sample_rate=16000
            )
            
            assert result is not None
            assert os.path.exists(result)
            
            # 清理
            if os.path.exists(result):
                os.unlink(result)
    
    @patch('app.utils.audio.librosa.load')
    @patch('app.utils.audio.librosa.feature.mfcc')
    def test_extract_audio_features(self, mock_mfcc, mock_librosa_load):
        """测试音频特征提取"""
        # 模拟返回值
        mock_librosa_load.return_value = (np.random.randn(16000), 16000)
        mock_mfcc.return_value = np.random.randn(13, 100)
        
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_file.write(b"fake audio data")
            audio_file.flush()
            
            features = extract_audio_features(audio_file.name)
            
            assert isinstance(features, dict)
            assert "mfcc" in features
            assert "duration" in features
            assert "sample_rate" in features


class TestLoggingConfig:
    """日志配置测试"""
    
    def test_configure_logging(self):
        """测试日志配置"""
        configure_logging()
        logger = get_logger("test")
        
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'debug')
    
    def test_logger_with_context(self):
        """测试带上下文的日志"""
        logger = get_logger("test")
        
        # 测试日志记录不会抛出异常
        try:
            logger.info("测试消息", extra={"key": "value"})
            logger.error("错误消息", extra={"error_code": 500})
        except Exception as e:
            pytest.fail(f"日志记录失败: {e}")


class TestMetrics:
    """指标收集测试"""
    
    def test_metrics_collector(self):
        """测试指标收集器"""
        # 测试计数器
        metrics_collector.request_count.inc()
        
        # 测试直方图
        metrics_collector.request_duration.observe(0.1)
        
        # 测试仪表
        metrics_collector.active_connections.set(5)
        
        # 获取指标
        metrics = get_metrics()
        assert isinstance(metrics, str)
        assert "request_count" in metrics
    
    def test_custom_metrics(self):
        """测试自定义指标"""
        # 测试ASR相关指标
        metrics_collector.asr_requests_total.inc()
        metrics_collector.asr_processing_duration.observe(1.5)
        
        # 测试声纹相关指标
        metrics_collector.speaker_recognition_requests.inc()
        metrics_collector.speaker_registration_total.inc()
        
        metrics = get_metrics()
        assert "asr_requests_total" in metrics
        assert "speaker_recognition_requests" in metrics