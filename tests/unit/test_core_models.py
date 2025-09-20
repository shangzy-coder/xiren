"""
核心模型模块单元测试
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np

from app.core.model import (
    ModelManager,
    get_model_manager,
    get_model_info,
    ModelLoadError
)


class TestModelManager:
    """模型管理器测试"""
    
    @pytest.fixture
    def model_manager(self):
        """创建模型管理器实例"""
        return ModelManager()
    
    def test_model_manager_initialization(self, model_manager):
        """测试模型管理器初始化"""
        assert model_manager.asr_model is None
        assert model_manager.speaker_model is None
        assert model_manager.vad_model is None
        assert model_manager._model_cache == {}
    
    @patch('app.core.model.sherpa_onnx')
    def test_load_asr_model_success(self, mock_sherpa, model_manager):
        """测试ASR模型加载成功"""
        # 模拟sherpa-onnx模型
        mock_model = MagicMock()
        mock_sherpa.OfflineRecognizer.return_value = mock_model
        
        result = model_manager.load_asr_model(
            model_path="/fake/path",
            use_gpu=False
        )
        
        assert result is True
        assert model_manager.asr_model is not None
        mock_sherpa.OfflineRecognizer.assert_called_once()
    
    def test_load_asr_model_failure(self, model_manager):
        """测试ASR模型加载失败"""
        with pytest.raises(ModelLoadError):
            model_manager.load_asr_model(
                model_path="/nonexistent/path",
                use_gpu=False
            )
    
    @patch('app.core.model.torch')
    @patch('app.core.model.torchaudio')
    def test_load_speaker_model_success(self, mock_torchaudio, mock_torch, model_manager):
        """测试声纹模型加载成功"""
        # 模拟PyTorch模型
        mock_model = MagicMock()
        mock_torch.jit.load.return_value = mock_model
        
        result = model_manager.load_speaker_model(
            model_path="/fake/speaker/model",
            use_gpu=False
        )
        
        assert result is True
        assert model_manager.speaker_model is not None
    
    @patch('app.core.model.webrtcvad')
    def test_load_vad_model_success(self, mock_webrtcvad, model_manager):
        """测试VAD模型加载成功"""
        mock_vad = MagicMock()
        mock_webrtcvad.Vad.return_value = mock_vad
        
        result = model_manager.load_vad_model(aggressiveness=3)
        
        assert result is True
        assert model_manager.vad_model is not None
        mock_webrtcvad.Vad.assert_called_once_with(3)
    
    def test_transcribe_without_model(self, model_manager):
        """测试没有加载模型时的转录"""
        audio_data = np.random.randn(16000).astype(np.float32)
        
        with pytest.raises(ModelLoadError, match="ASR模型未加载"):
            model_manager.transcribe(audio_data)
    
    @patch('app.core.model.sherpa_onnx')
    def test_transcribe_with_model(self, mock_sherpa, model_manager):
        """测试有模型时的转录"""
        # 设置模型
        mock_model = MagicMock()
        mock_stream = MagicMock()
        mock_model.create_stream.return_value = mock_stream
        mock_model.decode_stream.return_value = None
        mock_model.get_result.return_value = MagicMock(text="测试转录结果")
        
        model_manager.asr_model = mock_model
        
        audio_data = np.random.randn(16000).astype(np.float32)
        result = model_manager.transcribe(audio_data)
        
        assert result == "测试转录结果"
        mock_model.create_stream.assert_called_once()
    
    def test_extract_speaker_embedding_without_model(self, model_manager):
        """测试没有加载模型时的声纹提取"""
        audio_data = np.random.randn(16000).astype(np.float32)
        
        with pytest.raises(ModelLoadError, match="声纹模型未加载"):
            model_manager.extract_speaker_embedding(audio_data)
    
    def test_extract_speaker_embedding_with_model(self, model_manager):
        """测试有模型时的声纹提取"""
        # 模拟声纹模型
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(detach=MagicMock(
            return_value=MagicMock(numpy=MagicMock(
                return_value=np.random.randn(512)
            ))
        ))
        model_manager.speaker_model = mock_model
        
        audio_data = np.random.randn(16000).astype(np.float32)
        result = model_manager.extract_speaker_embedding(audio_data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 512
    
    def test_detect_voice_activity_without_model(self, model_manager):
        """测试没有加载模型时的VAD"""
        audio_data = np.random.randn(16000).astype(np.float32)
        
        with pytest.raises(ModelLoadError, match="VAD模型未加载"):
            model_manager.detect_voice_activity(audio_data)
    
    def test_detect_voice_activity_with_model(self, model_manager):
        """测试有模型时的VAD"""
        # 模拟VAD模型
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = True
        model_manager.vad_model = mock_vad
        
        audio_data = np.random.randn(16000).astype(np.int16)
        result = model_manager.detect_voice_activity(audio_data, sample_rate=16000)
        
        assert isinstance(result, list)
        # VAD应该被调用
        assert mock_vad.is_speech.called
    
    def test_model_caching(self, model_manager):
        """测试模型缓存"""
        model_path = "/fake/model/path"
        
        # 第一次加载应该创建新模型
        with patch('app.core.model.sherpa_onnx') as mock_sherpa:
            mock_model = MagicMock()
            mock_sherpa.OfflineRecognizer.return_value = mock_model
            
            model_manager.load_asr_model(model_path)
            assert model_path in model_manager._model_cache
            
            # 第二次加载应该使用缓存
            model_manager.load_asr_model(model_path)
            # sherpa_onnx应该只被调用一次
            assert mock_sherpa.OfflineRecognizer.call_count == 1


class TestModelManagerSingleton:
    """模型管理器单例测试"""
    
    @pytest.mark.asyncio
    async def test_get_model_manager_singleton(self):
        """测试模型管理器单例模式"""
        manager1 = await get_model_manager()
        manager2 = await get_model_manager()
        
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_get_model_info(self):
        """测试获取模型信息"""
        with patch('app.core.model.get_model_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.asr_model = MagicMock()
            mock_manager.speaker_model = MagicMock()
            mock_manager.vad_model = None
            mock_get_manager.return_value = mock_manager
            
            info = await get_model_info()
            
            assert isinstance(info, dict)
            assert "asr_loaded" in info
            assert "speaker_loaded" in info
            assert "vad_loaded" in info
            assert info["asr_loaded"] is True
            assert info["speaker_loaded"] is True
            assert info["vad_loaded"] is False