"""
综合处理API测试
"""
import pytest
import io
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from app.main import app


class TestComprehensiveAPI:
    """综合处理API测试"""
    
    def test_comprehensive_processing_success(self, client, mock_audio_file):
        """测试综合处理成功"""
        with patch('app.api.comprehensive.get_model_manager') as mock_get_manager:
            with patch('app.api.comprehensive.get_speaker_pool') as mock_get_pool:
                with patch('app.api.comprehensive.get_async_session') as mock_get_session:
                    # 设置模拟对象
                    mock_manager = AsyncMock()
                    mock_manager.transcribe.return_value = "综合处理测试结果"
                    mock_manager.extract_speaker_embedding.return_value = [0.1] * 512
                    mock_manager.detect_voice_activity.return_value = [(0.0, 2.0), (3.0, 5.0)]
                    mock_manager.asr_model = MagicMock()
                    mock_manager.speaker_model = MagicMock()
                    mock_manager.vad_model = MagicMock()
                    mock_get_manager.return_value = mock_manager
                    
                    mock_pool = AsyncMock()
                    mock_pool.identify_speaker.return_value = {
                        "speaker_id": "speaker-123",
                        "speaker_name": "张三",
                        "similarity": 0.92,
                        "confidence": 0.95
                    }
                    mock_get_pool.return_value = mock_pool
                    
                    mock_session = AsyncMock()
                    mock_get_session.return_value.__aenter__.return_value = mock_session
                    
                    with open(mock_audio_file, 'rb') as f:
                        response = client.post(
                            "/api/v1/process",
                            files={"audio_file": ("test.wav", f, "audio/wav")},
                            data={
                                "enable_asr": True,
                                "enable_speaker_id": True,
                                "enable_vad": True,
                                "enable_diarization": False,
                                "save_to_database": True
                            }
                        )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
                    assert data["asr_result"]["text"] == "综合处理测试结果"
                    assert data["speaker_result"]["speaker_id"] == "speaker-123"
                    assert len(data["vad_result"]) == 2
                    assert "processing_time" in data
                    assert "session_id" in data
    
    def test_comprehensive_processing_asr_only(self, client, mock_audio_file):
        """测试仅ASR处理"""
        with patch('app.api.comprehensive.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.transcribe.return_value = "仅ASR测试结果"
            mock_manager.asr_model = MagicMock()
            mock_get_manager.return_value = mock_manager
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/process",
                    files={"audio_file": ("test.wav", f, "audio/wav")},
                    data={
                        "enable_asr": True,
                        "enable_speaker_id": False,
                        "enable_vad": False,
                        "enable_diarization": False
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["asr_result"]["text"] == "仅ASR测试结果"
            assert data["speaker_result"] is None
            assert data["vad_result"] is None
    
    def test_comprehensive_processing_speaker_only(self, client, mock_audio_file):
        """测试仅声纹识别处理"""
        with patch('app.api.comprehensive.get_model_manager') as mock_get_manager:
            with patch('app.api.comprehensive.get_speaker_pool') as mock_get_pool:
                mock_manager = AsyncMock()
                mock_manager.extract_speaker_embedding.return_value = [0.1] * 512
                mock_manager.speaker_model = MagicMock()
                mock_get_manager.return_value = mock_manager
                
                mock_pool = AsyncMock()
                mock_pool.identify_speaker.return_value = {
                    "speaker_id": "speaker-456",
                    "similarity": 0.88
                }
                mock_get_pool.return_value = mock_pool
                
                with open(mock_audio_file, 'rb') as f:
                    response = client.post(
                        "/api/v1/process",
                        files={"audio_file": ("test.wav", f, "audio/wav")},
                        data={
                            "enable_asr": False,
                            "enable_speaker_id": True,
                            "enable_vad": False
                        }
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["asr_result"] is None
                assert data["speaker_result"]["speaker_id"] == "speaker-456"
                assert data["vad_result"] is None
    
    def test_comprehensive_processing_no_features_enabled(self, client, mock_audio_file):
        """测试没有启用任何功能"""
        with open(mock_audio_file, 'rb') as f:
            response = client.post(
                "/api/v1/process",
                files={"audio_file": ("test.wav", f, "audio/wav")},
                data={
                    "enable_asr": False,
                    "enable_speaker_id": False,
                    "enable_vad": False,
                    "enable_diarization": False
                }
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "至少需要启用一个处理功能" in data["detail"]
    
    def test_comprehensive_processing_invalid_file(self, client):
        """测试无效文件处理"""
        fake_file = io.BytesIO(b"not an audio file")
        
        response = client.post(
            "/api/v1/process",
            files={"audio_file": ("test.txt", fake_file, "text/plain")},
            data={"enable_asr": True}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "不支持的文件格式" in data["detail"] or "音频文件无效" in data["detail"]
    
    def test_comprehensive_processing_file_too_large(self, client):
        """测试文件过大"""
        with patch('app.config.settings.MAX_FILE_SIZE', 1):  # 1字节限制
            large_file = io.BytesIO(b"x" * 1000)
            
            response = client.post(
                "/api/v1/process",
                files={"audio_file": ("test.wav", large_file, "audio/wav")},
                data={"enable_asr": True}
            )
            
            assert response.status_code == 413
            data = response.json()
            assert "文件过大" in data["detail"]
    
    def test_comprehensive_processing_model_not_loaded(self, client, mock_audio_file):
        """测试模型未加载"""
        with patch('app.api.comprehensive.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.asr_model = None  # ASR模型未加载
            mock_get_manager.return_value = mock_manager
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/process",
                    files={"audio_file": ("test.wav", f, "audio/wav")},
                    data={"enable_asr": True}
                )
            
            assert response.status_code == 400
            data = response.json()
            assert "ASR模型未初始化" in data["detail"]
    
    def test_comprehensive_processing_with_diarization(self, client, mock_audio_file):
        """测试带说话人分离的处理"""
        with patch('app.api.comprehensive.get_model_manager') as mock_get_manager:
            with patch('app.api.comprehensive.perform_speaker_diarization') as mock_diarization:
                mock_manager = AsyncMock()
                mock_manager.transcribe.return_value = "分离测试结果"
                mock_manager.asr_model = MagicMock()
                mock_get_manager.return_value = mock_manager
                
                mock_diarization.return_value = [
                    {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
                    {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"}
                ]
                
                with open(mock_audio_file, 'rb') as f:
                    response = client.post(
                        "/api/v1/process",
                        files={"audio_file": ("test.wav", f, "audio/wav")},
                        data={
                            "enable_asr": True,
                            "enable_diarization": True
                        }
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "diarization_result" in data
                assert len(data["diarization_result"]) == 2
    
    def test_comprehensive_processing_with_custom_threshold(self, client, mock_audio_file):
        """测试自定义相似度阈值"""
        with patch('app.api.comprehensive.get_model_manager') as mock_get_manager:
            with patch('app.api.comprehensive.get_speaker_pool') as mock_get_pool:
                mock_manager = AsyncMock()
                mock_manager.extract_speaker_embedding.return_value = [0.1] * 512
                mock_manager.speaker_model = MagicMock()
                mock_get_manager.return_value = mock_manager
                
                mock_pool = AsyncMock()
                mock_pool.identify_speaker.return_value = None  # 低于阈值
                mock_get_pool.return_value = mock_pool
                
                with open(mock_audio_file, 'rb') as f:
                    response = client.post(
                        "/api/v1/process",
                        files={"audio_file": ("test.wav", f, "audio/wav")},
                        data={
                            "enable_speaker_id": True,
                            "speaker_threshold": 0.95  # 高阈值
                        }
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["speaker_result"] is None or data["speaker_result"]["speaker_id"] is None
    
    def test_comprehensive_processing_language_setting(self, client, mock_audio_file):
        """测试语言设置"""
        with patch('app.api.comprehensive.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.transcribe.return_value = "English test result"
            mock_manager.asr_model = MagicMock()
            mock_get_manager.return_value = mock_manager
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/process",
                    files={"audio_file": ("test.wav", f, "audio/wav")},
                    data={
                        "enable_asr": True,
                        "language": "en"
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["asr_result"]["text"] == "English test result"
            assert data["asr_result"]["language"] == "en"
    
    def test_comprehensive_processing_database_save(self, client, mock_audio_file):
        """测试数据库保存"""
        with patch('app.api.comprehensive.get_model_manager') as mock_get_manager:
            with patch('app.api.comprehensive.get_async_session') as mock_get_session:
                mock_manager = AsyncMock()
                mock_manager.transcribe.return_value = "数据库测试结果"
                mock_manager.asr_model = MagicMock()
                mock_get_manager.return_value = mock_manager
                
                mock_session = AsyncMock()
                mock_session.add = MagicMock()
                mock_session.commit = AsyncMock()
                mock_get_session.return_value.__aenter__.return_value = mock_session
                
                with open(mock_audio_file, 'rb') as f:
                    response = client.post(
                        "/api/v1/process",
                        files={"audio_file": ("test.wav", f, "audio/wav")},
                        data={
                            "enable_asr": True,
                            "save_to_database": True
                        }
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                
                # 验证数据库操作被调用
                mock_session.add.assert_called()
                mock_session.commit.assert_called()
    
    def test_comprehensive_processing_error_handling(self, client, mock_audio_file):
        """测试错误处理"""
        with patch('app.api.comprehensive.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.transcribe.side_effect = Exception("处理失败")
            mock_manager.asr_model = MagicMock()
            mock_get_manager.return_value = mock_manager
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/process",
                    files={"audio_file": ("test.wav", f, "audio/wav")},
                    data={"enable_asr": True}
                )
            
            assert response.status_code == 500
            data = response.json()
            assert "处理失败" in data["detail"]
    
    def test_comprehensive_processing_partial_failure(self, client, mock_audio_file):
        """测试部分功能失败"""
        with patch('app.api.comprehensive.get_model_manager') as mock_get_manager:
            with patch('app.api.comprehensive.get_speaker_pool') as mock_get_pool:
                mock_manager = AsyncMock()
                mock_manager.transcribe.return_value = "ASR成功"
                mock_manager.extract_speaker_embedding.side_effect = Exception("声纹失败")
                mock_manager.asr_model = MagicMock()
                mock_manager.speaker_model = MagicMock()
                mock_get_manager.return_value = mock_manager
                
                mock_pool = AsyncMock()
                mock_get_pool.return_value = mock_pool
                
                with open(mock_audio_file, 'rb') as f:
                    response = client.post(
                        "/api/v1/process",
                        files={"audio_file": ("test.wav", f, "audio/wav")},
                        data={
                            "enable_asr": True,
                            "enable_speaker_id": True
                        }
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["asr_result"]["text"] == "ASR成功"
                assert data["speaker_result"] is None
                assert "warnings" in data
    
    def test_comprehensive_processing_statistics(self, client, mock_audio_file):
        """测试处理统计信息"""
        with patch('app.api.comprehensive.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.transcribe.return_value = "统计测试"
            mock_manager.asr_model = MagicMock()
            mock_get_manager.return_value = mock_manager
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/process",
                    files={"audio_file": ("test.wav", f, "audio/wav")},
                    data={"enable_asr": True}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "processing_time" in data
            assert "file_info" in data
            assert data["file_info"]["size"] > 0
            assert data["file_info"]["format"] == "wav"