"""
ASR API测试
"""
import pytest
import io
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from app.main import app


class TestASRAPI:
    """ASR API测试"""
    
    def test_initialize_asr_models_success(self, client, mock_model):
        """测试ASR模型初始化成功"""
        with patch('app.api.asr.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.load_asr_model.return_value = True
            mock_manager.load_vad_model.return_value = True
            mock_manager.load_speaker_model.return_value = True
            mock_get_manager.return_value = mock_manager
            
            response = client.post("/api/v1/asr/initialize", data={
                "model_type": "sense_voice",
                "use_gpu": False,
                "enable_vad": True,
                "enable_speaker_id": True
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "message" in data
            assert "model_info" in data
    
    def test_initialize_asr_models_failure(self, client):
        """测试ASR模型初始化失败"""
        with patch('app.api.asr.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.load_asr_model.side_effect = Exception("模型加载失败")
            mock_get_manager.return_value = mock_manager
            
            response = client.post("/api/v1/asr/initialize", data={
                "model_type": "sense_voice",
                "use_gpu": False
            })
            
            assert response.status_code == 500
            data = response.json()
            assert "模型加载失败" in data["detail"]
    
    def test_transcribe_audio_success(self, client, mock_audio_file):
        """测试音频转录成功"""
        with patch('app.api.asr.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.transcribe.return_value = "测试转录结果"
            mock_manager.asr_model = MagicMock()  # 模型已加载
            mock_get_manager.return_value = mock_manager
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/asr/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"language": "zh", "enable_vad": True}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["text"] == "测试转录结果"
            assert data["success"] is True
            assert "processing_time" in data
            assert "session_id" in data
    
    def test_transcribe_audio_no_model(self, client, mock_audio_file):
        """测试没有模型时的转录"""
        with patch('app.api.asr.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.asr_model = None  # 模型未加载
            mock_get_manager.return_value = mock_manager
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/asr/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")},
                )
            
            assert response.status_code == 400
            data = response.json()
            assert "模型未初始化" in data["detail"]
    
    def test_transcribe_audio_invalid_file(self, client):
        """测试无效音频文件"""
        # 创建一个非音频文件
        fake_file = io.BytesIO(b"not an audio file")
        
        response = client.post(
            "/api/v1/asr/transcribe",
            files={"file": ("test.txt", fake_file, "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "不支持的文件格式" in data["detail"] or "音频文件无效" in data["detail"]
    
    def test_transcribe_audio_file_too_large(self, client):
        """测试文件过大"""
        with patch('app.config.settings.MAX_FILE_SIZE', 1):  # 1字节限制
            # 创建一个大文件
            large_file = io.BytesIO(b"x" * 1000)
            
            response = client.post(
                "/api/v1/asr/transcribe",
                files={"file": ("test.wav", large_file, "audio/wav")}
            )
            
            assert response.status_code == 413
            data = response.json()
            assert "文件过大" in data["detail"]
    
    def test_transcribe_async_success(self, client, mock_audio_file, mock_queue_manager):
        """测试异步转录成功"""
        with patch('app.api.asr.get_queue_manager') as mock_get_queue:
            mock_get_queue.return_value = mock_queue_manager
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/asr/transcribe-async",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"priority": "high"}
                )
            
            assert response.status_code == 202
            data = response.json()
            assert data["task_id"] == "test-task-id"
            assert data["status"] == "queued"
            assert "estimated_completion_time" in data
    
    def test_get_transcription_result_success(self, client, mock_queue_manager):
        """测试获取转录结果成功"""
        with patch('app.api.asr.get_queue_manager') as mock_get_queue:
            mock_get_queue.return_value = mock_queue_manager
            
            response = client.get("/api/v1/asr/result/test-task-id")
            
            assert response.status_code == 200
            data = response.json()
            assert data["task_id"] == "test-task-id"
            assert data["status"] == "completed"
            assert data["result"]["text"] == "测试结果"
    
    def test_get_transcription_result_not_found(self, client):
        """测试获取不存在的转录结果"""
        with patch('app.api.asr.get_queue_manager') as mock_get_queue:
            mock_manager = AsyncMock()
            mock_manager.get_task_status.return_value = None
            mock_get_queue.return_value = mock_manager
            
            response = client.get("/api/v1/asr/result/nonexistent-task")
            
            assert response.status_code == 404
            data = response.json()
            assert "任务不存在" in data["detail"]
    
    def test_get_model_info(self, client):
        """测试获取模型信息"""
        with patch('app.api.asr.get_model_info') as mock_get_info:
            mock_get_info.return_value = {
                "asr_loaded": True,
                "speaker_loaded": False,
                "vad_loaded": True,
                "model_type": "sense_voice"
            }
            
            response = client.get("/api/v1/asr/info")
            
            assert response.status_code == 200
            data = response.json()
            assert data["asr_loaded"] is True
            assert data["speaker_loaded"] is False
            assert data["vad_loaded"] is True
    
    def test_batch_transcribe(self, client, mock_audio_file):
        """测试批量转录"""
        with patch('app.api.asr.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.transcribe.return_value = "测试转录结果"
            mock_manager.asr_model = MagicMock()
            mock_get_manager.return_value = mock_manager
            
            # 准备多个文件
            files = []
            for i in range(3):
                with open(mock_audio_file, 'rb') as f:
                    files.append(("files", (f"test{i}.wav", f.read(), "audio/wav")))
            
            response = client.post(
                "/api/v1/asr/batch-transcribe",
                files=files,
                data={"language": "zh"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 3
            assert all(result["success"] for result in data["results"])
    
    def test_supported_formats_endpoint(self, client):
        """测试支持的格式端点"""
        response = client.get("/api/v1/asr/formats")
        
        assert response.status_code == 200
        data = response.json()
        assert "audio_formats" in data
        assert "languages" in data
        assert isinstance(data["audio_formats"], list)
        assert isinstance(data["languages"], list)
        
        # 检查常见格式
        assert "wav" in data["audio_formats"]
        assert "mp3" in data["audio_formats"]