"""
声纹识别API测试
"""
import pytest
import io
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from app.main import app


class TestSpeakerAPI:
    """声纹识别API测试"""
    
    def test_register_speaker_success(self, client, mock_audio_file):
        """测试声纹注册成功"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            with patch('app.api.speaker.get_model_manager') as mock_get_manager:
                # 设置模拟对象
                mock_pool = AsyncMock()
                mock_pool.register_speaker.return_value = "speaker-123"
                mock_get_pool.return_value = mock_pool
                
                mock_manager = AsyncMock()
                mock_manager.extract_speaker_embedding.return_value = [0.1] * 512
                mock_manager.speaker_model = MagicMock()  # 模型已加载
                mock_get_manager.return_value = mock_manager
                
                with open(mock_audio_file, 'rb') as f:
                    response = client.post(
                        "/api/v1/speaker/register",
                        files={"audio_file": ("speaker.wav", f, "audio/wav")},
                        data={"speaker_name": "张三"}
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["speaker_id"] == "speaker-123"
                assert data["speaker_name"] == "张三"
                assert "processing_time" in data
    
    def test_register_speaker_no_model(self, client, mock_audio_file):
        """测试没有模型时的声纹注册"""
        with patch('app.api.speaker.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.speaker_model = None  # 模型未加载
            mock_get_manager.return_value = mock_manager
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/speaker/register",
                    files={"audio_file": ("speaker.wav", f, "audio/wav")},
                    data={"speaker_name": "张三"}
                )
            
            assert response.status_code == 400
            data = response.json()
            assert "声纹模型未初始化" in data["detail"]
    
    def test_register_speaker_invalid_file(self, client):
        """测试无效文件的声纹注册"""
        fake_file = io.BytesIO(b"not an audio file")
        
        response = client.post(
            "/api/v1/speaker/register",
            files={"audio_file": ("test.txt", fake_file, "text/plain")},
            data={"speaker_name": "张三"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "不支持的文件格式" in data["detail"] or "音频文件无效" in data["detail"]
    
    def test_register_speaker_duplicate_name(self, client, mock_audio_file):
        """测试重复名称的声纹注册"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            with patch('app.api.speaker.get_model_manager') as mock_get_manager:
                mock_pool = AsyncMock()
                mock_pool.register_speaker.side_effect = Exception("说话人已存在")
                mock_get_pool.return_value = mock_pool
                
                mock_manager = AsyncMock()
                mock_manager.extract_speaker_embedding.return_value = [0.1] * 512
                mock_manager.speaker_model = MagicMock()
                mock_get_manager.return_value = mock_manager
                
                with open(mock_audio_file, 'rb') as f:
                    response = client.post(
                        "/api/v1/speaker/register",
                        files={"audio_file": ("speaker.wav", f, "audio/wav")},
                        data={"speaker_name": "张三"}
                    )
                
                assert response.status_code == 500
                data = response.json()
                assert "说话人已存在" in data["detail"]
    
    def test_identify_speaker_success(self, client, mock_audio_file):
        """测试声纹识别成功"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            with patch('app.api.speaker.get_model_manager') as mock_get_manager:
                mock_pool = AsyncMock()
                mock_pool.identify_speaker.return_value = {
                    "speaker_id": "speaker-123",
                    "speaker_name": "张三",
                    "similarity": 0.92,
                    "confidence": 0.95
                }
                mock_get_pool.return_value = mock_pool
                
                mock_manager = AsyncMock()
                mock_manager.extract_speaker_embedding.return_value = [0.1] * 512
                mock_manager.speaker_model = MagicMock()
                mock_get_manager.return_value = mock_manager
                
                with open(mock_audio_file, 'rb') as f:
                    response = client.post(
                        "/api/v1/speaker/identify",
                        files={"audio_file": ("unknown.wav", f, "audio/wav")},
                        data={"threshold": "0.8"}
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["speaker_id"] == "speaker-123"
                assert data["speaker_name"] == "张三"
                assert data["similarity"] == 0.92
                assert data["confidence"] == 0.95
    
    def test_identify_speaker_no_match(self, client, mock_audio_file):
        """测试声纹识别无匹配"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            with patch('app.api.speaker.get_model_manager') as mock_get_manager:
                mock_pool = AsyncMock()
                mock_pool.identify_speaker.return_value = None  # 无匹配
                mock_get_pool.return_value = mock_pool
                
                mock_manager = AsyncMock()
                mock_manager.extract_speaker_embedding.return_value = [0.1] * 512
                mock_manager.speaker_model = MagicMock()
                mock_get_manager.return_value = mock_manager
                
                with open(mock_audio_file, 'rb') as f:
                    response = client.post(
                        "/api/v1/speaker/identify",
                        files={"audio_file": ("unknown.wav", f, "audio/wav")}
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["speaker_id"] is None
                assert data["message"] == "未找到匹配的说话人"
    
    def test_get_all_speakers(self, client):
        """测试获取所有说话人"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.get_all_speakers.return_value = [
                {"id": "speaker-1", "name": "张三", "created_at": "2023-01-01"},
                {"id": "speaker-2", "name": "李四", "created_at": "2023-01-02"}
            ]
            mock_get_pool.return_value = mock_pool
            
            response = client.get("/api/v1/speaker/list")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["speakers"]) == 2
            assert data["speakers"][0]["name"] == "张三"
            assert data["speakers"][1]["name"] == "李四"
    
    def test_get_speaker_info(self, client):
        """测试获取特定说话人信息"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.get_speaker_info.return_value = {
                "id": "speaker-123",
                "name": "张三",
                "created_at": "2023-01-01",
                "audio_count": 5,
                "last_updated": "2023-01-05"
            }
            mock_get_pool.return_value = mock_pool
            
            response = client.get("/api/v1/speaker/info/speaker-123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "speaker-123"
            assert data["name"] == "张三"
            assert data["audio_count"] == 5
    
    def test_get_speaker_info_not_found(self, client):
        """测试获取不存在说话人信息"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.get_speaker_info.return_value = None
            mock_get_pool.return_value = mock_pool
            
            response = client.get("/api/v1/speaker/info/nonexistent")
            
            assert response.status_code == 404
            data = response.json()
            assert "说话人不存在" in data["detail"]
    
    def test_update_speaker_info(self, client):
        """测试更新说话人信息"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.update_speaker_info.return_value = True
            mock_get_pool.return_value = mock_pool
            
            response = client.put(
                "/api/v1/speaker/info/speaker-123",
                json={"name": "张三三", "description": "更新后的描述"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "说话人信息更新成功"
    
    def test_delete_speaker(self, client):
        """测试删除说话人"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.delete_speaker.return_value = True
            mock_get_pool.return_value = mock_pool
            
            response = client.delete("/api/v1/speaker/speaker-123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "说话人删除成功"
    
    def test_delete_speaker_not_found(self, client):
        """测试删除不存在的说话人"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.delete_speaker.return_value = False
            mock_get_pool.return_value = mock_pool
            
            response = client.delete("/api/v1/speaker/nonexistent")
            
            assert response.status_code == 404
            data = response.json()
            assert "说话人不存在" in data["detail"]
    
    def test_batch_register_speakers(self, client, mock_audio_file):
        """测试批量注册说话人"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            with patch('app.api.speaker.get_model_manager') as mock_get_manager:
                mock_pool = AsyncMock()
                mock_pool.register_speaker.side_effect = ["speaker-1", "speaker-2", "speaker-3"]
                mock_get_pool.return_value = mock_pool
                
                mock_manager = AsyncMock()
                mock_manager.extract_speaker_embedding.return_value = [0.1] * 512
                mock_manager.speaker_model = MagicMock()
                mock_get_manager.return_value = mock_manager
                
                # 准备多个文件
                files = []
                speaker_names = ["张三", "李四", "王五"]
                for i, name in enumerate(speaker_names):
                    with open(mock_audio_file, 'rb') as f:
                        files.append(("audio_files", (f"speaker{i}.wav", f.read(), "audio/wav")))
                
                response = client.post(
                    "/api/v1/speaker/batch-register",
                    files=files,
                    data={"speaker_names": ",".join(speaker_names)}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert len(data["results"]) == 3
                assert all(result["success"] for result in data["results"])
    
    def test_speaker_similarity_comparison(self, client, mock_audio_file):
        """测试说话人相似度比较"""
        with patch('app.api.speaker.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.extract_speaker_embedding.side_effect = [
                [0.1] * 512,  # 第一个音频的嵌入
                [0.2] * 512   # 第二个音频的嵌入
            ]
            mock_manager.speaker_model = MagicMock()
            mock_get_manager.return_value = mock_manager
            
            with patch('app.api.speaker.cosine_similarity') as mock_similarity:
                mock_similarity.return_value = 0.85
                
                files = []
                for i in range(2):
                    with open(mock_audio_file, 'rb') as f:
                        files.append(("audio_files", (f"audio{i}.wav", f.read(), "audio/wav")))
                
                response = client.post(
                    "/api/v1/speaker/compare",
                    files=files
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["similarity"] == 0.85
                assert "is_same_speaker" in data
    
    def test_speaker_statistics(self, client):
        """测试说话人统计信息"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.get_statistics.return_value = {
                "total_speakers": 10,
                "total_audio_samples": 50,
                "average_similarity": 0.87,
                "most_active_speaker": {"id": "speaker-1", "name": "张三", "count": 15}
            }
            mock_get_pool.return_value = mock_pool
            
            response = client.get("/api/v1/speaker/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_speakers"] == 10
            assert data["total_audio_samples"] == 50
            assert data["most_active_speaker"]["name"] == "张三"