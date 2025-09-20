"""
流水线API测试
"""
import pytest
import io
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from app.main import app


class TestPipelineAPI:
    """流水线API测试"""
    
    def test_submit_pipeline_success(self, client, mock_audio_file):
        """测试提交流水线成功"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.submit_pipeline.return_value = "pipeline-123"
            mock_get_orchestrator.return_value = mock_orchestrator
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/pipeline/submit",
                    files={"audio_file": ("test.wav", f, "audio/wav")},
                    data={
                        "enable_vad": True,
                        "enable_asr": True,
                        "enable_speaker_id": True,
                        "enable_diarization": False,
                        "priority": "normal"
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["pipeline_id"] == "pipeline-123"
            assert "estimated_completion_time" in data
            assert "submitted_at" in data
    
    def test_submit_pipeline_high_priority(self, client, mock_audio_file):
        """测试提交高优先级流水线"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.submit_pipeline.return_value = "pipeline-urgent-456"
            mock_get_orchestrator.return_value = mock_orchestrator
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/pipeline/submit",
                    files={"audio_file": ("urgent.wav", f, "audio/wav")},
                    data={
                        "enable_vad": True,
                        "enable_asr": True,
                        "priority": "urgent"
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["pipeline_id"] == "pipeline-urgent-456"
            assert data["priority"] == "urgent"
    
    def test_submit_pipeline_minimal_config(self, client, mock_audio_file):
        """测试最小配置流水线"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.submit_pipeline.return_value = "pipeline-minimal-789"
            mock_get_orchestrator.return_value = mock_orchestrator
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/pipeline/submit",
                    files={"audio_file": ("minimal.wav", f, "audio/wav")},
                    data={
                        "enable_vad": False,
                        "enable_asr": True,
                        "enable_speaker_id": False,
                        "enable_diarization": False
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["pipeline_id"] == "pipeline-minimal-789"
    
    def test_submit_pipeline_invalid_file(self, client):
        """测试提交无效文件"""
        fake_file = io.BytesIO(b"not an audio file")
        
        response = client.post(
            "/api/v1/pipeline/submit",
            files={"audio_file": ("test.txt", fake_file, "text/plain")},
            data={"enable_asr": True}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "不支持的文件格式" in data["detail"] or "音频文件无效" in data["detail"]
    
    def test_submit_pipeline_no_features_enabled(self, client, mock_audio_file):
        """测试没有启用任何功能"""
        with open(mock_audio_file, 'rb') as f:
            response = client.post(
                "/api/v1/pipeline/submit",
                files={"audio_file": ("test.wav", f, "audio/wav")},
                data={
                    "enable_vad": False,
                    "enable_asr": False,
                    "enable_speaker_id": False,
                    "enable_diarization": False
                }
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "至少需要启用一个处理功能" in data["detail"]
    
    def test_get_pipeline_status_success(self, client):
        """测试获取流水线状态成功"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_pipeline_status.return_value = {
                "pipeline_id": "pipeline-123",
                "status": "processing",
                "progress": 60,
                "current_stage": "asr",
                "stages": {
                    "preprocessing": {"status": "completed", "result": {"duration": 5.2}},
                    "vad": {"status": "completed", "result": {"segments": 3}},
                    "asr": {"status": "processing", "progress": 60}
                },
                "created_at": "2023-01-01T12:00:00Z",
                "estimated_completion": "2023-01-01T12:05:00Z"
            }
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/pipeline/status/pipeline-123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["pipeline_id"] == "pipeline-123"
            assert data["status"] == "processing"
            assert data["progress"] == 60
            assert data["current_stage"] == "asr"
            assert "stages" in data
    
    def test_get_pipeline_status_completed(self, client):
        """测试获取已完成流水线状态"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_pipeline_status.return_value = {
                "pipeline_id": "pipeline-456",
                "status": "completed",
                "progress": 100,
                "results": {
                    "asr": {"text": "完整转录结果", "confidence": 0.95},
                    "speaker": {"speaker_id": "speaker-1", "confidence": 0.9},
                    "vad": {"segments": [(0.0, 2.5), (3.0, 5.8)]}
                },
                "total_processing_time": 12.5,
                "completed_at": "2023-01-01T12:10:00Z"
            }
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/pipeline/status/pipeline-456")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["progress"] == 100
            assert "results" in data
            assert data["results"]["asr"]["text"] == "完整转录结果"
    
    def test_get_pipeline_status_not_found(self, client):
        """测试获取不存在的流水线状态"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_pipeline_status.return_value = None
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/pipeline/status/nonexistent")
            
            assert response.status_code == 404
            data = response.json()
            assert "流水线不存在" in data["detail"]
    
    def test_get_pipeline_status_failed(self, client):
        """测试获取失败流水线状态"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_pipeline_status.return_value = {
                "pipeline_id": "pipeline-failed",
                "status": "failed",
                "progress": 30,
                "error": "ASR模型加载失败",
                "failed_stage": "asr",
                "failed_at": "2023-01-01T12:03:00Z"
            }
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/pipeline/status/pipeline-failed")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "failed"
            assert data["error"] == "ASR模型加载失败"
            assert data["failed_stage"] == "asr"
    
    def test_cancel_pipeline_success(self, client):
        """测试取消流水线成功"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.cancel_pipeline.return_value = True
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post("/api/v1/pipeline/cancel/pipeline-123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "流水线已取消"
    
    def test_cancel_pipeline_not_found(self, client):
        """测试取消不存在的流水线"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.cancel_pipeline.return_value = False
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post("/api/v1/pipeline/cancel/nonexistent")
            
            assert response.status_code == 404
            data = response.json()
            assert "流水线不存在或无法取消" in data["detail"]
    
    def test_get_pipeline_results(self, client):
        """测试获取流水线结果"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_pipeline_results.return_value = {
                "pipeline_id": "pipeline-results",
                "results": {
                    "asr": {
                        "text": "这是完整的转录结果",
                        "confidence": 0.92,
                        "language": "zh",
                        "segments": [
                            {"start": 0.0, "end": 2.5, "text": "这是完整的"},
                            {"start": 2.5, "end": 5.0, "text": "转录结果"}
                        ]
                    },
                    "speaker": {
                        "speaker_id": "speaker-main",
                        "speaker_name": "主要说话人",
                        "confidence": 0.88
                    },
                    "vad": {
                        "segments": [(0.0, 5.0)],
                        "speech_duration": 5.0,
                        "silence_duration": 0.0
                    }
                },
                "metadata": {
                    "file_duration": 5.0,
                    "processing_time": 8.2,
                    "quality_score": 0.9
                }
            }
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/pipeline/results/pipeline-results")
            
            assert response.status_code == 200
            data = response.json()
            assert data["pipeline_id"] == "pipeline-results"
            assert "results" in data
            assert data["results"]["asr"]["text"] == "这是完整的转录结果"
            assert "metadata" in data
    
    def test_get_pipeline_list(self, client):
        """测试获取流水线列表"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_pipeline_list.return_value = {
                "pipelines": [
                    {
                        "pipeline_id": "pipeline-1",
                        "status": "completed",
                        "created_at": "2023-01-01T12:00:00Z",
                        "progress": 100
                    },
                    {
                        "pipeline_id": "pipeline-2", 
                        "status": "processing",
                        "created_at": "2023-01-01T12:05:00Z",
                        "progress": 45
                    }
                ],
                "total": 2,
                "page": 1,
                "per_page": 10
            }
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/pipeline/list")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["pipelines"]) == 2
            assert data["total"] == 2
            assert data["page"] == 1
    
    def test_get_pipeline_list_with_filters(self, client):
        """测试带过滤器的流水线列表"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_pipeline_list.return_value = {
                "pipelines": [
                    {
                        "pipeline_id": "pipeline-completed",
                        "status": "completed",
                        "created_at": "2023-01-01T12:00:00Z",
                        "progress": 100
                    }
                ],
                "total": 1,
                "page": 1,
                "per_page": 10
            }
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/pipeline/list?status=completed&page=1&per_page=5")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["pipelines"]) == 1
            assert data["pipelines"][0]["status"] == "completed"
    
    def test_get_pipeline_stats(self, client):
        """测试获取流水线统计"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_stats.return_value = {
                "total_pipelines": 100,
                "completed_pipelines": 85,
                "failed_pipelines": 5,
                "processing_pipelines": 10,
                "average_processing_time": 15.2,
                "success_rate": 0.85,
                "most_common_stages": {
                    "asr": 95,
                    "vad": 80,
                    "speaker_id": 60
                }
            }
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/pipeline/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_pipelines"] == 100
            assert data["success_rate"] == 0.85
            assert "most_common_stages" in data
    
    def test_test_pipeline_system(self, client):
        """测试流水线系统测试"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.submit_pipeline.return_value = "test-pipeline-123"
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post("/api/v1/pipeline/test")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["test_pipeline_id"] == "test-pipeline-123"
            assert "message" in data
    
    def test_pipeline_webhook_callback(self, client):
        """测试流水线Webhook回调"""
        webhook_data = {
            "pipeline_id": "pipeline-webhook",
            "status": "completed",
            "results": {
                "asr": {"text": "Webhook测试结果"}
            },
            "timestamp": "2023-01-01T12:00:00Z"
        }
        
        with patch('app.api.pipeline.process_webhook_callback') as mock_process:
            mock_process.return_value = {"success": True}
            
            response = client.post("/api/v1/pipeline/webhook", json=webhook_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_pipeline_retry(self, client):
        """测试重试失败的流水线"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.retry_pipeline.return_value = "pipeline-retry-456"
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post("/api/v1/pipeline/retry/pipeline-failed-123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["new_pipeline_id"] == "pipeline-retry-456"
    
    def test_pipeline_error_handling(self, client, mock_audio_file):
        """测试流水线错误处理"""
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.submit_pipeline.side_effect = Exception("流水线提交失败")
            mock_get_orchestrator.return_value = mock_orchestrator
            
            with open(mock_audio_file, 'rb') as f:
                response = client.post(
                    "/api/v1/pipeline/submit",
                    files={"audio_file": ("test.wav", f, "audio/wav")},
                    data={"enable_asr": True}
                )
            
            assert response.status_code == 500
            data = response.json()
            assert "流水线提交失败" in data["detail"]