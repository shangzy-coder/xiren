"""
流水线集成测试
"""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import json

from app.main import app


class TestPipelineIntegration:
    """流水线集成测试"""
    
    @pytest.mark.integration
    def test_full_pipeline_flow(self, client, mock_audio_file):
        """测试完整的流水线流程"""
        # 1. 首先初始化模型
        with patch('app.api.asr.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.load_asr_model.return_value = True
            mock_manager.load_vad_model.return_value = True
            mock_manager.load_speaker_model.return_value = True
            mock_get_manager.return_value = mock_manager
            
            init_response = client.post("/api/v1/asr/initialize", data={
                "model_type": "sense_voice",
                "use_gpu": False,
                "enable_vad": True,
                "enable_speaker_id": True
            })
            assert init_response.status_code == 200
        
        # 2. 提交流水线任务
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.submit_pipeline.return_value = "pipeline-123"
            mock_get_orchestrator.return_value = mock_orchestrator
            
            with open(mock_audio_file, 'rb') as f:
                pipeline_response = client.post(
                    "/api/v1/pipeline/submit",
                    files={"audio_file": ("test.wav", f, "audio/wav")},
                    data={
                        "enable_vad": True,
                        "enable_asr": True,
                        "enable_speaker_id": True,
                        "priority": "normal"
                    }
                )
            
            assert pipeline_response.status_code == 200
            pipeline_data = pipeline_response.json()
            assert pipeline_data["pipeline_id"] == "pipeline-123"
            assert pipeline_data["success"] is True
        
        # 3. 检查流水线状态
        with patch('app.api.pipeline.get_pipeline_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_pipeline_status.return_value = {
                "pipeline_id": "pipeline-123",
                "status": "completed",
                "progress": 100,
                "results": {
                    "asr": {"text": "测试转录结果"},
                    "speaker": {"speaker_id": "speaker-1", "confidence": 0.9}
                }
            }
            mock_get_orchestrator.return_value = mock_orchestrator
            
            status_response = client.get("/api/v1/pipeline/status/pipeline-123")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["status"] == "completed"
            assert status_data["progress"] == 100
    
    @pytest.mark.integration
    def test_comprehensive_processing_flow(self, client, mock_audio_file):
        """测试综合处理流程"""
        with patch('app.api.comprehensive.get_model_manager') as mock_get_manager:
            with patch('app.api.comprehensive.get_speaker_pool') as mock_get_pool:
                with patch('app.api.comprehensive.get_async_session') as mock_get_session:
                    # 设置模拟对象
                    mock_manager = AsyncMock()
                    mock_manager.transcribe.return_value = "综合处理测试结果"
                    mock_manager.extract_speaker_embedding.return_value = [0.1] * 512
                    mock_manager.detect_voice_activity.return_value = [(0.0, 2.0)]
                    mock_get_manager.return_value = mock_manager
                    
                    mock_pool = AsyncMock()
                    mock_pool.identify_speaker.return_value = {
                        "speaker_id": "speaker-1",
                        "similarity": 0.85,
                        "confidence": 0.9
                    }
                    mock_get_pool.return_value = mock_pool
                    
                    mock_session = AsyncMock()
                    mock_get_session.return_value.__aenter__.return_value = mock_session
                    
                    # 提交综合处理请求
                    with open(mock_audio_file, 'rb') as f:
                        response = client.post(
                            "/api/v1/process",
                            files={"audio_file": ("test.wav", f, "audio/wav")},
                            data={
                                "enable_asr": True,
                                "enable_speaker_id": True,
                                "enable_vad": True,
                                "save_to_database": True
                            }
                        )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
                    assert data["asr_result"]["text"] == "综合处理测试结果"
                    assert data["speaker_result"]["speaker_id"] == "speaker-1"
                    assert len(data["vad_result"]) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_integration(self):
        """测试WebSocket集成"""
        from fastapi.testclient import TestClient
        import websockets
        import base64
        
        with patch('app.core.websocket_manager.get_websocket_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.connect.return_value = "ws-connection-123"
            mock_manager.process_audio_chunk.return_value = {
                "type": "transcription",
                "text": "WebSocket测试结果",
                "timestamp": 1.23
            }
            mock_get_manager.return_value = mock_manager
            
            with TestClient(app) as client:
                with client.websocket_connect("/api/v1/websocket/stream") as websocket:
                    # 发送音频数据
                    audio_data = b'\x00' * 1000  # 模拟音频数据
                    message = {
                        "type": "audio",
                        "data": base64.b64encode(audio_data).decode()
                    }
                    websocket.send_json(message)
                    
                    # 接收响应
                    response = websocket.receive_json()
                    assert response["type"] == "transcription"
                    assert "text" in response
                    
                    # 结束会话
                    websocket.send_json({"type": "end"})
                    end_response = websocket.receive_json()
                    assert end_response["type"] == "end"
    
    @pytest.mark.integration
    def test_queue_system_integration(self, client, mock_audio_file):
        """测试队列系统集成"""
        with patch('app.api.queue_example.get_queue_manager') as mock_get_queue:
            mock_manager = AsyncMock()
            mock_manager.submit_task.return_value = "queue-task-123"
            mock_manager.get_task_status.return_value = {
                "task_id": "queue-task-123",
                "status": "completed",
                "result": {"text": "队列处理结果"}
            }
            mock_manager.get_stats.return_value = {
                "total_tasks": 10,
                "completed_tasks": 8,
                "failed_tasks": 1,
                "pending_tasks": 1
            }
            mock_get_queue.return_value = mock_manager
            
            # 提交队列任务
            with open(mock_audio_file, 'rb') as f:
                submit_response = client.post(
                    "/api/v1/queue/submit",
                    files={"audio_file": ("test.wav", f, "audio/wav")},
                    data={"priority": "high", "task_type": "asr"}
                )
            
            assert submit_response.status_code == 200
            submit_data = submit_response.json()
            assert submit_data["task_id"] == "queue-task-123"
            
            # 检查任务状态
            status_response = client.get("/api/v1/queue/status/queue-task-123")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["status"] == "completed"
            
            # 获取队列统计
            stats_response = client.get("/api/v1/queue/stats")
            assert stats_response.status_code == 200
            stats_data = stats_response.json()
            assert stats_data["total_tasks"] == 10
    
    @pytest.mark.integration
    def test_speaker_registration_and_identification(self, client, mock_audio_file):
        """测试声纹注册和识别集成"""
        with patch('app.api.speaker.get_speaker_pool') as mock_get_pool:
            with patch('app.api.speaker.get_model_manager') as mock_get_manager:
                # 设置模拟对象
                mock_pool = AsyncMock()
                mock_manager = AsyncMock()
                mock_manager.extract_speaker_embedding.return_value = [0.1] * 512
                mock_get_manager.return_value = mock_manager
                mock_get_pool.return_value = mock_pool
                
                # 1. 注册声纹
                mock_pool.register_speaker.return_value = "speaker-123"
                
                with open(mock_audio_file, 'rb') as f:
                    register_response = client.post(
                        "/api/v1/speaker/register",
                        files={"audio_file": ("speaker.wav", f, "audio/wav")},
                        data={"speaker_name": "测试说话人"}
                    )
                
                assert register_response.status_code == 200
                register_data = register_response.json()
                assert register_data["speaker_id"] == "speaker-123"
                
                # 2. 识别声纹
                mock_pool.identify_speaker.return_value = {
                    "speaker_id": "speaker-123",
                    "similarity": 0.92,
                    "confidence": 0.95
                }
                
                with open(mock_audio_file, 'rb') as f:
                    identify_response = client.post(
                        "/api/v1/speaker/identify",
                        files={"audio_file": ("unknown.wav", f, "audio/wav")}
                    )
                
                assert identify_response.status_code == 200
                identify_data = identify_response.json()
                assert identify_data["speaker_id"] == "speaker-123"
                assert identify_data["similarity"] > 0.9
    
    @pytest.mark.integration
    def test_error_handling_integration(self, client):
        """测试错误处理集成"""
        # 测试无效文件上传
        invalid_file = b"not an audio file"
        
        response = client.post(
            "/api/v1/asr/transcribe",
            files={"file": ("invalid.txt", invalid_file, "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        
        # 测试不存在的任务查询
        response = client.get("/api/v1/queue/status/nonexistent-task")
        assert response.status_code == 404
        
        # 测试不存在的流水线查询
        response = client.get("/api/v1/pipeline/status/nonexistent-pipeline")
        assert response.status_code == 404
    
    @pytest.mark.integration
    def test_monitoring_integration(self, client):
        """测试监控集成"""
        # 测试健康检查
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        
        # 测试指标端点
        with patch('app.main.settings.ENABLE_METRICS', True):
            metrics_response = client.get("/metrics")
            assert metrics_response.status_code == 200
            assert "text/plain" in metrics_response.headers["content-type"]