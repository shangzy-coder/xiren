"""
主应用API测试
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app


class TestMainApp:
    """主应用测试"""
    
    def test_app_creation(self):
        """测试应用创建"""
        assert app.title == "语音识别服务"
        assert app.version == "0.1.0"
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
    
    def test_health_endpoint(self, client):
        """测试健康检查端点"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_metrics_endpoint_enabled(self, client):
        """测试指标端点（启用时）"""
        with patch('app.main.settings.ENABLE_METRICS', True):
            response = client.get("/metrics")
            # 指标端点应该返回Prometheus格式的文本
            assert response.status_code == 200
            assert "text/plain" in response.headers["content-type"]
    
    def test_cors_headers(self, client):
        """测试CORS头"""
        response = client.options("/health")
        assert response.status_code == 200
        # 检查CORS头是否存在
        assert "access-control-allow-origin" in response.headers
    
    def test_root_redirect(self, client):
        """测试根路径重定向"""
        response = client.get("/", allow_redirects=False)
        # 根路径应该重定向到文档
        assert response.status_code in [307, 302]
    
    def test_openapi_schema(self, client):
        """测试OpenAPI模式"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert schema["info"]["title"] == "语音识别服务"
        assert schema["info"]["version"] == "0.1.0"
        
        # 检查主要路径是否存在
        paths = schema["paths"]
        assert "/api/v1/asr/transcribe" in paths
        assert "/api/v1/speaker/register" in paths
        assert "/api/v1/process" in paths
    
    def test_docs_endpoint(self, client):
        """测试文档端点"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self, client):
        """测试ReDoc端点"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_startup_event(self):
        """测试启动事件"""
        with patch('app.main.initialize_database') as mock_init_db:
            with patch('app.main.get_queue_manager') as mock_queue:
                with TestClient(app) as client:
                    # 启动事件应该被触发
                    mock_init_db.assert_called_once()
    
    def test_shutdown_event(self):
        """测试关闭事件"""
        with patch('app.main.shutdown_queue_manager') as mock_shutdown_queue:
            with patch('app.main.cleanup_websocket_manager') as mock_cleanup_ws:
                with TestClient(app) as client:
                    pass
                # 关闭事件应该被触发
                mock_shutdown_queue.assert_called_once()
                mock_cleanup_ws.assert_called_once()
    
    def test_middleware_order(self):
        """测试中间件顺序"""
        # 检查中间件是否正确注册
        middleware_types = [type(middleware.cls).__name__ for middleware in app.user_middleware]
        assert "CORSMiddleware" in middleware_types
    
    def test_router_inclusion(self):
        """测试路由器包含"""
        # 获取所有路由
        routes = [route.path for route in app.routes]
        
        # 检查主要API路由是否包含
        expected_prefixes = [
            "/api/v1/asr",
            "/api/v1/speaker", 
            "/api/v1",  # comprehensive
            "/api/v1/pipeline",
            "/api/v1/queue",
            "/api/v1/websocket"
        ]
        
        for prefix in expected_prefixes:
            # 至少应该有一个路由以该前缀开始
            assert any(route.startswith(prefix) for route in routes), f"Missing routes with prefix: {prefix}"
    
    def test_tags_configuration(self):
        """测试标签配置"""
        # 检查OpenAPI标签
        schema = TestClient(app).get("/openapi.json").json()
        
        # 提取所有标签
        all_tags = set()
        for path_info in schema["paths"].values():
            for method_info in path_info.values():
                if "tags" in method_info:
                    all_tags.update(method_info["tags"])
        
        expected_tags = {
            "语音识别",
            "声纹识别", 
            "综合处理",
            "语音处理流水线",
            "队列系统示例",
            "WebSocket实时通信",
            "监控"
        }
        
        # 检查预期标签是否存在
        assert expected_tags.issubset(all_tags)