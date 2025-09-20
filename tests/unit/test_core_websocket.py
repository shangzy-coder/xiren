"""
WebSocket管理器单元测试
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.core.websocket_manager import (
    WebSocketManager,
    ConnectionInfo,
    get_websocket_manager,
    cleanup_websocket_manager
)


class TestConnectionInfo:
    """连接信息测试"""
    
    def test_connection_info_creation(self):
        """测试连接信息创建"""
        websocket = MagicMock()
        conn_info = ConnectionInfo(
            connection_id="conn-123",
            websocket=websocket,
            client_ip="192.168.1.1"
        )
        
        assert conn_info.connection_id == "conn-123"
        assert conn_info.websocket is websocket
        assert conn_info.client_ip == "192.168.1.1"
        assert isinstance(conn_info.connected_at, datetime)
        assert conn_info.last_activity is not None
        assert conn_info.message_count == 0
    
    def test_update_activity(self):
        """测试更新活动时间"""
        conn_info = ConnectionInfo("conn-123", MagicMock())
        original_time = conn_info.last_activity
        
        # 等待一小段时间
        import time
        time.sleep(0.01)
        
        conn_info.update_activity()
        assert conn_info.last_activity > original_time
    
    def test_increment_message_count(self):
        """测试增加消息计数"""
        conn_info = ConnectionInfo("conn-123", MagicMock())
        assert conn_info.message_count == 0
        
        conn_info.increment_message_count()
        assert conn_info.message_count == 1
        
        conn_info.increment_message_count()
        assert conn_info.message_count == 2


class TestWebSocketManager:
    """WebSocket管理器测试"""
    
    @pytest.fixture
    def websocket_manager(self):
        """创建WebSocket管理器实例"""
        return WebSocketManager()
    
    def test_websocket_manager_initialization(self, websocket_manager):
        """测试WebSocket管理器初始化"""
        assert websocket_manager._connections == {}
        assert websocket_manager._connection_count == 0
        assert websocket_manager._message_count == 0
        assert websocket_manager._max_connections == 100
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self, websocket_manager):
        """测试WebSocket连接"""
        mock_websocket = MagicMock()
        mock_websocket.client = MagicMock()
        mock_websocket.client.host = "192.168.1.1"
        
        connection_id = await websocket_manager.connect(mock_websocket)
        
        assert isinstance(connection_id, str)
        assert len(connection_id) > 0
        assert connection_id in websocket_manager._connections
        assert websocket_manager._connection_count == 1
    
    @pytest.mark.asyncio
    async def test_connect_max_connections_exceeded(self, websocket_manager):
        """测试超过最大连接数"""
        websocket_manager._max_connections = 1
        
        # 添加第一个连接
        mock_ws1 = MagicMock()
        mock_ws1.client = MagicMock()
        mock_ws1.client.host = "192.168.1.1"
        
        conn1 = await websocket_manager.connect(mock_ws1)
        assert conn1 is not None
        
        # 尝试添加第二个连接（应该失败）
        mock_ws2 = MagicMock()
        mock_ws2.client = MagicMock()
        mock_ws2.client.host = "192.168.1.2"
        
        with pytest.raises(Exception, match="连接数已达上限"):
            await websocket_manager.connect(mock_ws2)
    
    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, websocket_manager):
        """测试WebSocket断开连接"""
        mock_websocket = MagicMock()
        mock_websocket.client = MagicMock()
        mock_websocket.client.host = "192.168.1.1"
        
        # 建立连接
        connection_id = await websocket_manager.connect(mock_websocket)
        assert websocket_manager._connection_count == 1
        
        # 断开连接
        await websocket_manager.disconnect(connection_id)
        assert connection_id not in websocket_manager._connections
        assert websocket_manager._connection_count == 0
    
    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_connection(self, websocket_manager):
        """测试断开不存在的连接"""
        # 断开不存在的连接不应该抛出异常
        await websocket_manager.disconnect("nonexistent-connection")
        assert websocket_manager._connection_count == 0
    
    @pytest.mark.asyncio
    async def test_send_message(self, websocket_manager):
        """测试发送消息"""
        mock_websocket = MagicMock()
        mock_websocket.client = MagicMock()
        mock_websocket.client.host = "192.168.1.1"
        mock_websocket.send_json = AsyncMock()
        
        # 建立连接
        connection_id = await websocket_manager.connect(mock_websocket)
        
        # 发送消息
        message = {"type": "test", "data": "hello"}
        await websocket_manager.send_message(connection_id, message)
        
        mock_websocket.send_json.assert_called_once_with(message)
        
        # 检查消息计数
        conn_info = websocket_manager._connections[connection_id]
        assert conn_info.message_count == 1
    
    @pytest.mark.asyncio
    async def test_send_message_to_nonexistent_connection(self, websocket_manager):
        """测试向不存在的连接发送消息"""
        message = {"type": "test", "data": "hello"}
        
        # 向不存在的连接发送消息不应该抛出异常
        await websocket_manager.send_message("nonexistent", message)
    
    @pytest.mark.asyncio
    async def test_send_message_websocket_error(self, websocket_manager):
        """测试发送消息时WebSocket错误"""
        mock_websocket = MagicMock()
        mock_websocket.client = MagicMock()
        mock_websocket.client.host = "192.168.1.1"
        mock_websocket.send_json = AsyncMock(side_effect=Exception("Connection closed"))
        
        # 建立连接
        connection_id = await websocket_manager.connect(mock_websocket)
        
        # 发送消息（应该处理异常）
        message = {"type": "test", "data": "hello"}
        await websocket_manager.send_message(connection_id, message)
        
        # 连接应该被自动清理
        assert connection_id not in websocket_manager._connections
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, websocket_manager):
        """测试广播消息"""
        # 建立多个连接
        connections = []
        for i in range(3):
            mock_ws = MagicMock()
            mock_ws.client = MagicMock()
            mock_ws.client.host = f"192.168.1.{i+1}"
            mock_ws.send_json = AsyncMock()
            
            conn_id = await websocket_manager.connect(mock_ws)
            connections.append((conn_id, mock_ws))
        
        # 广播消息
        message = {"type": "broadcast", "data": "hello all"}
        await websocket_manager.broadcast(message)
        
        # 验证所有连接都收到消息
        for conn_id, mock_ws in connections:
            mock_ws.send_json.assert_called_with(message)
    
    @pytest.mark.asyncio
    async def test_broadcast_with_exclude(self, websocket_manager):
        """测试排除特定连接的广播"""
        # 建立多个连接
        connections = []
        for i in range(3):
            mock_ws = MagicMock()
            mock_ws.client = MagicMock()
            mock_ws.client.host = f"192.168.1.{i+1}"
            mock_ws.send_json = AsyncMock()
            
            conn_id = await websocket_manager.connect(mock_ws)
            connections.append((conn_id, mock_ws))
        
        # 广播消息，排除第一个连接
        message = {"type": "broadcast", "data": "hello others"}
        exclude_conn = connections[0][0]
        await websocket_manager.broadcast(message, exclude=[exclude_conn])
        
        # 验证被排除的连接没有收到消息
        connections[0][1].send_json.assert_not_called()
        
        # 验证其他连接收到消息
        for conn_id, mock_ws in connections[1:]:
            mock_ws.send_json.assert_called_with(message)
    
    @pytest.mark.asyncio
    async def test_get_connection_info(self, websocket_manager):
        """测试获取连接信息"""
        mock_websocket = MagicMock()
        mock_websocket.client = MagicMock()
        mock_websocket.client.host = "192.168.1.1"
        
        # 建立连接
        connection_id = await websocket_manager.connect(mock_websocket)
        
        # 获取连接信息
        conn_info = await websocket_manager.get_connection_info(connection_id)
        
        assert conn_info is not None
        assert conn_info.connection_id == connection_id
        assert conn_info.client_ip == "192.168.1.1"
    
    @pytest.mark.asyncio
    async def test_get_connection_info_nonexistent(self, websocket_manager):
        """测试获取不存在连接的信息"""
        conn_info = await websocket_manager.get_connection_info("nonexistent")
        assert conn_info is None
    
    @pytest.mark.asyncio
    async def test_get_stats(self, websocket_manager):
        """测试获取统计信息"""
        # 建立一些连接
        for i in range(3):
            mock_ws = MagicMock()
            mock_ws.client = MagicMock()
            mock_ws.client.host = f"192.168.1.{i+1}"
            await websocket_manager.connect(mock_ws)
        
        # 模拟一些消息
        websocket_manager._message_count = 10
        
        stats = await websocket_manager.get_stats()
        
        assert stats["active_connections"] == 3
        assert stats["total_connections"] == 3
        assert stats["total_messages"] == 10
        assert "uptime_seconds" in stats
    
    @pytest.mark.asyncio
    async def test_cleanup_inactive_connections(self, websocket_manager):
        """测试清理不活跃连接"""
        mock_websocket = MagicMock()
        mock_websocket.client = MagicMock()
        mock_websocket.client.host = "192.168.1.1"
        
        # 建立连接
        connection_id = await websocket_manager.connect(mock_websocket)
        
        # 模拟连接不活跃（修改最后活动时间）
        conn_info = websocket_manager._connections[connection_id]
        conn_info.last_activity = datetime.now().timestamp() - 3700  # 1小时前
        
        # 清理不活跃连接
        cleaned_count = await websocket_manager.cleanup_inactive_connections(timeout=3600)
        
        assert cleaned_count == 1
        assert connection_id not in websocket_manager._connections
    
    @pytest.mark.asyncio
    async def test_process_audio_chunk(self, websocket_manager):
        """测试处理音频数据块"""
        with patch('app.core.websocket_manager.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.transcribe.return_value = "测试转录结果"
            mock_manager.asr_model = MagicMock()  # 模型已加载
            mock_get_manager.return_value = mock_manager
            
            mock_websocket = MagicMock()
            mock_websocket.client = MagicMock()
            mock_websocket.client.host = "192.168.1.1"
            
            # 建立连接
            connection_id = await websocket_manager.connect(mock_websocket)
            
            # 处理音频数据
            audio_data = b'\x00' * 1000  # 模拟音频数据
            result = await websocket_manager.process_audio_chunk(connection_id, audio_data)
            
            assert result is not None
            assert result["type"] == "transcription"
            assert result["text"] == "测试转录结果"
    
    @pytest.mark.asyncio
    async def test_process_audio_chunk_no_model(self, websocket_manager):
        """测试没有模型时处理音频"""
        with patch('app.core.websocket_manager.get_model_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.asr_model = None  # 模型未加载
            mock_get_manager.return_value = mock_manager
            
            mock_websocket = MagicMock()
            mock_websocket.client = MagicMock()
            mock_websocket.client.host = "192.168.1.1"
            
            # 建立连接
            connection_id = await websocket_manager.connect(mock_websocket)
            
            # 处理音频数据
            audio_data = b'\x00' * 1000
            result = await websocket_manager.process_audio_chunk(connection_id, audio_data)
            
            assert result is not None
            assert result["type"] == "error"
            assert "模型未初始化" in result["message"]


class TestWebSocketManagerSingleton:
    """WebSocket管理器单例测试"""
    
    @pytest.mark.asyncio
    async def test_get_websocket_manager_singleton(self):
        """测试WebSocket管理器单例模式"""
        manager1 = await get_websocket_manager()
        manager2 = await get_websocket_manager()
        
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_cleanup_websocket_manager(self):
        """测试清理WebSocket管理器"""
        manager = await get_websocket_manager()
        
        # 添加一些连接
        mock_ws = MagicMock()
        mock_ws.client = MagicMock()
        mock_ws.client.host = "192.168.1.1"
        await manager.connect(mock_ws)
        
        assert manager._connection_count > 0
        
        # 清理管理器
        await cleanup_websocket_manager()
        
        # 验证连接已清理
        assert manager._connection_count == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_connections(self):
        """测试并发连接"""
        manager = await get_websocket_manager()
        
        # 并发建立多个连接
        async def create_connection(i):
            mock_ws = MagicMock()
            mock_ws.client = MagicMock()
            mock_ws.client.host = f"192.168.1.{i}"
            return await manager.connect(mock_ws)
        
        # 创建10个并发连接
        connection_ids = await asyncio.gather(*[
            create_connection(i) for i in range(10)
        ])
        
        assert len(connection_ids) == 10
        assert len(set(connection_ids)) == 10  # 所有ID都是唯一的
        assert manager._connection_count == 10
        
        # 清理
        for conn_id in connection_ids:
            await manager.disconnect(conn_id)
        
        assert manager._connection_count == 0