"""
服务模块单元测试
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os

from app.services.db import (
    get_async_session,
    initialize_database,
    create_tables,
    Base
)
from app.services.storage import (
    StorageManager,
    get_storage_manager,
    StorageError
)


class TestDatabase:
    """数据库服务测试"""
    
    @pytest.mark.asyncio
    async def test_get_async_session(self):
        """测试获取异步数据库会话"""
        with patch('app.services.db.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            
            async with get_async_session() as session:
                assert session is mock_session
    
    @pytest.mark.asyncio
    async def test_initialize_database(self):
        """测试数据库初始化"""
        with patch('app.services.db.create_async_engine') as mock_create_engine:
            with patch('app.services.db.create_tables') as mock_create_tables:
                mock_engine = AsyncMock()
                mock_create_engine.return_value = mock_engine
                mock_create_tables.return_value = None
                
                await initialize_database()
                
                mock_create_engine.assert_called_once()
                mock_create_tables.assert_called_once_with(mock_engine)
    
    @pytest.mark.asyncio
    async def test_create_tables(self):
        """测试创建数据库表"""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        await create_tables(mock_engine)
        
        mock_engine.begin.assert_called_once()
        mock_conn.run_sync.assert_called_once()
    
    def test_base_model(self):
        """测试基础模型类"""
        assert hasattr(Base, 'metadata')
        assert hasattr(Base, 'registry')


class TestStorageManager:
    """存储管理器测试"""
    
    @pytest.fixture
    def storage_manager(self):
        """创建存储管理器实例"""
        return StorageManager(
            endpoint="localhost:9000",
            access_key="test_key",
            secret_key="test_secret",
            secure=False
        )
    
    def test_storage_manager_initialization(self, storage_manager):
        """测试存储管理器初始化"""
        assert storage_manager.endpoint == "localhost:9000"
        assert storage_manager.access_key == "test_key"
        assert storage_manager.secret_key == "test_secret"
        assert storage_manager.secure is False
        assert storage_manager.client is not None
    
    @pytest.mark.asyncio
    async def test_create_bucket_success(self, storage_manager):
        """测试创建存储桶成功"""
        with patch.object(storage_manager.client, 'bucket_exists') as mock_exists:
            with patch.object(storage_manager.client, 'make_bucket') as mock_make:
                mock_exists.return_value = False
                mock_make.return_value = None
                
                result = await storage_manager.create_bucket("test-bucket")
                
                assert result is True
                mock_exists.assert_called_once_with("test-bucket")
                mock_make.assert_called_once_with("test-bucket")
    
    @pytest.mark.asyncio
    async def test_create_bucket_already_exists(self, storage_manager):
        """测试创建已存在的存储桶"""
        with patch.object(storage_manager.client, 'bucket_exists') as mock_exists:
            mock_exists.return_value = True
            
            result = await storage_manager.create_bucket("existing-bucket")
            
            assert result is True
            mock_exists.assert_called_once_with("existing-bucket")
    
    @pytest.mark.asyncio
    async def test_create_bucket_failure(self, storage_manager):
        """测试创建存储桶失败"""
        with patch.object(storage_manager.client, 'bucket_exists') as mock_exists:
            with patch.object(storage_manager.client, 'make_bucket') as mock_make:
                mock_exists.return_value = False
                mock_make.side_effect = Exception("创建失败")
                
                with pytest.raises(StorageError, match="创建存储桶失败"):
                    await storage_manager.create_bucket("test-bucket")
    
    @pytest.mark.asyncio
    async def test_upload_file_success(self, storage_manager):
        """测试文件上传成功"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            
            try:
                with patch.object(storage_manager.client, 'fput_object') as mock_upload:
                    mock_upload.return_value = MagicMock()
                    
                    result = await storage_manager.upload_file(
                        bucket_name="test-bucket",
                        object_name="test.txt",
                        file_path=tmp_file.name
                    )
                    
                    assert result is True
                    mock_upload.assert_called_once_with(
                        "test-bucket", 
                        "test.txt", 
                        tmp_file.name,
                        content_type="text/plain"
                    )
            finally:
                os.unlink(tmp_file.name)
    
    @pytest.mark.asyncio
    async def test_upload_file_failure(self, storage_manager):
        """测试文件上传失败"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            
            try:
                with patch.object(storage_manager.client, 'fput_object') as mock_upload:
                    mock_upload.side_effect = Exception("上传失败")
                    
                    with pytest.raises(StorageError, match="上传文件失败"):
                        await storage_manager.upload_file(
                            bucket_name="test-bucket",
                            object_name="test.txt",
                            file_path=tmp_file.name
                        )
            finally:
                os.unlink(tmp_file.name)
    
    @pytest.mark.asyncio
    async def test_download_file_success(self, storage_manager):
        """测试文件下载成功"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                with patch.object(storage_manager.client, 'fget_object') as mock_download:
                    mock_download.return_value = None
                    
                    result = await storage_manager.download_file(
                        bucket_name="test-bucket",
                        object_name="test.txt",
                        file_path=tmp_file.name
                    )
                    
                    assert result is True
                    mock_download.assert_called_once_with(
                        "test-bucket",
                        "test.txt",
                        tmp_file.name
                    )
            finally:
                os.unlink(tmp_file.name)
    
    @pytest.mark.asyncio
    async def test_download_file_failure(self, storage_manager):
        """测试文件下载失败"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                with patch.object(storage_manager.client, 'fget_object') as mock_download:
                    mock_download.side_effect = Exception("下载失败")
                    
                    with pytest.raises(StorageError, match="下载文件失败"):
                        await storage_manager.download_file(
                            bucket_name="test-bucket",
                            object_name="test.txt",
                            file_path=tmp_file.name
                        )
            finally:
                os.unlink(tmp_file.name)
    
    @pytest.mark.asyncio
    async def test_delete_file_success(self, storage_manager):
        """测试文件删除成功"""
        with patch.object(storage_manager.client, 'remove_object') as mock_delete:
            mock_delete.return_value = None
            
            result = await storage_manager.delete_file(
                bucket_name="test-bucket",
                object_name="test.txt"
            )
            
            assert result is True
            mock_delete.assert_called_once_with("test-bucket", "test.txt")
    
    @pytest.mark.asyncio
    async def test_delete_file_failure(self, storage_manager):
        """测试文件删除失败"""
        with patch.object(storage_manager.client, 'remove_object') as mock_delete:
            mock_delete.side_effect = Exception("删除失败")
            
            with pytest.raises(StorageError, match="删除文件失败"):
                await storage_manager.delete_file(
                    bucket_name="test-bucket",
                    object_name="test.txt"
                )
    
    @pytest.mark.asyncio
    async def test_list_files_success(self, storage_manager):
        """测试列出文件成功"""
        mock_objects = [
            MagicMock(object_name="file1.txt", size=100),
            MagicMock(object_name="file2.txt", size=200)
        ]
        
        with patch.object(storage_manager.client, 'list_objects') as mock_list:
            mock_list.return_value = mock_objects
            
            files = await storage_manager.list_files(
                bucket_name="test-bucket",
                prefix="test/"
            )
            
            assert len(files) == 2
            assert files[0]["name"] == "file1.txt"
            assert files[0]["size"] == 100
            assert files[1]["name"] == "file2.txt"
            assert files[1]["size"] == 200
            
            mock_list.assert_called_once_with(
                "test-bucket",
                prefix="test/",
                recursive=True
            )
    
    @pytest.mark.asyncio
    async def test_list_files_failure(self, storage_manager):
        """测试列出文件失败"""
        with patch.object(storage_manager.client, 'list_objects') as mock_list:
            mock_list.side_effect = Exception("列出失败")
            
            with pytest.raises(StorageError, match="列出文件失败"):
                await storage_manager.list_files("test-bucket")
    
    @pytest.mark.asyncio
    async def test_get_file_url_success(self, storage_manager):
        """测试获取文件URL成功"""
        with patch.object(storage_manager.client, 'presigned_get_object') as mock_url:
            mock_url.return_value = "https://localhost:9000/test-bucket/test.txt?signature=xxx"
            
            url = await storage_manager.get_file_url(
                bucket_name="test-bucket",
                object_name="test.txt",
                expires=3600
            )
            
            assert url.startswith("https://localhost:9000")
            mock_url.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_file_url_failure(self, storage_manager):
        """测试获取文件URL失败"""
        with patch.object(storage_manager.client, 'presigned_get_object') as mock_url:
            mock_url.side_effect = Exception("获取URL失败")
            
            with pytest.raises(StorageError, match="获取文件URL失败"):
                await storage_manager.get_file_url(
                    bucket_name="test-bucket",
                    object_name="test.txt"
                )


class TestStorageManagerSingleton:
    """存储管理器单例测试"""
    
    @pytest.mark.asyncio
    async def test_get_storage_manager_singleton(self):
        """测试存储管理器单例模式"""
        with patch('app.services.storage.settings') as mock_settings:
            mock_settings.MINIO_ENDPOINT = "localhost:9000"
            mock_settings.MINIO_ACCESS_KEY = "test"
            mock_settings.MINIO_SECRET_KEY = "test"
            mock_settings.MINIO_SECURE = False
            
            manager1 = await get_storage_manager()
            manager2 = await get_storage_manager()
            
            assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_storage_manager_with_different_configs(self):
        """测试不同配置的存储管理器"""
        manager1 = StorageManager(
            endpoint="server1:9000",
            access_key="key1",
            secret_key="secret1"
        )
        
        manager2 = StorageManager(
            endpoint="server2:9000",
            access_key="key2",
            secret_key="secret2"
        )
        
        assert manager1.endpoint != manager2.endpoint
        assert manager1.access_key != manager2.access_key
        assert manager1.client is not manager2.client
    
    @pytest.mark.asyncio
    async def test_storage_operations_integration(self):
        """测试存储操作集成"""
        storage_manager = StorageManager(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
            secure=False
        )
        
        with patch.object(storage_manager.client, 'bucket_exists') as mock_exists:
            with patch.object(storage_manager.client, 'make_bucket') as mock_make:
                with patch.object(storage_manager.client, 'fput_object') as mock_upload:
                    with patch.object(storage_manager.client, 'list_objects') as mock_list:
                        # 设置模拟返回值
                        mock_exists.return_value = False
                        mock_make.return_value = None
                        mock_upload.return_value = MagicMock()
                        mock_list.return_value = [
                            MagicMock(object_name="test.txt", size=100)
                        ]
                        
                        # 执行操作序列
                        bucket_created = await storage_manager.create_bucket("test-bucket")
                        assert bucket_created is True
                        
                        # 创建临时文件用于上传
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                            tmp_file.write(b"test content")
                            tmp_file.flush()
                            
                            try:
                                file_uploaded = await storage_manager.upload_file(
                                    "test-bucket", "test.txt", tmp_file.name
                                )
                                assert file_uploaded is True
                                
                                files = await storage_manager.list_files("test-bucket")
                                assert len(files) == 1
                                assert files[0]["name"] == "test.txt"
                                
                            finally:
                                os.unlink(tmp_file.name)