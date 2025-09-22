"""
MinIO 对象存储服务模块
提供音频文件的存储、检索、版本管理和去重功能
"""

import os
import logging
import hashlib
import asyncio
from typing import List, Optional, Dict, Any, Tuple, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json

from minio import Minio
from minio.error import S3Error
try:
    from minio.commonconfig import CopySource
    from minio.lifecycle import LifecycleConfig, Rule, Expiration
except ImportError:
    # 处理旧版本的MinIO
    CopySource = None
    LifecycleConfig = None
    Rule = None
    Expiration = None
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """文件元数据"""
    filename: str
    file_path: str
    file_size: int
    content_type: str
    sha256_hash: str
    upload_time: datetime
    bucket: str
    version_id: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class StorageStats:
    """存储统计信息"""
    total_files: int
    total_size: int
    buckets: Dict[str, Dict[str, Any]]
    deduplication_savings: int = 0


class MinIOClient:
    """MinIO 客户端封装类"""
    
    def __init__(self):
        self.client: Optional[Minio] = None
        self._initialized = False
        self._connection_pool_size = 10
        
    async def initialize(self) -> bool:
        """初始化 MinIO 客户端"""
        if self._initialized:
            return True
            
        try:
            # 创建 MinIO 客户端
            self.client = Minio(
                endpoint=settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE
            )
            
            # 测试连接
            await self._test_connection()
            
            # 创建必要的存储桶
            await self._ensure_buckets()
            
            self._initialized = True
            logger.info("MinIO 客户端初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"MinIO 客户端初始化失败: {e}")
            return False
    
    async def _test_connection(self):
        """测试 MinIO 连接"""
        try:
            # 使用异步方式检查服务器状态
            async with httpx.AsyncClient() as client:
                url = f"{'https' if settings.MINIO_SECURE else 'http'}://{settings.MINIO_ENDPOINT}/minio/health/live"
                response = await client.get(url, timeout=5.0)
                if response.status_code != 200:
                    raise Exception(f"MinIO 健康检查失败: {response.status_code}")
        except Exception as e:
            logger.warning(f"MinIO 健康检查失败: {e}")
    
    async def _ensure_buckets(self):
        """确保必要的存储桶存在"""
        buckets = [
            settings.MINIO_BUCKET,
            settings.MINIO_TEMP_BUCKET,
            settings.MINIO_BACKUP_BUCKET
        ]
        
        for bucket_name in buckets:
            try:
                if not self.client.bucket_exists(bucket_name):
                    self.client.make_bucket(bucket_name)
                    logger.info(f"创建存储桶: {bucket_name}")
                    
                    # 为临时文件桶设置生命周期策略
                    if bucket_name == settings.MINIO_TEMP_BUCKET:
                        await self._set_temp_bucket_lifecycle(bucket_name)
                        
            except S3Error as e:
                logger.error(f"创建存储桶失败 {bucket_name}: {e}")
                raise
    
    async def _set_temp_bucket_lifecycle(self, bucket_name: str):
        """为临时文件桶设置生命周期策略"""
        try:
            if LifecycleConfig and Rule and Expiration:
                # 设置自动清理规则
                lifecycle_config = LifecycleConfig([
                    Rule(
                        rule_id="temp_file_cleanup",
                        rule_filter=None,
                        rule_status="Enabled",
                        expiration=Expiration(days=settings.TEMP_FILE_CLEANUP_HOURS // 24 or 1)
                    )
                ])
                self.client.set_bucket_lifecycle(bucket_name, lifecycle_config)
                logger.info(f"为存储桶 {bucket_name} 设置生命周期策略")
            else:
                logger.warning(f"MinIO lifecycle功能不可用，跳过为存储桶 {bucket_name} 设置生命周期策略")
        except Exception as e:
            logger.warning(f"设置生命周期策略失败: {e}")


class StorageService:
    """MinIO 存储服务类"""
    
    def __init__(self):
        self.minio_client = MinIOClient()
        self._initialized = False
        self._file_hash_cache: Dict[str, str] = {}
        
    async def initialize(self) -> bool:
        """初始化存储服务"""
        if self._initialized:
            return True
            
        success = await self.minio_client.initialize()
        if success:
            self._initialized = True
            logger.info("存储服务初始化成功")
        return success
    
    # ==================== 文件上传功能 ====================
    
    async def upload_audio_file(self, 
                               file_path: str,
                               filename: str = None,
                               bucket: str = None,
                               metadata: Dict[str, str] = None,
                               tags: Dict[str, str] = None) -> Optional[FileMetadata]:
        """上传音频文件到 MinIO"""
        if not self._initialized:
            await self.initialize()
            
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 文件大小检查
            file_size = file_path_obj.stat().st_size
            if file_size > settings.MAX_FILE_SIZE:
                raise ValueError(f"文件大小超出限制: {file_size} > {settings.MAX_FILE_SIZE}")
            
            # 文件格式验证
            filename = filename or file_path_obj.name
            if not self._is_supported_format(filename):
                raise ValueError(f"不支持的文件格式: {Path(filename).suffix}")
            
            # 生成唯一文件名避免冲突
            filename = await self._generate_unique_filename(filename, bucket or settings.MINIO_BUCKET)
            
            # 计算文件哈希
            file_hash = await self._calculate_file_hash(file_path)
            
            # 检查去重
            if settings.ENABLE_FILE_DEDUPLICATION:
                existing_file = await self._check_duplicate_file(file_hash)
                if existing_file:
                    logger.info(f"发现重复文件，返回现有文件信息: {existing_file.filename}")
                    return existing_file
            
            # 准备上传参数
            bucket = bucket or settings.MINIO_BUCKET
            content_type = self._get_content_type(filename)
            
            # 准备元数据
            upload_metadata = {
                "upload_time": datetime.now().isoformat(),
                "sha256_hash": file_hash,
                "original_filename": filename,
                **(metadata or {})
            }
            
            # 上传文件
            result = self.minio_client.client.fput_object(
                bucket_name=bucket,
                object_name=filename,
                file_path=file_path,
                content_type=content_type,
                metadata=upload_metadata,
                tags=tags
            )
            
            file_metadata = FileMetadata(
                filename=filename,
                file_path=f"{bucket}/{filename}",
                file_size=file_size,
                content_type=content_type,
                sha256_hash=file_hash,
                upload_time=datetime.now(),
                bucket=bucket,
                version_id=result.version_id if settings.ENABLE_FILE_VERSIONING else None,
                tags=tags
            )
            
            # 缓存哈希值用于去重
            self._file_hash_cache[file_hash] = f"{bucket}/{filename}"
            
            logger.info(f"文件上传成功: {filename} -> {bucket}")
            return file_metadata
            
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            return None
    
    async def upload_batch_files(self, 
                                file_paths: List[str],
                                bucket: str = None,
                                metadata: Dict[str, str] = None) -> List[FileMetadata]:
        """批量上传文件"""
        results = []
        
        # 使用信号量限制并发数
        semaphore = asyncio.Semaphore(5)
        
        async def upload_single_file(file_path: str):
            async with semaphore:
                return await self.upload_audio_file(
                    file_path=file_path,
                    bucket=bucket,
                    metadata=metadata
                )
        
        # 并发上传
        tasks = [upload_single_file(fp) for fp in file_paths]
        upload_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in upload_results:
            if isinstance(result, FileMetadata):
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"批量上传中的文件失败: {result}")
        
        logger.info(f"批量上传完成: {len(results)}/{len(file_paths)} 个文件成功")
        return results
    
    # ==================== 文件下载和访问功能 ====================
    
    async def download_audio_file(self, 
                                 filename: str,
                                 local_path: str,
                                 bucket: str = None) -> bool:
        """下载音频文件"""
        if not self._initialized:
            await self.initialize()
            
        try:
            bucket = bucket or settings.MINIO_BUCKET
            
            self.minio_client.client.fget_object(
                bucket_name=bucket,
                object_name=filename,
                file_path=local_path
            )
            
            logger.info(f"文件下载成功: {filename} -> {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件下载失败: {e}")
            return False
    
    async def get_audio_file_url(self, 
                                filename: str,
                                bucket: str = None,
                                expires: timedelta = timedelta(hours=1)) -> Optional[str]:
        """获取文件的预签名访问URL"""
        if not self._initialized:
            await self.initialize()
            
        try:
            bucket = bucket or settings.MINIO_BUCKET
            
            url = self.minio_client.client.presigned_get_object(
                bucket_name=bucket,
                object_name=filename,
                expires=expires
            )
            
            return url
            
        except Exception as e:
            logger.error(f"获取文件URL失败: {e}")
            return None
    
    async def get_file_info(self, 
                           filename: str,
                           bucket: str = None) -> Optional[FileMetadata]:
        """获取文件信息"""
        if not self._initialized:
            await self.initialize()
            
        try:
            bucket = bucket or settings.MINIO_BUCKET
            
            stat = self.minio_client.client.stat_object(bucket, filename)
            
            return FileMetadata(
                filename=filename,
                file_path=f"{bucket}/{filename}",
                file_size=stat.size,
                content_type=stat.content_type,
                sha256_hash=stat.metadata.get("sha256_hash", ""),
                upload_time=stat.last_modified,
                bucket=bucket,
                version_id=stat.version_id,
                tags=stat.tags
            )
            
        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return None
    
    # ==================== 文件删除和清理功能 ====================
    
    async def delete_audio_file(self, 
                               filename: str,
                               bucket: str = None,
                               version_id: str = None) -> bool:
        """删除音频文件"""
        if not self._initialized:
            await self.initialize()
            
        try:
            bucket = bucket or settings.MINIO_BUCKET
            
            self.minio_client.client.remove_object(
                bucket_name=bucket,
                object_name=filename,
                version_id=version_id
            )
            
            logger.info(f"文件删除成功: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"文件删除失败: {e}")
            return False
    
    async def delete_batch_files(self, 
                                filenames: List[str],
                                bucket: str = None) -> Dict[str, bool]:
        """批量删除文件"""
        results = {}
        bucket = bucket or settings.MINIO_BUCKET
        
        for filename in filenames:
            results[filename] = await self.delete_audio_file(filename, bucket)
        
        return results
    
    async def cleanup_temp_files(self, 
                                older_than_hours: int = None) -> int:
        """清理临时文件"""
        if not self._initialized:
            await self.initialize()
            
        try:
            older_than_hours = older_than_hours or settings.TEMP_FILE_CLEANUP_HOURS
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            
            deleted_count = 0
            objects = self.minio_client.client.list_objects(
                settings.MINIO_TEMP_BUCKET,
                recursive=True
            )
            
            for obj in objects:
                if obj.last_modified < cutoff_time:
                    try:
                        self.minio_client.client.remove_object(
                            settings.MINIO_TEMP_BUCKET,
                            obj.object_name
                        )
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"删除临时文件失败 {obj.object_name}: {e}")
            
            logger.info(f"清理临时文件完成，删除 {deleted_count} 个文件")
            return deleted_count
            
        except Exception as e:
            logger.error(f"清理临时文件失败: {e}")
            return 0
    
    # ==================== 存储统计功能 ====================
    
    async def get_storage_stats(self) -> StorageStats:
        """获取存储统计信息"""
        if not self._initialized:
            await self.initialize()
            
        try:
            stats = StorageStats(
                total_files=0,
                total_size=0,
                buckets={}
            )
            
            buckets = [
                settings.MINIO_BUCKET,
                settings.MINIO_TEMP_BUCKET,
                settings.MINIO_BACKUP_BUCKET
            ]
            
            for bucket_name in buckets:
                try:
                    bucket_stats = {
                        "file_count": 0,
                        "total_size": 0,
                        "latest_upload": None
                    }
                    
                    objects = self.minio_client.client.list_objects(
                        bucket_name,
                        recursive=True
                    )
                    
                    latest_time = None
                    for obj in objects:
                        bucket_stats["file_count"] += 1
                        bucket_stats["total_size"] += obj.size
                        
                        if latest_time is None or obj.last_modified > latest_time:
                            latest_time = obj.last_modified
                    
                    bucket_stats["latest_upload"] = latest_time
                    stats.buckets[bucket_name] = bucket_stats
                    stats.total_files += bucket_stats["file_count"]
                    stats.total_size += bucket_stats["total_size"]
                    
                except Exception as e:
                    logger.error(f"获取存储桶统计失败 {bucket_name}: {e}")
                    stats.buckets[bucket_name] = {"error": str(e)}
            
            return stats
            
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return StorageStats(total_files=0, total_size=0, buckets={})
    
    # ==================== 辅助方法 ====================
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件 SHA256 哈希值"""
        sha256_hash = hashlib.sha256()
        
        # 异步读取文件计算哈希
        loop = asyncio.get_event_loop()
        
        def _hash_file():
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        
        return await loop.run_in_executor(None, _hash_file)
    
    async def _check_duplicate_file(self, file_hash: str) -> Optional[FileMetadata]:
        """检查重复文件"""
        # 首先检查缓存
        if file_hash in self._file_hash_cache:
            bucket, filename = self._file_hash_cache[file_hash].split("/", 1)
            return await self.get_file_info(filename, bucket)
        
        # 搜索所有存储桶中的文件
        buckets = [settings.MINIO_BUCKET, settings.MINIO_BACKUP_BUCKET]
        
        for bucket in buckets:
            try:
                objects = self.minio_client.client.list_objects(bucket, recursive=True)
                for obj in objects:
                    # 获取文件元数据检查哈希
                    try:
                        stat = self.minio_client.client.stat_object(bucket, obj.object_name)
                        if stat.metadata.get("sha256_hash") == file_hash:
                            # 更新缓存
                            self._file_hash_cache[file_hash] = f"{bucket}/{obj.object_name}"
                            return await self.get_file_info(obj.object_name, bucket)
                    except Exception:
                        continue
            except Exception as e:
                logger.error(f"搜索重复文件失败 {bucket}: {e}")
        
        return None
    
    def _get_content_type(self, filename: str) -> str:
        """根据文件扩展名获取内容类型"""
        ext = Path(filename).suffix.lower()
        content_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.flac': 'audio/flac',
            '.m4a': 'audio/mp4',
            '.ogg': 'audio/ogg',
            '.amr': 'audio/amr',
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.mpeg': 'video/mpeg',
            '.webm': 'video/webm'
        }
        return content_types.get(ext, 'application/octet-stream')
    
    def _is_supported_format(self, filename: str) -> bool:
        """检查文件格式是否受支持"""
        ext = Path(filename).suffix.lower().lstrip('.')
        return ext in settings.SUPPORTED_FORMATS
    
    async def _generate_unique_filename(self, filename: str, bucket: str) -> str:
        """生成唯一的文件名，避免冲突"""
        base_name = Path(filename).stem
        extension = Path(filename).suffix
        counter = 0
        
        original_filename = filename
        
        # 检查文件是否已存在
        while True:
            try:
                # 尝试获取文件信息，如果成功说明文件存在
                self.minio_client.client.stat_object(bucket, filename)
                
                # 文件存在，生成新的文件名
                counter += 1
                filename = f"{base_name}_{counter}{extension}"
                
            except S3Error as e:
                if e.code == "NoSuchKey":
                    # 文件不存在，可以使用这个文件名
                    break
                else:
                    # 其他错误，记录并返回原始文件名
                    logger.warning(f"检查文件存在性时出错: {e}")
                    return original_filename
            except Exception as e:
                logger.warning(f"生成唯一文件名时出错: {e}")
                return original_filename
        
        if counter > 0:
            logger.info(f"文件名冲突，已重命名: {original_filename} -> {filename}")
        
        return filename
    
    async def close(self):
        """关闭存储服务"""
        self._initialized = False
        self._file_hash_cache.clear()
        logger.info("存储服务已关闭")


# 全局存储服务实例
storage_service: Optional[StorageService] = None

async def get_storage_service() -> StorageService:
    """获取全局存储服务实例"""
    global storage_service
    if storage_service is None:
        storage_service = StorageService()
        await storage_service.initialize()
    return storage_service

async def initialize_storage() -> StorageService:
    """初始化全局存储服务"""
    return await get_storage_service()
