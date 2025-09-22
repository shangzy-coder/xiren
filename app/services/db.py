"""
数据库服务模块
提供PostgreSQL连接、pgvector声纹存储和查询功能
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import json

import asyncpg
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseService:
    """数据库服务类"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.engine = None
        self.async_session = None
        self._initialized = False
    
    async def initialize(self):
        """初始化数据库连接"""
        if self._initialized:
            return True
        
        try:
            # 创建异步连接池
            self.pool = await asyncpg.create_pool(
                settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://"),
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # 创建SQLAlchemy异步引擎
            self.engine = create_async_engine(
                settings.DATABASE_URL,
                echo=settings.DEBUG,
                pool_pre_ping=True
            )
            
            # 创建异步会话工厂
            self.async_session = sessionmaker(
                self.engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            # 确保pgvector扩展和表存在
            await self._setup_database()
            
            self._initialized = True
            logger.info("数据库服务初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"数据库服务初始化失败: {e}")
            return False
    
    async def _setup_database(self):
        """通过读取init_db.sql脚本设置数据库表和扩展"""
        import os
        from pathlib import Path
        
        # 获取init_db.sql文件路径
        current_dir = Path(__file__).parent.parent.parent  # 项目根目录
        init_sql_path = current_dir / "scripts" / "init_db.sql"
        
        if not init_sql_path.exists():
            logger.error(f"找不到数据库初始化脚本: {init_sql_path}")
            raise FileNotFoundError(f"数据库初始化脚本不存在: {init_sql_path}")
        
        try:
            # 读取SQL脚本内容
            with open(init_sql_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            async with self.pool.acquire() as conn:
                # 执行初始化脚本
                await conn.execute(sql_content)
                logger.info(f"数据库初始化脚本执行完成: {init_sql_path.name}")
                
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    async def insert_speaker(self, 
                           name: str, 
                           embedding: np.ndarray, 
                           metadata: Dict[str, Any] = None,
                           speaker_id: str = None) -> Optional[int]:
        """插入新的说话人记录"""
        if not self._initialized:
            await self.initialize()
        
        # 如果没有提供speaker_id，则生成一个
        if speaker_id is None:
            import uuid
            speaker_id = f"speaker_{uuid.uuid4().hex[:8]}"
        
        try:
            # 将numpy数组转换为pgvector格式字符串
            if embedding is not None:
                embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                # pgvector需要向量格式为字符串 "[1,2,3,...]"
                embedding_str = str(embedding_list)
            else:
                embedding_str = None
            metadata = metadata or {}
            
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    INSERT INTO speakers (speaker_id, speaker_name, embedding, metadata) 
                    VALUES ($1, $2, $3, $4) 
                    RETURNING id
                """, speaker_id, name, embedding_str, json.dumps(metadata))
                
                speaker_id = result['id']
                logger.info(f"插入说话人记录: {name} (ID: {speaker_id})")
                return speaker_id
                
        except asyncpg.UniqueViolationError:
            logger.warning(f"说话人 {name} 已存在")
            return await self.update_speaker(name, embedding, metadata)
        except Exception as e:
            logger.error(f"插入说话人失败: {e}")
            return None
    
    async def update_speaker(self, 
                           name: str, 
                           embedding: np.ndarray = None, 
                           metadata: Dict[str, Any] = None) -> Optional[int]:
        """更新说话人记录"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                # 构建更新语句
                updates = ["updated_at = CURRENT_TIMESTAMP"]
                values = [name]
                param_count = 2
                
                if embedding is not None:
                    embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                    # pgvector需要向量格式为字符串 "[1,2,3,...]"
                    embedding_str = str(embedding_list)
                    updates.append(f"embedding = ${param_count}")
                    values.append(embedding_str)
                    param_count += 1
                
                if metadata is not None:
                    updates.append(f"metadata = ${param_count}")
                    values.append(json.dumps(metadata))
                    param_count += 1
                
                query = f"""
                    UPDATE speakers 
                    SET {', '.join(updates)}
                    WHERE speaker_name = $1 
                    RETURNING id
                """
                
                result = await conn.fetchrow(query, *values)
                if result:
                    speaker_id = result['id']
                    logger.info(f"更新说话人记录: {name} (ID: {speaker_id})")
                    return speaker_id
                else:
                    logger.warning(f"说话人 {name} 不存在")
                    return None
                    
        except Exception as e:
            logger.error(f"更新说话人失败: {e}")
            return None
    
    async def search_speakers_by_embedding(self, 
                                         embedding: np.ndarray, 
                                         threshold: float = 0.7, 
                                         limit: int = 5) -> List[Dict[str, Any]]:
        """通过声纹特征搜索相似的说话人"""
        if not self._initialized:
            await self.initialize()
        
        try:
            embedding_list = embedding.tolist()
            
            async with self.pool.acquire() as conn:
                # 使用余弦相似度搜索
                results = await conn.fetch("""
                    SELECT 
                        id, speaker_name as name, metadata, 
                        1 - (embedding <=> $1::vector) as similarity,
                        created_at, updated_at
                    FROM speakers 
                    WHERE embedding IS NOT NULL
                        AND (1 - (embedding <=> $1::vector)) >= $2
                    ORDER BY similarity DESC 
                    LIMIT $3
                """, embedding_list, threshold, limit)
                
                speakers = []
                for row in results:
                    speakers.append({
                        'id': row['id'],
                        'name': row['name'],
                        'similarity': float(row['similarity']),
                        'metadata': row['metadata'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    })
                
                logger.info(f"找到 {len(speakers)} 个相似说话人")
                return speakers
                
        except Exception as e:
            logger.error(f"声纹搜索失败: {e}")
            return []
    
    async def get_speaker_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """根据姓名获取说话人信息"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT id, speaker_name as name, embedding, metadata, created_at, updated_at 
                    FROM speakers 
                    WHERE speaker_name = $1
                """, name)
                
                if result:
                    return {
                        'id': result['id'],
                        'name': result['name'],
                        'embedding': np.array(result['embedding']) if result['embedding'] else None,
                        'metadata': result['metadata'],
                        'created_at': result['created_at'],
                        'updated_at': result['updated_at']
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"获取说话人失败: {e}")
            return None
    
    async def get_all_speakers(self, 
                             limit: int = 100, 
                             offset: int = 0) -> List[Dict[str, Any]]:
        """获取所有说话人列表"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT id, speaker_name as name, metadata, created_at, updated_at
                    FROM speakers 
                    ORDER BY created_at DESC 
                    LIMIT $1 OFFSET $2
                """, limit, offset)
                
                speakers = []
                for row in results:
                    speakers.append({
                        'id': row['id'],
                        'name': row['name'],
                        'metadata': row['metadata'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    })
                
                return speakers
                
        except Exception as e:
            logger.error(f"获取说话人列表失败: {e}")
            return []
    
    async def delete_speaker(self, name: str) -> bool:
        """删除说话人记录"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                result =                 await conn.execute("""
                    DELETE FROM speakers WHERE speaker_name = $1
                """, name)
                
                if result == "DELETE 1":
                    logger.info(f"删除说话人: {name}")
                    return True
                else:
                    logger.warning(f"说话人 {name} 不存在")
                    return False
                    
        except Exception as e:
            logger.error(f"删除说话人失败: {e}")
            return False
    
    async def insert_audio_file(self, 
                              filename: str,
                              file_path: str = None,
                              file_size: int = None,
                              duration: float = None,
                              sample_rate: int = None,
                              channels: int = None,
                              format: str = None,
                              speaker_id: int = None,
                              metadata: Dict[str, Any] = None,
                              # MinIO相关字段
                              bucket: str = None,
                              object_name: str = None,
                              sha256_hash: str = None,
                              content_type: str = None,
                              minio_version_id: str = None,
                              storage_class: str = "STANDARD",
                              file_tags: Dict[str, Any] = None,
                              upload_time: datetime = None,
                              is_duplicated: bool = False,
                              original_filename: str = None) -> Optional[int]:
        """插入音频文件记录（支持MinIO字段）"""
        if not self._initialized:
            await self.initialize()
        
        try:
            metadata = metadata or {}
            file_tags = file_tags or {}
            
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    INSERT INTO audio_files 
                    (filename, file_path, file_size, duration, sample_rate, 
                     channels, format, speaker_id, metadata,
                     bucket, object_name, sha256_hash, content_type, 
                     minio_version_id, storage_class, file_tags, 
                     upload_time, is_duplicated, original_filename) 
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9,
                            $10, $11, $12, $13, $14, $15, $16, $17, $18, $19) 
                    RETURNING id
                """, filename, file_path, file_size, duration, sample_rate, 
                    channels, format, speaker_id, json.dumps(metadata),
                    bucket, object_name, sha256_hash, content_type,
                    minio_version_id, storage_class, json.dumps(file_tags),
                    upload_time, is_duplicated, original_filename)
                
                file_id = result['id']
                logger.info(f"插入音频文件记录: {filename} (ID: {file_id}) 到存储桶: {bucket}")
                return file_id
                
        except Exception as e:
            logger.error(f"插入音频文件失败: {e}")
            return None
    
    async def insert_recognition_session(self,
                                       session_id: str,
                                       recognition_type: str,
                                       audio_file_id: int = None,
                                       status: str = "pending") -> Optional[int]:
        """插入识别会话记录"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    INSERT INTO recognition_sessions 
                    (session_id, recognition_type, audio_file_id, status) 
                    VALUES ($1, $2, $3, $4) 
                    RETURNING id
                """, session_id, recognition_type, audio_file_id, status)
                
                return result['id']
                
        except Exception as e:
            logger.error(f"插入识别会话失败: {e}")
            return None
    
    async def update_recognition_session(self,
                                       session_id: str,
                                       results: Dict[str, Any] = None,
                                       status: str = None,
                                       processing_time: float = None) -> bool:
        """更新识别会话结果"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                # 构建更新语句
                updates = ["updated_at = CURRENT_TIMESTAMP"]
                values = [session_id]
                param_count = 2
                
                if results is not None:
                    updates.append(f"results = ${param_count}")
                    values.append(json.dumps(results))
                    param_count += 1
                
                if status is not None:
                    updates.append(f"status = ${param_count}")
                    values.append(status)
                    param_count += 1
                    
                    if status in ['completed', 'failed']:
                        updates.append("completed_at = CURRENT_TIMESTAMP")
                
                if processing_time is not None:
                    updates.append(f"processing_time = ${param_count}")
                    values.append(processing_time)
                    param_count += 1
                
                query = f"""
                    UPDATE recognition_sessions 
                    SET {', '.join(updates)}
                    WHERE session_id = $1
                """
                
                result = await conn.execute(query, *values)
                return result == "UPDATE 1"
                
        except Exception as e:
            logger.error(f"更新识别会话失败: {e}")
            return False
    
    async def get_speaker_stats(self) -> Dict[str, Any]:
        """获取说话人统计信息"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                # 总说话人数
                total_speakers = await conn.fetchval(
                    "SELECT COUNT(*) FROM speakers"
                )
                
                # 有声纹特征的说话人数
                speakers_with_embedding = await conn.fetchval(
                    "SELECT COUNT(*) FROM speakers WHERE embedding IS NOT NULL"
                )
                
                # 音频文件总数
                total_audio_files = await conn.fetchval(
                    "SELECT COUNT(*) FROM audio_files"
                )
                
                # 识别会话统计
                session_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
                        COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_sessions,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_sessions
                    FROM recognition_sessions
                """)
                
                return {
                    'total_speakers': total_speakers,
                    'speakers_with_embedding': speakers_with_embedding,
                    'total_audio_files': total_audio_files,
                    'recognition_sessions': {
                        'total': session_stats['total_sessions'],
                        'completed': session_stats['completed_sessions'],
                        'pending': session_stats['pending_sessions'],
                        'failed': session_stats['failed_sessions']
                    }
                }
                
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    # ==================== MinIO 和文件版本管理方法 ====================
    
    async def insert_file_version(self,
                                 audio_file_id: int,
                                 bucket: str,
                                 object_name: str,
                                 file_size: int = None,
                                 sha256_hash: str = None,
                                 minio_version_id: str = None,
                                 is_current: bool = True,
                                 metadata: Dict[str, Any] = None) -> Optional[int]:
        """插入文件版本记录"""
        if not self._initialized:
            await self.initialize()
        
        try:
            metadata = metadata or {}
            
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    INSERT INTO file_versions 
                    (audio_file_id, bucket, object_name, file_size, 
                     sha256_hash, minio_version_id, is_current, metadata) 
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8) 
                    RETURNING id, version_number
                """, audio_file_id, bucket, object_name, file_size, 
                    sha256_hash, minio_version_id, is_current, json.dumps(metadata))
                
                version_id = result['id']
                version_number = result['version_number']
                logger.info(f"插入文件版本记录: 文件ID {audio_file_id}, 版本 {version_number} (ID: {version_id})")
                return version_id
                
        except Exception as e:
            logger.error(f"插入文件版本失败: {e}")
            return None
    
    async def get_audio_file_by_hash(self, sha256_hash: str) -> Optional[Dict[str, Any]]:
        """根据哈希值查找音频文件（用于去重）"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT id, filename, bucket, object_name, sha256_hash, 
                           content_type, file_size, upload_time, is_duplicated, 
                           original_filename, metadata
                    FROM audio_files 
                    WHERE sha256_hash = $1
                    LIMIT 1
                """, sha256_hash)
                
                if result:
                    return {
                        'id': result['id'],
                        'filename': result['filename'],
                        'bucket': result['bucket'],
                        'object_name': result['object_name'],
                        'sha256_hash': result['sha256_hash'],
                        'content_type': result['content_type'],
                        'file_size': result['file_size'],
                        'upload_time': result['upload_time'],
                        'is_duplicated': result['is_duplicated'],
                        'original_filename': result['original_filename'],
                        'metadata': result['metadata']
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"根据哈希查找文件失败: {e}")
            return None
    
    async def get_audio_file_by_object_name(self, bucket: str, object_name: str) -> Optional[Dict[str, Any]]:
        """根据存储桶和对象名查找音频文件"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT id, filename, bucket, object_name, sha256_hash, 
                           content_type, file_size, upload_time, is_duplicated, 
                           original_filename, metadata, created_at
                    FROM audio_files 
                    WHERE bucket = $1 AND object_name = $2
                    LIMIT 1
                """, bucket, object_name)
                
                if result:
                    return {
                        'id': result['id'],
                        'filename': result['filename'],
                        'bucket': result['bucket'],
                        'object_name': result['object_name'],
                        'sha256_hash': result['sha256_hash'],
                        'content_type': result['content_type'],
                        'file_size': result['file_size'],
                        'upload_time': result['upload_time'],
                        'is_duplicated': result['is_duplicated'],
                        'original_filename': result['original_filename'],
                        'metadata': result['metadata'],
                        'created_at': result['created_at']
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"根据对象名查找文件失败: {e}")
            return None
    
    async def get_file_versions(self, audio_file_id: int) -> List[Dict[str, Any]]:
        """获取文件的所有版本"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT id, version_number, bucket, object_name, file_size,
                           sha256_hash, minio_version_id, is_current, 
                           created_at, metadata
                    FROM file_versions 
                    WHERE audio_file_id = $1
                    ORDER BY version_number DESC
                """, audio_file_id)
                
                versions = []
                for row in results:
                    versions.append({
                        'id': row['id'],
                        'version_number': row['version_number'],
                        'bucket': row['bucket'],
                        'object_name': row['object_name'],
                        'file_size': row['file_size'],
                        'sha256_hash': row['sha256_hash'],
                        'minio_version_id': row['minio_version_id'],
                        'is_current': row['is_current'],
                        'created_at': row['created_at'],
                        'metadata': row['metadata']
                    })
                
                return versions
                
        except Exception as e:
            logger.error(f"获取文件版本失败: {e}")
            return []
    
    async def get_current_file_version(self, audio_file_id: int) -> Optional[Dict[str, Any]]:
        """获取文件的当前版本"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT id, version_number, bucket, object_name, file_size,
                           sha256_hash, minio_version_id, is_current, 
                           created_at, metadata
                    FROM file_versions 
                    WHERE audio_file_id = $1 AND is_current = true
                    LIMIT 1
                """, audio_file_id)
                
                if result:
                    return {
                        'id': result['id'],
                        'version_number': result['version_number'],
                        'bucket': result['bucket'],
                        'object_name': result['object_name'],
                        'file_size': result['file_size'],
                        'sha256_hash': result['sha256_hash'],
                        'minio_version_id': result['minio_version_id'],
                        'is_current': result['is_current'],
                        'created_at': result['created_at'],
                        'metadata': result['metadata']
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"获取当前文件版本失败: {e}")
            return None
    
    async def update_audio_file_minio_info(self,
                                          file_id: int,
                                          bucket: str = None,
                                          object_name: str = None,
                                          sha256_hash: str = None,
                                          content_type: str = None,
                                          minio_version_id: str = None,
                                          storage_class: str = None,
                                          file_tags: Dict[str, Any] = None,
                                          upload_time: datetime = None,
                                          is_duplicated: bool = None) -> bool:
        """更新音频文件的MinIO信息"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                # 构建更新语句
                updates = []
                values = [file_id]
                param_count = 2
                
                if bucket is not None:
                    updates.append(f"bucket = ${param_count}")
                    values.append(bucket)
                    param_count += 1
                
                if object_name is not None:
                    updates.append(f"object_name = ${param_count}")
                    values.append(object_name)
                    param_count += 1
                
                if sha256_hash is not None:
                    updates.append(f"sha256_hash = ${param_count}")
                    values.append(sha256_hash)
                    param_count += 1
                
                if content_type is not None:
                    updates.append(f"content_type = ${param_count}")
                    values.append(content_type)
                    param_count += 1
                
                if minio_version_id is not None:
                    updates.append(f"minio_version_id = ${param_count}")
                    values.append(minio_version_id)
                    param_count += 1
                
                if storage_class is not None:
                    updates.append(f"storage_class = ${param_count}")
                    values.append(storage_class)
                    param_count += 1
                
                if file_tags is not None:
                    updates.append(f"file_tags = ${param_count}")
                    values.append(json.dumps(file_tags))
                    param_count += 1
                
                if upload_time is not None:
                    updates.append(f"upload_time = ${param_count}")
                    values.append(upload_time)
                    param_count += 1
                
                if is_duplicated is not None:
                    updates.append(f"is_duplicated = ${param_count}")
                    values.append(is_duplicated)
                    param_count += 1
                
                if not updates:
                    logger.warning("没有提供要更新的字段")
                    return False
                
                query = f"""
                    UPDATE audio_files 
                    SET {', '.join(updates)}
                    WHERE id = $1
                """
                
                result = await conn.execute(query, *values)
                success = result == "UPDATE 1"
                
                if success:
                    logger.info(f"更新音频文件MinIO信息: ID {file_id}")
                else:
                    logger.warning(f"音频文件 ID {file_id} 不存在或未更新")
                
                return success
                
        except Exception as e:
            logger.error(f"更新音频文件MinIO信息失败: {e}")
            return False
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                # 总文件统计
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_files,
                        SUM(file_size) as total_size,
                        COUNT(CASE WHEN is_duplicated = true THEN 1 END) as duplicated_files,
                        COUNT(CASE WHEN bucket IS NOT NULL THEN 1 END) as minio_files
                    FROM audio_files
                """)
                
                # 按存储桶统计
                bucket_stats = await conn.fetch("""
                    SELECT 
                        bucket,
                        COUNT(*) as file_count,
                        SUM(file_size) as total_size,
                        COUNT(CASE WHEN is_duplicated = true THEN 1 END) as duplicated_count
                    FROM audio_files 
                    WHERE bucket IS NOT NULL
                    GROUP BY bucket
                    ORDER BY file_count DESC
                """)
                
                # 按内容类型统计
                content_type_stats = await conn.fetch("""
                    SELECT 
                        content_type,
                        COUNT(*) as file_count,
                        SUM(file_size) as total_size
                    FROM audio_files 
                    WHERE content_type IS NOT NULL
                    GROUP BY content_type
                    ORDER BY file_count DESC
                """)
                
                # 版本统计
                version_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_versions,
                        COUNT(DISTINCT audio_file_id) as files_with_versions,
                        COUNT(CASE WHEN is_current = true THEN 1 END) as current_versions
                    FROM file_versions
                """)
                
                return {
                    'total_statistics': {
                        'total_files': stats['total_files'],
                        'total_size': stats['total_size'] or 0,
                        'duplicated_files': stats['duplicated_files'],
                        'minio_files': stats['minio_files']
                    },
                    'bucket_statistics': [
                        {
                            'bucket': row['bucket'],
                            'file_count': row['file_count'],
                            'total_size': row['total_size'] or 0,
                            'duplicated_count': row['duplicated_count']
                        }
                        for row in bucket_stats
                    ],
                    'content_type_statistics': [
                        {
                            'content_type': row['content_type'],
                            'file_count': row['file_count'],
                            'total_size': row['total_size'] or 0
                        }
                        for row in content_type_stats
                    ],
                    'version_statistics': {
                        'total_versions': version_stats['total_versions'] if version_stats else 0,
                        'files_with_versions': version_stats['files_with_versions'] if version_stats else 0,
                        'current_versions': version_stats['current_versions'] if version_stats else 0
                    }
                }
                
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {}

    async def close(self):
        """关闭数据库连接"""
        if self.pool:
            await self.pool.close()
        if self.engine:
            await self.engine.dispose()
        self._initialized = False
        logger.info("数据库连接已关闭")


# 全局数据库服务实例
db_service: Optional[DatabaseService] = None

async def get_database_service() -> DatabaseService:
    """获取全局数据库服务实例"""
    global db_service
    if db_service is None:
        db_service = DatabaseService()
        await db_service.initialize()
    return db_service

async def initialize_database() -> DatabaseService:
    """初始化全局数据库服务"""
    return await get_database_service()
