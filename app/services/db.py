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
        """设置数据库表和扩展"""
        async with self.pool.acquire() as conn:
            try:
                # 启用pgvector扩展
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("pgvector扩展已启用")
                
                # 创建speakers表
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS speakers (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL UNIQUE,
                        embedding VECTOR(512),  -- 假设512维特征向量
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # 创建索引
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_speakers_name ON speakers(name);
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_speakers_embedding ON speakers 
                    USING ivfflat (embedding vector_cosine_ops);
                """)
                
                # 创建audio_files表
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS audio_files (
                        id SERIAL PRIMARY KEY,
                        filename VARCHAR(500) NOT NULL,
                        file_path VARCHAR(1000),
                        file_size BIGINT,
                        duration FLOAT,
                        sample_rate INTEGER,
                        channels INTEGER,
                        format VARCHAR(50),
                        speaker_id INTEGER REFERENCES speakers(id),
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # 创建recognition_sessions表
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS recognition_sessions (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(100) NOT NULL,
                        status VARCHAR(50) DEFAULT 'pending',
                        audio_file_id INTEGER REFERENCES audio_files(id),
                        results JSONB DEFAULT '{}',
                        recognition_type VARCHAR(50), -- 'asr', 'speaker_id', 'diarization'
                        processing_time FLOAT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP WITH TIME ZONE
                    );
                """)
                
                logger.info("数据库表结构设置完成")
                
            except Exception as e:
                logger.error(f"数据库设置失败: {e}")
                raise
    
    async def insert_speaker(self, 
                           name: str, 
                           embedding: np.ndarray, 
                           metadata: Dict[str, Any] = None) -> Optional[int]:
        """插入新的说话人记录"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # 将numpy数组转换为列表
            embedding_list = embedding.tolist() if embedding is not None else None
            metadata = metadata or {}
            
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    INSERT INTO speakers (name, embedding, metadata) 
                    VALUES ($1, $2, $3) 
                    RETURNING id
                """, name, embedding_list, json.dumps(metadata))
                
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
                    embedding_list = embedding.tolist()
                    updates.append(f"embedding = ${param_count}")
                    values.append(embedding_list)
                    param_count += 1
                
                if metadata is not None:
                    updates.append(f"metadata = ${param_count}")
                    values.append(json.dumps(metadata))
                    param_count += 1
                
                query = f"""
                    UPDATE speakers 
                    SET {', '.join(updates)}
                    WHERE name = $1 
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
                        id, name, metadata, 
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
                    SELECT id, name, embedding, metadata, created_at, updated_at 
                    FROM speakers 
                    WHERE name = $1
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
                    SELECT id, name, metadata, created_at, updated_at
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
                result = await conn.execute("""
                    DELETE FROM speakers WHERE name = $1
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
                              metadata: Dict[str, Any] = None) -> Optional[int]:
        """插入音频文件记录"""
        if not self._initialized:
            await self.initialize()
        
        try:
            metadata = metadata or {}
            
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    INSERT INTO audio_files 
                    (filename, file_path, file_size, duration, sample_rate, 
                     channels, format, speaker_id, metadata) 
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) 
                    RETURNING id
                """, filename, file_path, file_size, duration, sample_rate, 
                    channels, format, speaker_id, json.dumps(metadata))
                
                file_id = result['id']
                logger.info(f"插入音频文件记录: {filename} (ID: {file_id})")
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
