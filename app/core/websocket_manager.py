"""
WebSocket连接管理器

提供WebSocket连接的统一管理、监控和优化功能。
"""

import asyncio
import time
import json
import base64
import logging
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from app.config import settings
from app.core.model import recognize_audio, get_model_info
from app.utils.logging_config import get_logger
from app.utils.metrics import metrics_collector

logger = get_logger(__name__)


@dataclass
class ConnectionStats:
    """WebSocket连接统计信息"""
    connection_id: str
    client_address: str
    connected_at: datetime
    last_activity: datetime
    messages_sent: int = 0
    messages_received: int = 0
    audio_chunks_processed: int = 0
    recognition_requests: int = 0
    errors: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    
    def update_activity(self):
        """更新最后活动时间"""
        self.last_activity = datetime.now()
    
    def add_received_message(self, size: int = 0):
        """记录接收的消息"""
        self.messages_received += 1
        self.bytes_received += size
        self.update_activity()
    
    def add_sent_message(self, size: int = 0):
        """记录发送的消息"""
        self.messages_sent += 1
        self.bytes_sent += size
        self.update_activity()
    
    def add_audio_chunk(self):
        """记录处理的音频块"""
        self.audio_chunks_processed += 1
    
    def add_recognition_request(self):
        """记录识别请求"""
        self.recognition_requests += 1
    
    def add_error(self):
        """记录错误"""
        self.errors += 1


@dataclass
class AudioBuffer:
    """音频缓冲区管理"""
    samples: List[float] = field(default_factory=list)
    sample_rate: int = 16000
    buffer_size_seconds: float = 2.0
    overlap_ratio: float = 0.25
    min_process_seconds: float = 0.5
    
    @property
    def buffer_size_samples(self) -> int:
        """缓冲区大小（样本数）"""
        return int(self.sample_rate * self.buffer_size_seconds)
    
    @property
    def overlap_samples(self) -> int:
        """重叠样本数"""
        return int(self.buffer_size_samples * self.overlap_ratio)
    
    @property
    def min_process_samples(self) -> int:
        """最小处理样本数"""
        return int(self.sample_rate * self.min_process_seconds)
    
    def add_samples(self, new_samples: np.ndarray):
        """添加音频样本"""
        self.samples.extend(new_samples.tolist())
    
    def should_process(self) -> bool:
        """判断是否应该处理缓冲区"""
        return len(self.samples) >= self.buffer_size_samples
    
    def get_process_samples(self) -> np.ndarray:
        """获取待处理的样本并清理缓冲区"""
        if len(self.samples) < self.buffer_size_samples:
            return None
        
        # 取出处理样本
        process_samples = np.array(self.samples[:self.buffer_size_samples])
        
        # 清理缓冲区，保留重叠部分
        keep_samples = max(0, self.buffer_size_samples - self.overlap_samples)
        self.samples = self.samples[keep_samples:]
        
        return process_samples
    
    def get_remaining_samples(self) -> Optional[np.ndarray]:
        """获取剩余样本（用于会话结束时处理）"""
        if len(self.samples) >= self.min_process_samples:
            remaining = np.array(self.samples)
            self.samples.clear()
            return remaining
        return None
    
    def clear(self):
        """清空缓冲区"""
        self.samples.clear()


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_stats: Dict[str, ConnectionStats] = {}
        self.connection_buffers: Dict[str, AudioBuffer] = {}
        self._connection_counter = 0
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_started = False
    
    def _start_cleanup_task(self):
        """启动清理任务"""
        if not self._cleanup_started:
            try:
                if self._cleanup_task is None or self._cleanup_task.done():
                    self._cleanup_task = asyncio.create_task(self._cleanup_inactive_connections())
                    self._cleanup_started = True
            except RuntimeError:
                # 如果没有运行的事件循环，延迟启动
                pass
    
    async def _cleanup_inactive_connections(self):
        """清理不活跃的连接"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                
                current_time = datetime.now()
                inactive_threshold = timedelta(minutes=30)  # 30分钟不活跃则清理
                
                inactive_connections = []
                for conn_id, stats in self.connection_stats.items():
                    if current_time - stats.last_activity > inactive_threshold:
                        inactive_connections.append(conn_id)
                
                for conn_id in inactive_connections:
                    logger.info(f"清理不活跃的WebSocket连接: {conn_id}")
                    await self.disconnect(conn_id)
                
            except Exception as e:
                logger.error(f"清理连接时出错: {e}")
    
    def generate_connection_id(self, websocket: WebSocket) -> str:
        """生成连接ID"""
        self._connection_counter += 1
        client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
        return f"ws_{self._connection_counter}_{client_info}"
    
    async def connect(self, websocket: WebSocket) -> str:
        """建立WebSocket连接"""
        connection_id = self.generate_connection_id(websocket)

        # 确保清理任务已启动
        self._start_cleanup_task()

        # 存储连接
        self.active_connections[connection_id] = websocket
        
        # 创建统计信息
        client_address = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
        self.connection_stats[connection_id] = ConnectionStats(
            connection_id=connection_id,
            client_address=client_address,
            connected_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        # 创建音频缓冲区
        self.connection_buffers[connection_id] = AudioBuffer(
            sample_rate=settings.SAMPLE_RATE,
            buffer_size_seconds=2.0,
            overlap_ratio=0.25
        )
        
        # 记录监控指标
        if settings.ENABLE_METRICS:
            metrics_collector.record_websocket_connection('connect')
        
        logger.info("WebSocket连接已建立", 
                   connection_id=connection_id, 
                   client_address=client_address,
                   total_connections=len(self.active_connections))
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """断开WebSocket连接"""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                if not websocket.client_state.DISCONNECTED:
                    await websocket.close()
            except Exception as e:
                logger.warning(f"关闭WebSocket连接时出错: {e}")
            
            # 清理资源
            del self.active_connections[connection_id]
            
        if connection_id in self.connection_stats:
            stats = self.connection_stats[connection_id]
            
            # 计算连接持续时间并记录监控指标
            if settings.ENABLE_METRICS:
                connection_duration = (datetime.now() - stats.connected_at).total_seconds()
                metrics_collector.record_websocket_connection('disconnect')
                metrics_collector.record_websocket_connection_duration(connection_duration)
            
            logger.info("WebSocket连接已断开", 
                       connection_id=connection_id,
                       messages_received=stats.messages_received,
                       messages_sent=stats.messages_sent,
                       connection_duration=(datetime.now() - stats.connected_at).total_seconds(),
                       remaining_connections=len(self.active_connections) - 1)
            del self.connection_stats[connection_id]
            
        if connection_id in self.connection_buffers:
            del self.connection_buffers[connection_id]
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """发送消息到指定连接"""
        if connection_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[connection_id]
            message_str = json.dumps(message, ensure_ascii=False)
            await websocket.send_text(message_str)
            
            # 更新统计
            if connection_id in self.connection_stats:
                self.connection_stats[connection_id].add_sent_message(len(message_str))
            
            # 记录监控指标
            if settings.ENABLE_METRICS:
                message_type = message.get('type', 'unknown')
                metrics_collector.record_websocket_message('outbound', message_type)
            
            return True
        except Exception as e:
            logger.error(f"发送消息失败 {connection_id}: {e}")
            await self.disconnect(connection_id)
            return False
    
    async def send_error(self, connection_id: str, error_message: str):
        """发送错误消息"""
        await self.send_message(connection_id, {
            "type": "error",
            "message": error_message,
            "timestamp": time.time()
        })
    
    async def send_transcription(self, connection_id: str, result: Dict[str, Any]):
        """发送识别结果"""
        await self.send_message(connection_id, {
            "type": "transcription",
            "text": result["text"],
            "timestamp": result.get("start_time", 0),
            "speaker": result.get("speaker", "unknown"),
            "language": result.get("language", "unknown"),
            "emotion": result.get("emotion", "unknown"),
            "confidence": result.get("confidence", 0.0)
        })
    
    async def send_end(self, connection_id: str, message: str = "识别会话结束"):
        """发送会话结束消息"""
        await self.send_message(connection_id, {
            "type": "end",
            "message": message,
            "timestamp": time.time()
        })
    
    async def process_audio_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """处理音频消息"""
        try:
            # 解码音频数据
            audio_data = base64.b64decode(message["data"])
            audio_samples = np.frombuffer(audio_data, dtype=np.float32)
            
            # 更新统计
            if connection_id in self.connection_stats:
                stats = self.connection_stats[connection_id]
                stats.add_received_message(len(audio_data))
                stats.add_audio_chunk()
            
            # 添加到缓冲区
            buffer = self.connection_buffers.get(connection_id)
            if not buffer:
                logger.error(f"连接 {connection_id} 的缓冲区不存在")
                return False
            
            buffer.add_samples(audio_samples)
            
            # 检查是否需要处理
            if buffer.should_process():
                await self._process_audio_buffer(connection_id, buffer)
            
            return True
            
        except Exception as e:
            logger.error(f"处理音频消息失败 {connection_id}: {e}")
            await self.send_error(connection_id, f"音频处理错误: {str(e)}")
            return False
    
    async def process_end_message(self, connection_id: str) -> bool:
        """处理结束消息"""
        try:
            buffer = self.connection_buffers.get(connection_id)
            if buffer:
                # 处理剩余音频
                remaining_samples = buffer.get_remaining_samples()
                if remaining_samples is not None:
                    await self._perform_recognition(connection_id, remaining_samples)
            
            await self.send_end(connection_id)
            return True
            
        except Exception as e:
            logger.error(f"处理结束消息失败 {connection_id}: {e}")
            await self.send_error(connection_id, f"处理结束消息错误: {str(e)}")
            return False
    
    async def _process_audio_buffer(self, connection_id: str, buffer: AudioBuffer):
        """处理音频缓冲区"""
        process_samples = buffer.get_process_samples()
        if process_samples is not None:
            await self._perform_recognition(connection_id, process_samples)
    
    async def _perform_recognition(self, connection_id: str, audio_samples: np.ndarray):
        """执行语音识别"""
        try:
            # 更新统计
            if connection_id in self.connection_stats:
                self.connection_stats[connection_id].add_recognition_request()
            
            # 执行识别
            result = await recognize_audio(
                audio_data=audio_samples,
                sample_rate=settings.SAMPLE_RATE,
                enable_vad=True,
                enable_speaker_id=True,
                enable_punctuation=True  # 默认启用标点符号
            )
            
            # 发送结果
            if result["success"] and result["results"]:
                for segment in result["results"]:
                    if segment["text"].strip():  # 只发送非空结果
                        await self.send_transcription(connection_id, segment)
            
        except Exception as e:
            logger.error(f"语音识别失败 {connection_id}: {e}")
            if connection_id in self.connection_stats:
                self.connection_stats[connection_id].add_error()
            await self.send_error(connection_id, f"识别错误: {str(e)}")
    
    def get_connection_stats(self, connection_id: str) -> Optional[ConnectionStats]:
        """获取连接统计信息"""
        return self.connection_stats.get(connection_id)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有连接的统计信息"""
        total_connections = len(self.active_connections)
        total_messages = sum(stats.messages_received + stats.messages_sent 
                           for stats in self.connection_stats.values())
        total_audio_chunks = sum(stats.audio_chunks_processed 
                               for stats in self.connection_stats.values())
        total_recognitions = sum(stats.recognition_requests 
                               for stats in self.connection_stats.values())
        total_errors = sum(stats.errors for stats in self.connection_stats.values())
        
        return {
            "active_connections": total_connections,
            "total_messages": total_messages,
            "total_audio_chunks": total_audio_chunks,
            "total_recognitions": total_recognitions,
            "total_errors": total_errors,
            "connections": {
                conn_id: {
                    "client_address": stats.client_address,
                    "connected_at": stats.connected_at.isoformat(),
                    "last_activity": stats.last_activity.isoformat(),
                    "messages_sent": stats.messages_sent,
                    "messages_received": stats.messages_received,
                    "audio_chunks_processed": stats.audio_chunks_processed,
                    "recognition_requests": stats.recognition_requests,
                    "errors": stats.errors,
                    "bytes_received": stats.bytes_received,
                    "bytes_sent": stats.bytes_sent
                }
                for conn_id, stats in self.connection_stats.items()
            }
        }
    
    async def cleanup(self):
        """清理管理器资源"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 关闭所有连接
        for connection_id in list(self.active_connections.keys()):
            await self.disconnect(connection_id)


# 全局WebSocket管理器实例
websocket_manager = WebSocketManager()


async def get_websocket_manager() -> WebSocketManager:
    """获取WebSocket管理器实例"""
    return websocket_manager


async def cleanup_websocket_manager():
    """清理WebSocket管理器"""
    await websocket_manager.cleanup()