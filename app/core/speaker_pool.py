"""
声纹池管理模块
基于Sherpa-ONNX实现说话人注册、识别和管理功能

主要功能：
- 说话人声纹特征提取
- 说话人注册和匹配
- 临时声纹池管理
- 多说话人分离和识别
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
import pickle

import numpy as np
import sherpa_onnx

from app.config import settings

logger = logging.getLogger(__name__)


class SpeakerPool:
    """说话人声纹池管理器"""
    
    def __init__(self, 
                 embedding_model_path: str = None,
                 segmentation_model_path: str = None,
                 num_threads: int = 2,
                 provider: str = "cpu"):
        """
        初始化声纹池管理器
        
        Args:
            embedding_model_path: 声纹提取模型路径
            segmentation_model_path: 说话人分离模型路径 (可选)
            num_threads: 推理线程数
            provider: 推理提供者 (cpu, cuda, coreml)
        """
        self.embedding_model_path = embedding_model_path or settings.SPEAKER_MODEL_PATH
        self.segmentation_model_path = segmentation_model_path
        self.num_threads = num_threads
        self.provider = provider
        
        # 模型组件
        self.embedding_extractor: Optional[sherpa_onnx.SpeakerEmbeddingExtractor] = None
        self.speaker_manager: Optional[sherpa_onnx.SpeakerEmbeddingManager] = None
        self.diarization: Optional[sherpa_onnx.OfflineSpeakerDiarization] = None
        
        # 声纹存储
        self.registered_speakers: Dict[str, np.ndarray] = {}
        self.speaker_metadata: Dict[str, Dict[str, Any]] = {}
        self.speaker_counter = 0
        
        # 线程安全
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=num_threads)
        
        # 初始化标志
        self._initialized = False
    
    async def initialize(self) -> bool:
        """异步初始化模型"""
        if self._initialized:
            return True
            
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._init_models)
            self._initialized = True
            logger.info("声纹池管理器初始化成功")
            return True
        except Exception as e:
            logger.error(f"声纹池管理器初始化失败: {e}")
            return False
    
    def _init_models(self):
        """同步初始化模型"""
        # 初始化声纹提取器
        if Path(self.embedding_model_path).exists():
            embedding_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self.embedding_model_path,
                num_threads=self.num_threads,
                debug=False,
                provider=self.provider,
            )
            
            if embedding_config.validate():
                self.embedding_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(embedding_config)
                # 创建声纹管理器
                self.speaker_manager = sherpa_onnx.SpeakerEmbeddingManager(
                    self.embedding_extractor.dim
                )
                logger.info(f"声纹提取器初始化成功，特征维度: {self.embedding_extractor.dim}")
            else:
                logger.error(f"声纹提取器配置验证失败: {embedding_config}")
        
        # 初始化说话人分离器 (可选)
        if self.segmentation_model_path and Path(self.segmentation_model_path).exists():
            try:
                diarization_config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                            model=self.segmentation_model_path
                        ),
                    ),
                    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                        model=self.embedding_model_path
                    ),
                    clustering=sherpa_onnx.FastClusteringConfig(
                        num_clusters=-1,  # 自动检测说话人数量
                        threshold=settings.SPEAKER_SIMILARITY_THRESHOLD
                    ),
                    min_duration_on=0.3,
                    min_duration_off=0.5,
                )
                
                if diarization_config.validate():
                    self.diarization = sherpa_onnx.OfflineSpeakerDiarization(diarization_config)
                    logger.info("说话人分离器初始化成功")
            except Exception as e:
                logger.warning(f"说话人分离器初始化失败: {e}")
    
    async def extract_embedding(self, 
                              audio_data: np.ndarray, 
                              sample_rate: int) -> Optional[np.ndarray]:
        """
        提取音频的声纹特征
        
        Args:
            audio_data: 音频数据 (float32)
            sample_rate: 采样率
            
        Returns:
            声纹特征向量或None
        """
        if not self._initialized or not self.embedding_extractor:
            await self.initialize()
            
        if not self.embedding_extractor:
            logger.error("声纹提取器未初始化")
            return None
        
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._executor, 
                self._extract_embedding_sync, 
                audio_data, 
                sample_rate
            )
            return embedding
        except Exception as e:
            logger.error(f"声纹特征提取失败: {e}")
            return None
    
    def _extract_embedding_sync(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """同步提取声纹特征"""
        with self._lock:
            stream = self.embedding_extractor.create_stream()
            stream.accept_waveform(sample_rate=sample_rate, waveform=audio_data)
            stream.input_finished()
            
            if self.embedding_extractor.is_ready(stream):
                embedding = self.embedding_extractor.compute(stream)
                return np.array(embedding)
            else:
                raise RuntimeError("声纹提取器未就绪")
    
    async def register_speaker(self, 
                             speaker_name: str, 
                             audio_data: np.ndarray, 
                             sample_rate: int,
                             metadata: Dict[str, Any] = None) -> bool:
        """
        注册新的说话人
        
        Args:
            speaker_name: 说话人姓名
            audio_data: 音频数据
            sample_rate: 采样率
            metadata: 附加元数据
            
        Returns:
            注册是否成功
        """
        try:
            # 提取声纹特征
            embedding = await self.extract_embedding(audio_data, sample_rate)
            if embedding is None:
                logger.error(f"无法为说话人 {speaker_name} 提取声纹特征")
                return False
            
            with self._lock:
                # 检查是否已存在
                if speaker_name in self.registered_speakers:
                    # 平均已有特征
                    existing_embedding = self.registered_speakers[speaker_name]
                    averaged_embedding = (existing_embedding + embedding) / 2.0
                    self.registered_speakers[speaker_name] = averaged_embedding
                    logger.info(f"更新说话人 {speaker_name} 的声纹特征")
                else:
                    self.registered_speakers[speaker_name] = embedding
                    logger.info(f"注册新说话人 {speaker_name}")
                
                # 更新管理器
                if self.speaker_manager:
                    status = self.speaker_manager.add(speaker_name, embedding)
                    if not status:
                        logger.warning(f"声纹管理器注册失败: {speaker_name}")
                
                # 保存元数据
                if metadata is None:
                    metadata = {}
                metadata.update({
                    'registration_time': time.time(),
                    'embedding_dim': len(embedding),
                })
                self.speaker_metadata[speaker_name] = metadata
                
            return True
            
        except Exception as e:
            logger.error(f"注册说话人失败 {speaker_name}: {e}")
            return False
    
    async def identify_speaker(self, 
                             audio_data: np.ndarray, 
                             sample_rate: int,
                             threshold: float = None) -> Optional[Tuple[str, float]]:
        """
        识别说话人
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            threshold: 相似度阈值
            
        Returns:
            (说话人姓名, 相似度分数) 或 None
        """
        if threshold is None:
            threshold = settings.SPEAKER_SIMILARITY_THRESHOLD
            
        try:
            # 提取声纹特征
            embedding = await self.extract_embedding(audio_data, sample_rate)
            if embedding is None:
                return None
            
            with self._lock:
                if not self.registered_speakers:
                    logger.info("暂无注册的说话人")
                    return None
                
                # 使用Sherpa-ONNX管理器搜索
                if self.speaker_manager:
                    speaker_name = self.speaker_manager.search(embedding, threshold=threshold)
                    if speaker_name:
                        # 计算相似度分数
                        registered_embedding = self.registered_speakers[speaker_name]
                        similarity = self._compute_similarity(embedding, registered_embedding)
                        return speaker_name, similarity
                
                # 回退到手动搜索
                best_match = None
                best_similarity = -1.0
                
                for name, registered_embedding in self.registered_speakers.items():
                    similarity = self._compute_similarity(embedding, registered_embedding)
                    if similarity > best_similarity and similarity >= threshold:
                        best_similarity = similarity
                        best_match = name
                
                if best_match:
                    return best_match, best_similarity
                
            return None
            
        except Exception as e:
            logger.error(f"说话人识别失败: {e}")
            return None
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个声纹特征的余弦相似度"""
        # 归一化
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 余弦相似度
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    async def diarize_speakers(self, 
                             audio_data: np.ndarray, 
                             sample_rate: int,
                             num_speakers: int = -1) -> List[Dict[str, Any]]:
        """
        进行说话人分离
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            num_speakers: 预期说话人数量 (-1 为自动检测)
            
        Returns:
            分离结果列表 [{'start': float, 'end': float, 'speaker': str}]
        """
        if not self.diarization:
            logger.warning("说话人分离器未初始化")
            return []
        
        try:
            loop = asyncio.get_event_loop()
            segments = await loop.run_in_executor(
                self._executor,
                self._diarize_speakers_sync,
                audio_data,
                sample_rate,
                num_speakers
            )
            return segments
        except Exception as e:
            logger.error(f"说话人分离失败: {e}")
            return []
    
    def _diarize_speakers_sync(self, 
                             audio_data: np.ndarray, 
                             sample_rate: int,
                             num_speakers: int) -> List[Dict[str, Any]]:
        """同步进行说话人分离"""
        with self._lock:
            # 重新配置聚类参数
            if num_speakers > 0:
                self.diarization.config.clustering.num_clusters = num_speakers
            
            # 执行分离
            result = self.diarization.process(audio_data).sort_by_start_time()
            
            segments = []
            for segment in result:
                segments.append({
                    'start': float(segment.start),
                    'end': float(segment.end),
                    'speaker': f"Speaker_{segment.speaker:02d}",
                    'duration': float(segment.end - segment.start)
                })
            
            return segments
    
    def get_registered_speakers(self) -> List[Dict[str, Any]]:
        """获取已注册的说话人列表"""
        with self._lock:
            speakers = []
            for name, embedding in self.registered_speakers.items():
                metadata = self.speaker_metadata.get(name, {})
                speakers.append({
                    'name': name,
                    'embedding_dim': len(embedding),
                    'metadata': metadata
                })
            return speakers
    
    def remove_speaker(self, speaker_name: str) -> bool:
        """删除已注册的说话人"""
        with self._lock:
            if speaker_name in self.registered_speakers:
                del self.registered_speakers[speaker_name]
                self.speaker_metadata.pop(speaker_name, None)
                
                # 从管理器中移除
                if self.speaker_manager:
                    self.speaker_manager.remove(speaker_name)
                
                logger.info(f"删除说话人: {speaker_name}")
                return True
            return False
    
    def save_speakers(self, filepath: str) -> bool:
        """保存说话人数据到文件"""
        try:
            with self._lock:
                data = {
                    'speakers': self.registered_speakers,
                    'metadata': self.speaker_metadata,
                    'timestamp': time.time()
                }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                
                logger.info(f"说话人数据已保存到: {filepath}")
                return True
        except Exception as e:
            logger.error(f"保存说话人数据失败: {e}")
            return False
    
    def load_speakers(self, filepath: str) -> bool:
        """从文件加载说话人数据"""
        try:
            if not Path(filepath).exists():
                logger.warning(f"说话人数据文件不存在: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            with self._lock:
                self.registered_speakers = data.get('speakers', {})
                self.speaker_metadata = data.get('metadata', {})
                
                # 重新注册到管理器
                if self.speaker_manager:
                    for name, embedding in self.registered_speakers.items():
                        self.speaker_manager.add(name, embedding)
                
                logger.info(f"从文件加载 {len(self.registered_speakers)} 个说话人")
                return True
                
        except Exception as e:
            logger.error(f"加载说话人数据失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取声纹池统计信息"""
        with self._lock:
            return {
                'total_speakers': len(self.registered_speakers),
                'initialized': self._initialized,
                'has_extractor': self.embedding_extractor is not None,
                'has_manager': self.speaker_manager is not None,
                'has_diarization': self.diarization is not None,
                'embedding_dim': self.embedding_extractor.dim if self.embedding_extractor else 0,
                'provider': self.provider,
                'num_threads': self.num_threads
            }


# 全局声纹池实例
speaker_pool: Optional[SpeakerPool] = None

async def get_speaker_pool() -> SpeakerPool:
    """获取全局声纹池实例"""
    global speaker_pool
    if speaker_pool is None:
        speaker_pool = SpeakerPool()
        await speaker_pool.initialize()
    return speaker_pool

async def initialize_speaker_pool() -> SpeakerPool:
    """初始化全局声纹池"""
    return await get_speaker_pool()
