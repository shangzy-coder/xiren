"""
统一VAD (Voice Activity Detection) 处理模块
重构现有的分散VAD逻辑，提供统一、高效的语音活动检测接口
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import sherpa_onnx

from app.config import settings
from app.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """VAD配置类"""
    model_path: str = settings.VAD_MODEL_PATH
    threshold: float = settings.VAD_THRESHOLD
    min_silence_duration: float = settings.VAD_MIN_SILENCE_DURATION
    min_speech_duration: float = settings.VAD_MIN_SPEECH_DURATION
    max_speech_duration: float = settings.VAD_MAX_SPEECH_DURATION
    buffer_size_seconds: float = settings.VAD_BUFFER_SIZE_SECONDS
    provider: str = settings.VAD_PROVIDER
    num_threads: int = settings.VAD_THREADS
    sample_rate: int = settings.SAMPLE_RATE
    
    def __post_init__(self):
        """配置验证和后处理"""
        self.validate()
    
    def validate(self) -> bool:
        """验证配置参数"""
        errors = []
        
        # 验证模型路径
        if not Path(self.model_path).exists():
            errors.append(f"VAD模型文件不存在: {self.model_path}")
        
        # 验证数值参数
        if not 0.0 <= self.threshold <= 1.0:
            errors.append(f"VAD阈值必须在0-1之间: {self.threshold}")
        
        if self.min_silence_duration < 0:
            errors.append(f"最小静音时长不能为负: {self.min_silence_duration}")
        
        if self.min_speech_duration < 0:
            errors.append(f"最小语音时长不能为负: {self.min_speech_duration}")
        
        if self.max_speech_duration <= self.min_speech_duration:
            errors.append(f"最大语音时长必须大于最小语音时长: {self.max_speech_duration} <= {self.min_speech_duration}")
        
        if self.buffer_size_seconds <= 0:
            errors.append(f"缓冲区大小必须为正数: {self.buffer_size_seconds}")
        
        if self.num_threads < 1:
            errors.append(f"线程数必须至少为1: {self.num_threads}")
        
        if self.sample_rate <= 0:
            errors.append(f"采样率必须为正数: {self.sample_rate}")
        
        if errors:
            logger.error(f"VAD配置验证失败: {'; '.join(errors)}")
            raise ValueError(f"VAD配置错误: {'; '.join(errors)}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VADConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"忽略未知的VAD配置参数: {key}")
        self.validate()


@dataclass
class VADStats:
    """VAD统计信息"""
    total_processed_duration: float = 0.0
    total_segments: int = 0
    total_speech_duration: float = 0.0
    total_silence_duration: float = 0.0
    average_processing_time: float = 0.0
    peak_memory_usage: int = 0
    model_load_time: float = 0.0
    error_count: int = 0
    
    def update_processing_stats(self, audio_duration: float, processing_time: float, segments_count: int, speech_duration: float):
        """更新处理统计"""
        self.total_processed_duration += audio_duration
        self.total_segments += segments_count
        self.total_speech_duration += speech_duration
        self.total_silence_duration += audio_duration - speech_duration
        
        # 更新平均处理时间（滑动平均）
        if self.average_processing_time == 0:
            self.average_processing_time = processing_time
        else:
            self.average_processing_time = (self.average_processing_time * 0.9) + (processing_time * 0.1)


@dataclass
class SpeechSegment:
    """语音段落信息"""
    start: float      # 开始时间（秒）
    end: float        # 结束时间（秒）
    duration: float   # 持续时间（秒）
    confidence: float = 1.0  # 置信度
    samples: Optional[np.ndarray] = None  # 音频样本数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（不包含samples）"""
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "confidence": self.confidence
        }


class VADProcessor:
    """统一VAD处理器"""
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self.vad: Optional[sherpa_onnx.VoiceActivityDetector] = None
        self.stats = VADStats()
        self._lock = threading.Lock()
        self._initialized = False
        self._model_load_start = None
        
        # 内存优化
        self._audio_buffer_pool = []
        self._max_pool_size = 10
        
        # 性能监控
        self.metrics_collector: Optional[MetricsCollector] = None
        if hasattr(settings, 'ENABLE_METRICS') and settings.ENABLE_METRICS:
            try:
                self.metrics_collector = MetricsCollector()
            except Exception as e:
                logger.warning(f"无法初始化性能监控: {e}")
        
        logger.info(f"VAD处理器创建完成，配置: {self.config.to_dict()}")
    
    async def initialize(self) -> bool:
        """异步初始化VAD处理器"""
        if self._initialized:
            return True
            
        try:
            self._model_load_start = time.time()
            
            # 在线程池中加载模型，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, self._load_model)
            
            if success:
                self.stats.model_load_time = time.time() - self._model_load_start
                self._initialized = True
                logger.info(f"VAD处理器初始化成功，耗时: {self.stats.model_load_time:.2f}秒")
            
            return success
            
        except Exception as e:
            logger.error(f"VAD处理器初始化失败: {e}")
            self.stats.error_count += 1
            return False
    
    def _load_model(self) -> bool:
        """同步加载VAD模型"""
        try:
            with self._lock:
                # 创建VAD配置
                vad_config = sherpa_onnx.VadModelConfig()
                vad_config.silero_vad.model = str(self.config.model_path)
                vad_config.silero_vad.threshold = self.config.threshold
                vad_config.silero_vad.min_silence_duration = self.config.min_silence_duration
                vad_config.silero_vad.min_speech_duration = self.config.min_speech_duration
                vad_config.silero_vad.max_speech_duration = self.config.max_speech_duration
                vad_config.sample_rate = self.config.sample_rate
                vad_config.num_threads = self.config.num_threads
                vad_config.provider = self.config.provider
                
                # 创建VAD实例
                self.vad = sherpa_onnx.VoiceActivityDetector(
                    vad_config, 
                    buffer_size_in_seconds=self.config.buffer_size_seconds
                )
                
                logger.info("VAD模型加载成功")
                return True
                
        except Exception as e:
            logger.error(f"VAD模型加载失败: {e}")
            return False
    
    async def detect_speech_segments(
        self, 
        audio_data: np.ndarray, 
        sample_rate: Optional[int] = None,
        return_samples: bool = False
    ) -> List[SpeechSegment]:
        """
        检测音频中的语音段落
        
        Args:
            audio_data: 音频数据 (float32)
            sample_rate: 采样率，如果为None则使用配置中的默认值
            return_samples: 是否返回音频样本数据
            
        Returns:
            检测到的语音段落列表
        """
        if not self._initialized:
            await self.initialize()
        
        if self.vad is None:
            logger.error("VAD模型未初始化")
            return []
        
        start_time = time.time()
        sample_rate = sample_rate or self.config.sample_rate
        audio_duration = len(audio_data) / sample_rate
        
        try:
            # 为了避免状态累积，每次都创建新的VAD实例
            segments = await self._process_audio_with_fresh_vad(audio_data, sample_rate, return_samples)
            
            # 更新统计信息
            processing_time = time.time() - start_time
            speech_duration = sum(seg.duration for seg in segments)
            self.stats.update_processing_stats(audio_duration, processing_time, len(segments), speech_duration)
            
            # 性能监控
            if self.metrics_collector:
                self.metrics_collector.record_processing_time("vad", processing_time)
                self.metrics_collector.record_counter("vad_segments_detected", len(segments))
            
            logger.debug(f"VAD检测完成: {len(segments)}个段落, 耗时: {processing_time:.3f}秒")
            return segments
            
        except Exception as e:
            logger.error(f"VAD语音检测失败: {e}")
            self.stats.error_count += 1
            return []
    
    async def _process_audio_with_fresh_vad(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        return_samples: bool
    ) -> List[SpeechSegment]:
        """使用全新VAD实例处理音频（避免状态累积）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_process_audio, audio_data, sample_rate, return_samples)
    
    def _sync_process_audio(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        return_samples: bool
    ) -> List[SpeechSegment]:
        """同步处理音频数据"""
        segments = []
        
        try:
            # 创建新的VAD实例以确保状态清零
            vad_config = sherpa_onnx.VadModelConfig()
            vad_config.silero_vad.model = str(self.config.model_path)
            vad_config.silero_vad.threshold = self.config.threshold
            vad_config.silero_vad.min_silence_duration = self.config.min_silence_duration
            vad_config.silero_vad.min_speech_duration = self.config.min_speech_duration
            vad_config.silero_vad.max_speech_duration = self.config.max_speech_duration
            vad_config.sample_rate = sample_rate
            vad_config.num_threads = self.config.num_threads
            vad_config.provider = self.config.provider
            
            # 创建新的VAD实例
            fresh_vad = sherpa_onnx.VoiceActivityDetector(
                vad_config, 
                buffer_size_in_seconds=self.config.buffer_size_seconds
            )
            
            # 获取窗口大小
            window_size = vad_config.silero_vad.window_size
            total_samples_processed = 0
            
            logger.debug(f"开始VAD分割，音频长度: {len(audio_data)}，窗口大小: {window_size}")
            
            # 处理音频数据 - 使用完整窗口
            while len(audio_data) > total_samples_processed + window_size:
                chunk = audio_data[total_samples_processed:total_samples_processed + window_size]
                fresh_vad.accept_waveform(chunk)
                total_samples_processed += window_size
                
                # 获取检测到的语音段落
                while not fresh_vad.empty():
                    segment_samples = fresh_vad.front.samples
                    start_time = fresh_vad.front.start / sample_rate
                    duration = len(segment_samples) / sample_rate
                    end_time = start_time + duration
                    
                    segment = SpeechSegment(
                        start=start_time,
                        end=end_time,
                        duration=duration,
                        confidence=1.0,  # Silero VAD不提供置信度，默认为1.0
                        samples=segment_samples if return_samples else None
                    )
                    segments.append(segment)
                    fresh_vad.pop()
            
            # 处理剩余的音频数据
            fresh_vad.flush()
            while not fresh_vad.empty():
                segment_samples = fresh_vad.front.samples
                start_time = fresh_vad.front.start / sample_rate
                duration = len(segment_samples) / sample_rate
                end_time = start_time + duration
                
                segment = SpeechSegment(
                    start=start_time,
                    end=end_time,
                    duration=duration,
                    confidence=1.0,
                    samples=segment_samples if return_samples else None
                )
                segments.append(segment)
                fresh_vad.pop()
                
        except Exception as e:
            logger.error(f"VAD处理失败: {e}")
            # 处理失败时返回整段音频
            duration = len(audio_data) / sample_rate
            segments = [SpeechSegment(
                start=0.0,
                end=duration,
                duration=duration,
                confidence=1.0,
                samples=audio_data if return_samples else None
            )]
        
        return segments
    
    async def process_streaming_audio(
        self, 
        audio_chunk: np.ndarray, 
        sample_rate: Optional[int] = None
    ) -> List[SpeechSegment]:
        """
        处理流式音频数据
        注意：流式处理需要维护VAD状态，适用于实时音频流
        
        Args:
            audio_chunk: 音频块数据
            sample_rate: 采样率
            
        Returns:
            本次检测到的语音段落
        """
        if not self._initialized:
            await self.initialize()
        
        if self.vad is None:
            logger.error("VAD模型未初始化")
            return []
        
        sample_rate = sample_rate or self.config.sample_rate
        segments = []
        
        try:
            # 流式处理使用持久的VAD实例
            with self._lock:
                self.vad.accept_waveform(audio_chunk)
                
                # 获取检测到的语音段落
                while not self.vad.empty():
                    segment_samples = self.vad.front.samples
                    start_time = self.vad.front.start / sample_rate
                    duration = len(segment_samples) / sample_rate
                    end_time = start_time + duration
                    
                    segment = SpeechSegment(
                        start=start_time,
                        end=end_time,
                        duration=duration,
                        confidence=1.0,
                        samples=segment_samples
                    )
                    segments.append(segment)
                    self.vad.pop()
            
            return segments
            
        except Exception as e:
            logger.error(f"流式VAD处理失败: {e}")
            self.stats.error_count += 1
            return []
    
    def configure(self, **kwargs) -> bool:
        """动态配置VAD参数"""
        try:
            self.config.update(**kwargs)
            
            # 如果模型相关配置发生变化，需要重新加载
            model_related_keys = {'model_path', 'threshold', 'min_silence_duration', 
                                'min_speech_duration', 'max_speech_duration', 
                                'buffer_size_seconds', 'provider', 'num_threads', 'sample_rate'}
            
            if any(key in kwargs for key in model_related_keys):
                logger.info("检测到模型相关配置变更，将在下次调用时重新加载VAD模型")
                self._initialized = False
                self.vad = None
            
            return True
            
        except Exception as e:
            logger.error(f"VAD配置更新失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取VAD处理统计信息"""
        stats_dict = asdict(self.stats)
        
        # 添加配置信息
        stats_dict['config'] = self.config.to_dict()
        stats_dict['initialized'] = self._initialized
        
        # 计算衍生统计
        if self.stats.total_processed_duration > 0:
            stats_dict['speech_ratio'] = self.stats.total_speech_duration / self.stats.total_processed_duration
            stats_dict['processing_speed'] = self.stats.total_processed_duration / self.stats.average_processing_time if self.stats.average_processing_time > 0 else 0
        else:
            stats_dict['speech_ratio'] = 0
            stats_dict['processing_speed'] = 0
        
        return stats_dict
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = VADStats()
        logger.info("VAD统计信息已重置")
    
    async def close(self):
        """关闭VAD处理器，释放资源"""
        with self._lock:
            if self.vad:
                self.vad = None
            self._initialized = False
            self._audio_buffer_pool.clear()
        
        logger.info("VAD处理器已关闭")


# 全局VAD处理器实例
_global_vad_processor: Optional[VADProcessor] = None

async def get_vad_processor(config: Optional[VADConfig] = None) -> VADProcessor:
    """获取全局VAD处理器实例"""
    global _global_vad_processor
    
    if _global_vad_processor is None or config is not None:
        _global_vad_processor = VADProcessor(config)
        await _global_vad_processor.initialize()
    
    return _global_vad_processor

async def initialize_vad() -> VADProcessor:
    """初始化全局VAD处理器"""
    return await get_vad_processor()
