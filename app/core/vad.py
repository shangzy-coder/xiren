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
from app.utils.audio import audio_processor

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
        检测音频中的语音段落（VAD核心服务接口）
        
        专注于语音活动检测，为ASR、说话人识别等模块提供服务
        
        Args:
            audio_data: 音频数据 (float32，已预处理)
            sample_rate: 采样率，如果为None则使用配置中的默认值
            return_samples: 是否返回音频样本数据（通常为False）
            
        Returns:
            检测到的语音段落列表，包含时间戳信息
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
            # VAD处理：使用全新实例避免状态累积
            segments = await self._process_audio_with_fresh_vad(audio_data, sample_rate, return_samples)
            
            # 过滤过短的段落（基于配置）
            filtered_segments = []
            for segment in segments:
                if segment.duration >= self.config.min_speech_duration:
                    filtered_segments.append(segment)
                else:
                    logger.debug(f"过滤过短语音段落: {segment.duration:.3f}s < {self.config.min_speech_duration}s")
            
            # 更新统计信息
            processing_time = time.time() - start_time
            speech_duration = sum(seg.duration for seg in filtered_segments)
            self.stats.update_processing_stats(audio_duration, processing_time, len(filtered_segments), speech_duration)
            
            # 性能监控
            if self.metrics_collector:
                self.metrics_collector.record_processing_time("vad", processing_time)
                self.metrics_collector.record_counter("vad_segments_detected", len(filtered_segments))
                self.metrics_collector.record_gauge("vad_speech_ratio", speech_duration / audio_duration if audio_duration > 0 else 0)
            
            logger.debug(f"VAD检测完成: {len(filtered_segments)}个有效段落, 语音比例: {speech_duration/audio_duration:.2%}, 耗时: {processing_time:.3f}秒")
            return filtered_segments
            
        except Exception as e:
            logger.error(f"VAD语音检测失败: {e}")
            self.stats.error_count += 1
            if self.metrics_collector:
                self.metrics_collector.record_counter("vad_errors", 1)
            return []
    
    async def is_speech_active(
        self, 
        audio_data: np.ndarray, 
        sample_rate: Optional[int] = None,
        min_speech_ratio: float = 0.3
    ) -> bool:
        """
        简单的语音活动检测（布尔值返回）
        
        用于快速判断音频片段是否包含语音，适用于：
        - 流式处理中的实时判断
        - 预处理阶段的快速筛选
        - 降低后续处理的计算负载
        
        Args:
            audio_data: 音频数据片段
            sample_rate: 采样率
            min_speech_ratio: 最小语音比例阈值（0.3表示30%以上为语音才返回True）
            
        Returns:
            True如果检测到足够的语音活动，False否则
        """
        segments = await self.detect_speech_segments(audio_data, sample_rate, return_samples=False)
        
        if not segments:
            return False
        
        audio_duration = len(audio_data) / (sample_rate or self.config.sample_rate)
        speech_duration = sum(seg.duration for seg in segments)
        speech_ratio = speech_duration / audio_duration if audio_duration > 0 else 0
        
        is_active = speech_ratio >= min_speech_ratio
        logger.debug(f"语音活动检测: {speech_ratio:.2%} >= {min_speech_ratio:.2%} = {is_active}")
        
        return is_active
    
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
        sample_rate: Optional[int] = None,
        return_samples: bool = False
    ) -> List[SpeechSegment]:
        """
        处理流式音频数据（为WebSocket、实时处理提供服务）
        
        维护VAD状态，适用于连续的音频流处理
        主要服务于：websocket_manager、实时ASR等
        
        Args:
            audio_chunk: 音频块数据
            sample_rate: 采样率
            return_samples: 是否返回音频样本（流式处理通常需要）
            
        Returns:
            本次检测到的完整语音段落
        """
        if not self._initialized:
            await self.initialize()
        
        if self.vad is None:
            logger.error("VAD模型未初始化")
            return []
        
        sample_rate = sample_rate or self.config.sample_rate
        segments = []
        
        try:
            # 流式处理使用持久的VAD实例（保持状态）
            with self._lock:
                self.vad.accept_waveform(audio_chunk)
                
                # 获取检测到的完整语音段落
                while not self.vad.empty():
                    segment_samples = self.vad.front.samples
                    start_time = self.vad.front.start / sample_rate
                    duration = len(segment_samples) / sample_rate
                    end_time = start_time + duration
                    
                    # 只有满足最小时长要求的段落才返回
                    if duration >= self.config.min_speech_duration:
                        segment = SpeechSegment(
                            start=start_time,
                            end=end_time,
                            duration=duration,
                            confidence=1.0,
                            samples=segment_samples if return_samples else None
                        )
                        segments.append(segment)
                        logger.debug(f"流式VAD检测到语音段落: {start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s)")
                    else:
                        logger.debug(f"流式VAD过滤短段落: {duration:.3f}s < {self.config.min_speech_duration}s")
                    
                    self.vad.pop()
            
            # 更新统计（流式处理的简单统计）
            if segments and self.metrics_collector:
                self.metrics_collector.record_counter("vad_streaming_segments", len(segments))
            
            return segments
            
        except Exception as e:
            logger.error(f"流式VAD处理失败: {e}")
            self.stats.error_count += 1
            return []
    
    def reset_streaming_state(self):
        """
        重置流式处理状态
        
        在音频流结束或切换时调用，清除VAD内部状态
        """
        with self._lock:
            if self.vad:
                # 刷新剩余的音频数据
                self.vad.flush()
                # 重新初始化VAD以清除状态
                self._initialized = False
                self.vad = None
        
        logger.debug("VAD流式处理状态已重置")
    
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
        """
        获取VAD处理统计信息（服务状态监控）
        
        提供详细的性能指标，用于：
        - 系统监控和告警
        - 性能分析和优化
        - 服务健康检查
        
        Returns:
            包含统计信息的字典
        """
        stats_dict = asdict(self.stats)
        
        # 服务状态信息
        stats_dict['service_status'] = {
            'initialized': self._initialized,
            'model_loaded': self.vad is not None,
            'config_valid': True,  # 配置在创建时已验证
            'last_model_load_time': self.stats.model_load_time
        }
        
        # 配置信息（用于调试）
        stats_dict['current_config'] = self.config.to_dict()
        
        # 性能指标
        if self.stats.total_processed_duration > 0:
            stats_dict['performance_metrics'] = {
                'speech_ratio': self.stats.total_speech_duration / self.stats.total_processed_duration,
                'processing_speed_ratio': self.stats.total_processed_duration / self.stats.average_processing_time if self.stats.average_processing_time > 0 else 0,
                'segments_per_second': self.stats.total_segments / self.stats.total_processed_duration,
                'average_segment_duration': self.stats.total_speech_duration / self.stats.total_segments if self.stats.total_segments > 0 else 0,
                'error_rate': self.stats.error_count / (self.stats.total_segments + self.stats.error_count) if (self.stats.total_segments + self.stats.error_count) > 0 else 0
            }
        else:
            stats_dict['performance_metrics'] = {
                'speech_ratio': 0,
                'processing_speed_ratio': 0,
                'segments_per_second': 0,
                'average_segment_duration': 0,
                'error_rate': 0
            }
        
        # 内存和资源使用（如果可用）
        import psutil
        import os
        try:
            process = psutil.Process(os.getpid())
            stats_dict['resource_usage'] = {
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
            }
        except Exception:
            stats_dict['resource_usage'] = {
                'memory_usage_mb': 0,
                'cpu_percent': 0,
            }
        
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
