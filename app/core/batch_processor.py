"""
优化的批次处理模块
实现二阶段并行处理架构：
1. 阶段1：ASR批次并行处理
2. 阶段2：后处理并行（标点+声纹）
"""

import logging
import time
import asyncio
import psutil
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock

import numpy as np

from app.config import settings
from app.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class ResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self._cpu_threshold = 80.0  # CPU使用率阈值
        self._memory_threshold = 80.0  # 内存使用率阈值
        self._monitoring_enabled = True
    
    def get_cpu_usage(self) -> float:
        """获取当前CPU使用率"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception as e:
            logger.warning(f"获取CPU使用率失败: {e}")
            return 50.0  # 返回保守值
    
    def get_memory_usage(self) -> Tuple[float, int, int]:
        """获取内存使用情况"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent, memory.available, memory.total
        except Exception as e:
            logger.warning(f"获取内存使用率失败: {e}")
            return 50.0, 4 * 1024**3, 8 * 1024**3  # 返回保守值
    
    def should_reduce_concurrency(self) -> bool:
        """检查是否应该降低并发度"""
        if not self._monitoring_enabled:
            return False
        
        cpu_usage = self.get_cpu_usage()
        memory_percent, _, _ = self.get_memory_usage()
        
        return cpu_usage > self._cpu_threshold or memory_percent > self._memory_threshold
    
    def get_optimal_thread_count(self, base_threads: int) -> int:
        """根据系统资源计算最优线程数"""
        if not self._monitoring_enabled:
            return base_threads
        
        cpu_usage = self.get_cpu_usage()
        memory_percent, _, _ = self.get_memory_usage()
        
        # 根据CPU和内存使用率调整线程数
        cpu_factor = max(0.3, 1.0 - (cpu_usage - 50) / 100)
        memory_factor = max(0.3, 1.0 - (memory_percent - 50) / 100)
        
        adjustment_factor = min(cpu_factor, memory_factor)
        optimal_threads = max(1, int(base_threads * adjustment_factor))
        
        if optimal_threads != base_threads:
            logger.info(f"根据资源使用率调整线程数: {base_threads} -> {optimal_threads} "
                       f"(CPU: {cpu_usage:.1f}%, 内存: {memory_percent:.1f}%)")
        
        return optimal_threads


@dataclass
class ErrorRecoveryManager:
    """错误恢复和降级管理器"""
    
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 1.0  # 秒
        self.degradation_threshold = 0.5  # 失败率阈值
        self.failure_history: List[Dict] = []
        self.degradation_active = False
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """判断是否应该重试"""
        if attempt >= self.max_retries:
            return False
        
        # 针对不同类型的异常采用不同策略
        if isinstance(exception, (TimeoutError, asyncio.TimeoutError)):
            return True
        elif isinstance(exception, MemoryError):
            return False  # 内存错误不重试
        elif isinstance(exception, (ConnectionError, OSError)):
            return True
        else:
            return attempt < 2  # 其他错误最多重试1次
    
    def get_retry_delay(self, attempt: int) -> float:
        """获取重试延迟时间（指数退避）"""
        return self.retry_delay * (2 ** attempt)
    
    def should_degrade(self, failure_rate: float) -> bool:
        """判断是否应该启用降级处理"""
        return failure_rate > self.degradation_threshold
    
    def record_failure(self, operation: str, exception: Exception):
        """记录失败信息"""
        self.failure_history.append({
            'operation': operation,
            'exception': str(exception),
            'timestamp': time.time()
        })
        
        # 保持最近的100个错误记录
        if len(self.failure_history) > 100:
            self.failure_history = self.failure_history[-100:]
    
    def get_recent_failure_rate(self, window_minutes: int = 5) -> float:
        """获取最近时间窗口内的失败率"""
        if not self.failure_history:
            return 0.0
        
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        recent_failures = [f for f in self.failure_history if f['timestamp'] >= window_start]
        
        # 简化计算：假设每个失败对应10次尝试
        total_attempts = len(self.failure_history) * 10
        failure_count = len(recent_failures)
        
        return failure_count / max(total_attempts, 1)


@dataclass
class BatchProcessingConfig:
    """批次处理配置"""
    enable_optimized_processing: bool = settings.ENABLE_OPTIMIZED_BATCH_PROCESSING
    enable_parallel_post_processing: bool = settings.ENABLE_PARALLEL_POST_PROCESSING
    
    # 阶段1：ASR批次配置
    max_batch_threads: int = settings.MAX_BATCH_THREADS
    min_batch_size: int = settings.MIN_BATCH_SIZE
    max_batch_size: int = settings.MAX_BATCH_SIZE
    asr_threads_per_batch: int = settings.ASR_THREADS_PER_BATCH
    
    # 阶段2：后处理并行配置
    punctuation_threads_per_batch: int = settings.PUNCTUATION_THREADS_PER_BATCH
    speaker_threads_per_batch: int = settings.SPEAKER_THREADS_PER_BATCH
    post_processing_batch_size: int = settings.POST_PROCESSING_BATCH_SIZE
    post_processing_timeout: int = settings.POST_PROCESSING_TIMEOUT
    
    def validate(self) -> bool:
        """验证配置参数"""
        errors = []
        
        if self.max_batch_threads < 1:
            errors.append(f"max_batch_threads必须至少为1: {self.max_batch_threads}")
        
        if self.min_batch_size < 1:
            errors.append(f"min_batch_size必须至少为1: {self.min_batch_size}")
        
        if self.max_batch_size < self.min_batch_size:
            errors.append(f"max_batch_size必须大于等于min_batch_size: {self.max_batch_size} < {self.min_batch_size}")
        
        if self.post_processing_batch_size < 1:
            errors.append(f"post_processing_batch_size必须至少为1: {self.post_processing_batch_size}")
        
        if self.post_processing_timeout < 1:
            errors.append(f"post_processing_timeout必须至少为1: {self.post_processing_timeout}")
        
        if errors:
            logger.error(f"批次处理配置验证失败: {'; '.join(errors)}")
            raise ValueError(f"批次处理配置错误: {'; '.join(errors)}")
        
        return True


@dataclass
class BatchProcessingStats:
    """批次处理统计信息"""
    total_segments_processed: int = 0
    total_processing_time: float = 0.0
    stage1_time: float = 0.0  # ASR阶段时间
    stage2_time: float = 0.0  # 后处理阶段时间
    batches_created: int = 0
    batches_completed: int = 0
    batches_failed: int = 0
    batches_retried: int = 0  # 重试的批次数
    partial_failures: int = 0  # 部分失败的处理数
    degraded_processing: int = 0  # 降级处理次数
    parallel_efficiency: float = 0.0  # 并行效率
    error_recovery_success: int = 0  # 错误恢复成功次数
    
    def update_stage1_stats(self, processing_time: float, segments_count: int, batches_count: int):
        """更新阶段1统计"""
        self.stage1_time += processing_time
        self.total_segments_processed += segments_count
        self.batches_created += batches_count
    
    def update_stage2_stats(self, processing_time: float):
        """更新阶段2统计"""
        self.stage2_time += processing_time
    
    def update_completion_stats(self, completed: int, failed: int):
        """更新完成统计"""
        self.batches_completed += completed
        self.batches_failed += failed
        self.total_processing_time = self.stage1_time + self.stage2_time
        
        # 计算并行效率
        if self.total_processing_time > 0:
            theoretical_sequential_time = self.stage1_time + self.stage2_time
            self.parallel_efficiency = theoretical_sequential_time / self.total_processing_time


@dataclass
class ProcessingSegment:
    """处理段落数据结构"""
    index: int
    audio_samples: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ASRResult:
    """ASR结果数据结构"""
    index: int
    text: str
    confidence: float
    start_time: float
    end_time: float
    language: str = "unknown"
    emotion: str = "unknown"
    event: str = "unknown"


@dataclass
class PostProcessingResult:
    """后处理结果数据结构"""
    index: int
    text_with_punct: str = ""
    speaker_info: str = "unknown"
    speaker_confidence: float = 0.0


class OptimizedBatchProcessor:
    """优化的批次处理器 - 二阶段并行处理"""
    
    def __init__(self, config: Optional[BatchProcessingConfig] = None):
        self.config = config or BatchProcessingConfig()
        self.config.validate()
        
        self.stats = BatchProcessingStats()
        self._lock = RLock()
        
        # 资源监控器
        self.resource_monitor = ResourceMonitor()
        
        # 错误恢复管理器
        self.error_recovery = ErrorRecoveryManager()
        
        # 专用线程池管理
        self._asr_thread_pool: Optional[ThreadPoolExecutor] = None
        self._punctuation_thread_pool: Optional[ThreadPoolExecutor] = None
        self._speaker_thread_pool: Optional[ThreadPoolExecutor] = None
        self._thread_pools_initialized = False
        
        # 性能监控
        self.metrics_collector: Optional[MetricsCollector] = None
        if hasattr(settings, 'ENABLE_METRICS') and settings.ENABLE_METRICS:
            try:
                self.metrics_collector = MetricsCollector()
            except Exception as e:
                logger.warning(f"无法初始化性能监控: {e}")
        
        logger.info(f"优化批次处理器创建完成，配置: {asdict(self.config)}")
    
    def _initialize_thread_pools(self):
        """初始化专用线程池"""
        if self._thread_pools_initialized:
            return
        
        with self._lock:
            if self._thread_pools_initialized:
                return
            
            # 根据系统资源动态调整线程池大小
            base_asr_threads = self.config.max_batch_threads
            base_post_threads = (self.config.punctuation_threads_per_batch + 
                               self.config.speaker_threads_per_batch)
            
            optimal_asr_threads = self.resource_monitor.get_optimal_thread_count(base_asr_threads)
            optimal_post_threads = self.resource_monitor.get_optimal_thread_count(base_post_threads)
            
            # 创建ASR专用线程池
            self._asr_thread_pool = ThreadPoolExecutor(
                max_workers=optimal_asr_threads,
                thread_name_prefix="asr_batch"
            )
            
            # 创建后处理专用线程池
            self._post_processing_thread_pool = ThreadPoolExecutor(
                max_workers=optimal_post_threads,
                thread_name_prefix="post_processing"
            )
            
            self._thread_pools_initialized = True
            logger.info(f"线程池初始化完成 - ASR: {optimal_asr_threads} 线程, 后处理: {optimal_post_threads} 线程")
    
    def _cleanup_thread_pools(self):
        """清理线程池资源"""
        with self._lock:
            if self._asr_thread_pool:
                self._asr_thread_pool.shutdown(wait=True)
                self._asr_thread_pool = None
            
            if self._post_processing_thread_pool:
                self._post_processing_thread_pool.shutdown(wait=True)
                self._post_processing_thread_pool = None
            
            self._thread_pools_initialized = False
            logger.info("线程池资源已清理")
    
    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            self._cleanup_thread_pools()
        except Exception as e:
            logger.warning(f"清理线程池资源时出错: {e}")
    
    async def process_segments_optimized(
        self,
        segments: List[Dict[str, Any]],
        enable_punctuation: bool = True,
        enable_speaker_id: bool = True,
        asr_model=None,
        punctuation_processor=None,
        speaker_extractor=None
    ) -> List[Dict[str, Any]]:
        """
        优化的段落处理流程
        
        阶段1：批次级ASR并行
        阶段2：后处理并行 (标点 + 声纹)
        """
        if not segments:
            return []
        
        total_start_time = time.time()
        
        # 检查是否启用优化处理
        if not self.config.enable_optimized_processing:
            logger.info("使用传统批次处理（优化处理已禁用）")
            return await self._fallback_processing(segments, enable_punctuation, enable_speaker_id, asr_model, punctuation_processor, speaker_extractor)
        
        # 初始化线程池
        self._initialize_thread_pools()
        
        # 检查系统资源并调整处理策略
        if self.resource_monitor.should_reduce_concurrency():
            logger.warning("系统资源使用率较高，将降低并发处理强度")
        
        logger.info(f"🚀 开始优化批次处理，共 {len(segments)} 个段落")
        cpu_usage = self.resource_monitor.get_cpu_usage()
        memory_percent, memory_available, memory_total = self.resource_monitor.get_memory_usage()
        logger.info(f"📊 系统资源状态 - CPU: {cpu_usage:.1f}%, 内存: {memory_percent:.1f}% ({memory_available//1024**2}MB 可用)")
        
        try:
            # 准备处理段落
            processing_segments = self._prepare_segments(segments)
            
            # 阶段1：ASR批次并行处理
            asr_results = await self._stage1_parallel_asr_processing(processing_segments, asr_model)
            
            # 阶段2：后处理并行
            if self.config.enable_parallel_post_processing:
                final_results = await self._stage2_parallel_post_processing(
                    asr_results, processing_segments, enable_punctuation, enable_speaker_id,
                    punctuation_processor, speaker_extractor
                )
            else:
                final_results = await self._stage2_sequential_post_processing(
                    asr_results, processing_segments, enable_punctuation, enable_speaker_id,
                    punctuation_processor, speaker_extractor
                )
            
            # 更新统计
            total_time = time.time() - total_start_time
            self.stats.update_completion_stats(len(final_results), 0)
            
            # 性能监控
            if self.metrics_collector:
                self.metrics_collector.record_processing_time("optimized_batch_processing", total_time)
                self.metrics_collector.record_counter("segments_processed", len(segments))
            
            logger.info(f"🎉 优化批次处理完成，耗时: {total_time:.2f}秒，平均每段: {total_time/len(segments):.3f}秒")
            logger.info(f"📊 阶段1耗时: {self.stats.stage1_time:.2f}秒，阶段2耗时: {self.stats.stage2_time:.2f}秒")
            logger.info(f"⚡ 并行效率: {self.stats.parallel_efficiency:.2%}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"优化批次处理失败: {e}")
            self.error_recovery.record_failure("optimized_batch_processing", e)
            self.stats.batches_failed += 1
            
            # 检查是否应该启用降级处理
            failure_rate = self.error_recovery.get_recent_failure_rate()
            if self.error_recovery.should_degrade(failure_rate):
                logger.warning(f"失败率过高 ({failure_rate:.2%})，将启用更保守的降级处理策略")
                self.stats.degraded_processing += 1
            
            # 降级到传统处理
            logger.info("降级到传统批次处理模式")
            return await self._fallback_processing(segments, enable_punctuation, enable_speaker_id, asr_model, punctuation_processor, speaker_extractor)
    
    def _prepare_segments(self, segments: List[Dict[str, Any]]) -> List[ProcessingSegment]:
        """准备处理段落"""
        processing_segments = []
        
        for i, segment in enumerate(segments):
            processing_segment = ProcessingSegment(
                index=i,
                audio_samples=segment['samples'],
                sample_rate=segment['sample_rate'],
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                metadata=segment.get('metadata', {})
            )
            processing_segments.append(processing_segment)
        
        return processing_segments
    
    async def _stage1_parallel_asr_processing(
        self,
        segments: List[ProcessingSegment],
        asr_model
    ) -> List[ASRResult]:
        """阶段1：ASR批次并行处理"""
        if not asr_model:
            logger.error("ASR模型未提供，无法进行语音识别")
            return []
        
        stage1_start_time = time.time()
        logger.info(f"🎯 阶段1：ASR批次并行处理开始，{len(segments)} 个段落")
        
        # 计算批次划分
        batch_strategy = self._calculate_batch_strategy(len(segments))
        batches = self._create_batches(segments, batch_strategy['batch_size'])
        
        logger.info(f"📦 创建 {len(batches)} 个批次，每批 {batch_strategy['batch_size']} 个段落，使用 {batch_strategy['max_workers']} 个线程")
        
        # 确保线程池已初始化
        if not self._thread_pools_initialized:
            self._initialize_thread_pools()
        
        # 并行处理所有批次
        loop = asyncio.get_event_loop()
        batch_futures = []
        
        for batch_idx, batch in enumerate(batches):
            future = loop.run_in_executor(
                self._asr_thread_pool,  # 使用专用ASR线程池
                self._process_asr_batch,
                batch,
                batch_idx + 1,
                len(batches),
                asr_model
            )
            batch_futures.append(future)
        
        # 等待所有批次完成
        batch_results = await asyncio.gather(*batch_futures, return_exceptions=True)
        
        # 收集结果并处理失败的批次
        all_asr_results = []
        successful_batches = 0
        failed_batches = 0
        retry_batches = []
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"ASR批次 {i+1} 处理失败: {result}")
                self.error_recovery.record_failure(f"asr_batch_{i+1}", result)
                
                # 检查是否应该重试
                if self.error_recovery.should_retry(0, result):
                    retry_batches.append((i, batches[i]))
                    logger.info(f"将重试ASR批次 {i+1}")
                else:
                    failed_batches += 1
            else:
                all_asr_results.extend(result)
                successful_batches += 1
        
        # 处理需要重试的批次
        if retry_batches:
            logger.info(f"开始重试 {len(retry_batches)} 个失败的ASR批次")
            retry_results = await self._retry_failed_batches(retry_batches, asr_model)
            all_asr_results.extend(retry_results)
            self.stats.batches_retried += len(retry_batches)
        
        # 按索引排序结果
        all_asr_results.sort(key=lambda x: x.index)
        
        stage1_time = time.time() - stage1_start_time
        self.stats.update_stage1_stats(stage1_time, len(segments), len(batches))
        
        logger.info(f"✅ 阶段1完成，耗时: {stage1_time:.2f}秒，成功批次: {successful_batches}/{len(batches)}")
        
        return all_asr_results
    
    async def _retry_failed_batches(
        self,
        retry_batches: List[Tuple[int, List[ProcessingSegment]]],
        asr_model
    ) -> List[ASRResult]:
        """重试失败的ASR批次"""
        retry_results = []
        
        for batch_idx, batch_segments in retry_batches:
            for attempt in range(1, self.error_recovery.max_retries + 1):
                try:
                    # 添加重试延迟
                    if attempt > 1:
                        delay = self.error_recovery.get_retry_delay(attempt - 1)
                        await asyncio.sleep(delay)
                        logger.info(f"重试ASR批次 {batch_idx + 1}，第 {attempt} 次尝试（延迟 {delay:.1f}秒）")
                    
                    # 尝试重新处理批次
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self._asr_thread_pool,
                        self._process_asr_batch,
                        batch_segments,
                        batch_idx + 1,
                        1,  # 重试时只有这一个批次
                        asr_model
                    )
                    
                    retry_results.extend(result)
                    self.stats.error_recovery_success += 1
                    logger.info(f"ASR批次 {batch_idx + 1} 重试成功")
                    break
                    
                except Exception as e:
                    logger.warning(f"ASR批次 {batch_idx + 1} 第 {attempt} 次重试失败: {e}")
                    self.error_recovery.record_failure(f"asr_batch_{batch_idx + 1}_retry_{attempt}", e)
                    
                    if attempt == self.error_recovery.max_retries:
                        logger.error(f"ASR批次 {batch_idx + 1} 重试次数已用完，放弃处理")
                        # 为失败的段落创建空结果
                        for segment in batch_segments:
                            error_result = ASRResult(
                                index=segment.index,
                                text="",
                                confidence=0.0,
                                start_time=segment.start_time,
                                end_time=segment.end_time
                            )
                            retry_results.append(error_result)
        
        return retry_results
    
    def _calculate_batch_strategy(self, total_segments: int) -> Dict[str, int]:
        """计算批次划分策略，考虑系统资源约束和错误历史"""
        if total_segments <= 10:
            return {
                'batch_size': total_segments,
                'max_workers': 1,
                'num_batches': 1
            }
        
        # 根据系统资源动态调整基础配置
        base_max_threads = self.config.max_batch_threads
        optimal_max_threads = self.resource_monitor.get_optimal_thread_count(base_max_threads)
        
        # 检查错误历史，如果最近失败率较高，启用保守策略
        failure_rate = self.error_recovery.get_recent_failure_rate()
        conservative_mode = self.error_recovery.should_degrade(failure_rate)
        
        if conservative_mode:
            optimal_max_threads = max(1, optimal_max_threads // 2)
            self.stats.degraded_processing += 1
            logger.warning(f"检测到高失败率 ({failure_rate:.2%})，启用保守处理模式")
        
        # 根据配置和资源约束计算批次大小
        batch_size = max(
            self.config.min_batch_size,
            min(self.config.max_batch_size, total_segments // optimal_max_threads)
        )
        
        # 如果系统资源紧张，增加批次大小以减少并发度
        if self.resource_monitor.should_reduce_concurrency():
            batch_size = min(self.config.max_batch_size, batch_size * 2)
            logger.info(f"系统资源紧张，调整批次大小为: {batch_size}")
        
        # 如果是保守模式，进一步增加批次大小
        if conservative_mode:
            batch_size = min(self.config.max_batch_size, batch_size * 2)
            logger.info(f"保守模式下，进一步调整批次大小为: {batch_size}")
        
        # 计算批次数量
        num_batches = (total_segments + batch_size - 1) // batch_size
        
        # 根据批次数量和资源约束调整线程数
        max_workers = min(optimal_max_threads, num_batches)
        
        return {
            'batch_size': batch_size,
            'max_workers': max_workers,
            'num_batches': num_batches
        }
    
    def _create_batches(self, segments: List[ProcessingSegment], batch_size: int) -> List[List[ProcessingSegment]]:
        """创建批次"""
        batches = []
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def _process_asr_batch(
        self,
        batch_segments: List[ProcessingSegment],
        batch_idx: int,
        total_batches: int,
        asr_model
    ) -> List[ASRResult]:
        """处理单个ASR批次"""
        batch_start_time = time.time()
        logger.info(f"🔄 ASR批次 {batch_idx}/{total_batches} 开始，包含 {len(batch_segments)} 个段落")
        
        try:
            # 创建识别流
            streams = []
            for segment in batch_segments:
                stream = asr_model.offline_recognizer.create_stream()
                stream.accept_waveform(segment.sample_rate, segment.audio_samples)
                streams.append(stream)
            
            # 批量识别
            asr_model.offline_recognizer.decode_streams(streams)
            
            # 收集结果
            batch_results = []
            for i, stream in enumerate(streams):
                result = stream.result
                segment = batch_segments[i]
                
                asr_result = ASRResult(
                    index=segment.index,
                    text=result.text,
                    confidence=getattr(result, 'confidence', 0.0),
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    language=getattr(result, 'lang', 'unknown'),
                    emotion=getattr(result, 'emotion', 'unknown'),
                    event=getattr(result, 'event', 'unknown')
                )
                batch_results.append(asr_result)
            
            batch_time = time.time() - batch_start_time
            logger.info(f"✅ ASR批次 {batch_idx} 完成，耗时: {batch_time:.2f}秒")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"ASR批次 {batch_idx} 处理失败: {e}")
            # 返回错误占位结果
            error_results = []
            for segment in batch_segments:
                error_result = ASRResult(
                    index=segment.index,
                    text="",
                    confidence=0.0,
                    start_time=segment.start_time,
                    end_time=segment.end_time
                )
                error_results.append(error_result)
            return error_results
    
    async def _stage2_parallel_post_processing(
        self,
        asr_results: List[ASRResult],
        segments: List[ProcessingSegment],
        enable_punctuation: bool,
        enable_speaker_id: bool,
        punctuation_processor,
        speaker_extractor
    ) -> List[Dict[str, Any]]:
        """阶段2：后处理并行（标点+声纹）"""
        stage2_start_time = time.time()
        logger.info(f"🎯 阶段2：后处理并行开始，处理 {len(asr_results)} 个结果")
        
        # 确保线程池已初始化
        if not self._thread_pools_initialized:
            self._initialize_thread_pools()
        
        # 使用专用后处理线程池
        executor = self._post_processing_thread_pool
        
        # 提交标点处理任务
        punctuation_futures = {}
        if enable_punctuation and punctuation_processor:
            for asr_result in asr_results:
                if asr_result.text.strip():
                    future = executor.submit(
                        self._process_punctuation_single,
                        asr_result.text,
                        asr_result.index,
                        punctuation_processor
                    )
                    punctuation_futures[future] = asr_result.index
        
        # 提交声纹识别任务
        speaker_futures = {}
        if enable_speaker_id and speaker_extractor:
            for i, segment in enumerate(segments):
                future = executor.submit(
                    self._process_speaker_single,
                    segment.audio_samples,
                    segment.sample_rate,
                    segment.index,
                    speaker_extractor
                )
                speaker_futures[future] = segment.index
        
        # 收集并行结果
        punctuation_results = {}
        speaker_results = {}
        
        # 等待标点处理完成，增加错误计数和超时处理
        punctuation_success = 0
        punctuation_failures = 0
        
        try:
            for future in as_completed(punctuation_futures, timeout=self.config.post_processing_timeout):
                try:
                    punctuated_text, index = future.result()
                    punctuation_results[index] = punctuated_text
                    punctuation_success += 1
                except Exception as e:
                    index = punctuation_futures[future]
                    logger.warning(f"标点处理失败 (段落 {index}): {e}")
                    self.error_recovery.record_failure(f"punctuation_{index}", e)
                    punctuation_failures += 1
                    # 使用原始文本作为降级方案
                    original_text = next((r.text for r in asr_results if r.index == index), "")
                    punctuation_results[index] = original_text
        except asyncio.TimeoutError:
            logger.error(f"标点处理超时，已处理: {punctuation_success}，失败: {punctuation_failures}")
            self.stats.partial_failures += 1
            # 为超时的任务提供降级结果
            for future, index in punctuation_futures.items():
                if index not in punctuation_results:
                    original_text = next((r.text for r in asr_results if r.index == index), "")
                    punctuation_results[index] = original_text
        
        # 等待声纹识别完成，增加错误计数和超时处理
        speaker_success = 0
        speaker_failures = 0
        
        try:
            for future in as_completed(speaker_futures, timeout=self.config.post_processing_timeout):
                try:
                    speaker_info, confidence, index = future.result()
                    speaker_results[index] = {
                        'speaker': speaker_info,
                        'confidence': confidence
                    }
                    speaker_success += 1
                except Exception as e:
                    index = speaker_futures[future]
                    logger.warning(f"声纹识别失败 (段落 {index}): {e}")
                    self.error_recovery.record_failure(f"speaker_id_{index}", e)
                    speaker_failures += 1
                    speaker_results[index] = {
                        'speaker': 'unknown',
                        'confidence': 0.0
                    }
        except asyncio.TimeoutError:
            logger.error(f"声纹识别超时，已处理: {speaker_success}，失败: {speaker_failures}")
            self.stats.partial_failures += 1
            # 为超时的任务提供降级结果
            for future, index in speaker_futures.items():
                if index not in speaker_results:
                    speaker_results[index] = {
                        'speaker': 'unknown',
                        'confidence': 0.0
                    }
        
        # 合并最终结果
        final_results = []
        for asr_result in asr_results:
            final_result = {
                'text': asr_result.text,
                'start_time': asr_result.start_time,
                'end_time': asr_result.end_time,
                'language': asr_result.language,
                'emotion': asr_result.emotion,
                'event': asr_result.event,
                'confidence': asr_result.confidence
            }
            
            # 添加标点结果
            if asr_result.index in punctuation_results:
                final_result['text_with_punct'] = punctuation_results[asr_result.index]
                final_result['text'] = punctuation_results[asr_result.index]
            else:
                final_result['text_with_punct'] = asr_result.text
            
            # 添加声纹结果
            if asr_result.index in speaker_results:
                final_result['speaker'] = speaker_results[asr_result.index]['speaker']
                final_result['speaker_confidence'] = speaker_results[asr_result.index]['confidence']
            else:
                final_result['speaker'] = 'unknown'
                final_result['speaker_confidence'] = 0.0
            
            final_results.append(final_result)
        
        stage2_time = time.time() - stage2_start_time
        self.stats.update_stage2_stats(stage2_time)
        
        logger.info(f"✅ 阶段2完成，耗时: {stage2_time:.2f}秒")
        logger.info(f"📈 标点处理: {len(punctuation_results)}/{len(asr_results)}, 声纹识别: {len(speaker_results)}/{len(segments)}")
        
        return final_results
    
    def _process_punctuation_single(self, text: str, index: int, punctuation_processor) -> Tuple[str, int]:
        """处理单个段落的标点"""
        try:
            punctuated_text = punctuation_processor.add_punctuation(text)
            return punctuated_text, index
        except Exception as e:
            logger.warning(f"标点处理失败 (段落 {index}): {e}")
            return text, index
    
    def _process_speaker_single(self, audio_samples: np.ndarray, sample_rate: int, index: int, speaker_extractor) -> Tuple[str, float, int]:
        """处理单个段落的声纹识别"""
        try:
            # 提取说话人嵌入
            stream = speaker_extractor.create_stream()
            stream.accept_waveform(sample_rate, audio_samples)
            stream.input_finished()
            
            embedding = speaker_extractor.compute(stream)
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            # 简化的说话人识别逻辑（实际应该使用SpeakerManager）
            speaker_id = f"Speaker_{hash(tuple(embedding.tolist()[:10])) % 1000:03d}"
            confidence = 0.8  # 简化的置信度
            
            return speaker_id, confidence, index
            
        except Exception as e:
            logger.warning(f"声纹识别失败 (段落 {index}): {e}")
            return 'unknown', 0.0, index
    
    async def _stage2_sequential_post_processing(
        self,
        asr_results: List[ASRResult],
        segments: List[ProcessingSegment],
        enable_punctuation: bool,
        enable_speaker_id: bool,
        punctuation_processor,
        speaker_extractor
    ) -> List[Dict[str, Any]]:
        """阶段2：顺序后处理（降级方案）"""
        logger.info("🔄 使用顺序后处理（并行处理已禁用）")
        
        final_results = []
        for i, asr_result in enumerate(asr_results):
            result = {
                'text': asr_result.text,
                'start_time': asr_result.start_time,
                'end_time': asr_result.end_time,
                'language': asr_result.language,
                'emotion': asr_result.emotion,
                'event': asr_result.event,
                'confidence': asr_result.confidence
            }
            
            # 标点处理
            if enable_punctuation and punctuation_processor and asr_result.text.strip():
                try:
                    result['text_with_punct'] = punctuation_processor.add_punctuation(asr_result.text)
                    result['text'] = result['text_with_punct']
                except Exception as e:
                    logger.warning(f"标点处理失败: {e}")
                    result['text_with_punct'] = asr_result.text
            else:
                result['text_with_punct'] = asr_result.text
            
            # 声纹识别
            if enable_speaker_id and speaker_extractor and i < len(segments):
                try:
                    speaker_info, confidence, _ = self._process_speaker_single(
                        segments[i].audio_samples, segments[i].sample_rate, i, speaker_extractor
                    )
                    result['speaker'] = speaker_info
                    result['speaker_confidence'] = confidence
                except Exception as e:
                    logger.warning(f"声纹识别失败: {e}")
                    result['speaker'] = 'unknown'
                    result['speaker_confidence'] = 0.0
            else:
                result['speaker'] = 'unknown'
                result['speaker_confidence'] = 0.0
            
            final_results.append(result)
        
        return final_results
    
    async def _fallback_processing(
        self,
        segments: List[Dict[str, Any]],
        enable_punctuation: bool,
        enable_speaker_id: bool,
        asr_model,
        punctuation_processor,
        speaker_extractor
    ) -> List[Dict[str, Any]]:
        """降级到传统批次处理"""
        logger.info("🔄 使用传统批次处理（降级模式）")
        
        if asr_model and hasattr(asr_model, '_parallel_recognize_segments'):
            return await asr_model._parallel_recognize_segments(
                segments, enable_speaker_id, enable_punctuation
            )
        else:
            # 最基础的顺序处理
            results = []
            for segment in segments:
                result = {
                    'text': '',
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'language': 'unknown',
                    'emotion': 'unknown',
                    'event': 'unknown',
                    'speaker': 'unknown',
                    'text_with_punct': '',
                    'confidence': 0.0,
                    'speaker_confidence': 0.0
                }
                results.append(result)
            return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats_dict = asdict(self.stats)
        stats_dict['config'] = asdict(self.config)
        return stats_dict
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = BatchProcessingStats()
        logger.info("批次处理统计信息已重置")


# 全局批次处理器实例
_global_batch_processor: Optional[OptimizedBatchProcessor] = None

async def get_batch_processor(config: Optional[BatchProcessingConfig] = None) -> OptimizedBatchProcessor:
    """获取全局批次处理器实例"""
    global _global_batch_processor
    
    if _global_batch_processor is None or config is not None:
        _global_batch_processor = OptimizedBatchProcessor(config)
    
    return _global_batch_processor

async def initialize_batch_processor() -> OptimizedBatchProcessor:
    """初始化全局批次处理器"""
    return await get_batch_processor()
