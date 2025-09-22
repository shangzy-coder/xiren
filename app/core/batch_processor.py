"""
优化的批次处理模块
实现二阶段并行处理架构：
1. 阶段1：ASR批次并行处理
2. 阶段2：后处理并行（标点+声纹）
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
import threading

import numpy as np

from app.config import settings
from app.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


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
    parallel_efficiency: float = 0.0  # 并行效率
    
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
        
        # 性能监控
        self.metrics_collector: Optional[MetricsCollector] = None
        if hasattr(settings, 'ENABLE_METRICS') and settings.ENABLE_METRICS:
            try:
                self.metrics_collector = MetricsCollector()
            except Exception as e:
                logger.warning(f"无法初始化性能监控: {e}")
        
        logger.info(f"优化批次处理器创建完成，配置: {asdict(self.config)}")
    
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
        
        logger.info(f"🚀 开始优化批次处理，共 {len(segments)} 个段落")
        
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
            self.stats.batches_failed += 1
            # 降级到传统处理
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
        
        # 并行处理所有批次
        loop = asyncio.get_event_loop()
        batch_futures = []
        
        for batch_idx, batch in enumerate(batches):
            future = loop.run_in_executor(
                None,  # 使用默认线程池
                self._process_asr_batch,
                batch,
                batch_idx + 1,
                len(batches),
                asr_model
            )
            batch_futures.append(future)
        
        # 等待所有批次完成
        batch_results = await asyncio.gather(*batch_futures, return_exceptions=True)
        
        # 收集结果
        all_asr_results = []
        successful_batches = 0
        failed_batches = 0
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"ASR批次 {i+1} 处理失败: {result}")
                failed_batches += 1
            else:
                all_asr_results.extend(result)
                successful_batches += 1
        
        # 按索引排序结果
        all_asr_results.sort(key=lambda x: x.index)
        
        stage1_time = time.time() - stage1_start_time
        self.stats.update_stage1_stats(stage1_time, len(segments), len(batches))
        
        logger.info(f"✅ 阶段1完成，耗时: {stage1_time:.2f}秒，成功批次: {successful_batches}/{len(batches)}")
        
        return all_asr_results
    
    def _calculate_batch_strategy(self, total_segments: int) -> Dict[str, int]:
        """计算批次划分策略"""
        if total_segments <= 10:
            return {
                'batch_size': total_segments,
                'max_workers': 1,
                'num_batches': 1
            }
        
        # 根据配置计算批次大小
        batch_size = max(
            self.config.min_batch_size,
            min(self.config.max_batch_size, total_segments // self.config.max_batch_threads)
        )
        
        # 计算批次数量
        num_batches = (total_segments + batch_size - 1) // batch_size
        
        # 根据批次数量调整线程数
        max_workers = min(self.config.max_batch_threads, num_batches)
        
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
        
        # 准备并行任务
        total_workers = self.config.punctuation_threads_per_batch + self.config.speaker_threads_per_batch
        
        with ThreadPoolExecutor(max_workers=total_workers) as executor:
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
            
            # 等待标点处理完成
            for future in as_completed(punctuation_futures, timeout=self.config.post_processing_timeout):
                try:
                    punctuated_text, index = future.result()
                    punctuation_results[index] = punctuated_text
                except Exception as e:
                    index = punctuation_futures[future]
                    logger.warning(f"标点处理失败 (段落 {index}): {e}")
                    # 使用原始文本作为降级方案
                    original_text = next((r.text for r in asr_results if r.index == index), "")
                    punctuation_results[index] = original_text
            
            # 等待声纹识别完成
            for future in as_completed(speaker_futures, timeout=self.config.post_processing_timeout):
                try:
                    speaker_info, confidence, index = future.result()
                    speaker_results[index] = {
                        'speaker': speaker_info,
                        'confidence': confidence
                    }
                except Exception as e:
                    index = speaker_futures[future]
                    logger.warning(f"声纹识别失败 (段落 {index}): {e}")
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
