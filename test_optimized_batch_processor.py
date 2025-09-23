#!/usr/bin/env python3
"""
测试优化批次处理器的性能和功能
"""

import asyncio
import time
import logging
from typing import Dict, List, Any

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockASRModel:
    """模拟ASR模型，用于测试"""
    
    def __init__(self):
        self.offline_recognizer = MockOfflineRecognizer()


class MockOfflineRecognizer:
    """模拟离线识别器"""
    
    def create_stream(self):
        return MockStream()
    
    def decode_streams(self, streams):
        # 模拟批量解码过程
        time.sleep(0.1)  # 模拟处理时间
        for stream in streams:
            stream.result.text = f"Mock result for stream {id(stream)}"


class MockStream:
    """模拟识别流"""
    
    def __init__(self):
        self.result = MockResult()
    
    def accept_waveform(self, sample_rate, audio_samples):
        pass


class MockResult:
    """模拟识别结果"""
    
    def __init__(self):
        self.text = ""


class MockPunctuationProcessor:
    """模拟标点处理器"""
    
    def add_punctuation(self, text):
        time.sleep(0.05)  # 模拟处理时间
        return text + "。"


class MockSpeakerExtractor:
    """模拟声纹提取器"""
    
    def create_stream(self):
        return MockSpeakerStream()
    
    def compute(self, stream):
        time.sleep(0.05)  # 模拟处理时间
        return [0.1] * 512  # 返回模拟的声纹向量


class MockSpeakerStream:
    """模拟声纹流"""
    
    def accept_waveform(self, sample_rate, audio_samples):
        pass
    
    def input_finished(self):
        pass


def create_mock_segments(num_segments: int) -> List[Dict[str, Any]]:
    """创建模拟的音频段落"""
    segments = []
    
    for i in range(num_segments):
        segment = {
            'samples': [0.1] * 16000,  # 1秒的音频数据（模拟）
            'sample_rate': 16000,
            'start_time': i * 1.0,
            'end_time': (i + 1) * 1.0,
            'metadata': {'segment_id': i}
        }
        segments.append(segment)
    
    return segments


async def test_optimized_batch_processor():
    """测试优化批次处理器"""
    from app.core.batch_processor import OptimizedBatchProcessor, BatchProcessingConfig
    
    # 创建配置
    config = BatchProcessingConfig()
    config.enable_optimized_processing = True
    config.enable_parallel_post_processing = True
    config.max_batch_threads = 4
    config.min_batch_size = 2
    config.max_batch_size = 8
    
    # 创建处理器
    processor = OptimizedBatchProcessor(config)
    
    # 创建模拟对象
    asr_model = MockASRModel()
    punctuation_processor = MockPunctuationProcessor()
    speaker_extractor = MockSpeakerExtractor()
    
    # 测试不同大小的段落数量
    test_cases = [5, 10, 20, 50]
    
    for num_segments in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试 {num_segments} 个段落的批次处理")
        logger.info(f"{'='*60}")
        
        # 创建测试数据
        segments = create_mock_segments(num_segments)
        
        # 开始测试
        start_time = time.time()
        
        try:
            results = await processor.process_segments_optimized(
                segments=segments,
                enable_punctuation=True,
                enable_speaker_id=True,
                asr_model=asr_model,
                punctuation_processor=punctuation_processor,
                speaker_extractor=speaker_extractor
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 验证结果
            assert len(results) == num_segments, f"结果数量不匹配: 期望 {num_segments}, 实际 {len(results)}"
            
            # 检查结果格式
            for i, result in enumerate(results):
                assert 'text' in result, f"结果 {i} 缺少 text 字段"
                assert 'start_time' in result, f"结果 {i} 缺少 start_time 字段"
                assert 'end_time' in result, f"结果 {i} 缺少 end_time 字段"
                assert 'speaker' in result, f"结果 {i} 缺少 speaker 字段"
                assert 'text_with_punct' in result, f"结果 {i} 缺少 text_with_punct 字段"
            
            # 打印统计信息
            stats = processor.get_stats()
            logger.info(f"✅ 测试成功完成")
            logger.info(f"⏱️  处理时间: {processing_time:.2f}秒")
            logger.info(f"📊 平均每段: {processing_time/num_segments:.3f}秒")
            logger.info(f"🔧 阶段1时间: {stats['stage1_time']:.2f}秒")
            logger.info(f"🔧 阶段2时间: {stats['stage2_time']:.2f}秒")
            logger.info(f"📦 批次统计: 创建={stats['batches_created']}, 完成={stats['batches_completed']}, 失败={stats['batches_failed']}")
            logger.info(f"🔄 重试统计: 重试={stats['batches_retried']}, 恢复成功={stats['error_recovery_success']}")
            logger.info(f"⚡ 并行效率: {stats['parallel_efficiency']:.2%}")
            
        except Exception as e:
            logger.error(f"❌ 测试失败: {e}")
            raise
        
        # 重置统计信息
        processor.reset_stats()


async def test_error_handling():
    """测试错误处理机制"""
    from app.core.batch_processor import OptimizedBatchProcessor, BatchProcessingConfig
    
    logger.info(f"\n{'='*60}")
    logger.info("测试错误处理和降级机制")
    logger.info(f"{'='*60}")
    
    class FailingASRModel:
        """故意失败的ASR模型"""
        def __init__(self):
            self.offline_recognizer = FailingOfflineRecognizer()
    
    class FailingOfflineRecognizer:
        def create_stream(self):
            return MockStream()
        
        def decode_streams(self, streams):
            raise RuntimeError("模拟ASR处理失败")
    
    # 创建配置
    config = BatchProcessingConfig()
    config.enable_optimized_processing = True
    
    # 创建处理器
    processor = OptimizedBatchProcessor(config)
    
    # 创建失败的模型
    failing_asr_model = FailingASRModel()
    punctuation_processor = MockPunctuationProcessor()
    speaker_extractor = MockSpeakerExtractor()
    
    # 创建测试数据
    segments = create_mock_segments(10)
    
    # 测试错误处理
    start_time = time.time()
    
    try:
        results = await processor.process_segments_optimized(
            segments=segments,
            enable_punctuation=True,
            enable_speaker_id=True,
            asr_model=failing_asr_model,
            punctuation_processor=punctuation_processor,
            speaker_extractor=speaker_extractor
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"✅ 错误处理测试完成")
        logger.info(f"⏱️  处理时间: {processing_time:.2f}秒")
        logger.info(f"📊 结果数量: {len(results)}")
        
        # 检查统计信息
        stats = processor.get_stats()
        logger.info(f"📦 批次统计: 创建={stats['batches_created']}, 完成={stats['batches_completed']}, 失败={stats['batches_failed']}")
        logger.info(f"🔄 错误统计: 重试={stats['batches_retried']}, 降级={stats['degraded_processing']}")
        
        # 验证降级处理是否工作
        assert stats['batches_failed'] > 0 or stats['degraded_processing'] > 0, "应该有失败或降级处理"
        
    except Exception as e:
        logger.error(f"❌ 错误处理测试失败: {e}")
        raise


async def main():
    """主测试函数"""
    logger.info("开始优化批次处理器性能测试")
    
    try:
        # 测试正常功能
        await test_optimized_batch_processor()
        
        # 测试错误处理
        await test_error_handling()
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 所有测试通过！优化批次处理器工作正常")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # 运行测试
    success = asyncio.run(main())
    
    if success:
        print("✅ 测试成功完成")
        exit(0)
    else:
        print("❌ 测试失败")
        exit(1)
