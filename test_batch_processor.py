#!/usr/bin/env python3
"""
优化批次处理器测试
测试新的二阶段并行处理架构
"""

import asyncio
import numpy as np
import time
from typing import List, Dict, Any
import random

from app.core.batch_processor import OptimizedBatchProcessor, BatchProcessingConfig
from app.config import settings

class MockASRModel:
    """模拟ASR模型"""
    
    def __init__(self):
        self.offline_recognizer = MockOfflineRecognizer()

class MockOfflineRecognizer:
    """模拟离线识别器"""
    
    def create_stream(self):
        return MockStream()
    
    def decode_streams(self, streams: List):
        """模拟批量解码"""
        time.sleep(0.1)  # 模拟处理时间
        for stream in streams:
            stream._process()

class MockStream:
    """模拟识别流"""
    
    def __init__(self):
        self.audio_data = None
        self.sample_rate = None
        self.result = None
    
    def accept_waveform(self, sample_rate: int, audio_data: np.ndarray):
        self.sample_rate = sample_rate
        self.audio_data = audio_data
    
    def _process(self):
        """模拟处理生成结果"""
        self.result = MockResult()

class MockResult:
    """模拟识别结果"""
    
    def __init__(self):
        # 生成随机文本结果
        words = ["hello", "world", "this", "is", "a", "test", "speech", "recognition", "system"]
        self.text = " ".join(random.choices(words, k=random.randint(3, 8)))
        self.confidence = random.uniform(0.7, 0.95)
        self.lang = "zh"
        self.emotion = "neutral"
        self.event = "speech"

class MockPunctuationProcessor:
    """模拟标点处理器"""
    
    def add_punctuation(self, text: str) -> str:
        time.sleep(0.01)  # 模拟处理时间
        if text.strip():
            return text.strip() + "。"
        return text

class MockSpeakerExtractor:
    """模拟声纹提取器"""
    
    def create_stream(self):
        return MockSpeakerStream()
    
    def compute(self, stream):
        return np.random.random(512)  # 模拟512维声纹特征

class MockSpeakerStream:
    """模拟声纹流"""
    
    def accept_waveform(self, sample_rate: int, audio_data: np.ndarray):
        pass
    
    def input_finished(self):
        pass

def create_test_segments(num_segments: int = 100) -> List[Dict[str, Any]]:
    """创建测试音频段落"""
    segments = []
    sample_rate = 16000
    
    for i in range(num_segments):
        # 生成随机长度的音频数据
        duration = random.uniform(1.0, 5.0)  # 1-5秒
        samples = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        start_time = i * 6.0  # 每段间隔6秒
        end_time = start_time + duration
        
        segment = {
            'samples': samples,
            'sample_rate': sample_rate,
            'start_time': start_time,
            'end_time': end_time,
            'metadata': {'index': i}
        }
        segments.append(segment)
    
    return segments

async def test_basic_processing():
    """测试基本处理功能"""
    print("=== 基本处理功能测试 ===")
    
    # 创建测试数据
    segments = create_test_segments(20)
    print(f"创建了 {len(segments)} 个测试段落")
    
    # 创建模拟组件
    asr_model = MockASRModel()
    punctuation_processor = MockPunctuationProcessor()
    speaker_extractor = MockSpeakerExtractor()
    
    # 创建批次处理器
    processor = OptimizedBatchProcessor()
    
    # 测试处理
    start_time = time.time()
    results = await processor.process_segments_optimized(
        segments=segments,
        enable_punctuation=True,
        enable_speaker_id=True,
        asr_model=asr_model,
        punctuation_processor=punctuation_processor,
        speaker_extractor=speaker_extractor
    )
    processing_time = time.time() - start_time
    
    print(f"✅ 处理完成，耗时: {processing_time:.2f}秒")
    print(f"   - 输入段落: {len(segments)}")
    print(f"   - 输出结果: {len(results)}")
    print(f"   - 平均每段耗时: {processing_time/len(segments):.3f}秒")
    
    # 验证结果
    if len(results) == len(segments):
        print("✅ 结果数量正确")
    else:
        print(f"❌ 结果数量不匹配: 期望 {len(segments)}, 实际 {len(results)}")
        return False
    
    # 检查结果结构
    sample_result = results[0]
    required_fields = ['text', 'text_with_punct', 'speaker', 'start_time', 'end_time', 'confidence']
    for field in required_fields:
        if field not in sample_result:
            print(f"❌ 结果缺少字段: {field}")
            return False
    
    print("✅ 结果结构正确")
    
    # 显示统计信息
    stats = processor.get_stats()
    print(f"📊 处理统计:")
    print(f"   - 总段落数: {stats['total_segments_processed']}")
    print(f"   - 阶段1耗时: {stats['stage1_time']:.2f}秒")
    print(f"   - 阶段2耗时: {stats['stage2_time']:.2f}秒")
    print(f"   - 并行效率: {stats['parallel_efficiency']:.2%}")
    print(f"   - 创建批次: {stats['batches_created']}")
    print(f"   - 完成批次: {stats['batches_completed']}")
    
    return True

async def test_performance_comparison():
    """测试性能对比"""
    print("\n=== 性能对比测试 ===")
    
    # 创建较大的测试数据集
    test_sizes = [50, 100, 200]
    
    for size in test_sizes:
        print(f"\n📊 测试 {size} 个段落的处理性能")
        segments = create_test_segments(size)
        
        # 创建模拟组件
        asr_model = MockASRModel()
        punctuation_processor = MockPunctuationProcessor()
        speaker_extractor = MockSpeakerExtractor()
        
        # 测试优化处理
        processor_optimized = OptimizedBatchProcessor(BatchProcessingConfig(
            enable_optimized_processing=True,
            enable_parallel_post_processing=True
        ))
        
        start_time = time.time()
        results_optimized = await processor_optimized.process_segments_optimized(
            segments=segments,
            enable_punctuation=True,
            enable_speaker_id=True,
            asr_model=asr_model,
            punctuation_processor=punctuation_processor,
            speaker_extractor=speaker_extractor
        )
        optimized_time = time.time() - start_time
        
        # 测试传统处理
        processor_traditional = OptimizedBatchProcessor(BatchProcessingConfig(
            enable_optimized_processing=False,
            enable_parallel_post_processing=False
        ))
        
        start_time = time.time()
        results_traditional = await processor_traditional.process_segments_optimized(
            segments=segments,
            enable_punctuation=True,
            enable_speaker_id=True,
            asr_model=asr_model,
            punctuation_processor=punctuation_processor,
            speaker_extractor=speaker_extractor
        )
        traditional_time = time.time() - start_time
        
        # 计算性能提升
        if traditional_time > 0:
            improvement = (traditional_time - optimized_time) / traditional_time * 100
        else:
            improvement = 0
        
        print(f"   🚀 优化处理: {optimized_time:.2f}秒")
        print(f"   🔄 传统处理: {traditional_time:.2f}秒")
        print(f"   ⚡ 性能提升: {improvement:.1f}%")
        
        # 验证结果一致性
        if len(results_optimized) == len(results_traditional) == len(segments):
            print(f"   ✅ 结果数量一致")
        else:
            print(f"   ❌ 结果数量不一致")

async def test_configuration_options():
    """测试配置选项"""
    print("\n=== 配置选项测试 ===")
    
    segments = create_test_segments(30)
    asr_model = MockASRModel()
    punctuation_processor = MockPunctuationProcessor()
    speaker_extractor = MockSpeakerExtractor()
    
    # 测试不同配置
    configs = [
        ("优化+并行后处理", BatchProcessingConfig(
            enable_optimized_processing=True,
            enable_parallel_post_processing=True,
            max_batch_threads=4,
            punctuation_threads_per_batch=2,
            speaker_threads_per_batch=2
        )),
        ("优化+顺序后处理", BatchProcessingConfig(
            enable_optimized_processing=True,
            enable_parallel_post_processing=False
        )),
        ("传统处理", BatchProcessingConfig(
            enable_optimized_processing=False
        ))
    ]
    
    for config_name, config in configs:
        print(f"\n🔧 测试配置: {config_name}")
        
        processor = OptimizedBatchProcessor(config)
        
        start_time = time.time()
        results = await processor.process_segments_optimized(
            segments=segments,
            enable_punctuation=True,
            enable_speaker_id=True,
            asr_model=asr_model,
            punctuation_processor=punctuation_processor,
            speaker_extractor=speaker_extractor
        )
        processing_time = time.time() - start_time
        
        stats = processor.get_stats()
        
        print(f"   ⏱️ 耗时: {processing_time:.2f}秒")
        print(f"   📈 并行效率: {stats['parallel_efficiency']:.2%}")
        print(f"   🎯 结果数量: {len(results)}")

async def test_error_handling():
    """测试错误处理"""
    print("\n=== 错误处理测试 ===")
    
    segments = create_test_segments(10)
    
    # 测试缺少模型的情况
    processor = OptimizedBatchProcessor()
    
    print("🧪 测试缺少ASR模型")
    results = await processor.process_segments_optimized(
        segments=segments,
        enable_punctuation=True,
        enable_speaker_id=True,
        asr_model=None,  # 故意传入None
        punctuation_processor=MockPunctuationProcessor(),
        speaker_extractor=MockSpeakerExtractor()
    )
    
    if len(results) == 0:
        print("✅ 正确处理了缺少ASR模型的情况")
    else:
        print("❌ 未正确处理缺少ASR模型的情况")
    
    print("🧪 测试空段落列表")
    results = await processor.process_segments_optimized(
        segments=[],
        enable_punctuation=True,
        enable_speaker_id=True,
        asr_model=MockASRModel(),
        punctuation_processor=MockPunctuationProcessor(),
        speaker_extractor=MockSpeakerExtractor()
    )
    
    if len(results) == 0:
        print("✅ 正确处理了空段落列表")
    else:
        print("❌ 未正确处理空段落列表")

async def main():
    """主测试函数"""
    print("开始优化批次处理器测试...")
    print(f"测试配置:")
    print(f"  - 优化处理: {settings.ENABLE_OPTIMIZED_BATCH_PROCESSING}")
    print(f"  - 并行后处理: {settings.ENABLE_PARALLEL_POST_PROCESSING}")
    print(f"  - 最大批次线程: {settings.MAX_BATCH_THREADS}")
    print(f"  - 标点线程: {settings.PUNCTUATION_THREADS_PER_BATCH}")
    print(f"  - 声纹线程: {settings.SPEAKER_THREADS_PER_BATCH}")
    
    test_results = []
    
    # 运行各项测试
    test_results.append(await test_basic_processing())
    await test_performance_comparison()
    await test_configuration_options()
    await test_error_handling()
    
    # 总结
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n=== 测试总结 ===")
    print(f"通过: {passed}/{total} 项测试")
    
    if passed == total:
        print("🎉 所有批次处理器测试通过！")
    else:
        print(f"❌ {total - passed} 项测试失败")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
