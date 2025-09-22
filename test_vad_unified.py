#!/usr/bin/env python3
"""
统一VAD模块测试
测试新的VADProcessor类的功能
"""

import asyncio
import numpy as np
import time
import tempfile
import soundfile as sf
from pathlib import Path

from app.core.vad import VADProcessor, VADConfig, get_vad_processor
from app.config import settings

async def test_vad_config():
    """测试VAD配置"""
    print("=== VAD配置测试 ===")
    
    # 测试默认配置
    try:
        config = VADConfig()
        print("✅ 默认配置创建成功")
        print(f"   - 模型路径: {config.model_path}")
        print(f"   - 阈值: {config.threshold}")
        print(f"   - 最小静音时长: {config.min_silence_duration}")
        print(f"   - 最小语音时长: {config.min_speech_duration}")
        print(f"   - 最大语音时长: {config.max_speech_duration}")
    except Exception as e:
        print(f"❌ 默认配置创建失败: {e}")
        return False
    
    # 测试配置验证
    try:
        invalid_config = VADConfig(threshold=1.5)  # 无效阈值
        print("❌ 配置验证失败 - 应该抛出异常")
        return False
    except ValueError:
        print("✅ 配置验证正常工作")
    
    # 测试配置更新
    try:
        config.update(threshold=0.6, min_speech_duration=0.3)
        print("✅ 配置更新成功")
        print(f"   - 新阈值: {config.threshold}")
        print(f"   - 新最小语音时长: {config.min_speech_duration}")
    except Exception as e:
        print(f"❌ 配置更新失败: {e}")
        return False
    
    return True

async def test_vad_processor_basic():
    """测试VAD处理器基本功能"""
    print("\n=== VAD处理器基本功能测试 ===")
    
    # 创建VAD处理器
    try:
        processor = VADProcessor()
        print("✅ VAD处理器创建成功")
    except Exception as e:
        print(f"❌ VAD处理器创建失败: {e}")
        return False
    
    # 初始化处理器
    try:
        success = await processor.initialize()
        if success:
            print("✅ VAD处理器初始化成功")
        else:
            print("❌ VAD处理器初始化失败")
            return False
    except Exception as e:
        print(f"❌ VAD处理器初始化异常: {e}")
        return False
    
    # 创建测试音频数据
    sample_rate = 16000
    duration = 3.0  # 3秒
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 生成包含语音和静音的音频信号
    audio_data = np.zeros_like(t, dtype=np.float32)
    
    # 添加几个语音段落（正弦波模拟）
    # 段落1: 0.5-1.5秒
    mask1 = (t >= 0.5) & (t <= 1.5)
    audio_data[mask1] = 0.3 * np.sin(2 * np.pi * 440 * t[mask1])  # 440Hz音调
    
    # 段落2: 2.0-2.8秒
    mask2 = (t >= 2.0) & (t <= 2.8)
    audio_data[mask2] = 0.3 * np.sin(2 * np.pi * 880 * t[mask2])  # 880Hz音调
    
    print(f"创建了 {duration}秒 的测试音频，包含2个语音段落")
    
    # 测试语音检测
    try:
        start_time = time.time()
        segments = await processor.detect_speech_segments(audio_data, sample_rate)
        processing_time = time.time() - start_time
        
        print(f"✅ 语音检测完成，耗时: {processing_time:.3f}秒")
        print(f"   - 检测到 {len(segments)} 个语音段落")
        
        for i, segment in enumerate(segments):
            print(f"   - 段落 {i+1}: {segment.start:.2f}s - {segment.end:.2f}s "
                  f"(时长: {segment.duration:.2f}s, 置信度: {segment.confidence})")
        
        if len(segments) > 0:
            print("✅ 语音检测功能正常")
        else:
            print("⚠️ 未检测到语音段落，可能需要调整VAD参数")
    
    except Exception as e:
        print(f"❌ 语音检测失败: {e}")
        return False
    
    # 测试统计信息
    try:
        stats = processor.get_stats()
        print(f"✅ 统计信息获取成功")
        print(f"   - 处理总时长: {stats['total_processed_duration']:.2f}秒")
        print(f"   - 语音总时长: {stats['total_speech_duration']:.2f}秒")
        print(f"   - 语音比例: {stats['speech_ratio']:.2%}")
        print(f"   - 平均处理时间: {stats['average_processing_time']:.3f}秒")
        print(f"   - 模型加载时间: {stats['model_load_time']:.2f}秒")
    except Exception as e:
        print(f"❌ 统计信息获取失败: {e}")
    
    # 关闭处理器
    await processor.close()
    print("✅ VAD处理器已关闭")
    
    return True

async def test_streaming_vad():
    """测试流式VAD处理"""
    print("\n=== 流式VAD处理测试 ===")
    
    processor = VADProcessor()
    await processor.initialize()
    
    sample_rate = 16000
    chunk_duration = 0.5  # 每块0.5秒
    chunk_size = int(sample_rate * chunk_duration)
    total_chunks = 6  # 总共3秒
    
    print(f"模拟流式音频处理，每块 {chunk_duration}秒，共 {total_chunks} 块")
    
    all_segments = []
    
    for i in range(total_chunks):
        # 生成音频块
        t_start = i * chunk_duration
        t_end = (i + 1) * chunk_duration
        t = np.linspace(t_start, t_end, chunk_size)
        
        # 在某些块中添加语音信号
        if 1 <= i <= 3:  # 第2-4块包含语音
            audio_chunk = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        else:
            audio_chunk = np.zeros(chunk_size, dtype=np.float32)
        
        # 处理音频块
        try:
            segments = await processor.process_streaming_audio(audio_chunk, sample_rate)
            all_segments.extend(segments)
            
            if segments:
                print(f"   块 {i+1}: 检测到 {len(segments)} 个语音段落")
            else:
                print(f"   块 {i+1}: 未检测到语音")
                
        except Exception as e:
            print(f"❌ 流式处理失败 (块 {i+1}): {e}")
            await processor.close()
            return False
    
    print(f"✅ 流式处理完成，总共检测到 {len(all_segments)} 个语音段落")
    
    await processor.close()
    return True

async def test_global_vad_processor():
    """测试全局VAD处理器"""
    print("\n=== 全局VAD处理器测试 ===")
    
    try:
        # 获取全局处理器
        processor1 = await get_vad_processor()
        processor2 = await get_vad_processor()
        
        # 应该是同一个实例
        if processor1 is processor2:
            print("✅ 全局VAD处理器单例模式正常")
        else:
            print("❌ 全局VAD处理器单例模式失败")
            return False
        
        # 测试配置更新
        success = processor1.configure(threshold=0.6)
        if success:
            print("✅ 全局处理器配置更新成功")
        else:
            print("❌ 全局处理器配置更新失败")
    
    except Exception as e:
        print(f"❌ 全局VAD处理器测试失败: {e}")
        return False
    
    return True

async def test_vad_performance():
    """测试VAD性能"""
    print("\n=== VAD性能测试 ===")
    
    processor = VADProcessor()
    await processor.initialize()
    
    # 创建较长的测试音频（10秒）
    sample_rate = 16000
    duration = 10.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 生成复杂的音频信号
    audio_data = np.zeros_like(t, dtype=np.float32)
    
    # 添加多个语音段落
    speech_intervals = [(1, 2.5), (3, 4), (5.5, 7), (8, 9.5)]
    for start, end in speech_intervals:
        mask = (t >= start) & (t <= end)
        frequency = 440 + (start * 100)  # 不同频率
        audio_data[mask] = 0.3 * np.sin(2 * np.pi * frequency * t[mask])
    
    print(f"创建了 {duration}秒 的复杂音频，包含 {len(speech_intervals)} 个语音段落")
    
    # 性能测试
    num_runs = 5
    processing_times = []
    
    for i in range(num_runs):
        start_time = time.time()
        segments = await processor.detect_speech_segments(audio_data, sample_rate)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        print(f"   运行 {i+1}: {processing_time:.3f}秒, 检测到 {len(segments)} 个段落")
    
    # 计算性能指标
    avg_time = np.mean(processing_times)
    std_time = np.std(processing_times)
    min_time = np.min(processing_times)
    max_time = np.max(processing_times)
    
    real_time_factor = duration / avg_time
    
    print(f"✅ 性能测试完成:")
    print(f"   - 平均处理时间: {avg_time:.3f} ± {std_time:.3f}秒")
    print(f"   - 最快/最慢: {min_time:.3f}/{max_time:.3f}秒")
    print(f"   - 实时倍数: {real_time_factor:.1f}x")
    
    if real_time_factor > 1:
        print(f"✅ 处理速度超过实时 ({real_time_factor:.1f}倍)")
    else:
        print(f"⚠️ 处理速度低于实时 ({real_time_factor:.1f}倍)")
    
    # 获取最终统计
    stats = processor.get_stats()
    print(f"   - 总处理时长: {stats['total_processed_duration']:.1f}秒")
    print(f"   - 总语音时长: {stats['total_speech_duration']:.1f}秒")
    print(f"   - 错误次数: {stats['error_count']}")
    
    await processor.close()
    return True

async def main():
    """主测试函数"""
    print("开始统一VAD模块测试...")
    print(f"VAD模型路径: {settings.VAD_MODEL_PATH}")
    print(f"模型文件存在: {Path(settings.VAD_MODEL_PATH).exists()}")
    
    test_results = []
    
    # 运行各项测试
    test_results.append(await test_vad_config())
    test_results.append(await test_vad_processor_basic())
    test_results.append(await test_streaming_vad())
    test_results.append(await test_global_vad_processor())
    test_results.append(await test_vad_performance())
    
    # 总结
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n=== 测试总结 ===")
    print(f"通过: {passed}/{total} 项测试")
    
    if passed == total:
        print("🎉 所有VAD模块测试通过！")
    else:
        print(f"❌ {total - passed} 项测试失败")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
