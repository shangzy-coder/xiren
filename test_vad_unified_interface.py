#!/usr/bin/env python3
"""
VAD统一接口测试脚本
验证VAD作为内部服务组件的功能
"""

import asyncio
import logging
import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from app.core.vad import VADProcessor, VADConfig, get_vad_processor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_audio(duration: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """生成测试音频数据"""
    # 生成包含语音和静音的测试音频
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)
    
    # 添加一些"语音"段落（简单的正弦波）
    speech_start1 = int(0.2 * sample_rate)  # 0.2s开始
    speech_end1 = int(0.8 * sample_rate)    # 0.8s结束
    speech_start2 = int(1.2 * sample_rate)  # 1.2s开始
    speech_end2 = int(1.8 * sample_rate)    # 1.8s结束
    
    # 生成正弦波作为"语音"
    freq = 440  # 440Hz
    t1 = np.linspace(0, (speech_end1 - speech_start1) / sample_rate, speech_end1 - speech_start1)
    t2 = np.linspace(0, (speech_end2 - speech_start2) / sample_rate, speech_end2 - speech_start2)
    
    audio[speech_start1:speech_end1] = 0.5 * np.sin(2 * np.pi * freq * t1)
    audio[speech_start2:speech_end2] = 0.3 * np.sin(2 * np.pi * freq * t2)
    
    return audio


async def test_vad_basic_functionality():
    """测试VAD基础功能"""
    print("\n=== 测试VAD基础功能 ===")
    
    # 创建VAD处理器
    vad_processor = VADProcessor()
    
    try:
        # 初始化
        success = await vad_processor.initialize()
        if not success:
            print("❌ VAD初始化失败")
            return False
        print("✅ VAD初始化成功")
        
        # 生成测试音频
        test_audio = generate_test_audio(duration=2.0)
        print(f"✅ 生成测试音频: {len(test_audio)}个样本, {len(test_audio)/16000:.1f}秒")
        
        # 测试语音段落检测
        segments = await vad_processor.detect_speech_segments(test_audio)
        print(f"✅ 检测到 {len(segments)} 个语音段落:")
        for i, segment in enumerate(segments):
            print(f"   段落{i+1}: {segment.start:.2f}s - {segment.end:.2f}s ({segment.duration:.2f}s)")
        
        # 测试语音活动检测
        is_active = await vad_processor.is_speech_active(test_audio)
        print(f"✅ 语音活动检测: {'有语音' if is_active else '无语音'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        await vad_processor.close()


async def test_vad_streaming():
    """测试VAD流式处理"""
    print("\n=== 测试VAD流式处理 ===")
    
    vad_processor = VADProcessor()
    
    try:
        await vad_processor.initialize()
        
        # 模拟流式音频块
        chunk_duration = 0.5  # 每块0.5秒
        chunk_size = int(chunk_duration * 16000)
        total_chunks = 4
        
        all_segments = []
        
        for i in range(total_chunks):
            # 生成音频块
            chunk = generate_test_audio(duration=chunk_duration)
            
            # 处理流式音频
            segments = await vad_processor.process_streaming_audio(chunk, return_samples=False)
            
            print(f"✅ 处理音频块{i+1}: 检测到 {len(segments)} 个段落")
            for segment in segments:
                print(f"   段落: {segment.start:.2f}s - {segment.end:.2f}s ({segment.duration:.2f}s)")
            
            all_segments.extend(segments)
        
        # 重置流式状态
        vad_processor.reset_streaming_state()
        print("✅ 流式状态已重置")
        
        print(f"✅ 流式处理完成: 总共检测到 {len(all_segments)} 个段落")
        return True
        
    except Exception as e:
        print(f"❌ 流式测试失败: {e}")
        return False
    finally:
        await vad_processor.close()


async def test_vad_configuration():
    """测试VAD配置管理"""
    print("\n=== 测试VAD配置管理 ===")
    
    try:
        # 创建自定义配置
        config = VADConfig(
            threshold=0.6,
            min_speech_duration=0.3,
            max_speech_duration=4.0
        )
        
        vad_processor = VADProcessor(config)
        await vad_processor.initialize()
        
        print("✅ 自定义配置创建成功")
        
        # 测试动态配置更新
        success = vad_processor.configure(
            threshold=0.7,
            min_speech_duration=0.4
        )
        
        if success:
            print("✅ 动态配置更新成功")
        else:
            print("❌ 动态配置更新失败")
            return False
        
        # 测试配置验证
        try:
            vad_processor.configure(threshold=1.5)  # 无效值
            print("❌ 配置验证失败，应该拒绝无效值")
            return False
        except ValueError:
            print("✅ 配置验证正常，正确拒绝无效值")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False
    finally:
        await vad_processor.close()


async def test_vad_statistics():
    """测试VAD统计功能"""
    print("\n=== 测试VAD统计功能 ===")
    
    vad_processor = VADProcessor()
    
    try:
        await vad_processor.initialize()
        
        # 处理一些音频以生成统计数据
        test_audio = generate_test_audio(duration=3.0)
        await vad_processor.detect_speech_segments(test_audio)
        
        # 获取统计信息
        stats = vad_processor.get_stats()
        
        print("✅ 获取统计信息成功:")
        print(f"   服务状态: 初始化={stats['service_status']['initialized']}, 模型加载={stats['service_status']['model_loaded']}")
        print(f"   处理统计: 总时长={stats['total_processed_duration']:.2f}s, 段落数={stats['total_segments']}")
        print(f"   性能指标: 语音比例={stats['performance_metrics']['speech_ratio']:.2%}")
        print(f"   资源使用: 内存={stats['resource_usage']['memory_usage_mb']:.1f}MB")
        
        # 重置统计
        vad_processor.reset_stats()
        stats_after_reset = vad_processor.get_stats()
        
        if stats_after_reset['total_processed_duration'] == 0:
            print("✅ 统计重置成功")
        else:
            print("❌ 统计重置失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 统计测试失败: {e}")
        return False
    finally:
        await vad_processor.close()


async def test_global_vad_processor():
    """测试全局VAD处理器"""
    print("\n=== 测试全局VAD处理器 ===")
    
    try:
        # 获取全局实例
        vad1 = await get_vad_processor()
        vad2 = await get_vad_processor()
        
        # 验证是同一个实例
        if vad1 is vad2:
            print("✅ 全局VAD处理器单例模式正常")
        else:
            print("❌ 全局VAD处理器单例模式失败")
            return False
        
        # 测试功能
        test_audio = generate_test_audio(duration=1.0)
        segments = await vad1.detect_speech_segments(test_audio)
        
        print(f"✅ 全局VAD处理器功能正常: 检测到 {len(segments)} 个段落")
        return True
        
    except Exception as e:
        print(f"❌ 全局处理器测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("=" * 60)
    print("VAD统一接口测试")
    print("=" * 60)
    
    tests = [
        ("基础功能", test_vad_basic_functionality),
        ("流式处理", test_vad_streaming),
        ("配置管理", test_vad_configuration),
        ("统计功能", test_vad_statistics),
        ("全局处理器", test_global_vad_processor)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n开始测试: {test_name}")
        try:
            success = await test_func()
            if success:
                print(f"✅ {test_name} 测试通过")
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 60)
    
    if passed == total:
        print("🎉 所有测试通过！VAD统一接口验证成功！")
    else:
        print("⚠️  部分测试失败，请检查VAD实现")


if __name__ == "__main__":
    asyncio.run(main())
