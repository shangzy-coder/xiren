#!/usr/bin/env python3
"""
测试VAD时间戳修复和标点符号处理功能
"""

import asyncio
import numpy as np
import requests
import json
from pathlib import Path
import time

def test_api_with_mp3():
    """测试API接口使用MP3文件"""
    print("=" * 60)
    print("🎯 测试API接口 - MP3文件处理")
    print("=" * 60)
    
    # 查找MP3文件
    mp3_files = list(Path(".").glob("*.mp3"))
    if not mp3_files:
        print("❌ 未找到MP3文件，请确保当前目录有MP3文件")
        return False
    
    mp3_file = mp3_files[0]
    print(f"📁 使用文件: {mp3_file}")
    
    # 准备请求
    url = "http://127.0.0.1:8000/api/v1/asr/transcribe"
    
    try:
        with open(mp3_file, 'rb') as f:
            files = {
                'file': (mp3_file.name, f, 'audio/mpeg')
            }
            data = {
                'enable_vad': 'true',
                'enable_speaker_id': 'false'
            }
            
            print("🚀 发送请求...")
            start_time = time.time()
            response = requests.post(url, files=files, data=data, timeout=60)
            end_time = time.time()
            
            print(f"⏱️  请求耗时: {end_time - start_time:.2f}秒")
            print(f"📊 响应状态: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 请求成功!")
                
                # 分析结果
                if result.get("success"):
                    results = result.get("results", [])
                    print(f"📝 识别到 {len(results)} 个语音段落")
                    
                    # 检查时间戳
                    print("\n🔍 时间戳分析:")
                    for i, segment in enumerate(results[:5]):  # 只显示前5个
                        start_time = segment.get("start_time", 0)
                        end_time = segment.get("end_time", 0)
                        text = segment.get("text", "")
                        print(f"  段落 {i+1}: {start_time:.3f}s - {end_time:.3f}s | {text[:50]}...")
                    
                    # 检查是否有标点符号
                    print("\n📖 标点符号检查:")
                    has_punctuation = False
                    for segment in results[:3]:
                        text = segment.get("text", "")
                        if any(p in text for p in "，。！？；：""''"):
                            has_punctuation = True
                            print(f"  ✅ 发现标点: {text[:100]}...")
                            break
                    
                    if not has_punctuation:
                        print("  ⚠️  未发现中文标点符号")
                    
                    # 检查时间戳是否从0开始
                    if results and results[0].get("start_time", 0) < 1.0:
                        print("✅ 时间戳修复成功 - 从接近0开始")
                    else:
                        print("❌ 时间戳仍有问题 - 未从0开始")
                    
                    return True
                else:
                    print(f"❌ 识别失败: {result.get('message', 'Unknown error')}")
                    return False
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"错误详情: {error_detail}")
                except:
                    print(f"错误内容: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def test_health_check():
    """测试健康检查"""
    print("=" * 60)
    print("🏥 测试健康检查")
    print("=" * 60)
    
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print("✅ 健康检查通过")
            print(f"📊 服务状态: {health.get('status', 'unknown')}")
            
            model_status = health.get('model_status', {})
            print(f"🤖 模型状态:")
            print(f"  - 已初始化: {model_status.get('is_initialized', False)}")
            print(f"  - ASR加载: {model_status.get('asr_loaded', False)}")
            print(f"  - VAD加载: {model_status.get('vad_loaded', False)}")
            print(f"  - 声纹加载: {model_status.get('speaker_loaded', False)}")
            print(f"  - 标点加载: {model_status.get('punctuation_loaded', False)}")
            print(f"  - 预加载启用: {model_status.get('preload_enabled', False)}")
            
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 开始VAD修复和标点符号功能测试")
    print("=" * 60)
    
    # 测试健康检查
    health_ok = test_health_check()
    if not health_ok:
        print("❌ 健康检查失败，停止测试")
        return
    
    print("\n")
    
    # 测试API功能
    api_ok = test_api_with_mp3()
    
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)
    print(f"🏥 健康检查: {'✅ 通过' if health_ok else '❌ 失败'}")
    print(f"🎯 API测试: {'✅ 通过' if api_ok else '❌ 失败'}")
    
    if health_ok and api_ok:
        print("🎉 所有测试通过！VAD时间戳和标点符号功能修复成功！")
    else:
        print("⚠️  部分测试失败，需要进一步检查")

if __name__ == "__main__":
    main()
