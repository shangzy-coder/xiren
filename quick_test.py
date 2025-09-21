#!/usr/bin/env python3
"""
快速测试修复后的功能
"""

import requests
import json
import time

def test_with_mp3():
    """使用test.mp3测试"""
    print("🎯 测试VAD时间戳修复和标点符号功能")
    print("=" * 50)
    
    url = "http://127.0.0.1:8000/api/v1/asr/transcribe"
    
    try:
        with open("test.mp3", 'rb') as f:
            files = {
                'file': ('test.mp3', f, 'audio/mpeg')
            }
            data = {
                'enable_vad': 'true',
                'enable_speaker_id': 'false'
            }
            
            print("🚀 发送请求...")
            start_time = time.time()
            response = requests.post(url, files=files, data=data, timeout=120)
            end_time = time.time()
            
            print(f"⏱️  请求耗时: {end_time - start_time:.2f}秒")
            print(f"📊 响应状态: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 请求成功!")
                
                if result.get("success"):
                    results = result.get("results", [])
                    print(f"📝 识别到 {len(results)} 个语音段落")
                    
                    # 检查时间戳
                    print("\n🔍 时间戳分析:")
                    for i, segment in enumerate(results[:3]):
                        start_time = segment.get("start_time", 0)
                        end_time = segment.get("end_time", 0)
                        text = segment.get("text", "")
                        print(f"  段落 {i+1}: {start_time:.3f}s - {end_time:.3f}s")
                        print(f"    文本: {text}")
                    
                    # 检查第一个段落的时间戳
                    if results:
                        first_start = results[0].get("start_time", 0)
                        if first_start < 1.0:
                            print("✅ 时间戳修复成功 - 从接近0开始")
                        else:
                            print(f"❌ 时间戳仍有问题 - 从{first_start:.3f}秒开始")
                    
                    # 检查标点符号
                    print("\n📖 标点符号检查:")
                    has_punctuation = False
                    for segment in results:
                        text = segment.get("text", "")
                        if any(p in text for p in "，。！？；：""''"):
                            has_punctuation = True
                            print(f"  ✅ 发现标点: {text}")
                            break
                    
                    if not has_punctuation:
                        print("  ⚠️  未发现中文标点符号")
                    
                    return True
                else:
                    print(f"❌ 识别失败: {result}")
                    return False
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                print(f"错误内容: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

if __name__ == "__main__":
    test_with_mp3()
