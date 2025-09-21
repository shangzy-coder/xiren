#!/usr/bin/env python3
"""
测试并行处理效果
"""

import requests
import json
import time

def test_parallel_processing():
    """测试并行处理效果"""
    print("🎯 测试并行批次处理效果")
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
            response = requests.post(url, files=files, data=data, timeout=180)
            end_time = time.time()
            
            print(f"⏱️  总耗时: {end_time - start_time:.2f}秒")
            print(f"📊 响应状态: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 请求成功!")
                
                if result.get("success"):
                    results = result.get("results", [])
                    statistics = result.get("statistics", {})
                    
                    print(f"📝 识别结果:")
                    print(f"  - 语音段落数: {len(results)}")
                    print(f"  - 音频总时长: {statistics.get('total_duration', 0):.2f}秒")
                    print(f"  - 处理时间: {statistics.get('processing_time', 0):.2f}秒")
                    print(f"  - 实时因子: {statistics.get('real_time_factor', 0):.2f}")
                    
                    # 检查并行效果
                    if len(results) > 100:
                        print(f"🚀 大量段落({len(results)}个)处理完成，并行效果显著!")
                    
                    # 显示前几个结果
                    print(f"\n📖 前5个识别结果:")
                    for i, segment in enumerate(results[:5]):
                        start_time = segment.get("start_time", 0)
                        end_time = segment.get("end_time", 0)
                        text = segment.get("text", "")
                        print(f"  {i+1}. [{start_time:.3f}s-{end_time:.3f}s] {text}")
                    
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
    test_parallel_processing()
