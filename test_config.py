#!/usr/bin/env python3
"""
测试不同配置的性能效果
"""

import os
import requests
import json
import time
import subprocess

def test_with_config(config_name, env_vars):
    """使用指定配置测试性能"""
    print(f"\n🔧 测试配置: {config_name}")
    print("=" * 50)
    
    # 设置环境变量
    for key, value in env_vars.items():
        os.environ[key] = str(value)
        print(f"  {key}={value}")
    
    print("\n🚀 重启服务...")
    
    # 重启服务 (这里只是示例，实际需要重启uvicorn)
    # 在实际使用中，你需要重启服务来应用新的环境变量
    
    # 等待服务启动
    time.sleep(5)
    
    # 测试请求
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
            
            print("📤 发送测试请求...")
            start_time = time.time()
            response = requests.post(url, files=files, data=data, timeout=180)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    results = result.get("results", [])
                    statistics = result.get("statistics", {})
                    
                    print(f"✅ 测试成功!")
                    print(f"  - 总耗时: {end_time - start_time:.2f}秒")
                    print(f"  - 段落数: {len(results)}")
                    print(f"  - 处理时间: {statistics.get('processing_time', 0):.2f}秒")
                    print(f"  - 实时因子: {statistics.get('real_time_factor', 0):.3f}")
                    
                    return {
                        'config': config_name,
                        'total_time': end_time - start_time,
                        'segments': len(results),
                        'processing_time': statistics.get('processing_time', 0),
                        'rtf': statistics.get('real_time_factor', 0)
                    }
                else:
                    print(f"❌ 识别失败: {result}")
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                
    except Exception as e:
        print(f"❌ 测试异常: {e}")
    
    return None

def main():
    """主测试函数"""
    print("🎯 性能配置对比测试")
    print("=" * 60)
    
    # 测试配置列表
    test_configs = [
        {
            'name': '保守配置 (2线程)',
            'env': {
                'MAX_BATCH_THREADS': 2,
                'MIN_BATCH_SIZE': 30,
                'MAX_BATCH_SIZE': 60,
                'ASR_THREADS_PER_BATCH': 1,
                'PUNCTUATION_THREADS_PER_BATCH': 1
            }
        },
        {
            'name': '平衡配置 (4线程)',
            'env': {
                'MAX_BATCH_THREADS': 4,
                'MIN_BATCH_SIZE': 20,
                'MAX_BATCH_SIZE': 80,
                'ASR_THREADS_PER_BATCH': 2,
                'PUNCTUATION_THREADS_PER_BATCH': 2
            }
        },
        {
            'name': '激进配置 (6线程)',
            'env': {
                'MAX_BATCH_THREADS': 6,
                'MIN_BATCH_SIZE': 15,
                'MAX_BATCH_SIZE': 100,
                'ASR_THREADS_PER_BATCH': 2,
                'PUNCTUATION_THREADS_PER_BATCH': 2
            }
        }
    ]
    
    results = []
    
    print("⚠️  注意: 此脚本仅展示配置参数，实际测试需要重启服务")
    print("请手动设置环境变量并重启服务进行测试\n")
    
    for config in test_configs:
        print(f"📋 {config['name']} 配置:")
        for key, value in config['env'].items():
            print(f"  export {key}={value}")
        print()
    
    print("🔧 推荐的测试步骤:")
    print("1. 设置环境变量")
    print("2. 重启服务: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    print("3. 运行测试: python test_parallel.py")
    print("4. 记录性能数据")
    print("5. 重复步骤1-4测试不同配置")

if __name__ == "__main__":
    main()
