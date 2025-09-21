#!/usr/bin/env python3
"""
音频格式支持测试脚本
"""
import requests
import json
from pathlib import Path

# 测试配置
BASE_URL = "http://localhost:8000"
TEST_FORMATS = [
    ("test.mp3", "audio/mpeg"),
    ("test.wav", "audio/wav"),
    ("test.m4a", "audio/mp4"),
    ("test.flac", "audio/flac"),
    ("test.ogg", "audio/ogg"),
    ("test.amr", "audio/amr"),
    ("test.mpga", "audio/mpeg"),
    ("test.mp4", "video/mp4"),
    ("test.mov", "video/quicktime"),
    ("test.webm", "video/webm"),
    ("test.mpeg", "video/mpeg"),
]

def test_format_validation():
    """测试格式验证"""
    print("🔍 测试格式验证...")
    
    for filename, content_type in TEST_FORMATS:
        # 创建一个虚拟的音频数据（实际上是空数据，只测试格式验证）
        files = {
            'file': (filename, b'fake_audio_data', content_type)
        }
        data = {
            'enable_vad': 'true',
            'enable_speaker_id': 'false'
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/asr/transcribe",
                files=files,
                data=data,
                timeout=10
            )
            
            if response.status_code == 400:
                error_detail = response.json().get('detail', '')
                if "不支持的文件格式" in error_detail:
                    print(f"❌ {filename} ({content_type}): 格式不支持")
                elif "音频文件处理失败" in error_detail:
                    print(f"✅ {filename} ({content_type}): 格式支持，但数据无效（预期）")
                else:
                    print(f"⚠️  {filename} ({content_type}): 其他错误 - {error_detail}")
            elif response.status_code == 200:
                print(f"✅ {filename} ({content_type}): 完全成功")
            else:
                print(f"⚠️  {filename} ({content_type}): HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {filename} ({content_type}): 请求失败 - {e}")

def test_unsupported_formats():
    """测试不支持的格式"""
    print("\n🚫 测试不支持的格式...")
    
    unsupported_formats = [
        ("test.txt", "text/plain"),
        ("test.pdf", "application/pdf"),
        ("test.doc", "application/msword"),
        ("test.xyz", "application/octet-stream"),
    ]
    
    for filename, content_type in unsupported_formats:
        files = {
            'file': (filename, b'fake_data', content_type)
        }
        data = {
            'enable_vad': 'true',
            'enable_speaker_id': 'false'
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/asr/transcribe",
                files=files,
                data=data,
                timeout=10
            )
            
            if response.status_code == 400:
                error_detail = response.json().get('detail', '')
                if "不支持的文件格式" in error_detail:
                    print(f"✅ {filename} ({content_type}): 正确拒绝不支持的格式")
                else:
                    print(f"⚠️  {filename} ({content_type}): 意外错误 - {error_detail}")
            else:
                print(f"❌ {filename} ({content_type}): 应该被拒绝但没有")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {filename} ({content_type}): 请求失败 - {e}")

def test_health_endpoint():
    """测试健康检查端点"""
    print("\n💚 测试健康检查...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ 健康检查成功")
            print(f"   服务状态: {health_data.get('status')}")
            print(f"   模型状态: {health_data.get('model_status', {}).get('is_initialized')}")
            print(f"   支持的模型: ASR={health_data.get('model_status', {}).get('asr_loaded')}, "
                  f"VAD={health_data.get('model_status', {}).get('vad_loaded')}, "
                  f"Speaker={health_data.get('model_status', {}).get('speaker_loaded')}, "
                  f"Punctuation={health_data.get('model_status', {}).get('punctuation_loaded')}")
        else:
            print(f"❌ 健康检查失败: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ 健康检查请求失败: {e}")

if __name__ == "__main__":
    print("🎵 音频格式支持测试")
    print("=" * 50)
    
    # 测试健康检查
    test_health_endpoint()
    
    # 测试支持的格式
    test_format_validation()
    
    # 测试不支持的格式
    test_unsupported_formats()
    
    print("\n" + "=" * 50)
    print("✨ 测试完成")
