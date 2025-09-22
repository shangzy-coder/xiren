#!/usr/bin/env python3
"""
测试简单语音服务
"""
import sys
import os
from pathlib import Path
import requests
import time

# 添加simple_speech_service到Python路径
sys.path.insert(0, str(Path(__file__).parent / "simple_speech_service"))

def test_health():
    """测试健康检查"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ 健康检查通过")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 连接服务失败: {e}")
        return False

def test_root():
    """测试根路径"""
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("✅ 根路径测试通过")
            return True
        else:
            print(f"❌ 根路径测试失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 连接服务失败: {e}")
        return False

def test_speakers_list():
    """测试说话人列表"""
    try:
        response = requests.get("http://localhost:8000/api/speakers")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 说话人列表测试通过，当前有 {data['data']['total']} 个说话人")
            return True
        else:
            print(f"❌ 说话人列表测试失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 连接服务失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试简单语音服务...")
    print("=" * 50)

    # 等待服务启动
    print("等待服务启动...")
    time.sleep(2)

    tests = [
        ("健康检查", test_health),
        ("根路径", test_root),
        ("说话人列表", test_speakers_list),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n测试: {test_name}")
        if test_func():
            passed += 1

    print("\n" + "=" * 50)
    print(f"测试完成: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！服务运行正常")
        return True
    else:
        print("⚠️  部分测试失败，请检查服务状态")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)