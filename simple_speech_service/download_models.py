#!/usr/bin/env python3
"""
下载语音识别模型
"""
import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import shutil

from config import settings

def download_file(url, dest_path):
    """下载文件"""
    print(f"下载 {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print("下载完成")

def extract_zip(zip_path, extract_to):
    """解压ZIP文件"""
    print(f"解压 {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("解压完成")

def extract_tar(tar_path, extract_to):
    """解压TAR文件"""
    print(f"解压 {tar_path} -> {extract_to}")
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)
    print("解压完成")

def download_asr_model():
    """下载ASR模型"""
    print("下载SenseVoice ASR模型...")

    # SenseVoice模型
    model_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
    model_file = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"

    # 下载
    download_file(model_url, model_file)

    # 解压
    extract_tar(model_file, ".")

    # 移动到正确位置
    model_dir = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    if os.path.exists(model_dir):
        shutil.move(model_dir, settings.MODELS_DIR)

    # 清理
    os.remove(model_file)
    print("ASR模型下载完成")

def download_speaker_model():
    """下载说话人识别模型"""
    print("下载说话人识别模型...")

    model_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
    model_file = os.path.join(settings.MODELS_DIR, "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")

    # 确保目录存在
    os.makedirs(settings.MODELS_DIR, exist_ok=True)

    # 下载
    download_file(model_url, model_file)
    print("说话人识别模型下载完成")

def download_vad_model():
    """下载VAD模型"""
    print("下载VAD模型...")

    model_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    model_file = os.path.join(settings.MODELS_DIR, "silero_vad.onnx")

    # 确保目录存在
    os.makedirs(settings.MODELS_DIR, exist_ok=True)

    # 下载
    download_file(model_url, model_file)
    print("VAD模型下载完成")

def download_punctuation_model():
    """下载标点模型"""
    print("下载标点符号模型...")

    model_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2"
    model_file = "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2"

    # 下载
    download_file(model_url, model_file)

    # 解压
    extract_tar(model_file, ".")

    # 移动到正确位置
    model_dir = "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"
    if os.path.exists(model_dir):
        shutil.move(model_dir, settings.MODELS_DIR)

    # 清理
    os.remove(model_file)
    print("标点符号模型下载完成")

def main():
    """主函数"""
    print("开始下载语音识别模型...")
    print(f"模型将下载到: {settings.MODELS_DIR}")

    # 确保目录存在
    os.makedirs(settings.MODELS_DIR, exist_ok=True)

    try:
        # 下载各个模型
        download_asr_model()
        download_speaker_model()
        download_vad_model()
        download_punctuation_model()

        print("\n所有模型下载完成！")
        print("现在可以运行 python main.py 启动服务")

    except Exception as e:
        print(f"下载过程中出错: {e}")
        print("请检查网络连接并重试")

if __name__ == "__main__":
    main()