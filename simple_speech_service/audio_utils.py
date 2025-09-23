"""
音频处理工具类
基于demo中的音频处理功能简化而来
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import io

from .config import settings

class AudioProcessor:
    """音频处理器"""

    def __init__(self):
        self.supported_formats = settings.SUPPORTED_FORMATS

    def is_supported_format(self, filename: str, content_type: Optional[str] = None) -> bool:
        """检查文件格式是否支持"""
        if not filename:
            return False

        # 检查文件扩展名
        file_ext = Path(filename).suffix.lower().lstrip('.')
        if file_ext not in self.supported_formats:
            return False

        # 如果提供了content_type，也检查
        if content_type:
            # 简单的content-type检查
            audio_types = ['audio/', 'video/']  # 视频文件也可能包含音频
            if not any(content_type.startswith(t) for t in audio_types):
                return False

        return True

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        支持多种格式，自动转换为16kHz单声道
        """
        print(f"正在加载音频文件: {file_path}")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            # 首先尝试使用librosa加载，支持更多格式
            samples, sample_rate = librosa.load(file_path, sr=None, mono=True)
            print("使用librosa加载成功")
        except Exception as e:
            print(f"librosa加载失败: {e}，尝试使用soundfile")
            try:
                samples, sample_rate = sf.read(file_path, dtype='float32')
                # 转换为单声道
                if len(samples.shape) > 1:
                    samples = np.mean(samples, axis=1)
            except Exception as e2:
                raise RuntimeError(f"无法加载音频文件 {file_path}: {e2}")

        # 重采样到16kHz（如果需要）
        if sample_rate != settings.SAMPLE_RATE:
            print(f"原始采样率: {sample_rate}Hz, 重采样到{settings.SAMPLE_RATE}Hz")
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=settings.SAMPLE_RATE)
            sample_rate = settings.SAMPLE_RATE

        duration = len(samples) / sample_rate
        print(f"音频信息: 采样率={sample_rate}Hz, 时长={duration:.2f}秒")
        return samples, sample_rate

    def load_audio_from_bytes(self, audio_data: bytes, filename: str) -> Tuple[np.ndarray, int]:
        """
        从字节数据加载音频
        """
        print(f"正在处理音频数据: {filename}")

        try:
            # 先尝试librosa（支持更多格式）
            audio_io = io.BytesIO(audio_data)
            samples, sample_rate = librosa.load(audio_io, sr=None, mono=True)
            print("使用librosa处理成功")
        except Exception as e:
            print(f"librosa处理失败: {e}，尝试使用soundfile")
            try:
                audio_io = io.BytesIO(audio_data)
                samples, sample_rate = sf.read(audio_io, dtype='float32')
                # 转换为单声道
                if len(samples.shape) > 1:
                    samples = np.mean(samples, axis=1)
            except Exception as e2:
                raise RuntimeError(f"无法处理音频数据 {filename}: {e2}")

        # 重采样到16kHz（如果需要）
        if sample_rate != settings.SAMPLE_RATE:
            print(f"原始采样率: {sample_rate}Hz, 重采样到{settings.SAMPLE_RATE}Hz")
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=settings.SAMPLE_RATE)
            sample_rate = settings.SAMPLE_RATE

        duration = len(samples) / sample_rate
        print(f"音频信息: 采样率={sample_rate}Hz, 时长={duration:.2f}秒")
        return samples, sample_rate

    def convert_and_resample(self, audio_data: bytes, output_sample_rate: int = None) -> bytes:
        """
        转换音频格式并重采样
        简化版本，主要是为了兼容现有接口
        """
        if output_sample_rate is None:
            output_sample_rate = settings.SAMPLE_RATE

        # 加载音频
        samples, sample_rate = self.load_audio_from_bytes(audio_data, "temp")

        # 重采样（如果需要）
        if sample_rate != output_sample_rate:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=output_sample_rate)

        # 转换为WAV格式的字节数据
        output_io = io.BytesIO()
        sf.write(output_io, samples, output_sample_rate, format='wav')
        output_io.seek(0)

        return output_io.read()

# 全局音频处理器实例
audio_processor = AudioProcessor()