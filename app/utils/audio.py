"""
音频处理工具模块 - 使用FFmpeg
"""
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional
import logging

from app.config import settings

logger = logging.getLogger(__name__)

class AudioProcessor:
    """音频处理器，基于FFmpeg"""
    
    def __init__(self):
        self.ffmpeg_path = settings.FFMPEG_PATH
        self.supported_formats = settings.AUDIO_FORMATS
        
    def convert_to_wav(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        转换音频文件为WAV格式
        """
        if output_file is None:
            output_file = tempfile.mktemp(suffix='.wav')
            
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', input_file,
                '-ar', str(settings.SAMPLE_RATE),  # 采样率
                '-ac', '1',  # 单声道
                '-f', 'wav',
                '-y',  # 覆盖输出文件
                output_file
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            logger.info(f"音频转换成功: {input_file} -> {output_file}")
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg转换失败: {e.stderr}")
            raise RuntimeError(f"音频转换失败: {e.stderr}")
    
    def get_audio_info(self, input_file: str) -> dict:
        """
        获取音频文件信息
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                input_file
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            import json
            info = json.loads(result.stdout)
            
            # 提取音频流信息
            audio_stream = None
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                raise ValueError("未找到音频流")
            
            return {
                'duration': float(info['format'].get('duration', 0)),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
                'codec': audio_stream.get('codec_name', 'unknown'),
                'bitrate': int(info['format'].get('bit_rate', 0))
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"获取音频信息失败: {e.stderr}")
            raise RuntimeError(f"获取音频信息失败: {e.stderr}")
    
    def split_audio(self, input_file: str, start_time: float, duration: float, output_file: Optional[str] = None) -> str:
        """
        分割音频文件
        """
        if output_file is None:
            output_file = tempfile.mktemp(suffix='.wav')
            
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', input_file,
                '-ss', str(start_time),
                '-t', str(duration),
                '-ar', str(settings.SAMPLE_RATE),
                '-ac', '1',
                '-f', 'wav',
                '-y',
                output_file
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"音频分割成功: {start_time}s-{start_time+duration}s")
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"音频分割失败: {e.stderr}")
            raise RuntimeError(f"音频分割失败: {e.stderr}")
    
    def validate_audio_file(self, file_path: str) -> bool:
        """
        验证音频文件是否有效
        """
        try:
            info = self.get_audio_info(file_path)
            return info['duration'] > 0 and info['sample_rate'] > 0
        except:
            return False

# 全局实例
audio_processor = AudioProcessor()
