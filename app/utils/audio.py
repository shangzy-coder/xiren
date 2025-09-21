"""
音频处理工具模块 - 使用FFmpeg
"""
import subprocess
import tempfile
import os
import asyncio
import time
import threading
from pathlib import Path
from typing import Tuple, Optional, Union
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# 临时文件管理
class TempFileManager:
    """临时文件管理器"""

    def __init__(self, cleanup_interval: int = 3600, max_age: int = 7200):
        self.temp_files = {}  # {file_path: creation_time}
        self.cleanup_interval = cleanup_interval  # 清理间隔（秒）
        self.max_age = max_age  # 文件最大存活时间（秒）
        self.lock = threading.Lock()
        self._cleanup_thread = None
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """启动清理线程"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()

    def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self.cleanup_old_files()
            except Exception as e:
                logger.error(f"临时文件清理失败: {e}")

    def register_temp_file(self, file_path: str):
        """注册临时文件"""
        with self.lock:
            self.temp_files[file_path] = time.time()
            logger.debug(f"注册临时文件: {file_path}")

    def unregister_temp_file(self, file_path: str):
        """注销临时文件"""
        with self.lock:
            if file_path in self.temp_files:
                del self.temp_files[file_path]
                logger.debug(f"注销临时文件: {file_path}")

    def cleanup_old_files(self):
        """清理过期的临时文件"""
        current_time = time.time()
        files_to_remove = []

        with self.lock:
            for file_path, creation_time in self.temp_files.items():
                if current_time - creation_time > self.max_age:
                    files_to_remove.append(file_path)

        for file_path in files_to_remove:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.info(f"清理过期临时文件: {file_path}")
                self.unregister_temp_file(file_path)
            except Exception as e:
                logger.error(f"清理临时文件失败 {file_path}: {e}")

    def safe_remove_file(self, file_path: str):
        """安全删除文件"""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"删除临时文件: {file_path}")
            self.unregister_temp_file(file_path)
        except Exception as e:
            logger.warning(f"删除临时文件失败 {file_path}: {e}")

# 全局临时文件管理器
temp_file_manager = TempFileManager()

# 音频处理异常类
class AudioProcessingError(Exception):
    """音频处理基础异常"""
    pass

class UnsupportedFormatError(AudioProcessingError):
    """不支持的格式异常"""
    pass

class AudioConversionError(AudioProcessingError):
    """音频转换异常"""
    pass

class AudioFileError(AudioProcessingError):
    """音频文件异常"""
    pass

# 性能监控
class AudioProcessingStats:
    """音频处理统计"""

    def __init__(self):
        self.total_processed = 0
        self.total_time = 0.0
        self.total_size = 0
        self.error_count = 0
        self.lock = threading.Lock()

    def record_processing(self, processing_time: float, file_size: int, success: bool = True):
        """记录处理统计"""
        with self.lock:
            self.total_processed += 1
            self.total_time += processing_time
            self.total_size += file_size
            if not success:
                self.error_count += 1

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self.lock:
            if self.total_processed == 0:
                return {
                    "total_processed": 0,
                    "average_time": 0.0,
                    "average_size": 0,
                    "success_rate": 1.0,
                    "throughput": 0.0
                }

            return {
                "total_processed": self.total_processed,
                "average_time": self.total_time / self.total_processed,
                "average_size": self.total_size / self.total_processed,
                "success_rate": (self.total_processed - self.error_count) / self.total_processed,
                "throughput": self.total_size / self.total_time if self.total_time > 0 else 0.0
            }

# 全局统计实例
audio_stats = AudioProcessingStats()

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

    async def convert_and_resample(
        self,
        audio_data: Union[bytes, str],
        output_sample_rate: int = None,
        output_channels: int = 1
    ) -> bytes:
        """
        异步转换和重采样音频数据

        Args:
            audio_data: 音频数据（字节流）或文件路径
            output_sample_rate: 输出采样率，默认使用配置中的采样率
            output_channels: 输出声道数，默认为1（单声道）

        Returns:
            转换后的WAV格式音频数据（字节流）
        """
        if output_sample_rate is None:
            output_sample_rate = settings.SAMPLE_RATE

        start_time = time.time()
        input_size = len(audio_data) if isinstance(audio_data, bytes) else 0

        logger.info(f"开始音频转换: 目标采样率={output_sample_rate}, 声道数={output_channels}, 输入大小={input_size}字节")

        # 验证输入参数
        if not audio_data:
            raise AudioFileError("音频数据为空")

        if isinstance(audio_data, bytes) and len(audio_data) == 0:
            raise AudioFileError("音频数据长度为0")

        # 大文件处理检查
        max_file_size = getattr(settings, 'MAX_AUDIO_FILE_SIZE', 100 * 1024 * 1024)  # 默认100MB
        if input_size > max_file_size:
            raise AudioFileError(f"音频文件过大: {input_size}字节 > {max_file_size}字节")

        # 创建临时文件
        input_file = None
        output_file = None

        try:
            # 处理输入数据
            if isinstance(audio_data, bytes):
                # 如果是字节流，先保存为临时文件
                input_file = tempfile.mktemp(suffix='.tmp')
                temp_file_manager.register_temp_file(input_file)
                with open(input_file, 'wb') as f:
                    f.write(audio_data)
            else:
                # 如果是文件路径，直接使用
                input_file = audio_data

            # 创建输出临时文件
            output_file = tempfile.mktemp(suffix='.wav')
            temp_file_manager.register_temp_file(output_file)

            # 构建FFmpeg命令
            cmd = [
                self.ffmpeg_path,
                '-i', input_file,
                '-ar', str(output_sample_rate),  # 采样率
                '-ac', str(output_channels),     # 声道数
                '-f', 'wav',                     # 输出格式
                '-acodec', 'pcm_f32le',         # 32位浮点PCM
                '-y',                           # 覆盖输出文件
                output_file
            ]

            # 异步执行FFmpeg命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown FFmpeg error"
                logger.error(f"FFmpeg转换失败 (返回码: {process.returncode}): {error_msg}")

                # 根据错误信息分类异常
                if "Invalid data found" in error_msg or "could not find codec" in error_msg:
                    raise UnsupportedFormatError(f"不支持的音频格式或编码: {error_msg}")
                elif "No such file" in error_msg or "Permission denied" in error_msg:
                    raise AudioFileError(f"文件访问错误: {error_msg}")
                else:
                    raise AudioConversionError(f"音频转换失败: {error_msg}")

            # 读取转换后的音频数据
            if not os.path.exists(output_file):
                raise AudioConversionError("转换后的音频文件不存在")

            file_size = os.path.getsize(output_file)
            if file_size == 0:
                raise AudioConversionError("转换后的音频文件为空")

            with open(output_file, 'rb') as f:
                converted_data = f.read()

            processing_time = time.time() - start_time
            logger.info(f"音频转换成功: 采样率={output_sample_rate}, 声道数={output_channels}, 输出大小={len(converted_data)}字节, 耗时={processing_time:.2f}秒")

            # 记录统计信息
            audio_stats.record_processing(processing_time, input_size, True)

            return converted_data

        except AudioProcessingError:
            # 记录失败统计
            processing_time = time.time() - start_time
            audio_stats.record_processing(processing_time, input_size, False)
            # 重新抛出我们自定义的异常
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            audio_stats.record_processing(processing_time, input_size, False)
            logger.error(f"异步音频转换失败: {e}")
            raise AudioConversionError(f"音频转换过程中发生未知错误: {str(e)}")
        finally:
            # 清理临时文件
            if input_file and isinstance(audio_data, bytes):
                temp_file_manager.safe_remove_file(input_file)
            if output_file:
                temp_file_manager.safe_remove_file(output_file)

    def is_supported_format(self, filename: str, content_type: str = None) -> bool:
        """
        检查文件格式是否受支持

        Args:
            filename: 文件名
            content_type: MIME类型

        Returns:
            是否支持该格式
        """
        # 通过文件扩展名检查
        if filename:
            ext = Path(filename).suffix.lower().lstrip('.')
            if ext in settings.SUPPORTED_FORMATS:
                return True

        # 通过MIME类型检查
        if content_type:
            if content_type.startswith('audio/') or content_type.startswith('video/'):
                return True

        return False

    def get_format_from_filename(self, filename: str) -> str:
        """
        从文件名获取格式
        """
        return Path(filename).suffix.lower().lstrip('.')

    def get_processing_stats(self) -> dict:
        """
        获取音频处理统计信息
        """
        return audio_stats.get_stats()

# 全局实例
audio_processor = AudioProcessor()
