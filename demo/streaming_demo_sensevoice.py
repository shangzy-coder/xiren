#!/usr/bin/env python3
"""
基于SenseVoice模型的伪流式语音识别演示

由于SenseVoice是离线模型，此演示通过VAD分段实现伪流式效果，
提供接近实时的语音识别体验。
"""

import argparse
import queue
import threading
import time
import numpy as np
import pyaudio
import sherpa_onnx


class StreamingSenseVoice:
    def __init__(self, model_dir, use_gpu=False, sample_rate=16000):
        """
        初始化伪流式SenseVoice系统
        
        Args:
            model_dir: SenseVoice模型目录路径
            use_gpu: 是否使用GPU
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        self.use_gpu = use_gpu
        
        # 创建离线识别器配置
        recognizer_config = sherpa_onnx.OfflineRecognizerConfig(
            feat_config=sherpa_onnx.FeatureExtractorConfig(
                sampling_rate=sample_rate,
                feature_dim=80
            ),
            model_config=sherpa_onnx.OfflineModelConfig(
                sense_voice=sherpa_onnx.OfflineSenseVoiceModelConfig(
                    model=f"{model_dir}/model.int8.onnx",
                    language="auto",
                    use_itn=True
                ),
                tokens=f"{model_dir}/tokens.txt",
                num_threads=4,
                provider="cuda" if use_gpu else "cpu",
                model_type="sense_voice"
            )
        )
        
        # 创建识别器
        self.recognizer = sherpa_onnx.OfflineRecognizer(recognizer_config)
        
        # 创建VAD配置
        vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model="models/silero_vad.onnx",
                threshold=0.5,
                min_silence_duration=0.25,
                min_speech_duration=0.25,
                window_size=512
            ),
            sample_rate=sample_rate,
            num_threads=2,
            provider="cpu"
        )
        
        # 创建VAD
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
        
        # 音频缓冲区和队列
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        self.is_recording = False
        self.buffer_lock = threading.Lock()
        
        print(f"SenseVoice模型已加载: {model_dir}")
        print(f"使用设备: {'CUDA' if use_gpu else 'CPU'}")
        print(f"采样率: {sample_rate}Hz")
        print("=" * 50)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio回调函数"""
        if self.is_recording:
            # 将音频数据转换为float32并放入队列
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
    
    def find_microphone_device(self):
        """自动找到最佳的麦克风设备"""
        audio = pyaudio.PyAudio()
        
        print("扫描可用的音频设备...")
        device_count = audio.get_device_count()
        best_device = None
        best_score = -1
        
        for i in range(device_count):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:  # 输入设备
                score = 0
                name = info['name'].lower()
                
                # 优先选择支持16kHz的设备
                if info['defaultSampleRate'] == 16000:
                    score += 100
                elif info['defaultSampleRate'] == 48000:
                    score += 50
                
                # 避免虚拟设备
                if 'pulse' not in name and 'virtual' not in name:
                    score += 20
                
                # 优先选择USB或内置麦克风
                if any(keyword in name for keyword in ['usb', 'microphone', 'mic']):
                    score += 10
                
                print(f"设备 {i}: {info['name']} (采样率: {info['defaultSampleRate']}, 得分: {score})")
                
                if score > best_score:
                    best_score = score
                    best_device = i
        
        audio.terminate()
        
        if best_device is not None:
            print(f"选择设备: {best_device}")
            return best_device
        else:
            print("未找到合适的麦克风设备，使用默认设备")
            return None
    
    def start_recording(self, device_id=None):
        """开始录音"""
        if device_id is None:
            device_id = self.find_microphone_device()
        
        # 初始化PyAudio
        self.audio = pyaudio.PyAudio()
        
        # 获取设备信息
        if device_id is not None:
            device_info = self.audio.get_device_info_by_index(device_id)
            input_sample_rate = int(device_info['defaultSampleRate'])
            print(f"使用设备: {device_info['name']} (采样率: {input_sample_rate}Hz)")
        else:
            input_sample_rate = 16000
            print("使用默认设备")
        
        # 打开音频流
        self.audio_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=input_sample_rate,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=1024,
            stream_callback=self.audio_callback
        )
        
        self.input_sample_rate = input_sample_rate
        self.is_recording = True
        self.audio_stream.start_stream()
        
        print("开始录音...")
        print("说话时会进行实时VAD检测和语音识别，按Ctrl+C结束")
        print("-" * 50)
    
    def resample_audio(self, audio_data, input_rate, output_rate):
        """音频重采样"""
        if input_rate == output_rate:
            return audio_data
        
        # 简单的线性插值重采样
        input_length = len(audio_data)
        output_length = int(input_length * output_rate / input_rate)
        
        indices = np.linspace(0, input_length - 1, output_length)
        resampled = np.interp(indices, np.arange(input_length), audio_data)
        
        return resampled.astype(np.float32)
    
    def recognize_segment(self, audio_segment):
        """识别音频段落"""
        try:
            # 创建离线流
            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, audio_segment)
            
            # 执行识别
            self.recognizer.decode_stream(stream)
            result = stream.result
            
            return {
                'text': result.text,
                'language': getattr(result, 'lang', 'unknown'),
                'emotion': getattr(result, 'emotion', 'unknown')
            }
        except Exception as e:
            print(f"识别出错: {e}")
            return {'text': '', 'language': 'unknown', 'emotion': 'unknown'}
    
    def process_audio(self):
        """处理音频数据的主循环"""
        print("🎤 正在监听...")
        
        while self.is_recording:
            try:
                # 从队列获取音频数据
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # 重采样到16kHz（如果需要）
                if self.input_sample_rate != self.sample_rate:
                    audio_chunk = self.resample_audio(
                        audio_chunk, self.input_sample_rate, self.sample_rate
                    )
                
                # 添加到缓冲区
                with self.buffer_lock:
                    self.audio_buffer.extend(audio_chunk)
                
                # 发送到VAD
                self.vad.accept_waveform(audio_chunk)
                
                # 检查VAD结果
                if self.vad.is_speech_detected():
                    print("🔊 检测到语音...", end="\r", flush=True)
                
                # 检查是否有完整的语音段落
                if not self.vad.empty():
                    speech_segment = self.vad.front()
                    
                    # 使用缓冲区的音频进行识别
                    with self.buffer_lock:
                        if len(self.audio_buffer) > 8000:  # 至少0.5秒的音频
                            audio_array = np.array(self.audio_buffer, dtype=np.float32)
                            
                            # 清理输出并显示处理中
                            print(" " * 50, end="\r")
                            print("🔄 正在识别...", end="", flush=True)
                            
                            # 识别音频段落
                            result = self.recognize_segment(audio_array)
                            
                            # 显示结果
                            if result['text'].strip():
                                print(f"\r✓ 识别结果: {result['text']}")
                                if result['language'] != 'unknown':
                                    print(f"  语言: {result['language']}")
                                if result['emotion'] != 'unknown':
                                    print(f"  情感: {result['emotion']}")
                                print("-" * 30)
                            else:
                                print("\r❌ 未识别到有效语音")
                            
                            # 清空缓冲区，保留最近的一小部分
                            keep_samples = int(self.sample_rate * 0.5)  # 保留0.5秒
                            if len(self.audio_buffer) > keep_samples:
                                self.audio_buffer = self.audio_buffer[-keep_samples:]
                    
                    # 弹出已处理的语音段落
                    self.vad.pop()
                    
                    print("🎤 继续监听...")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n处理音频时出错: {e}")
                break
    
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if hasattr(self, 'audio'):
            self.audio.terminate()
        
        print("\n录音已停止")
    
    def run_interactive(self, device_id=None):
        """运行交互式识别"""
        try:
            # 开始录音
            self.start_recording(device_id)
            
            # 启动音频处理线程
            processing_thread = threading.Thread(target=self.process_audio)
            processing_thread.daemon = True
            processing_thread.start()
            
            # 等待用户中断
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n正在退出...")
        finally:
            self.stop_recording()


def main():
    parser = argparse.ArgumentParser(
        description="基于SenseVoice的伪流式语音识别演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python streaming_demo_sensevoice.py
  python streaming_demo_sensevoice.py --model models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
  python streaming_demo_sensevoice.py --use-gpu --device 0
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        help="SenseVoice模型目录路径"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="使用GPU加速"
    )
    
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="指定音频设备ID"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="音频采样率 (默认: 16000)"
    )
    
    args = parser.parse_args()
    
    print("基于SenseVoice的伪流式语音识别演示")
    print("=" * 45)
    
    try:
        # 创建伪流式ASR系统
        asr = StreamingSenseVoice(
            model_dir=args.model,
            use_gpu=args.use_gpu,
            sample_rate=args.sample_rate
        )
        
        # 运行交互式识别
        asr.run_interactive(device_id=args.device)
        
    except FileNotFoundError as e:
        print(f"模型文件未找到: {e}")
        print("请确保SenseVoice模型路径正确")
    except Exception as e:
        print(f"启动失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

