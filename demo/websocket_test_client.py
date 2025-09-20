#!/usr/bin/env python3
"""
WebSocket 实时语音识别测试客户端

测试与语音识别服务的WebSocket连接，发送音频数据并接收识别结果。
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import argparse
import time
import sys
import os
from pathlib import Path

# 添加项目根目录到路径，以便导入项目模块
sys.path.append(str(Path(__file__).parent.parent))


class WebSocketTestClient:
    def __init__(self, server_url="ws://localhost:8002/api/v1/asr/stream"):
        """
        初始化WebSocket测试客户端
        
        Args:
            server_url: WebSocket服务器地址
        """
        self.server_url = server_url
        self.websocket = None
        
    async def connect(self):
        """建立WebSocket连接"""
        try:
            print(f"正在连接到 {self.server_url}...")
            self.websocket = await websockets.connect(self.server_url)
            print("✅ WebSocket连接已建立")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    async def disconnect(self):
        """断开WebSocket连接"""
        if self.websocket:
            await self.websocket.close()
            print("🔌 WebSocket连接已断开")
    
    async def send_audio_chunk(self, audio_data):
        """
        发送音频数据块
        
        Args:
            audio_data: numpy数组格式的音频数据 (float32)
        """
        if not self.websocket:
            raise Exception("WebSocket未连接")
        
        # 将音频数据编码为base64
        audio_bytes = audio_data.astype(np.float32).tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # 构造消息
        message = {
            "type": "audio",
            "data": audio_base64
        }
        
        # 发送消息
        await self.websocket.send(json.dumps(message))
        print(f"📤 已发送音频数据块 ({len(audio_data)} 样本)")
    
    async def send_end_signal(self):
        """发送结束信号"""
        if not self.websocket:
            raise Exception("WebSocket未连接")
        
        message = {"type": "end"}
        await self.websocket.send(json.dumps(message))
        print("🔚 已发送结束信号")
    
    async def receive_messages(self):
        """接收并处理服务器消息"""
        if not self.websocket:
            raise Exception("WebSocket未连接")
        
        print("👂 开始监听服务器消息...")
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(data)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON解码错误: {e}")
                except Exception as e:
                    print(f"❌ 消息处理错误: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("🔌 服务器关闭了连接")
        except Exception as e:
            print(f"❌ 接收消息时出错: {e}")
    
    async def handle_message(self, data):
        """处理接收到的消息"""
        message_type = data.get("type", "unknown")
        
        if message_type == "transcription":
            # 识别结果
            text = data.get("text", "")
            timestamp = data.get("timestamp", 0)
            speaker = data.get("speaker", "unknown")
            language = data.get("language", "unknown")
            emotion = data.get("emotion", "unknown")
            
            print(f"🎯 识别结果: {text}")
            print(f"   时间戳: {timestamp:.2f}s")
            print(f"   说话人: {speaker}")
            print(f"   语言: {language}")
            print(f"   情感: {emotion}")
            print("-" * 50)
            
        elif message_type == "error":
            # 错误消息
            error_msg = data.get("message", "未知错误")
            print(f"❌ 服务器错误: {error_msg}")
            
        elif message_type == "end":
            # 会话结束
            end_msg = data.get("message", "会话结束")
            print(f"🏁 {end_msg}")
            
        else:
            # 未知消息类型
            print(f"❓ 未知消息类型: {message_type}")
            print(f"   数据: {data}")
    
    async def test_with_audio_file(self, audio_file_path, chunk_duration=2.0, sample_rate=16000):
        """
        使用音频文件测试WebSocket
        
        Args:
            audio_file_path: 音频文件路径
            chunk_duration: 每个音频块的持续时间(秒)
            sample_rate: 音频采样率
        """
        print(f"📁 加载音频文件: {audio_file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(audio_file_path):
            print(f"❌ 音频文件不存在: {audio_file_path}")
            return
        
        try:
            # 这里简化处理，假设是WAV文件
            # 在实际应用中，应该使用librosa或soundfile来正确加载音频
            import wave
            
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate_file = wav_file.getframerate()
                audio_data = wav_file.readframes(frames)
                
                # 转换为float32格式
                if wav_file.getsampwidth() == 2:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    print(f"❌ 不支持的音频格式，采样位数: {wav_file.getsampwidth()}")
                    return
                
                print(f"✅ 音频文件加载成功:")
                print(f"   采样率: {sample_rate_file}Hz")
                print(f"   时长: {len(audio_array) / sample_rate_file:.2f}秒")
                print(f"   样本数: {len(audio_array)}")
                
                # 重采样到目标采样率(简化版本)
                if sample_rate_file != sample_rate:
                    print(f"🔄 重采样从 {sample_rate_file}Hz 到 {sample_rate}Hz")
                    # 简单的线性插值重采样
                    target_length = int(len(audio_array) * sample_rate / sample_rate_file)
                    indices = np.linspace(0, len(audio_array) - 1, target_length)
                    audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)
                
                # 分块发送音频
                chunk_samples = int(sample_rate * chunk_duration)
                total_chunks = (len(audio_array) + chunk_samples - 1) // chunk_samples
                
                print(f"📦 将音频分为 {total_chunks} 个块，每块 {chunk_duration}秒")
                print("🚀 开始发送音频数据...")
                
                for i in range(total_chunks):
                    start_idx = i * chunk_samples
                    end_idx = min(start_idx + chunk_samples, len(audio_array))
                    chunk = audio_array[start_idx:end_idx]
                    
                    await self.send_audio_chunk(chunk)
                    
                    # 模拟实时发送，等待一段时间
                    await asyncio.sleep(chunk_duration * 0.8)  # 稍微快一点以避免缓冲区积压
                
                # 发送结束信号
                await self.send_end_signal()
                
                # 等待处理完成
                print("⏳ 等待服务器处理完成...")
                await asyncio.sleep(2.0)
                
        except Exception as e:
            print(f"❌ 音频文件处理错误: {e}")
    
    async def test_with_synthetic_audio(self, duration=10.0, sample_rate=16000):
        """
        使用合成音频测试WebSocket
        
        Args:
            duration: 音频持续时间(秒)
            sample_rate: 采样率
        """
        print(f"🎵 生成 {duration}秒 的合成音频数据 (采样率: {sample_rate}Hz)")
        
        # 生成正弦波测试音频
        total_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, total_samples, False)
        
        # 混合多个频率的正弦波
        frequency1 = 440.0  # A4音符
        frequency2 = 554.37  # C#5音符
        audio_data = (np.sin(2 * np.pi * frequency1 * t) * 0.3 + 
                     np.sin(2 * np.pi * frequency2 * t) * 0.2).astype(np.float32)
        
        # 分块发送
        chunk_duration = 2.0
        chunk_samples = int(sample_rate * chunk_duration)
        total_chunks = (len(audio_data) + chunk_samples - 1) // chunk_samples
        
        print(f"📦 将音频分为 {total_chunks} 个块，每块 {chunk_duration}秒")
        print("🚀 开始发送合成音频数据...")
        
        for i in range(total_chunks):
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, len(audio_data))
            chunk = audio_data[start_idx:end_idx]
            
            await self.send_audio_chunk(chunk)
            
            # 模拟实时发送
            await asyncio.sleep(chunk_duration * 0.8)
        
        # 发送结束信号
        await self.send_end_signal()
        
        # 等待处理完成
        print("⏳ 等待服务器处理完成...")
        await asyncio.sleep(2.0)
    
    async def run_test(self, test_type="synthetic", audio_file=None, duration=10.0):
        """
        运行WebSocket测试
        
        Args:
            test_type: 测试类型 ("synthetic" 或 "file")
            audio_file: 音频文件路径 (test_type="file"时使用)
            duration: 合成音频持续时间 (test_type="synthetic"时使用)
        """
        print("🧪 WebSocket 语音识别测试客户端")
        print("=" * 50)
        
        # 建立连接
        if not await self.connect():
            return
        
        try:
            # 启动消息接收任务
            receive_task = asyncio.create_task(self.receive_messages())
            
            # 等待一小段时间确保连接稳定
            await asyncio.sleep(1.0)
            
            # 根据测试类型运行不同的测试
            if test_type == "file" and audio_file:
                await self.test_with_audio_file(audio_file)
            else:
                await self.test_with_synthetic_audio(duration)
            
            # 等待接收完所有消息
            await asyncio.sleep(3.0)
            
            # 取消接收任务
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
            
        except KeyboardInterrupt:
            print("\n⏹️  用户中断测试")
        except Exception as e:
            print(f"❌ 测试过程中出错: {e}")
        finally:
            await self.disconnect()
        
        print("✅ 测试完成")


def main():
    parser = argparse.ArgumentParser(
        description="WebSocket 语音识别测试客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python websocket_test_client.py
  python websocket_test_client.py --server ws://localhost:8002/api/v1/asr/stream
  python websocket_test_client.py --type file --audio test.wav
  python websocket_test_client.py --type synthetic --duration 15
        """
    )
    
    parser.add_argument(
        "--server",
        type=str,
        default="ws://localhost:8002/api/v1/asr/stream",
        help="WebSocket服务器地址"
    )
    
    parser.add_argument(
        "--type",
        choices=["synthetic", "file"],
        default="synthetic",
        help="测试类型: synthetic(合成音频) 或 file(音频文件)"
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        help="音频文件路径 (仅当type=file时使用)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="合成音频持续时间，单位秒 (仅当type=synthetic时使用)"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.type == "file" and not args.audio:
        print("❌ 使用文件测试时必须指定 --audio 参数")
        return 1
    
    try:
        # 创建客户端并运行测试
        client = WebSocketTestClient(args.server)
        
        # 运行异步测试
        asyncio.run(client.run_test(
            test_type=args.type,
            audio_file=args.audio,
            duration=args.duration
        ))
        
    except Exception as e:
        print(f"❌ 客户端启动失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())