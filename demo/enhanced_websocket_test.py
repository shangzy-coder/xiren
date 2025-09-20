#!/usr/bin/env python3
"""
增强的WebSocket实时语音识别测试客户端

测试新的增强WebSocket API端点，包括连接管理、统计信息和心跳检测等功能。
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

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))


class EnhancedWebSocketTestClient:
    def __init__(self, server_url="ws://localhost:8002/api/v1/websocket/stream"):
        """
        初始化增强的WebSocket测试客户端
        
        Args:
            server_url: WebSocket服务器地址
        """
        self.server_url = server_url
        self.websocket = None
        self.connection_id = None
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "audio_chunks_sent": 0,
            "transcriptions_received": 0,
            "errors_received": 0
        }
        
    async def connect(self):
        """建立WebSocket连接"""
        try:
            print(f"正在连接到 {self.server_url}...")
            self.websocket = await websockets.connect(self.server_url)
            print("✅ WebSocket连接已建立")
            
            # 等待连接确认消息
            initial_message = await self.websocket.recv()
            data = json.loads(initial_message)
            
            if data.get("type") == "connected":
                self.connection_id = data.get("connection_id")
                model_status = data.get("model_status", {})
                print(f"🔗 连接ID: {self.connection_id}")
                print(f"🤖 模型状态: ASR={model_status.get('asr_model', 'unknown')}, "
                      f"VAD={model_status.get('vad_model', 'unknown')}, "
                      f"Speaker={model_status.get('speaker_id', 'unknown')}")
                return True
            else:
                print(f"❌ 意外的初始消息: {data}")
                return False
                
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    async def disconnect(self):
        """断开WebSocket连接"""
        if self.websocket:
            await self.websocket.close()
            print("🔌 WebSocket连接已断开")
    
    async def send_message(self, message):
        """发送消息"""
        if not self.websocket:
            raise Exception("WebSocket未连接")
        
        await self.websocket.send(json.dumps(message))
        self.stats["messages_sent"] += 1
    
    async def send_audio_chunk(self, audio_data):
        """发送音频数据块"""
        audio_bytes = audio_data.astype(np.float32).tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        message = {
            "type": "audio",
            "data": audio_base64
        }
        
        await self.send_message(message)
        self.stats["audio_chunks_sent"] += 1
        print(f"📤 已发送音频数据块 ({len(audio_data)} 样本)")
    
    async def send_end_signal(self):
        """发送结束信号"""
        message = {"type": "end"}
        await self.send_message(message)
        print("🔚 已发送结束信号")
    
    async def send_ping(self):
        """发送心跳"""
        message = {
            "type": "ping",
            "timestamp": time.time()
        }
        await self.send_message(message)
        print("💓 已发送心跳")
    
    async def request_stats(self):
        """请求连接统计信息"""
        message = {"type": "get_stats"}
        await self.send_message(message)
        print("📊 已请求统计信息")
    
    async def receive_messages(self):
        """接收并处理服务器消息"""
        if not self.websocket:
            raise Exception("WebSocket未连接")
        
        print("👂 开始监听服务器消息...")
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    self.stats["messages_received"] += 1
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
            self.stats["transcriptions_received"] += 1
            text = data.get("text", "")
            timestamp = data.get("timestamp", 0)
            speaker = data.get("speaker", "unknown")
            language = data.get("language", "unknown")
            emotion = data.get("emotion", "unknown")
            confidence = data.get("confidence", 0.0)
            
            print(f"🎯 识别结果: {text}")
            print(f"   时间戳: {timestamp:.2f}s, 说话人: {speaker}")
            print(f"   语言: {language}, 情感: {emotion}, 置信度: {confidence:.2f}")
            print("-" * 50)
            
        elif message_type == "error":
            # 错误消息
            self.stats["errors_received"] += 1
            error_msg = data.get("message", "未知错误")
            print(f"❌ 服务器错误: {error_msg}")
            
        elif message_type == "end":
            # 会话结束
            end_msg = data.get("message", "会话结束")
            print(f"🏁 {end_msg}")
            
        elif message_type == "pong":
            # 心跳响应
            timestamp = data.get("timestamp")
            if timestamp:
                latency = time.time() - timestamp
                print(f"💗 心跳响应 (延迟: {latency*1000:.1f}ms)")
            else:
                print("💗 心跳响应")
                
        elif message_type == "stats":
            # 统计信息
            print("📊 服务器端统计信息:")
            print(f"   连接ID: {data.get('connection_id', 'unknown')}")
            print(f"   接收消息: {data.get('messages_received', 0)}")
            print(f"   发送消息: {data.get('messages_sent', 0)}")
            print(f"   音频块处理: {data.get('audio_chunks_processed', 0)}")
            print(f"   识别请求: {data.get('recognition_requests', 0)}")
            print(f"   错误次数: {data.get('errors', 0)}")
            print(f"   接收字节: {data.get('bytes_received', 0)}")
            print(f"   发送字节: {data.get('bytes_sent', 0)}")
            print(f"   连接时长: {data.get('connected_duration', 0):.1f}秒")
            print("-" * 50)
            
        else:
            # 未知消息类型
            print(f"❓ 未知消息类型: {message_type}")
            print(f"   数据: {data}")
    
    async def test_with_synthetic_audio(self, duration=10.0, sample_rate=16000, chunk_duration=2.0):
        """使用合成音频测试"""
        print(f"🎵 生成 {duration}秒 的合成音频数据 (采样率: {sample_rate}Hz)")
        
        # 生成正弦波测试音频
        total_samples = int(duration * sample_rate)
        chunk_samples = int(sample_rate * chunk_duration)
        total_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        
        print(f"📦 将音频分为 {total_chunks} 个块，每块 {chunk_duration}秒")
        print("🚀 开始发送合成音频数据...")
        
        for i in range(total_chunks):
            start_sample = i * chunk_samples
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk_length = end_sample - start_sample
            
            # 生成音频块
            t_start = start_sample / sample_rate
            t = np.linspace(t_start, t_start + chunk_length/sample_rate, chunk_length, False)
            
            # 混合多个频率的正弦波
            frequency1 = 440.0  # A4音符
            frequency2 = 554.37  # C#5音符
            audio_chunk = (np.sin(2 * np.pi * frequency1 * t) * 0.3 + 
                          np.sin(2 * np.pi * frequency2 * t) * 0.2).astype(np.float32)
            
            await self.send_audio_chunk(audio_chunk)
            
            # 模拟实时发送
            await asyncio.sleep(chunk_duration * 0.8)
        
        print("🎵 合成音频发送完成")
    
    async def interactive_test(self):
        """交互式测试"""
        print("🎮 进入交互式测试模式")
        print("命令:")
        print("  ping - 发送心跳")
        print("  stats - 请求统计信息")
        print("  audio <duration> - 发送合成音频")
        print("  end - 结束会话")
        print("  quit - 退出测试")
        print("-" * 50)
        
        while True:
            try:
                command = input("请输入命令: ").strip().lower()
                
                if command == "quit":
                    break
                elif command == "ping":
                    await self.send_ping()
                elif command == "stats":
                    await self.request_stats()
                elif command.startswith("audio"):
                    parts = command.split()
                    duration = float(parts[1]) if len(parts) > 1 else 5.0
                    await self.test_with_synthetic_audio(duration=duration)
                elif command == "end":
                    await self.send_end_signal()
                    break
                else:
                    print(f"未知命令: {command}")
                    
            except KeyboardInterrupt:
                print("\n⏹️  用户中断")
                break
            except Exception as e:
                print(f"❌ 命令执行错误: {e}")
    
    def print_client_stats(self):
        """打印客户端统计信息"""
        print("📊 客户端统计信息:")
        print(f"   发送消息: {self.stats['messages_sent']}")
        print(f"   接收消息: {self.stats['messages_received']}")
        print(f"   音频块发送: {self.stats['audio_chunks_sent']}")
        print(f"   识别结果接收: {self.stats['transcriptions_received']}")
        print(f"   错误接收: {self.stats['errors_received']}")
    
    async def run_test(self, test_mode="synthetic", duration=10.0):
        """运行测试"""
        print("🧪 增强WebSocket语音识别测试客户端")
        print("=" * 60)
        
        # 建立连接
        if not await self.connect():
            return
        
        try:
            # 启动消息接收任务
            receive_task = asyncio.create_task(self.receive_messages())
            
            # 等待连接稳定
            await asyncio.sleep(1.0)
            
            # 根据测试模式运行
            if test_mode == "interactive":
                await self.interactive_test()
            elif test_mode == "synthetic":
                await self.test_with_synthetic_audio(duration)
                await asyncio.sleep(2.0)
                await self.send_end_signal()
            
            # 等待处理完成
            await asyncio.sleep(3.0)
            
            # 请求最终统计信息
            await self.request_stats()
            await asyncio.sleep(1.0)
            
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
            self.print_client_stats()
        
        print("✅ 测试完成")


def main():
    parser = argparse.ArgumentParser(
        description="增强WebSocket语音识别测试客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python enhanced_websocket_test.py
  python enhanced_websocket_test.py --server ws://localhost:8002/api/v1/websocket/stream
  python enhanced_websocket_test.py --mode interactive
  python enhanced_websocket_test.py --mode synthetic --duration 15
        """
    )
    
    parser.add_argument(
        "--server",
        type=str,
        default="ws://localhost:8002/api/v1/websocket/stream",
        help="WebSocket服务器地址"
    )
    
    parser.add_argument(
        "--mode",
        choices=["synthetic", "interactive"],
        default="synthetic",
        help="测试模式: synthetic(合成音频) 或 interactive(交互式)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="合成音频持续时间，单位秒"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建客户端并运行测试
        client = EnhancedWebSocketTestClient(args.server)
        
        # 运行异步测试
        asyncio.run(client.run_test(
            test_mode=args.mode,
            duration=args.duration
        ))
        
    except Exception as e:
        print(f"❌ 客户端启动失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())