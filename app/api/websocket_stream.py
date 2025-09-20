"""
增强的WebSocket实时语音识别API

提供更好的连接管理、错误处理和监控功能。
"""

import asyncio
import json
import logging
from typing import Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.model import get_model_info
from app.core.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/stream")
async def enhanced_websocket_stream(websocket: WebSocket):
    """
    增强的实时音频流识别 (WebSocket)
    
    协议说明:
    - 客户端发送: {"type": "audio", "data": base64_audio_data}
    - 服务端返回: {"type": "transcription", "text": "识别结果", "timestamp": 1.23, "speaker": "说话人"}
    - 客户端发送: {"type": "end"} 结束会话
    - 服务端返回: {"type": "error", "message": "错误信息"} 错误消息
    - 服务端返回: {"type": "end", "message": "会话结束"} 会话结束
    
    增强功能:
    - 连接管理和统计
    - 更好的错误处理
    - 自动清理不活跃连接
    - 性能监控
    """
    connection_id = None
    manager = await get_websocket_manager()
    
    try:
        # 接受WebSocket连接
        await websocket.accept()
        connection_id = await manager.connect(websocket)
        
        # 检查模型状态
        model_info = get_model_info()
        if not model_info["initialized"]:
            await manager.send_error(connection_id, "ASR模型未初始化，请先初始化模型")
            return
        
        # 发送连接确认消息
        await manager.send_message(connection_id, {
            "type": "connected",
            "message": "WebSocket连接已建立",
            "connection_id": connection_id,
            "model_status": {
                "asr_model": "ok" if model_info["offline_recognizer"] else "unavailable",
                "vad_model": "ok" if model_info["vad"] else "unavailable",
                "speaker_id": "ok" if model_info["speaker_extractor"] else "unavailable"
            }
        })
        
        # 消息处理循环
        while True:
            try:
                # 接收消息（设置超时）
                message_data = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=300.0  # 5分钟超时
                )
                
                try:
                    message = json.loads(message_data)
                except json.JSONDecodeError as e:
                    await manager.send_error(connection_id, f"JSON格式错误: {str(e)}")
                    continue
                
                # 处理不同类型的消息
                message_type = message.get("type", "unknown")
                
                if message_type == "audio":
                    # 处理音频数据
                    if "data" not in message:
                        await manager.send_error(connection_id, "音频消息缺少data字段")
                        continue
                    
                    success = await manager.process_audio_message(connection_id, message)
                    if not success:
                        continue
                
                elif message_type == "end":
                    # 处理会话结束
                    await manager.process_end_message(connection_id)
                    break
                
                elif message_type == "ping":
                    # 处理心跳消息
                    await manager.send_message(connection_id, {
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    })
                
                elif message_type == "get_stats":
                    # 获取连接统计
                    stats = manager.get_connection_stats(connection_id)
                    if stats:
                        await manager.send_message(connection_id, {
                            "type": "stats",
                            "connection_id": connection_id,
                            "messages_received": stats.messages_received,
                            "messages_sent": stats.messages_sent,
                            "audio_chunks_processed": stats.audio_chunks_processed,
                            "recognition_requests": stats.recognition_requests,
                            "errors": stats.errors,
                            "bytes_received": stats.bytes_received,
                            "bytes_sent": stats.bytes_sent,
                            "connected_duration": (stats.last_activity - stats.connected_at).total_seconds()
                        })
                
                else:
                    # 未知消息类型
                    await manager.send_error(connection_id, f"未知消息类型: {message_type}")
                
            except asyncio.TimeoutError:
                # 发送心跳检测
                await manager.send_message(connection_id, {
                    "type": "ping",
                    "message": "心跳检测"
                })
                continue
            
            except WebSocketDisconnect:
                logger.info(f"WebSocket客户端主动断开连接: {connection_id}")
                break
            
            except Exception as e:
                logger.error(f"处理WebSocket消息时出错 {connection_id}: {e}")
                await manager.send_error(connection_id, f"消息处理错误: {str(e)}")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket连接处理出错 {connection_id}: {e}")
        if connection_id:
            try:
                await manager.send_error(connection_id, f"服务器错误: {str(e)}")
            except:
                pass
    finally:
        # 清理连接
        if connection_id:
            await manager.disconnect(connection_id)


@router.get("/stream/stats")
async def get_websocket_stats():
    """
    获取WebSocket连接统计信息
    """
    try:
        manager = await get_websocket_manager()
        stats = manager.get_all_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "service": "websocket-stream",
                "status": "healthy",
                "statistics": stats
            }
        )
    except Exception as e:
        logger.error(f"获取WebSocket统计信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计信息失败: {str(e)}"
        )


@router.get("/stream/health")
async def websocket_health_check():
    """
    WebSocket服务健康检查
    """
    try:
        model_info = get_model_info()
        manager = await get_websocket_manager()
        stats = manager.get_all_stats()
        
        # 判断服务健康状态
        is_healthy = (
            model_info["initialized"] and
            stats["total_errors"] < stats["total_recognitions"] * 0.1  # 错误率低于10%
        )
        
        health_status = {
            "service": "websocket-stream",
            "status": "healthy" if is_healthy else "degraded",
            "model_initialized": model_info["initialized"],
            "active_connections": stats["active_connections"],
            "total_recognitions": stats["total_recognitions"],
            "total_errors": stats["total_errors"],
            "error_rate": stats["total_errors"] / max(1, stats["total_recognitions"]),
            "components": {
                "asr_model": "ok" if model_info["offline_recognizer"] else "unavailable",
                "vad_model": "ok" if model_info["vad"] else "unavailable", 
                "speaker_id": "ok" if model_info["speaker_extractor"] else "unavailable"
            }
        }
        
        status_code = 200 if is_healthy else 503
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
        
    except Exception as e:
        logger.error(f"WebSocket健康检查失败: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "websocket-stream",
                "status": "unhealthy",
                "error": str(e)
            }
        )