"""
语音处理流水线系统

将语音处理分解为独立的队列任务：VAD、ASR、Speaker识别等
每个处理步骤都是独立的任务，通过流水线编排器协调执行
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from app.config import settings
from app.core.queue import TaskType, TaskPriority, get_queue_manager
from app.core.request_manager import get_request_manager

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """流水线阶段"""
    PREPROCESSING = "preprocessing"    # 音频预处理
    VAD = "vad"                       # 语音活动检测
    ASR = "asr"                       # 语音识别
    SPEAKER_EMBEDDING = "speaker_embedding"  # 声纹特征提取
    SPEAKER_IDENTIFICATION = "speaker_identification"  # 声纹识别
    SPEAKER_DIARIZATION = "speaker_diarization"  # 说话人分离
    POSTPROCESSING = "postprocessing"  # 后处理
    STORAGE = "storage"               # 结果存储


@dataclass
class PipelineData:
    """流水线数据容器"""
    pipeline_id: str
    session_id: str
    audio_data: bytes
    filename: str
    
    # 处理配置
    enable_vad: bool = True
    enable_asr: bool = True
    enable_speaker_id: bool = True
    enable_diarization: bool = False
    sample_rate: int = 16000
    
    # 中间结果
    processed_audio: Optional[np.ndarray] = None
    vad_segments: Optional[List[Dict]] = None
    asr_results: Optional[List[Dict]] = None
    speaker_embeddings: Optional[List[np.ndarray]] = None
    speaker_identifications: Optional[List[str]] = None
    diarization_results: Optional[Dict] = None
    
    # 最终结果
    final_results: Optional[Dict] = None
    
    # 元数据
    created_at: float = 0
    stages_completed: List[str] = None
    stages_failed: List[str] = None
    total_processing_time: float = 0
    
    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()
        if self.stages_completed is None:
            self.stages_completed = []
        if self.stages_failed is None:
            self.stages_failed = []


class PipelineOrchestrator:
    """流水线编排器"""
    
    def __init__(self):
        self._active_pipelines: Dict[str, PipelineData] = {}
        self._request_manager = None
        
    async def initialize(self):
        """初始化编排器"""
        self._request_manager = await get_request_manager()
        logger.info("流水线编排器初始化完成")
    
    async def submit_pipeline(self,
                            audio_data: bytes,
                            filename: str,
                            enable_vad: bool = True,
                            enable_asr: bool = True,
                            enable_speaker_id: bool = True,
                            enable_diarization: bool = False,
                            sample_rate: int = 16000,
                            priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        提交音频处理流水线
        
        Returns:
            pipeline_id: 流水线ID
        """
        pipeline_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # 创建流水线数据
        pipeline_data = PipelineData(
            pipeline_id=pipeline_id,
            session_id=session_id,
            audio_data=audio_data,
            filename=filename,
            enable_vad=enable_vad,
            enable_asr=enable_asr,
            enable_speaker_id=enable_speaker_id,
            enable_diarization=enable_diarization,
            sample_rate=sample_rate
        )
        
        self._active_pipelines[pipeline_id] = pipeline_data
        
        # 开始执行流水线
        await self._execute_pipeline(pipeline_data, priority)
        
        logger.info(f"流水线已提交: {pipeline_id}, 文件: {filename}")
        return pipeline_id
    
    async def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict]:
        """获取流水线状态"""
        if pipeline_id not in self._active_pipelines:
            return None
        
        pipeline_data = self._active_pipelines[pipeline_id]
        return {
            "pipeline_id": pipeline_id,
            "session_id": pipeline_data.session_id,
            "filename": pipeline_data.filename,
            "stages_completed": pipeline_data.stages_completed,
            "stages_failed": pipeline_data.stages_failed,
            "total_processing_time": pipeline_data.total_processing_time,
            "is_completed": pipeline_data.final_results is not None,
            "created_at": pipeline_data.created_at
        }
    
    async def get_pipeline_result(self, pipeline_id: str, timeout: float = 300) -> Optional[Dict]:
        """获取流水线结果（阻塞等待）"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if pipeline_id in self._active_pipelines:
                pipeline_data = self._active_pipelines[pipeline_id]
                if pipeline_data.final_results is not None:
                    return pipeline_data.final_results
                elif pipeline_data.stages_failed:
                    return {
                        "success": False,
                        "error": f"流水线执行失败，失败阶段: {pipeline_data.stages_failed}",
                        "pipeline_id": pipeline_id
                    }
            
            await asyncio.sleep(0.1)
        
        raise asyncio.TimeoutError(f"等待流水线结果超时: {pipeline_id}")
    
    async def _execute_pipeline(self, pipeline_data: PipelineData, priority: TaskPriority):
        """执行流水线"""
        try:
            # 1. 音频预处理任务
            await self._submit_preprocessing_task(pipeline_data, priority)
            
        except Exception as e:
            logger.error(f"流水线执行失败: {pipeline_data.pipeline_id}, 错误: {e}")
            pipeline_data.stages_failed.append("pipeline_orchestration")
    
    async def _submit_preprocessing_task(self, pipeline_data: PipelineData, priority: TaskPriority):
        """提交音频预处理任务"""
        task_id = await self._request_manager.submit_asr_request(
            func=_process_audio_preprocessing,
            args=(pipeline_data.pipeline_id, pipeline_data.audio_data, pipeline_data.sample_rate),
            priority=priority,
            timeout=60
        )
        
        logger.info(f"音频预处理任务已提交: {task_id}, 流水线: {pipeline_data.pipeline_id}")
    
    async def _continue_pipeline(self, pipeline_id: str, stage: PipelineStage):
        """继续执行流水线的下一个阶段"""
        if pipeline_id not in self._active_pipelines:
            logger.error(f"流水线不存在: {pipeline_id}")
            return
        
        pipeline_data = self._active_pipelines[pipeline_id]
        
        try:
            if stage == PipelineStage.PREPROCESSING:
                # 预处理完成，开始VAD
                if pipeline_data.enable_vad:
                    await self._submit_vad_task(pipeline_data)
                elif pipeline_data.enable_asr:
                    await self._submit_asr_task(pipeline_data)
                elif pipeline_data.enable_speaker_id:
                    await self._submit_speaker_task(pipeline_data)
                else:
                    await self._finalize_pipeline(pipeline_data)
                    
            elif stage == PipelineStage.VAD:
                # VAD完成，开始ASR
                if pipeline_data.enable_asr:
                    await self._submit_asr_task(pipeline_data)
                elif pipeline_data.enable_speaker_id:
                    await self._submit_speaker_task(pipeline_data)
                else:
                    await self._finalize_pipeline(pipeline_data)
                    
            elif stage == PipelineStage.ASR:
                # ASR完成，开始Speaker处理
                if pipeline_data.enable_speaker_id:
                    await self._submit_speaker_task(pipeline_data)
                else:
                    await self._finalize_pipeline(pipeline_data)
                    
            elif stage == PipelineStage.SPEAKER_IDENTIFICATION:
                # Speaker识别完成，检查是否需要分离
                if pipeline_data.enable_diarization:
                    await self._submit_diarization_task(pipeline_data)
                else:
                    await self._finalize_pipeline(pipeline_data)
                    
            elif stage == PipelineStage.SPEAKER_DIARIZATION:
                # 所有处理完成
                await self._finalize_pipeline(pipeline_data)
                
        except Exception as e:
            logger.error(f"流水线阶段执行失败: {pipeline_id}, 阶段: {stage.value}, 错误: {e}")
            pipeline_data.stages_failed.append(stage.value)
    
    async def _submit_vad_task(self, pipeline_data: PipelineData):
        """提交VAD任务"""
        task_id = await self._request_manager.submit_asr_request(
            func=_process_vad,
            args=(pipeline_data.pipeline_id, pipeline_data.processed_audio, pipeline_data.sample_rate),
            priority=TaskPriority.NORMAL,
            timeout=120
        )
        logger.info(f"VAD任务已提交: {task_id}, 流水线: {pipeline_data.pipeline_id}")
    
    async def _submit_asr_task(self, pipeline_data: PipelineData):
        """提交ASR任务"""
        # 如果有VAD结果，使用分段音频；否则使用全部音频
        audio_segments = pipeline_data.vad_segments if pipeline_data.vad_segments else [{"audio": pipeline_data.processed_audio}]
        
        task_id = await self._request_manager.submit_asr_request(
            func=_process_asr,
            args=(pipeline_data.pipeline_id, audio_segments, pipeline_data.sample_rate),
            priority=TaskPriority.NORMAL,
            timeout=300
        )
        logger.info(f"ASR任务已提交: {task_id}, 流水线: {pipeline_data.pipeline_id}")
    
    async def _submit_speaker_task(self, pipeline_data: PipelineData):
        """提交声纹识别任务"""
        # 使用ASR结果中的音频段，或者原始音频
        audio_segments = []
        if pipeline_data.asr_results:
            for result in pipeline_data.asr_results:
                if "audio_segment" in result:
                    audio_segments.append(result["audio_segment"])
        
        if not audio_segments:
            audio_segments = [pipeline_data.processed_audio]
        
        task_id = await self._request_manager.submit_speaker_request(
            func=_process_speaker_identification,
            args=(pipeline_data.pipeline_id, audio_segments, pipeline_data.sample_rate),
            priority=TaskPriority.NORMAL,
            timeout=180
        )
        logger.info(f"声纹识别任务已提交: {task_id}, 流水线: {pipeline_data.pipeline_id}")
    
    async def _submit_diarization_task(self, pipeline_data: PipelineData):
        """提交说话人分离任务"""
        task_id = await self._request_manager.submit_speaker_request(
            func=_process_speaker_diarization,
            args=(pipeline_data.pipeline_id, pipeline_data.processed_audio, pipeline_data.sample_rate),
            priority=TaskPriority.LOW,  # 分离任务优先级较低
            timeout=600
        )
        logger.info(f"说话人分离任务已提交: {task_id}, 流水线: {pipeline_data.pipeline_id}")
    
    async def _finalize_pipeline(self, pipeline_data: PipelineData):
        """完成流水线处理"""
        try:
            # 汇总所有结果
            final_results = {
                "success": True,
                "pipeline_id": pipeline_data.pipeline_id,
                "session_id": pipeline_data.session_id,
                "filename": pipeline_data.filename,
                "processing_time": time.time() - pipeline_data.created_at,
                "stages_completed": pipeline_data.stages_completed,
                "results": {}
            }
            
            # 添加各阶段结果
            if pipeline_data.vad_segments:
                final_results["results"]["vad"] = pipeline_data.vad_segments
            
            if pipeline_data.asr_results:
                final_results["results"]["transcription"] = pipeline_data.asr_results
            
            if pipeline_data.speaker_identifications:
                final_results["results"]["speaker_identification"] = pipeline_data.speaker_identifications
            
            if pipeline_data.diarization_results:
                final_results["results"]["diarization"] = pipeline_data.diarization_results
            
            pipeline_data.final_results = final_results
            pipeline_data.total_processing_time = final_results["processing_time"]
            
            logger.info(f"流水线处理完成: {pipeline_data.pipeline_id}, 耗时: {final_results['processing_time']:.2f}秒")
            
        except Exception as e:
            logger.error(f"流水线最终化失败: {pipeline_data.pipeline_id}, 错误: {e}")
            pipeline_data.stages_failed.append("finalization")


# 全局流水线编排器实例
_pipeline_orchestrator: Optional[PipelineOrchestrator] = None


async def get_pipeline_orchestrator() -> PipelineOrchestrator:
    """获取流水线编排器实例"""
    global _pipeline_orchestrator
    if _pipeline_orchestrator is None:
        _pipeline_orchestrator = PipelineOrchestrator()
        await _pipeline_orchestrator.initialize()
    return _pipeline_orchestrator


# ============================================================================
# 各个处理阶段的任务函数
# ============================================================================

async def _process_audio_preprocessing(pipeline_id: str, audio_data: bytes, sample_rate: int) -> Dict[str, Any]:
    """音频预处理任务"""
    try:
        from app.utils.audio import AudioProcessor
        
        orchestrator = await get_pipeline_orchestrator()
        pipeline_data = orchestrator._active_pipelines[pipeline_id]
        
        # 音频预处理
        audio_processor = AudioProcessor()
        processed_audio = await audio_processor.convert_and_resample(
            audio_data, 
            output_sample_rate=sample_rate
        )
        
        # 转换为numpy数组
        audio_array = np.frombuffer(processed_audio, dtype=np.float32)
        pipeline_data.processed_audio = audio_array
        pipeline_data.stages_completed.append(PipelineStage.PREPROCESSING.value)
        
        # 继续下一阶段
        await orchestrator._continue_pipeline(pipeline_id, PipelineStage.PREPROCESSING)
        
        return {
            "success": True,
            "pipeline_id": pipeline_id,
            "stage": PipelineStage.PREPROCESSING.value,
            "audio_duration": len(audio_array) / sample_rate
        }
        
    except Exception as e:
        logger.error(f"音频预处理失败: {pipeline_id}, 错误: {e}")
        return {
            "success": False,
            "pipeline_id": pipeline_id,
            "stage": PipelineStage.PREPROCESSING.value,
            "error": str(e)
        }


async def _process_vad(pipeline_id: str, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """VAD语音活动检测任务"""
    try:
        from app.core.vad import VADProcessor
        
        orchestrator = await get_pipeline_orchestrator()
        pipeline_data = orchestrator._active_pipelines[pipeline_id]
        
        # VAD处理
        vad_processor = VADProcessor()
        segments = await vad_processor.detect_speech_segments(audio_data, sample_rate)
        
        # 提取语音段音频
        vad_segments = []
        for segment in segments:
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            vad_segments.append({
                "start": segment.start,
                "end": segment.end,
                "duration": segment.duration,
                "audio": segment_audio,
                "confidence": segment.confidence
            })
        
        pipeline_data.vad_segments = vad_segments
        pipeline_data.stages_completed.append(PipelineStage.VAD.value)
        
        # 继续下一阶段
        await orchestrator._continue_pipeline(pipeline_id, PipelineStage.VAD)
        
        return {
            "success": True,
            "pipeline_id": pipeline_id,
            "stage": PipelineStage.VAD.value,
            "segments_count": len(vad_segments),
            "total_speech_duration": sum(seg["duration"] for seg in vad_segments)
        }
        
    except Exception as e:
        logger.error(f"VAD处理失败: {pipeline_id}, 错误: {e}")
        return {
            "success": False,
            "pipeline_id": pipeline_id,
            "stage": PipelineStage.VAD.value,
            "error": str(e)
        }


async def _process_asr(pipeline_id: str, audio_segments: List[Dict], sample_rate: int) -> Dict[str, Any]:
    """ASR语音识别任务"""
    try:
        from app.core.model import recognize_audio
        
        orchestrator = await get_pipeline_orchestrator()
        pipeline_data = orchestrator._active_pipelines[pipeline_id]
        
        asr_results = []
        
        for i, segment in enumerate(audio_segments):
            audio_data = segment.get("audio", segment)
            if not isinstance(audio_data, np.ndarray):
                continue
            
            # 执行语音识别
            result = await recognize_audio(
                audio_data=audio_data,
                sample_rate=sample_rate,
                enable_vad=False,  # 已经做过VAD了
                enable_speaker_id=False,  # 在后续阶段处理
                enable_punctuation=True  # 默认启用标点符号
            )
            
            if result["success"]:
                for text_result in result.get("results", []):
                    asr_results.append({
                        "segment_index": i,
                        "text": text_result.get("text", ""),
                        "confidence": text_result.get("confidence", 0.0),
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                        "audio_segment": audio_data
                    })
        
        pipeline_data.asr_results = asr_results
        pipeline_data.stages_completed.append(PipelineStage.ASR.value)
        
        # 继续下一阶段
        await orchestrator._continue_pipeline(pipeline_id, PipelineStage.ASR)
        
        return {
            "success": True,
            "pipeline_id": pipeline_id,
            "stage": PipelineStage.ASR.value,
            "transcription_count": len(asr_results),
            "total_text": " ".join(result["text"] for result in asr_results)
        }
        
    except Exception as e:
        logger.error(f"ASR处理失败: {pipeline_id}, 错误: {e}")
        return {
            "success": False,
            "pipeline_id": pipeline_id,
            "stage": PipelineStage.ASR.value,
            "error": str(e)
        }


async def _process_speaker_identification(pipeline_id: str, audio_segments: List[np.ndarray], sample_rate: int) -> Dict[str, Any]:
    """声纹识别任务"""
    try:
        from app.core.speaker_pool import SpeakerPool
        
        orchestrator = await get_pipeline_orchestrator()
        pipeline_data = orchestrator._active_pipelines[pipeline_id]
        
        # 初始化声纹池
        speaker_pool = SpeakerPool()
        await speaker_pool.initialize()
        
        speaker_identifications = []
        
        for i, audio_segment in enumerate(audio_segments):
            if not isinstance(audio_segment, np.ndarray):
                continue
            
            # 声纹识别
            identification_result = await speaker_pool.identify_speaker(
                audio_segment, 
                sample_rate,
                threshold=settings.SPEAKER_SIMILARITY_THRESHOLD
            )
            
            if identification_result:
                speaker_name, similarity = identification_result
                speaker_identifications.append({
                    "segment_index": i,
                    "speaker": speaker_name,
                    "similarity": similarity,
                    "confidence": similarity
                })
            else:
                speaker_identifications.append({
                    "segment_index": i,
                    "speaker": "unknown",
                    "similarity": 0.0,
                    "confidence": 0.0
                })
        
        pipeline_data.speaker_identifications = speaker_identifications
        pipeline_data.stages_completed.append(PipelineStage.SPEAKER_IDENTIFICATION.value)
        
        # 继续下一阶段
        await orchestrator._continue_pipeline(pipeline_id, PipelineStage.SPEAKER_IDENTIFICATION)
        
        return {
            "success": True,
            "pipeline_id": pipeline_id,
            "stage": PipelineStage.SPEAKER_IDENTIFICATION.value,
            "identified_segments": len(speaker_identifications),
            "unique_speakers": len(set(id["speaker"] for id in speaker_identifications if id["speaker"] != "unknown"))
        }
        
    except Exception as e:
        logger.error(f"声纹识别失败: {pipeline_id}, 错误: {e}")
        return {
            "success": False,
            "pipeline_id": pipeline_id,
            "stage": PipelineStage.SPEAKER_IDENTIFICATION.value,
            "error": str(e)
        }


async def _process_speaker_diarization(pipeline_id: str, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """说话人分离任务"""
    try:
        from app.core.speaker_pool import SpeakerPool
        
        orchestrator = await get_pipeline_orchestrator()
        pipeline_data = orchestrator._active_pipelines[pipeline_id]
        
        # 初始化声纹池
        speaker_pool = SpeakerPool()
        await speaker_pool.initialize()
        
        # 执行说话人分离
        diarization_result = await speaker_pool.diarize_speakers(audio_data, sample_rate)
        
        pipeline_data.diarization_results = diarization_result
        pipeline_data.stages_completed.append(PipelineStage.SPEAKER_DIARIZATION.value)
        
        # 继续下一阶段（完成）
        await orchestrator._continue_pipeline(pipeline_id, PipelineStage.SPEAKER_DIARIZATION)
        
        return {
            "success": True,
            "pipeline_id": pipeline_id,
            "stage": PipelineStage.SPEAKER_DIARIZATION.value,
            "speakers_found": len(diarization_result.get("speakers", [])),
            "segments_count": len(diarization_result.get("segments", []))
        }
        
    except Exception as e:
        logger.error(f"说话人分离失败: {pipeline_id}, 错误: {e}")
        return {
            "success": False,
            "pipeline_id": pipeline_id,
            "stage": PipelineStage.SPEAKER_DIARIZATION.value,
            "error": str(e)
        }
