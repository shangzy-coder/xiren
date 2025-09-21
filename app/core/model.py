"""
语音识别模型加载和推理模块

基于Sherpa-ONNX实现ASR模型的封装，支持：
- 离线/非流式语音识别 (SenseVoice, Paraformer, Whisper等)
- 在线/流式语音识别 (Zipformer等)
- 声纹识别和说话人识别
- 语音活动检测 (VAD)
- 多种推理后端 (CPU, CUDA, ONNX Runtime)
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import sherpa_onnx

from app.config import settings

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """模型加载异常"""
    pass


class InferenceError(Exception):
    """推理异常"""
    pass


class ASRModelManager:
    """ASR模型管理器"""
    
    def __init__(self):
        self.offline_recognizer: Optional[sherpa_onnx.OfflineRecognizer] = None
        self.online_recognizer: Optional[sherpa_onnx.OnlineRecognizer] = None
        self.vad: Optional[sherpa_onnx.VoiceActivityDetector] = None
        self.speaker_extractor: Optional[sherpa_onnx.SpeakerEmbeddingExtractor] = None
        self.speaker_manager: Optional[sherpa_onnx.SpeakerEmbeddingManager] = None
        self.punctuation_processor: Optional[sherpa_onnx.OfflinePunctuation] = None
        
        self._is_initialized = False
        self._lock = threading.Lock()
        self._thread_pool = ThreadPoolExecutor(
            max_workers=settings.MAX_WORKERS,
            thread_name_prefix="asr_inference"
        )
        
        logger.info("ASR模型管理器初始化完成")

    async def initialize(
        self,
        model_type: str = "sense_voice",
        use_gpu: bool = False,
        enable_vad: bool = True,
        enable_speaker_id: bool = False,
        enable_punctuation: bool = False
    ) -> None:
        """
        异步初始化所有模型
        
        Args:
            model_type: 模型类型 (sense_voice, paraformer, whisper, zipformer等)
            use_gpu: 是否使用GPU加速
            enable_vad: 是否启用VAD
            enable_speaker_id: 是否启用声纹识别
            enable_punctuation: 是否启用标点符号处理
        """
        if self._is_initialized:
            logger.warning("模型已经初始化，跳过重复初始化")
            return
            
        logger.info(f"开始初始化ASR模型: {model_type}")
        start_time = time.time()
        
        try:
            # 在线程池中执行模型加载（避免阻塞事件循环）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._thread_pool,
                self._load_models,
                model_type,
                use_gpu,
                enable_vad,
                enable_speaker_id,
                enable_punctuation
            )
            
            # 保存初始化参数
            self._model_type = model_type
            self._use_gpu = use_gpu
            self._is_initialized = True
            elapsed = time.time() - start_time
            logger.info(f"模型初始化完成，耗时: {elapsed:.2f}秒")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise ModelLoadError(f"Failed to initialize models: {e}")

    def _load_models(
        self,
        model_type: str,
        use_gpu: bool,
        enable_vad: bool,
        enable_speaker_id: bool,
        enable_punctuation: bool
    ) -> None:
        """在线程池中执行的同步模型加载"""
        
        provider = "cuda" if use_gpu else "cpu"
        logger.info(f"使用推理设备: {provider}")
        
        # 1. 加载离线ASR模型
        self._load_offline_asr(model_type, provider)
        
        # 2. 加载VAD模型
        if enable_vad:
            self._load_vad_model(provider)
            
        # 3. 加载声纹识别模型
        if enable_speaker_id:
            self._load_speaker_models(provider)
            
        # 4. 加载标点符号处理模型
        if enable_punctuation:
            self._load_punctuation_model(provider)

    def _load_offline_asr(self, model_type: str, provider: str) -> None:
        """加载离线ASR识别器"""
        try:
            if model_type == "sense_voice":
                # SenseVoice模型 - 支持中英日韩粤多语言
                model_path = Path(settings.ASR_MODEL_PATH) / "model.int8.onnx"
                tokens_path = Path(settings.ASR_MODEL_PATH) / "tokens.txt"
                
                if not model_path.exists() or not tokens_path.exists():
                    raise FileNotFoundError(f"SenseVoice模型文件不存在: {model_path}")
                    
                self.offline_recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(model_path),
                    tokens=str(tokens_path),
                    num_threads=settings.MAX_WORKERS,
                    provider=provider,
                    language="auto",  # 自动检测语言
                    use_itn=True,     # 使用逆文本规范化
                    debug=False
                )
                
            elif model_type == "paraformer":
                # Paraformer模型 - 阿里达摩院模型
                model_path = Path(settings.ASR_MODEL_PATH) / "model.onnx"
                tokens_path = Path(settings.ASR_MODEL_PATH) / "tokens.txt"
                
                if not model_path.exists() or not tokens_path.exists():
                    raise FileNotFoundError(f"Paraformer模型文件不存在: {model_path}")
                    
                self.offline_recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                    paraformer=str(model_path),
                    tokens=str(tokens_path),
                    num_threads=settings.MAX_WORKERS,
                    sample_rate=settings.SAMPLE_RATE,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    provider=provider,
                    debug=False
                )
                
            elif model_type == "whisper":
                # Whisper模型
                encoder_path = Path(settings.ASR_MODEL_PATH) / "encoder.onnx"
                decoder_path = Path(settings.ASR_MODEL_PATH) / "decoder.onnx"
                tokens_path = Path(settings.ASR_MODEL_PATH) / "tokens.txt"
                
                if not all(p.exists() for p in [encoder_path, decoder_path, tokens_path]):
                    raise FileNotFoundError(f"Whisper模型文件不完整")
                    
                self.offline_recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                    encoder=str(encoder_path),
                    decoder=str(decoder_path),
                    tokens=str(tokens_path),
                    num_threads=settings.MAX_WORKERS,
                    decoding_method="greedy_search",
                    language="",  # 自动检测
                    task="transcribe",
                    provider=provider,
                    debug=False
                )
                
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
                
            logger.info(f"离线ASR模型加载成功: {model_type}")
            
        except Exception as e:
            logger.error(f"加载离线ASR模型失败: {e}")
            raise ModelLoadError(f"Failed to load offline ASR model: {e}")

    def _load_vad_model(self, provider: str) -> None:
        """加载VAD语音活动检测模型"""
        try:
            vad_model_path = Path(settings.VAD_MODEL_PATH)
            
            if not vad_model_path.exists():
                raise FileNotFoundError(f"VAD模型文件不存在: {vad_model_path}")
                
            # 创建VAD配置
            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = str(vad_model_path)
            config.silero_vad.threshold = 0.5
            config.silero_vad.min_silence_duration = 0.25  # 最小静音时长
            config.silero_vad.min_speech_duration = 0.25   # 最小语音时长
            config.silero_vad.max_speech_duration = 5.0    # 最大语音时长
            config.sample_rate = settings.SAMPLE_RATE
            config.num_threads = 2
            config.provider = provider
            
            self.vad = sherpa_onnx.VoiceActivityDetector(
                config, 
                buffer_size_in_seconds=30
            )
            
            logger.info("VAD模型加载成功")
            
        except Exception as e:
            logger.error(f"加载VAD模型失败: {e}")
            raise ModelLoadError(f"Failed to load VAD model: {e}")

    def _load_speaker_models(self, provider: str) -> None:
        """加载声纹识别模型"""
        try:
            speaker_model_path = Path(settings.SPEAKER_MODEL_PATH)
            
            if not speaker_model_path.exists():
                raise FileNotFoundError(f"声纹模型文件不存在: {speaker_model_path}")
                
            # 创建声纹嵌入提取器
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
            config.model = str(speaker_model_path)
            config.provider = provider
            config.num_threads = 2
            
            self.speaker_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            
            # 创建声纹管理器
            self.speaker_manager = sherpa_onnx.SpeakerEmbeddingManager(
                self.speaker_extractor.dim
            )
            
            logger.info(f"声纹识别模型加载成功 (嵌入维度: {self.speaker_extractor.dim})")
            
        except Exception as e:
            logger.error(f"加载声纹识别模型失败: {e}")
            raise ModelLoadError(f"Failed to load speaker model: {e}")

    def _load_punctuation_model(self, provider: str) -> None:
        """加载标点符号处理模型"""
        try:
            punctuation_model_path = Path(settings.PUNCTUATION_MODEL_PATH)

            if not punctuation_model_path.exists():
                raise FileNotFoundError(f"标点符号模型目录不存在: {punctuation_model_path}")

            # 检查模型文件
            model_file = punctuation_model_path / "model.int8.onnx"
            tokens_file = punctuation_model_path / "tokens.json"

            if not model_file.exists():
                raise FileNotFoundError(f"标点符号模型文件不存在: {model_file}")
            if not tokens_file.exists():
                raise FileNotFoundError(f"标点符号tokens文件不存在: {tokens_file}")

            # 创建标点符号处理器配置
            config = sherpa_onnx.OfflinePunctuationConfig()
            config.model.ct_transformer = str(model_file)
            config.model.num_threads = settings.MAX_WORKERS
            config.model.provider = provider
            config.model.debug = False

            # 创建标点符号处理器
            self.punctuation_processor = sherpa_onnx.OfflinePunctuation(config)

            logger.info(f"标点符号处理模型加载成功: {model_file}")

        except Exception as e:
            logger.error(f"加载标点符号模型失败: {e}")
            raise ModelLoadError(f"Failed to load punctuation model: {e}")

    async def recognize_audio(
        self,
        audio_data: Union[np.ndarray, bytes],
        sample_rate: int = None,
        enable_vad: bool = True,
        enable_speaker_id: bool = False
    ) -> Dict[str, Any]:
        """
        异步音频识别接口
        
        Args:
            audio_data: 音频数据 (numpy数组或字节流)
            sample_rate: 采样率，如果为None则使用默认值
            enable_vad: 是否启用VAD语音段落分割
            enable_speaker_id: 是否启用声纹识别
            
        Returns:
            识别结果字典，包含文本、时间戳、声纹等信息
        """
        if not self._is_initialized:
            raise InferenceError("模型未初始化，请先调用initialize()")
            
        if self.offline_recognizer is None:
            raise InferenceError("离线识别器未加载")
            
        # 预处理音频数据
        if isinstance(audio_data, bytes):
            audio_samples = np.frombuffer(audio_data, dtype=np.float32)
        else:
            audio_samples = audio_data.astype(np.float32)
            
        if sample_rate is None:
            sample_rate = settings.SAMPLE_RATE
            
        # 在线程池中执行推理
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._thread_pool,
            self._perform_recognition,
            audio_samples,
            sample_rate,
            enable_vad,
            enable_speaker_id
        )
        
        return result

    def _perform_recognition(
        self,
        audio_samples: np.ndarray,
        sample_rate: int,
        enable_vad: bool,
        enable_speaker_id: bool
    ) -> Dict[str, Any]:
        """执行同步识别推理"""
        
        start_time = time.time()
        
        try:
            if enable_vad and self.vad is not None:
                # 使用VAD分割音频
                segments = self._segment_audio_with_vad(audio_samples, sample_rate)
            else:
                # 不使用VAD，整段识别
                segments = [{
                    'samples': audio_samples,
                    'sample_rate': sample_rate,
                    'start_time': 0.0,
                    'end_time': len(audio_samples) / sample_rate
                }]
            
            # 批量识别所有段落
            results = []
            for segment in segments:
                # 创建识别流
                stream = self.offline_recognizer.create_stream()
                stream.accept_waveform(segment['sample_rate'], segment['samples'])
                
                # 执行识别
                self.offline_recognizer.decode_stream(stream)
                result = stream.result
                
                segment_result = {
                    'text': result.text,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'language': getattr(result, 'lang', 'unknown'),
                    'emotion': getattr(result, 'emotion', 'unknown'),
                    'event': getattr(result, 'event', 'unknown')
                }
                
                # 添加声纹识别
                if enable_speaker_id and self.speaker_extractor is not None:
                    speaker_info = self._identify_speaker(segment['samples'], segment['sample_rate'])
                    segment_result['speaker'] = speaker_info
                else:
                    segment_result['speaker'] = 'unknown'
                    
                results.append(segment_result)
            
            # 计算处理统计信息
            total_duration = len(audio_samples) / sample_rate
            processing_time = time.time() - start_time
            rtf = processing_time / total_duration if total_duration > 0 else 0
            
            return {
                'success': True,
                'results': results,
                'statistics': {
                    'total_duration': total_duration,
                    'processing_time': processing_time,
                    'real_time_factor': rtf,
                    'segments_count': len(results)
                }
            }
            
        except Exception as e:
            logger.error(f"音频识别失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'statistics': {}
            }

    def _segment_audio_with_vad(
        self, 
        audio_samples: np.ndarray, 
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """使用VAD分割音频"""
        
        if self.vad is None:
            # VAD不可用，返回整段音频
            return [{
                'samples': audio_samples,
                'sample_rate': sample_rate,
                'start_time': 0.0,
                'end_time': len(audio_samples) / sample_rate
            }]
        
        segments = []
        window_size = 1600  # 假设16kHz下100ms窗口
        total_samples_processed = 0
        
        try:
            # 逐窗口处理音频
            while total_samples_processed < len(audio_samples):
                end_idx = min(total_samples_processed + window_size, len(audio_samples))
                chunk = audio_samples[total_samples_processed:end_idx]
                
                self.vad.accept_waveform(chunk)
                total_samples_processed = end_idx
                
                # 获取检测到的语音段落
                while not self.vad.empty():
                    segment_samples = self.vad.front.samples
                    start_time = self.vad.front.start / sample_rate
                    duration = len(segment_samples) / sample_rate
                    end_time = start_time + duration
                    
                    segments.append({
                        'samples': segment_samples,
                        'sample_rate': sample_rate,
                        'start_time': start_time,
                        'end_time': end_time
                    })
                    
                    self.vad.pop()
            
            # 处理剩余音频
            self.vad.flush()
            while not self.vad.empty():
                segment_samples = self.vad.front.samples
                start_time = self.vad.front.start / sample_rate
                duration = len(segment_samples) / sample_rate
                end_time = start_time + duration
                
                segments.append({
                    'samples': segment_samples,
                    'sample_rate': sample_rate,
                    'start_time': start_time,
                    'end_time': end_time
                })
                
                self.vad.pop()
                
        except Exception as e:
            logger.error(f"VAD分割失败: {e}")
            # 分割失败，返回整段音频
            return [{
                'samples': audio_samples,
                'sample_rate': sample_rate,
                'start_time': 0.0,
                'end_time': len(audio_samples) / sample_rate
            }]
        
        logger.debug(f"VAD分割完成，得到 {len(segments)} 个语音段落")
        return segments

    def _identify_speaker(self, audio_samples: np.ndarray, sample_rate: int) -> str:
        """识别说话人"""
        
        if self.speaker_extractor is None or self.speaker_manager is None:
            return 'unknown'
        
        try:
            # 提取说话人嵌入
            stream = self.speaker_extractor.create_stream()
            stream.accept_waveform(sample_rate, audio_samples)
            stream.input_finished()
            
            embedding = self.speaker_extractor.compute(stream)
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            # 在已注册说话人中搜索
            embedding_list = embedding.tolist()
            matched_speaker = self.speaker_manager.search(embedding_list, threshold=0.3)
            
            if matched_speaker:
                return matched_speaker
            else:
                # 未匹配到已注册说话人，返回临时标识
                return f"Speaker_{hash(tuple(embedding_list[:10])) % 1000:03d}"
                
        except Exception as e:
            logger.error(f"声纹识别失败: {e}")
            return 'unknown'

    async def register_speaker(
        self,
        speaker_name: str,
        audio_data: Union[np.ndarray, bytes],
        sample_rate: int = None
    ) -> bool:
        """
        异步注册说话人
        
        Args:
            speaker_name: 说话人名称
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            注册是否成功
        """
        if self.speaker_extractor is None or self.speaker_manager is None:
            logger.error("声纹识别模型未加载")
            return False
        
        # 预处理音频数据
        if isinstance(audio_data, bytes):
            audio_samples = np.frombuffer(audio_data, dtype=np.float32)
        else:
            audio_samples = audio_data.astype(np.float32)
            
        if sample_rate is None:
            sample_rate = settings.SAMPLE_RATE
        
        # 在线程池中执行注册
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            self._thread_pool,
            self._register_speaker_sync,
            speaker_name,
            audio_samples,
            sample_rate
        )
        
        return success

    def _register_speaker_sync(
        self,
        speaker_name: str,
        audio_samples: np.ndarray,
        sample_rate: int
    ) -> bool:
        """同步注册说话人"""
        try:
            # 提取嵌入
            stream = self.speaker_extractor.create_stream()
            stream.accept_waveform(sample_rate, audio_samples)
            stream.input_finished()
            
            embedding = self.speaker_extractor.compute(stream)
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            # 注册到管理器
            success = self.speaker_manager.add(speaker_name, embedding.tolist())
            
            if success:
                logger.info(f"说话人 '{speaker_name}' 注册成功")
            else:
                logger.warning(f"说话人 '{speaker_name}' 注册失败")
                
            return success
            
        except Exception as e:
            logger.error(f"注册说话人 '{speaker_name}' 时出错: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'is_initialized': self._is_initialized,
            'model_type': getattr(self, '_model_type', 'none'),
            'use_gpu': getattr(self, '_use_gpu', False),
            'asr_loaded': self.offline_recognizer is not None,
            'vad_loaded': self.vad is not None,
            'speaker_loaded': self.speaker_extractor is not None and self.speaker_manager is not None,
            'punctuation_loaded': self.punctuation_processor is not None,
            'sample_rate': settings.SAMPLE_RATE,
            'max_workers': settings.MAX_WORKERS,
            'models_dir': settings.MODELS_DIR,
            'asr_model_path': settings.ASR_MODEL_PATH,
            'vad_model_path': settings.VAD_MODEL_PATH,
            'speaker_model_path': settings.SPEAKER_MODEL_PATH,
            'punctuation_model_path': settings.PUNCTUATION_MODEL_PATH
        }

    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("正在清理ASR模型资源...")
        
        # 关闭线程池
        self._thread_pool.shutdown(wait=True)
        
        # 清理模型引用
        self.offline_recognizer = None
        self.online_recognizer = None
        self.vad = None
        self.speaker_extractor = None
        self.speaker_manager = None
        self.punctuation_processor = None
        
        self._is_initialized = False
        logger.info("ASR模型资源清理完成")


# 全局模型管理器实例
model_manager = ASRModelManager()


async def initialize_models(**kwargs) -> None:
    """初始化全局模型管理器"""
    await model_manager.initialize(**kwargs)


async def recognize_audio(**kwargs) -> Dict[str, Any]:
    """全局音频识别接口"""
    return await model_manager.recognize_audio(**kwargs)


async def register_speaker(**kwargs) -> bool:
    """全局说话人注册接口"""
    return await model_manager.register_speaker(**kwargs)


def get_model_info() -> Dict[str, Any]:
    """获取全局模型信息"""
    return model_manager.get_model_info()


async def cleanup_models() -> None:
    """清理全局模型资源"""
    await model_manager.cleanup()
