"""
语音处理器
基于demo中的语音处理功能，整合ASR、VAD、说话人识别等
"""
import sherpa_onnx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time

from .config import settings
from .speaker_manager import SpeakerManager

class SpeechProcessor:
    """语音处理器"""

    def __init__(self, use_gpu: bool = False):
        """
        初始化语音处理器

        Args:
            use_gpu: 是否使用GPU
        """
        self.use_gpu = use_gpu
        self.asr_recognizer = None
        self.vad_detector = None
        self.punctuation_processor = None
        self.speaker_extractor = None
        self.speaker_manager = None

        # 初始化各个组件
        self._init_asr()
        self._init_vad()
        self._init_punctuation()
        self._init_speaker_recognition()

        print("语音处理器初始化完成")

    def _init_asr(self):
        """初始化ASR识别器"""
        try:
            provider = "cuda" if self.use_gpu else "cpu"
            print(f"初始化ASR模型，使用设备: {provider}")

            # 使用SenseVoice模型
            self.asr_recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=f"{settings.asr_model_path}/model.onnx",
                tokens=f"{settings.asr_model_path}/tokens.txt",
                provider=provider,
                num_threads=4,
                language=settings.DEFAULT_LANGUAGE,
                debug=False,
            )
            print("ASR模型初始化成功")
        except Exception as e:
            print(f"ASR模型初始化失败: {e}")
            self.asr_recognizer = None

    def _init_vad(self):
        """初始化语音活动检测器"""
        try:
            print("初始化VAD模型")
            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = settings.vad_model_path
            config.silero_vad.threshold = 0.5
            config.silero_vad.min_silence_duration = 0.25
            config.silero_vad.min_speech_duration = 0.25
            config.silero_vad.max_speech_duration = 5
            config.sample_rate = settings.SAMPLE_RATE

            self.vad_detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
            print("VAD模型初始化成功")
        except Exception as e:
            print(f"VAD模型初始化失败: {e}")
            self.vad_detector = None

    def _init_punctuation(self):
        """初始化标点符号处理器"""
        try:
            provider = "cuda" if self.use_gpu else "cpu"
            print(f"初始化标点模型，使用设备: {provider}")

            punct_model_dir = settings.punctuation_model_path
            model_config = sherpa_onnx.OfflinePunctuationModelConfig()
            model_config.ct_transformer = f"{punct_model_dir}/model.onnx"
            model_config.num_threads = 2
            model_config.provider = provider

            punctuation_config = sherpa_onnx.OfflinePunctuationConfig()
            punctuation_config.model = model_config

            self.punctuation_processor = sherpa_onnx.OfflinePunctuation(punctuation_config)
            print("标点模型初始化成功")
        except Exception as e:
            print(f"标点模型初始化失败: {e}")
            self.punctuation_processor = None

    def _init_speaker_recognition(self):
        """初始化说话人识别系统"""
        try:
            provider = "cuda" if self.use_gpu else "cpu"
            print(f"初始化说话人识别模型，使用设备: {provider}")

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
            config.model = settings.speaker_model_path
            config.provider = provider
            config.num_threads = 2

            self.speaker_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            print(f"说话人识别模型初始化成功，嵌入维度: {self.speaker_extractor.dim}")

            # 初始化说话人管理器
            self.speaker_manager = SpeakerManager(storage_type=settings.STORAGE_TYPE)

        except Exception as e:
            print(f"说话人识别模型初始化失败: {e}")
            self.speaker_extractor = None
            self.speaker_manager = None

    def detect_speech_segments(self, samples: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """
        检测语音段落

        Args:
            samples: 音频样本
            sample_rate: 采样率

        Returns:
            List[Dict]: 语音段落列表
        """
        if not self.vad_detector:
            # 如果没有VAD，整个音频作为一个段落
            return [{
                'samples': samples,
                'sample_rate': sample_rate,
                'start_time': 0.0,
                'end_time': len(samples) / sample_rate
            }]

        print("正在进行语音段落检测...")
        segments = []
        total_samples_processed = 0
        window_size = self.vad_detector.config.silero_vad.window_size

        while len(samples) > total_samples_processed + window_size:
            chunk = samples[total_samples_processed:total_samples_processed + window_size]
            self.vad_detector.accept_waveform(chunk)
            total_samples_processed += window_size

            while not self.vad_detector.empty():
                segment_samples = self.vad_detector.front.samples
                start_time = self.vad_detector.front.start / sample_rate
                duration = len(segment_samples) / sample_rate
                end_time = start_time + duration

                segments.append({
                    'samples': segment_samples,
                    'sample_rate': sample_rate,
                    'start_time': start_time,
                    'end_time': end_time
                })
                self.vad_detector.pop()

        # 处理剩余音频
        self.vad_detector.flush()
        while not self.vad_detector.empty():
            segment_samples = self.vad_detector.front.samples
            start_time = self.vad_detector.front.start / sample_rate
            duration = len(segment_samples) / sample_rate
            end_time = start_time + duration

            segments.append({
                'samples': segment_samples,
                'sample_rate': sample_rate,
                'start_time': start_time,
                'end_time': end_time
            })
            self.vad_detector.pop()

        print(f"检测到 {len(segments)} 个语音段落")
        return segments

    def recognize_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        识别单个语音段落

        Args:
            segment: 语音段落数据

        Returns:
            Dict: 识别结果
        """
        if not self.asr_recognizer:
            return {
                'text': '',
                'emotion': 'unknown',
                'event': 'unknown',
                'language': 'unknown',
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'speaker': 'Unknown'
            }

        try:
            # 创建识别流
            stream = self.asr_recognizer.create_stream()
            stream.accept_waveform(segment['sample_rate'], segment['samples'])
            self.asr_recognizer.decode_stream(stream)

            result = stream.result

            # 基础结果
            result_data = {
                'text': result.text,
                'emotion': result.emotion.strip('<|>'),
                'event': result.event.strip('<|>'),
                'language': result.lang.strip('<|>'),
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'speaker': 'Unknown'
            }

            # 添加标点（如果有标点处理器）
            if self.punctuation_processor and result_data['text'].strip():
                try:
                    result_data['text_with_punct'] = self.punctuation_processor.add_punctuation(result_data['text'])
                except:
                    result_data['text_with_punct'] = result_data['text']

            # 说话人识别（如果有说话人系统）
            if self.speaker_extractor and self.speaker_manager:
                result_data['speaker'] = self._identify_speaker_in_segment(segment)

            return result_data

        except Exception as e:
            print(f"识别段落失败: {e}")
            return {
                'text': '',
                'emotion': 'unknown',
                'event': 'unknown',
                'language': 'unknown',
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'speaker': 'Unknown',
                'error': str(e)
            }

    def _identify_speaker_in_segment(self, segment: Dict[str, Any]) -> str:
        """
        在语音段落中识别说话人

        Args:
            segment: 语音段落数据

        Returns:
            str: 说话人标识
        """
        try:
            # 提取嵌入向量
            stream = self.speaker_extractor.create_stream()
            stream.accept_waveform(segment['sample_rate'], segment['samples'])
            stream.input_finished()

            embedding = self.speaker_extractor.compute(stream)
            if isinstance(embedding, list):
                embedding = np.array(embedding)

            # 识别说话人
            speaker_name, similarity = self.speaker_manager.identify_speaker(embedding)

            if speaker_name:
                return speaker_name
            else:
                # 为未注册说话人分配临时标签
                return f"Speaker{len(self.speaker_manager.list_speakers()) + 1}"

        except Exception as e:
            print(f"说话人识别失败: {e}")
            return "Unknown"

    def recognize_audio(self, samples: np.ndarray, sample_rate: int,
                       enable_vad: bool = True) -> List[Dict[str, Any]]:
        """
        识别音频内容

        Args:
            samples: 音频样本
            sample_rate: 采样率
            enable_vad: 是否启用语音活动检测

        Returns:
            List[Dict]: 识别结果列表
        """
        print("开始语音识别...")

        start_time = time.time()

        # 检测语音段落
        if enable_vad and self.vad_detector:
            segments = self.detect_speech_segments(samples, sample_rate)
        else:
            # 不使用VAD，整个音频作为一个段落
            segments = [{
                'samples': samples,
                'sample_rate': sample_rate,
                'start_time': 0.0,
                'end_time': len(samples) / sample_rate
            }]

        if not segments:
            print("未检测到语音内容")
            return []

        # 识别所有段落
        results = []
        for segment in segments:
            result = self.recognize_segment(segment)
            results.append(result)

        processing_time = time.time() - start_time
        total_duration = len(samples) / sample_rate

        print(f"语音识别完成，处理了 {len(results)} 个段落")
        print(".2f"
        return results

    def register_speaker(self, name: str, samples: np.ndarray, sample_rate: int,
                        metadata: Dict[str, Any] = None) -> bool:
        """
        注册说话人

        Args:
            name: 说话人姓名
            samples: 音频样本
            sample_rate: 采样率
            metadata: 元数据

        Returns:
            bool: 注册是否成功
        """
        if not self.speaker_extractor or not self.speaker_manager:
            print("说话人识别系统未初始化")
            return False

        try:
            # 提取嵌入向量
            stream = self.speaker_extractor.create_stream()
            stream.accept_waveform(sample_rate, samples)
            stream.input_finished()

            embedding = self.speaker_extractor.compute(stream)
            if isinstance(embedding, list):
                embedding = np.array(embedding)

            # 注册到管理器
            return self.speaker_manager.register_speaker(name, embedding, metadata)

        except Exception as e:
            print(f"注册说话人失败: {e}")
            return False

    def get_speaker_manager(self) -> SpeakerManager:
        """获取说话人管理器"""
        return self.speaker_manager

    def identify_speaker_from_audio(self, samples: np.ndarray, sample_rate: int,
                                   threshold: float = None) -> Tuple[Optional[str], float]:
        """
        从音频中识别说话人

        Args:
            samples: 音频样本
            sample_rate: 采样率
            threshold: 相似度阈值

        Returns:
            Tuple[Optional[str], float]: (说话人姓名, 相似度)
        """
        if not self.speaker_extractor or not self.speaker_manager:
            return None, 0.0

        try:
            # 提取嵌入向量
            stream = self.speaker_extractor.create_stream()
            stream.accept_waveform(sample_rate, samples)
            stream.input_finished()

            embedding = self.speaker_extractor.compute(stream)
            if isinstance(embedding, list):
                embedding = np.array(embedding)

            # 识别说话人
            return self.speaker_manager.identify_speaker(embedding, threshold)

        except Exception as e:
            print(f"说话人识别失败: {e}")
            return None, 0.0

    def get_stats(self) -> Dict[str, Any]:
        """
        获取处理器统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            'asr_loaded': self.asr_recognizer is not None,
            'vad_loaded': self.vad_detector is not None,
            'punctuation_loaded': self.punctuation_processor is not None,
            'speaker_loaded': self.speaker_extractor is not None,
            'use_gpu': self.use_gpu,
            'speaker_stats': self.speaker_manager.get_stats() if self.speaker_manager else None
        }