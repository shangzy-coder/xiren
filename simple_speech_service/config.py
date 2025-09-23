"""
简单语音识别服务配置
"""
import os
from pathlib import Path

class SimpleSettings:
    """简单配置类"""

    # 服务配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # 模型配置
    MODELS_DIR: str = os.getenv("MODELS_DIR", "./models")
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # 说话人配置
    SPEAKER_THRESHOLD: float = float(os.getenv("SPEAKER_THRESHOLD", "0.6"))
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "memory")  # memory 或 file
    SPEAKER_DATA_FILE: str = os.getenv("SPEAKER_DATA_FILE", "./data/speakers.json")
    EMBEDDINGS_DIR: str = os.getenv("EMBEDDINGS_DIR", "./data/embeddings")

    # 音频配置
    SUPPORTED_FORMATS: list = ["wav", "mp3", "flac", "m4a", "ogg", "amr"]
    MAX_AUDIO_SIZE: int = int(os.getenv("MAX_AUDIO_SIZE", "50")) * 1024 * 1024  # 50MB

    # 语言配置
    SUPPORTED_LANGUAGES: list = ["auto", "zh", "en", "ja", "ko", "yue"]
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "auto")

    # 模型路径配置
    ASR_MODEL_DIR: str = os.getenv("ASR_MODEL_DIR", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
    SPEAKER_MODEL_FILE: str = os.getenv("SPEAKER_MODEL_FILE", "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
    VAD_MODEL_FILE: str = os.getenv("VAD_MODEL_FILE", "silero_vad.onnx")
    PUNCTUATION_MODEL_DIR: str = os.getenv("PUNCTUATION_MODEL_DIR", "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12")

    @property
    def asr_model_path(self) -> str:
        """ASR模型完整路径"""
        return f"{self.MODELS_DIR}/{self.ASR_MODEL_DIR}"

    @property
    def speaker_model_path(self) -> str:
        """说话人模型完整路径"""
        return f"{self.MODELS_DIR}/{self.SPEAKER_MODEL_FILE}"

    @property
    def vad_model_path(self) -> str:
        """VAD模型完整路径"""
        return f"{self.MODELS_DIR}/{self.VAD_MODEL_FILE}"

    @property
    def punctuation_model_path(self) -> str:
        """标点模型完整路径"""
        return f"{self.MODELS_DIR}/{self.PUNCTUATION_MODEL_DIR}"

    @property
    def data_dir(self) -> Path:
        """数据目录路径"""
        return Path("./data")

    @property
    def speakers_file(self) -> Path:
        """说话人数据文件路径"""
        return Path(self.SPEAKER_DATA_FILE)

    @property
    def embeddings_dir(self) -> Path:
        """向量数据目录路径"""
        return Path(self.EMBEDDINGS_DIR)

# 全局配置实例
settings = SimpleSettings()