"""
配置文件
"""
import os
from typing import Optional

class Settings:
    # 服务配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8001"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # 数据库配置
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql+asyncpg://speech_user:speech_pass@localhost:5432/speech_recognition"
    )
    
    # MinIO配置
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "audio-files")
    MINIO_TEMP_BUCKET: str = os.getenv("MINIO_TEMP_BUCKET", "temp-files")
    MINIO_BACKUP_BUCKET: str = os.getenv("MINIO_BACKUP_BUCKET", "backup-files")
    
    # 文件存储配置
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024  # 100MB
    TEMP_FILE_CLEANUP_HOURS: int = int(os.getenv("TEMP_FILE_CLEANUP_HOURS", "24"))
    ENABLE_FILE_VERSIONING: bool = os.getenv("ENABLE_FILE_VERSIONING", "false").lower() == "true"
    ENABLE_FILE_DEDUPLICATION: bool = os.getenv("ENABLE_FILE_DEDUPLICATION", "true").lower() == "true"
    
    # 模型配置
    MODELS_DIR: str = os.getenv("MODELS_DIR", "./models")
    ASR_MODEL_PATH: str = os.getenv("ASR_MODEL_PATH", "./models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
    SPEAKER_MODEL_PATH: str = os.getenv("SPEAKER_MODEL_PATH", "./models/speaker-recongition/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx")
    VAD_MODEL_PATH: str = os.getenv("VAD_MODEL_PATH", "./models/vad/silero_vad.onnx")
    
    # VAD 模块配置
    VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
    VAD_MIN_SILENCE_DURATION: float = float(os.getenv("VAD_MIN_SILENCE_DURATION", "0.25"))
    VAD_MIN_SPEECH_DURATION: float = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.25"))
    VAD_MAX_SPEECH_DURATION: float = float(os.getenv("VAD_MAX_SPEECH_DURATION", "5.0"))
    VAD_BUFFER_SIZE_SECONDS: float = float(os.getenv("VAD_BUFFER_SIZE_SECONDS", "30.0"))
    VAD_PROVIDER: str = os.getenv("VAD_PROVIDER", "cpu")
    VAD_THREADS: int = int(os.getenv("VAD_THREADS", "2"))
    PUNCTUATION_MODEL_PATH: str = os.getenv("PUNCTUATION_MODEL_PATH", "./models/punction/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8")
    
    # FFmpeg配置
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "ffmpeg")
    AUDIO_FORMATS: list = ["wav", "mp3", "flac", "m4a", "ogg", "mpga", "amr"]
    VIDEO_FORMATS: list = ["mp4", "mov", "mpeg", "webm"]
    SUPPORTED_FORMATS: list = AUDIO_FORMATS + VIDEO_FORMATS
    
    # 队列配置
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    QUEUE_SIZE: int = int(os.getenv("QUEUE_SIZE", "100"))

    # 并发处理配置
    THREAD_POOL_SIZE: int = int(os.getenv("THREAD_POOL_SIZE", "8"))
    MAX_QUEUE_SIZE: int = int(os.getenv("MAX_QUEUE_SIZE", "1000"))
    ENABLE_QUEUE_METRICS: bool = os.getenv("ENABLE_QUEUE_METRICS", "true").lower() == "true"
    TASK_TIMEOUT: int = int(os.getenv("TASK_TIMEOUT", "300"))  # 5分钟
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50"))
    BATCH_SIZE_LIMIT: int = int(os.getenv("BATCH_SIZE_LIMIT", "10"))

    # 批次处理性能配置
    MAX_BATCH_THREADS: int = int(os.getenv("MAX_BATCH_THREADS", "4"))  # 最大批次线程数
    MIN_BATCH_SIZE: int = int(os.getenv("MIN_BATCH_SIZE", "20"))  # 最小批次大小
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "100"))  # 最大批次大小
    ASR_THREADS_PER_BATCH: int = int(os.getenv("ASR_THREADS_PER_BATCH", "2"))  # 每个批次的ASR线程数
    PUNCTUATION_THREADS_PER_BATCH: int = int(os.getenv("PUNCTUATION_THREADS_PER_BATCH", "2"))  # 每个批次的标点线程数
    
    # 设备配置
    DEVICE_TYPE: str = os.getenv("DEVICE_TYPE", "gpu")  # auto, cpu, gpu
    
    # 声纹配置
    SPEAKER_EMBEDDING_DIM: int = int(os.getenv("SPEAKER_EMBEDDING_DIM", "512"))
    SPEAKER_SIMILARITY_THRESHOLD: float = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.75"))
    
    # 音频配置
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
    MAX_AUDIO_SIZE: int = int(os.getenv("MAX_AUDIO_SIZE", "100")) * 1024 * 1024  # 100MB
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")  # development, production
    
    # 模型预加载配置
    ENABLE_MODEL_PRELOAD: bool = os.getenv("ENABLE_MODEL_PRELOAD", "true").lower() == "true"
    DEFAULT_MODEL_TYPE: str = os.getenv("DEFAULT_MODEL_TYPE", "sense_voice")
    DEFAULT_USE_GPU: bool = os.getenv("DEFAULT_USE_GPU", "true").lower() == "true"
    DEFAULT_ENABLE_VAD: bool = os.getenv("DEFAULT_ENABLE_VAD", "true").lower() == "true"
    DEFAULT_ENABLE_SPEAKER_ID: bool = os.getenv("DEFAULT_ENABLE_SPEAKER_ID", "true").lower() == "true"
    DEFAULT_ENABLE_PUNCTUATION: bool = os.getenv("DEFAULT_ENABLE_PUNCTUATION", "true").lower() == "true"

    # 监控配置
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "false").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "8001"))
    ENABLE_SYSTEM_METRICS: bool = os.getenv("ENABLE_SYSTEM_METRICS", "false").lower() == "true"
    METRICS_UPDATE_INTERVAL: int = int(os.getenv("METRICS_UPDATE_INTERVAL", "30"))  # 秒

settings = Settings()