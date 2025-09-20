"""
配置文件
"""
import os
from typing import Optional

class Settings:
    # 服务配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
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
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "audio-files")
    
    # 模型配置
    MODELS_DIR: str = os.getenv("MODELS_DIR", "./models")
    ASR_MODEL_PATH: str = os.getenv("ASR_MODEL_PATH", "./models/asr_model")
    SPEAKER_MODEL_PATH: str = os.getenv("SPEAKER_MODEL_PATH", "./models/speaker_model.onnx")
    VAD_MODEL_PATH: str = os.getenv("VAD_MODEL_PATH", "./models/vad_model.onnx")
    
    # FFmpeg配置
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "ffmpeg")
    AUDIO_FORMATS: list = ["wav", "mp3", "flac", "m4a", "ogg"]
    
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
    
    # 设备配置
    DEVICE_TYPE: str = os.getenv("DEVICE_TYPE", "auto")  # auto, cpu, gpu
    
    # 声纹配置
    SPEAKER_EMBEDDING_DIM: int = int(os.getenv("SPEAKER_EMBEDDING_DIM", "512"))
    SPEAKER_SIMILARITY_THRESHOLD: float = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.75"))
    
    # 音频配置
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
    MAX_AUDIO_SIZE: int = int(os.getenv("MAX_AUDIO_SIZE", "100")) * 1024 * 1024  # 100MB
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")  # development, production
    
    # 监控配置
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "8001"))
    ENABLE_SYSTEM_METRICS: bool = os.getenv("ENABLE_SYSTEM_METRICS", "true").lower() == "true"
    METRICS_UPDATE_INTERVAL: int = int(os.getenv("METRICS_UPDATE_INTERVAL", "30"))  # 秒

settings = Settings()