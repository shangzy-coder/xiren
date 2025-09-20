-- 初始化数据库脚本
-- 创建pgvector扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建声纹表
CREATE TABLE IF NOT EXISTS speakers (
    id SERIAL PRIMARY KEY,
    speaker_id VARCHAR(255) UNIQUE NOT NULL,
    speaker_name VARCHAR(255) NOT NULL,
    audio_file_path TEXT,
    embedding vector(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建识别记录表
CREATE TABLE IF NOT EXISTS recognition_records (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    audio_file_path TEXT,
    transcription TEXT,
    speaker_results JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_speakers_embedding ON speakers USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_recognition_records_session ON recognition_records(session_id);
CREATE INDEX IF NOT EXISTS idx_recognition_records_created_at ON recognition_records(created_at);
