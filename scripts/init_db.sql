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
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建音频文件表
CREATE TABLE IF NOT EXISTS audio_files (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000),
    file_size BIGINT,
    duration FLOAT,
    sample_rate INTEGER,
    channels INTEGER,
    format VARCHAR(50),
    speaker_id INTEGER REFERENCES speakers(id),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建识别会话表
CREATE TABLE IF NOT EXISTS recognition_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    audio_file_id INTEGER REFERENCES audio_files(id),
    results JSONB DEFAULT '{}',
    recognition_type VARCHAR(50),
    processing_time FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- 创建识别记录表 (保持向后兼容)
CREATE TABLE IF NOT EXISTS recognition_records (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    audio_file_path TEXT,
    transcription TEXT,
    speaker_results JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_speakers_name ON speakers(speaker_name);
CREATE INDEX IF NOT EXISTS idx_speakers_embedding ON speakers USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_audio_files_speaker ON audio_files(speaker_id);
CREATE INDEX IF NOT EXISTS idx_recognition_sessions_session ON recognition_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_recognition_sessions_status ON recognition_sessions(status);
CREATE INDEX IF NOT EXISTS idx_recognition_records_session ON recognition_records(session_id);
CREATE INDEX IF NOT EXISTS idx_recognition_records_created_at ON recognition_records(created_at);
