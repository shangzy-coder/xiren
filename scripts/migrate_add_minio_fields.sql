-- 数据库迁移脚本：为 MinIO 对象存储添加字段
-- 版本：v1.1 - 添加 MinIO 支持
-- 创建时间：2025-09-22

-- 开始事务
BEGIN;

-- 1. 扩展 audio_files 表添加 MinIO 相关字段
DO $$
BEGIN
    -- 检查并添加 bucket 字段
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'audio_files' AND column_name = 'bucket') THEN
        ALTER TABLE audio_files ADD COLUMN bucket VARCHAR(100);
        RAISE NOTICE 'Added bucket column to audio_files table';
    END IF;

    -- 检查并添加 object_name 字段
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'audio_files' AND column_name = 'object_name') THEN
        ALTER TABLE audio_files ADD COLUMN object_name VARCHAR(1000);
        RAISE NOTICE 'Added object_name column to audio_files table';
    END IF;

    -- 检查并添加 sha256_hash 字段
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'audio_files' AND column_name = 'sha256_hash') THEN
        ALTER TABLE audio_files ADD COLUMN sha256_hash VARCHAR(64);
        RAISE NOTICE 'Added sha256_hash column to audio_files table';
    END IF;

    -- 检查并添加 content_type 字段
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'audio_files' AND column_name = 'content_type') THEN
        ALTER TABLE audio_files ADD COLUMN content_type VARCHAR(200);
        RAISE NOTICE 'Added content_type column to audio_files table';
    END IF;

    -- 检查并添加 minio_version_id 字段
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'audio_files' AND column_name = 'minio_version_id') THEN
        ALTER TABLE audio_files ADD COLUMN minio_version_id VARCHAR(255);
        RAISE NOTICE 'Added minio_version_id column to audio_files table';
    END IF;

    -- 检查并添加 storage_class 字段
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'audio_files' AND column_name = 'storage_class') THEN
        ALTER TABLE audio_files ADD COLUMN storage_class VARCHAR(50) DEFAULT 'STANDARD';
        RAISE NOTICE 'Added storage_class column to audio_files table';
    END IF;

    -- 检查并添加 file_tags 字段
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'audio_files' AND column_name = 'file_tags') THEN
        ALTER TABLE audio_files ADD COLUMN file_tags JSONB DEFAULT '{}';
        RAISE NOTICE 'Added file_tags column to audio_files table';
    END IF;

    -- 检查并添加 upload_time 字段（用于记录MinIO上传时间）
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'audio_files' AND column_name = 'upload_time') THEN
        ALTER TABLE audio_files ADD COLUMN upload_time TIMESTAMP WITH TIME ZONE;
        RAISE NOTICE 'Added upload_time column to audio_files table';
    END IF;

    -- 检查并添加 is_duplicated 字段（标记是否为去重后的文件）
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'audio_files' AND column_name = 'is_duplicated') THEN
        ALTER TABLE audio_files ADD COLUMN is_duplicated BOOLEAN DEFAULT false;
        RAISE NOTICE 'Added is_duplicated column to audio_files table';
    END IF;

    -- 检查并添加 original_filename 字段（保存原始文件名）
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'audio_files' AND column_name = 'original_filename') THEN
        ALTER TABLE audio_files ADD COLUMN original_filename VARCHAR(500);
        RAISE NOTICE 'Added original_filename column to audio_files table';
    END IF;
END $$;

-- 2. 创建 file_versions 表支持版本管理
CREATE TABLE IF NOT EXISTS file_versions (
    id SERIAL PRIMARY KEY,
    audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL DEFAULT 1,
    bucket VARCHAR(100) NOT NULL,
    object_name VARCHAR(1000) NOT NULL,
    file_size BIGINT,
    sha256_hash VARCHAR(64),
    minio_version_id VARCHAR(255),
    is_current BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    
    -- 约束：确保每个文件只有一个当前版本
    CONSTRAINT unique_current_version_per_file 
        EXCLUDE (audio_file_id WITH =) WHERE (is_current = true)
);

-- 添加注释
COMMENT ON TABLE file_versions IS '音频文件版本管理表，支持MinIO版本控制';
COMMENT ON COLUMN file_versions.audio_file_id IS '关联的音频文件ID';
COMMENT ON COLUMN file_versions.version_number IS '版本号，从1开始递增';
COMMENT ON COLUMN file_versions.bucket IS 'MinIO存储桶名称';
COMMENT ON COLUMN file_versions.object_name IS 'MinIO对象名称';
COMMENT ON COLUMN file_versions.minio_version_id IS 'MinIO版本ID';
COMMENT ON COLUMN file_versions.is_current IS '是否为当前活跃版本';

-- 3. 添加数据库索引以优化查询性能
DO $$
BEGIN
    -- 为 audio_files 表新字段创建索引
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audio_files_bucket') THEN
        CREATE INDEX idx_audio_files_bucket ON audio_files(bucket);
        RAISE NOTICE 'Created index idx_audio_files_bucket';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audio_files_hash') THEN
        CREATE INDEX idx_audio_files_hash ON audio_files(sha256_hash);
        RAISE NOTICE 'Created index idx_audio_files_hash';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audio_files_object_name') THEN
        CREATE INDEX idx_audio_files_object_name ON audio_files(object_name);
        RAISE NOTICE 'Created index idx_audio_files_object_name';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audio_files_content_type') THEN
        CREATE INDEX idx_audio_files_content_type ON audio_files(content_type);
        RAISE NOTICE 'Created index idx_audio_files_content_type';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audio_files_upload_time') THEN
        CREATE INDEX idx_audio_files_upload_time ON audio_files(upload_time);
        RAISE NOTICE 'Created index idx_audio_files_upload_time';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audio_files_duplicated') THEN
        CREATE INDEX idx_audio_files_duplicated ON audio_files(is_duplicated);
        RAISE NOTICE 'Created index idx_audio_files_duplicated';
    END IF;

    -- 为 file_versions 表创建索引
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_file_versions_audio_file') THEN
        CREATE INDEX idx_file_versions_audio_file ON file_versions(audio_file_id);
        RAISE NOTICE 'Created index idx_file_versions_audio_file';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_file_versions_current') THEN
        CREATE INDEX idx_file_versions_current ON file_versions(is_current) WHERE is_current = true;
        RAISE NOTICE 'Created index idx_file_versions_current';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_file_versions_hash') THEN
        CREATE INDEX idx_file_versions_hash ON file_versions(sha256_hash);
        RAISE NOTICE 'Created index idx_file_versions_hash';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_file_versions_bucket_object') THEN
        CREATE INDEX idx_file_versions_bucket_object ON file_versions(bucket, object_name);
        RAISE NOTICE 'Created index idx_file_versions_bucket_object';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_file_versions_created_at') THEN
        CREATE INDEX idx_file_versions_created_at ON file_versions(created_at);
        RAISE NOTICE 'Created index idx_file_versions_created_at';
    END IF;
END $$;

-- 4. 添加外键约束确保数据完整性
DO $$
BEGIN
    -- 确保 file_versions 表的外键约束存在
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_file_versions_audio_file' 
        AND table_name = 'file_versions'
    ) THEN
        ALTER TABLE file_versions 
        ADD CONSTRAINT fk_file_versions_audio_file 
        FOREIGN KEY (audio_file_id) REFERENCES audio_files(id) ON DELETE CASCADE;
        RAISE NOTICE 'Added foreign key constraint fk_file_versions_audio_file';
    END IF;
END $$;

-- 5. 创建触发器函数来自动管理版本号
CREATE OR REPLACE FUNCTION update_file_version_number()
RETURNS TRIGGER AS $$
BEGIN
    -- 如果没有指定版本号，自动设置为下一个版本号
    IF NEW.version_number IS NULL OR NEW.version_number = 0 THEN
        SELECT COALESCE(MAX(version_number), 0) + 1 
        INTO NEW.version_number 
        FROM file_versions 
        WHERE audio_file_id = NEW.audio_file_id;
    END IF;

    -- 如果设置为当前版本，将其他版本设置为非当前
    IF NEW.is_current = true THEN
        UPDATE file_versions 
        SET is_current = false 
        WHERE audio_file_id = NEW.audio_file_id AND id != COALESCE(NEW.id, -1);
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 创建触发器
DROP TRIGGER IF EXISTS trigger_update_file_version ON file_versions;
CREATE TRIGGER trigger_update_file_version
    BEFORE INSERT OR UPDATE ON file_versions
    FOR EACH ROW
    EXECUTE FUNCTION update_file_version_number();

-- 6. 为新字段添加约束和检查
ALTER TABLE audio_files 
ADD CONSTRAINT check_sha256_format 
CHECK (sha256_hash IS NULL OR LENGTH(sha256_hash) = 64);

ALTER TABLE audio_files 
ADD CONSTRAINT check_bucket_format 
CHECK (bucket IS NULL OR (LENGTH(bucket) >= 3 AND LENGTH(bucket) <= 63));

ALTER TABLE file_versions 
ADD CONSTRAINT check_version_number_positive 
CHECK (version_number > 0);

-- 提交事务
COMMIT;

-- 输出完成信息
\echo '数据库迁移完成！'
\echo '已添加以下功能：'
\echo '1. audio_files 表新增 MinIO 相关字段'
\echo '2. file_versions 表支持版本管理'
\echo '3. 相关索引用于查询优化'
\echo '4. 数据完整性约束'
\echo '5. 自动版本管理触发器'
