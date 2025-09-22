-- 数据库回滚脚本：移除 MinIO 对象存储字段
-- 版本：v1.1 回滚脚本
-- 创建时间：2025-09-22
-- 警告：此脚本将删除所有MinIO相关数据，请谨慎使用

-- 开始事务
BEGIN;

-- 1. 删除触发器和函数
DROP TRIGGER IF EXISTS trigger_update_file_version ON file_versions;
DROP FUNCTION IF EXISTS update_file_version_number();

-- 2. 删除 file_versions 表
DROP TABLE IF EXISTS file_versions CASCADE;

-- 3. 删除 audio_files 表的MinIO相关索引
DO $$
BEGIN
    -- 删除为MinIO字段创建的索引
    IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audio_files_bucket') THEN
        DROP INDEX idx_audio_files_bucket;
        RAISE NOTICE 'Dropped index idx_audio_files_bucket';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audio_files_hash') THEN
        DROP INDEX idx_audio_files_hash;
        RAISE NOTICE 'Dropped index idx_audio_files_hash';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audio_files_object_name') THEN
        DROP INDEX idx_audio_files_object_name;
        RAISE NOTICE 'Dropped index idx_audio_files_object_name';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audio_files_content_type') THEN
        DROP INDEX idx_audio_files_content_type;
        RAISE NOTICE 'Dropped index idx_audio_files_content_type';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audio_files_upload_time') THEN
        DROP INDEX idx_audio_files_upload_time;
        RAISE NOTICE 'Dropped index idx_audio_files_upload_time';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audio_files_duplicated') THEN
        DROP INDEX idx_audio_files_duplicated;
        RAISE NOTICE 'Dropped index idx_audio_files_duplicated';
    END IF;
END $$;

-- 4. 删除约束
DO $$
BEGIN
    -- 删除检查约束
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'check_sha256_format' 
        AND table_name = 'audio_files'
    ) THEN
        ALTER TABLE audio_files DROP CONSTRAINT check_sha256_format;
        RAISE NOTICE 'Dropped constraint check_sha256_format';
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'check_bucket_format' 
        AND table_name = 'audio_files'
    ) THEN
        ALTER TABLE audio_files DROP CONSTRAINT check_bucket_format;
        RAISE NOTICE 'Dropped constraint check_bucket_format';
    END IF;
END $$;

-- 5. 删除 audio_files 表的MinIO相关字段
DO $$
BEGIN
    -- 删除 bucket 字段
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'audio_files' AND column_name = 'bucket') THEN
        ALTER TABLE audio_files DROP COLUMN bucket;
        RAISE NOTICE 'Dropped bucket column from audio_files table';
    END IF;

    -- 删除 object_name 字段
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'audio_files' AND column_name = 'object_name') THEN
        ALTER TABLE audio_files DROP COLUMN object_name;
        RAISE NOTICE 'Dropped object_name column from audio_files table';
    END IF;

    -- 删除 sha256_hash 字段
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'audio_files' AND column_name = 'sha256_hash') THEN
        ALTER TABLE audio_files DROP COLUMN sha256_hash;
        RAISE NOTICE 'Dropped sha256_hash column from audio_files table';
    END IF;

    -- 删除 content_type 字段
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'audio_files' AND column_name = 'content_type') THEN
        ALTER TABLE audio_files DROP COLUMN content_type;
        RAISE NOTICE 'Dropped content_type column from audio_files table';
    END IF;

    -- 删除 minio_version_id 字段
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'audio_files' AND column_name = 'minio_version_id') THEN
        ALTER TABLE audio_files DROP COLUMN minio_version_id;
        RAISE NOTICE 'Dropped minio_version_id column from audio_files table';
    END IF;

    -- 删除 storage_class 字段
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'audio_files' AND column_name = 'storage_class') THEN
        ALTER TABLE audio_files DROP COLUMN storage_class;
        RAISE NOTICE 'Dropped storage_class column from audio_files table';
    END IF;

    -- 删除 file_tags 字段
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'audio_files' AND column_name = 'file_tags') THEN
        ALTER TABLE audio_files DROP COLUMN file_tags;
        RAISE NOTICE 'Dropped file_tags column from audio_files table';
    END IF;

    -- 删除 upload_time 字段
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'audio_files' AND column_name = 'upload_time') THEN
        ALTER TABLE audio_files DROP COLUMN upload_time;
        RAISE NOTICE 'Dropped upload_time column from audio_files table';
    END IF;

    -- 删除 is_duplicated 字段
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'audio_files' AND column_name = 'is_duplicated') THEN
        ALTER TABLE audio_files DROP COLUMN is_duplicated;
        RAISE NOTICE 'Dropped is_duplicated column from audio_files table';
    END IF;

    -- 删除 original_filename 字段
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'audio_files' AND column_name = 'original_filename') THEN
        ALTER TABLE audio_files DROP COLUMN original_filename;
        RAISE NOTICE 'Dropped original_filename column from audio_files table';
    END IF;
END $$;

-- 提交事务
COMMIT;

-- 输出完成信息
\echo 'MinIO相关字段回滚完成！'
\echo '已删除以下内容：'
\echo '1. audio_files 表的所有 MinIO 相关字段'
\echo '2. file_versions 表（包括所有数据）'
\echo '3. 相关索引和约束'
\echo '4. 版本管理触发器和函数'
\echo ''
\echo '警告：此操作不可逆，所有MinIO相关数据已被永久删除！'
