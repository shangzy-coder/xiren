-- 数据库迁移验证脚本
-- 版本：v1.1 验证脚本
-- 创建时间：2025-09-22
-- 用途：验证MinIO字段迁移是否成功

\echo '开始验证MinIO数据库迁移...'
\echo ''

-- 1. 验证 audio_files 表的新字段
\echo '1. 验证 audio_files 表结构:'
SELECT 
    column_name,
    data_type,
    character_maximum_length,
    column_default,
    is_nullable
FROM information_schema.columns 
WHERE table_name = 'audio_files' 
AND column_name IN (
    'bucket', 'object_name', 'sha256_hash', 'content_type', 
    'minio_version_id', 'storage_class', 'file_tags', 
    'upload_time', 'is_duplicated', 'original_filename'
)
ORDER BY column_name;

\echo ''

-- 2. 验证 file_versions 表是否创建成功
\echo '2. 验证 file_versions 表结构:'
SELECT 
    column_name,
    data_type,
    character_maximum_length,
    column_default,
    is_nullable
FROM information_schema.columns 
WHERE table_name = 'file_versions'
ORDER BY ordinal_position;

\echo ''

-- 3. 验证索引是否创建成功
\echo '3. 验证索引创建情况:'
SELECT 
    indexname,
    tablename,
    indexdef
FROM pg_indexes 
WHERE tablename IN ('audio_files', 'file_versions')
AND (
    indexname LIKE 'idx_audio_files_%' 
    OR indexname LIKE 'idx_file_versions_%'
)
ORDER BY tablename, indexname;

\echo ''

-- 4. 验证约束是否添加成功
\echo '4. 验证约束情况:'
SELECT 
    constraint_name,
    table_name,
    constraint_type
FROM information_schema.table_constraints 
WHERE table_name IN ('audio_files', 'file_versions')
AND (
    constraint_name LIKE 'check_%'
    OR constraint_name LIKE 'fk_%'
    OR constraint_name LIKE 'unique_%'
)
ORDER BY table_name, constraint_name;

\echo ''

-- 5. 验证触发器是否创建成功
\echo '5. 验证触发器情况:'
SELECT 
    trigger_name,
    event_object_table,
    action_timing,
    event_manipulation
FROM information_schema.triggers 
WHERE event_object_table = 'file_versions'
ORDER BY trigger_name;

\echo ''

-- 6. 验证函数是否创建成功
\echo '6. 验证函数情况:'
SELECT 
    routine_name,
    routine_type,
    data_type
FROM information_schema.routines 
WHERE routine_name = 'update_file_version_number';

\echo ''

-- 7. 测试基本功能
\echo '7. 测试基本数据操作:'

-- 7.1 测试插入audio_files记录
\echo '   7.1 测试插入 audio_files 记录...'
DO $$
DECLARE
    test_file_id INTEGER;
BEGIN
    -- 插入测试记录
    INSERT INTO audio_files (
        filename, bucket, object_name, sha256_hash, 
        content_type, storage_class, file_tags, 
        upload_time, is_duplicated, original_filename
    ) VALUES (
        'test_migration.wav', 
        'audio-files', 
        'test/test_migration_20250922.wav',
        'abcd1234567890abcd1234567890abcd1234567890abcd1234567890abcd1234',
        'audio/wav',
        'STANDARD',
        '{"test": true, "migration": "v1.1"}',
        CURRENT_TIMESTAMP,
        false,
        'test_migration.wav'
    ) RETURNING id INTO test_file_id;
    
    RAISE NOTICE 'Successfully inserted test audio file with ID: %', test_file_id;
    
    -- 7.2 测试插入file_versions记录
    INSERT INTO file_versions (
        audio_file_id, bucket, object_name, file_size,
        sha256_hash, minio_version_id, is_current, metadata
    ) VALUES (
        test_file_id,
        'audio-files',
        'test/test_migration_20250922.wav',
        1024000,
        'abcd1234567890abcd1234567890abcd1234567890abcd1234567890abcd1234',
        'version_123456',
        true,
        '{"original_size": 1024000, "compression": "none"}'
    );
    
    RAISE NOTICE 'Successfully inserted test file version';
    
    -- 7.3 验证自动版本号功能
    INSERT INTO file_versions (
        audio_file_id, bucket, object_name, file_size,
        sha256_hash, minio_version_id, is_current, metadata
    ) VALUES (
        test_file_id,
        'audio-files',
        'test/test_migration_20250922_v2.wav',
        1024000,
        'efgh5678901234efgh5678901234efgh5678901234efgh5678901234efgh5678',
        'version_789012',
        true,
        '{"original_size": 1024000, "compression": "gzip"}'
    );
    
    RAISE NOTICE 'Successfully inserted second file version (should auto-increment version number)';
    
    -- 7.4 查看测试结果
    RAISE NOTICE 'Test data inserted successfully';
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Error during test data insertion: %', SQLERRM;
        ROLLBACK;
END $$;

-- 7.5 查看插入的测试数据
\echo '   7.5 查看测试数据:'
SELECT 
    af.id,
    af.filename,
    af.bucket,
    af.object_name,
    af.sha256_hash,
    af.content_type,
    af.is_duplicated
FROM audio_files af 
WHERE af.filename = 'test_migration.wav';

SELECT 
    fv.id,
    fv.audio_file_id,
    fv.version_number,
    fv.bucket,
    fv.object_name,
    fv.is_current,
    fv.minio_version_id
FROM file_versions fv 
JOIN audio_files af ON fv.audio_file_id = af.id
WHERE af.filename = 'test_migration.wav'
ORDER BY fv.version_number;

\echo ''

-- 8. 测试查询性能（简单测试）
\echo '8. 测试索引查询性能:'
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM audio_files WHERE sha256_hash = 'abcd1234567890abcd1234567890abcd1234567890abcd1234567890abcd1234';

EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM file_versions WHERE is_current = true;

\echo ''

-- 9. 清理测试数据
\echo '9. 清理测试数据:'
DELETE FROM audio_files WHERE filename = 'test_migration.wav';

\echo ''
\echo '验证完成！'
\echo ''
\echo '检查上述输出结果：'
\echo '- 确保所有MinIO字段已添加到 audio_files 表'
\echo '- 确保 file_versions 表已创建'
\echo '- 确保所有索引已创建'
\echo '- 确保约束和触发器正常工作'
\echo '- 确保基本数据操作正常'
