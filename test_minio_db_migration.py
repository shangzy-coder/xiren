#!/usr/bin/env python3
"""
MinIO数据库迁移测试脚本
测试数据库表结构扩展是否成功
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from app.services.db import DatabaseService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_database_migration():
    """测试数据库迁移功能"""
    print("开始测试MinIO数据库迁移...")
    
    db_service = DatabaseService()
    
    try:
        # 1. 初始化数据库连接
        print("\n1. 初始化数据库连接...")
        success = await db_service.initialize()
        if not success:
            print("❌ 数据库初始化失败！")
            return False
        print("✅ 数据库初始化成功")
        
        # 2. 测试插入带有MinIO字段的音频文件记录
        print("\n2. 测试插入音频文件记录（包含MinIO字段）...")
        
        test_file_id = await db_service.insert_audio_file(
            filename="test_minio_migration.wav",
            file_path="/tmp/test_minio_migration.wav",
            file_size=1024000,
            duration=60.5,
            sample_rate=16000,
            channels=1,
            format="wav",
            metadata={"test": True, "migration": "v1.1"},
            # MinIO相关字段
            bucket="audio-files",
            object_name="test/test_minio_migration_20250922.wav",
            sha256_hash="abcd1234567890abcd1234567890abcd1234567890abcd1234567890abcd1234",
            content_type="audio/wav",
            minio_version_id="version_123456",
            storage_class="STANDARD",
            file_tags={"test": True, "environment": "migration_test"},
            upload_time=datetime.now(),
            is_duplicated=False,
            original_filename="test_minio_migration.wav"
        )
        
        if test_file_id:
            print(f"✅ 成功插入音频文件记录，ID: {test_file_id}")
        else:
            print("❌ 插入音频文件记录失败！")
            return False
        
        # 3. 测试插入文件版本记录
        print("\n3. 测试插入文件版本记录...")
        
        version_id = await db_service.insert_file_version(
            audio_file_id=test_file_id,
            bucket="audio-files",
            object_name="test/test_minio_migration_20250922.wav",
            file_size=1024000,
            sha256_hash="abcd1234567890abcd1234567890abcd1234567890abcd1234567890abcd1234",
            minio_version_id="version_123456",
            is_current=True,
            metadata={"original_size": 1024000, "compression": "none"}
        )
        
        if version_id:
            print(f"✅ 成功插入文件版本记录，ID: {version_id}")
        else:
            print("❌ 插入文件版本记录失败！")
            return False
        
        # 4. 测试根据哈希值查找文件（去重功能）
        print("\n4. 测试根据哈希值查找文件...")
        
        found_file = await db_service.get_audio_file_by_hash(
            "abcd1234567890abcd1234567890abcd1234567890abcd1234567890abcd1234"
        )
        
        if found_file:
            print(f"✅ 成功根据哈希值查找到文件: {found_file['filename']}")
            print(f"   存储桶: {found_file['bucket']}")
            print(f"   对象名: {found_file['object_name']}")
        else:
            print("❌ 根据哈希值查找文件失败！")
            return False
        
        # 5. 测试根据存储桶和对象名查找文件
        print("\n5. 测试根据存储桶和对象名查找文件...")
        
        found_file2 = await db_service.get_audio_file_by_object_name(
            "audio-files", 
            "test/test_minio_migration_20250922.wav"
        )
        
        if found_file2:
            print(f"✅ 成功根据存储桶和对象名查找到文件: {found_file2['filename']}")
        else:
            print("❌ 根据存储桶和对象名查找文件失败！")
            return False
        
        # 6. 测试获取文件版本信息
        print("\n6. 测试获取文件版本信息...")
        
        versions = await db_service.get_file_versions(test_file_id)
        
        if versions:
            print(f"✅ 成功获取文件版本信息，共 {len(versions)} 个版本")
            for version in versions:
                print(f"   版本 {version['version_number']}: {version['object_name']} "
                      f"({'当前' if version['is_current'] else '历史'})")
        else:
            print("❌ 获取文件版本信息失败！")
            return False
        
        # 7. 测试获取当前版本
        print("\n7. 测试获取当前文件版本...")
        
        current_version = await db_service.get_current_file_version(test_file_id)
        
        if current_version:
            print(f"✅ 成功获取当前版本: 版本 {current_version['version_number']}")
        else:
            print("❌ 获取当前文件版本失败！")
            return False
        
        # 8. 测试存储统计功能
        print("\n8. 测试存储统计功能...")
        
        stats = await db_service.get_storage_statistics()
        
        if stats:
            print("✅ 成功获取存储统计信息:")
            print(f"   总文件数: {stats['total_statistics']['total_files']}")
            print(f"   总大小: {stats['total_statistics']['total_size']} 字节")
            print(f"   MinIO文件数: {stats['total_statistics']['minio_files']}")
            print(f"   重复文件数: {stats['total_statistics']['duplicated_files']}")
            
            if stats['bucket_statistics']:
                print("   存储桶统计:")
                for bucket_stat in stats['bucket_statistics']:
                    print(f"     {bucket_stat['bucket']}: {bucket_stat['file_count']} 个文件, "
                          f"{bucket_stat['total_size']} 字节")
        else:
            print("❌ 获取存储统计信息失败！")
            return False
        
        # 9. 测试更新MinIO信息
        print("\n9. 测试更新MinIO信息...")
        
        update_success = await db_service.update_audio_file_minio_info(
            file_id=test_file_id,
            storage_class="COLD",
            file_tags={"test": True, "environment": "migration_test", "updated": True}
        )
        
        if update_success:
            print("✅ 成功更新MinIO信息")
        else:
            print("❌ 更新MinIO信息失败！")
            return False
        
        # 10. 测试第二个版本（自动版本号递增）
        print("\n10. 测试插入第二个文件版本...")
        
        version_id2 = await db_service.insert_file_version(
            audio_file_id=test_file_id,
            bucket="audio-files",
            object_name="test/test_minio_migration_20250922_v2.wav",
            file_size=1024000,
            sha256_hash="efgh5678901234efgh5678901234efgh5678901234efgh5678901234efgh5678",
            minio_version_id="version_789012",
            is_current=True,  # 这应该会自动将之前的版本设为非当前
            metadata={"original_size": 1024000, "compression": "gzip"}
        )
        
        if version_id2:
            print(f"✅ 成功插入第二个文件版本，ID: {version_id2}")
            
            # 验证版本管理
            updated_versions = await db_service.get_file_versions(test_file_id)
            current_count = sum(1 for v in updated_versions if v['is_current'])
            if current_count == 1:
                print("✅ 版本管理正常，只有一个当前版本")
            else:
                print(f"❌ 版本管理异常，有 {current_count} 个当前版本")
                return False
        else:
            print("❌ 插入第二个文件版本失败！")
            return False
        
        # 11. 清理测试数据
        print("\n11. 清理测试数据...")
        
        async with db_service.pool.acquire() as conn:
            # 删除测试记录（因为有外键约束，file_versions会自动删除）
            await conn.execute("DELETE FROM audio_files WHERE filename = $1", 
                              "test_minio_migration.wav")
            print("✅ 清理测试数据完成")
        
        print("\n🎉 所有测试通过！MinIO数据库迁移验证成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        logger.exception("测试失败")
        return False
        
    finally:
        # 关闭数据库连接
        await db_service.close()


async def test_database_structure():
    """测试数据库结构是否正确"""
    print("\n检查数据库表结构...")
    
    db_service = DatabaseService()
    
    try:
        await db_service.initialize()
        
        async with db_service.pool.acquire() as conn:
            # 检查 audio_files 表的新字段
            audio_files_columns = await conn.fetch("""
                SELECT column_name, data_type, character_maximum_length, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'audio_files' 
                AND column_name IN (
                    'bucket', 'object_name', 'sha256_hash', 'content_type', 
                    'minio_version_id', 'storage_class', 'file_tags', 
                    'upload_time', 'is_duplicated', 'original_filename'
                )
                ORDER BY column_name
            """)
            
            print(f"✅ audio_files 表新增字段 ({len(audio_files_columns)} 个):")
            for col in audio_files_columns:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
                print(f"   {col['column_name']}: {col['data_type']}"
                      f"({col['character_maximum_length'] or ''}) {nullable}{default}")
            
            # 检查 file_versions 表
            file_versions_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'file_versions'
                )
            """)
            
            if file_versions_exists:
                print("✅ file_versions 表已创建")
                
                # 获取表结构
                file_versions_columns = await conn.fetch("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = 'file_versions'
                    ORDER BY ordinal_position
                """)
                
                print(f"   字段 ({len(file_versions_columns)} 个):")
                for col in file_versions_columns:
                    nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                    default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
                    print(f"     {col['column_name']}: {col['data_type']} {nullable}{default}")
            else:
                print("❌ file_versions 表未创建")
                return False
            
            # 检查索引
            indexes = await conn.fetch("""
                SELECT indexname, tablename
                FROM pg_indexes 
                WHERE tablename IN ('audio_files', 'file_versions')
                AND (
                    indexname LIKE 'idx_audio_files_%' 
                    OR indexname LIKE 'idx_file_versions_%'
                )
                ORDER BY tablename, indexname
            """)
            
            print(f"✅ 相关索引 ({len(indexes)} 个):")
            for idx in indexes:
                print(f"   {idx['indexname']} ({idx['tablename']})")
                
    except Exception as e:
        print(f"❌ 检查数据库结构失败: {e}")
        return False
        
    finally:
        await db_service.close()
    
    return True


async def main():
    """主函数"""
    print("=" * 60)
    print("MinIO 数据库迁移验证测试")
    print("=" * 60)
    
    # 测试数据库结构
    structure_ok = await test_database_structure()
    if not structure_ok:
        print("\n❌ 数据库结构检查失败，请先运行迁移脚本")
        return
    
    # 测试功能
    function_ok = await test_database_migration()
    if not function_ok:
        print("\n❌ 功能测试失败")
        return
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！数据库迁移验证成功！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
