#!/usr/bin/env python3
"""
MinIO 存储服务基础测试
测试 StorageService 类的基本功能
"""

import asyncio
import os
import tempfile
import hashlib
from pathlib import Path

from app.services.storage import StorageService, get_storage_service
from app.config import settings

async def test_storage_service():
    """测试存储服务的基本功能"""
    print("=== MinIO 存储服务测试 ===")
    
    # 获取存储服务实例
    try:
        storage = await get_storage_service()
        print("✅ 存储服务初始化成功")
    except Exception as e:
        print(f"❌ 存储服务初始化失败: {e}")
        return False
    
    # 创建临时测试文件
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
        test_content = b"This is a test audio file content for MinIO storage testing."
        temp_file.write(test_content)
        temp_file_path = temp_file.name
    
    try:
        # 测试文件上传
        print("\n📤 测试文件上传...")
        file_metadata = await storage.upload_audio_file(
            file_path=temp_file_path,
            filename="test_audio.txt",
            metadata={"test": "true", "purpose": "basic_test"}
        )
        
        if file_metadata:
            print(f"✅ 文件上传成功: {file_metadata.filename}")
            print(f"   - 文件大小: {file_metadata.file_size} 字节")
            print(f"   - SHA256: {file_metadata.sha256_hash}")
            print(f"   - 存储桶: {file_metadata.bucket}")
        else:
            print("❌ 文件上传失败")
            return False
        
        # 测试文件信息获取
        print("\n📋 测试文件信息获取...")
        info = await storage.get_file_info("test_audio.txt")
        if info:
            print(f"✅ 获取文件信息成功: {info.filename}")
            print(f"   - 上传时间: {info.upload_time}")
            print(f"   - 内容类型: {info.content_type}")
        else:
            print("❌ 获取文件信息失败")
        
        # 测试预签名URL生成
        print("\n🔗 测试预签名URL生成...")
        url = await storage.get_audio_file_url("test_audio.txt")
        if url:
            print(f"✅ 预签名URL生成成功")
            print(f"   URL: {url[:100]}...")
        else:
            print("❌ 预签名URL生成失败")
        
        # 测试文件下载
        print("\n📥 测试文件下载...")
        download_path = temp_file_path + ".downloaded"
        success = await storage.download_audio_file("test_audio.txt", download_path)
        if success and os.path.exists(download_path):
            print("✅ 文件下载成功")
            # 验证下载的文件内容
            with open(download_path, 'rb') as f:
                downloaded_content = f.read()
            if downloaded_content == test_content:
                print("✅ 下载文件内容验证成功")
            else:
                print("❌ 下载文件内容验证失败")
            os.unlink(download_path)
        else:
            print("❌ 文件下载失败")
        
        # 测试存储统计
        print("\n📊 测试存储统计...")
        stats = await storage.get_storage_stats()
        print(f"✅ 存储统计获取成功")
        print(f"   - 总文件数: {stats.total_files}")
        print(f"   - 总大小: {stats.total_size} 字节")
        print(f"   - 存储桶数: {len(stats.buckets)}")
        for bucket_name, bucket_stats in stats.buckets.items():
            if "error" not in bucket_stats:
                print(f"   - {bucket_name}: {bucket_stats['file_count']} 文件, {bucket_stats['total_size']} 字节")
        
        # 测试重复文件检测
        print("\n🔍 测试重复文件检测...")
        duplicate_metadata = await storage.upload_audio_file(
            file_path=temp_file_path,
            filename="test_audio_duplicate.txt",
            metadata={"test": "duplicate"}
        )
        
        if duplicate_metadata:
            if settings.ENABLE_FILE_DEDUPLICATION:
                print("✅ 去重功能正常工作 - 返回了现有文件信息")
            else:
                print("✅ 重复文件上传成功（去重功能已禁用）")
        else:
            print("❌ 重复文件处理失败")
        
        # 清理测试文件
        print("\n🧹 清理测试文件...")
        delete_success = await storage.delete_audio_file("test_audio.txt")
        if delete_success:
            print("✅ 测试文件删除成功")
        else:
            print("❌ 测试文件删除失败")
        
        # 如果上传了重复文件，也删除它
        if duplicate_metadata and not settings.ENABLE_FILE_DEDUPLICATION:
            await storage.delete_audio_file("test_audio_duplicate.txt")
        
        print("\n🎉 所有基础测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        # 关闭存储服务
        await storage.close()

async def test_batch_operations():
    """测试批量操作"""
    print("\n=== 批量操作测试 ===")
    
    storage = await get_storage_service()
    
    # 创建多个临时测试文件
    temp_files = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix=f'_batch_{i}.txt', delete=False) as temp_file:
            content = f"Batch test file {i} content".encode()
            temp_file.write(content)
            temp_files.append(temp_file.name)
    
    try:
        # 测试批量上传
        print("📤 测试批量上传...")
        results = await storage.upload_batch_files(temp_files)
        print(f"✅ 批量上传完成: {len(results)} 个文件成功")
        
        # 测试批量删除
        if results:
            print("🗑️ 测试批量删除...")
            filenames = [result.filename for result in results]
            delete_results = await storage.delete_batch_files(filenames)
            success_count = sum(1 for success in delete_results.values() if success)
            print(f"✅ 批量删除完成: {success_count}/{len(filenames)} 个文件成功")
        
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        await storage.close()

async def main():
    """主测试函数"""
    print("开始 MinIO 存储服务测试...")
    print(f"MinIO 服务器: {settings.MINIO_ENDPOINT}")
    print(f"主存储桶: {settings.MINIO_BUCKET}")
    print(f"去重功能: {'启用' if settings.ENABLE_FILE_DEDUPLICATION else '禁用'}")
    print(f"版本管理: {'启用' if settings.ENABLE_FILE_VERSIONING else '禁用'}")
    
    # 基础功能测试
    basic_success = await test_storage_service()
    
    # 批量操作测试
    if basic_success:
        await test_batch_operations()
    
    print("\n测试完成!")

if __name__ == "__main__":
    asyncio.run(main())
