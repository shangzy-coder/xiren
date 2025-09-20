"""
pytest配置和fixture定义
"""
import asyncio
import os
import tempfile
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.services.db import get_async_session, Base


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """创建测试客户端"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """创建异步测试客户端"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_audio_file():
    """模拟音频文件"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # 创建一个简单的WAV文件头
        wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00'
        # 添加一些音频数据
        audio_data = b'\x00\x00' * 1000  # 1000个样本的静音
        tmp.write(wav_header + audio_data)
        tmp.flush()
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def mock_model():
    """模拟模型"""
    mock = MagicMock()
    mock.transcribe = MagicMock(return_value="测试转录结果")
    mock.extract_speaker_embedding = MagicMock(return_value=[0.1] * 512)
    return mock


@pytest.fixture
def mock_database():
    """模拟数据库会话"""
    # 创建内存数据库
    SQLALCHEMY_TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
    engine = create_async_engine(
        SQLALCHEMY_TEST_DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
    
    async def get_test_session():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        TestingSessionLocal = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async with TestingSessionLocal() as session:
            yield session
    
    return get_test_session


@pytest.fixture
def override_dependencies():
    """覆盖应用依赖项"""
    overrides = {}
    
    def _override(dependency, override_with):
        overrides[dependency] = override_with
        app.dependency_overrides[dependency] = override_with
    
    yield _override
    
    # 清理
    for dependency in overrides:
        if dependency in app.dependency_overrides:
            del app.dependency_overrides[dependency]


@pytest.fixture
def mock_queue_manager():
    """模拟队列管理器"""
    mock = AsyncMock()
    mock.submit_task = AsyncMock(return_value="test-task-id")
    mock.get_task_status = AsyncMock(return_value={
        "task_id": "test-task-id",
        "status": "completed",
        "result": {"text": "测试结果"}
    })
    mock.get_stats = AsyncMock(return_value={
        "total_tasks": 10,
        "completed_tasks": 8,
        "failed_tasks": 1,
        "pending_tasks": 1
    })
    return mock


@pytest.fixture
def mock_websocket_manager():
    """模拟WebSocket管理器"""
    mock = AsyncMock()
    mock.connect = AsyncMock(return_value="connection-id")
    mock.disconnect = AsyncMock()
    mock.get_stats = AsyncMock(return_value={
        "active_connections": 2,
        "total_connections": 10,
        "total_messages": 100
    })
    return mock


@pytest.fixture
def mock_speaker_pool():
    """模拟声纹池"""
    mock = AsyncMock()
    mock.register_speaker = AsyncMock(return_value="speaker-id")
    mock.identify_speaker = AsyncMock(return_value={
        "speaker_id": "speaker-id",
        "similarity": 0.85,
        "confidence": 0.9
    })
    mock.get_all_speakers = AsyncMock(return_value=[
        {"id": "speaker-1", "name": "测试说话人1"},
        {"id": "speaker-2", "name": "测试说话人2"}
    ])
    return mock


@pytest.fixture
def sample_audio_data():
    """样本音频数据"""
    import numpy as np
    # 生成1秒16kHz的正弦波
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4音符
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio.tobytes()


# 环境变量模拟
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """模拟环境变量"""
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test")
    monkeypatch.setenv("ENABLE_METRICS", "false")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")