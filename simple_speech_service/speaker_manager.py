"""
说话人管理器
负责说话人的注册、识别和管理
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
from datetime import datetime
import pickle

from .config import settings

class SpeakerManager:
    """说话人管理器"""

    def __init__(self, storage_type: str = 'memory'):
        """
        初始化说话人管理器

        Args:
            storage_type: 存储类型 ('memory' 或 'file')
        """
        self.storage_type = storage_type
        self.speakers = {}  # name -> speaker_info
        self.embeddings = {}  # name -> embedding_vector

        # 确保数据目录存在
        settings.data_dir.mkdir(exist_ok=True)
        settings.embeddings_dir.mkdir(exist_ok=True)

        # 加载已存在的说话人数据
        self._load_speakers()

        print(f"说话人管理器初始化完成，存储类型: {storage_type}")

    def _load_speakers(self):
        """加载说话人数据"""
        try:
            if settings.speakers_file.exists():
                with open(settings.speakers_file, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)

                for name, info in speaker_data.items():
                    self.speakers[name] = info

                    # 加载向量数据
                    if self.storage_type == 'file':
                        embedding_file = settings.embeddings_dir / f"{name}.pkl"
                        if embedding_file.exists():
                            with open(embedding_file, 'rb') as f:
                                self.embeddings[name] = pickle.load(f)
                    # memory模式下，embeddings会在register时加载

                print(f"加载了 {len(self.speakers)} 个已注册说话人")
            else:
                print("没有找到已保存的说话人数据，将创建新的数据文件")
        except Exception as e:
            print(f"加载说话人数据失败: {e}")

    def _save_speakers(self):
        """保存说话人数据"""
        try:
            # 保存说话人信息
            with open(settings.speakers_file, 'w', encoding='utf-8') as f:
                json.dump(self.speakers, f, ensure_ascii=False, indent=2)

            # 保存向量数据（文件模式）
            if self.storage_type == 'file':
                for name, embedding in self.embeddings.items():
                    embedding_file = settings.embeddings_dir / f"{name}.pkl"
                    with open(embedding_file, 'wb') as f:
                        pickle.dump(embedding, f)

            print(f"保存了 {len(self.speakers)} 个说话人数据")
        except Exception as e:
            print(f"保存说话人数据失败: {e}")
            raise

    def register_speaker(self, name: str, embedding: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """
        注册说话人

        Args:
            name: 说话人姓名
            embedding: 说话人向量
            metadata: 附加元数据

        Returns:
            bool: 注册是否成功
        """
        try:
            if name in self.speakers:
                print(f"说话人 '{name}' 已存在，将更新信息")
            else:
                print(f"注册新说话人: {name}")

            # 保存向量
            self.embeddings[name] = embedding.copy()

            # 保存说话人信息
            speaker_info = {
                'name': name,
                'embedding_dim': len(embedding),
                'registered_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            self.speakers[name] = speaker_info

            # 保存到文件
            self._save_speakers()

            print(f"说话人 '{name}' 注册成功")
            return True

        except Exception as e:
            print(f"注册说话人 '{name}' 失败: {e}")
            return False

    def identify_speaker(self, embedding: np.ndarray, threshold: float = None) -> Tuple[Optional[str], float]:
        """
        识别说话人

        Args:
            embedding: 待识别的向量
            threshold: 相似度阈值

        Returns:
            Tuple[Optional[str], float]: (说话人姓名, 相似度)，如果没有匹配返回(None, 0.0)
        """
        if threshold is None:
            threshold = settings.SPEAKER_THRESHOLD

        if not self.embeddings:
            return None, 0.0

        best_match = None
        best_similarity = 0.0

        for name, registered_embedding in self.embeddings.items():
            # 计算余弦相似度
            similarity = np.dot(embedding, registered_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(registered_embedding)
            )

            if similarity > best_similarity and similarity >= threshold:
                best_match = name
                best_similarity = similarity

        return best_match, best_similarity

    def list_speakers(self) -> List[Dict[str, Any]]:
        """
        列出所有已注册的说话人

        Returns:
            List[Dict]: 说话人信息列表
        """
        return list(self.speakers.values())

    def get_speaker_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取特定说话人的信息

        Args:
            name: 说话人姓名

        Returns:
            Optional[Dict]: 说话人信息，如果不存在返回None
        """
        return self.speakers.get(name)

    def remove_speaker(self, name: str) -> bool:
        """
        删除说话人

        Args:
            name: 说话人姓名

        Returns:
            bool: 删除是否成功
        """
        try:
            if name not in self.speakers:
                return False

            # 删除内存中的数据
            if name in self.embeddings:
                del self.embeddings[name]
            del self.speakers[name]

            # 删除文件中的数据
            if self.storage_type == 'file':
                embedding_file = settings.embeddings_dir / f"{name}.pkl"
                if embedding_file.exists():
                    embedding_file.unlink()

            # 保存更新
            self._save_speakers()

            print(f"说话人 '{name}' 已删除")
            return True

        except Exception as e:
            print(f"删除说话人 '{name}' 失败: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            'total_speakers': len(self.speakers),
            'storage_type': self.storage_type,
            'embedding_dim': list(self.embeddings.values())[0].shape[0] if self.embeddings else 0,
            'speakers_file': str(settings.speakers_file),
            'embeddings_dir': str(settings.embeddings_dir)
        }