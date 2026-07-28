"""
语义编码器模块 - 用于计算语义相似度
"""
from typing import List, Optional
import numpy as np
from config import ENCODER_CONFIG

class SemanticEncoder:
    """语义编码器 - 使用SBERT等模型编码文本"""
    
    def __init__(self):
        self.model_name = ENCODER_CONFIG["model_name"]
        self.device = ENCODER_CONFIG["device"]
        self.batch_size = ENCODER_CONFIG["batch_size"]
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载编码模型"""
        try:
            from sentence_transformers import SentenceTransformer
            import os
            
            # 优先使用本地模型路径
            local_model_path = ENCODER_CONFIG.get("local_model_path")
            
            if local_model_path and os.path.exists(local_model_path):
                print(f"从本地路径加载语义编码模型: {local_model_path}")
                self.model = SentenceTransformer(local_model_path, device=self.device)
                print("模型加载完成（本地）")
                return
            
            # 尝试从本地缓存加载（避免网络请求）
            print(f"尝试加载语义编码模型: {self.model_name}")
            
            # 检查 Hugging Face 缓存目录
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache_path = os.path.join(cache_dir, f"models--sentence-transformers--{self.model_name.replace('/', '--')}")
            
            if os.path.exists(model_cache_path):
                print(f"在本地缓存中找到模型: {model_cache_path}")
                # 查找snapshots目录
                snapshots_dir = os.path.join(model_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        latest_snapshot = os.path.join(snapshots_dir, snapshots[0])
                        print(f"使用缓存快照: {latest_snapshot}")
                        self.model = SentenceTransformer(latest_snapshot, device=self.device)
                        print("模型加载完成（从本地缓存）")
                        return
            
            # 如果本地没有，尝试从网络加载（可能会失败）
            print("本地缓存中未找到模型，尝试从网络加载...")
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                print("模型加载完成（从Hugging Face）")
            except Exception as network_error:
                print(f"网络加载失败: {network_error}")
                print("\n建议解决方案：")
                print("1. 设置环境变量 LOCAL_MODEL_PATH 指向本地模型目录")
                print("2. 或者手动下载模型到本地，然后设置 local_model_path")
                print("3. 或者在有网络的环境下载模型后，复制到 ~/.cache/huggingface/hub/")
                raise network_error
                
        except ImportError:
            print("警告: sentence-transformers未安装，将使用简单的文本特征")
            print("请运行: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            print(f"模型加载失败: {e}，将使用简单特征")
            print("注意：将使用简单的文本特征，可能影响语义相似度计算效果")
            self.model = None
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        编码文本列表
        
        Args:
            texts: 文本列表
            
        Returns:
            编码向量矩阵 (n_samples, embedding_dim)
        """
        if self.model is None:
            # 使用简单的TF-IDF特征作为备用
            return self._simple_encode(texts)
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            print(f"编码失败: {e}，使用简单特征")
            return self._simple_encode(texts)
    
    def _simple_encode(self, texts: List[str]) -> np.ndarray:
        """简单的文本编码（备用方法）"""
        # 使用字符级别的简单特征
        max_len = 512
        vocab_size = 256
        
        features = []
        for text in texts:
            text_bytes = text.encode('utf-8')[:max_len]
            feature = np.zeros(vocab_size)
            for byte in text_bytes:
                feature[byte % vocab_size] += 1
            # 归一化
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            features.append(feature)
        
        return np.array(features)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def batch_cosine_similarity(self, query_vec: np.ndarray, 
                               candidate_vecs: np.ndarray) -> np.ndarray:
        """
        批量计算余弦相似度
        
        Args:
            query_vec: 查询向量 (embedding_dim,)
            candidate_vecs: 候选向量矩阵 (n_candidates, embedding_dim)
            
        Returns:
            相似度数组 (n_candidates,)
        """
        # 归一化
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        candidate_norms = candidate_vecs / (np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-8)
        
        # 计算点积
        similarities = np.dot(candidate_norms, query_norm)
        return similarities

