"""
Embedding 缓存工具 - 提前计算并存储数据的 embedding
"""
import json
import os
import numpy as np
import random
from typing import List, Dict, Optional
from semantic_encoder import SemanticEncoder
from graph_builder import GraphBuilder
from config import DATA_PATHS

class EmbeddingCache:
    """Embedding 缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.encoder = SemanticEncoder()
        self.graph_builder = GraphBuilder()
        
        # 缓存文件路径
        self.embeddings_file = os.path.join(cache_dir, "embeddings_cache.npz")
        self.metadata_file = os.path.join(cache_dir, "embeddings_metadata.jsonl")
    
    def prepare_icl_pool(self, pool_size: Optional[int] = None, force_recompute: bool = False) -> List[Dict]:
        """
        准备 ICL 候选池（随机选择数据集的1/8）并计算 embedding
        
        Args:
            pool_size: 候选池大小（如果为None，则使用数据集的1/8）
            force_recompute: 是否强制重新计算
            
        Returns:
            候选池数据列表，每个元素包含 text, label, graph, embedding
        """
        # 检查是否已有缓存
        if not force_recompute and os.path.exists(self.embeddings_file) and os.path.exists(self.metadata_file):
            print(f"从缓存加载 ICL 候选池...")
            return self._load_from_cache()
        
        # 加载数据
        from data_loader import DataLoader
        data_loader = DataLoader()
        
        # 加载所有数据（不限制数量）
        print("加载所有数据...")
        all_fake_data = data_loader.load_misinformation()
        all_real_data = data_loader.load_verified_information()
        
        # 过滤出有文本的数据
        fake_data_with_text = []
        for item in all_fake_data:
            article = item.get("article", {})
            news_text = article.get("article_content", "")
            if news_text:
                fake_data_with_text.append({
                    "item": item,
                    "text": news_text,
                    "label": 0,
                    "label_str": "假新闻"
                })
        
        real_data_with_text = []
        for item in all_real_data:
            article = item.get("article", {})
            news_text = article.get("article_content", "")
            if news_text:
                real_data_with_text.append({
                    "item": item,
                    "text": news_text,
                    "label": 1,
                    "label_str": "真新闻"
                })
        
        # 计算总数据集大小
        total_size = len(fake_data_with_text) + len(real_data_with_text)
        
        # 如果pool_size为None，使用数据集的1/8
        if pool_size is None:
            pool_size = max(1, total_size // 40)
        
        # 确保pool_size不超过总数据集大小
        pool_size = min(pool_size, total_size)
        
        # 从真假新闻中各随机选择一半
        fake_count = pool_size // 2
        real_count = pool_size - fake_count  # 确保总数是pool_size
        
        # 确保不超过可用数据量
        fake_count = min(fake_count, len(fake_data_with_text))
        real_count = min(real_count, len(real_data_with_text))
        
        # 如果某一类数据不足，用另一类补充
        if fake_count + real_count < pool_size:
            if fake_count < len(fake_data_with_text):
                fake_count = min(pool_size - real_count, len(fake_data_with_text))
            else:
                real_count = min(pool_size - fake_count, len(real_data_with_text))
        
        print(f"数据集总大小: {total_size} 条（假新闻: {len(fake_data_with_text)}, 真新闻: {len(real_data_with_text)}）")
        print(f"随机选择 {pool_size} 条数据（假新闻: {fake_count}, 真新闻: {real_count}）作为 ICL 候选池...")
        
        # 随机选择
        random.seed(42)  # 设置随机种子以确保可重复性
        selected_fake = random.sample(fake_data_with_text, fake_count)
        selected_real = random.sample(real_data_with_text, real_count)
        
        # 合并数据
        all_data = selected_fake + selected_real
        
        # 再次随机打乱，确保真假新闻混合
        random.shuffle(all_data)
        
        print(f"已准备 {len(all_data)} 条数据")
        
        # 提取文本并计算 embedding
        texts = [d["text"] for d in all_data]
        print("计算 embedding...")
        embeddings = self.encoder.encode(texts)
        print(f"Embedding 计算完成，形状: {embeddings.shape}")
        
        # 构建图并准备候选池
        pool = []
        print("构建传播图...")
        for i, data in enumerate(all_data):
            item = data["item"]
            news_text = data["text"]
            news_id = f"news_{hash(news_text) % 1000000}"
            
            # 检查是否有增强数据
            has_enhanced_data = ("enhanced_nodes" in item or "enhanced_edges" in item)
            
            if has_enhanced_data:
                graph = self.graph_builder.build_enhanced_from_data(item, news_id)
            else:
                graph = self.graph_builder.build_from_data(item, news_id)
            
            pool.append({
                "text": news_text,
                "label": data["label"],
                "label_str": data["label_str"],
                "graph": graph,
                "embedding": embeddings[i],
                "news_id": news_id,
                "item": item  # 添加item字段，供evolve_meta_paths使用
            })
        
        # 保存到缓存
        print("保存到缓存...")
        self._save_to_cache(pool, embeddings)
        print(f"缓存保存完成: {self.cache_dir}")
        
        return pool
    
    def _save_to_cache(self, pool: List[Dict], embeddings: np.ndarray):
        """保存到缓存文件"""
        # 保存 embedding 矩阵
        np.savez_compressed(self.embeddings_file, embeddings=embeddings)
        
        # 保存元数据（不包含 graph 和 embedding，因为 graph 太大）
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            for data in pool:
                metadata = {
                    "text": data["text"],
                    "label": data["label"],
                    "label_str": data["label_str"],
                    "news_id": data["news_id"]
                }
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
    
    def _load_from_cache(self) -> List[Dict]:
        """从缓存文件加载"""
        # 加载 embedding
        cache_data = np.load(self.embeddings_file)
        embeddings = cache_data["embeddings"]
        
        # 加载元数据
        metadata_list = []
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    metadata_list.append(json.loads(line))
        
        # 重新构建图（因为图没有缓存）
        print("从原始数据重新构建图...")
        from data_loader import DataLoader
        data_loader = DataLoader()
        
        fake_data = data_loader.load_misinformation()
        real_data = data_loader.load_verified_information()
        
        # 创建文本到数据的映射
        text_to_item = {}
        for item in fake_data + real_data:
            article = item.get("article", {})
            news_text = article.get("article_content", "")
            if news_text:
                text_to_item[news_text] = item
        
        pool = []
        for i, metadata in enumerate(metadata_list):
            text = metadata["text"]
            news_id = metadata["news_id"]
            
            # 查找对应的原始数据
            item = text_to_item.get(text)
            if item is None:
                print(f"警告: 未找到文本对应的原始数据: {text[:50]}...")
                continue
            
            # 确保item字段被包含
            
            # 构建图
            has_enhanced_data = ("enhanced_nodes" in item or "enhanced_edges" in item)
            if has_enhanced_data:
                graph = self.graph_builder.build_enhanced_from_data(item, news_id)
            else:
                graph = self.graph_builder.build_from_data(item, news_id)
            
            pool.append({
                "text": text,
                "label": metadata["label"],
                "label_str": metadata["label_str"],
                "graph": graph,
                "embedding": embeddings[i],
                "news_id": news_id,
                "item": item  # 添加item字段，供evolve_meta_paths使用
            })
        
        print(f"从缓存加载了 {len(pool)} 条数据")
        return pool


def main():
    """主函数 - 用于独立运行embedding cache"""
    import argparse
    
    parser = argparse.ArgumentParser(description="准备 ICL 候选池并计算 embedding")
    parser.add_argument("--pool_size", type=int, default=None,
                       help="候选池大小（如果为None，则使用数据集的1/8）")
    parser.add_argument("--force_recompute", action="store_true",
                       help="强制重新计算，即使缓存已存在")
    parser.add_argument("--cache_dir", type=str, default="cache",
                       help="缓存目录（默认: cache）")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Embedding Cache 工具")
    print("="*80)
    
    # 创建 EmbeddingCache 实例
    embedding_cache = EmbeddingCache(cache_dir=args.cache_dir)
    
    # 准备 ICL 候选池
    pool = embedding_cache.prepare_icl_pool(
        pool_size=args.pool_size,
        force_recompute=args.force_recompute
    )
    
    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"候选池大小: {len(pool)} 条")
    print(f"缓存目录: {args.cache_dir}")
    print(f"Embedding 文件: {embedding_cache.embeddings_file}")
    print(f"元数据文件: {embedding_cache.metadata_file}")


if __name__ == "__main__":
    main()

