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
        准备 ICL 监督池并计算 embedding。

        论文 §5.1：从数据集中随机采样 100 条带标签样本作为监督池，
        因此默认池大小固定为 100（显式传入 pool_size 时以传入值为准）。

        Args:
            pool_size: 候选池大小；为 None 时使用默认值 100
            force_recompute: 是否强制重新计算

        Returns:
            候选池数据列表，每个元素包含 text, label, graph, embedding
        """
        from data_loader import DataLoader
        data_loader = DataLoader()

        # 1) 判断数据源：存在增强数据集则优先使用，使“池图”与“查询图”同为增强图
        #    （论文 Alg.1 要求对 D_L 中每条 v 都先用 L_A 增强，再计算 fitness）
        source = "raw"
        enhanced_items = None
        try:
            enhanced_items = data_loader.load_enhanced_dataset()
            if enhanced_items:
                source = "enhanced"
        except Exception:
            enhanced_items = None

        # 2) 目标池大小：未指定时按论文设为 100
        requested_size = 100 if pool_size is None else pool_size

        # 3) 缓存命中需同时满足：池大小、数据集、数据源一致；否则重建
        if (not force_recompute and os.path.exists(self.embeddings_file)
                and os.path.exists(self.metadata_file)
                and self._cache_matches(requested_size, source)):
            print("从缓存加载 ICL 候选池...")
            return self._load_from_cache()
        if os.path.exists(self.metadata_file) and not force_recompute:
            print(f"缓存与目标不一致（目标池大小={requested_size}, 数据源={source}），重新构建候选池...")

        # 4) 组织带文本、带标签的样本（优先增强数据，保证池图已增强）
        fake_data_with_text = []
        real_data_with_text = []
        if source == "enhanced":
            print("从增强数据集采样 ICL 监督池（池图与查询图口径一致）...")
            for item in enhanced_items:
                news_text = item.get("article", {}).get("article_content", "")
                if not news_text:
                    continue
                label = item.get("label", 0)
                rec = {"item": item, "text": news_text, "label": label,
                       "label_str": "假新闻" if label == 0 else "真新闻"}
                (fake_data_with_text if label == 0 else real_data_with_text).append(rec)
        else:
            print("未找到增强数据集，回退为从原始数据采样（池图未增强）...")
            for item in data_loader.load_misinformation():
                news_text = item.get("article", {}).get("article_content", "")
                if news_text:
                    fake_data_with_text.append({"item": item, "text": news_text, "label": 0, "label_str": "假新闻"})
            for item in data_loader.load_verified_information():
                news_text = item.get("article", {}).get("article_content", "")
                if news_text:
                    real_data_with_text.append({"item": item, "text": news_text, "label": 1, "label_str": "真新闻"})

        total_size = len(fake_data_with_text) + len(real_data_with_text)

        # 确保pool_size不超过总数据集大小
        if pool_size is None:
            pool_size = 100
        pool_size = min(pool_size, total_size)

        # 从真假新闻中各随机选择一半
        fake_count = pool_size // 2
        real_count = pool_size - fake_count

        # 确保不超过可用数据量，并在某类不足时用另一类补充
        fake_count = min(fake_count, len(fake_data_with_text))
        real_count = min(real_count, len(real_data_with_text))
        if fake_count + real_count < pool_size:
            if fake_count < len(fake_data_with_text):
                fake_count = min(pool_size - real_count, len(fake_data_with_text))
            else:
                real_count = min(pool_size - fake_count, len(real_data_with_text))

        print(f"数据源: {source} | 总样本: {total_size}（假:{len(fake_data_with_text)}, 真:{len(real_data_with_text)}）")
        print(f"随机选择 {pool_size} 条（假新闻: {fake_count}, 真新闻: {real_count}）作为 ICL 监督池...")

        # 随机选择（固定种子，保证可复现）
        random.seed(42)
        selected_fake = random.sample(fake_data_with_text, fake_count) if fake_count else []
        selected_real = random.sample(real_data_with_text, real_count) if real_count else []
        all_data = selected_fake + selected_real
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
        
        # 保存到缓存（记录数据源，便于下次校验）
        print("保存到缓存...")
        self._save_to_cache(pool, embeddings, source)
        print(f"缓存保存完成: {self.cache_dir}")
        
        return pool
    
    def _save_to_cache(self, pool: List[Dict], embeddings: np.ndarray, source: str = "raw"):
        """保存到缓存文件（含数据源标记）"""
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

        # 记录候选池配置（池大小 + 数据集 + 数据源），供下次判断缓存是否可复用
        with open(self._pool_config_path(), 'w', encoding='utf-8') as f:
            json.dump({"pool_size": len(pool), "dataset": self._dataset_name(),
                       "source": source, "seed": 42}, f, ensure_ascii=False)

    def _pool_config_path(self) -> str:
        """候选池配置文件路径（记录 pool_size/数据集，用于校验缓存是否可复用）"""
        return os.path.join(self.cache_dir, "icl_pool_config.json")

    def _dataset_name(self) -> str:
        """从数据路径推断当前数据集名称（用于校验缓存归属，避免跨数据集复用）"""
        path = DATA_PATHS.get("misinformation", "").lower()
        for name in ("mcfend", "gossipcop", "politifact"):
            if name in path:
                return name
        return "unknown"

    def _cache_matches(self, pool_size: int, source: str = "raw") -> bool:
        """判断已有缓存是否与目标池大小、数据集、数据源一致；缺失/损坏视为不一致"""
        config_path = self._pool_config_path()
        if not os.path.exists(config_path):
            return False
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return (int(cfg.get("pool_size", -1)) == int(pool_size)
                    and cfg.get("dataset", "unknown") == self._dataset_name()
                    and cfg.get("source", "raw") == source)
        except Exception:
            return False

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
        
        # 读取缓存配置，确定数据源（增强/原始），据此重建对应的图
        source = "raw"
        config_path = self._pool_config_path()
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    source = json.load(f).get("source", "raw")
            except Exception:
                source = "raw"

        from data_loader import DataLoader
        data_loader = DataLoader()

        # 创建文本到数据的映射（增强数据源优先，保证重建出增强图）
        text_to_item = {}
        if source == "enhanced":
            print("从增强数据重新构建图...")
            try:
                base_items = data_loader.load_enhanced_dataset()
            except Exception:
                base_items = data_loader.load_misinformation() + data_loader.load_verified_information()
        else:
            print("从原始数据重新构建图...")
            base_items = data_loader.load_misinformation() + data_loader.load_verified_information()

        for item in base_items:
            news_text = item.get("article", {}).get("article_content", "")
            if news_text and news_text not in text_to_item:
                text_to_item[news_text] = item
        
        pool = []
        for i, metadata in enumerate(metadata_list):
            text = metadata["text"]
            # news_id 依赖 Python 的 hash()（跨进程随机化），缓存里存的值可能来自旧进程，
            # 这里按当前进程重算，保证与查询样本的 news_id 口径一致（去泄漏/排除自身才生效）
            news_id = f"news_{hash(text) % 1000000}"

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
                       help="候选池大小（如果为None，则使用论文默认的100条）")
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

