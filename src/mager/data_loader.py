"""
数据加载模块 - 加载假新闻、真新闻和用户历史数据
"""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import DATA_PATHS

class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.misinformation_path = DATA_PATHS["misinformation"]
        self.verified_information_path = DATA_PATHS["verified_information"]
        self.user_instances_path = DATA_PATHS["user_instances"]
        
        # 缓存数据
        self._misinformation_cache: Optional[List[Dict]] = None
        self._verified_information_cache: Optional[List[Dict]] = None
        self._user_instances_cache: Optional[Dict[str, Dict]] = None
    
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """加载JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"解析JSON行失败: {e}, 跳过该行")
                        continue
        return data
    
    def load_misinformation(self, limit: Optional[int] = None) -> List[Dict]:
        """
        加载假新闻数据
        
        Args:
            limit: 限制加载的数量，None表示加载全部
            
        Returns:
            假新闻数据列表
        """
        if self._misinformation_cache is None:
            print(f"加载假新闻数据: {self.misinformation_path}")
            self._misinformation_cache = self._load_jsonl(self.misinformation_path)
            print(f"已加载 {len(self._misinformation_cache)} 条假新闻")
        
        if limit:
            return self._misinformation_cache[:limit]
        return self._misinformation_cache
    
    def load_verified_information(self, limit: Optional[int] = None) -> List[Dict]:
        """
        加载真新闻数据
        
        Args:
            limit: 限制加载的数量，None表示加载全部
            
        Returns:
            真新闻数据列表
        """
        if self._verified_information_cache is None:
            print(f"加载真新闻数据: {self.verified_information_path}")
            self._verified_information_cache = self._load_jsonl(self.verified_information_path)
            print(f"已加载 {len(self._verified_information_cache)} 条真新闻")
        
        if limit:
            return self._verified_information_cache[:limit]
        return self._verified_information_cache
    
    def load_user_instances(self) -> Dict[str, Dict]:
        """
        加载用户历史信息
        
        Returns:
            用户ID到用户信息的字典
        """
        if self._user_instances_cache is None:
            print(f"加载用户历史数据: {self.user_instances_path}")
            user_data = self._load_jsonl(self.user_instances_path)
            self._user_instances_cache = {}
            for user_info in user_data:
                uid = user_info.get("uid")
                if uid:
                    self._user_instances_cache[uid] = user_info
            print(f"已加载 {len(self._user_instances_cache)} 个用户的信息")
        
        return self._user_instances_cache
    
    def get_labeled_dataset(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取带标签的数据集（用于训练和评估）
        
        Args:
            limit: 限制加载的数量
            
        Returns:
            带标签的数据列表，每个数据包含：
            - 'article': 文章信息
            - 'comment_graphs': 评论图
            - 'repost_graph': 转发图
            - 'comment_users': 评论用户列表
            - 'repost_users': 转发用户列表
            - 'attitude_users': 点赞用户列表
            - 'label': 标签 (0=假新闻, 1=真新闻)
        """
        labeled_data = []
        
        # 加载假新闻（标签为0）
        misinformation = self.load_misinformation(limit)
        print("假新闻数量：",len(misinformation))
        for item in misinformation:
            labeled_item = item.copy()
            labeled_item['label'] = 0  # 假新闻
            labeled_data.append(labeled_item)
        print("添加假新闻成功！")
        print("添加假新闻后的数量：",len(labeled_data))
        # 加载真新闻（标签为1）
        verified = self.load_verified_information(limit)
        print("真新闻数量：",len(verified))
        for item in verified:
            labeled_item = item.copy()
            labeled_item['label'] = 1  # 真新闻
            labeled_data.append(labeled_item)
        print("添加真新闻成功！")
        print("添加真新闻后的数量：",len(labeled_data))
        return labeled_data
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        获取指定用户的信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户信息字典，如果不存在返回None
        """
        user_instances = self.load_user_instances()
        return user_instances.get(user_id)
    
    def _get_dataset_name(self) -> str:
        """从数据路径中提取数据集名称"""
        misinformation_path = DATA_PATHS.get("misinformation", "").lower()
        
        if "mcfend" in misinformation_path:
            return "mcfend"
        elif "gossipcop" in misinformation_path:
            return "gossipcop"
        elif "politifact" in misinformation_path:
            return "politifact"
        else:
            return "unknown"
    
    def _get_enhancement_model_name(self) -> str:
        """获取增强智能体使用的模型名称"""
        from config import get_model_config
        model_config = get_model_config("agent_a")
        model_name = model_config.model_name
        model_name_clean = model_name.replace("/", "-").replace(" ", "-").replace(".", "-")
        return model_name_clean
    
    def load_enhanced_dataset(self, file_path: Optional[str] = None, 
                             dataset_name: Optional[str] = None,
                             model_name: Optional[str] = None) -> List[Dict]:
        """
        加载增强后的数据集
        
        Args:
            file_path: 增强数据文件路径，如果为None则根据数据集和模型名称自动生成
            dataset_name: 数据集名称（如果为None则自动从DATA_PATHS推断）
            model_name: 模型名称（如果为None则自动从agent_a配置推断）
            
        Returns:
            增强后的数据列表
        """
        import os
        if file_path is None:
            if dataset_name is None:
                dataset_name = self._get_dataset_name()
            if model_name is None:
                model_name = self._get_enhancement_model_name()
            
            from config import DATA_PATHS
            file_path = DATA_PATHS.get("enhanced")
            if file_path is None:
                base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", dataset_name)
                filename = f"enhanced_dataset_{dataset_name}_{model_name}.jsonl"
                file_path = os.path.join(base_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"增强数据文件不存在: {file_path}\n"
                f"请先运行 preprocess_dataset.py 进行数据预处理，或指定正确的文件路径。"
            )
        
        enhanced_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        enhanced_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"解析JSON行失败: {e}，跳过该行")
                        continue
        
        # print(f"已加载 {len(enhanced_data)} 条增强数据")
        return enhanced_data

