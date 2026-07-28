"""
图内上下文学习 (GICL) - 结构-语义协同检索
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from graph_builder import PropagationGraph
from meta_path import MetaPath
from semantic_encoder import SemanticEncoder
from config import GA_CONFIG

class GraphInContextLearning:
    """图内上下文学习模块"""
    
    def __init__(self, top_k: Optional[int] = None):
        """
        初始化 GICL
        
        Args:
            top_k: 检索的示例数量，如果为 None 则从配置读取
        """
        self.encoder = SemanticEncoder()
        self.lambda_structure = GA_CONFIG["lambda_structure"]
        self.top_k = top_k if top_k is not None else GA_CONFIG["top_k_retrieval"]
        self.epsilon = GA_CONFIG["epsilon"]
    
    def compute_semantic_similarity(self, text1: str, text2: str, 
                                   embedding1: Optional[np.ndarray] = None,
                                   embedding2: Optional[np.ndarray] = None) -> float:
        """
        计算语义相似度 s_sem (式2)
        
        Args:
            text1: 文本1
            text2: 文本2
            embedding1: 文本1的预计算 embedding（可选）
            embedding2: 文本2的预计算 embedding（可选）
            
        Returns:
            语义相似度 (0-1)
        """
        if embedding1 is not None and embedding2 is not None:
            # 检查维度是否匹配
            if embedding1.shape != embedding2.shape:
                # 维度不匹配，重新编码
                vec1 = self.encoder.encode([text1])[0]
                vec2 = self.encoder.encode([text2])[0]
                return max(0.0, self.encoder.cosine_similarity(vec1, vec2))
            else:
                # 使用预计算的 embedding
                return max(0.0, self.encoder.cosine_similarity(embedding1, embedding2))
        else:
            # 动态计算
            vec1 = self.encoder.encode([text1])[0]
            vec2 = self.encoder.encode([text2])[0]
            return max(0.0, self.encoder.cosine_similarity(vec1, vec2))
    
    def compute_structure_coverage(self, graph: PropagationGraph, 
                                  meta_path: MetaPath) -> float:
        """
        计算元路径在节点邻域中的覆盖率 ρ_M(v) (式3)
        
        Args:
            graph: 传播图
            meta_path: 元路径
            
        Returns:
            覆盖率 (0-1)
        """
        if not meta_path.path:
            return 0.0
        
        # 从图中抽取满足元路径的实例
        from graph_builder import GraphBuilder
        instances = GraphBuilder.extract_meta_path_instances(graph, meta_path.path)
        
        # 计算所有节点的邻域大小
        total_neighbors = 0
        matching_neighbors = 0
        
        # 遍历所有节点，统计满足元路径的交互
        for node_id, node in graph.nodes.items():
            if node.node_type == "News":
                # 计算该新闻节点的一阶交互数量
                neighbors = graph.get_neighbors(node_id)
                total_neighbors += len(neighbors)
                
                # 计算满足元路径的交互数量
                for instance in instances:
                    if len(instance) > 0 and instance[0] == node_id:
                        matching_neighbors += 1
        
        if total_neighbors == 0:
            return 0.0
        
        return matching_neighbors / (total_neighbors + self.epsilon)
    
    def compute_structure_similarity(self, graph1: PropagationGraph, 
                                    graph2: PropagationGraph,
                                    meta_path: MetaPath) -> float:
        """
        计算结构相似度 s_str (式4)
        
        Args:
            graph1: 图1
            graph2: 图2
            meta_path: 元路径
            
        Returns:
            结构相似度 (0-1)
        """
        rho1 = self.compute_structure_coverage(graph1, meta_path)
        rho2 = self.compute_structure_coverage(graph2, meta_path)
        
        # s_str = 1 - |ρ_M(v_q) - ρ_M(v_i)|
        structure_sim = 1.0 - abs(rho1 - rho2)
        return max(0.0, structure_sim)
    
    def compute_dual_view_score(self, query_text: str, query_graph: PropagationGraph,
                               candidate_text: str, candidate_graph: PropagationGraph,
                               meta_path: MetaPath,
                               query_embedding: Optional[np.ndarray] = None,
                               candidate_embedding: Optional[np.ndarray] = None) -> float:
        """
        计算双视角检索得分 s(v_q, v_i | M) (式5)
        
        Args:
            query_text: 查询新闻文本
            query_graph: 查询新闻的传播图
            candidate_text: 候选新闻文本
            candidate_graph: 候选新闻的传播图
            meta_path: 元路径
            query_embedding: 查询文本的预计算 embedding（可选）
            candidate_embedding: 候选文本的预计算 embedding（可选）
            
        Returns:
            综合相似度得分 (0-1)
        """
        # 语义相似度（使用预计算的 embedding 如果提供）
        s_sem = self.compute_semantic_similarity(
            query_text, candidate_text, 
            query_embedding, candidate_embedding
        )
        
        # 结构相似度
        s_str = self.compute_structure_similarity(query_graph, candidate_graph, meta_path)
        
        # 综合得分
        score = (1 - self.lambda_structure) * s_sem + self.lambda_structure * s_str
        return score
    
    def retrieve_examples(self, query_text: str, query_graph: PropagationGraph,
                         candidate_pool: List[Dict], meta_path: MetaPath,
                         query_embedding: Optional[np.ndarray] = None) -> List[Dict]:
        """
        检索Top-k示例 D^{(M)}_{v_q} (式5后)
        
        Args:
            query_text: 查询新闻文本
            query_graph: 查询新闻的传播图
            candidate_pool: 候选新闻池（包含文本、图和可选的 embedding）
            meta_path: 元路径
            query_embedding: 查询文本的预计算 embedding（可选）
            
        Returns:
            Top-k示例列表
        """
        scores = []
        
        for candidate in candidate_pool:
            candidate_text = candidate.get("text", "")
            candidate_graph = candidate.get("graph")
            candidate_embedding = candidate.get("embedding")  # 预计算的 embedding
            
            if candidate_graph is None:
                continue
            
            score = self.compute_dual_view_score(
                query_text, query_graph,
                candidate_text, candidate_graph,
                meta_path,
                query_embedding, candidate_embedding
            )
            
            scores.append((score, candidate))
        
        # 按得分排序，返回Top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k_examples = [item[1] for item in scores[:self.top_k]]
        
        return top_k_examples
    
    def compute_retrieval_confidence(self, query_text: str, query_graph: PropagationGraph,
                                    examples: List[Dict], meta_path: MetaPath,
                                    query_embedding: Optional[np.ndarray] = None) -> float:
        """
        计算检索置信度得分 c_k(v_q) (式9)
        
        Args:
            query_text: 查询新闻文本
            query_graph: 查询新闻的传播图
            examples: 检索到的示例列表
            meta_path: 元路径
            query_embedding: 查询文本的预计算 embedding（可选）
            
        Returns:
            检索置信度得分 (0-1)
        """
        if not examples:
            return 0.0
        
        total_score = 0.0
        for example in examples:
            candidate_text = example.get("text", "")
            candidate_graph = example.get("graph")
            candidate_embedding = example.get("embedding")
            
            if candidate_graph:
                score = self.compute_dual_view_score(
                    query_text, query_graph,
                    candidate_text, candidate_graph,
                    meta_path,
                    query_embedding, candidate_embedding
                )
                total_score += score
        
        return total_score / len(examples)

