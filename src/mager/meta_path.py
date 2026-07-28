"""
元路径定义和管理模块
"""
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MetaPath:
    """元路径类"""
    path: List[Tuple[str, str]]  # [(node_type1, edge_type1), (node_type2, edge_type2), ...]
    description: str  # 元路径的文本描述
    fitness: float = 0.0  # 适应度得分
    
    def __str__(self):
        """将元路径转换为可读字符串"""
        return self.to_formal_string()
    
    def to_formal_string(self) -> str:
        """转换为正式格式的字符串表示
        
        格式: Node1 --[Edge1]--> Node2 --[Edge2]--> Node3
        path格式: [(起始节点, 边1), (节点2, 边2), ..., (结束节点, "")]
        """
        if not self.path:
            return "Empty"
        
        # 解析路径：path格式为 [(node, edge), (node, edge), ...]
        # 第一个元素是起始节点和第一条边
        # 后续元素是中间/目标节点和下一条边
        # 最后一个元素的边为空字符串（表示路径结束）
        
        parts = []
        start_node, first_edge = self.path[0]
        parts.append(start_node)
        
        # 遍历后续节点
        for i in range(1, len(self.path)):
            node, edge = self.path[i]
            # 添加边（使用前一个元素的边）
            if first_edge and first_edge.strip():
                parts.append(f"--[{first_edge}]-->")
            else:
                # 如果边类型缺失，这是一个无效的元路径，但仍然显示（用于调试）
                parts.append("--[?]-->")  # 边类型缺失（无效元路径）
            # 添加节点
            parts.append(node)
            # 更新边类型为当前元素的边（用于下一次迭代）
            first_edge = edge
        
        return " ".join(parts)
    
    def __hash__(self):
        """使MetaPath可哈希（用于集合操作）"""
        return hash(str(self))
    
    def __eq__(self, other):
        """判断两个元路径是否相等"""
        if not isinstance(other, MetaPath):
            return False
        return self.path == other.path

class MetaPathInitializer:
    """元路径初始化器"""
    
    @staticmethod
    def get_initial_meta_paths() -> List[MetaPath]:
        """
        生成初始元路径种群
        
        包括常用模板和简单类型序列
        """
        meta_paths = []
        
        # 1. 基础元路径模板（1跳）
        # 格式: [(起始节点, 边类型), (目标节点, "")]
        basic_paths = [
            # User --[Comment]--> News (用户评论新闻，单向边)
            MetaPath(
                path=[("User", "Comment"), ("News", "")],
                description="用户评论新闻"
            ),
            # User --[Reply]--> User (用户回复用户)
            MetaPath(
                path=[("User", "Reply"), ("User", "")],
                description="用户回复用户"
            ),
            # User --[Repost]--> News (用户转发新闻)
            MetaPath(
                path=[("User", "Repost"), ("News", "")],
                description="用户转发新闻"
            ),
        ]
        
        # 2. 2跳元路径
        two_hop_paths = [
            # User --[Comment]--> News --[Repost]--> User (用户评论新闻，新闻被转发)
            MetaPath(
                path=[("User", "Comment"), ("News", "Repost"), ("User", "")],
                description="用户评论新闻，新闻被转发"
            ),
            # User --[Reply]--> User --[Comment]--> News (用户回复用户，用户评论新闻)
            MetaPath(
                path=[("User", "Reply"), ("User", "Comment"), ("News", "")],
                description="用户回复用户后评论新闻"
            ),
            # User --[Reply]--> User --[Reply]--> User (用户回复用户，用户再回复其他用户)
            MetaPath(
                path=[("User", "Reply"), ("User", "Reply"), ("User", "")],
                description="用户回复用户，形成回复链"
            ),
            # User --[Comment]--> News (注意：Comment是单向边，不能从News出发)
            # 改为：User1 --[Reply]--> User2 --[Comment]--> News (用户1回复用户2，用户2评论新闻)
            # 但上面已经有了，这里添加一个不同的：User --[Repost]--> News --[Comment]--> User (但Comment不能从News出发)
            # 改为：User --[Repost]--> News (这是1跳，已在basic_paths中)
            # 添加：User --[Comment]--> News (但这是1跳)
            # 改为：User --[Reply]--> User --[Repost]--> News (用户回复用户，用户转发新闻)
            MetaPath(
                path=[("User", "Reply"), ("User", "Repost"), ("News", "")],
                description="用户回复用户后转发新闻"
            ),
        ]
        
        # 3. 3跳元路径
        three_hop_paths = [
            # User --[Reply]--> User --[Reply]--> User --[Comment]--> News (用户回复链后评论新闻)
            MetaPath(
                path=[("User", "Reply"), ("User", "Reply"), ("User", "Comment"), ("News", "")],
                description="用户回复链后评论新闻"
            ),
            # User --[Reply]--> User --[Comment]--> News --[Repost]--> User (用户回复后评论新闻，新闻被转发)
            MetaPath(
                path=[("User", "Reply"), ("User", "Comment"), ("News", "Repost"), ("User", "")],
                description="用户回复后评论新闻，新闻被转发"
            ),
            # User --[Reply]--> User --[Reply]--> User --[Reply]--> User (用户形成深度回复链)
            MetaPath(
                path=[("User", "Reply"), ("User", "Reply"), ("User", "Reply"), ("User", "")],
                description="用户形成深度回复链"
            ),
            # User --[Reply]--> User --[Reply]--> User --[Reply]--> User (用户形成深度回复链)
            MetaPath(
                path=[("User", "Reply"), ("User", "Reply"), ("User", "Reply"), ("User", "")],
                description="用户形成深度回复链"
            ),
        ]
        
        # 4. 4跳元路径
        four_hop_paths = [
            # User --[Reply]--> User --[Reply]--> User --[Reply]--> User --[Comment]--> News (深度回复链后评论新闻)
            MetaPath(
                path=[("User", "Reply"), ("User", "Reply"), ("User", "Reply"), ("User", "Comment"), ("News", "")],
                description="深度回复链后评论新闻"
            ),
            # User --[Reply]--> User --[Reply]--> User --[Comment]--> News --[Repost]--> User (用户回复链后评论新闻，新闻被转发)
            MetaPath(
                path=[("User", "Reply"), ("User", "Reply"), ("User", "Comment"), ("News", "Repost"), ("User", "")],
                description="用户回复链后评论新闻，新闻被转发"
            ),
            # User --[Reply]--> User --[Comment]--> News --[Repost]--> User --[Reply]--> User (用户回复后评论新闻，新闻被转发，用户回复)
            MetaPath(
                path=[("User", "Reply"), ("User", "Comment"), ("News", "Repost"), ("User", "Reply"), ("User", "")],
                description="用户回复后评论新闻，新闻被转发，用户回复"
            ),
        ]
        
        # 3. 增强后的元路径（包含细化类型）
        enhanced_paths = [
            # 基于立场的路径
            MetaPath(
                path=[("User-Suspicious", "Comment-Agree"), ("News", "")],
                description="可疑用户同意并评论新闻"
            ),
            MetaPath(
                path=[("User-Normal", "Comment-Disagree"), ("News", "")],
                description="正常用户反对并评论新闻"
            ),
            MetaPath(
                path=[("User-Suspicious", "Comment-Discuss"), ("News", "")],
                description="可疑用户讨论新闻"
            ),
            # 基于情感的路径
            MetaPath(
                path=[("User-Suspicious", "Comment-Angry"), ("News", "")],
                description="可疑用户愤怒地评论新闻"
            ),
            MetaPath(
                path=[("User-Normal", "Comment-Happy"), ("News", "")],
                description="正常用户高兴地评论新闻"
            ),
            MetaPath(
                path=[("User-Suspicious", "Comment-Neutral"), ("News", "")],
                description="可疑用户中性地评论新闻"
            ),
            # 组合路径
            MetaPath(
                path=[("User-Suspicious", "Comment-Agree-Happy"), ("News", "")],
                description="可疑用户同意并高兴地评论新闻"
            ),
            MetaPath(
                path=[("User-Suspicious", "Comment-Disagree-Angry"), ("News", "")],
                description="可疑用户反对并愤怒地评论新闻"
            ),
        ]
        
        meta_paths.extend(basic_paths)
        meta_paths.extend(two_hop_paths)
        meta_paths.extend(three_hop_paths)
        meta_paths.extend(four_hop_paths)
        meta_paths.extend(enhanced_paths)
        
        return meta_paths
    
    @staticmethod
    def extract_simple_paths_from_graph(graph) -> List[MetaPath]:
        """
        从增强图中抽取简单的类型序列作为初始元路径
        
        Args:
            graph: 增强后的传播图
            
        Returns:
            提取的元路径列表
        """
        meta_paths = []
        
        # 从图中统计常见的边类型组合
        edge_sequences = []
        
        # 遍历所有边，构建简单的路径
        for edge in graph.edges:
            source_node = graph.nodes.get(edge.source)
            target_node = graph.nodes.get(edge.target)
            # print("source_node", source_node)
            # print("target_node", target_node)
            # print("edge.edge_type", edge.edge_type)
            
            if source_node and target_node:
                # 创建长度为1的元路径
                # 格式: [(起始节点, 边类型), (目标节点, "")]
                meta_path = MetaPath(
                    path=[(source_node.node_type, edge.edge_type), (target_node.node_type, "")],
                    description=f"{source_node.node_type}通过{edge.edge_type}连接到{target_node.node_type}"
                )
                if meta_path not in meta_paths:
                    meta_paths.append(meta_path)
                
                # 尝试创建长度为2的路径
                # 格式: [(起始节点, 边1), (中间节点, 边2), (目标节点, "")]
                for next_edge in graph.adjacency_list.get(edge.target, []):
                    next_target_node = graph.nodes.get(next_edge.target)
                    if next_target_node:
                        meta_path = MetaPath(
                            path=[
                                (source_node.node_type, edge.edge_type),
                                (target_node.node_type, next_edge.edge_type),
                                (next_target_node.node_type, "")
                            ],
                            description=f"{source_node.node_type}通过{edge.edge_type}到{target_node.node_type}，再通过{next_edge.edge_type}到{next_target_node.node_type}"
                        )
                        if meta_path not in meta_paths:
                            meta_paths.append(meta_path)
        
        return meta_paths

