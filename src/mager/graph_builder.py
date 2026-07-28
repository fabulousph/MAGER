"""
传播图构建模块 - 从数据构建异质信息网络（HIN）
"""
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Node:
    """图节点"""
    node_id: str
    node_type: str  # News, User, Topic等
    attributes: Dict[str, Any]  # 节点属性
    
@dataclass
class Edge:
    """图边"""
    source: str  # 源节点ID
    target: str  # 目标节点ID
    edge_type: str  # Comment, Reply, Repost, Agree等
    attributes: Dict[str, Any]  # 边属性（立场、情感等）

class PropagationGraph:
    """传播图类"""
    
    def __init__(self, news_id: str):
        """
        初始化传播图
        
        Args:
            news_id: 新闻ID
        """
        self.news_id = news_id
        self.nodes: Dict[str, Node] = {}  # node_id -> Node
        self.edges: List[Edge] = []  # 边列表
        self.adjacency_list: Dict[str, List[Edge]] = defaultdict(list)  # 邻接表
        
        # 添加新闻节点
        news_node = Node(
            node_id=news_id,
            node_type="News",
            attributes={}
        )
        self.nodes[news_id] = news_node
    
    def add_node(self, node_id: str, node_type: str, attributes: Optional[Dict] = None):
        """添加节点"""
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(
                node_id=node_id,
                node_type=node_type,
                attributes=attributes or {}
            )
        else:
            # 更新属性
            self.nodes[node_id].attributes.update(attributes or {})
    
    def add_edge(self, source: str, target: str, edge_type: str, 
                 attributes: Optional[Dict] = None):
        """添加边"""
        edge = Edge(
            source=source,
            target=target,
            edge_type=edge_type,
            attributes=attributes or {}
        )
        self.edges.append(edge)
        self.adjacency_list[source].append(edge)
    
    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """
        获取节点的邻居
        
        Args:
            node_id: 节点ID
            edge_type: 可选的边类型过滤
            
        Returns:
            邻居节点ID列表
        """
        neighbors = []
        for edge in self.adjacency_list.get(node_id, []):
            if edge_type is None or edge.edge_type == edge_type:
                neighbors.append(edge.target)
        return neighbors
    
    def get_edges(self, node_id: Optional[str] = None, 
                  edge_type: Optional[str] = None) -> List[Edge]:
        """
        获取边
        
        Args:
            node_id: 可选的节点ID过滤（源节点或目标节点）
            edge_type: 可选的边类型过滤
            
        Returns:
            边列表
        """
        if node_id is None and edge_type is None:
            return self.edges
        
        filtered_edges = []
        for edge in self.edges:
            match_node = node_id is None or edge.source == node_id or edge.target == node_id
            match_type = edge_type is None or edge.edge_type == edge_type
            
            if match_node and match_type:
                filtered_edges.append(edge)
        
        return filtered_edges

class GraphBuilder:
    """传播图构建器"""
    
    @staticmethod
    def build_from_data(news_item: Dict, news_id: Optional[str] = None) -> PropagationGraph:
        """
        从数据项构建传播图
        
        Args:
            news_item: 新闻数据项
            news_id: 新闻ID（如果为None，则自动生成）
            
        Returns:
            传播图对象
        """
        # 生成新闻ID
        if news_id is None:
            article_content = news_item.get("article", {}).get("article_content", "")
            news_id = f"news_{hash(article_content) % 1000000}"
        
        graph = PropagationGraph(news_id)
        
        article = news_item.get("article", {})
        comment_graphs = news_item.get("comment_graphs", [])
        repost_graph = news_item.get("repost_graph", {})
        comment_users = news_item.get("comment_users", [])
        repost_users = news_item.get("repost_users", [])
        attitude_users = news_item.get("attitude_users", [])
        
        # 添加评论用户节点和边
        for user_id in comment_users:
            graph.add_node(user_id, "User", {"interaction_type": "comment"})
            # 添加Comment边（未细化，后续会由增强智能体细化）
            graph.add_edge(user_id, news_id, "Comment")
        
        # 处理评论图
        for comment_graph in comment_graphs:
            nodes = comment_graph.get("nodes", [])
            edges = comment_graph.get("edges", [])
            
            # 添加评论节点和边
            for node in nodes:
                user_from = node.get("user_from")
                user_to = node.get("user_to")
                text = node.get("text", "")
                
                if user_from:
                    graph.add_node(user_from, "User")
                    if user_to:
                        # 用户之间的回复关系
                        graph.add_edge(user_from, user_to, "Reply", {"text": text})
                    else:
                        # 用户对新闻的评论
                        graph.add_edge(user_from, news_id, "Comment", {"text": text})
            
            # 处理评论图中的边（如果有）
            for edge in edges:
                # 边的格式可能是 [source_index, target_index] 或其他格式
                # 这里需要根据实际数据格式处理
                pass
        
        # 添加转发用户节点和边
        # 收集转发用户和对应的文字内容
        repost_user_text_map = {}  # user_id -> text
        
        # 从 repost_users 列表添加（可能没有文字）
        for user_id in repost_users:
            if user_id:
                repost_user_text_map[user_id] = ""  # 没有文字内容
        
        # 从 repost_graph 节点添加（可能有文字）
        repost_nodes = repost_graph.get("nodes", [])
        for node in repost_nodes:
            user_name = node.get("name")
            text = node.get("text", "")  # 获取转发文字内容
            if user_name:
                # 如果用户不存在，或者用户存在但没有文字而新节点有文字，则更新
                if user_name not in repost_user_text_map:
                    repost_user_text_map[user_name] = text
                elif not repost_user_text_map[user_name] and text:
                    # 用户已存在但没有文字，新节点有文字，则更新
                    repost_user_text_map[user_name] = text
        
        # 统一添加转发节点和边（包含文字内容）
        for user_id, text in repost_user_text_map.items():
            graph.add_node(user_id, "User", {"interaction_type": "repost"})
            # 添加转发边，包含文字内容（如果有）
            graph.add_edge(user_id, news_id, "Repost", {"text": text})
        
        # 处理转发图中的边（如果有用户之间的转发关系）
        # repost_edges = repost_graph.get("edges", [])
        
        # 添加点赞用户节点和边
        for user_id in attitude_users:
            graph.add_node(user_id, "User", {"interaction_type": "attitude"})
            graph.add_edge(user_id, news_id, "Attitude")
        
        return graph
    
    @staticmethod
    def build_enhanced_from_data(enhanced_item: Dict, news_id: Optional[str] = None) -> PropagationGraph:
        """
        从增强后的数据项直接构建增强图（不需要再次调用增强智能体）
        
        Args:
            enhanced_item: 增强后的新闻数据项（包含enhanced_nodes和enhanced_edges字段）
            news_id: 新闻ID（如果为None，则自动生成）
            
        Returns:
            增强后的传播图对象
        """
        # 生成新闻ID
        if news_id is None:
            article_content = enhanced_item.get("article", {}).get("article_content", "")
            news_id = f"news_{hash(article_content) % 1000000}"
        
        graph = PropagationGraph(news_id)
        
        # 获取增强信息
        enhanced_nodes = enhanced_item.get("enhanced_nodes", {})  # user_id -> refined_type
        enhanced_edges = enhanced_item.get("enhanced_edges", [])  # 细化后的边信息
        
        # 创建边映射，方便查找
        enhanced_edge_map = {}
        for edge_info in enhanced_edges:
            key = (edge_info["source"], edge_info["target"], edge_info["edge_type"])
            enhanced_edge_map[key] = edge_info
        
        article = enhanced_item.get("article", {})
        comment_graphs = enhanced_item.get("comment_graphs", [])
        repost_graph = enhanced_item.get("repost_graph", {})
        comment_users = enhanced_item.get("comment_users", [])
        repost_users = enhanced_item.get("repost_users", [])
        attitude_users = enhanced_item.get("attitude_users", [])
        
        # 添加评论用户节点和边（使用增强后的节点类型）
        for user_id in comment_users:
            # 使用增强后的节点类型，如果没有则默认为User
            refined_type = enhanced_nodes.get(user_id, "User")
            graph.add_node(user_id, refined_type, {"interaction_type": "comment"})
            
            # 查找是否有增强的边信息
            edge_key = (user_id, news_id, "Comment")
            if edge_key in enhanced_edge_map:
                edge_info = enhanced_edge_map[edge_key]
                stance = edge_info.get('cognitive_stance', 'Discuss')
                sentiment = edge_info.get('affective_sentiment', 'Neutral')
                
                # 添加多种细粒度的边类型，以便元路径可以从不同维度进行匹配
                # 1. 只有立场的边
                graph.add_edge(user_id, news_id, f"Comment-{stance}", {
                    "text": edge_info.get("text", ""),
                    "cognitive_stance": stance,
                    "affective_sentiment": sentiment
                })
                # 2. 只有情感的边
                graph.add_edge(user_id, news_id, f"Comment-{sentiment}", {
                    "text": edge_info.get("text", ""),
                    "cognitive_stance": stance,
                    "affective_sentiment": sentiment
                })
                # 3. 组合边（可选）
                graph.add_edge(user_id, news_id, f"Comment-{stance}-{sentiment}", {
                    "text": edge_info.get("text", ""),
                    "cognitive_stance": stance,
                    "affective_sentiment": sentiment
                })
                # 4. 原始边
                graph.add_edge(user_id, news_id, "Comment", {"text": edge_info.get("text", "")})
            else:
                # 如果没有增强信息，使用原始边
                graph.add_edge(user_id, news_id, "Comment")
        
        # 处理评论图
        for comment_graph in comment_graphs:
            nodes = comment_graph.get("nodes", [])
            
            # 添加评论节点和边
            for node in nodes:
                user_from = node.get("user_from")
                user_to = node.get("user_to")
                text = node.get("text", "")
                
                if user_from:
                    # 使用增强后的节点类型
                    refined_type = enhanced_nodes.get(user_from, "User")
                    graph.add_node(user_from, refined_type)
                    
                    if user_to:
                        # 用户之间的回复关系
                        refined_type_to = enhanced_nodes.get(user_to, "User")
                        graph.add_node(user_to, refined_type_to)
                        
                        # 查找增强的边信息
                        edge_key = (user_from, user_to, "Reply")
                        if edge_key in enhanced_edge_map:
                            edge_info = enhanced_edge_map[edge_key]
                            stance = edge_info.get('cognitive_stance', 'Discuss')
                            sentiment = edge_info.get('affective_sentiment', 'Neutral')
                            
                            # 添加多种细粒度的边类型
                            graph.add_edge(user_from, user_to, f"Reply-{stance}", {
                                "text": edge_info.get("text", ""),
                                "cognitive_stance": stance,
                                "affective_sentiment": sentiment
                            })
                            graph.add_edge(user_from, user_to, f"Reply-{sentiment}", {
                                "text": edge_info.get("text", ""),
                                "cognitive_stance": stance,
                                "affective_sentiment": sentiment
                            })
                            graph.add_edge(user_from, user_to, f"Reply-{stance}-{sentiment}", {
                                "text": edge_info.get("text", ""),
                                "cognitive_stance": stance,
                                "affective_sentiment": sentiment
                            })
                            graph.add_edge(user_from, user_to, "Reply", {"text": text})
                        else:
                            graph.add_edge(user_from, user_to, "Reply", {"text": text})
                    else:
                        # 用户对新闻的评论
                        edge_key = (user_from, news_id, "Comment")
                        if edge_key in enhanced_edge_map:
                            edge_info = enhanced_edge_map[edge_key]
                            stance = edge_info.get('cognitive_stance', 'Discuss')
                            sentiment = edge_info.get('affective_sentiment', 'Neutral')
                            
                            # 添加多种细粒度的边类型
                            graph.add_edge(user_from, news_id, f"Comment-{stance}", {
                                "text": edge_info.get("text", ""),
                                "cognitive_stance": stance,
                                "affective_sentiment": sentiment
                            })
                            graph.add_edge(user_from, news_id, f"Comment-{sentiment}", {
                                "text": edge_info.get("text", ""),
                                "cognitive_stance": stance,
                                "affective_sentiment": sentiment
                            })
                            graph.add_edge(user_from, news_id, f"Comment-{stance}-{sentiment}", {
                                "text": edge_info.get("text", ""),
                                "cognitive_stance": stance,
                                "affective_sentiment": sentiment
                            })
                            graph.add_edge(user_from, news_id, "Comment", {"text": text})
                        else:
                            graph.add_edge(user_from, news_id, "Comment", {"text": text})
        
        # 添加转发用户节点和边
        repost_user_text_map = {}
        
        for user_id in repost_users:
            if user_id:
                repost_user_text_map[user_id] = ""
        
        repost_nodes = repost_graph.get("nodes", [])
        for node in repost_nodes:
            user_name = node.get("name")
            text = node.get("text", "")
            if user_name:
                if user_name not in repost_user_text_map:
                    repost_user_text_map[user_name] = text
                elif not repost_user_text_map[user_name] and text:
                    repost_user_text_map[user_name] = text
        
        for user_id, text in repost_user_text_map.items():
            refined_type = enhanced_nodes.get(user_id, "User")
            graph.add_node(user_id, refined_type, {"interaction_type": "repost"})
            graph.add_edge(user_id, news_id, "Repost", {"text": text})
        
        # 添加点赞用户节点和边
        for user_id in attitude_users:
            refined_type = enhanced_nodes.get(user_id, "User")
            graph.add_node(user_id, refined_type, {"interaction_type": "attitude"})
            graph.add_edge(user_id, news_id, "Attitude")
        
        return graph
    
    @staticmethod
    def extract_meta_path_instances(graph: PropagationGraph, 
                                   meta_path: List[Tuple[str, str]]) -> List[List[str]]:
        """
        从图中抽取满足元路径的实例
        
        Args:
            graph: 传播图
            meta_path: 元路径，格式为 [(起始节点, 边1), (节点2, 边2), ..., (结束节点, "")]
            
        Returns:
            满足元路径的节点序列列表
        """
        instances = []
        
        if not meta_path:
            return instances
        
        # 从第一个节点类型开始
        first_node_type, first_edge = meta_path[0]
        start_nodes = [node_id for node_id, node in graph.nodes.items() 
                      if node.node_type == first_node_type]
        
        def dfs(current_node_id: str, path: List[str], path_index: int):
            """深度优先搜索查找元路径实例"""
            if path_index >= len(meta_path) - 1:
                # 到达最后一个节点（边为空字符串）
                instances.append(path[:])
                return
            
            # 获取当前需要的边类型和目标节点类型
            # path[path_index] = (当前节点类型, 边类型)
            # path[path_index+1] = (目标节点类型, 下一条边)
            _, edge_type = meta_path[path_index]
            next_node_type, _ = meta_path[path_index + 1]
            
            # 查找满足条件的边
            for edge in graph.adjacency_list.get(current_node_id, []):
                # 检查边类型是否匹配
                # 1. 完全匹配
                # 2. 边是元路径要求的细化（如 Comment-Agree 满足 Comment）
                # 3. 元路径要求的是边的细化（如 Comment 满足 Comment-Agree）
                edge_matches = (edge.edge_type == edge_type or 
                               edge.edge_type.startswith(f"{edge_type}-") or
                               edge_type.startswith(f"{edge.edge_type}-"))
                
                if edge_matches:
                    target_node = graph.nodes.get(edge.target)
                    if target_node and target_node.node_type == next_node_type:
                        path.append(edge.target)
                        dfs(edge.target, path, path_index + 1)
                        path.pop()  # 回溯
        
        for start_node_id in start_nodes:
            dfs(start_node_id, [start_node_id], 0)
        
        return instances

