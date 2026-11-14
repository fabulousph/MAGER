"""
Few-shot 虚假新闻检测（选择元路径版本）- Prompt V2: 对比分析版本
从100条样例池中检索4条最相关的示例作为few-shot示例。

Prompt V2 特点：
- 强调当前新闻与示例的对比分析
- 引导模型找出相似点和差异点
- 通过对比来推断真实性
"""
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from data_loader import DataLoader
from llm_client import LLMClient
from config import get_model_config, DATA_PATHS
from graph_builder import GraphBuilder, node_type_matches, edge_type_matches
from meta_path import MetaPath
from gicl import GraphInContextLearning
from embedding_cache import EmbeddingCache

class FewShotSelectedMetaPathDetector:
    """Few-shot虚假新闻检测器（选择元路径版本）- 提供示例，从100条中选择4条"""
    
    def __init__(self, model_config_name: str = "agent_d", 
                 selected_meta_paths: Optional[List[MetaPath]] = None,
                 icl_pool_size: int = 100,
                 top_k_examples: int = 4):
        """
        初始化检测器
        
        Args:
            model_config_name: 使用的模型配置名称（默认使用检测智能体配置）
            selected_meta_paths: 用于推理的元路径列表（≥1 个）；
                论文 §4.5 会用整个进化种群 P*（K=20 条）做 Eq(10) 加权聚合
            icl_pool_size: ICL 候选池大小（默认100，与论文 §5.1 监督池一致）
            top_k_examples: 检索的示例数量（默认4）
        """
        self.llm_client = LLMClient(model_config_name)
        self.data_loader = DataLoader()
        self.graph_builder = GraphBuilder()
        self.gicl = GraphInContextLearning(top_k=top_k_examples)
        self.embedding_cache = EmbeddingCache()
        self.icl_pool_size = icl_pool_size
        self.top_k_examples = top_k_examples
        self.prompt_version = "v2"  # Prompt 版本标识
        
        if selected_meta_paths is None:
            # 默认选择一些常用的元路径
            selected_meta_paths = [
                MetaPath(path=[("User", "Comment"), ("News", "")], description="用户评论新闻"),
                MetaPath(path=[("User", "Reply"), ("User", "")], description="用户回复用户"),
            ]
        
        # 论文 §4.5 以整个进化种群 P* 做加权聚合，条数为 K（如 20），因此不再限制为 2-3 条
        if len(selected_meta_paths) < 1:
            raise ValueError("至少需要 1 个元路径")
        
        self.selected_meta_paths = selected_meta_paths
        print(f"已选择 {len(selected_meta_paths)} 个元路径:")
        for i, mp in enumerate(selected_meta_paths, 1):
            print(f"  {i}. {mp.description}: {mp.to_formal_string()}")
        
        # 缓存 ICL 候选池
        self._icl_pool: Optional[List[Dict]] = None
    
    def _load_icl_pool(self):
        """加载 ICL 候选池（使用 embedding 缓存）"""
        if self._icl_pool is not None:
            return
        
        print("加载 ICL 候选池...")
        self._icl_pool = self.embedding_cache.prepare_icl_pool(
            pool_size=self.icl_pool_size,
            force_recompute=False
        )
        print(f"ICL 候选池已加载: {len(self._icl_pool)} 条数据")
    
    def _retrieve_examples(self, news_text: str, graph, meta_path: MetaPath,
                           query_embedding=None, exclude_news_id=None) -> List[Dict]:
        """
        使用 GICL 从候选池中检索最相关的示例（式4）

        Args:
            news_text: 查询新闻文本
            graph: 查询新闻的传播图
            meta_path: 元路径（用于结构相似度计算）
            query_embedding: 查询文本的预计算 embedding（可选）
            exclude_news_id: 需要排除的查询新闻 ID（D_L \\ {v_q}）

        Returns:
            检索到的示例列表（最多 top_k_examples 个）
        """
        if self._icl_pool is None:
            self._load_icl_pool()

        # 计算查询文本的 embedding（如果候选池中的 embedding 已预计算，我们也计算查询的）
        if query_embedding is None:
            query_embedding = self.gicl.encoder.encode([news_text])[0]

        # 使用 GICL 检索示例（排除查询自身）
        examples = self.gicl.retrieve_examples(
            query_text=news_text,
            query_graph=graph,
            candidate_pool=self._icl_pool,
            meta_path=meta_path,
            query_embedding=query_embedding,
            exclude_news_id=exclude_news_id
        )

        return examples[:self.top_k_examples]
    
    def _node_type_matches(self, actual_type: str, required_type: str) -> bool:
        """
        检查实际节点类型是否匹配要求的节点类型
        
        支持灵活的匹配：
        - 精确匹配：User == User
        - 基础类型匹配：User-Normal 匹配 User（如果required是User）
        - 细化类型匹配：User-Suspicious 匹配 User-Suspicious
        
        Args:
            actual_type: 图中节点的实际类型
            required_type: 元路径要求的节点类型
            
        Returns:
            是否匹配
        """
        if actual_type == required_type:
            return True
        
        # 如果要求的是基础类型（如 User），实际类型可能是细化类型（如 User-Normal）
        if required_type in ["User", "News"] and actual_type.startswith(f"{required_type}-"):
            return True
        
        # 如果要求的是细化类型，但实际是基础类型，也允许（放宽匹配）
        if actual_type in ["User", "News"] and required_type.startswith(f"{actual_type}-"):
            return True
        
        return False
    
    def _extract_subgraph_by_meta_paths(self, graph, meta_paths: Optional[List[MetaPath]] = None) -> Dict[str, List[List[str]]]:
        """
        使用选择的元路径从完整图中提取子图实例

        Args:
            graph: PropagationGraph对象
            meta_paths: 需要抽取的元路径列表（默认使用全部已选元路径）

        Returns:
            {meta_path_description: [实例列表]}，每个实例是一个节点序列
        """
        subgraph_instances = {}

        for meta_path in (meta_paths if meta_paths is not None else self.selected_meta_paths):
            # 使用改进的提取方法（考虑节点类型灵活匹配）
            instances = self._extract_meta_path_instances_flexible(graph, meta_path.path)

            # 存储实例（最多保留100个实例，避免过多）
            subgraph_instances[meta_path.description] = instances[:100]

        return subgraph_instances
    
    def _extract_meta_path_instances_flexible(self, graph, meta_path: List[Tuple[str, str]]) -> List[List[str]]:
        """
        从图中抽取满足元路径的实例（支持灵活的节点类型匹配）
        
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
                      if self._node_type_matches(node.node_type, first_node_type)]
        
        def dfs(current_node_id: str, path: List[str], path_index: int):
            """深度优先搜索查找元路径实例"""
            if path_index >= len(meta_path) - 1:
                # 到达最后一个节点（边为空字符串）
                instances.append(path[:])
                return
            
            # 获取当前需要的边类型和目标节点类型
            _, edge_type = meta_path[path_index]
            next_node_type, _ = meta_path[path_index + 1]
            
            # 查找满足条件的边
            for edge in graph.adjacency_list.get(current_node_id, []):
                # 检查边类型是否匹配（兼容增强后的细粒度/组合边类型）
                if edge_type_matches(edge.edge_type, edge_type):
                    target_node = graph.nodes.get(edge.target)
                    if target_node and node_type_matches(target_node.node_type, next_node_type):
                        path.append(edge.target)
                        dfs(edge.target, path, path_index + 1)
                        path.pop()  # 回溯
        
        for start_node_id in start_nodes:
            dfs(start_node_id, [start_node_id], 0)
        
        return instances
    
    def _subgraph_to_text(self, subgraph_instances: Dict[str, List[List[str]]], graph) -> str:
        """
        将提取的子图实例转换为文本描述
        
        Args:
            subgraph_instances: {meta_path_description: [实例列表]}
            graph: PropagationGraph对象（用于获取节点和边的详细信息）
            
        Returns:
            子图文本描述
        """
        lines = []
        
        lines.append("【基于选择的元路径提取的子图信息】")
        lines.append("")
        
        for meta_path_desc, instances in subgraph_instances.items():
            lines.append(f"元路径: {meta_path_desc}")
            lines.append(f"  匹配到的实例数: {len(instances)}")
            
            if instances:
                # 显示前20个实例
                display_instances = instances[:20]
                for i, instance in enumerate(display_instances, 1):
                    # 实例是一个节点序列 [node1, node2, ...]
                    instance_str = " -> ".join(instance[:5])  # 限制显示前5个节点
                    if len(instance) > 5:
                        instance_str += f" -> ... (共{len(instance)}个节点)"
                    
                    # 获取实例中的详细信息（评论文本等）
                    details = []
                    for j in range(len(instance) - 1):
                        source_id = instance[j]
                        target_id = instance[j + 1]
                        
                        # 查找对应的边
                        for edge in graph.adjacency_list.get(source_id, []):
                            if edge.target == target_id:
                                text = edge.attributes.get("text", "")
                                if text:
                                    text_display = text[:100] if len(text) <= 100 else text[:100] + "..."
                                    details.append(f"[{source_id}->{target_id}: {text_display}]")
                                break
                    
                    if details:
                        lines.append(f"    实例{i}: {instance_str}")
                        for detail in details[:2]:  # 最多显示2个详细信息
                            lines.append(f"      {detail}")
                    else:
                        lines.append(f"    实例{i}: {instance_str}")
                
                if len(instances) > 20:
                    lines.append(f"    ... (还有 {len(instances) - 20} 个实例未显示)")
            else:
                lines.append("  (未匹配到任何实例)")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _graph_to_text(self, graph, max_comments: int = 150) -> tuple:
        """
        将传播图转换为论文格式的文本描述（基础信息）
        按照论文的格式：[Claim], [Comments], [Propagation Structure]
        
        Args:
            graph: PropagationGraph对象
            max_comments: 最大评论数量（避免超过token限制）
            
        Returns:
            (comments_text, propagation_structure_text): 评论文本和传播结构文本
        """
        # 收集所有评论（Comment 边，兼容 Comment-{stance}-{sentiment} 细化类型）
        comment_edges = [e for e in graph.edges if e.edge_type.split("-")[0] == "Comment"]

        # 收集所有用户间回复（Reply 边，兼容 Reply-{stance}-{sentiment} 细化类型）
        reply_edges = [e for e in graph.edges if e.edge_type.split("-")[0] == "Reply"]
        
        # 1. Comments格式: ['id1:comment_text1', 'id2:comment_text2', ...]
        comments_list = []
        comment_id_map = {}
        user_to_comment_edge = {}
        
        comments_with_text = [e for e in comment_edges if e.attributes.get("text", "").strip()]
        comments_without_text = [e for e in comment_edges if not e.attributes.get("text", "").strip()]
        
        display_comments = comments_with_text[:max_comments]
        
        for i, edge in enumerate(display_comments, 1):
            user_id = edge.source
            text = edge.attributes.get("text", "").strip()
            if text:
                if user_id and len(user_id) <= 50:
                    comment_id = user_id
                else:
                    comment_id = f"comment_{i}"
                
                text_display = text[:500] if len(text) <= 500 else text[:500] + "..."
                comment_id_map[user_id] = comment_id
                user_to_comment_edge[user_id] = edge
                comments_list.append(f"'{comment_id}:{text_display}'")
        
        remaining_slots = max_comments - len(display_comments)
        if remaining_slots > 0 and comments_without_text:
            for i, edge in enumerate(comments_without_text[:min(remaining_slots, 10)], len(display_comments) + 1):
                user_id = edge.source
                if user_id and len(user_id) <= 50:
                    comment_id = user_id
                else:
                    comment_id = f"comment_{i}"
                comment_id_map[user_id] = comment_id
                user_to_comment_edge[user_id] = edge
                comments_list.append(f"'{comment_id}:(无文本内容)'")
        
        if comments_list:
            comments_text = f"[{', '.join(comments_list)}]"
        else:
            comments_text = "[]"
        
        # 2. Propagation Structure格式
        propagation_list = []
        comments_in_reply_as_source = set()
        
        for edge in reply_edges:
            source_id = edge.source
            target_id = edge.target
            
            if target_id == graph.news_id and source_id in comment_id_map:
                source_comment_id = comment_id_map[source_id]
                comments_in_reply_as_source.add(source_id)
                propagation_list.append(f"'{source_comment_id} replied to news'")
            elif source_id in comment_id_map and target_id in comment_id_map:
                source_comment_id = comment_id_map[source_id]
                target_comment_id = comment_id_map[target_id]
                comments_in_reply_as_source.add(source_id)
                propagation_list.append(f"'{source_comment_id} replied to {target_comment_id}'")
        
        for user_id, comment_id in comment_id_map.items():
            if user_id not in comments_in_reply_as_source:
                propagation_list.append(f"'{comment_id} replied to news'")
        
        if propagation_list:
            propagation_structure_text = f"[{', '.join(propagation_list)}]"
        else:
            propagation_structure_text = "[]"
        
        return comments_text, propagation_structure_text
    
    def detect(self, news_item: Dict) -> int:
        """
        检测新闻真实性（使用选择的元路径提取子图，并使用 GICL 检索 few-shot 示例）
        
        Args:
            news_item: 新闻数据项（包含article、comment_graphs等信息）
            
        Returns:
            预测标签: 0=假新闻, 1=真新闻
        """
        try:
            # 确保 ICL 候选池已加载
            if self._icl_pool is None:
                self._load_icl_pool()
            
            # 构建传播图（自动检测是否使用增强数据）
            article = news_item.get("article", {})
            news_text = article.get("article_content", "")
            news_id = f"news_{hash(news_text) % 1000000}"
            
            # 检查是否有增强数据（enhanced_nodes 或 enhanced_edges）
            has_enhanced_data = ("enhanced_nodes" in news_item or "enhanced_edges" in news_item)
            
            if has_enhanced_data:
                # 使用增强数据构建增强图
                graph = self.graph_builder.build_enhanced_from_data(news_item, news_id)
            else:
                # 使用原始数据构建图
                graph = self.graph_builder.build_from_data(news_item, news_id)
            
            # 获取基础图信息（评论和传播结构）
            comments_text, propagation_structure_text = self._graph_to_text(graph)

            system_prompt = """你是一个专业的虚假新闻检测专家（Prompt V2: 对比分析专家）。你需要通过对比待检测新闻与参考案例，找出相似点和差异点，从而判断新闻的真实性。

请只返回数字 0 或 1，不要返回其他任何内容：
- 0 表示假新闻
- 1 表示真新闻"""

            # 查询文本 embedding 只计算一次，供检索与置信度复用
            query_embedding = self.gicl.encoder.encode([news_text])[0]

            # §4.5 检索加权推理：对每个有效元路径独立预测，再用检索置信度 c_k 加权聚合
            path_predictions = []  # [(c_k, y_k), ...]
            for meta_path in self.selected_meta_paths:
                # 式(10)：过滤有效元路径（覆盖率 rho_M(v_q) > 0）
                rho = self.gicl.compute_structure_coverage(graph, meta_path)
                if rho <= 0.0:
                    continue

                # 抽取该元路径诱导的子图，并转换为文本
                subgraph_instances = self._extract_subgraph_by_meta_paths(graph, [meta_path])
                subgraph_text = self._subgraph_to_text(subgraph_instances, graph)

                # 式(4)：检索该元路径下的 top-k 示范（排除查询自身 v_q）
                examples = self._retrieve_examples(
                    news_text, graph, meta_path,
                    query_embedding=query_embedding,
                    exclude_news_id=news_id
                )

                # 构建该元路径专属的检测 prompt，得到该路径的预测 y_k
                prompt = self._build_prompt(
                    news_text, comments_text, propagation_structure_text,
                    subgraph_text, examples,
                    prompt_meta_paths=[meta_path]
                )
                response = self.llm_client.call(prompt, system_prompt)
                y_k = self._parse_label(response)

                # 式(9)：检索置信度 c_k(v_q) = 检索到的示例相似度均值
                c_k = self.gicl.compute_retrieval_confidence(
                    news_text, graph, examples, meta_path,
                    query_embedding=query_embedding
                )
                path_predictions.append((c_k, y_k))

            # 式(10)：没有任何有效元路径时，回退到仅基于内容的预测
            if not path_predictions:
                fallback_prompt = self._build_content_only_prompt(
                    news_text, comments_text, propagation_structure_text
                )
                response = self.llm_client.call(fallback_prompt, system_prompt)
                return self._parse_label(response)

            # 式(10)：置信度加权聚合 y_hat = I( sum(c_k*y_k)/sum(c_k) > 0.5 )
            total_confidence = sum(c for c, _ in path_predictions)
            if total_confidence > 0:
                weighted_score = sum(c * y for c, y in path_predictions) / total_confidence
            else:
                # 所有置信度均为 0 时退化为等权投票
                weighted_score = sum(y for _, y in path_predictions) / len(path_predictions)

            return 1 if weighted_score > 0.5 else 0
            
        except Exception as e:
            print(f"检测失败: {e}")
            import traceback
            traceback.print_exc()
            return 0  # 默认返回假新闻
    
    def _build_prompt(self, news_text: str, comments_text: str, 
                     propagation_structure_text: str, subgraph_text: str,
                     examples: List[Dict],
                     prompt_meta_paths: Optional[List[MetaPath]] = None) -> str:
        """
        构建Few-shot检测提示词（V2: 对比分析版本）
        
        Args:
            news_text: 新闻文本
            comments_text: 评论文本
            propagation_structure_text: 传播结构文本
            subgraph_text: 子图文本
            examples: 检索到的示例列表
        """
        prompt_parts = []
        
        # 限制新闻文本长度
        if len(news_text) > 2000:
            news_text = news_text[:2000] + "... (已截断)"
        
        # 构建元路径描述
        meta_path_descriptions = []
        for mp in (prompt_meta_paths if prompt_meta_paths is not None else self.selected_meta_paths):
            meta_path_descriptions.append(f"- {mp.description}: {mp.to_formal_string()}")
        meta_paths_info = "\n".join(meta_path_descriptions)
        
        # 添加目标新闻的完整信息
        prompt_parts.append("【待检测新闻】")
        prompt_parts.append(f"[Claim[id:{news_text}]]")
        prompt_parts.append(f"[Comments[{comments_text}]]")
        prompt_parts.append(f"[Propagation Structure:{propagation_structure_text}]")
        prompt_parts.append(f"[Selected Meta-paths Subgraph:")
        prompt_parts.append("选择的元路径：")
        prompt_parts.append(meta_paths_info)
        prompt_parts.append("")
        prompt_parts.append(subgraph_text)
        prompt_parts.append("]")
        prompt_parts.append("")
        
        # 添加few-shot示例（强调对比）
        prompt_parts.append("="*60)
        prompt_parts.append("【参考示例 - 用于对比分析】")
        prompt_parts.append("以下是几个与当前新闻最相关的历史案例（通过语义和结构相似度检索得到）：\n")
        
        for i, example in enumerate(examples, 1):
            example_text = example.get('text', '')[:500] + "..." if len(example.get('text', '')) > 500 else example.get('text', '')
            label_str = example.get('label_str', '未知')
            prompt_parts.append(f"参考案例 {i} - 标签: {label_str}")
            prompt_parts.append(f"新闻内容: {example_text}")
            prompt_parts.append("")
        
        prompt_parts.append("="*60)
        prompt_parts.append("\n【对比分析任务】")
        prompt_parts.append("请通过对比分析来判断待检测新闻的真实性。")
        prompt_parts.append("")
        prompt_parts.append("对比分析步骤：")
        prompt_parts.append("1. 内容对比：")
        prompt_parts.append("   - 待检测新闻与参考案例在主题、表述方式、细节上的相似性")
        prompt_parts.append("   - 是否存在可疑的表述模式（如夸大、煽动性语言等）")
        prompt_parts.append("")
        prompt_parts.append("2. 传播结构对比：")
        prompt_parts.append("   - 待检测新闻的传播结构与参考案例的传播模式是否相似")
        prompt_parts.append("   - 评论的立场分布、用户交互模式是否与已知的真实/虚假新闻模式一致")
        prompt_parts.append("")
        prompt_parts.append("3. 元路径子图对比：")
        prompt_parts.append("   - 待检测新闻的子图特征（实例数量、交互深度等）与参考案例的对比")
        prompt_parts.append("   - 子图中的传播模式是否支持或质疑新闻真实性")
        prompt_parts.append("")
        prompt_parts.append("4. 综合判断：")
        prompt_parts.append("   - 如果待检测新闻与多个真新闻案例相似，倾向于判断为真新闻")
        prompt_parts.append("   - 如果待检测新闻与多个假新闻案例相似，倾向于判断为假新闻")
        prompt_parts.append("   - 考虑传播结构和子图模式的一致性")
        prompt_parts.append("")
        prompt_parts.append("请基于以上对比分析，判断待检测新闻的真实性。")
        prompt_parts.append("只返回数字 0（假新闻）或 1（真新闻），不要返回其他内容。")
        
        return "\n".join(prompt_parts)

    def _build_content_only_prompt(self, news_text: str, comments_text: str,
                                   propagation_structure_text: str) -> str:
        """
        构建仅基于内容的回退提示词（式10：当没有任何有效元路径时使用）

        Args:
            news_text: 新闻文本
            comments_text: 评论文本
            propagation_structure_text: 传播结构文本

        Returns:
            内容回退 prompt
        """
        if len(news_text) > 2000:
            news_text = news_text[:2000] + "... (已截断)"

        prompt_parts = [
            "【待检测新闻】",
            f"[Claim[id:{news_text}]]",
            f"[Comments[{comments_text}]]",
            f"[Propagation Structure:{propagation_structure_text}]",
            "",
            "请根据新闻内容、评论内容和传播结构，判断该新闻的真实性。",
            "只返回数字 0（假新闻）或 1（真新闻），不要返回其他内容。",
        ]
        return "\n".join(prompt_parts)
    
    def _parse_label(self, response: str) -> int:
        """解析LLM响应，提取标签（0或1）"""
        response = response.strip()
        
        import re
        numbers = re.findall(r'\d+', response)
        
        if numbers:
            label = int(numbers[0])
            if label == 1:
                return 1
            else:
                return 0
        
        response_lower = response.lower()
        if '1' in response or '真' in response or 'true' in response_lower or 'real' in response_lower:
            return 1
        elif '0' in response or '假' in response or 'false' in response_lower or 'fake' in response_lower:
            return 0
        
        return 0
    
    def _get_dataset_name(self) -> str:
        """从数据路径中提取数据集名称"""
        from config import DATA_PATHS
        misinformation_path = DATA_PATHS.get("misinformation", "").lower()
        
        if "mcfend" in misinformation_path:
            return "mcfend"
        elif "gossipcop" in misinformation_path:
            return "gossipcop"
        elif "politifact" in misinformation_path:
            return "politifact"
        else:
            return "unknown"
    
    def _get_model_name(self) -> str:
        """获取当前使用的模型名称"""
        model_config = self.llm_client.config
        model_name = model_config.model_name
        model_name_clean = model_name.replace("/", "-").replace(" ", "-").replace(".", "-")
        return model_name_clean
    
    def _get_meta_paths_str(self) -> str:
        """获取元路径集合的短标识（用于文件名）

        当元路径较多（如整个种群 K=20）时，直接拼接描述会让文件名超长，
        因此超过 3 条时改用 "K{数量}_{哈希}" 的紧凑形式。
        """
        import hashlib

        mp_strs = []
        for mp in self.selected_meta_paths:
            # 使用描述的前几个词作为标识
            desc_words = mp.description.split()[:2]
            mp_strs.append("_".join(desc_words))

        # 条数较多时用数量+短哈希，避免文件名过长导致创建失败
        if len(mp_strs) > 3:
            digest = hashlib.md5("|".join(mp_strs).encode("utf-8")).hexdigest()[:8]
            return f"K{len(mp_strs)}_{digest}"

        return "_".join(mp_strs)

    def _news_id_for_item(self, item: Dict) -> str:
        """按检测器统一规则计算某条新闻的 news_id（与 detect 内部保持一致）"""
        news_text = item.get("article", {}).get("article_content", "")
        return f"news_{hash(news_text) % 1000000}"

    def evaluate(self, test_dataset: Optional[List[Dict]] = None, 
                 limit: Optional[int] = None,
                 save_results: bool = True,
                 output_dir: str = "result") -> Dict:
        """
        评估检测性能（使用增强后的数据集）。

        指标口径与论文 §5.2 一致：以“假新闻(label=0)”为正类计算
        Precision / Recall / F1；Accuracy 为整体正确率。
        """
        import json
        import os
        from datetime import datetime
        
        # 确保 ICL 候选池已加载
        if self._icl_pool is None:
            self._load_icl_pool()
        
        if test_dataset is None:
            # 使用增强后的数据集作为评测集
            test_dataset = self.data_loader.load_enhanced_dataset()

            # 去泄漏（论文 §5.1）：评测集应为“监督池之外的剩余样本”，
            # 先剔除与 ICL 监督池重合的样本，再应用 limit。
            pool_ids = {ex.get("news_id") for ex in (self._icl_pool or []) if ex.get("news_id")}
            if pool_ids:
                before = len(test_dataset)
                test_dataset = [it for it in test_dataset
                                if self._news_id_for_item(it) not in pool_ids]
                print(f"去泄漏: 从评测集剔除监督池重合样本 {before - len(test_dataset)} 条 "
                      f"({before} -> {len(test_dataset)})")

            # 随机抽样（固定种子）取 limit 条，避免按文件顺序取到单一类别导致指标失真
            if limit and len(test_dataset) > limit:
                import random
                random.seed(42)
                test_dataset = random.sample(test_dataset, limit)

        print("len(test_dataset):", len(test_dataset))
        
        # 准备数据
        data_items = []
        for i, item in enumerate(test_dataset):
            article = item.get("article", {})
            news_text = article.get("article_content", "")
            if news_text:
                data_items.append({
                    "index": i + 1,
                    "item": item,
                    "true_label": item.get("label", 0)
                })
        
        dataset_size = len(data_items)
        
        # 获取数据集和模型信息
        dataset_name = self._get_dataset_name()
        model_name = self._get_model_name()
        meta_paths_str = self._get_meta_paths_str()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建文件名
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 
                f"few_shot_selected_meta_path_{self.prompt_version}_{dataset_name}_{model_name}_{meta_paths_str}_size{dataset_size}_{timestamp}.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                pass
            print(f"检测结果将保存到: {output_file}")
        else:
            output_file = None
        
        print("开始评估...")
        
        correct = 0
        total = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        file_lock = Lock() if save_results else None
        
        def process_single_sample(data_item):
            index = data_item["index"]
            item = data_item["item"]
            true_label = data_item["true_label"]
            
            try:
                predicted_label = self.detect(item)
            except Exception as e:
                predicted_label = 0
            
            article = item.get("article", {})
            news_text = article.get("article_content", "")
            
            sample_result = {
                "sample_id": index,
                "news_text": news_text[:500] + "..." if len(news_text) > 500 else news_text,
                "news_text_length": len(news_text),
                "true_label": true_label,
                "true_label_str": "真新闻" if true_label == 1 else "假新闻",
                "predicted_label": predicted_label,
                "predicted_label_str": "真新闻" if predicted_label == 1 else "假新闻",
                "is_correct": predicted_label == true_label
            }
            
            if save_results and output_file:
                with file_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(sample_result, ensure_ascii=False) + '\n')
            
            return sample_result
        
        sample_results = [None] * len(data_items)
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(process_single_sample, item) for item in data_items]
            for future in tqdm(as_completed(futures), total=len(futures), desc="评估进度"):
                try:
                    result = future.result()
                    if result:
                        idx = result["sample_id"] - 1
                        if 0 <= idx < len(sample_results):
                            sample_results[idx] = result
                except Exception as e:
                    print(f"处理样本时出错: {e}")
        
        sample_results = [r for r in sample_results if r is not None]
        
        # 按论文 §5.2 口径统计混淆矩阵：以“假新闻(label=0)”为正类
        for sample_result in sample_results:
            predicted_label = sample_result["predicted_label"]
            true_label = sample_result["true_label"]

            if predicted_label == true_label:
                correct += 1
                # 真正例 TP：真实为假且预测为假；真负例 TN：真实为真且预测为真
                if true_label == 0:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                # 假负例 FN：真实为假但预测为真；假正例 FP：真实为真但预测为假
                if true_label == 0 and predicted_label == 1:
                    false_negatives += 1
                elif true_label == 1 and predicted_label == 0:
                    false_positives += 1

            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if save_results:
            summary_file = os.path.join(output_dir,
                f"few_shot_selected_meta_path_{self.prompt_version}_summary_{dataset_name}_{model_name}_{meta_paths_str}_size{dataset_size}_{timestamp}.json")
            summary = {
                "timestamp": timestamp,
                "dataset": dataset_name,
                "model": model_name,
                "method": f"few-shot-selected-meta-path-gicl-{self.prompt_version}",
                # 正类定义：与论文一致，采用“假新闻”(label=0) 为正类
                "positive_class": "fake(label=0)",
                "icl_pool_size": self.icl_pool_size,
                "top_k_examples": self.top_k_examples,
                "selected_meta_paths": [{"description": mp.description, "path": str(mp.to_formal_string())} 
                                       for mp in self.selected_meta_paths],
                "total_samples": total,
                "correct": correct,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_negatives": true_negatives,
                "results_file": output_file
            }
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"汇总信息已保存到: {summary_file}")
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "positive_class": "fake(label=0)",
            "total_samples": total,
            "correct": correct,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "results_file": output_file if save_results else None,
            "summary_file": summary_file if save_results else None
        }
        
        return results


def _parse_path_string(path_str: str) -> List[Tuple[str, str]]:
    """
    解析路径字符串，转换为MetaPath的path格式
    
    Args:
        path_str: 格式如 "User --[Comment]--> News --[Repost]--> User"
        
    Returns:
        path格式: [(起始节点, 边1), (节点2, 边2), ..., (结束节点, "")]
    """
    import re
    # 匹配模式: --[Edge]--> 来分割路径
    # 先找到所有的边
    edge_pattern = r'--\[([^\]]+)\]-->'
    edges = re.findall(edge_pattern, path_str)
    
    if not edges:
        raise ValueError(f"无法解析路径字符串（未找到边）: {path_str}")
    
    # 使用边模式来分割字符串，提取节点
    # 格式: Node1 --[Edge1]--> Node2 --[Edge2]--> Node3
    # 分割后得到: [Node1, Node2, Node3]
    parts = re.split(r'\s*--\[[^\]]+\]-->\s*', path_str)
    
    if len(parts) != len(edges) + 1:
        raise ValueError(f"路径解析错误: 节点数({len(parts)}) != 边数+1({len(edges)+1})")
    
    # 构建path格式
    path = []
    for i in range(len(edges)):
        node = parts[i].strip()
        edge = edges[i]
        path.append((node, edge))
    
    # 最后一个节点，边为空字符串
    last_node = parts[-1].strip()
    path.append((last_node, ""))
    
    return path


def _load_meta_paths_from_json(json_file: str) -> List[MetaPath]:
    """
    从JSON文件加载已选择的元路径
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        MetaPath对象列表
    """
    import json
    import os
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON文件不存在: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    selected_meta_paths = []
    for mp_data in data.get("selected_meta_paths", []):
        description = mp_data.get("description", "")
        path_str = mp_data.get("path", "")
        
        if not path_str:
            continue
        
        # 解析路径字符串
        path = _parse_path_string(path_str)
        
        # 创建MetaPath对象
        meta_path = MetaPath(path=path, description=description)
        selected_meta_paths.append(meta_path)
    
    return selected_meta_paths


def main():
    """主函数"""
    import argparse
    import os
    from meta_path import MetaPath, MetaPathInitializer
    
    # 默认JSON文件路径（qwen3的politifact数据上已选择好的元路径）
    default_json_file = "/root/autodl-tmp/fnd/code/result/few_shot_selected_meta_path_v2_summary_politifact_qwen3-8b_用户评论新闻，新闻被转发_用户回复用户后转发新闻_交叉:_从_size728_20251226_182027.json"
    
    parser = argparse.ArgumentParser(description="Few-shot虚假新闻检测（选择元路径版本）")
    parser.add_argument("--mode", type=str, choices=["detect", "evaluate"],
                       default="evaluate", help="运行模式")
    parser.add_argument("--limit", type=int, default=30000,
                       help="评估时的样本数量限制")
    parser.add_argument("--model_config", type=str, default="agent_d",
                       help="使用的模型配置名称")
    parser.add_argument("--output_dir", type=str, default="result",
                       help="结果文件保存目录")
    parser.add_argument("--no_save", action="store_true",
                       help="不保存检测结果到文件")
    parser.add_argument("--icl_pool_size", type=int, default=None,
                       help="ICL 候选池大小（默认100）")
    parser.add_argument("--top_k", type=int, default=4,
                       help="检索的示例数量（默认4）")
    
    # 元路径选择参数
    parser.add_argument("--meta_path_json", type=str, default=default_json_file,
                       help=f"从JSON文件加载已选择的元路径（默认: {default_json_file}）")
    parser.add_argument("--meta_path_indices", type=int, nargs="+",
                       help="选择的元路径索引（从初始元路径列表中选择，2-3个，用空格分隔，如：0 1 2）")
    parser.add_argument("--list_meta_paths", action="store_true",
                       help="列出所有可用的初始元路径并退出")
    
    args = parser.parse_args()
    
    # 列出可用元路径
    if args.list_meta_paths:
        print("可用的初始元路径:")
        initial_meta_paths = MetaPathInitializer.get_initial_meta_paths()
        for i, mp in enumerate(initial_meta_paths):
            print(f"  [{i}] {mp.description}: {mp.to_formal_string()}")
        return
    
    # 选择元路径
    selected_meta_paths = None
    
    # 优先从JSON文件加载（如果文件存在）
    if args.meta_path_json and os.path.exists(args.meta_path_json):
        try:
            print(f"从JSON文件加载元路径: {args.meta_path_json}")
            selected_meta_paths = _load_meta_paths_from_json(args.meta_path_json)
            print(f"成功加载 {len(selected_meta_paths)} 个元路径")
        except Exception as e:
            print(f"从JSON文件加载元路径失败: {e}")
            selected_meta_paths = None
    
    # 如果JSON加载失败或未指定，尝试使用索引选择
    if selected_meta_paths is None and args.meta_path_indices:
        initial_meta_paths = MetaPathInitializer.get_initial_meta_paths()
        selected_meta_paths = [initial_meta_paths[i] for i in args.meta_path_indices if 0 <= i < len(initial_meta_paths)]
        if len(selected_meta_paths) < 2:
            print("错误：至少需要选择2个元路径")
            return
        if len(selected_meta_paths) > 3:
            print("警告：最多选择3个元路径，将只使用前3个")
            selected_meta_paths = selected_meta_paths[:3]
    
    # 如果仍然没有选择元路径，使用默认值
    if selected_meta_paths is None:
        print("警告：未指定元路径，使用默认元路径")
        selected_meta_paths = [
            MetaPath(path=[("User", "Comment"), ("News", "")], description="用户评论新闻"),
            MetaPath(path=[("User", "Reply"), ("User", "")], description="用户回复用户"),
        ]
    
    # 创建检测器
    detector = FewShotSelectedMetaPathDetector(
        model_config_name=args.model_config,
        selected_meta_paths=selected_meta_paths,
        icl_pool_size=args.icl_pool_size,
        top_k_examples=args.top_k
    )
    
    if args.mode == "detect":
        # 检测示例
        print("检测示例新闻...")
        from data_loader import DataLoader
        data_loader = DataLoader()
        test_data = data_loader.get_labeled_dataset(limit=5)
        
        for item in test_data:
            article = item.get("article", {})
            news_text = article.get("article_content", "")
            true_label = item.get("label", 0)
            
            if not news_text:
                continue
            
            print(f"\n{'='*60}")
            print(f"新闻内容: {news_text[:200]}...")
            print(f"真实标签: {true_label} ({'真新闻' if true_label == 1 else '假新闻'})")
            
            predicted_label = detector.detect(item)
            print(f"预测结果: {predicted_label} ({'真新闻' if predicted_label == 1 else '假新闻'})")
            print(f"是否正确: {'✓' if predicted_label == true_label else '✗'}")
    
    elif args.mode == "evaluate":
        # 评估性能
        print("开始评估（Few-shot选择元路径版本）...")
        results = detector.evaluate(
            limit=args.limit,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*60)
        print("评估结果（Few-shot选择元路径版本）")
        print("="*60)
        print(f"使用的示例: 从{args.icl_pool_size}条候选池中检索{args.top_k}条最相关的示例")
        print(f"总样本数: {results['total_samples']}")
        print(f"正确预测: {results['correct']}")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"F1分数: {results['f1_score']:.4f}")
        print(f"\n混淆矩阵:")
        print(f"  真新闻->真新闻: {results['true_positives']}")
        print(f"  真新闻->假新闻: {results['false_negatives']}")
        print(f"  假新闻->真新闻: {results['false_positives']}")
        print(f"  假新闻->假新闻: {results['true_negatives']}")
        
        if results.get('results_file'):
            print(f"\n详细结果已保存到: {results['results_file']}")
        if results.get('summary_file'):
            print(f"汇总信息已保存到: {results['summary_file']}")


if __name__ == "__main__":
    main()

