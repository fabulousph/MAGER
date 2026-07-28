"""
增强智能体 (L_A) - 对传播图进行语义增强
"""
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from graph_builder import PropagationGraph, Node, Edge
from llm_client import LLMClient
from config import ENHANCEMENT_CONFIG
import json
import re

class EnhancementAgent:
    """增强智能体 - 负责节点和边的语义细化"""
    
    def __init__(self):
        self.llm_client = LLMClient("agent_a")
        
        # 预定义的节点类型和边类型
        self.user_refined_types = ["User-Normal", "User-Suspicious"] # 用户类型
        self.news_topics = ["politics", "health", "entertainment", "technology", 
                           "sports", "economy", "education", "other"] # 新闻主题
        
        self.cognitive_stances = ["Agree", "Disagree", "Discuss", "Unrelated"] # 认知立场
        self.affective_sentiments = ["Angry", "Happy", "Neutral", "Fear", "Sad"] # 情感极性
    
    def enhance_graph(self, graph: PropagationGraph, 
                     news_content: str) -> PropagationGraph:
        """
        增强传播图
        
        Args:
            graph: 原始传播图
            news_content: 新闻内容
            
        Returns:
            增强后的传播图
        """
        enhanced_graph = PropagationGraph(graph.news_id)
        
        # 复制所有节点
        for node_id, node in graph.nodes.items():
            enhanced_graph.nodes[node_id] = Node(
                node_id=node.node_id,
                node_type=node.node_type,
                attributes=node.attributes.copy()
            )
        
        # 1. 节点细化
        print("节点细化")
        enhanced_graph = self._refine_nodes(enhanced_graph, news_content)
        
        # 2. 边细化
        enhanced_graph = self._refine_edges(enhanced_graph, graph)
        
        # 3. 添加主题节点
        # enhanced_graph = self._add_topic_nodes(enhanced_graph, news_content)
        
        return enhanced_graph
    
    def _refine_nodes(self, graph: PropagationGraph, news_content: str) -> PropagationGraph:
        """节点细化 - 将用户分为Normal和Suspicious"""
        user_nodes = [node for node in graph.nodes.values() if node.node_type == "User"]
        
        for node in user_nodes:
            # 获取用户的历史信息（如果有）
            user_attributes = node.attributes
            
            # 使用LLM判断用户类型
            user_type = self._classify_user_type(node.node_id, user_attributes, news_content)
            
            # 更新节点类型
            node.node_type = user_type
            node.attributes["refined_type"] = user_type
        
        return graph
    
    def _classify_user_types_batch(self, user_list: List[Tuple[str, Dict]], 
                                  batch_size: int = 10, show_progress: bool = True) -> Dict[str, str]:
        """
        批量分类用户类型（优化：一次处理多个用户，支持并行处理多个批次）
        
        Args:
            user_list: [(user_id, user_attributes), ...]
            batch_size: 每批处理的用户数量
            show_progress: 是否显示进度条
            
        Returns:
            {user_id: refined_type, ...}
        """
        results = {}
        total_batches = (len(user_list) + batch_size - 1) // batch_size
        
        # 导入tqdm（如果可用）
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
        
        # 准备所有批次
        batches = []
        for i in range(0, len(user_list), batch_size):
            batches.append((i, user_list[i:i+batch_size]))
        
        # 并行处理批次（如果批次数量较多，使用并行）
        parallel_batch_threshold = ENHANCEMENT_CONFIG.get("parallel_batch_threshold", 3)
        use_parallel = len(batches) >= parallel_batch_threshold
        
        if use_parallel:
            # 并行处理多个批次
            max_workers = min(ENHANCEMENT_CONFIG.get("batch_parallel_workers", 5), len(batches))
            if show_progress:
                print(f"  使用 {max_workers} 个线程并行处理 {len(batches)} 个批次...")
            
            def process_single_batch(batch_data):
                """处理单个批次"""
                batch_idx, batch = batch_data
                return batch_idx, self._process_user_batch(batch)
            
            batch_results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_batch, batch_data): batch_data 
                          for batch_data in batches}
                
                batch_range = as_completed(futures) if not (show_progress and has_tqdm) else \
                             tqdm(as_completed(futures), total=len(futures), 
                                 desc="  并行处理用户批次", unit="批", leave=False)
                
                for future in batch_range:
                    try:
                        batch_idx, batch_result = future.result()
                        batch_results[batch_idx] = batch_result
                    except Exception as e:
                        batch_idx = futures[future][0]
                        print(f"  警告：处理批次 {batch_idx} 时出错: {e}")
                        # 使用默认值
                        batch_results[batch_idx] = {}
            
            # 合并结果
            for batch_idx in sorted(batch_results.keys()):
                results.update(batch_results[batch_idx])
        else:
            # 串行处理（批次较少时）
            batch_range = range(0, len(user_list), batch_size)
            if show_progress and has_tqdm:
                batch_range = tqdm(batch_range, desc="  批量处理用户节点", unit="批", leave=False)
            
            for i in batch_range:
                batch = user_list[i:i+batch_size]
                batch_result = self._process_user_batch(batch)
                results.update(batch_result)
        
        return results
    
    def _process_user_batch(self, batch: List[Tuple[str, Dict]]) -> Dict[str, str]:
        """处理单个用户批次（内部方法）"""
        results = {}
        
        # 构建批量prompt
        user_items = []
        for idx, (user_id, user_attributes) in enumerate(batch):
            # 提取用户信息
            user_info = {
                "编号": idx + 1,
                "用户ID": user_id
            }
            
            # 添加基本属性
            if user_attributes:
                if 'interaction_type' in user_attributes:
                    user_info["交互类型"] = user_attributes['interaction_type']
                if 'name' in user_attributes:
                    user_info["名称"] = user_attributes['name']
                if 'verified' in user_attributes:
                    user_info["认证状态"] = user_attributes['verified']
            
            # 添加评论内容（最重要的信息）
            if 'comments' in user_attributes and user_attributes['comments']:
                comments = user_attributes['comments']
                user_info["评论内容"] = comments
                user_info["评论数量"] = user_attributes.get('comment_count', len(comments))
            else:
                user_info["评论内容"] = []
                user_info["评论数量"] = 0
            
            user_items.append(user_info)
        
        # 构建更清晰的prompt，包含评论内容
        user_list_str = ""
        for item in user_items:
            user_list_str += f"\n{item['编号']}. 用户ID: {item['用户ID']}"
            if '交互类型' in item:
                user_list_str += f", 交互类型: {item['交互类型']}"
            if '评论数量' in item and item['评论数量'] > 0:
                user_list_str += f", 评论数量: {item['评论数量']}"
                if '评论内容' in item and item['评论内容']:
                    user_list_str += f"\n   评论示例:"
                    for i, comment in enumerate(item['评论内容'][:3], 1):  # 最多显示3条
                        user_list_str += f"\n     {i}. {comment}"
            else:
                user_list_str += f", 无评论内容"
            user_list_str += "\n"
        
        prompt = f"""请批量判断以下用户的类型。

{user_list_str}

判断标准：
- User-Normal: 正常用户，生成内容自然、有语境相关性、逻辑连贯
- User-Suspicious: 可疑用户，呈现机器人或恶意行为特征（大量刷屏、仇恨言论、高度雷同内容、无意义评论）

请仔细分析每个用户的评论内容，判断其行为特征。

请对每个用户，严格按照以下格式返回，每行一个结果：
1 User-Normal
2 User-Suspicious
3 User-Normal
4 User-Suspicious
...

严格要求：
1. 每行格式：编号 空格 User-Normal 或 User-Suspicious
2. 不要添加任何其他文字、标点、解释或空行
3. 必须返回所有用户的结果"""

        try:
            # 根据批量大小动态计算max_tokens
            estimated_tokens = len(batch) * 30 + 100
            max_tokens = max(estimated_tokens, 500)
            
            response = self.llm_client.call(prompt, max_tokens=max_tokens)
            if response is None:
                # 默认全部为Normal
                for user_id, _ in batch:
                    results[user_id] = "User-Normal"
                return results
            
            # 解析批量结果
            lines = response.strip().split('\n')
            parsed_count = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 解析格式：编号 User-Normal 或 编号 User-Suspicious
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        idx = int(parts[0]) - 1  # 转为0-based索引
                        if 0 <= idx < len(batch):
                            user_type_str = parts[1].strip()
                            # 匹配用户类型
                            if "Suspicious" in user_type_str or "suspicious" in user_type_str.lower():
                                user_type = "User-Suspicious"
                            else:
                                user_type = "User-Normal"
                            
                            user_id, _ = batch[idx]
                            results[user_id] = user_type
                            parsed_count += 1
                    except (ValueError, IndexError):
                        continue
            
            # 如果解析结果不完整，为未解析的用户设置默认值
            if parsed_count < len(batch):
                for user_id, _ in batch:
                    if user_id not in results:
                        results[user_id] = "User-Normal"
        
        except Exception as e:
            print(f"  处理用户批次时出错: {e}")
            # 默认全部为Normal
            for user_id, _ in batch:
                results[user_id] = "User-Normal"
        
        return results
    
    def _classify_user_type(self, user_id: str, user_attributes: Dict, 
                           news_content: str) -> str:
        """
        使用LLM分类用户类型（优化版：减少输入token，只输出结果）
        
        Returns:
            "User-Normal" 或 "User-Suspicious"
        """
        # 只提取关键用户属性，减少token量
        key_attrs = {}
        if user_attributes:
            # 只保留最重要的几个属性
            for key in ['interaction_type', 'name', 'verified']:
                if key in user_attributes:
                    key_attrs[key] = user_attributes[key]
            # 如果属性很少，保留前3个
            if len(key_attrs) < 2 and len(user_attributes) > 0:
                for i, (k, v) in enumerate(list(user_attributes.items())[:3]):
                    if k not in key_attrs:
                        key_attrs[k] = v
        
        # 限制新闻内容长度
        news_preview = news_content[:100] if news_content else ""
        
        prompt = f"""判断用户类型。

            用户ID: {user_id}
            关键属性: {json.dumps(key_attrs, ensure_ascii=False) if key_attrs else "无"}

            标准：
            - User-Normal: 正常用户
            - User-Suspicious: 可疑用户（机器人、水军、刷屏、仇恨言论）

            只返回：User-Normal 或 User-Suspicious，不要其他内容。"""

        try:
            response = self.llm_client.call(prompt)
            if response is None:
                return "User-Normal"
            
            response = response.strip()
            
            # 快速判断
            if "Suspicious" in response or "可疑" in response:
                return "User-Suspicious"
            else:
                return "User-Normal"
        except Exception as e:
            print(f"用户分类失败: {e}，默认返回User-Normal")
            return "User-Normal"
    
    def _refine_edges(self, enhanced_graph: PropagationGraph, 
                     original_graph: PropagationGraph) -> PropagationGraph:
        """边细化 - 添加认知立场和情感极性"""
        # 先复制所有边
        for edge in original_graph.edges:
            enhanced_graph.edges.append(Edge(
                source=edge.source,
                target=edge.target,
                edge_type=edge.edge_type,
                attributes=edge.attributes.copy()
            ))
            enhanced_graph.adjacency_list[edge.source].append(enhanced_graph.edges[-1])
        
        # 对每条Comment和Reply边进行细化
        for edge in list(enhanced_graph.edges):
            if edge.edge_type in ["Comment", "Reply"]:
                text = edge.attributes.get("text", "")
                if text:
                    # 分析认知立场和情感极性
                    stance, sentiment = self._analyze_interaction(text)
                    
                    # 创建细化的边类型
                    refined_edge_type = f"{edge.edge_type}-{stance}"
                    
                    # 更新边属性
                    edge.attributes["cognitive_stance"] = stance
                    edge.attributes["affective_sentiment"] = sentiment
                    
                    # 可选：创建新的边类型（保留原边，同时添加细化边）
                    if edge.source in enhanced_graph.nodes and edge.target in enhanced_graph.nodes:
                        new_edge = Edge(
                            source=edge.source,
                            target=edge.target,
                            edge_type=refined_edge_type,
                            attributes={
                                "text": text,
                                "cognitive_stance": stance,
                                "affective_sentiment": sentiment
                            }
                        )
                        enhanced_graph.edges.append(new_edge)
                        enhanced_graph.adjacency_list[edge.source].append(new_edge)
        
        return enhanced_graph
    
    def _analyze_interactions_batch(self, text_list: List[str], 
                                   batch_size: int = 15, show_progress: bool = True) -> List[Tuple[str, str]]:
        """
        批量分析交互的认知立场和情感极性（优化：一次处理多条边，支持并行处理多个批次）
        
        Args:
            text_list: 评论文本列表
            batch_size: 每批处理的评论数量
            show_progress: 是否显示进度条
            
        Returns:
            [(stance, sentiment), ...]
        """
        results = []
        total_batches = (len(text_list) + batch_size - 1) // batch_size
        
        # 导入tqdm（如果可用）
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
        
        # 准备所有批次
        batches = []
        for i in range(0, len(text_list), batch_size):
            batches.append((i, text_list[i:i+batch_size]))
        
        # 并行处理批次（如果批次数量较多，使用并行）
        parallel_batch_threshold = ENHANCEMENT_CONFIG.get("parallel_batch_threshold", 3)
        use_parallel = len(batches) >= parallel_batch_threshold
        
        if use_parallel:
            # 并行处理多个批次
            max_workers = min(ENHANCEMENT_CONFIG.get("batch_parallel_workers", 5), len(batches))
            if show_progress:
                print(f"  使用 {max_workers} 个线程并行处理 {len(batches)} 个批次...")
            
            def process_single_batch(batch_data):
                """处理单个批次"""
                batch_idx, batch = batch_data
                return batch_idx, self._process_edge_batch(batch)
            
            batch_results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_batch, batch_data): batch_data 
                          for batch_data in batches}
                
                batch_range = as_completed(futures) if not (show_progress and has_tqdm) else \
                             tqdm(as_completed(futures), total=len(futures), 
                                 desc="  并行处理边批次", unit="批", leave=False)
                
                for future in batch_range:
                    try:
                        batch_idx, batch_result = future.result()
                        batch_results[batch_idx] = batch_result
                    except Exception as e:
                        batch_idx = futures[future][0]
                        print(f"  警告：处理批次 {batch_idx} 时出错: {e}")
                        # 使用默认值
                        batch_results[batch_idx] = []
            
            # 合并结果（按批次索引排序）
            for batch_idx in sorted(batch_results.keys()):
                results.extend(batch_results[batch_idx])
        else:
            # 串行处理（批次较少时）
            batch_range = range(0, len(text_list), batch_size)
            if show_progress and has_tqdm:
                batch_range = tqdm(batch_range, desc="  批量处理边", unit="批", leave=False)
            
            for i in batch_range:
                batch = text_list[i:i+batch_size]
                batch_result = self._process_edge_batch(batch)
                results.extend(batch_result)
        
        return results
    
    def _process_edge_batch(self, batch: List[str]) -> List[Tuple[str, str]]:
        """处理单个边批次（内部方法）"""
        results = []
        
        # 构建批量prompt
        comments = []
        for idx, text in enumerate(batch):
            text_short = text[:80] if text else ""
            comments.append({
                "编号": idx + 1,
                "评论": text_short
            })
        
        # 构建更清晰的prompt
        comments_str = "\n".join([f"{item['编号']}. {item['评论']}" for item in comments])
        
        prompt = f"""请批量分析以下评论的认知立场和情感极性。

            {comments_str}

            立场选项（必须选择其一）: Agree, Disagree, Discuss, Unrelated
            情感选项（必须选择其一）: Angry, Happy, Neutral, Fear, Sad

            请对每个评论，严格按照以下格式返回，每行一个结果：
            1 Agree Happy
            2 Disagree Angry
            3 Discuss Neutral
            4 Unrelated Fear
            5 Agree Angry
            ...

            严格要求：
            1. 每行格式：编号 空格 立场 空格 情感
            2. 立场必须是：Agree, Disagree, Discuss, Unrelated 之一
            3. 情感必须是：Angry, Happy, Neutral, Fear, Sad 之一
            4. 不要添加任何其他文字、标点、解释或空行
            5. 必须返回所有评论的结果"""

        try:
            # 根据批量大小动态计算max_tokens
            estimated_tokens = len(batch) * 40 + 100
            max_tokens = max(estimated_tokens, 500)
            
            response = self.llm_client.call(prompt, max_tokens=max_tokens)
            if response is None:
                # 默认值
                for _ in batch:
                    results.append(("Discuss", "Neutral"))
                return results
            
            # 解析批量结果
            lines = response.strip().split('\n')
            batch_results = {}
            parsed_count = 0
            
            import re
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 支持多种格式：1 Agree Happy, 1. Agree Happy, 1: Agree Happy
                match = re.match(r'^(\d+)[\.:、\s]+(\w+)[\s]+(\w+)', line, re.IGNORECASE)
                if match:
                    try:
                        idx = int(match.group(1)) - 1  # 转为0-based索引
                        stance_str = match.group(2).strip()
                        sentiment_str = match.group(3).strip()
                        
                        # 匹配立场（精确匹配优先）
                        stance = None
                        for option in self.cognitive_stances:
                            if option.lower() == stance_str.lower():
                                stance = option
                                break
                        if stance is None:
                            # 模糊匹配
                            for option in self.cognitive_stances:
                                if option.lower() in stance_str.lower() or stance_str.lower() in option.lower():
                                    stance = option
                                    break
                        if stance is None:
                            stance = "Discuss"  # 默认值
                        
                        # 匹配情感（精确匹配优先）
                        sentiment = None
                        for option in self.affective_sentiments:
                            if option.lower() == sentiment_str.lower():
                                sentiment = option
                                break
                        if sentiment is None:
                            # 模糊匹配
                            for option in self.affective_sentiments:
                                if option.lower() in sentiment_str.lower() or sentiment_str.lower() in option.lower():
                                    sentiment = option
                                    break
                        if sentiment is None:
                            sentiment = "Neutral"  # 默认值
                        
                        if 0 <= idx < len(batch):
                            batch_results[idx] = (stance, sentiment)
                            parsed_count += 1
                    except (ValueError, IndexError):
                        continue
            
            # 按索引顺序构建结果列表
            for idx in range(len(batch)):
                if idx in batch_results:
                    results.append(batch_results[idx])
                else:
                    results.append(("Discuss", "Neutral"))  # 默认值
        
        except Exception as e:
            print(f"  处理边批次时出错: {e}")
            # 默认值
            for _ in batch:
                results.append(("Discuss", "Neutral"))
        
        return results
    
    def _analyze_interaction(self, text: str) -> Tuple[str, str]:
        """
        分析交互的认知立场和情感极性（优化版：只输出词，不要推理过程）
        
        Returns:
            (认知立场, 情感极性)
        """
        # 限制评论内容长度，减少token
        text_short = text[:80] if text else ""
        
        prompt = f"""分析评论的立场和情感。

评论: {text_short}

立场选项: Agree, Disagree, Discuss, Unrelated
情感选项: Angry, Happy, Neutral, Fear, Sad

只输出两个词，格式：立场 情感
例如：Agree Happy 或 Disagree Angry
不要任何解释或推理过程，只输出两个词。"""

        try:
            response = self.llm_client.call(prompt)
            
            # 解析响应（快速提取）
            stance = "Discuss"  # 默认值
            sentiment = "Neutral"  # 默认值
            
            # 先尝试提取两个词（空格分隔）
            words = response.strip().split()
            if len(words) >= 2:
                word1, word2 = words[0], words[1]
                # 检查第一个词是否是立场
                for option in self.cognitive_stances:
                    if option.lower() in word1.lower() or word1.lower() in option.lower():
                        stance = option
                        break
                # 检查第二个词是否是情感
                for option in self.affective_sentiments:
                    if option.lower() in word2.lower() or word2.lower() in option.lower():
                        sentiment = option
                        break
            else:
                # 如果格式不对，回退到原来的解析方式
                for option in self.cognitive_stances:
                    if option in response or option.lower() in response.lower():
                        stance = option
                        break
                
                for option in self.affective_sentiments:
                    if option in response or option.lower() in response.lower():
                        sentiment = option
                        break
            
            return stance, sentiment
        except Exception as e:
            print(f"交互分析失败: {e}，使用默认值")
            return "Discuss", "Neutral"
    
    def _add_topic_nodes(self, graph: PropagationGraph, news_content: str) -> PropagationGraph:
        """添加主题节点"""
        # 使用LLM识别新闻主题
        topic = self._classify_news_topic(news_content)
        
        # 创建主题节点
        topic_node_id = f"topic_{topic}"
        graph.add_node(topic_node_id, "Topic", {"topic_name": topic})
        
        # 将新闻节点与主题节点连接
        if graph.news_id in graph.nodes:
            graph.add_edge(graph.news_id, topic_node_id, "HasTopic")
            graph.add_edge(topic_node_id, graph.news_id, "HasTopic_inv")
        
        return graph
    
    def _classify_news_topic(self, news_content: str) -> str:
        """使用LLM分类新闻主题"""
        prompt = f"""请判断以下新闻的主题类别。

        新闻内容: {news_content[:500]}  # 限制长度

        主题类别: {', '.join(self.news_topics)}

        请只返回主题类别名称（英文），不要返回其他内容。"""

        try:
            response = self.llm_client.call(prompt)
            response = response.strip()
            
            # 检查响应是否包含预定义的主题
            for topic in self.news_topics:
                if topic.lower() in response.lower():
                    return topic
            
            return "other"
        except Exception as e:
            print(f"主题分类失败: {e}，默认返回other")
            return "other"

