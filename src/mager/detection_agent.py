"""
检测智能体 (L_D) - 基于元路径进行虚假新闻检测推理
"""
from typing import List, Dict, Optional, Tuple
from graph_builder import PropagationGraph
from meta_path import MetaPath
from llm_client import LLMClient
from gicl import GraphInContextLearning

class DetectionAgent:
    """检测智能体 - 负责虚假新闻检测"""
    
    def __init__(self):
        self.llm_client = LLMClient("agent_d")
        self.gicl = GraphInContextLearning()
    
    def detect(self, news_text: str, news_graph: PropagationGraph,
               examples: List[Dict], meta_path: MetaPath) -> Tuple[int, str]:
        """
        基于元路径和示例进行虚假新闻检测
        
        Args:
            news_text: 新闻文本
            news_graph: 新闻的传播图
            examples: 检索到的示例列表（包含文本、标签、图）
            meta_path: 元路径
            
        Returns:
            (预测标签, 推理链)
            标签: 0=假新闻, 1=真新闻
        """
        # 构建提示词
        prompt = self._build_detection_prompt(news_text, news_graph, examples, meta_path)
        
        system_prompt = """你是一个专业的虚假新闻检测专家。你需要基于新闻内容和传播结构特征，判断新闻的真实性。

            请仔细分析：
            1. 新闻内容本身的可信度
            2. 传播结构中的模式（如用户行为、交互类型等）
            3. 与历史示例的相似性

            输出格式：
            预测: [0或1]  # 0表示假新闻，1表示真新闻
            推理: [你的推理过程，解释为什么做出这个判断]
            """

        try:
            response = self.llm_client.call(prompt, system_prompt)
            
            # 解析响应
            label, reasoning = self._parse_response(response)
            return label, reasoning
        except Exception as e:
            print(f"检测失败: {e}")
            # 默认返回
            return 0, f"检测过程出错: {str(e)}"
    
    def _build_detection_prompt(self, news_text: str, news_graph: PropagationGraph,
                               examples: List[Dict], meta_path: MetaPath) -> str:
        """
        构建检测提示词（使用论文格式：[Claim], [Comments], [Propagation Structure]）
        """
        # 将图转换为论文格式的文本描述
        comments_text, propagation_structure_text = self._graph_to_text(news_graph)
        
        # 限制新闻文本长度
        if len(news_text) > 2000:
            news_text = news_text[:2000] + "... (已截断)"
        
        prompt_parts = []
        
        # 按照论文的"Propagation Structure" prompt格式
        prompt_parts.append("请判断以下新闻的真实性。\n")
        prompt_parts.append(f"[Claim[id:{news_text}]]")
        prompt_parts.append(f"[Comments[{comments_text}]]")
        prompt_parts.append(f"[Propagation Structure:{propagation_structure_text}]")
        
        # 添加元路径信息（自适应方法的核心）
        prompt_parts.append(f"\n用于分析的元路径: {meta_path.to_formal_string()}")
        prompt_parts.append(f"元路径描述: {meta_path.description}")
        
        # 添加元路径匹配的子图信息
        from graph_builder import GraphBuilder
        instances = GraphBuilder.extract_meta_path_instances(news_graph, meta_path.path)
        prompt_parts.append(f"满足元路径的实例数量: {len(instances)}")
        if instances:
            prompt_parts.append(f"前5个实例: {instances[:5]}")
        
        # 添加示例（如果有）
        if examples:
            prompt_parts.append("\n相关历史示例（用于上下文学习）:")
            for i, example in enumerate(examples, 1):
                example_text = example.get("text", "")[:300]
                example_label = example.get("label", -1)
                label_str = "真新闻" if example_label == 1 else "假新闻"
                
                prompt_parts.append(f"\n示例 {i} ({label_str}):")
                prompt_parts.append(f"{example_text}")
        
        prompt_parts.append("\n\n说明：")
        prompt_parts.append("- [Claim[id:claim]] 表示新闻内容和ID")
        prompt_parts.append("- [Comments[id:comment_texts]] 表示评论列表，格式为 ['id1:text1', 'id2:text2', ...]")
        prompt_parts.append("- [Propagation Structure: ['id1 replied to id2', ...]] 表示传播结构")
        prompt_parts.append("- 元路径用于从完整图中提取相关的子图模式")
        
        prompt_parts.append("\n请根据新闻内容、评论内容、传播结构和元路径匹配的子图来判断该新闻是否为虚假新闻。")
        prompt_parts.append("注意分析：")
        prompt_parts.append("1. 评论中是否有质疑、反驳或指出问题的内容（可能暗示假新闻）")
        prompt_parts.append("2. 评论的情感倾向和可信度（愤怒、怀疑等负面情绪可能暗示假新闻）")
        prompt_parts.append("3. 传播结构的特征（如是否形成讨论链，用户是否进行深入讨论）")
        prompt_parts.append("4. 元路径匹配的子图模式是否揭示可疑的传播行为")
        
        prompt_parts.append("\n输出格式：\n预测: [0或1]\n推理: [你的推理过程]")
        
        return "\n".join(prompt_parts)
    
    def _graph_to_text(self, graph: PropagationGraph, max_comments: int = 200) -> tuple:
        """
        将传播图转换为论文格式的文本描述
        按照论文的格式：[Claim], [Comments], [Propagation Structure]
        
        Args:
            graph: PropagationGraph对象
            max_comments: 最大评论数量（避免超过token限制）
            
        Returns:
            (comments_text, propagation_structure_text): 评论文本和传播结构文本
        """
        # 收集所有评论（Comment边，用户对新闻的评论）
        comment_edges = [e for e in graph.edges if e.edge_type.startswith("Comment")]
        
        # 收集所有用户间回复（Reply边）
        reply_edges = [e for e in graph.edges if e.edge_type.startswith("Reply")]
        
        # 1. Comments格式: ['id1:comment_text1', 'id2:comment_text2', ...]
        comments_list = []
        comment_id_map = {}
        user_to_comment_edge = {}
        
        # 优先选择有文本的评论
        comments_with_text = [e for e in comment_edges if e.attributes.get("text", "").strip()]
        comments_without_text = [e for e in comment_edges if not e.attributes.get("text", "").strip()]
        
        # 先添加有文本的评论（最多max_comments条）
        display_comments = comments_with_text[:max_comments]
        
        for i, edge in enumerate(display_comments, 1):
            user_id = edge.source
            text = edge.attributes.get("text", "").strip()
            if text:
                # 使用user_id作为comment_id（如果user_id存在且不太长），否则使用序号
                if user_id and len(user_id) <= 50:
                    comment_id = user_id
                else:
                    comment_id = f"comment_{i}"
                
                # 限制评论长度（单条评论最多500字符，避免过长）
                text_display = text[:500] if len(text) <= 500 else text[:500] + "..."
                comment_id_map[user_id] = comment_id
                user_to_comment_edge[user_id] = edge
                comments_list.append(f"'{comment_id}:{text_display}'")
        
        # 如果还有空间，添加一些无文本的评论（用于完整性）
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
        
        # 2. Propagation Structure格式: ['id1 replied to id2', 'id3 replied to id1', ...]
        propagation_list = []
        
        # 记录哪些评论已经作为source出现在回复关系中
        comments_in_reply_as_source = set()
        
        # 处理用户间的回复关系（Reply边）
        for edge in reply_edges:
            source_id = edge.source
            target_id = edge.target
            
            # 如果target是新闻节点，则添加 'comment_id replied to news'
            if target_id == graph.news_id and source_id in comment_id_map:
                source_comment_id = comment_id_map[source_id]
                comments_in_reply_as_source.add(source_id)
                propagation_list.append(f"'{source_comment_id} replied to news'")
            # 如果source和target都有对应的评论ID，则添加用户间回复关系
            elif source_id in comment_id_map and target_id in comment_id_map:
                source_comment_id = comment_id_map[source_id]
                target_comment_id = comment_id_map[target_id]
                comments_in_reply_as_source.add(source_id)
                propagation_list.append(f"'{source_comment_id} replied to {target_comment_id}'")
        
        # 处理评论对新闻的直接回复关系（Comment边）
        # 只添加那些没有在reply_edges中作为source出现的评论
        for user_id, comment_id in comment_id_map.items():
            if user_id not in comments_in_reply_as_source:
                # 该评论还没有在传播结构中，添加它作为对新闻的回复
                propagation_list.append(f"'{comment_id} replied to news'")
        
        if propagation_list:
            propagation_structure_text = f"[{', '.join(propagation_list)}]"
        else:
            propagation_structure_text = "[]"
        
        return comments_text, propagation_structure_text
    
    def _extract_structure_info(self, graph: PropagationGraph, 
                               meta_path: MetaPath) -> str:
        """提取传播结构信息"""
        info_parts = []
        
        # 统计节点类型
        node_type_counts = {}
        for node in graph.nodes.values():
            node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1
        
        info_parts.append(f"节点类型统计: {node_type_counts}")
        
        # 统计边类型
        edge_type_counts = {}
        for edge in graph.edges:
            edge_type_counts[edge.edge_type] = edge_type_counts.get(edge.edge_type, 0) + 1
        
        info_parts.append(f"边类型统计: {edge_type_counts}")
        
        # 抽取满足元路径的实例
        from graph_builder import GraphBuilder
        instances = GraphBuilder.extract_meta_path_instances(graph, meta_path.path)
        info_parts.append(f"满足元路径的实例数量: {len(instances)}")
        
        if instances:
            info_parts.append(f"前3个实例: {instances[:3]}")
        
        return "\n".join(info_parts)
    
    def _parse_response(self, response: str) -> Tuple[int, str]:
        """解析LLM响应"""
        label = 0  # 默认值
        reasoning = response
        
        # 尝试提取预测标签
        lines = response.split('\n')
        for line in lines:
            if '预测:' in line or '预测：' in line:
                # 提取数字
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    label = int(numbers[0])
                    if label not in [0, 1]:
                        label = 0
            elif '推理:' in line or '推理：' in line:
                # 提取推理部分
                reasoning = line.split(':', 1)[-1].strip()
        
        # 如果没有找到推理部分，使用整个响应
        if '推理:' not in response and '推理：' not in response:
            reasoning = response
        
        return label, reasoning
    
    def generate_reasoning(self, news_text: str, news_graph: PropagationGraph,
                          meta_path: MetaPath, predicted_label: int) -> str:
        """
        生成显式推理链 R_D(v, M) (用于评估)
        
        Args:
            news_text: 新闻文本
            news_graph: 传播图
            meta_path: 元路径
            predicted_label: 预测标签
            
        Returns:
            推理链文本
        """
        prompt = f"""请详细解释为什么基于元路径 {meta_path.to_formal_string()} 抽取的子图支持你的预测。

新闻文本: {news_text[:500]}
预测标签: {"假新闻" if predicted_label == 0 else "真新闻"}

传播结构信息:
{self._extract_structure_info(news_graph, meta_path)}

请提供详细的推理过程，说明：
1. 元路径揭示的结构模式
2. 这些模式如何支持你的预测
3. 结构证据与预测的关联性

请用清晰、逻辑连贯的方式解释。"""

        try:
            reasoning = self.llm_client.call(prompt)
            return reasoning
        except Exception as e:
            return f"推理生成失败: {str(e)}"