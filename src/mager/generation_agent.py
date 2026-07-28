"""
生成智能体 (L_G) - 通过语义变异和交叉生成新元路径
"""
import random
from typing import List, Tuple, Optional
from meta_path import MetaPath
from llm_client import LLMClient
from config import GA_CONFIG
import json

class GenerationAgent:
    """生成智能体 - 负责元路径的语义变异和交叉"""
    
    def __init__(self):
        self.llm_client = LLMClient("agent_g")
        self.crossover_prob = GA_CONFIG["crossover_prob"]
        
        # 可用的节点类型和边类型（移除主题相关）
        self.available_node_types = [
            "News", "User", "User-Normal", "User-Suspicious"
        ]
        
        self.available_edge_types = [
            "Comment", "Reply", "Repost", "Attitude",
            # 立场细化
            "Comment-Agree", "Comment-Disagree", "Comment-Discuss", "Comment-Unrelated",
            "Reply-Agree", "Reply-Disagree", "Reply-Discuss", "Reply-Unrelated",
            # 情感细化
            "Comment-Angry", "Comment-Happy", "Comment-Neutral", "Comment-Fear", "Comment-Sad",
            "Reply-Angry", "Reply-Happy", "Reply-Neutral", "Reply-Fear", "Reply-Sad",
            # 组合细化示例
            "Comment-Agree-Happy", "Comment-Agree-Angry", "Comment-Agree-Neutral", "Comment-Agree-Fear", "Comment-Agree-Sad",
            "Comment-Disagree-Angry", "Comment-Disagree-Happy", "Comment-Disagree-Neutral", "Comment-Disagree-Fear", "Comment-Disagree-Sad",
            "Comment-Discuss-Angry", "Comment-Discuss-Happy", "Comment-Discuss-Neutral", "Comment-Discuss-Fear", "Comment-Discuss-Sad",
            "Comment-Unrelated-Angry", "Comment-Unrelated-Happy", "Comment-Unrelated-Neutral", "Comment-Unrelated-Fear", "Comment-Unrelated-Sad",
            "Reply-Agree-Happy", "Reply-Agree-Angry", "Reply-Agree-Neutral", "Reply-Agree-Fear", "Reply-Agree-Sad",
            "Reply-Disagree-Happy", "Reply-Disagree-Angry", "Reply-Disagree-Neutral", "Reply-Disagree-Fear", "Reply-Disagree-Sad",
            "Reply-Discuss-Happy", "Reply-Discuss-Angry", "Reply-Discuss-Neutral", "Reply-Discuss-Fear", "Reply-Discuss-Sad",
            "Reply-Unrelated-Happy", "Reply-Unrelated-Angry", "Reply-Unrelated-Neutral", "Reply-Unrelated-Fear", "Reply-Unrelated-Sad",
        ]
    
    def select_parents(self, population: List[MetaPath], 
                      tournament_size: int = 3) -> List[MetaPath]:
        """
        锦标赛选择父代元路径
        
        Args:
            population: 当前种群
            tournament_size: 锦标赛大小
            
        Returns:
            选中的父代元路径列表（1个用于变异，2个用于交叉）
        """
        selected = []
        
        # 选择用于变异的父代（单个）
        tournament = random.sample(population, min(tournament_size, len(population)))
        parent_mutation = max(tournament, key=lambda x: x.fitness)
        selected.append(parent_mutation)
        
        # 选择用于交叉的父代（两个）
        tournament1 = random.sample(population, min(tournament_size, len(population)))
        parent1 = max(tournament1, key=lambda x: x.fitness)
        
        tournament2 = random.sample(population, min(tournament_size, len(population)))
        parent2 = max(tournament2, key=lambda x: x.fitness)
        
        selected.extend([parent1, parent2])
        
        return selected
    
    def generate_candidate(self, parents: List[MetaPath]) -> MetaPath:
        """
        生成候选元路径（通过变异或交叉）
        
        Args:
            parents: 父代元路径列表 [mutation_parent, crossover_parent1, crossover_parent2]
            
        Returns:
            新生成的候选元路径
        """
        # 根据概率决定是变异还是交叉
        if random.random() < self.crossover_prob:
            # 语义交叉
            result = self._semantic_crossover(parents[1], parents[2])
            result.description = f"交叉: {result.description}"
            return result
        else:
            # 语义变异
            result = self._semantic_mutation(parents[0])
            result.description = f"变异: {result.description}"
            return result
    
    def _semantic_mutation(self, parent: MetaPath) -> MetaPath:
        """
        语义变异 - 对单个父代元路径进行局部细化
        
        Args:
            parent: 父代元路径
            
        Returns:
            变异后的新元路径
        """
        prompt = f"""请对以下元路径进行语义变异，生成一个新的、更有表达力的元路径。

原始元路径: {parent.to_formal_string()}
描述: {parent.description}

可用操作：
1. 类型细化：将通用类型细化为更具体的类型（如 User -> User-Suspicious）
2. 关系替换：将粗粒度关系替换为细粒度关系（如 Comment -> Comment-Agree）
3. 添加类型约束：在路径中添加更具体的节点或边类型
4. **增加边（扩展路径长度）**：在路径中添加新的边和节点，扩展路径长度（如从2跳扩展到3跳或4跳）
   - 可以添加 Reply 边连接用户
   - 可以添加 Comment 边连接用户和新闻（注意：Comment是单向边，只能从User指向News）
   - 可以添加 Repost 边连接用户和新闻

重要约束：
- Comment边是单向的，只能从User指向News，不能反向
- 路径长度可以是1跳、2跳、3跳或4跳
- 确保路径在语义上合理（如用户可以先回复其他用户，再评论新闻）

请生成一个新的元路径，格式为：
节点类型1 --[边类型1]--> 节点类型2 --[边类型2]--> 节点类型3 ...

可用节点类型: {', '.join(self.available_node_types)}
可用边类型: {', '.join(self.available_edge_types)}

只返回元路径字符串，不要其他说明。"""

        try:
            response = self.llm_client.call(prompt)
            new_path = self._parse_meta_path_from_string(response)
            
            if new_path and self._validate_meta_path(new_path):
                return MetaPath(
                    path=new_path,
                    description=f"从 {parent.description} 变异得到"
                )
            else:
                # 如果解析失败，返回轻微的随机变异
                print(f"  警告: 解析失败或路径无效，使用随机变异")
                return self._random_mutation(parent)
        except Exception as e:
            print(f"  警告: 语义变异失败: {e}，使用随机变异")
            return self._random_mutation(parent)
    
    def _semantic_crossover(self, parent1: MetaPath, parent2: MetaPath) -> MetaPath:
        """
        语义交叉 - 组合两个父代元路径的互补优势
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            交叉后的新元路径
        """
        prompt = f"""请对以下两个元路径进行语义交叉，生成一个新的、结合两者优势的元路径。

父代元路径1: {parent1.to_formal_string()}
描述: {parent1.description}

父代元路径2: {parent2.to_formal_string()}
描述: {parent2.description}

请将两条路径中互补的结构优势进行合成，生成一个拓扑上合法且语义上新颖的元路径。

交叉策略：
1. 可以组合两条路径的不同部分
2. 可以扩展路径长度（生成2跳、3跳或4跳的路径）
3. 可以结合两条路径的边类型和节点类型

重要约束：
- Comment边是单向的，只能从User指向News，不能反向
- 路径长度可以是1跳、2跳、3跳或4跳
- 确保路径在语义上合理

格式为：
节点类型1 --[边类型1]--> 节点类型2 --[边类型2]--> 节点类型3 ...

可用节点类型: {', '.join(self.available_node_types)}
可用边类型: {', '.join(self.available_edge_types)}

只返回元路径字符串，不要其他说明。"""

        try:
            response = self.llm_client.call(prompt)
            new_path = self._parse_meta_path_from_string(response)
            
            if new_path and self._validate_meta_path(new_path):
                return MetaPath(
                    path=new_path,
                    description=f"从 {parent1.description} 和 {parent2.description} 交叉得到"
                )
            else:
                # 如果解析失败，使用简单的路径拼接
                print(f"  警告: 解析失败或路径无效，使用简单交叉")
                return self._simple_crossover(parent1, parent2)
        except Exception as e:
            print(f"  警告: 语义交叉失败: {e}，使用简单交叉")
            return self._simple_crossover(parent1, parent2)
    
    def _parse_meta_path_from_string(self, text: str) -> Optional[List[Tuple[str, str]]]:
        """从字符串解析元路径"""
        # 移除多余的空格和换行
        text = text.strip()
        
        # 移除可能的markdown代码块标记
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join([l for l in lines if not l.strip().startswith("```")])
        text = text.strip()
        
        # 尝试解析格式: NodeType1 --[EdgeType]--> NodeType2 ...
        # 使用正则表达式更准确地解析
        import re
        
        # 匹配模式: NodeType --[EdgeType]--> NodeType
        pattern = r'(\w+(?:-\w+)*)\s*--\[([^\]]+)\]-->\s*(\w+(?:-\w+)*)'
        matches = re.findall(pattern, text)
        
        if not matches:
            # 如果正则匹配失败，尝试简单分割
            return self._parse_meta_path_simple(text)
        
        # 构建路径
        path = []
        for i, match in enumerate(matches):
            start_node, edge_type, end_node = match
            
            # 清理节点和边类型
            start_node = start_node.strip()
            edge_type = edge_type.strip()
            end_node = end_node.strip()
            
            # 验证节点类型和边类型
            if not start_node or not end_node or not edge_type:
                return None
            
            # 验证节点类型是否包含User或News
            if "User" not in start_node and "News" not in start_node:
                return None
            if "User" not in end_node and "News" not in end_node:
                return None
            
            # 验证边类型是否包含Comment、Reply或Repost
            if "Comment" not in edge_type and "Reply" not in edge_type and "Repost" not in edge_type:
                return None
            
            # 第一个元素：起始节点和第一条边
            if i == 0:
                path.append((start_node, edge_type))
            
            # 最后一个元素：结束节点，边为空
            if i == len(matches) - 1:
                path.append((end_node, ""))
            else:
                # 中间元素：节点和下一条边
                next_edge = matches[i + 1][1].strip()
                if not next_edge:
                    return None
                path.append((end_node, next_edge))
        
        # 验证路径格式：path格式为 [(node, edge), (node, edge), ..., (node, "")]
        if len(path) < 2:
            return None
        
        # 确保最后一个元素的边为空
        if path[-1][1] != "":
            path[-1] = (path[-1][0], "")
        
        # 确保没有空边类型（除了最后一个）
        for i, (node, edge) in enumerate(path[:-1]):
            if not edge or edge.strip() == "":
                return None
        
        return path
    
    def _parse_meta_path_simple(self, text: str) -> Optional[List[Tuple[str, str]]]:
        """简单解析方法（备用）"""
        # 分割路径部分
        parts = text.split("-->")
        
        if len(parts) < 2:
            return None
        
        path = []
        
        # 解析第一部分（起始节点和第一条边）
        first_part = parts[0].strip()
        if "--[" in first_part:
            start_node = first_part.split("--[")[0].strip()
            edge_match = first_part.split("[")[1].split("]")[0].strip() if "[" in first_part else ""
        else:
            start_node = first_part.strip()
            edge_match = ""
        
        if not start_node or not edge_match:
            return None
        
        path.append((start_node, edge_match))
        
        # 解析后续部分
        for i, part in enumerate(parts[1:], 1):
            part = part.strip()
            
            # 提取节点类型
            if "--[" in part:
                node_type = part.split("--[")[0].strip()
                # 提取下一条边（如果存在）
                if i < len(parts) - 1:
                    next_part = parts[i + 1].strip()
                    if "--[" in next_part:
                        next_edge = next_part.split("[")[1].split("]")[0].strip() if "[" in next_part else ""
                    else:
                        next_edge = ""
                else:
                    next_edge = ""
            else:
                node_type = part.strip()
                next_edge = ""
            
            if not node_type:
                break
            
            path.append((node_type, next_edge))
        
        # 确保最后一个元素的边为空
        if path and path[-1][1] != "":
            path[-1] = (path[-1][0], "")
        
        # 验证路径
        if len(path) < 2:
            return None
        
        # 确保没有空边类型（除了最后一个）
        for i, (node, edge) in enumerate(path[:-1]):
            if not edge or edge.strip() == "":
                return None
        
        return path
    
    def _random_mutation(self, parent: MetaPath) -> MetaPath:
        """随机变异（备用方法）"""
        new_path = parent.path.copy()
        
        if not new_path:
            return parent
        
        # 随机选择变异位置
        mutate_pos = random.randint(0, len(new_path) - 1)
        node_type, edge_type = new_path[mutate_pos]
        
        # 随机修改节点类型或边类型
        if random.random() < 0.5:
            # 变异节点类型
            if "User" in node_type:
                new_node_type = random.choice(["User", "User-Normal", "User-Suspicious"])
                new_path[mutate_pos] = (new_node_type, edge_type)
        else:
            # 变异边类型
            base_type = edge_type.split('-')[0]
            if base_type in ["Comment", "Reply"]:
                # 随机选择一种细化方式
                new_edge_type = random.choice([t for t in self.available_edge_types if t.startswith(base_type)])
                new_path[mutate_pos] = (node_type, new_edge_type)
        
        return MetaPath(
            path=new_path,
            description=f"随机变异自 {parent.description}"
        )
    
    def _validate_meta_path(self, path: List[Tuple[str, str]]) -> bool:
        """验证元路径是否有效"""
        if not path or len(path) < 2:
            return False
        
        # 检查是否有空边类型（除了最后一个）
        for i, (node, edge) in enumerate(path[:-1]):
            if not edge or edge.strip() == "":
                return False
        
        # 检查最后一个元素的边是否为空
        if path[-1][1] != "":
            return False
        
        # 检查节点类型是否有效
        for node, _ in path:
            if not node or node.strip() == "":
                return False
        
        return True
    
    def _simple_crossover(self, parent1: MetaPath, parent2: MetaPath) -> MetaPath:
        """简单交叉（备用方法）"""
        # 简单地将两条路径拼接或组合
        if len(parent1.path) >= 2 and len(parent2.path) >= 1:
            # 取parent1的前半部分和parent2的后半部分
            split_point = len(parent1.path) // 2
            new_path = parent1.path[:split_point] + parent2.path[-1:]
        else:
            # 如果路径太短，随机选择一条
            new_path = random.choice([parent1.path, parent2.path]).copy()
        
        # 确保路径有效
        if not self._validate_meta_path(new_path):
            # 如果无效，返回parent1的副本
            new_path = parent1.path.copy()
        
        return MetaPath(
            path=new_path,
            description=f"简单交叉自 {parent1.description} 和 {parent2.description}"
        )

