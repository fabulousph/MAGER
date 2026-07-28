"""
评估智能体 (L_E) - 评估检测智能体的推理质量
"""
from typing import Tuple
from llm_client import LLMClient
from graph_builder import PropagationGraph
from meta_path import MetaPath

class EvaluationAgent:
    """评估智能体 - 负责评估推理质量"""
    
    def __init__(self):
        self.llm_client = LLMClient("agent_e")
    
    def evaluate_reasoning(self, reasoning: str, true_label: int,
                          meta_path: MetaPath, structure_evidence: str) -> float:
        """
        评估推理质量 R_rea (式7)
        
        Args:
            reasoning: 检测智能体的推理链
            true_label: 真实标签 (0=假新闻, 1=真新闻)
            meta_path: 元路径
            structure_evidence: 结构证据信息
            
        Returns:
            推理质量得分 (0-1)
        """
        prompt = self._build_evaluation_prompt(reasoning, true_label, meta_path, structure_evidence)
        
        system_prompt = """你是一个专业的推理质量评估专家。你需要严格评估一段推理过程的质量。

评估标准（三个维度，每个维度0-1分，综合评分取平均值）：
1. **与真实标签的一致性（Label Consistency）**：
   - 如果推理得出的结论与真实标签一致，且推理过程支持该结论，得1.0分
   - 如果推理得出的结论与真实标签一致，但推理过程不够充分，得0.6-0.8分
   - 如果推理得出的结论与真实标签不一致，得0.0-0.3分（即使推理过程看似合理）

2. **逻辑连贯性（Logical Coherence）**：
   - 推理过程逻辑清晰、步骤明确、前后一致，得0.8-1.0分
   - 推理过程基本连贯，但存在一些跳跃或不够清晰的地方，得0.5-0.7分
   - 推理过程混乱、前后矛盾、逻辑不清，得0.0-0.4分

3. **结构证据忠实性（Structural Evidence Faithfulness）**：
   - 推理充分使用了元路径提供的结构证据，且解释合理，得0.8-1.0分
   - 推理提到了结构证据，但使用不够充分或解释不够深入，得0.5-0.7分
   - 推理忽略了结构证据或对结构证据的解释明显错误，得0.0-0.4分

**重要原则**：
- 如果推理得出的结论与真实标签不一致，即使推理过程看似合理，综合评分也应该较低（不超过0.5）
- 评分要严格，避免给分过高
- 必须同时考虑三个维度，不能只看其中一个

请给出一个0-1之间的综合评分（取三个维度的平均值），并简要说明每个维度的得分。"""

        try:
            response = self.llm_client.call(prompt, system_prompt)
            score = self._parse_score(response)
            return score
        except Exception as e:
            print(f"推理评估失败: {e}")
            return 0.5  # 默认得分
    
    def _build_evaluation_prompt(self, reasoning: str, true_label: int,
                                 meta_path: MetaPath, structure_evidence: str) -> str:
        """构建评估提示词"""
        label_str = "假新闻" if true_label == 0 else "真新闻"
        
        prompt = f"""请严格评估以下推理过程的质量。

**真实标签**: {label_str}
**元路径**: {meta_path.to_formal_string()}
**结构证据**: {structure_evidence}

**检测智能体的推理**:
{reasoning}

**评估任务**：
请从以下三个维度分别评估，然后计算综合评分（取三个维度的平均值）：

1. **与真实标签的一致性（Label Consistency）**：
   - 推理得出的结论是什么？（真新闻/假新闻）
   - 该结论是否与真实标签一致？
   - 推理过程是否充分支持该结论？
   - 得分：___（0-1之间）

2. **逻辑连贯性（Logical Coherence）**：
   - 推理过程是否逻辑清晰、步骤明确？
   - 推理步骤之间是否前后一致？
   - 是否存在逻辑跳跃或矛盾？
   - 得分：___（0-1之间）

3. **结构证据忠实性（Structural Evidence Faithfulness）**：
   - 推理是否充分使用了元路径提供的结构证据？
   - 对结构证据的解释是否合理？
   - 是否忽略了重要的结构信息？
   - 得分：___（0-1之间）

**重要提醒**：
- 如果推理得出的结论与真实标签不一致，即使推理过程看似合理，综合评分也应该较低（不超过0.5）
- 评分要严格，避免给分过高
- 综合评分 = (维度1得分 + 维度2得分 + 维度3得分) / 3

**输出格式**（必须严格遵循）：
维度1得分: [0-1之间的数字]
维度2得分: [0-1之间的数字]
维度3得分: [0-1之间的数字]
综合评分: [0-1之间的数字]
评估理由: [简要说明每个维度的评估理由]"""

        return prompt
    
    def _parse_score(self, response: str) -> float:
        """解析评分"""
        import re
        
        # 优先尝试提取"综合评分"后的数字
        patterns = [
            r'综合评分[：:]\s*([0-9.]+)',
            r'综合得分[：:]\s*([0-9.]+)',
            r'最终评分[：:]\s*([0-9.]+)',
            r'评分[：:]\s*([0-9.]+)',  # 通用评分
            r'得分[：:]\s*([0-9.]+)',  # 通用得分
            r'score[：:]\s*([0-9.]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[0])
                    # 如果数字大于1，可能是百分比，需要除以100
                    if score > 1.0 and score <= 100.0:
                        score = score / 100.0
                    if 0.0 <= score <= 1.0:
                        return score
                except ValueError:
                    continue
        
        # 尝试提取三个维度的得分并计算平均值
        dimension_scores = []
        dim_patterns = [
            r'维度1得分[：:]\s*([0-9.]+)',
            r'维度2得分[：:]\s*([0-9.]+)',
            r'维度3得分[：:]\s*([0-9.]+)',
        ]
        for pattern in dim_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[0])
                    if score > 1.0 and score <= 100.0:
                        score = score / 100.0
                    if 0.0 <= score <= 1.0:
                        dimension_scores.append(score)
                except ValueError:
                    continue
        
        if len(dimension_scores) >= 2:
            # 如果找到至少2个维度得分，计算平均值
            avg_score = sum(dimension_scores) / len(dimension_scores)
            return avg_score
        
        # 尝试提取所有0-1之间的数字，取第一个有效的
        float_pattern = r'\b(0\.\d+|1\.0|1|0)\b'
        matches = re.findall(float_pattern, response)
        if matches:
            for match in matches:
                try:
                    score = float(match)
                    if 0.0 <= score <= 1.0:
                        return score
                except ValueError:
                    continue
        
        # 如果没有找到明确的数字，尝试从文本中推断（但降低默认得分）
        if '优秀' in response or '完美' in response or 'excellent' in response.lower():
            return 0.8  # 降低默认高分
        elif '良好' in response or 'good' in response.lower():
            return 0.6
        elif '一般' in response or 'average' in response.lower():
            return 0.4
        elif '较差' in response or 'poor' in response.lower() or '不一致' in response:
            return 0.2
        else:
            return 0.4  # 降低默认得分，更严格

