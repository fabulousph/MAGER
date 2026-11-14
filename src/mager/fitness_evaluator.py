"""
适应度评估模块 - 计算元路径的适应度
"""
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from meta_path import MetaPath
from graph_builder import PropagationGraph
from detection_agent import DetectionAgent
from evaluation_agent import EvaluationAgent
from gicl import GraphInContextLearning
from embedding_cache import EmbeddingCache
from config import GA_CONFIG

class FitnessEvaluator:
    """适应度评估器"""
    
    def __init__(self):
        self.detection_agent = DetectionAgent()
        self.evaluation_agent = EvaluationAgent()
        self.gicl = GraphInContextLearning(top_k=4)  # 检索4条
        self.embedding_cache = EmbeddingCache()
        self._icl_pool = None  # 100条候选池
        self.max_workers = GA_CONFIG.get("max_workers", 10)
    
    def _load_icl_pool(self):
        """加载embedding_cache中的100条候选池"""
        if self._icl_pool is not None:
            return
        print("      [加载embedding_cache的候选池（论文默认 100 条监督池）]")
        self._icl_pool = self.embedding_cache.prepare_icl_pool(pool_size=None, force_recompute=False)  # None 表示使用论文默认的 100 条
        print(f"      [已加载 {len(self._icl_pool)} 条候选池数据]")
    
    def evaluate_fitness(self, meta_path: MetaPath, labeled_dataset: List[Dict],
                        enhanced_graphs: Dict[str, PropagationGraph]) -> float:
        """
        评估元路径的适应度 R_final (式8)
        
        Args:
            meta_path: 候选元路径
            labeled_dataset: 带标签的数据集
            enhanced_graphs: 增强后的图字典 {news_id: graph}
            
        Returns:
            总适应度得分
        """
        print(f"\n    {'='*70}")
        print(f"    [开始评估] 元路径: {meta_path.to_formal_string()}")
        print(f"              描述: {meta_path.description}")
        print(f"    {'='*70}")
        
        # 计算检测准确度 R_det (式6)
        print(f"\n    [步骤1] 计算检测准确度 R_det...")
        r_det = self._compute_detection_accuracy(meta_path, labeled_dataset, enhanced_graphs)
        
        # 计算推理质量 R_rea (式7)
        print(f"\n    [步骤2] 计算推理质量 R_rea...")
        r_rea = self._compute_reasoning_quality(meta_path, labeled_dataset, enhanced_graphs)
        
        # 总适应度（式7）：R(M) = R_det(M) + μ * R_rea(M)
        mu = GA_CONFIG.get("mu_reasoning", 0.25)
        r_final = r_det + mu * r_rea
        
        # 输出详细适应度分解
        print(f"\n    {'='*70}")
        print(f"    [评估结果汇总]")
        print(f"    {'='*70}")
        print(f"    R_det (检测准确度): {r_det:.4f}  (正确预测数 / 总样本数)")
        print(f"    R_rea (推理质量):   {r_rea:.4f}  (平均推理质量得分)")
        print(f"    R_final (总得分):  {r_final:.4f}  (R_det + {mu:g} * R_rea)")
        print(f"    {'='*70}\n")
        
        meta_path.fitness = r_final
        return r_final
    
    def _compute_detection_accuracy(self, meta_path: MetaPath,
                                   labeled_dataset: List[Dict],
                                   enhanced_graphs: Dict[str, PropagationGraph]) -> float:
        """
        并行计算检测准确度 R_det (式6)
        使用embedding_cache的100条候选池，从中检索4条最相关的进行评估
        """
        # 加载embedding_cache的100条候选池
        if self._icl_pool is None:
            self._load_icl_pool()
        
        # labeled_dataset是查询集（用于评估的样本）
        query_samples = labeled_dataset
        
        if len(query_samples) == 0:
            print(f"      ⚠️  警告: 查询集为空")
            return 0.0
        
        # 使用embedding_cache的100条候选池作为上下文数据
        context_data = []
        for example in self._icl_pool:
            context_data.append({
                "text": example.get("text", ""),
                "label": example.get("label", 0),
                "graph": example.get("graph"),
                "embedding": example.get("embedding"),
                "news_id": example.get("news_id")
            })
        
        print(f"      [使用embedding_cache的100条候选池，从中检索4条进行评估]")
        print(f"      [查询集大小: {len(query_samples)}]")

        # 存储每个样本的详细结果
        sample_results = []
        
        def process_single_sample(query_item):
            article = query_item.get("article", {})
            news_text = article.get("article_content", "")
            if not news_text:
                return None, None, None, None, None
            
            news_id = f"news_{hash(news_text) % 1000000}"
            true_label = query_item.get("label", 0)
            
            # 获取增强后的图
            graph = enhanced_graphs.get(news_id)
            if graph is None:
                # 如果enhanced_graphs中没有，尝试从query_item构建
                return None, None, None, None, None
            
            # 使用GICL从100条候选池中检索4条最相关的示例
            try:
                query_embedding = self.gicl.encoder.encode([news_text])[0]
                examples = self.gicl.retrieve_examples(
                    news_text, graph, context_data, meta_path, query_embedding=query_embedding,
                    exclude_news_id=news_id
                )
                
                # 确保只使用检索到的4条（或更少）
                examples = examples[:4]
                
                if len(examples) == 0:
                    print(f"      ⚠️  警告: 样本 {news_id[:20]}... 未检索到示例")
                    return None, None, None, None, None
                
                # 检测（只用检索到的4条示例）
                predicted_label, reasoning = self.detection_agent.detect(
                    news_text, graph, examples, meta_path
                )
                
                is_correct = predicted_label == true_label
                return news_id, true_label, predicted_label, is_correct, reasoning
            except Exception as e:
                print(f"      ⚠️  错误: 处理样本 {news_id[:20]}... 时出错: {str(e)}")
                return None, None, None, None, None

        correct_predictions = 0
        total_samples = 0
        
        # 导入tqdm并使用线程池并行执行
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(query_samples), desc="    [并行准确度评估]", leave=False, unit="条")
        except ImportError:
            pbar = None
            
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sample = {executor.submit(process_single_sample, item): item for item in query_samples}
            for future in as_completed(future_to_sample):
                result = future.result()
                if result[0] is not None:  # news_id不为None
                    news_id, true_label, predicted_label, is_correct, reasoning = result
                    sample_results.append({
                        "news_id": news_id,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "is_correct": is_correct,
                        "reasoning": reasoning[:200] if reasoning else ""  # 只保存前200字符
                    })
                    if is_correct:
                        correct_predictions += 1
                    total_samples += 1
                if pbar:
                    pbar.update(1)
        
        if pbar:
            pbar.close()

        if total_samples == 0:
            print(f"      ⚠️  警告: 没有有效样本进行评估")
            return 0.0
        
        accuracy = correct_predictions / total_samples
        
        # 打印详细结果
        print(f"      [R_det 详细结果]")
        print(f"      - 总样本数: {total_samples}")
        print(f"      - 正确预测: {correct_predictions}")
        print(f"      - 错误预测: {total_samples - correct_predictions}")
        print(f"      - 准确率: {accuracy:.4f}")
        
        # 打印前10个样本的详细结果
        # print(f"\n      [前10个样本的详细预测结果]")
        # for i, result in enumerate(sample_results[:10], 1):
        #     status = "✓" if result["is_correct"] else "✗"
        #     true_str = "真新闻" if result["true_label"] == 1 else "假新闻"
        #     pred_str = "真新闻" if result["predicted_label"] == 1 else "假新闻"
        #     print(f"      样本{i}: {status} 真实={true_str} 预测={pred_str} (ID: {result['news_id'][:30]}...)")
        #     if result["reasoning"]:
        #         print(f"              推理: {result['reasoning'][:100]}...")
        
        # if len(sample_results) > 10:
        #     print(f"      ... (还有 {len(sample_results) - 10} 个样本未显示)")
        
        return accuracy
    
    def _compute_reasoning_quality(self, meta_path: MetaPath,
                                  labeled_dataset: List[Dict],
                                  enhanced_graphs: Dict[str, PropagationGraph]) -> float:
        """
        并行计算推理质量得分 R_rea (式7)
        使用embedding_cache的100条候选池，从中检索4条最相关的进行评估
        """
        # 加载embedding_cache的100条候选池
        if self._icl_pool is None:
            self._load_icl_pool()
        
        # 使用全部数据集进行评估（与检测准确度评估保持一致）
        eval_samples = labeled_dataset
        
        # 使用embedding_cache的100条候选池作为上下文数据
        context_data = []
        for example in self._icl_pool:
            context_data.append({
                "text": example.get("text", ""),
                "label": example.get("label", 0),
                "graph": example.get("graph"),
                "embedding": example.get("embedding")
            })
        
        print(f"      [使用embedding_cache的100条候选池，从中检索4条进行评估]")
        print(f"      [评估样本数: {len(eval_samples)}]")

        # 存储每个样本的详细结果
        reasoning_results = []
        
        def process_reasoning_sample(eval_item):
            article = eval_item.get("article", {})
            news_text = article.get("article_content", "")
            news_id = f"news_{hash(news_text) % 1000000}"
            true_label = eval_item.get("label", 0)
            
            # 获取增强后的图
            graph = enhanced_graphs.get(news_id)
            if graph is None:
                return None, None, None, None, None, None
            
            # 使用GICL从100条候选池中检索4条最相关的示例
            query_embedding = self.gicl.encoder.encode([news_text])[0]
            examples = self.gicl.retrieve_examples(
                news_text, graph, context_data, meta_path, query_embedding=query_embedding,
                exclude_news_id=news_id
            )
            
            # 确保只使用检索到的4条（或更少）
            examples = examples[:4]
            
            # 检测并生成推理（只用检索到的4条示例）
            predicted_label, _ = self.detection_agent.detect(
                news_text, graph, examples, meta_path
            )
            
            # 生成显式推理链
            reasoning = self.detection_agent.generate_reasoning(
                news_text, graph, meta_path, predicted_label
            )
            
            # 提取结构证据
            structure_evidence = self.detection_agent._extract_structure_info(graph, meta_path)
            
            # 评估推理质量
            reasoning_score = self.evaluation_agent.evaluate_reasoning(
                reasoning, true_label, meta_path, structure_evidence
            )
            
            return news_id, true_label, predicted_label, reasoning_score, reasoning, structure_evidence

        total_score = 0.0
        total_samples = 0
        
        # 导入tqdm并使用线程池并行执行
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(eval_samples), desc="    [并行推理评估]", leave=False, unit="条")
        except ImportError:
            pbar = None
            
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sample = {executor.submit(process_reasoning_sample, item): item for item in eval_samples}
            for future in as_completed(future_to_sample):
                result = future.result()
                if result[0] is not None:  # news_id不为None
                    news_id, true_label, predicted_label, reasoning_score, reasoning, structure_evidence = result
                    reasoning_results.append({
                        "news_id": news_id,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "reasoning_score": reasoning_score,
                        "reasoning": reasoning[:200] if reasoning else "",
                        "structure_evidence": structure_evidence[:200] if structure_evidence else ""
                    })
                    total_score += reasoning_score
                    total_samples += 1
                if pbar:
                    pbar.update(1)
        
        if pbar:
            pbar.close()

        if total_samples == 0:
            print(f"      ⚠️  警告: 没有有效样本进行推理质量评估")
            return 0.0
        
        avg_score = total_score / total_samples
        
        # 打印详细结果
        print(f"      [R_rea 详细结果]")
        print(f"      - 评估样本数: {total_samples}")
        print(f"      - 总推理得分: {total_score:.4f}")
        print(f"      - 平均推理质量: {avg_score:.4f}")
        print(f"      - 最高得分: {max(r['reasoning_score'] for r in reasoning_results):.4f}")
        print(f"      - 最低得分: {min(r['reasoning_score'] for r in reasoning_results):.4f}")
        
        # 打印前10个样本的详细结果
        print(f"\n      [前10个样本的推理质量评估结果]")
        for i, result in enumerate(reasoning_results[:10], 1):
            true_str = "真新闻" if result["true_label"] == 1 else "假新闻"
            pred_str = "真新闻" if result["predicted_label"] == 1 else "假新闻"
            label_match = "✓" if result["true_label"] == result["predicted_label"] else "✗"
            print(f"      样本{i}: 得分={result['reasoning_score']:.4f} {label_match} 真实={true_str} 预测={pred_str}")
            print(f"              (ID: {result['news_id'][:30]}...)")
            if result["reasoning"]:
                print(f"              推理: {result['reasoning'][:100]}...")
            if result["structure_evidence"]:
                print(f"              结构证据: {result['structure_evidence'][:100]}...")
        
        if len(reasoning_results) > 10:
            print(f"      ... (还有 {len(reasoning_results) - 10} 个样本未显示)")
        
        return avg_score

