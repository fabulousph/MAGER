"""
元路径进化脚本 - 找到最好的3条元路径，然后使用v2方式进行评估
"""
import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from data_loader import DataLoader
from graph_builder import GraphBuilder, PropagationGraph
from meta_path import MetaPathInitializer, MetaPath
from genetic_algorithm import GeneticAlgorithm
from detector import FewShotSelectedMetaPathDetector
from config import GA_CONFIG

def evolve_and_evaluate(
    dataset_limit: Optional[int] = None,
    max_iterations: Optional[int] = None,
    population_size: Optional[int] = None,
    use_enhanced_dataset: bool = True,
    test_limit: Optional[int] = None
):
    """
    进化元路径并评估
    
    Args:
        dataset_limit: 用于进化的数据集大小限制（前100条作为候选池，剩余作为训练集）
        max_iterations: 最大迭代次数
        population_size: 种群大小
        use_enhanced_dataset: 是否使用增强数据集
        test_limit: 最终测试时的样本数量限制
    """
    print("="*80)
    print("元路径进化与评估流程")
    print("="*80)
    
    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    data_loader = DataLoader()
    graph_builder = GraphBuilder()
    
    if use_enhanced_dataset:
        try:
            labeled_dataset = data_loader.load_enhanced_dataset()
            print(f"已加载 {len(labeled_dataset)} 条增强数据")
            if dataset_limit:
                labeled_dataset = labeled_dataset[:dataset_limit]
                print(f"限制为前 {len(labeled_dataset)} 条")
        except FileNotFoundError:
            print("警告: 未找到增强数据集，使用原始数据集")
            labeled_dataset = data_loader.get_labeled_dataset(limit=dataset_limit)
    else:
        labeled_dataset = data_loader.get_labeled_dataset(limit=dataset_limit)
        print(f"已加载 {len(labeled_dataset)} 条带标签数据")
    
    # 2. 使用embedding_cache的100条作为训练集（用于进化评估）
    print("\n[步骤2] 准备训练集（使用embedding_cache的100条）...")
    from embedding_cache import EmbeddingCache
    
    embedding_cache = EmbeddingCache()
    icl_pool = embedding_cache.prepare_icl_pool(pool_size=None, force_recompute=False)  # None 表示使用论文默认的 100 条监督池
    
    print(f"训练集大小: {len(icl_pool)} 条（来自embedding_cache）")
    
    # 构建训练集的图（用于进化时的适应度评估）
    enhanced_graphs = {}
    training_dataset = []
    
    print("  构建训练集的图...")
    for i, example in enumerate(icl_pool):
        if (i + 1) % 20 == 0:
            print(f"    进度: {i + 1}/{len(icl_pool)}")
        
        item = example.get("item")
        news_text = example.get("text", "")
        if not news_text or item is None:
            continue
        
        news_id = f"news_{hash(news_text) % 1000000}"
        
        # 使用已有的图（如果example中有）
        graph = example.get("graph")
        if graph is None:
            # 如果没有图，需要构建
            has_enhanced_data = ("enhanced_nodes" in item or "enhanced_edges" in item)
            if has_enhanced_data:
                graph = graph_builder.build_enhanced_from_data(item, news_id)
            else:
                graph = graph_builder.build_from_data(item, news_id)
        
        enhanced_graphs[news_id] = graph
        
        # 准备训练数据集项
        training_item = item.copy()
        training_item['label'] = example.get("label", 0)
        training_dataset.append(training_item)
    
    print(f"已构建 {len(enhanced_graphs)} 个增强图（用于进化）")
    print(f"训练集大小: {len(training_dataset)} 条")
    
    # 3. 初始化元路径种群
    print("\n[步骤3] 初始化元路径种群...")
    meta_path_initializer = MetaPathInitializer()
    initial_population = meta_path_initializer.get_initial_meta_paths()
    
    if population_size:
        initial_population = initial_population[:population_size]
    
    print(f"初始种群大小: {len(initial_population)}")
    print("初始元路径示例:")
    for i, mp in enumerate(initial_population[:5], 1):
        print(f"  {i}. {mp.description}: {mp.to_formal_string()}")
    
    # 4. 执行进化
    print("\n[步骤4] 执行元路径进化...")
    genetic_algorithm = GeneticAlgorithm()
    
    if max_iterations:
        genetic_algorithm.max_iterations = max_iterations
    
    evolved_population = genetic_algorithm.evolve(
        initial_population=initial_population,
        labeled_dataset=training_dataset,
        enhanced_graphs=enhanced_graphs
    )
    
    # 5. 使用整个进化种群 P* 做推理（论文 §4.5：Eq(10) 对整个种群加权聚合，而非 Top-3）
    print("\n[步骤5] 汇总用于推理的元路径集合 P*（整个进化种群）...")
    # 按适应度降序排列；对形式化字符串去重，避免重复路径在 Eq(10) 中被重复计票
    sorted_population = sorted(evolved_population, key=lambda x: x.fitness, reverse=True)
    best_meta_paths = []
    seen_formal = set()
    for mp in sorted_population:
        formal = mp.to_formal_string()
        if formal in seen_formal:
            continue
        seen_formal.add(formal)
        best_meta_paths.append(mp)

    print(f"\n推理将聚合 {len(best_meta_paths)} 条元路径（去重后）:")
    for i, mp in enumerate(best_meta_paths, 1):
        print(f"  {i}. {mp.description}: {mp.to_formal_string()}")
        print(f"     适应度: {mp.fitness:.4f}")

    # 保存元路径信息
    print("\n[步骤5.5] 保存元路径信息...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_paths_file = os.path.join("result", f"evolved_meta_paths_{timestamp}.json")
    os.makedirs("result", exist_ok=True)

    meta_paths_data = {
        "timestamp": timestamp,
        # 与论文 Eq(7) 一致：R = R_det + mu * R_rea
        "evolution_method": f"R_final = R_det + {GA_CONFIG.get('mu_reasoning', 0.25)} * R_rea",
        "total_population_size": len(evolved_population),
        "num_inference_meta_paths": len(best_meta_paths),
        "max_iterations": genetic_algorithm.max_iterations,
        "best_meta_paths": [
            {
                "rank": i + 1,
                "description": mp.description,
                "formal_string": mp.to_formal_string(),
                "path": mp.path,  # 保存完整路径结构
                "fitness": mp.fitness,
                "fitness_breakdown": {
                    "R_det": "N/A (需要重新评估)",
                    "R_rea": "N/A (需要重新评估)",
                    "R_final": mp.fitness
                }
            }
            for i, mp in enumerate(best_meta_paths)
        ],
        "all_population_fitness": [
            {
                "description": mp.description,
                "formal_string": mp.to_formal_string(),
                "fitness": mp.fitness
            }
            for mp in sorted(evolved_population, key=lambda x: x.fitness, reverse=True)
        ]
    }
    
    with open(meta_paths_file, 'w', encoding='utf-8') as f:
        json.dump(meta_paths_data, f, ensure_ascii=False, indent=2)
    print(f"元路径信息已保存到: {meta_paths_file}")
    
    # 6. 使用v2方式进行最终评估
    print("\n[步骤6] 使用Few-shot Selected Meta-path (V2)方式进行最终评估...")
    detector = FewShotSelectedMetaPathDetector(
        model_config_name="agent_d",
        selected_meta_paths=best_meta_paths,
        icl_pool_size=100,
        top_k_examples=4
    )
    
    # 评估（使用全部测试集或限制数量）
    results = detector.evaluate(
        test_dataset=None,  # 使用全部增强数据集
        limit=test_limit,
        save_results=True,
        output_dir="result"
    )
    
    print("\n" + "="*80)
    print("最终评估结果")
    print("="*80)
    print(f"使用的元路径集合 P*（{len(best_meta_paths)} 条，去重后）:")
    for i, mp in enumerate(best_meta_paths, 1):
        print(f"  {i}. {mp.description}: {mp.to_formal_string()}")
    print(f"\n评估指标:")
    print(f"  总样本数: {results['total_samples']}")
    print(f"  正确预测: {results['correct']}")
    print(f"  准确率: {results['accuracy']:.4f}")
    print(f"  精确率: {results['precision']:.4f}")
    print(f"  召回率: {results['recall']:.4f}")
    print(f"  F1分数: {results['f1_score']:.4f}")
    # 正类=假新闻(label=0)：TP=假->假，FN=假->真，FP=真->假，TN=真->真
    print(f"\n混淆矩阵（正类=假新闻）:")
    print(f"  假新闻->假新闻(TP): {results['true_positives']}")
    print(f"  假新闻->真新闻(FN): {results['false_negatives']}")
    print(f"  真新闻->假新闻(FP): {results['false_positives']}")
    print(f"  真新闻->真新闻(TN): {results['true_negatives']}")
    
    if results.get('results_file'):
        print(f"\n详细结果已保存到: {results['results_file']}")
    if results.get('summary_file'):
        print(f"汇总信息已保存到: {results['summary_file']}")
    print(f"元路径信息已保存到: {meta_paths_file}")
    
    return best_meta_paths, results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="元路径进化与评估")
    parser.add_argument("--dataset_limit", type=int, default=None,
                       help="用于进化的数据集大小限制（前100条作为候选池）")
    parser.add_argument("--max_iterations", type=int, default=None,
                       help="最大迭代次数（默认使用GA_CONFIG中的值）")
    parser.add_argument("--population_size", type=int, default=None,
                       help="种群大小（默认使用GA_CONFIG中的值）")
    parser.add_argument("--test_limit", type=int, default=None,
                       help="最终测试时的样本数量限制")
    parser.add_argument("--no_enhanced", action="store_true",
                       help="不使用增强数据集（使用原始数据）")
    
    args = parser.parse_args()
    
    evolve_and_evaluate(
        dataset_limit=args.dataset_limit,
        max_iterations=args.max_iterations,
        population_size=args.population_size,
        use_enhanced_dataset=not args.no_enhanced,
        test_limit=args.test_limit
    )


if __name__ == "__main__":
    main()

