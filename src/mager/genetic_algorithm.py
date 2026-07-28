"""
遗传算法模块 - 元路径进化
"""
from typing import List, Dict
from meta_path import MetaPath
from generation_agent import GenerationAgent
from fitness_evaluator import FitnessEvaluator
from graph_builder import PropagationGraph
from config import GA_CONFIG

class GeneticAlgorithm:
    """遗传算法 - 用于元路径进化"""
    
    def __init__(self):
        self.generation_agent = GenerationAgent()
        self.fitness_evaluator = FitnessEvaluator()
        self.population_size = GA_CONFIG["population_size"]
        self.max_iterations = GA_CONFIG["max_iterations"]
        self.tournament_size = GA_CONFIG["tournament_size"]
    
    def evolve(self, initial_population: List[MetaPath],
              labeled_dataset: List[Dict],
              enhanced_graphs: Dict[str, PropagationGraph]) -> List[MetaPath]:
        """
        执行遗传进化过程（算法1）
        
        Args:
            initial_population: 初始元路径种群
            labeled_dataset: 带标签的数据集
            enhanced_graphs: 增强后的图字典
            
        Returns:
            进化后的元路径种群
        """
        population = initial_population.copy()
        
        # 初始化：计算初始种群的适应度
        print("\n" + "="*60)
        print("初始化阶段：计算初始种群适应度")
        print("="*60)
        for idx, meta_path in enumerate(population, 1):
            if meta_path.fitness == 0.0:
                print(f"[{idx}/{len(population)}] 评估: {meta_path.to_formal_string()}")
                fitness = self.fitness_evaluator.evaluate_fitness(
                    meta_path, labeled_dataset, enhanced_graphs
                )
                meta_path.fitness = fitness
                print(f"  适应度: {fitness:.4f}")
        
        # 排序并显示初始种群
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        print(f"\n初始种群适应度范围: {min(mp.fitness for mp in population):.4f} - {max(mp.fitness for mp in population):.4f}")
        print("\n初始种群Top-5:")
        for i, mp in enumerate(sorted_pop[:5], 1):
            print(f"  {i}. {mp.to_formal_string()} (适应度: {mp.fitness:.4f})")
        
        # 遗传优化阶段
        print("\n" + "="*60)
        print("遗传优化阶段")
        print("="*60)
        for iteration in range(self.max_iterations):
            print(f"\n{'='*60}")
            print(f"迭代 {iteration + 1}/{self.max_iterations}")
            print(f"{'='*60}")
            
            # 选择父代
            print("\n[1/4] 选择父代（锦标赛选择）...")
            parents = self.generation_agent.select_parents(population, self.tournament_size)
            print(f"  变异父代: {parents[0].to_formal_string()} (适应度: {parents[0].fitness:.4f})")
            print(f"  交叉父代1: {parents[1].to_formal_string()} (适应度: {parents[1].fitness:.4f})")
            print(f"  交叉父代2: {parents[2].to_formal_string()} (适应度: {parents[2].fitness:.4f})")
            
            # 生成新候选
            print("\n[2/4] 生成新候选...")
            new_candidate = self.generation_agent.generate_candidate(parents)
            print(f"  操作类型: {new_candidate.description}")
            print(f"  新候选: {new_candidate.to_formal_string()}")
            
            # 评估适应度
            print("\n[3/4] 评估新候选适应度...")
            new_fitness = self.fitness_evaluator.evaluate_fitness(
                new_candidate, labeled_dataset, enhanced_graphs
            )
            new_candidate.fitness = new_fitness
            print(f"  新候选适应度: {new_fitness:.4f}")
            
            # 稳态替换
            print("\n[4/4] 稳态替换...")
            min_fitness_meta_path = min(population, key=lambda x: x.fitness)
            print(f"  当前最差元路径: {min_fitness_meta_path.to_formal_string()} (适应度: {min_fitness_meta_path.fitness:.4f})")
            
            if new_fitness > min_fitness_meta_path.fitness:
                # 替换适应度最低的元路径
                idx = population.index(min_fitness_meta_path)
                population[idx] = new_candidate
                print(f"  ✓ 替换成功！")
                print(f"    被替换: {min_fitness_meta_path.to_formal_string()} (适应度: {min_fitness_meta_path.fitness:.4f})")
                print(f"    新元路径: {new_candidate.to_formal_string()} (适应度: {new_fitness:.4f})")
                print(f"    提升: {new_fitness - min_fitness_meta_path.fitness:.4f}")
            else:
                print(f"  ✗ 新候选未优于最差个体，不替换 (差值: {new_fitness - min_fitness_meta_path.fitness:.4f})")
            
            # 输出当前种群统计
            fitnesses = [mp.fitness for mp in population]
            sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
            print(f"\n当前种群统计:")
            print(f"  平均适应度: {sum(fitnesses)/len(fitnesses):.4f}")
            print(f"  最大适应度: {max(fitnesses):.4f}")
            print(f"  最小适应度: {min(fitnesses):.4f}")
            print(f"  Top-3元路径:")
            for i, mp in enumerate(sorted_pop[:3], 1):
                print(f"    {i}. {mp.to_formal_string()} (适应度: {mp.fitness:.4f})")
        
        # 最终结果
        print("\n" + "="*60)
        print("进化完成！最终种群统计")
        print("="*60)
        sorted_final = sorted(population, key=lambda x: x.fitness, reverse=True)
        fitnesses = [mp.fitness for mp in population]
        print(f"最终适应度范围: {min(fitnesses):.4f} - {max(fitnesses):.4f}")
        print(f"平均适应度: {sum(fitnesses)/len(fitnesses):.4f}")
        print(f"\n最终种群Top-5:")
        for i, mp in enumerate(sorted_final[:5], 1):
            print(f"  {i}. {mp.to_formal_string()}")
            print(f"     适应度: {mp.fitness:.4f}, 描述: {mp.description}")
        
        return population
    
    def get_best_meta_paths(self, population: List[MetaPath], top_k: int = 5) -> List[MetaPath]:
        """获取适应度最高的top-k元路径"""
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_population[:top_k]

