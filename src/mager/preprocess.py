"""
数据预处理脚本 - 对数据集进行增强（节点和边细化）
先对数据集进行增强，保存增强后的数据集，避免在构建图时重复调用LLM
"""
import json
import os
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，使用简单的进度显示
    def tqdm(iterable, desc="", leave=True, total=None):
        return iterable

from data_loader import DataLoader
from enhancement_agent import EnhancementAgent
from graph_builder import GraphBuilder
from config import ENHANCEMENT_CONFIG, GA_CONFIG, DATA_PATHS, get_model_config

def process_single_news_item(item: Dict, graph_builder: GraphBuilder, 
                              enhancement_agent: EnhancementAgent, 
                              index: int, total_items: int) -> Tuple[Dict, bool, str]:
    """
    处理单条新闻的增强
    
    Args:
        item: 新闻数据项
        graph_builder: 图构建器
        enhancement_agent: 增强智能体
        index: 当前索引（用于显示）
        total_items: 总数量（用于显示）
        
    Returns:
        (enhanced_item, success, message)
        enhanced_item: 增强后的数据项，如果失败则为None
        success: 是否成功处理
        message: 处理信息（成功/跳过/错误）
    """
    try:
        # 获取新闻内容
        article = item.get("article", {})
        news_text = article.get("article_content", "")
        news_id = f"news_{hash(news_text) % 1000000}"
        
        # 构建原始图
        original_graph = graph_builder.build_from_data(item, news_id)
        
        # 统计需要处理的节点和边
        user_nodes = [node for node in original_graph.nodes.values() if node.node_type == "User"]
        comment_reply_edges = [edge for edge in original_graph.edges 
                              if edge.edge_type in ["Comment", "Reply"]]
        
        # 跳过用户节点数超过1万的新闻（处理时间过长）
        if len(user_nodes) >= 10000:
            return None, False, f"[{index+1}/{total_items}] {news_id}: 跳过（用户节点数过多 {len(user_nodes)} >= 10000）"
        
        # 创建增强数据项（复制原始数据）
        enhanced_item = item.copy()
        enhanced_item['enhanced_nodes'] = {}
        enhanced_item['enhanced_edges'] = []
        
        # 1. 节点细化：批量处理用户节点分类
        if user_nodes:
            # 准备批量数据：收集用户的评论内容
            user_list = []
            for node in user_nodes:
                user_id = node.node_id
                user_attributes = node.attributes.copy()
                
                # 收集该用户的所有评论内容
                user_comments = []
                for edge in original_graph.edges:
                    if edge.source == user_id and edge.attributes.get("text"):
                        comment_text = edge.attributes.get("text", "")
                        if comment_text:
                            user_comments.append(comment_text[:100])  # 每条评论限制100字符
                
                # 将评论内容添加到用户属性中
                if user_comments:
                    user_attributes["comments"] = user_comments[:5]  # 最多保留5条评论
                    user_attributes["comment_count"] = len(user_comments)
                
                user_list.append((user_id, user_attributes))
            
            # 批量分类
            batch_results = enhancement_agent._classify_user_types_batch(
                user_list, 
                batch_size=ENHANCEMENT_CONFIG["user_batch_size"],
                show_progress=False  # 并行时关闭进度条，避免混乱
            )
            
            enhanced_item['enhanced_nodes'] = batch_results
        else:
            enhanced_item['enhanced_nodes'] = {}
        
        # 2. 边细化：批量处理边分析
        if comment_reply_edges:
            # 分离有文本和无文本的边
            edges_with_text = []
            edges_without_text = []
            
            for edge in comment_reply_edges:
                text = edge.attributes.get("text", "")
                if text:
                    edges_with_text.append(edge)
                else:
                    edges_without_text.append(edge)
            
            # 批量处理有文本的边
            if edges_with_text:
                text_list = [edge.attributes.get("text", "")[:80] for edge in edges_with_text]
                # 批量分析
                batch_results = enhancement_agent._analyze_interactions_batch(
                    text_list, 
                    batch_size=ENHANCEMENT_CONFIG["edge_batch_size"],
                    show_progress=False  # 并行时关闭进度条
                )
                
                # 保存有文本边的结果
                for edge, (stance, sentiment) in zip(edges_with_text, batch_results):
                    enhanced_item['enhanced_edges'].append({
                        "source": edge.source,
                        "target": edge.target,
                        "edge_type": edge.edge_type,
                        "cognitive_stance": stance,
                        "affective_sentiment": sentiment,
                        "text": edge.attributes.get("text", "")
                    })
            
            # 无文本的边使用默认值
            for edge in edges_without_text:
                enhanced_item['enhanced_edges'].append({
                    "source": edge.source,
                    "target": edge.target,
                    "edge_type": edge.edge_type,
                    "cognitive_stance": "Discuss",
                    "affective_sentiment": "Neutral",
                    "text": ""
                })
        else:
            enhanced_item['enhanced_edges'] = []
        
        return enhanced_item, True, f"[{index+1}/{total_items}] {news_id}: 完成（用户{len(user_nodes)}个，边{len(comment_reply_edges)}条）"
    
    except Exception as e:
        import traceback
        error_msg = f"[{index+1}/{total_items}] 处理出错: {str(e)}"
        return None, False, error_msg

def get_processed_count() -> int:
    """检查已处理的数据数量（通过临时文件）"""
    base_dir = os.path.dirname(get_enhanced_data_path())
    dataset_name = _get_dataset_name()
    model_name = _get_model_name()
    temp_prefix = f"enhanced_dataset_{dataset_name}_{model_name}_temp_"
    
    max_count = 0
    
    if not os.path.isdir(base_dir):
        return 0
    
    # 查找所有临时文件（匹配当前数据集和模型）
    for filename in os.listdir(base_dir):
        if filename.startswith(temp_prefix) and filename.endswith(".jsonl"):
            try:
                count_str = filename.replace(temp_prefix, "").replace(".jsonl", "")
                count = int(count_str)
                max_count = max(max_count, count)
            except ValueError:
                continue
    
    return max_count

def enhance_dataset(limit: int = 100):
    """
    对数据集进行增强处理（支持断点续传）
    
    Args:
        limit: 每种类型新闻的处理数量（真假各limit条）
    """
    print("=" * 60)
    print("开始数据预处理 - 增强数据集")
    print("=" * 60)
    
    # 初始化组件
    data_loader = DataLoader()
    enhancement_agent = EnhancementAgent()
    graph_builder = GraphBuilder()
    
    # 检查是否有已处理的数据
    processed_count = get_processed_count()
    if processed_count > 0:
        print(f"\n发现已处理数据: {processed_count} 条")
        print(f"将从第 {processed_count + 1} 条继续处理...")
        # 加载已处理的数据
        dataset_name = _get_dataset_name()
        model_name = _get_model_name()
        base_dir = os.path.dirname(get_enhanced_data_path())
        temp_file = os.path.join(base_dir, f"enhanced_dataset_{dataset_name}_{model_name}_temp_{processed_count}.jsonl")
        if os.path.exists(temp_file):
            enhanced_data = []
            with open(temp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            enhanced_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            print(f"已加载 {len(enhanced_data)} 条已处理数据")
        else:
            enhanced_data = []
            processed_count = 0
    else:
        enhanced_data = []
    
    # 加载数据（真假各limit条）
    print(f"\n加载数据（真假各{limit}条）...")
    misinformation = data_loader.load_misinformation(limit=limit)
    verified = data_loader.load_verified_information(limit=limit)
    
    print(f"假新闻: {len(misinformation)} 条")
    print(f"真新闻: {len(verified)} 条")
    
    # 合并数据并添加标签
    labeled_data = []
    for item in misinformation:
        labeled_item = item.copy()
        labeled_item['label'] = 0  # 假新闻
        labeled_data.append(labeled_item)
    
    for item in verified:
        labeled_item = item.copy()
        labeled_item['label'] = 1  # 真新闻
        labeled_data.append(labeled_item)
    
    print(f"总计: {len(labeled_data)} 条数据")
    
    # 跳过已处理的数据
    if processed_count > 0:
        labeled_data = labeled_data[processed_count:]
        print(f"跳过已处理的 {processed_count} 条，剩余 {len(labeled_data)} 条待处理")
    
    total_items = len(labeled_data) + processed_count
    
    if len(labeled_data) == 0:
        print("\n所有数据已处理完成！")
        # 保存最终结果
        save_enhanced_data(enhanced_data, total_items, is_final=True)
        return
    
    print(f"\n开始增强处理（从第 {processed_count + 1} 条开始，共需处理 {total_items} 条）...")
    
    # 获取并行线程数
    max_workers = ENHANCEMENT_CONFIG.get("max_workers", 10)
    print(f"使用 {max_workers} 个线程并行处理...")
    
    # 准备带索引的数据项
    indexed_items = [(processed_count + idx, item) for idx, item in enumerate(labeled_data)]
    
    # 线程安全的保存锁和结果列表
    save_lock = Lock()
    skipped_count = 0
    processed_results = {}  # index -> enhanced_item
    
    # 并行处理函数
    def process_with_index(indexed_item):
        idx, item = indexed_item
        enhanced_item, success, message = process_single_news_item(
            item, graph_builder, enhancement_agent, idx, total_items
        )
        return idx, enhanced_item, success, message
    
    # 并行处理
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_with_index, item): item[0] for item in indexed_items}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            try:
                idx, enhanced_item, success, message = future.result()
                
                if success and enhanced_item:
                    processed_results[idx] = enhanced_item
                else:
                    skipped_count += 1
                
                # 输出处理信息（简化输出，避免太多打印）
                if not success and "跳过" not in message:
                    print(f"\n{message}")
                
                completed_count += 1
                
                # 每完成4条数据保存一次（防止中途中断丢失数据）
                if completed_count % 4 == 0:
                    with save_lock:
                        # 按索引排序并合并结果
                        sorted_indices = sorted([k for k in processed_results.keys()])
                        sorted_results = [processed_results[k] for k in sorted_indices]
                        
                        # 合并之前已处理的数据
                        all_data = enhanced_data + sorted_results
                        current_total = processed_count + completed_count
                        
                        save_enhanced_data(all_data, current_total)
                        print(f"\n✓ 已保存中间结果（共 {current_total} 条）")
            
            except Exception as e:
                idx = futures.get(future, "unknown")
                print(f"\n处理索引 {idx} 时发生异常: {e}")
                skipped_count += 1
                import traceback
                traceback.print_exc()
    
    # 合并所有处理结果（按索引排序）
    sorted_indices = sorted(processed_results.keys())
    all_enhanced_data = enhanced_data + [processed_results[idx] for idx in sorted_indices]
    
    # 保存最终结果
    save_enhanced_data(all_enhanced_data, total_items, is_final=True)
    
    # 清理临时文件
    base_dir = os.path.dirname(get_enhanced_data_path())
    dataset_name = _get_dataset_name()
    model_name = _get_model_name()
    temp_prefix = f"enhanced_dataset_{dataset_name}_{model_name}_temp_"
    
    for filename in os.listdir(base_dir):
        if filename.startswith(temp_prefix) and filename.endswith(".jsonl"):
            try:
                os.remove(os.path.join(base_dir, filename))
            except:
                pass
    
    print(f"\n数据预处理完成！")
    print(f"成功处理: {len(processed_results)}/{total_items - processed_count} 条新数据")
    if processed_count > 0:
        print(f"包含之前已处理的: {processed_count} 条")
    print(f"总计: {len(all_enhanced_data)} 条数据")
    if skipped_count > 0:
        print(f"跳过处理: {skipped_count} 条（用户节点数 >= 10000）")
    print(f"增强后的数据已保存到: {get_enhanced_data_path()}")

def save_enhanced_data(enhanced_data: List[Dict], count: int, is_final: bool = False):
    """保存增强后的数据"""
    output_path = get_enhanced_data_path()
    
    # 如果是中间保存，使用临时文件名
    if not is_final:
        output_path = output_path.replace('.jsonl', f'_temp_{count}.jsonl')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为JSONL格式
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in enhanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    if is_final:
        print(f"\n最终数据已保存到: {output_path}")

def _get_dataset_name() -> str:
    """从数据路径中提取数据集名称"""
    misinformation_path = DATA_PATHS.get("misinformation", "").lower()
    
    if "mcfend" in misinformation_path:
        return "mcfend"
    elif "gossipcop" in misinformation_path:
        return "gossipcop"
    elif "politifact" in misinformation_path:
        return "politifact"
    else:
        return "unknown"

def _get_model_name() -> str:
    """获取增强智能体使用的模型名称"""
    model_config = get_model_config("agent_a")
    model_name = model_config.model_name
    model_name_clean = model_name.replace("/", "-").replace(" ", "-").replace(".", "-")
    return model_name_clean

def get_enhanced_data_path() -> str:
    """获取增强数据的保存路径（包含数据集名称和模型名称）"""
    configured_path = DATA_PATHS.get("enhanced")
    if configured_path:
        return configured_path
    dataset_name = _get_dataset_name()
    model_name = _get_model_name()
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", dataset_name)
    filename = f"enhanced_dataset_{dataset_name}_{model_name}.jsonl"
    return os.path.join(base_dir, filename)

# 注意：load_enhanced_dataset 函数已移至 data_loader.py 中
# 请使用 data_loader.load_enhanced_dataset() 来加载增强数据集

if __name__ == "__main__":
    # 处理前100条真假新闻
    enhance_dataset(limit=400)

