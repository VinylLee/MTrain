import json
from collections import defaultdict
import os
from tqdm import tqdm

# 示例数据：
# {"premise":"Two men in colorful uniforms bike past a crowd of people.","hypothesis":"Two people are wearing uniforms.","label":0,"pair_id":2032,"is_syn":false,"weight":1.0}
# {"premise":"Two men in colorful uniforms bike past a crowd of people.","hypothesis":"Two individuals are donning outfits.","label":0,"pair_id":2032,"is_syn":true,"weight":1.0}
# {"premise":"Two men in colorful uniforms bike past a crowd of people.","hypothesis":"No two people are wearing uniforms.","label":2,"pair_id":2032,"is_syn":true,"weight":1.0}
# {"premise":"Two men in colorful uniforms bike past a crowd of people.","hypothesis":"Uniforms are being worn by two people.","label":0,"pair_id":2032,"is_syn":true,"weight":1.0}
# {"premise":"Two men in colorful uniforms bike past a crowd of people.","hypothesis":"Two people are not wearing uniforms.","label":2,"pair_id":2032,"is_syn":true,"weight":1.0}
# {"premise":"Two men in colorful uniforms bike past a crowd of people.","hypothesis":"Two men biked.","label":0,"pair_id":2032,"is_syn":true,"weight":1.0}
# {"premise":"Two men in colorful uniforms bike past a crowd of people.","hypothesis":"Two people in colorful uniforms bike past a crowd of individuals.","label":0,"pair_id":2032,"is_syn":true,"weight":1.0}
# {"premise":"Two men in colorful uniforms bike past a crowd of people.","hypothesis":"Two of them are wearing uniforms.","label":0,"pair_id":2032,"is_syn":true,"weight":1.0}
# {"premise":"Two men in colorful uniforms bike past a crowd of people.","hypothesis":"2 persons are there","label":0,"pair_id":2032,"is_syn":true,"weight":1.0}
# {"premise":"Two men in colorful uniforms bike past a crowd of people.","hypothesis":"They are not wearing uniforms.","label":2,"pair_id":2032,"is_syn":true,"weight":1.0}

# 使用 sentence-transformers 库来计算语义相似度
try:
    from sentence_transformers import SentenceTransformer, util
    import torch

    # 初始化模型（全局变量，避免重复加载）
    _similarity_model = None

    def get_similarity_model():
        global _similarity_model
        if _similarity_model is None:
            print("正在加载语义相似度模型...")
            # try:
            #     # 尝试使用多语言模型，支持中英文
            #     _similarity_model = SentenceTransformer(
            #         'paraphrase-multilingual-MiniLM-L12-v2')
            #     print("模型加载完成!")
            # except Exception as e:
            #     print(f"加载多语言模型失败: {e}")
            #     print("尝试加载英文模型...")
            try:
                _similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("英文模型加载完成!")
            except Exception as e2:
                print(f"加载英文模型也失败: {e2}")
                raise e2
        return _similarity_model

    def llm_are_semantically_similar(text1, text2, threshold=0.9):
        """
        使用sentence-transformers计算语义相似度
        """
        try:
            model = get_similarity_model()

            # 编码文本为向量
            embeddings = model.encode([text1, text2], convert_to_tensor=True)

            # 计算余弦相似度
            cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1])
            similarity_score = cosine_scores.item()

            return similarity_score > threshold
        except Exception as e:
            print(f"语义相似度计算失败: {e}")
            print("降级使用词重叠方法...")
            return word_overlap_similarity(text1, text2,
                                           threshold * 0.9)  # 调整阈值

    SENTENCE_TRANSFORMERS_AVAILABLE = True

except ImportError as e:
    print(f"警告: sentence-transformers未安装或有问题 ({e})，使用简化的词重叠方法")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def word_overlap_similarity(text1, text2, threshold=0.85):
    """
    简化的基于词重叠的相似度计算（备用方案）
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if len(words1.union(words2)) == 0:
        return True

    overlap = len(words1.intersection(words2)) / len(words1.union(words2))

    return overlap > threshold


if not SENTENCE_TRANSFORMERS_AVAILABLE:

    def llm_are_semantically_similar(text1, text2, threshold=0.85):
        """
        使用词重叠方法
        """
        return word_overlap_similarity(text1, text2, threshold)


def deduplicate_nli_data(input_file_path,
                         output_file_path,
                         similarity_threshold=0.85):
    """
    通过移除语义相似的假设来对NLI数据进行去重。
    
    Args:
        input_file_path: 输入的JSONL文件路径
        output_file_path: 输出的去重后JSONL文件路径
        similarity_threshold: 语义相似度阈值，超过此值认为是冗余
    """
    print(f"开始处理文件: {input_file_path}")
    print(f"相似度阈值: {similarity_threshold}")

    # 明确显示使用的去重方法
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("🚀 使用方法: 语义嵌入模型 (sentence-transformers)")
    else:
        print("⚡ 使用方法: 词重叠算法 (基于词汇交集)")

    all_items = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                all_items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析错误: {e}")
                continue

    print(f"成功读取 {len(all_items)} 条数据")

    # 按 pair_id 分组
    items_by_pair_id = defaultdict(list)
    for item in all_items:
        items_by_pair_id[item['pair_id']].append(item)

    print(f"共有 {len(items_by_pair_id)} 个不同的pair_id")

    deduplicated_items = []
    total_removed = 0

    # 使用进度条处理每个pair_id
    for pair_id, items_in_group in tqdm(items_by_pair_id.items(), desc="去重处理"):
        if len(items_in_group) <= 1:
            # 如果组内只有一个或没有数据，直接保留
            deduplicated_items.extend(items_in_group)
            continue

        # 标记哪些项目应该被移除
        items_to_remove = set()

        # 比较组内每两个数据的hypothesis相似性
        for i in range(len(items_in_group)):
            if i in items_to_remove:
                continue

            for j in range(i + 1, len(items_in_group)):
                if j in items_to_remove:
                    continue

                item_i = items_in_group[i]
                item_j = items_in_group[j]

                # 比较两个hypothesis的相似性
                if llm_are_semantically_similar(item_i['hypothesis'],
                                                item_j['hypothesis'],
                                                similarity_threshold):
                    # 如果hypothesis相似，则比较它们与premise的相似性
                    premise = item_i['premise']  # 同一pair_id的premise应该相同

                    # 计算hypothesis与premise的相似性
                    sim_i_premise = llm_are_semantically_similar(
                        item_i['hypothesis'], premise, 0.5)  # 使用较低阈值判断相似性
                    sim_j_premise = llm_are_semantically_similar(
                        item_j['hypothesis'], premise, 0.5)

                    # 保留与premise不太相似的那个（更有价值的数据）
                    if sim_i_premise and not sim_j_premise:
                        # item_i与premise相似，保留item_j
                        items_to_remove.add(i)
                        total_removed += 1
                    elif sim_j_premise and not sim_i_premise:
                        # item_j与premise相似，保留item_i
                        items_to_remove.add(j)
                        total_removed += 1
                    elif sim_i_premise and sim_j_premise:
                        # 两个都与premise相似，保留第一个
                        items_to_remove.add(j)
                        total_removed += 1
                    elif not sim_i_premise and not sim_j_premise:
                        # 两个都与premise不相似，保留第一个
                        items_to_remove.add(j)
                        total_removed += 1

        # 保留未被标记移除的项目
        for i, item in enumerate(items_in_group):
            if i not in items_to_remove:
                deduplicated_items.append(item)

    # 保存结果
    print("正在保存去重后的数据...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in tqdm(deduplicated_items, desc="保存数据"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n======== 去重完成 ========")
    print(f"原始数据条数: {len(all_items)}")
    print(f"去重后数据条数: {len(deduplicated_items)}")
    print(f"移除条数: {total_removed}")
    print(f"压缩率: {total_removed/len(all_items)*100:.1f}%")
    print(f"去重后的数据已保存到: {output_file_path}")


def main():
    """主函数"""
    # 默认路径设置
    default_input = '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlim_lr0.0051_phi-4/augmented_data_三合一.json'
    default_output = '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlim_lr0.0051_phi-4/augmented_data_三合一_0.8.json'

    # 检查输入文件是否存在
    if os.path.exists(default_input):
        input_path = default_input
        output_path = default_output
        print(f"使用默认输入文件: {input_path}")
    else:
        print(f"默认文件不存在: {default_input}")
        # 寻找其他可能的输入文件
        possible_files = []
        base_dir = '/home/ubuntu/MTrain/MTrainNLI/nli_aug_data'
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(
                            '.json'
                    ) and 'unlabeled_data' in file and 'deduplicated' not in file:
                        possible_files.append(os.path.join(root, file))

        if possible_files:
            print("找到以下可能的输入文件:")
            for i, file in enumerate(possible_files):
                print(f"  {i+1}. {file}")

            choice = input(
                f"请选择文件 (1-{len(possible_files)}) 或输入自定义路径: ").strip()

            if choice.isdigit() and 1 <= int(choice) <= len(possible_files):
                input_path = possible_files[int(choice) - 1]
                output_path = input_path.replace('.json', '_deduplicated.json')
            else:
                input_path = choice
                output_path = input_path.replace('.json', '_deduplicated.json')
        else:
            input_path = input("请输入输入文件路径: ").strip()
            output_path = input_path.replace('.json', '_deduplicated.json')

    # 确认输出路径
    print(f"输出文件将保存到: {output_path}")

    # 设置相似度阈值
    default_threshold = 0.85 if not SENTENCE_TRANSFORMERS_AVAILABLE else 0.9
    threshold_input = input(
        f"请输入相似度阈值 (0.0-1.0, 默认{default_threshold}): ").strip()
    try:
        threshold = float(
            threshold_input) if threshold_input else default_threshold
        threshold = max(0.0, min(1.0, threshold))  # 限制在0-1之间
    except ValueError:
        threshold = default_threshold
        print(f"无效输入，使用默认阈值 {default_threshold}")

    # 执行去重
    try:
        deduplicate_nli_data(input_path, output_path, threshold)
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
