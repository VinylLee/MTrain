import json
import os


def load_jsonl(file_path):
    """加载JSONL格式文件"""
    data = []
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """保存数据为JSONL格式"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def merge_datasets(unlabeled_file, augmented_file, output_file):
    """
    融合两个数据集文件
    关键处理：将augmented_file的pair_id接在unlabeled_file的pair_id后面
    """
    # 加载两个数据集
    print("正在加载数据集...")
    unlabeled_data = load_jsonl(unlabeled_file)
    augmented_data = load_jsonl(augmented_file)

    print(f"未标注数据集条数: {len(unlabeled_data)}")
    print(f"增强数据集条数: {len(augmented_data)}")

    # 找到第一个文件中的最大pair_id
    max_pair_id = -1
    if unlabeled_data:
        max_pair_id = max(item.get('pair_id', -1) for item in unlabeled_data)

    print(f"未标注数据集中最大pair_id: {max_pair_id}")

    # 调整第二个文件的pair_id
    adjusted_augmented_data = []
    for item in augmented_data:
        new_item = item.copy()
        # 将pair_id偏移到第一个文件的最大值之后
        new_item['pair_id'] = item['pair_id'] + max_pair_id + 1
        adjusted_augmented_data.append(new_item)

    # 合并数据集
    merged_data = unlabeled_data + adjusted_augmented_data

    print(f"融合后总条数: {len(merged_data)}")
    print(
        f"新的pair_id范围: {min(item['pair_id'] for item in merged_data)} - {max(item['pair_id'] for item in merged_data)}"
    )

    # 保存融合后的数据
    save_jsonl(merged_data, output_file)
    print(f"融合后的数据已保存到: {output_file}")

    return merged_data


def main():
    # 定义文件路径
    base_dir = "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/mnlim_lr0.0051_phi-4"
    unlabeled_file = os.path.join(base_dir, "unlabeled_data.json")
    augmented_file = os.path.join(base_dir, "augmented_data.json")
    output_file = os.path.join(base_dir, "augmented_data_三合一.json")

    # 执行融合
    try:
        merged_data = merge_datasets(unlabeled_file, augmented_file,
                                     output_file)

        # 验证融合结果
        print("\n=== 融合结果验证 ===")
        pair_ids = [item['pair_id'] for item in merged_data]
        print(
            f"pair_id是否连续: {list(range(min(pair_ids), max(pair_ids) + 1)) == sorted(set(pair_ids))}"
        )
        print(f"是否存在重复pair_id: {len(pair_ids) != len(set(pair_ids))}")

        # 显示一些示例
        print("\n=== 前5条示例 ===")
        for i, item in enumerate(merged_data[:5]):
            print(
                f"条目 {i+1}: pair_id={item['pair_id']}, premise='{item['premise'][:50]}...'"
            )

        print("\n=== 后5条示例 ===")
        for i, item in enumerate(merged_data[-5:]):
            print(
                f"条目 {len(merged_data)-4+i}: pair_id={item['pair_id']}, premise='{item['premise'][:50]}...'"
            )

    except Exception as e:
        print(f"融合过程中出现错误: {e}")


if __name__ == "__main__":
    main()

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
