import json
import os


def remove_empty_lines_from_jsonl(file_path):
    """
    读取JSONL文件并删除所有空行
    
    Args:
        file_path (str): JSONL文件路径
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 过滤空行
    non_empty_lines = [line for line in lines if line.strip()]

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(non_empty_lines)

    print(f"已处理文件: {file_path}")
    print(f"原始行数: {len(lines)}, 处理后行数: {len(non_empty_lines)}")


# 使用示例
if __name__ == "__main__":
    # 替换为您的JSONL文件路径
    jsonl_file_path = "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr0.0037_phi-4/unlabeled_data.json"
    remove_empty_lines_from_jsonl(jsonl_file_path)
