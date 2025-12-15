#!/usr/bin/env python3
"""
删除JSON文件中 "mr_id":"none" 的行 - 自动执行版本
"""

import json
import os
from pathlib import Path
import shutil


def remove_none_mr_id(input_file, output_file=None):
    """
    删除JSON文件中包含 "mr_id":"none" 的行
    
    Args:
        input_file (str): 输入文件路径
        output_file (str, optional): 输出文件路径，如果为None则覆盖原文件
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"错误：文件 {input_file} 不存在")
        return

    # 如果没有指定输出文件，创建备份并覆盖原文件
    if output_file is None:
        backup_file = input_path.with_suffix(input_path.suffix + '.backup')
        print(f"创建备份文件: {backup_file}")
        shutil.copy2(input_file, backup_file)
        output_file = input_file

    filtered_lines = []
    removed_count = 0
    total_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                total_count += 1

                try:
                    # 解析JSON
                    data = json.loads(line)

                    # 检查是否包含 "mr_id":"none"
                    if data.get("mr_id") == "none":
                        removed_count += 1
                        if removed_count <= 10:  # 只显示前10个删除的行
                            print(
                                f"删除第 {line_num} 行: mr_id=none, pair_id={data.get('pair_id', 'unknown')}"
                            )
                        elif removed_count == 11:
                            print("... (更多行被删除)")
                    else:
                        filtered_lines.append(line)

                except json.JSONDecodeError as e:
                    print(f"警告：第 {line_num} 行JSON格式错误: {e}")
                    print(f"内容: {line[:100]}...")
                    # 保留格式错误的行
                    filtered_lines.append(line)

        # 写入过滤后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in filtered_lines:
                f.write(line + '\n')

        print(f"\n处理完成:")
        print(f"  总行数: {total_count}")
        print(f"  删除行数: {removed_count}")
        print(f"  保留行数: {len(filtered_lines)}")
        print(f"  输出文件: {output_file}")

        return True

    except Exception as e:
        print(f"错误：处理文件时出现异常: {e}")
        return False


if __name__ == "__main__":
    # 处理指定文件
    input_file = "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/snli_lr4e-06_llama-3.2-3b-instruct/unlabeled_data.json"

    print(f"开始处理文件: {input_file}")
    print("删除所有包含 'mr_id':'none' 的行...")

    success = remove_none_mr_id(input_file)

    if success:
        print("✅ 处理成功完成！")
    else:
        print("❌ 处理失败！")
