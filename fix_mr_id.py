"""
fix_mr_id.py
~~~~~~~~~~~~
批量修正 mr_test_data_full 下所有 JSONL 文件中记录的 mr_id：
仅当 is_source=false（后续用例）时，把 mr_id 设为它所在文件名（MR 名）；
源记录（is_source=true）保持不变。

用法：
  python fix_mr_id.py [root_dir]     # 默认 root_dir = mr_test_data_full
"""

import json
import pathlib
import sys


def fix_dir(root: pathlib.Path) -> None:
    total = changed = 0
    for path in sorted(root.rglob("*.jsonl")):
        mr_name = path.stem  # 文件名即 MR 名
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("is_source", True) and rec.get("mr_id") != mr_name:
                rec["mr_id"] = mr_name  # 原地改键，保留键序
                changed += 1
            lines.append(json.dumps(rec, ensure_ascii=False))
            total += 1
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"共 {total} 条记录，修改 {changed} 条")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "mr_test_data_full"
    fix_dir(pathlib.Path(root))
