"""
merge_mr_data.py
~~~~~~~~~~~~~~~~~
把 mr_test_data_full 下每个数据集目录里的所有 MR 文件合并成一个大数据集。

规则（用户确认）：
- 整体按 pair_id 从小到大排序；同一个 pair_id 内，源（is_source=true）在前，后续按 MR 注册顺序排列。
- 同样的 source 只保留一个（按 pair_id 去重）。
- idx 全局唯一连续（0..N-1，按输出顺序编号）。

用法：
  python merge_mr_data.py [src_root] [out_root]
  默认 src_root=mr_test_data_full, out_root=mr_test_data_merged
"""

import json
import pathlib
import sys

# MT_MRS 注册顺序
MR_ORDER = [
    "negation_flip",
    "antonym_substitution",
    "adding_contradiction",
    "pronoun_substitution",
    "voice_switch",
    "synonym_replacement",
    "uninformative",
    "conditional_clause",
]


def merge_dataset(dataset_dir: pathlib.Path) -> list:
    """合并单个数据集目录下的全部 MR 文件，返回按规则排好序并重编号的记录列表。"""
    sources = {}  # pair_id -> 源记录（去重）
    fups = {}     # pair_id -> {mr_name: 后续记录}
    for path in sorted(dataset_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["is_source"]:
                sources.setdefault(rec["pair_id"], rec)  # 同一 pair 只保留一个源
            else:
                fups.setdefault(rec["pair_id"], {})[rec["mr_id"]] = rec

    out = []
    for pid in sorted(sources):
        out.append(sources[pid])  # 源在前
        for mr in MR_ORDER:       # 后续按注册顺序
            if mr in fups.get(pid, {}):
                out.append(fups[pid][mr])
    for i, rec in enumerate(out):  # idx 全局唯一连续
        rec["idx"] = i
    return out


def main(src_root: pathlib.Path, out_root: pathlib.Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for ds_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        recs = merge_dataset(ds_dir)
        out_path = out_root / f"{ds_dir.name}.jsonl"
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
            encoding="utf-8")
        print(f"{ds_dir.name}: {len(recs)} 条 -> {out_path}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "mr_test_data_full"
    out = sys.argv[2] if len(sys.argv) > 2 else "mr_test_data_merged"
    main(pathlib.Path(src), pathlib.Path(out))
