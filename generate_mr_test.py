"""
generate_mr_test.py
~~~~~~~~~~~~~~~~~~~
在原始测试集上生成 MR（蜕变关系）后续测试用例，并按 MR 类型保存为自包含的 JSONL 文件。

每个 MR 文件（{dataset}/{mr_name}.jsonl）同时包含：
  - 源用例记录（is_source=true，来自测试集的原始 premise/hypothesis）
  - 该 MR 生成的后续用例记录（is_source=false）

每条记录包含 8 个字段：
  idx        该输出文件内的递增序号（源记录与后续记录各自独立编号 0..S-1 / 0..F-1）
  premise    前提
  hypothesis 假设
  mr_id      MR 类型（如 negation_flip）；源记录为 "none"
  pair_id    用例对编号（mnlim/mnlimm 用原始 idx，sick/snli 用运行索引）
  label      金标（int：0=entailment, 1=neutral, 2=contradiction）
  mr_type    MR 类别（inv/flip/neutral）；源记录为 "inv"
  is_source  是否源用例

用法：
  python generate_mr_test.py                  # 真实 LLM 全量生成
  python generate_mr_test.py --dry-run        # 假 LLM，快速验证 I/O 与记录结构
  python generate_mr_test.py --datasets=snli  # 只处理指定数据集
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import Any, Dict, List

from mylogging import logger
from MRAugment import NLIMRTool
from LMstudio import OpenAIChat, HOST, PORT

# 标签映射（与 LabeledEnhance.py 一致）：0=entailment, 1=neutral, 2=contradiction
INT_TO_STR = {0: "entailment", 1: "neutral", 2: "contradiction"}
STR_TO_INT = {v: k for k, v in INT_TO_STR.items()}

CONFIG = {
    "original_root": "original_dataset",   # 测试集根目录（相对脚本所在目录）
    "output_root": "mr_test_data",         # 输出根目录（相对脚本所在目录）
    "datasets": ["mnlim", "mnlimm", "sick", "snli"],
    "mr_names": None,                      # None -> 使用全部 NLIMRTool.MT_MRS
    "model_name": "gemma-3-4b-it-qat",     # LLM 模型名（LM Studio）
    "dry_run": False,                      # True -> 使用 _FakeLLM，不连接 LLM 服务
    "dry_run_limit": 10,                   # dry_run 模式下每个数据集最多处理的样本数
    "sample_limit": None,                  # None=全部；非 dry_run 模式每数据集最多处理的样本数（小规模测试用）
    "batch_size": 0,                       # 0=单样本逐条；>0=批量路径（如 8/16，一次 LLM 请求处理多条）
    "resume": False,                       # True=跳过已写源样本、续接后续 idx（追加模式）
}

REQUIRED_FIELDS = ("idx", "premise", "hypothesis", "mr_id", "pair_id",
                   "label", "mr_type", "is_source")


def _script_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def load_sources(test_path: pathlib.Path) -> List[Dict[str, Any]]:
    """流式读取 JSONL 测试集，过滤无金标（label=-1）/坏行/缺字段，返回带 pair_id 的源样本。"""
    sources: List[Dict[str, Any]] = []
    if not test_path.exists():
        logger.error(f"测试集文件不存在: {test_path}")
        return sources
    with open(test_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(f"跳过无法解析的行: {exc}")
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("label") == -1:
                continue  # 无金标，过滤（与 NLIdataset.clean_dataset 一致）
            if not all(k in rec for k in ("premise", "hypothesis", "label")):
                logger.warning("跳过缺字段的样本")
                continue
            sources.append({
                "premise": str(rec["premise"]),
                "hypothesis": str(rec["hypothesis"]),
                "label": int(rec["label"]),
                "raw_idx": rec.get("idx"),  # mnlim/mnlimm 自带 idx；sick/snli 为 None
            })
    # pair_id：优先用原始 idx，否则用运行索引
    for i, src in enumerate(sources):
        src["pair_id"] = src["raw_idx"] if src["raw_idx"] is not None else i
    return sources


def make_source_record(src: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """构造 8 字段源记录。"""
    return {
        "idx": idx,
        "premise": src["premise"],
        "hypothesis": src["hypothesis"],
        "mr_id": "none",
        "pair_id": src["pair_id"],
        "label": src["label"],
        "mr_type": "inv",
        "is_source": True,
    }


def make_followup_record(aug: Dict[str, Any], idx: int, pair_id: int,
                         mr_name: str, mr_type: str,
                         fallback_label: int) -> Dict[str, Any]:
    """构造 8 字段后续记录；MR 返回的未知标签回退为源标签。"""
    label = STR_TO_INT.get(aug.get("label"), fallback_label)
    if aug.get("label") not in STR_TO_INT:
        logger.warning(f"{mr_name} 返回未知标签 {aug.get('label')!r}，回退为源标签 {fallback_label}")
    return {
        "idx": idx,
        "premise": aug.get("premise", ""),
        "hypothesis": aug.get("hypothesis", ""),
        "mr_id": mr_name,
        "pair_id": pair_id,
        "label": label,
        "mr_type": mr_type,
        "is_source": False,
    }


def make_followup_records(aug: Any, followup_idx: int, pair_id: int,
                          mr_name: str, fallback_label: int) -> List[Dict[str, Any]]:
    """防御性处理 tuple 返回（当前 8 个 MR 均非 dual，属冗余但无害）。"""
    items = aug if isinstance(aug, tuple) else (aug,)
    mr_type = NLIMRTool.MR_TYPE.get(mr_name, "inv")
    records: List[Dict[str, Any]] = []
    for j, item in enumerate(items):
        if item is None:
            continue
        records.append(
            make_followup_record(item, followup_idx + j, pair_id, mr_name,
                                 mr_type, fallback_label))
    return records


def dispatch(tool: NLIMRTool, mr_name: str,
             mr_input: Dict[str, Any]) -> Any:
    """调用 MR。MT_MRS 的值是类方法（unbound），需显式传入 tool 作为 self。"""
    return NLIMRTool.MT_MRS[mr_name](tool, mr_input)


def _write_followup(fh: Any, rec: Dict[str, Any], src: Dict[str, Any],
                    stats: Dict[str, int]) -> bool:
    """写入一条后续记录；与源完全一致则跳过，返回是否写入。"""
    if (rec["premise"], rec["hypothesis"], rec["label"]) == (
            src["premise"], src["hypothesis"], src["label"]):
        stats["skipped"] += 1
        return False
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats["followup"] += 1
    return True


def _process_single(fh: Any, tool: NLIMRTool, dataset: str, mr_name: str,
                    src: Dict[str, Any], source_idx: int, followup_idx: int,
                    stats: Dict[str, int]) -> tuple:
    """单样本路径：写源记录 + 调 MR + 写后续记录，返回 (源idx, 后续idx)。"""
    fh.write(json.dumps(make_source_record(src, source_idx),
                        ensure_ascii=False) + "\n")
    stats["source"] += 1
    source_idx += 1

    mr_input = {
        "premise": src["premise"],
        "hypothesis": src["hypothesis"],
        "label": INT_TO_STR.get(src["label"]),
    }
    try:
        result = dispatch(tool, mr_name, mr_input)
    except Exception as exc:
        logger.warning(
            f"[{dataset}/{mr_name}] pair_id={src['pair_id']} MR 调用异常: {exc}")
        stats["failed"] += 1
        return source_idx, followup_idx
    if result is None or result == "":
        stats["skipped"] += 1
        return source_idx, followup_idx
    for rec in make_followup_records(result, followup_idx, src["pair_id"],
                                     mr_name, src["label"]):
        if _write_followup(fh, rec, src, stats):
            followup_idx += 1
    return source_idx, followup_idx


def _process_batch(fh: Any, tool: NLIMRTool, dataset: str, mr_name: str,
                   batch: List[Dict[str, Any]], source_idx: int,
                   followup_idx: int, stats: Dict[str, int]) -> tuple:
    """批量路径：写源记录 + mr_process_batch + 写后续记录，返回 (源idx, 后续idx)。"""
    for src in batch:
        fh.write(json.dumps(make_source_record(src, source_idx),
                            ensure_ascii=False) + "\n")
        source_idx += 1
        stats["source"] += 1

    batch_work = [{
        "premise": s["premise"],
        "hypothesis": s["hypothesis"],
        "label": INT_TO_STR.get(s["label"]),
    } for s in batch]
    try:
        followups = tool.mr_process_batch(batch_work, mr_name)
    except Exception as exc:
        logger.warning(f"[{dataset}/{mr_name}] 批量处理异常: {exc}")
        stats["failed"] += len(batch)
        return source_idx, followup_idx

    for src, fu in zip(batch, followups):
        if fu is None:
            stats["skipped"] += 1
            continue
        for rec in make_followup_records(fu, followup_idx, src["pair_id"],
                                         mr_name, src["label"]):
            if _write_followup(fh, rec, src, stats):
                followup_idx += 1
    return source_idx, followup_idx


def _existing_counts(path: pathlib.Path) -> tuple:
    """读取已存在的输出文件，返回 (已写源记录数, 已写后续记录数)。"""
    s = f = 0
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("is_source"):
                    s += 1
                else:
                    f += 1
    return s, f


def process_dataset(tool: NLIMRTool, dataset: str, sources: List[Dict[str, Any]],
                    mr_names: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """对单个数据集：按 MR 逐文件写入源记录与后续记录，返回 {mr_name: stats}。

    resume=True 时对已存在的 (dataset, mr_name) 文件跳过已写的源样本、续接后续 idx。
    """
    out_dir = _script_dir() / cfg["output_root"] / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = cfg["dry_run_limit"] if cfg["dry_run"] else cfg.get("sample_limit")
    batch_size = int(cfg.get("batch_size", 0) or 0)
    resume = bool(cfg.get("resume"))
    mr_stats: Dict[str, Any] = {}
    for mr_name in mr_names:
        stats = {"source": 0, "followup": 0, "skipped": 0, "failed": 0}
        out_path = out_dir / f"{mr_name}.jsonl"
        mode = "w"
        source_skip = 0
        followup_start = 0
        if resume and out_path.exists():
            source_skip, followup_start = _existing_counts(out_path)
            mode = "a"
        with open(out_path, mode, encoding="utf-8") as fh:
            source_idx = source_skip
            followup_idx = followup_start
            # 跳过已写的源样本；再套用 limit
            remaining = sources[source_skip:]
            if limit is not None:
                remaining = remaining[:max(0, limit - source_skip)]
            i = 0
            while i < len(remaining):
                if batch_size > 0:
                    batch = remaining[i:i + batch_size]
                    if not batch:
                        break
                    source_idx, followup_idx = _process_batch(
                        fh, tool, dataset, mr_name, batch, source_idx,
                        followup_idx, stats)
                    i += len(batch)
                else:
                    src = remaining[i]
                    source_idx, followup_idx = _process_single(
                        fh, tool, dataset, mr_name, src, source_idx,
                        followup_idx, stats)
                    i += 1
        mr_stats[mr_name] = stats
        logger.info(
            f"[{dataset}] {mr_name}: 源={stats['source']} 后续={stats['followup']} "
            f"跳过={stats['skipped']} 失败={stats['failed']}"
            f"{' (resume: 已跳过%d源/%d后续)' % (source_skip, followup_start) if resume and source_skip else ''}"
            f" -> {out_path}")
    return mr_stats


class _FakeLLM:
    """dry-run 用假 LLM：兼容单样本（Sentence:\\n{input}）与批量（JSON 数组）prompt。"""

    def chat(self, content: str, role: str = "user", save: bool = False) -> str:
        # 批量验证 prompt：返回每项答案的 JSON 数组
        if "answer strings" in content:
            m = re.search(r"exactly (\d+) answer strings", content)
            n = int(m.group(1)) if m else 1
            return json.dumps(["yes"] * n)
        # 批量变换 prompt：返回每句（加后缀）的 JSON 数组
        m = re.search(r"exactly (\d+) strings", content)
        if m:
            n = int(m.group(1))
            lines = [l for l in content.split("\n") if re.match(r"^\d+\. ", l)]
            sents = [re.sub(r"^\d+\. ", "", l) for l in lines][:n]
            return json.dumps([s + " batched" for s in sents])
        # 单样本 prompt：返回输入句（`Sentence:\\n{input}` 结尾）
        return content.rsplit("Sentence:\n", 1)[-1].strip()


def build_tool(cfg: Dict[str, Any]) -> NLIMRTool:
    """构造 NLIMRTool。dry_run 用假 LLM；否则必须显式传有效模型名（不能用占位符默认）。"""
    if cfg["dry_run"]:
        logger.info("dry_run 模式：使用假 LLM，不连接 LLM 服务")
        return NLIMRTool(llm=_FakeLLM())
    model_name = cfg["model_name"]
    if not model_name or model_name == "默认大模型名称":
        raise ValueError(
            "model_name 无效，不能使用 LMstudio 的占位符默认模型，请在 CONFIG 中指定")
    llm = OpenAIChat(model_name, HOST, PORT)
    return NLIMRTool(llm=llm)


def validate_record(rec: Dict[str, Any], expect_mr: str, expect_mr_type: str,
                    expect_source: bool) -> None:
    """断言一条输出记录的结构与关键字段值。"""
    for key in REQUIRED_FIELDS:
        assert key in rec, f"缺少字段 {key}: {rec}"
    assert isinstance(rec["idx"], int) and rec["idx"] >= 0, f"idx 非法: {rec}"
    assert isinstance(rec["premise"], str), f"premise 非法: {rec}"
    assert isinstance(rec["hypothesis"], str), f"hypothesis 非法: {rec}"
    assert isinstance(rec["label"], int) and rec["label"] in (0, 1, 2), f"label 非法: {rec}"
    assert isinstance(rec["pair_id"], int), f"pair_id 非法: {rec}"
    assert isinstance(rec["is_source"], bool), f"is_source 非法: {rec}"
    assert rec["mr_id"] == expect_mr, f"mr_id={rec['mr_id']!r} != {expect_mr!r}"
    assert rec["mr_type"] == expect_mr_type, f"mr_type={rec['mr_type']!r} != {expect_mr_type!r}"
    assert rec["is_source"] == expect_source, f"is_source={rec['is_source']!r} != {expect_source!r}"


def _self_check_shape() -> None:
    """无 LLM 的确定性自检：断言源记录与后续记录的 8 字段精确输出。"""
    src = {"premise": "P", "hypothesis": "H", "label": 0, "pair_id": 7}
    s = make_source_record(src, 3)
    assert s == {
        "idx": 3, "premise": "P", "hypothesis": "H", "mr_id": "none",
        "pair_id": 7, "label": 0, "mr_type": "inv", "is_source": True,
    }, s
    aug = {"premise": "P", "hypothesis": "H'", "label": "contradiction"}
    f = make_followup_record(aug, 0, 7, "negation_flip", "flip", 0)
    assert f == {
        "idx": 0, "premise": "P", "hypothesis": "H'", "mr_id": "negation_flip",
        "pair_id": 7, "label": 2, "mr_type": "flip", "is_source": False,
    }, f
    logger.info("记录结构自检通过")


def _verify_outputs(output_root: pathlib.Path, datasets: List[str],
                    mr_names: List[str], source_counts: Dict[str, int]) -> None:
    """dry-run 后校验：重读所有输出文件，验证记录结构与 idx 序列。"""
    logger.info("开始校验输出文件...")
    checked = 0
    for dataset in datasets:
        for mr_name in mr_names:
            path = output_root / dataset / f"{mr_name}.jsonl"
            if not path.exists():
                logger.error(f"缺少输出文件: {path}")
                continue
            source_ids: List[int] = []
            followup_ids: List[int] = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec["is_source"]:
                        validate_record(rec, "none", "inv", True)
                        source_ids.append(rec["idx"])
                    else:
                        validate_record(rec, mr_name,
                                        NLIMRTool.MR_TYPE.get(mr_name, "inv"),
                                        False)
                        followup_ids.append(rec["idx"])
            assert source_ids == list(range(len(source_ids))), (
                f"{path} 源记录 idx 不连续: {source_ids[:5]}...")
            assert followup_ids == list(range(len(followup_ids))), (
                f"{path} 后续记录 idx 不连续: {followup_ids[:5]}...")
            expected = source_counts.get(dataset)
            if expected is not None:
                assert len(source_ids) == expected, (
                    f"{path} 源记录条数 {len(source_ids)} != 期望 {expected}")
            checked += 1
    logger.info(f"输出文件校验通过，共 {checked} 个文件")


def main() -> None:
    cfg = dict(CONFIG)
    if "--dry-run" in sys.argv:
        cfg["dry_run"] = True
    for arg in sys.argv:
        if arg.startswith("--datasets="):
            cfg["datasets"] = [
                d.strip() for d in arg.split("=", 1)[1].split(",") if d.strip()
            ]
        if arg.startswith("--sample-limit="):
            cfg["sample_limit"] = int(arg.split("=", 1)[1])
        if arg.startswith("--batch-size="):
            cfg["batch_size"] = int(arg.split("=", 1)[1])
        if arg.startswith("--output-root="):
            cfg["output_root"] = arg.split("=", 1)[1].strip() or "mr_test_data"
        if arg == "--resume":
            cfg["resume"] = True

    if cfg["dry_run"]:
        _self_check_shape()

    base = _script_dir()
    original_root = base / cfg["original_root"]
    output_root = base / cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    mr_names = cfg["mr_names"] or list(NLIMRTool.MT_MRS)
    logger.info(f"待处理 MR: {mr_names}")
    logger.info(f"待处理数据集: {cfg['datasets']}")
    tool = build_tool(cfg)

    summary: Dict[str, Any] = {}
    source_counts: Dict[str, int] = {}
    for dataset in cfg["datasets"]:
        test_path = original_root / dataset / "test.json"
        logger.info(f"加载测试集: {test_path}")
        sources = load_sources(test_path)
        logger.info(
            f"[{dataset}] 有效源样本: {len(sources)}（已过滤无金标样本）")
        if not sources:
            logger.warning(f"[{dataset}] 无有效样本，跳过")
            continue
        try:
            mr_stats = process_dataset(tool, dataset, sources, mr_names, cfg)
        except Exception as exc:
            logger.error(f"[{dataset}] 处理失败: {exc}")
            continue
        summary[dataset] = mr_stats
        # 写入的源记录条数 = min(有效样本数, 生效上限)
        limit = cfg["dry_run_limit"] if cfg["dry_run"] else cfg.get("sample_limit")
        source_counts[dataset] = min(limit, len(sources)) if limit else len(sources)

    logger.info("=" * 60)
    for dataset, mr_stats in summary.items():
        total_followup = sum(st["followup"] for st in mr_stats.values())
        total_skipped = sum(st["skipped"] for st in mr_stats.values())
        total_failed = sum(st["failed"] for st in mr_stats.values())
        logger.info(
            f"[{dataset}] 汇总: 后续={total_followup} 跳过={total_skipped} 失败={total_failed}")
    logger.info(f"输出目录: {output_root}")

    if cfg["dry_run"]:
        _verify_outputs(output_root, list(summary.keys()), mr_names, source_counts)


if __name__ == "__main__":
    main()
