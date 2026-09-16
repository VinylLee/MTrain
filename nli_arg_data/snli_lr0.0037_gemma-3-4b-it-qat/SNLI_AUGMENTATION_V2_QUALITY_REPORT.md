# SNLI 数据增强 v2.3 质量报告

- 完成日期：2026-09-10
- 输入：`labeled_data.json`，2033 条
- 生成、候选判定和审计模型：仅 `gemma-3-4b-it-qat`
- 最终输出：`augmented_data.json`
- 旧版备份：`augmented_data.pre_v2_20260910.json`

## 结果对比

| 指标 | 旧版 | v2.3 最终版 |
|---|---:|---:|
| 总行数 | 5340 | 2813 |
| 原始行 | 2033 | 2033 |
| 合成行 | 3307 | 780 |
| Entailment | 1273 | 930 |
| Neutral | 1855 | 959 |
| Contradiction | 2212 | 924 |
| 重复 NLI 三元组 | 127 | 0 |
| 提示词污染 | 48 | 0 |

最终合成数据包含 inv 196、flip 271、neutral 313。质量优先策略主动减少低置信样本，三类总标签 930/959/924，接近平衡。

## 质量验证

1. 固定 seed 的 60 条小样本：人工审阅 12/12 正确；Gemma 独立审计 11/12，唯一分歧是模型把 premise 未说明的红色柱塞误判为 entailment。
2. 对未人工过滤的 786 条合成数据做全量 Gemma 审计：成功 786/786；标签准确率 97.71%，语法率 100%，inv 等价率 97.01%，完全通过率 96.95%；四项总体质量门槛全部通过。
3. 人工复核全部 24 条 Gemma 边界项，删除 6 条确认风险样本。
4. 最终确定性审计：2033 原始行与 780 合成行；0 重复、0 未变化、0 缺失源 pair、0 提示词污染。
5. 人工过滤后 780 条的 Gemma 指标：标签准确率 97.82%，语法率 100%，操作有效率 100%，inv 等价率 98.47%，完全通过率 97.44%。

## 人工删除项

| pair_id | 操作 | 删除原因 |
|---:|---|---|
| 137 | inv | `cannot dance` 改成 `does not dance`，能力与当前动作不等价 |
| 226 | inv | 删除 `young`，发生泛化 |
| 268 | neutral | premise 已说明使用 cards，新增 `a deck of cards` 过于接近蕴含 |
| 981 | inv | `different dresses` 缩窄为 `different colored dresses` |
| 1018 | inv | 删除数量 `Two` |
| 1545 | inv | `nothing to eat` 改成当前 `does not eat anything` |

## v2.3 关键改动

- 每次生成同时提供 premise、源 hypothesis、源标签、目标标签和严格 SNLI 定义。
- inv 检查内容词、数字、否定、premise 泄漏与语义等价。
- flip 只接受可证明的直接全句否定，拒绝危险量词、已有否定和不自然的机械否定。
- neutral 必须保留完整源 hypothesis，只添加一个 premise 未确定的具体事实。
- 每个候选由隔离的 Gemma 请求重判源标签、候选标签、语法和操作有效性。
- 每 10 条原子检查点并支持 `--resume`；恢复测试证明输出与日志哈希不变。

## 复现命令

    python -u augment_snli_v2.py --all --seed 42 --generator-model gemma-3-4b-it-qat --judge-model gemma-3-4b-it-qat --output nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data_v2_full.json --checkpoint-every 10 --overwrite

    python -u audit_augmentation_quality.py --source nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/labeled_data.json --data nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data_v2_full.json --judge-sample 786 --judge-model gemma-3-4b-it-qat --batch-size 8 --seed 42

## SHA-256

- 输入：`441e72a3c2499cf896e699d541e1e772d2ae77075605516b88f8a00bc90c646e`
- 旧版备份：`0c03838e4f6f1a71e97a4cfd3813b2dfcbe964526f361258912b6861119d1b7d`
- 最终数据：`b2b0039a3e7af7cf8b651e15fa18a28a7b936b3c3b900460af98d2d2fbbcf36b`
- 生成脚本：`0e150438266aa66b3953962086a4b044863a8e2f923211d54039404f6db8780b`
- 审计脚本：`88b86b2595f6a5c16842b5220e29dd021f9009a43214666d1aeecce73a959457`
