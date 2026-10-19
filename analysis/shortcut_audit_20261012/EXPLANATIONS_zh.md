# 补充诊断：表层探针、长度与剩余词线索

日期：2026-10-12。复用主报告的冻结源组划分。重放三 seed 表层探针及 seed42 raw/masked 词探针，所有四集的准确率均与主报告逐 seed 指标完全一致。

## 表层探针的不确定性与类别平衡

| 数据 | 表层 accuracy | 源组 bootstrap 95% CI | balanced accuracy | 多数类基线 | 高于基线（百分点） |
| --- | --- | --- | --- | --- | --- |
| snli | 98.08% | 97.86%–98.31% | 96.85% | 51.16% | 46.92 |
| mnlim | 93.98% | 93.53%–94.46% | 82.89% | 64.04% | 29.94 |
| mnlimm | 93.47% | 93.00%–93.95% | 81.10% | 63.68% | 29.80 |
| sick | 94.06% | 93.42%–94.72% | 90.11% | 52.09% | 41.98 |

该探针只有 8 个词面布尔特征和 3 个长度/标点计数。高于多数类基线的幅度为当前数据的表层标签可预测性证据，不能解释成正式 LLM 中有同样比例的预测因果上依赖捷径。

## follow-up 长度与目标标签

| 数据 | entailment 平均/中位词数 | neutral 平均/中位词数 | contradiction 平均/中位词数 |
| --- | --- | --- | --- |
| train | 5.67 / 5.0 | 13.02 / 13.0 | 8.25 / 7.0 |
| snli | 5.63 / 5.0 | 13.16 / 13.0 | 8.29 / 7.0 |
| mnlim | 8.26 / 8.0 | 17.12 / 17.0 | 10.58 / 10.0 |
| mnlimm | 9.04 / 9.0 | 18.32 / 18.0 | 11.36 / 11.0 |
| sick | 7.31 / 7.0 | 14.99 / 14.0 | 9.30 / 8.0 |

训练增强中 E 平均约 5.67 词，N 约 13.02 词，C 约 8.25 词。增补整段 clause 的 MR 与局部 INV 编辑的长度差异，也可被单句分类器利用。以后平衡长度需要在真实语义与 MR 合同允许范围内实现，不能删掉决定标签的细节或机械截断。

## 词探针的高权重特征（seed42）

| 探针 | 标签 | 最高的 8 个正权重 unigram/bigram |
| --- | --- | --- |
| raw_word | entailment | `and white`, `clothes`, `not standing`, `man and`, `watching concert`, `his owner`, `in water`, `blue cap` |
| raw_word | neutral | `and`, `and he`, `only if`, `and the`, `if`, `and they`, `and she`, `only` |
| raw_word | contradiction | `no`, `not`, `nobody`, `is not`, `are not`, `but`, `in`, `nothing` |
| masked_word | entailment | `blue cap`, `are outside`, `in field`, `the pool`, `are outdoors`, `some people`, `the woman`, `watching concert` |
| masked_word | neutral | `wearing`, `using`, `has`, `nearby`, `for`, `the`, `red`, `uses` |
| masked_word | contradiction | `does`, `do`, `with red`, `in`, `at the`, `in blue`, `are indoors`, `has to` |

raw_word 的 N 特征主要是 and/if，C 特征主要是 no/not/nobody/but，与预定义线索检测一致。masked_word 中 does/do 等特征也可能来自删除否定后的语法残留；不能把它们全部认定为原始生成器中新发现的捷径。线性系数反映相关性，完整权重及训练行标签计数见 explanatory_probes.json。

## 结构检查中的待复核样本

本轮规范化 P/H 标签冲突在训练和四集均为 0；但有两条 follow-up 的规范化 hypothesis 与其源相同：

| 数据 | pair_id | MR | source hypothesis | follow-up hypothesis |
| --- | --- | --- | --- | --- |
| snli | 3746 | composite_inv | A person in a shirt | A person in a shirt |
| mnlim | 9378 | synonym_replacement | In both Britain and America the term covers almost everyone. | In both Britain and America, the term covers almost everyone. |

这两条应人工复核操作有效性；本轮未删除或改写它们。标签自洽、审核元数据与原始/增强文本的操作有效性需要分别判断。

补充计算未读取 premise 作为单句探针输入，未改动任何输入文件。
