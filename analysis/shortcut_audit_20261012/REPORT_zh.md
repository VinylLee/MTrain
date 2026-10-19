# 增强训练集与 v1.1 测试集捷径诊断报告

日期：2026-10-12。所有数字均为本轮重新计算，不沿用历史准确率。

**主要发现：只使用连接词、否定标记与长度等 11 个表层特征，就能在四个测试集 follow-up 上达到 93.47%–98.08%；词级 hypothesis-only TF-IDF 为 89.01%–94.95%。** 两类探针均未读取 premise，也未将 MR、label、component_mrs 或 ensemble 审核信息作为输入。

这说明当前数据存在明显的单句标签线索。文本重合检查与模板捷径是两类证据，不能将高单句准确率直接称为训练测试泄漏。

本报告没有修改数据；与其他检测配置之间的小幅指标差异不能解释为捷径已经降低。这里的准确率衡量线索可利用性，不能直接当作正式 LLM 实际依赖捷径的比例。

## 1. 数据范围与可复现性

| 数据 | 原始样本 | follow-up | 合计 |
| --- | --- | --- | --- |
| train | 2033 | 3307 | 5340 |
| snli | 9824 | 13328 | 23152 |
| mnlim | 9815 | 10396 | 20211 |
| mnlimm | 9832 | 10393 | 20225 |
| sick | 4906 | 6015 | 10921 |

训练输入是完整 v3.3 的 5,340 行数据；测试输入是本机导入的四个 v1.1 JSONL。各训练划分由原始 pair_id 分组，源行与全部 follow-up 共同进入同一侧。不同数据集的 pair_id 命名空间分别处理。

每个 seed 使用 80% 原始 pair groups 拟合，20% 留作训练域诊断验证；按原始标签分层。seed=42/43/44。四个正式测试集只用于评估，未用于选择特征、模型超参数或阈值。

TF-IDF 的词表与 IDF 仅在每个 seed 的训练部分拟合。词特征为 unigram/bigram，min_df=2，max_features=30,000；字符特征为 char_wb 3–5 gram，max_features=60,000。LinearSVC C=1；表层探针使用标准化的 11 个数值特征与 LogisticRegression C=1。

11 个特征为：only-if、comma-and、comma-but、普通 but、显式否定、否定词开头、普通 if、不确定性 modal 共 8 个布尔特征，加 hypothesis 词数、字符数、逗号数。它们不编码实体、动作或 premise/hypothesis 的语义对应。

这是轻量 CPU 诊断分类器拟合，不是 Gemma/LoRA 训练，也未调用 LLM 或 GPU。原有输入、标签与提示词均未修改。

输入路径与 SHA-256：

| 数据 | 路径 | SHA-256 |
| --- | --- | --- |
| train | `nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data_all_label_mrs_v3_3_full.json` | `648946c122110d808818df5cef9166602c9166b660c9c8a7719ecde4c010ab1a` |
| snli | `mr_test_data_merged_v1.1/snli.jsonl` | `21c1236c6b196be3382af3f4f04136d350ff73e03c9970db4f2f66f5763115fc` |
| mnlim | `mr_test_data_merged_v1.1/mnlim.jsonl` | `4d172e49a05a27dadc37e5cea25b9f16467db0267b1c22270528d5a4cec15de1` |
| mnlimm | `mr_test_data_merged_v1.1/mnlimm.jsonl` | `1cc44cc9f27655a00a531e79e4b3a8a8678c09665c691b9a9d3a1ae5c743ddb1` |
| sick | `mr_test_data_merged_v1.1/sick.jsonl` | `6d6ce729d669699434759d9b1fe0c4286dca3f4c643c87052e3ac0e891009ffd` |

## 2. 文本重合与近重复

| 测试集 | 精确 P/H | 规范化 P/H | P/H 对调 | 双侧 Jaccard ≥0.9 | 双侧 Jaccard ≥0.8 | 仅 H 规范化重合 |
| --- | --- | --- | --- | --- | --- | --- |
| snli | 0 | 0 | 0 | 0 | 0 | 453 |
| mnlim | 0 | 0 | 0 | 0 | 0 | 0 |
| mnlimm | 0 | 0 | 0 | 0 | 0 | 0 |
| sick | 0 | 0 | 0 | 0 | 0 | 62 |

规范化使用 Unicode NFKC、大小写折叠和词 token。近重复要求同一个训练行与测试行的 premise、hypothesis 各自 token-set Jaccard 都达到阈值；使用完整候选检索，无随机抽样或候选数量截断。它是词面重复筛查，不是穷尽语义污染证明。仅 H 重合可能来自常见短句，不能单独认定完整样本泄漏。

## 3. 固定连接词的标签绑定

| 数据 | 线索 | follow-up 命中数 | 覆盖率 | 占优标签 | 标签纯度 |
| --- | --- | --- | --- | --- | --- |
| train | only_if | 565 | 17.08% | neutral | 100.00% |
| train | comma_and | 1362 | 41.19% | neutral | 99.93% |
| train | comma_but | 184 | 5.56% | contradiction | 100.00% |
| train | explicit_negation | 944 | 28.55% | contradiction | 98.52% |
| snli | only_if | 2546 | 19.10% | neutral | 100.00% |
| snli | comma_and | 4266 | 32.01% | neutral | 99.77% |
| snli | comma_but | 896 | 6.72% | contradiction | 100.00% |
| snli | explicit_negation | 4437 | 33.29% | contradiction | 98.69% |
| mnlim | only_if | 2689 | 25.87% | neutral | 100.00% |
| mnlim | comma_and | 4061 | 39.06% | neutral | 98.42% |
| mnlim | comma_but | 297 | 2.86% | contradiction | 71.38% |
| mnlim | explicit_negation | 3808 | 36.63% | contradiction | 71.17% |
| mnlimm | only_if | 2710 | 26.08% | neutral | 100.00% |
| mnlimm | comma_and | 4023 | 38.71% | neutral | 98.26% |
| mnlimm | comma_but | 228 | 2.19% | contradiction | 78.95% |
| mnlimm | explicit_negation | 3610 | 34.73% | contradiction | 75.04% |
| sick | only_if | 1216 | 20.22% | neutral | 100.00% |
| sick | comma_and | 1904 | 31.65% | neutral | 99.53% |
| sick | comma_but | 171 | 2.84% | contradiction | 100.00% |
| sick | explicit_negation | 1904 | 31.65% | contradiction | 99.16% |

生成器 `augment_snli_all_label_mrs.py` 的 build_generation_prompt 强制固定前缀和连接词：uninformative 使用 `, and`，adding_contradiction 使用 `, but`，conditional_clause 使用 `only if`。生成测试适配器直接导入该模块。当前测试文本也呈现同样关联；由于导入目录缺少完整生成日志，本报告不声称已认证所有测试行的历史生成命令。

预定义规则：先将 only-if / comma-and 判为 neutral，再将 comma-but 判为 contradiction；扩展规则在此前基础上将普通 but / 显式否定判为 contradiction。其余行弃权，规则没有依据测试结果调整。

| 数据 | 三连接词覆盖 | 覆盖部分准确率 | 加否定覆盖 | 覆盖部分准确率 | 扩展规则正确行 / 全部 follow-up |
| --- | --- | --- | --- | --- | --- |
| train | 63.71% | 99.95% | 86.30% | 99.93% | 86.24% |
| snli | 57.77% | 99.87% | 83.92% | 99.87% | 83.81% |
| mnlim | 66.56% | 98.99% | 91.41% | 98.01% | 89.59% |
| mnlimm | 66.01% | 98.92% | 90.89% | 98.14% | 89.19% |
| sick | 54.63% | 99.73% | 83.11% | 99.78% | 82.93% |

覆盖部分准确率是条件准确率，不能写成整个测试集准确率。未覆盖行不计为预测正确。否定本身可以是合法语义操作；风险来自它与目标标签的单侧绑定，而不是出现否定词就必然有问题。

## 4. 单句探针与消融

| 探针：follow-up accuracy（三 seed 均值） | snli | mnlim | mnlimm | sick |
| --- | --- | --- | --- | --- |
| Hypothesis word TF-IDF | 94.95% | 89.81% | 89.01% | 91.50% |
| Hypothesis word + char TF-IDF | 94.83% | 89.04% | 88.25% | 90.67% |
| Cue-masked hypothesis word TF-IDF | 66.18% | 67.66% | 66.70% | 64.54% |
| Surface cues + length | 98.08% | 93.98% | 93.47% | 94.06% |
| Original-only hypothesis word TF-IDF | 39.88% | 44.43% | 45.31% | 39.10% |
| Premise word TF-IDF | 43.97% | 50.23% | 49.25% | 42.34% |
| Training majority | 51.16% | 64.04% | 63.68% | 52.09% |
| MR-ID label prior (metadata diagnostic) | 99.23% | 98.02% | 97.81% | 92.40% |

| 测试集 | 词+字符 H accuracy ± seed SD | 按 pair-group bootstrap 95% CI | balanced accuracy | macro-F1 |
| --- | --- | --- | --- | --- |
| snli | 94.83% ± 0.23% | 94.50%–95.17% | 90.04% | 0.9207 |
| mnlim | 89.04% ± 0.11% | 88.51%–89.57% | 70.70% | 0.7345 |
| mnlimm | 88.25% ± 0.28% | 87.72%–88.83% | 69.41% | 0.7219 |
| sick | 90.67% ± 0.21% | 89.99%–91.35% | 82.22% | 0.8436 |

bootstrap 按有 follow-up 的源 pair groups 整组抽样，1,500 次，统计量为三 seed follow-up micro accuracy 的均值。seed SD 描述分组拟合随机性，bootstrap 区间描述当前测试源组抽样不确定性，均不覆盖生成器或语义标签错误的不确定性。

原始样本训练探针使用与增强探针相同的 80% 源组，但仅拟合源行，因此样本数更少；它用于衡量增强暴露是否带来额外单句标签可预测性，不能当作严格预算匹配的增强效果比较。

MR-ID label prior 只根据训练部分的 MR ID 预测占优标签，是结构诊断参照，不是理论上界；普通 NLI 测试未必提供 MR ID，不能把这个值称为真实测试标签泄漏。

遮蔽探针删除 only-if、comma-and、comma-but、but 和显式否定。raw_word 与 masked_word 的特征和模型参数相同，可直接比较。这个处理会破坏部分语义与语法，仅用于诊断，不生成新 NLI 数据，也不能将准确率下降当作安全消除捷径的证据。

| 测试集 | 原始词 H | 遮蔽词 H | 下降（百分点） | 仅表层特征 |
| --- | --- | --- | --- | --- |
| snli | 94.95% | 66.18% | 28.77 | 98.08% |
| mnlim | 89.81% | 67.66% | 22.15 | 93.98% |
| mnlimm | 89.01% | 66.70% | 22.31 | 93.47% |
| sick | 91.50% | 64.54% | 26.96 | 94.06% |

## 5. 原始测试与配对联合正确率

| 测试集 | H 词+字符：全部原始行 | follow-up | 对应源行（按 follow-up 加权） | source-follow-up 联合正确率 |
| --- | --- | --- | --- | --- |
| snli | 49.99% | 94.83% | 48.86% | 48.15% |
| mnlim | 38.46% | 89.04% | 18.41% | 17.60% |
| mnlimm | 38.33% | 88.25% | 14.74% | 14.09% |
| sick | 43.31% | 90.67% | 32.17% | 31.17% |

联合指标以每条 source-follow-up 边为单位，只有两端预测均正确才计为正确；一个源有多个 follow-up 会产生多条边。本指标不是所有 follow-up 必须同时正确的整组指标，也不是项目正式 MR/MSR 指标的重计算。

| 测试集 | 表层探针：全部原始行 | follow-up | 配对源行（按 follow-up 加权） | 联合正确率 |
| --- | --- | --- | --- | --- |
| snli | 40.78% | 98.08% | 87.61% | 86.79% |
| mnlim | 41.88% | 93.98% | 43.10% | 42.92% |
| mnlimm | 42.81% | 93.47% | 35.41% | 35.17% |
| sick | 47.83% | 94.06% | 67.14% | 66.81% |

**联合指标也不能完全消除捷径。** 例如 SNLI 表层探针的联合正确率仍较高，而其全部原始源行准确率很低：有 follow-up 的源组及其边高度偏向 entailment，使简单的源标签倾向也能获益。应同时保留对全部原始源行的评估，以及按源标签分层的联合评估。

对于 neutral MR，若 MSR 只检验 follow-up 是否为 neutral，固定连接词可让 MSR 很高而无需判断源样本。应并列报告原始 NLI、逐标签 balanced accuracy、逐 MR 指标和配对联合正确率，避免单一 follow-up/MSR 掩盖这种差异。

## 6. 源样本选择与 MR 配比

| 数据 | 原始源 E 占比 | 有 follow-up 的源组数 | 这些源组 E 占比 | 按 follow-up 加权的源 E 占比 |
| --- | --- | --- | --- | --- |
| train | 36.35% | 760 | 96.58% | 99.18% |
| snli | 34.28% | 3446 | 97.39% | 99.23% |
| mnlim | 35.45% | 3620 | 94.61% | 98.02% |
| mnlimm | 35.22% | 3632 | 94.03% | 97.81% |
| sick | 28.62% | 1768 | 78.85% | 92.40% |

flip/neutral 操作在当前合同下多作用于 entailment 源，INV 在接受过滤后也可能偏向 E。这属于实际选择偏差；不能用跨 relation-family 或改标签的方法人为抹平。

| MR | 训练 follow-up 数 | 训练占优标签/纯度 | snli H词+字符 accuracy (n) | mnlim H词+字符 accuracy (n) | mnlimm H词+字符 accuracy (n) | sick H词+字符 accuracy (n) |
| --- | --- | --- | --- | --- | --- | --- |
| adding_contradiction | 180 | contradiction / 100.00% | 99.12% (875) | 98.26% (192) | 93.13% (165) | 99.79% (159) |
| antonym_substitution | 13 | contradiction / 100.00% | 48.76% (67) | 0.00% (2) | 66.67% (3) | 88.89% (6) |
| composite_flip | 255 | contradiction / 100.00% | 99.26% (1175) | 91.25% (461) | 91.93% (500) | 99.16% (478) |
| composite_inv | 23 | entailment / 95.65% | 73.10% (145) | 42.50% (40) | 31.94% (48) | 45.24% (28) |
| composite_neutral | 208 | neutral / 100.00% | 99.63% (998) | 98.23% (639) | 98.69% (611) | 99.89% (615) |
| conditional_clause | 565 | neutral / 100.00% | 99.70% (2546) | 99.13% (2689) | 99.50% (2710) | 99.92% (1216) |
| negation_flip | 480 | contradiction / 100.00% | 99.02% (2275) | 91.53% (1904) | 89.14% (1873) | 97.63% (970) |
| pronoun_substitution | 355 | entailment / 94.37% | 72.03% (1598) | 31.30% (835) | 30.38% (881) | 56.35% (966) |
| synonym_replacement | 33 | entailment / 84.85% | 64.78% (177) | 31.43% (193) | 30.34% (167) | 40.93% (79) |
| uninformative | 1150 | neutral / 100.00% | 99.41% (3252) | 97.95% (3315) | 98.22% (3289) | 99.79% (1275) |
| voice_switch | 45 | entailment / 97.78% | 82.88% (220) | 20.11% (126) | 16.44% (146) | 79.67% (223) |

稀少 MR 的准确率波动很大，不宜用 2–13 条测试/训练样本作稳健结论。逐 MR 高准确率也可能来自固定目标标签与模板，而非具体变换理解。

## 7. 降低捷径的建议（本轮未实施）

1. **优先处理固定模板与标签绑定。** 单纯找不同主题或删除近重复不会处理 only-if/comma-and/comma-but 的绑定。对允许表达变体的 MR，增加结构与连接形式多样性，并保留独立操作有效性与 NLI 标签审核。对于当前定义明确要求 only-if 的 conditional_clause，改变该格式需要单独评审并升级合同版本，不能偷偷改成其他 MR 或只删词。

2. **保持测试独立性。** 已用这四集诊断过生成问题，后续若据此调整，应将它们作为诊断基准，另冻结未参与调优的源样本/模板组合用于确认评估。训练和测试的实现形式可以区分，但 MR 的操作、relation-family 和组合顺序必须保持明确一致。

3. **考虑合法反例与源标签覆盖。** 同一表面线索在不同上下文中应能对应多个标签；这些反例必须具有真实语义依据。可补充独立注明来源的 ordinary-NLI 控制样本，或提高既有 INV 中 C/N 源的有效覆盖。不能强行给固定 E→N/E→C 的 MR 分配其他标签，也不能靠无条件提高拒绝率获得漂亮数字。

4. **诊断与质量同步设门槛。** 比较相同源组、训练预算和标签分布下的 hypothesis-only raw/masked、表层探针、原始测试、逐标签和联合指标；同时报告语义审核、MR fidelity、各 MR 接受率与覆盖。hypothesis-only 下降而语义/覆盖恶化不能算成功。

5. **保留当前版本。** 新生成数据使用独立目录、新合同版本与完整哈希/日志；保留 v3.3 训练和 v1.1 测试作为固定基准。先做小规模代表性试验，审阅之后再考虑全量重生成。

补充报告 [EXPLANATIONS_zh.md](EXPLANATIONS_zh.md) 给出表层探针的源组 bootstrap 区间、逐标签长度差异、高权重词特征与两条待复核的无实质文本变化样本。

目前证据可支持“单句表面线索很强、与生成合同一致”。它不能单独证明正式 Gemma 模型具体依赖了哪条捷径，也不能证明全部标签在语义上正确。

## 8. 输出与复现

- `metrics.json`：数据统计、线索纯度、重合、逐 seed 与逐 MR 指标、混淆矩阵及 bootstrap。
- `splits.json`：三 seed 的训练/验证源组，供复现并核对分组隔离。
- `shortcut_accuracy.png` / `.pdf`：诊断概览图。
- `explanatory_probes.json` / `EXPLANATIONS_zh.md`：冻结划分重放及表层/词线索解释。
- `audit_script_at_probe_run.py`：主探针执行时的脚本快照，其 SHA-256 对应 metrics.json 的 script_sha256。
- `audit_nli_shortcuts.py`：完整计算脚本位于上一级目录。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/ubuntu/anaconda3/bin/python analysis/audit_nli_shortcuts.py \
  --output analysis/shortcut_audit_new_run
```

补充解释可在主诊断完成后运行：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/ubuntu/anaconda3/bin/python analysis/explain_nli_shortcuts.py \
  --output analysis/shortcut_audit_new_run
```

环境：Python 3.11.3；scikit-learn 1.2.2；NumPy 1.24.3。
输入哈希复核：True。耗时：94.7 秒。
