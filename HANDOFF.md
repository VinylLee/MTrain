# MetTrain 增效数据 —— 交接文档（给接手的 agent）

> 最后更新：2026-09-18
> 工作仓库：`/home/ubuntu/MTrain`（**生成端**）
> 配套文档：`AUGMENTATION_PIPELINE.md`（生成流程 + 提示词逐字摘录，写论文时用）
> 本文档覆盖：SNLI v3.3 增效数据的生成、溯源字段 `component_mrs` 的回填、以及**全部开放任务**

---

## 0. 先读这一节

**一句话现状**：SNLI v3.3 增效数据已生成、`component_mrs` 溯源字段已回填、生成端仓库已推送 GitHub、生成流程文档已写完。
**但 `component_mrs` 目前只到 RQ2，还没到 RQ1**；数据配比欠配的问题仍未解决。

### 开放任务（按建议优先级）

| # | 任务 | 主要落点 | 代价 / 阻塞 |
|---|---|---|---|
| **A** | 让 **RQ1/NLI** 用上 `component_mrs` | `LLMTrain`（生成端无需改） | 改下游渲染代码 + 重跑 sample/convert；**不消耗 GPU**，但会牵动 shuffle 与校验器 |
| **B** | 修 **MR 配比欠配**（synonym/antonym） | `MTrain`（需重新生成） | 约 **18 小时** GPU；会产出与现有数据文本不同的新数据集，**现有 RQ1/RQ2 结果全部作废** |
| **C** | 生成端文档与工具维护 | `MTrain` | 无阻塞，随时可做 |

### 三条铁律

1. **不要动 `PROMPT_VERSION` / `OPERATION_SPECS` / `COMPOSITE_STAGE_PLANS`**（除非在做任务 B 且明确知道后果）。
   改这三个中任何一个，`config_fingerprint` 就会变，既有数据与已完成的实验全部失去可比性。
2. **不要删 `*.generation_log.jsonl`**。它是 `component_mrs` 的**唯一来源**，丢了就再也无法回填或复核。
3. **生成不可复现**。`temperature=0.65` 且不传 server 端 seed，重跑一次得到的文本完全不同。任何"重新生成"都是新数据集，不是修复。

---

## 1. 仓库地图

| 仓库 | 角色 | 远端 | 状态 |
|---|---|---|---|
| `/home/ubuntu/MTrain` | **生成端**：MR 增效数据生成器 | `git@ssh.github.com:VinylLee/MTrain.git` | 已推送，main = `c422dcb` |
| `/home/ubuntu/LLMTrain/LLMTrain` | **实验端**：采样 / 转换 / LoRA 训练 / 评测 | `git@github.com:VinylLee/LLMTrain.git` | 见该仓库 `AGENTS.md` |

**数据流**

```
MTrain/augment_snli_all_label_mrs.py          ← 生成（Gemma-3-4B-IT-QAT，本地 OpenAI 兼容服务）
        │
        ▼  nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/
           ├─ augmented_data_all_label_mrs_v3_3_full.json      ← 数据（5340 行）
           ├─ ...v3_3_full.generation_log.jsonl                ← 逐次尝试的完整证据（唯一溯源）
           ├─ ...v3_3_full.report.json                         ← 运行配置 + 哈希 + 指纹
           ├─ ...v3_3_full.quality_report.json / .quality_review.jsonl   ← 独立质量审计
           └─ ...v3_3_full.component_mrs_backfill.json         ← 回填记录
        │
        ▼  cp（人工）
   LLMTrain/data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/
        │
        ├─► RQ1/NLI：sample_mettrain_pairid.py → convert_nli_to_ft.py → LoRA
        │            ⚠️ 当前 sampled.json / full_train.json 里 component_mrs = 0 条
        │
        └─► RQ2：RQ2/data/cohorts_v4/ → converted_v4/ → grounded_v4
                     ✅ 已消费（ordered transformation trace）
```

---

## 2. 已完成的工作（2026-09-15 ~ 09-18）

1. **`component_mrs` 溯源字段回填**。composite MR（`composite_inv`/`composite_flip`/`composite_neutral`）由固定 plan 表把两个原子 MR 串行组合，但"由哪两个、按什么顺序"从未写进输出数据。该信息原本存在于 `generation_log.jsonl` 的 `applied_steps` 字段，本次用 `backfill_component_mrs.py` 回填进数据，**零 LLM 调用**。
   - 486 条 composite 行获得该字段；其余 **4854 行逐字节未变**
   - `2bc38a5b…` → `648946c1…`
   - 回填前原件保留为 `..._full.pre_component_mrs_20260915.json`
2. **生成器改造**：`synthetic_record()` 支持该字段（插在 `mr_id` 之后、`mr_type` 之前，**仅 composite 行携带**）；`report.json` 补齐运行溯源（argv/cwd/effective_args/host/port/operations + 4 个文件 sha256 + `config_fingerprint`）。
3. **新增共享模块** `provenance_schema.py`（纯标准库）与 **32 个测试**（`tests/`）。
4. **仓库推送**：`.gitignore` 忽略 `original_dataset/`、`mr_test_data*/`（可重生成）；`nli_arg_data/` **刻意纳入版本控制**。
5. **文档**：`AUGMENTATION_PIPELINE.md`（生成流程与提示词逐字）。

**MTrain 提交历史**

```
c422dcb  docs: 增效数据生成流程说明（供论文方法部分参考）
c8b534b  feat: composite MR 溯源字段 component_mrs + 生成过程记录
fde2e8d  perf(MRAugment): 新增原子 MR 的批量生成/验证路径   ← 更早的未提交改动
```

---

## 3. 数据现状（可直接核查）

**文件**：`nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data_all_label_mrs_v3_3_full.json`（JSONL，5340 行）

| 项 | 值 |
|---|---|
| sha256 | `648946c122110d808818df5cef9166602c9166b660c9c8a7719ecde4c010ab1a` |
| 字节 / 行数 | 1,370,347 / 5,340（= 2,033 源行 + 3,307 合成行） |
| 回填前原件 | `..._full.pre_component_mrs_20260915.json`，sha `2bc38a5b…`，1,341,235 字节 |
| generation log | sha `8eb92745…`，12,535 行，24,552,243 字节（**未改动过**） |
| 带 `component_mrs` 的行 | **486**（全部为 composite 行） |
| `config_fingerprint` | `3f7ed9ff3cd82904ebc6f6d5a711f212397a7fb6c25d330a886be42625cf74c0` —— **记在 `..._full.component_mrs_backfill.json` 的 `generator_report` 里**，来源是 `ast_snapshot_of_generator_source`（对改造前的生成器源码 `53002411578a2bfe…` 取 AST 快照算出来的）。 |

> ⚠️ **`.report.json` 里没有 `config_fingerprint`**。那份报告是 2026-09-11 那次运行写的，
> 而该字段是 09-15 才加进生成器的 —— **只有改造后新跑的运行才会自带它**。
> 判断"这批数据是不是当前这套 MR 规格产出的"，在 v3.3 上要读回填报告，不能读 `.report.json`。

**486 条 composite 的组件组合（有序）**

| component_mrs | 条数 |
|---|---|
| pronoun_substitution + negation_flip | 248 |
| pronoun_substitution + uninformative | 177 |
| voice_switch + uninformative | 20 |
| pronoun_substitution + synonym_replacement | 18 |
| synonym_replacement + uninformative | 11 |
| synonym_replacement + pronoun_substitution | 4 |
| voice_switch + adding_contradiction | 4 |
| synonym_replacement + negation_flip | 3 |
| voice_switch + synonym_replacement | 1 |

**`component_mrs` 的语义（已逐条验证）**

- **有序**：从左到右 = 从先到后。486/486 条满足 `component_mrs[i] == 日志中第 i 个执行阶段的 operation`。
- **逐级传递**：第二阶段作用在**第一阶段的输出**上，不是源 hypothesis 上。
- 486 条里两个阶段**都真实改动了文本**，不存在顺序空转。
- 反向顺序是**不同的东西**：`[pronoun, synonym]` 18 条 vs `[synonym, pronoun]` 4 条。

**逐 MR 接受率（`--all`, seed 42）**

| MR | attempted | accepted | 接受率 |
|---|---|---|---|
| uninformative | 1,308 | 1,150 | 0.8792 |
| conditional_clause | 739 | 565 | 0.7645 |
| negation_flip | 717 | 480 | 0.6695 |
| composite_flip | 723 | 255 | 0.3527 |
| composite_neutral | 739 | 208 | 0.2815 |
| adding_contradiction | 723 | 180 | 0.2490 |
| pronoun_substitution | 2,033 | 355 | 0.1746 |
| voice_switch | 1,119 | 45 | 0.0402 |
| antonym_substitution | 368 | 13 | 0.0353 |
| synonym_replacement | 2,033 | 33 | 0.0162 |
| composite_inv | 2,033 | 23 | 0.0113 |

**质量门**（`audit_augmentation_quality.py`，judge-sample 110）

| 指标 | 值 | 门槛 | |
|---|---|---|---|
| label_accuracy | 0.9909 | ≥0.95 | ✅ |
| grammatical_rate | 1.0000 | ≥0.95 | ✅ |
| **inv_source_equivalence_rate** | **0.9000** | ≥0.95 | ❌ |
| fully_accepted_rate | 0.9545 | ≥0.90 | ✅ |

未达标项里最弱的是 composite：`composite_inv` 0.8、`composite_neutral` 0.9。

---

## 4. 开放任务 A：让 RQ1/NLI 消费 `component_mrs`

### 动机

这是当初加这个字段的**原始动机**。RQ1 的 MR-as-Instruction 有两种模式需要操作描述：

- `relation_aware` / `full_oracle`（旧命名）→ 现在是 RQ2 的 `pair_relation` / `full_specification` 等

composite 样本此前只能用含糊描述（`"Several controlled textual edits were applied."`），导致这两种模式对 composite 样本**没有实际区分度**。RQ2 已经改成由 `component_mrs` 展开的有序 trace，**RQ1/NLI 还没有**。

### 现状证据

```bash
# 都是 0
data/ft_datasets/rq1_mr_snli_v3_3_gemma-3-4b-it_seed42_none/sampled.json      # 5340 行，0 条带 component_mrs
data/ft_datasets/rq1_mr_snli_v3_3_gemma-3-4b-it_seed42_none/full_train.json   # 5102 行，0 条
```

原因：这些文件是**回填之前**抽样/转换出来的快照。字段在源数据里，但没流下去。

### 做法

1. **重跑 sample + convert**（`sample_mettrain_pairid.py` 会逐字保留整行，所以 `component_mrs` 能直接传下去；`convert_nli_to_ft.py` 只输出 `{instruction, input, output}` 三个键，字段**不会**进 `full_train.json`——这是刻意的，别改）。
2. **改描述渲染**：`scripts/convert_nli_to_ft.py` 的 `MR_OPERATION_DESCRIPTIONS`（composite 三条是含糊描述）。渲染点在这里：

   ```python
   # convert_nli_to_ft.py:730-737
   operation_desc = ""
   if mode in MODES_REQUIRING_OPERATION:
       operation_desc = operation_descriptions.get(effective_mr_id, "")
   ```

   要在这里按行取 `s.get("component_mrs")` 展开成有序 trace，而不是按 `effective_mr_id` 查表。

### 坑（务必先读）

- **`shuffled_operation` 模式假定"描述是 `mr_id` 的纯函数"**：`select_shuffled_descriptions`（L596–628）按**描述字符串**做 derangement，并有固定点检查；`build_shuffle_audit`（L631–672）的混淆矩阵也以描述文本为键。描述变成逐行相关后，这两处都要重新设计。
- **校验器有字面量断言**：`scripts/validate_mrinstr_conversion.py:503/525/531` 断言 `MR_OPERATION_DESCRIPTIONS[mr_id] in instruction`。composite 改成组件展开后这三条会失败。
- **不要破坏 `data_signature`**：`convert_nli_to_ft.py:796` 的 `compute_data_signature` 用白名单 `("pair_id","mr_id","premise","hypothesis","label","idx","id","_source")`——**不含 `component_mrs`**。这是好事（既有切分清单继续有效），但如果有人"顺手"把新字段加进白名单，所有 `conversion_report.json` 的 `data_signature` 会集体失配。
- **RQ2 已有实现可参考**：`RQ2/RQ2_INSTRUCTION_DESIGN_V4.md`、`scripts/audit_rq2_v4_provenance.py`。照它的 trace 格式做，能保持两个 RQ 的措辞一致。

---

## 5. 开放任务 B：修 MR 配比欠配

### 证据

| MR | SNLI 训练（占 MR 行） | SNLI MR 测试（占 MR 行） | 配比 |
|---|---|---|---|
| synonym_replacement | **33 (1.0%)** | 1,856 (14.6%) | **0.07×（14.6× 欠配）** |
| antonym_substitution | **13 (0.4%)** | 1,471 (11.6%) | **0.03×（29× 欠配）** |
| uninformative | 1,150 (34.8%) | 2,601 (20.4%) | 1.7× 过配 |
| conditional_clause | 565 (17.1%) | 1,781 (14.0%) | 1.22× |
| negation_flip | 480 (14.5%) | 2,384 (18.7%) | 0.78× |
| pronoun_substitution | 355 (10.7%) | 1,332 (10.5%) | 1.02× |

**composite 的组件分布同样欠配**：486 条里 `synonym_replacement` 作为组件出现 33 次，`antonym_substitution` **0 次**。

### 两条路径

| 路径 | 做法 | 代价 |
|---|---|---|
| **B1 定向补采** | 只对 `synonym_replacement` / `antonym_substitution` 跑 refill（用 `--operations` 限定 + 提高 `--min-total-rows`） | 仍需 LLM 调用，但不重跑全部 11 个 MR |
| **B2 重新生成** | 调低这两个 MR 的过滤门槛或提高其尝试次数后全量重跑 | **~18 小时**；产出新数据集，**现有 RQ1/RQ2 结果全部作废** |

### 为什么这两个 MR 接受率极低

- `synonym_replacement` 0.0162：确定性门要求内容词对称差 ≤ 2（几乎只能动 1–2 个词），而 Gemma 倾向整句改写。
- `antonym_substitution` 0.0353：要求**恰好一个 token 差异**且该差异对在 32 组人工白名单里（`ANTONYM_PAIRS`，L68–83）。

**因此"提高尝试次数"没用**——瓶颈是门的严格度与模型行为不匹配。要真正解决，得改门或改 instruction（属于**任务 B2，且会改变生成契约**）。

> 关联：SA 任务也撞上同一堵墙——记忆里的 "SA MR 保真度失败" 显示 Gemma-3-4b 不做受控最小编辑而做自由改写，8 个 SA MR 里 6 个 transformation validity 为 0%。**这是同一个根因**，两个任务可以共用结论。

---

## 6. 开放任务 C：生成端文档与工具维护

- `README.md` 仍描述 8 月的旧管线（`new_run.py` / `NLImain.py` / RTE 示例），**0 处**提及 `augment_snli_all_label_mrs.py` / `component_mrs` / v2-v3 工作。已在本次加入指向本文件与 `AUGMENTATION_PIPELINE.md` 的入口，但旧章节尚未重写。
- `MRAugment.py` 里有一批**死代码**：`composite_mr_inv/flip/neutral` 用 `random.randint`/`random.sample` 随机组合，且引用了未定义的 `METAMORPHIC_RELATIONS`。实际 v3 流程**完全不用**这些方法（只用 `NLIMRTool.LABEL_MRS` 做一致性校验）。清理前先确认没有别的入口在调。
- `augment_snli_v2.py`（前身，3 个粗粒度操作 `*_v2`）与 `audit_augmentation_quality.py` 有 import 关系（audit 复用 v2 的 helper），**不要直接删 v2**。

---

## 7. 工具速查

全部在 `/home/ubuntu/MTrain`。**默认 `python3` 没有 `openai`**；需要 import 生成器时用 `/home/ubuntu/anaconda3/bin/python`。

| 脚本 | 用途 | 命令 |
|---|---|---|
| `augment_snli_all_label_mrs.py` | 增效数据生成器 | 见 §7.1 |
| `backfill_component_mrs.py` | 回填 `component_mrs`（纯标准库） | `python3 backfill_component_mrs.py --check` → 去掉 `--check` 执行 |
| `audit_augmentation_quality.py` | 独立质量审计 | 见 §7.2 |
| `provenance_schema.py` | 共享溯源工具（被上面两个 import） | — |
| `tests/` | 32 个测试 | `/home/ubuntu/anaconda3/bin/python -m pytest tests/ -q` |

### 7.1 复现 v3.3_full 的生成

```bash
# 需要本地 OpenAI 兼容服务（LM Studio / llama.cpp）在 127.0.0.1:1234，且已加载 gemma-3-4b-it-qat
python -u augment_snli_all_label_mrs.py \
  --all --seed 42 \
  --generator-model gemma-3-4b-it-qat --judge-model gemma-3-4b-it-qat \
  --output nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data_all_label_mrs_v3_3_full.json \
  --candidate-count 3 --retries 2 --workers 8 \
  --min-total-rows 5340 --max-fill-rounds 4 --checkpoint-every 10 --resume
```

⚠️ 这是**历史命令的复原**（原先只存在于 codex 会话日志），实测耗时 **65,437 秒 ≈ 18.2 小时**。重跑会得到**不同的文本**。

### 7.2 质量审计

```bash
/home/ubuntu/anaconda3/bin/python audit_augmentation_quality.py \
  --source nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/labeled_data.json \
  --data   nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data_all_label_mrs_v3_3_full.json \
  --judge-sample 110 --batch-size 8 --seed 42 \
  --report /tmp/audit.json --review-output /tmp/audit_review.jsonl
```
（不给 `--report` 会覆盖在用的 `.quality_report.json`，别那么干。）

---

## 8. 已知陷阱（都踩过）

| 陷阱 | 说明 |
|---|---|
| **生成不可复现** | `temperature=0.65`，不传 server 端 seed。`--seed` 只控制"取哪些源样本"。 |
| **`--resume` 不支持样本内断点** | 以"某 `pair_id` 是否已有非合成行"判断完成，一旦写了源行就整条跳过。中断可能留下不完整的样本。 |
| **`--overwrite` 与 `--resume` 互斥**，且两者都不给而输出已存在 → 直接报错。 |
| **跑完生成后必须重跑回填** | 改造后的生成器会原生写 `component_mrs`，但**若用 `--resume` 续跑旧文件**，旧行仍是 8 键、新行 9 键 → 混杂 schema。回填脚本幂等，跑一遍即可补齐。 |
| **`count_dataset_tokens.py` 不识别 `token_matched_*.json`**（LLMTrain 侧） | 它只 glob `full*`，要统计得显式给路径。 |
| **回填的 `--force` 语义** | 仅在"已有值与日志不符"时需要；它会把该行**规范化**回正确键序。默认不加 `--force` 时遇到不符会中止（fail-closed）。 |
| **回填备份不能重复创建** | 备份保存的是**回填前**状态。二次运行时数据已变，脚本按"零改动则不动备份"处理；若手工删了备份再跑，会以当前（已回填）内容做备份 → 误导。 |
| **GitHub push 很慢且抖动** | 见记忆 `git-push-github-port443`。给每次 `git push` 留 ≥60s，别用短超时探测就断定 SSH 挂了。 |

---

## 9. 铁律（重复一次，因为很贵）

1. **不碰 `PROMPT_VERSION` / `OPERATION_SPECS` / `COMPOSITE_STAGE_PLANS`**（任务 B 除外，且要明白代价）。
2. **不删 `*.generation_log.jsonl`**——`component_mrs` 的唯一来源。
3. **不把 `component_mrs` 加进下游 `compute_data_signature` 的白名单**——会让所有既有 `conversion_report.json` 的 `data_signature` 失配。
4. **不把 `component_mrs` 写进 `full_train.json`**——训练文件只应有 `{instruction, input, output}`，加字段会改变 LLaMA-Factory 的输入 schema。
5. **`original_dataset/` 与 `mr_test_data*/` 不要提交**（已 ignore，可重新下载/生成）。

---

## 10. 相关文档索引

**生成端（本仓库）**

| 文档 | 内容 |
|---|---|
| `HANDOFF.md` | 本文件：状态、开放任务、工具、陷阱 |
| `AUGMENTATION_PIPELINE.md` | 生成流程逐节说明 + **全部提示词逐字摘录** + 源码行号索引（写论文用） |
| `README.md` | 旧版项目说明（RTE 示例管线）；顶部有指向本文件的入口 |
| `nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/*.report.json` | 每次运行自己的记录（全部超参 + 文件哈希 + argv/cwd）；**09-15 之后的运行才有 `config_fingerprint`** |
| `...*.component_mrs_backfill.json` | 回填记录（18 项校验、join 统计、组件分布、**v3.3 的 `config_fingerprint` 在这里**） |

**实验端（`/home/ubuntu/LLMTrain/LLMTrain`）**

| 文档 | 内容 |
|---|---|
| `AGENTS.md` | 项目记忆与规则（**接手实验端前先读**） |
| `PROJECT_OVERVIEW.md` | 仓库架构、数据流、脚本职责 |
| `RQ2/RQ2_INSTRUCTION_DESIGN_V4.md` | **已消费 `component_mrs` 的实现**，任务 A 可照抄其 trace 格式 |
| `RQ1/SA_MR_PIPELINE_HANDOFF.md` | SA 任务交接（同源问题：Gemma 自由改写） |
