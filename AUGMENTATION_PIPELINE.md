# MetTrain SNLI 增效数据生成流程（v3.3）

> 本文档描述 `/home/ubuntu/MTrain/augment_snli_all_label_mrs.py` 的完整生成流程，
> 供论文方法部分写作参考。所有提示词均为源码逐字摘录，标注了行号，便于核对。
>
> 生成器：`augment_snli_all_label_mrs.py`（`PROMPT_VERSION = "snli-pair-aware-all-label-mrs-v3.3"`）
> 数据：SNLI → `nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data_all_label_mrs_v3_3_full.json`
> 溯源字段 `component_mrs` 于 2026-09-15 由 `backfill_component_mrs.py` 回填，见第 10 节。

---

## 1. 概览

MetTrain 用**蜕变关系（MR）作为训练期数据生成规则**：给定一条已标注的 NLI 样本（premise + hypothesis + label），按某个 MR 对 hypothesis 做受控改写，产出一条新的 premise–hypothesis–label 三元组。改写由一个小型冻结 LLM（Gemma-3-4B-IT-QAT）执行，并由**第二个独立的 LLM 请求**进行判题，只有通过判题才被接受。

```
labeled_data.json  (2033 条 SNLI entailment/neutral/contradiction)
        │
        ▼
 [0] 采样：全量或按标签分层抽样（seed 控制）
        │
        ▼
 对每条源样本 × 每个适用的 MR（eligibility 门控）
        │
        ├─ 原子 MR ─────────────────────► [A] 生成 → [B] 确定性过滤 → [C] 判题 → [D] 选择
        │                                                                      │
        └─ composite MR ──► 两阶段串行：每个阶段跑一遍 [A][B][C][D]            │
                            再对整个结果补一次「终审判题」                      │
                                                                               ▼
                                                                   接受 → synthetic 行
                                                                   拒绝 → 丢弃（重试 retries 次）
        │
        ▼
 [E] Refill：若未达 --min-total-rows，按固定优先级补采若干轮
        │
        ▼
 输出  augmented_data_all_label_mrs_v3_3_full.json（JSONL）
       + .generation_log.jsonl（逐次尝试的完整证据）
       + .report.json（逐 MR 接受率与运行配置）
```

**关键设计点（论文可用）**

| 设计 | 说明 |
|---|---|
| **Pair-aware prompt** | premise、源 hypothesis、源 label、目标 label 全部进入生成提示词，避免"只看到 hypothesis 就改写"导致的语义漂移 |
| **独立判题** | 判题使用一个独立的、无状态的请求（temperature=0），不看生成过程，只看结果 |
| **生成与判题模型相同但调用独立** | 同为 `gemma-3-4b-it-qat`，但判题不得复用生成的上下文 |
| **确定性门 + LLM 判题双层过滤** | 先跑廉价的正则/词表门，挡掉明显捷径，再交给 LLM 判断 |
| **标签由 MR 语义决定，不由模型决定** | `target_label` 由 `OPERATION_SPECS[op]["target"]` 查表得到，模型只在候选里声明，不参与决定 |
| **composite 显式串行** | composite MR 由固定 plan 表定义，两阶段各自独立调用 LLM，第二阶段输入 = 第一阶段输出 |

---

## 2. 输入

| 项 | 值 |
|---|---|
| 输入文件 | `nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/labeled_data.json` |
| SHA-256 | `441e72a3c2499cf896e699d541e1e772d2ae77075605516b88f8a00bc90c646e` |
| 规模 | 2033 条 SNLI 样本 |
| 字段 | `premise`, `hypothesis`, `label`(0=entailment/1=neutral/2=contradiction) |
| 标签分布 | entailment 739 / neutral 644 / contradiction 650 |

---

## 3. 采样

`stratified_sample()`（L356–374）。`--all` 时全量使用；否则按标签分层抽样：

1. 按 label 分桶，用 `random.Random(seed)` 各自打乱；
2. 每桶取 `size // 3` 条，余数依次分配给按 label 排序靠前的桶；
3. 合并后再整体打乱。

**seed 只影响"取哪些源样本"，不影响 LLM 生成**——生成端 temperature=0.65 且不传 server 端 seed，因此**生成结果不可复现**，只有源集合可复现。这一点在论文里若要声明可复现性需要说明。

---

## 4. MR 目录（11 个）

`OPERATION_SPECS`（L105–209），共 **8 个原子 MR + 3 个 composite MR**。每个 MR 由 `mr_type`（inv/flip/neutral）、`family`、`instruction`（生成指令）、`target`（目标标签）四要素定义。

| MR | family | target | 语义 |
|---|---|---|---|
| `negation_flip` | flip | contradiction | 对整个 hypothesis 做最小语法改动的直接否定 |
| `antonym_substitution` | flip | contradiction | 替换恰好一个关键词为其反义词 |
| `adding_contradiction` | flip | contradiction | 复制 hypothesis 后追加一个与 premise 相矛盾的短句 |
| `pronoun_substitution` | inv | *同源标签* | 把一个无歧义名词短语替换为代词 |
| `voice_switch` | inv | *同源标签* | 主被动转换 |
| `synonym_replacement` | inv | *同源标签* | 替换一到两个常用词为同义词 |
| `uninformative` | neutral | neutral | 追加一个 premise 既不支持也不排除的具体细节 |
| `conditional_clause` | neutral | neutral | 追加 "only if + 一个前提未声明的必要条件" |
| `composite_inv` | inv | *同源标签* | 两个保义变换的组合 |
| `composite_flip` | flip | contradiction | 先一个保义变换，再一个决定性翻转 |
| `composite_neutral` | neutral | neutral | 先一个保义变换，再加一个未支持从句 |

> `target = "source"` 表示目标标签 = 源样本自身的标签（`target_label()`，L381–383）。
> 即 inv 类 MR 保标签，flip 类把 entailment 翻成 contradiction，neutral 类把 entailment 弱化为 neutral。

各 MR 的 `instruction` 原文见**附录 A**。这些文本会整段插入生成提示词的 `TASK` 段。

### 4.1 适用性门控 `eligible()`（L386–408）

在生成之前先判断某条源样本是否适用某 MR，不适用则直接跳过（不消耗 LLM 调用）：

- **inv 家族**：恒适用，但 `voice_switch` 要求 hypothesis 含明确的"动作/助动词"结构（`has_voice_predicate()`，L269–283，用词表 + `-ing/-ed/-en` 后缀启发式）；
- **flip / neutral 家族**：**只作用于 entailment 源样本**（`label != 0 → False`）；
- flip 家族额外要求 `is_safe_flip_source()`（L263–267）：源 hypothesis 中出现的强量化词（all/every/each/many/most/few/several）必须在 premise 里也出现，否则可能因量化范围问题产生错误的"矛盾"；
- `negation_flip` / `antonym_substitution`：源 hypothesis 已含否定则跳过；`antonym_substitution` 另要求源不以 `a/an/some/someone/somebody/one` 开头（存在量词取反会改变语义重心）。

---

## 5. 阶段 A：生成

`_generate_atomic()`（L752–808）中的第一次调用。

| 项 | 值 |
|---|---|
| 模型 | `gemma-3-4b-it-qat`（`ALLOWED_MODEL`，L45，硬校验：生成与判题都必须是它） |
| system prompt | `GENERATOR_SYSTEM_PROMPT`（L88–96） |
| user prompt | `build_generation_prompt()`（L411–470） |
| temperature | **0.65** |
| max_tokens | 420 |
| 候选数 | `--candidate-count`，默认 3 |

### 5.1 System prompt（逐字）

```
You are a meticulous SNLI data editor.
Create short, natural English hypotheses for natural-language inference.
Treat the premise as the only known world. Use the strict SNLI definitions:
- entailment: the premise makes the hypothesis definitely true;
- contradiction: the premise makes the hypothesis definitely false;
- neutral: the hypothesis may be true, but the premise neither proves nor
  disproves it.
Do not rely on ordinary-world assumptions to call something a contradiction.
Do not explain your reasoning. Return only the requested JSON array.
```

### 5.2 User prompt 模板（逐字）

`{spec['instruction']}` 处插入第 4 节对应 MR 的 instruction 原文；`{...}` 为变量。

```
TASK
{spec['instruction']}

PAIR
{
  "premise": "<premise>",
  "source_hypothesis": "<源 hypothesis>",
  "source_label": "entailment",
  "target_label": "<目标标签名>"
}

OUTPUT CONTRACT
Return a JSON array containing up to 3 distinct objects.
Each object must have string fields "hypothesis" and "target_label".
For adding_contradiction it must also have the string field "evidence_fact";
other MRs must not add fields.
{"hypothesis": "...", "target_label": "<目标标签名>"}
The hypothesis must be one self-contained English statement with no labels,
quotes, commentary, prefixes, or prompt text. Never copy the premise verbatim.
{neutral_contract}

SNLI PITFALLS
- An extra fact absent from the premise is neutral, not contradiction.
- A less specific statement may be entailed but is not an exact paraphrase.
- Changing person, number, color, location, direction, or polarity is forbidden
  unless the named MR explicitly requires that exact change.
- A flip must negate a fact guaranteed by the premise; difference alone is neutral.
- A neutral edit must add uncertainty or one unsupported plausible fact without
  deleting the supported source event.
- Perform the named MR only. A valid target label does not excuse the wrong edit.
```

### 5.3 三种 MR 的额外格式约束 `neutral_contract`（L424–447）

对 `uninformative` / `adding_contradiction` / `conditional_clause` 三个 MR，强制模型输出以源 hypothesis 原文为前缀，只允许追加一个新从句——这是为了让确定性门能按固定连接词切分：

| MR | 强制连接词 | 追加到提示词的内容 |
|---|---|---|
| `uninformative` | `, and ` | `UNINFORMATIVE FORMAT: Every hypothesis must begin exactly with:` + `"<源hypothesis去尾标点>, and "` + `Then complete exactly one new clause; do not paraphrase the prefix.` |
| `adding_contradiction` | `, but ` | 同上，标签为 `ADDING_CONTRADICTION`，另加 `evidence_fact` 字段要求 |
| `conditional_clause` | ` only if ` | 同上，标签为 `CONDITIONAL_CLAUSE`（无 evidence_fact） |

`adding_contradiction` 额外要求输出对象含 `evidence_fact` 字段——即被追加从句所否定的、**premise 中的最短连续原文引文**，提示词里给了例子：

```
Each output object must also contain evidence_fact: the shortest
exact contiguous quote from the PREMISE that the added clause
explicitly negates. Example: premise 'Two dogs are jumping in a
field', source 'The dogs are outside' -> hypothesis 'The dogs are
outside, but the dogs are not jumping', evidence_fact 'dogs are
jumping'. Do not quote the whole premise.
```

---

## 6. 阶段 B：确定性过滤（`parse_candidates`，L530–699）

在调用 LLM 判题**之前**先用廉价规则挡掉不合格候选。这是本流程与"直接让 LLM 生成即用"的主要区别。

**格式与去重**

- 必须是 JSON 数组；截断最多 `candidate_count` 个元素；
- 每个对象须含 `hypothesis`（str）与 `target_label`，且 `target_label` 必须等于期望目标标签（大小写不敏感）；
- 空白归一化后去重，且不得等于源 hypothesis 或 premise；
- 词数 `3 ≤ n ≤ max(30, len(源hypothesis词数)*2 + 8)`。

**内容重叠下界**：候选与源 hypothesis 的内容词交集须达到下限（内容词 = 去掉停用词与否定词）：

| MR | 下界 |
|---|---|
| `negation_flip` | ⌈3/4 · |源内容词|⌉ |
| `antonym_substitution` | \|源内容词\| − 1（要求几乎完全一致） |
| 其他 | ⌈|源内容词|/2⌉ |

**MR 专属门（拒绝"走捷径"的候选）**

| MR | 检查 |
|---|---|
| `uninformative` | 必须以 `<源>norm, and ` 开头；后缀内容词须含 premise 未出现的词；后缀不得使用与 premise 性别不符的代词 |
| `adding_contradiction` | 必须以 `<源>norm, but ` 开头；`evidence_fact` 必须是 premise 的连续子串；后缀须含否定；后缀内容词须与 evidence 内容词有足够重叠且**不得有 evidence 之外的内容词** |
| `conditional_clause` | 必须以 `<源>norm only if ` 开头；后缀须引入 premise 未出现的新内容；后缀不得含模糊条件词（may/maybe/perhaps/possibly 等） |
| `pronoun_substitution` | 代词数量必须**增加**；不得引入源中不存在的新名词短语；被替换掉的名词短语不超过 2 个 |
| `voice_switch` | `by` 的出现与否必须在源与候选间**反转**（判断主被动确实切换了） |
| `synonym_replacement` | 内容词对称差 ≤ 2（即只动了很少的词） |
| `negation_flip` | 须通过 `is_direct_negation()`（L245）——最短编辑式否定校验 |
| `antonym_substitution` | 不得含否定；数字 token 集合须不变；须通过 `is_whitelisted_antonym_substitution()` |

**inv 家族全局约束**：数字 token 集合不变 **且** 否定状态不变（保义）。

**通用"污染"正则**：命中即丢弃——`premise:` / `hypothesis:` / `target_label` / `source_label` / 三反引号 / `<think>` / `1 are|1 were|1 have`（主谓数不一致）/ `likely|maybe|perhaps|probably|possibly`（模糊限定）/ 数字短语作主语。

> `ANTONYM_PAIRS`（L68–83）是 32 组人工白名单反义词对（inside/outside、up/down、open/closed……），`antonym_substitution` 要求**恰好一个 token 差异**且该对在白名单内。这是 11 个 MR 里通过率最低的原因之一（0.0353）。

---

## 7. 阶段 C：LLM 判题

`_generate_atomic()` 中的第二次调用，使用**独立**的 client 与请求。

| 项 | 值 |
|---|---|
| 模型 | `gemma-3-4b-it-qat` |
| system prompt | `JUDGE_SYSTEM_PROMPT`（L98–103） |
| user prompt | `build_judge_prompt()`（L473–507） |
| temperature | **0.0** |
| max_tokens | 520 |

### 7.1 System prompt（逐字）

```
You are a strict, conservative SNLI quality judge.
Classify from the premise alone. Contradiction requires logical incompatibility,
not merely different, missing, unlikely, or newly added information. Entailment
requires the hypothesis to be guaranteed. Otherwise choose neutral.
Judge grammar and operation compliance independently. Return only JSON and do
not provide hidden reasoning; the reason field must be a short error tag.
```

### 7.2 User prompt 模板（逐字）

```
Evaluate every candidate in this payload:
{
  "premise": "<premise>",
  "source_hypothesis": "<源 hypothesis>",
  "source_label": "entailment",
  "requested_operation": "<MR 名>",
  "operation_family": "inv|flip|neutral",
  "operation_requirement": "<该 MR 的 instruction 原文>",
  "target_label": "<目标标签>",
  "candidates": [
    {"index": 0, "hypothesis": "<候选 1>"},
    {"index": 1, "hypothesis": "<候选 2>"}
  ]
}

For each candidate independently determine:
1. source_label: the strict SNLI label of premise -> source_hypothesis; do not
   trust the supplied source_label blindly (playing is not necessarily enjoying,
   setting a table is not seated, and multiple does not necessarily mean many);
2. label: the strict SNLI label of premise -> candidate hypothesis;
3. grammatical: whether the candidate is natural, self-contained English;
4. source_equivalent: whether candidate and source have identical truth conditions;
5. operation_valid: whether it performs the named MR exactly, including its edit
   count and structural constraints, with no shortcut or artifact. A correct NLI
   label alone is insufficient.

Return exactly one JSON object of this form and preserve candidate indices:
{"evaluations":[{"index":0,"source_label":"entailment","label":"entailment",
"grammatical":true,"source_equivalent":true,"operation_valid":true,"reason":"ok"}]}
Allowed reason values: ok, wrong_label, meaning_drift, grammar, artifact,
not_changed, or other. No prose outside JSON.
```

判题输出 5 个维度：`source_label`（对源的独立复核，作为审计信号）、`label`、`grammatical`、`source_equivalent`、`operation_valid`，外加 `reason` 短标签（7 个枚举值）。

---

## 8. 阶段 D：选择（`choose_candidate`，L728–747）

按候选顺序取**第一个**同时满足以下条件的候选：

1. `judgment["label"] == target_label`（标签正确）；
2. `grammatical == true` 且 `operation_valid == true`；
3. 若 `family == "inv"`，额外要求 `source_equivalent == true`（保义）。

**注意**：判题给出的 `source_label` **不用于否决**——源码注释明确说明"curated SNLI 源标签是权威的，用同一个小模型做否决会错误拒绝有效候选"，该字段只作为日志里的审计信号保留。这是论文里值得写的一个设计选择。

若无候选通过，则整次尝试记为 `judge_rejected`，进入下一次 retry（共 `--retries`，默认 2 次）。

---

## 9. Composite MR：两阶段串行流水线

### 9.1 plan 表（`COMPOSITE_STAGE_PLANS`，L810–830，逐字）

```python
COMPOSITE_STAGE_PLANS = {
    "composite_inv": (
        ("pronoun_substitution", "synonym_replacement"),
        ("voice_switch", "synonym_replacement"),
        ("synonym_replacement", "pronoun_substitution"),
    ),
    "composite_flip": (
        ("pronoun_substitution", "negation_flip"),
        ("synonym_replacement", "negation_flip"),
        ("voice_switch", "adding_contradiction"),
    ),
    "composite_neutral": (
        ("pronoun_substitution", "uninformative"),
        ("synonym_replacement", "uninformative"),
        ("voice_switch", "uninformative"),
    ),
}
```

### 9.2 执行语义（`_generate_composite`，L832–912）

- **顺序有意义**：`plan` 是有序元组，**从左到右 = 从先到后**。`component_mrs[i]` 就是第 i 次执行的原子 MR。
- **逐级传递**：`working = dict(source)` 起步；每个阶段跑完整的 `_generate_atomic`（生成 → 确定性过滤 → 判题 → 选择），成功后 `working["hypothesis"] = hypothesis`，`working["label"] = LABEL_IDS[target_label(working, stage_operation)]`。**因此第二阶段作用在第一阶段的输出上，而不是源 hypothesis 上。**
- **标签重推**：`working["label"]` 必须更新，否则第二阶段的 `eligible()` 会失败（flip/neutral 家族要求当前标签为 entailment）。
- **plan 优先级**：按 plan 表顺序尝试，**取第一个完全成功的 plan**（即两阶段都被接受、且通过终审判题）。没有任何随机选择或加权采样。
- **终审判题**：两阶段都成功后，再用一次独立判题调用评估最终结果——**这次才使用 composite MR 自己的 instruction**（生成阶段从未用过它）：

  ```python
  raw_judgment = _completion(judge_client, judge_model, JUDGE_SYSTEM_PROMPT,
                             build_judge_prompt(source, operation, [final_hypothesis]),
                             temperature=0.0, max_tokens=320)
  ```

  注意 `build_judge_prompt` 的第一个参数是**原始 source**（不是 `working`），`operation` 是 composite 名——终审是在问"这个两阶段结果相对于最初源样本，是否构成一个有效的 composite MR"。

### 9.3 成本

一次 composite 尝试 = 2 × (生成 + 判题) + 1 次终审判题 = **5 次 LLM 调用**（每个阶段内部还可能因 retries 重试），最多尝试 3 个 plan。

> **论文表述提醒**：composite 的 instruction 文本（附录 A 最后三条）只出现在**终审判题**里，生成阶段用的是各原子 MR 自己的 instruction。若在方法部分描述"composite 由 X 和 Y 两个操作组合"，这与代码一致；但不要写成"我们把 composite 的 instruction 交给模型生成"。

---

## 10. 输出与溯源

### 10.1 数据结构

输出是 **JSONL**（尽管扩展名是 `.json`）。每行 8–9 个字段：

| 字段 | 含义 |
|---|---|
| `premise` | 源 premise（**生成过程中全程不变**） |
| `hypothesis` | 改写后的 hypothesis |
| `label` | 0/1/2 |
| `pair_id` | 源样本编号（同一源的派生存共享） |
| `is_syn` | `false` = 源行（原样拷贝），`true` = 合成行 |
| `weight` | 恒 1.0 |
| `mr_id` | MR 名，源行为 `"none"` |
| `mr_type` | `inv` / `flip` / `neutral` |
| `component_mrs` | **仅 composite 行有**：2 个原子 MR 的有序列表，顺序 = 执行顺序 |

### 10.2 v3.3_full 实际产出

| 指标 | 值 |
|---|---|
| 总行数 | 5,340 = 2,033 源行 + 3,307 合成行 |
| 合成行 MR 分布 | uninformative 1150、conditional_clause 565、negation_flip 480、pronoun_substitution 355、composite_flip 255、composite_neutral 208、adding_contradiction 180、voice_switch 45、synonym_replacement 33、composite_inv 23、antonym_substitution 13 |
| composite 行 | **486 条**，9 种组合（见下） |

**486 条 composite 的实际组件组合**

| component_mrs（有序） | 条数 |
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

### 10.3 逐 MR 接受率（`--all`，seed 42）

| MR | attempted | accepted | 接受率 |
|---|---|---|---|
| uninformative | 1,308 | 1,150 | **0.8792** |
| conditional_clause | 739 | 565 | 0.7645 |
| negation_flip | 717 | 480 | 0.6695 |
| composite_flip | 723 | 255 | 0.3527 |
| composite_neutral | 739 | 208 | 0.2815 |
| adding_contradiction | 723 | 180 | 0.2490 |
| pronoun_substitution | 2,033 | 355 | 0.1746 |
| voice_switch | 1,119 | 45 | 0.0402 |
| antonym_substitution | 368 | 13 | 0.0353 |
| synonym_replacement | 2,033 | 33 | 0.0162 |
| composite_inv | 2,033 | 23 | **0.0113** |

### 10.4 溯源：`.generation_log.jsonl`

**每个（源样本 × MR）组合写一条日志**，无论接受与否。这是整套流程的完整证据链，也是 `component_mrs` 的唯一来源。

| 键 | 出现条件 | 内容 |
|---|---|---|
| `operation` / `attempts` / `status` / `pair_id` / `source_label` / `target_label` | 全部 | 基本标识 |
| `mode` | 仅 composite | `"verified_two_stage"` |
| `applied_steps` | 仅被接受的 composite | 采用的有序 plan，例如 `["voice_switch","uninformative"]` |
| `fill_round` | 仅 refill 阶段 | 第几轮补采 |
| `duplicate_hypothesis` | 生成结果与已有行重复时 | 被丢弃的 hypothesis |

`attempts[]` 内层：原子 MR 每次尝试记录 `raw_generation`（截断 3000 字符）、`candidates`、`raw_judgment`（截断 3000 字符）、`judgments`（逐候选 5 维判题）、`status`、`selected`。composite 的 `attempts[]` 每个 plan 记录 `stages`、`stage_logs`（**每个阶段的完整内层日志**）、 `status`、`failed_stage`、`raw_final_judgment`、`final_judgments`、`selected`。

> 阶段中间态可从 `attempts[-1].stage_logs[i].attempts[-1].selected` 取出——这在生成 `relation_aware` / `full_oracle` 类描述时有用（composite 的第二阶段作用在第一阶段输出上，描述里若缺中间态会少一环）。

### 10.5 溯源：`.report.json`

运行级记录。2026-09-15 起新增以下键（此前只记 argparse 参数）：

`provenance_schema_version`、`config_fingerprint`（对 `PROMPT_VERSION` + `OPERATION_SPECS` + `COMPOSITE_STAGE_PLANS` 取 SHA-256，用于证明"这批数据由当前这套 MR 规格产出"）、`argv`、`cwd`、`effective_args`、`host`/`port`/`base_url`、`operations`、`all`/`sample_size`、以及 `input_sha256`/`output_sha256`/`generation_log_sha256`/`generator_source_sha256`。

---

## 11. Refill（补采）机制

若 `--min-total-rows` 未达成，进入补采循环（`main()`，L1280–1330）：

- **固定优先级顺序**：`uninformative, conditional_clause, negation_flip, pronoun_substitution, synonym_replacement, antonym_substitution, voice_switch, adding_contradiction, composite_neutral, composite_inv, composite_flip`
- 依次遍历上述 MR × 全部源样本，逐个生成直到达标；
- 最多 `--max-fill-rounds`（默认 4）轮；
- 补采产生的行会被去重（与已有 `(normalize(premise), normalize(hypothesis))` 比对），重复则记 `duplicate_hypothesis` 并丢弃。

**v3.3_full 实际**：`fill_attempts=569`，`fill_accepted=431`，`fill_rounds_completed=1`。

> 论文提示：补采按 MR 优先级而非按比例，因此**最终的数据集 MR 配比由接受率和优先级共同决定**，不是预先设定的目标分布。这一点在报告中若要主张"配比对齐测试集"需要额外说明。

---

## 12. 质量审计（独立脚本）

`audit_augmentation_quality.py`，与生成器分离，生成后单独运行。

**确定性检查**（不调 LLM）：行数/源行/合成行计数、标签分布、MR 分布、`mr_type` 分布、精确重复合成行、重复 NLI 三元组、与源完全未变的行、缺失源 `pair_id` 的行、提示词污染（`prompt_artifacts`）、hypothesis 词数分布。

**LLM 复审**（`--judge-sample N`）：按 `mr_id` 分层抽样 N 条，用**盲判**（不告诉判题模型该行是否来自哪个 MR 的正确实现）重新评定，计算 `label_accuracy`、`grammatical_rate`、`operation_valid_rate`、`inv_source_equivalence_rate`、`fully_accepted_rate`，并给出 4 项质量门。

**v3.3_full 实测**（judge-sample 110）：

| 指标 | 值 | 门槛 |
|---|---|---|
| label_accuracy | 0.9909 | ≥0.95 ✅ |
| grammatical_rate | 1.0000 | ≥0.95 ✅ |
| operation_valid_rate | 1.0000 | — |
| **inv_source_equivalence_rate** | **0.9000** | **≥0.95 ❌** |
| fully_accepted_rate | 0.9545 | ≥0.90 ✅ |

> `inv_equivalence` 门**未通过**，且最弱的三项正是 composite：`composite_inv` 0.8、`composite_neutral` 0.9，以及 `pronoun_substitution` / `synonym_replacement` 各 0.9。若论文要报告质量门，这一项需要如实说明。

---

## 13. 复现配置与命令

**v3.3_full 的确切调用**（从 codex 会话记录恢复，此前未记录在仓库任何地方）：

```bash
python -u augment_snli_all_label_mrs.py \
  --all --seed 42 \
  --generator-model gemma-3-4b-it-qat \
  --judge-model gemma-3-4b-it-qat \
  --output nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data_all_label_mrs_v3_3_full.json \
  --candidate-count 3 --retries 2 --workers 8 \
  --min-total-rows 5340 --max-fill-rounds 4 \
  --checkpoint-every 10 --resume
```

| 参数 | 值 |
|---|---|
| 生成/判题模型 | `gemma-3-4b-it-qat`（硬校验，其它模型被拒绝） |
| 服务端点 | 本地 OpenAI 兼容服务 `http://127.0.0.1:1234/v1`（LM Studio / llama.cpp） |
| 生成 temperature / max_tokens | 0.65 / 420 |
| 判题 temperature / max_tokens | 0.0 / 520（终审 320） |
| candidate-count / retries / workers | 3 / 2 / 8 |
| seed | 42（仅控制源样本选取） |
| checkpoint-every | 10（原子写入 `<output>.json.tmp` 后 replace） |
| 实测耗时 | 65,437 秒 ≈ 18.2 小时 |

**断点续跑**：`--resume` 读取已有输出与日志，以"某 `pair_id` 是否已有非合成行"判断该源是否完成，完成后整条源样本跳过（**不支持样本内断点**）。`--overwrite` 与 `--resume` 互斥。

---

## 14. 论文写作要点与已知局限

**可用作方法贡献的表述**

1. **MR 作为训练期数据生成规则**，而非仅测试期判据；标签由 MR 语义查表决定（`target` 字段），不由生成模型决定。
2. **双层过滤**：确定性门（正则 + 词表 + 白名单反义词 + 编辑距离约束）挡掉捷径候选，再交独立 LLM 判题；判题显式分离 `label` / `grammatical` / `source_equivalent` / `operation_valid` 四个维度。
3. **inv 类的额外保义约束**：`source_equivalent` 强制成立，且数字集合与否定状态必须不变。
4. **composite 为显式两阶段流水线**，组件有序、逐级传递，接受时把组件列表作为溯源写入 `component_mrs`。

**需要在论文中如实声明的局限**

1. **生成不可复现**：temperature=0.65 且不传 server 端 seed，换一次运行就会得到完全不同的文本。
2. **`negation_flip` 缺席于早期版本**：v3.3 之前的主实验 cohort 完全没有 `negation_flip` 训练样本，而它占 SNLI MR 测试集的 18.7%。
3. **MR 配比与测试集不匹配**：`synonym_replacement` 仅 33 条（占训练 MR 行的 1.0%），`antonym_substitution` 仅 13 条（0.4%），而二者在 SNLI MR 测试集中分别占 14.6% 和 11.6%——相应单元格的结论样本量不足。
4. **composite 含大量训练期不考的类别**：486 条 composite 中，组件 `synonym_replacement` 出现 33 次、`antonym_substitution` **0 次**，即 composite 的组件分布同样带有上述欠配。
5. **`inv_source_equivalence` 质量门未达标（0.90 < 0.95）**，最弱的是 composite 类别。
6. **人工抽查证据有限**：`artifacts/mrinstr_annotation/MR_QUICK_SAMPLE_AUDIT_40.md` 是一次 40 条的单人分层抽查（17 条标签不同意、11 条不确定、21 条变换无效），既不能估计全体噪声率，也不构成 Human Validation PASS，只能作为 exploratory pilot 的证据。

---

## 附录 A：11 个 MR 的 instruction 原文

> 以下为 `OPERATION_SPECS` 中 `instruction` 字段逐字摘录，会整段插入生成提示词的 `TASK` 段。

**negation_flip** (flip → contradiction)
```
Directly negate the entire SOURCE HYPOTHESIS with the
smallest grammatical edit. Preserve every entity, action, object, count, attribute,
and location. Quantifier scope is critical: an existential source beginning with
A/An/Some/Someone must become No/Nobody/No one (A man walks -> No man walks);
'A man does not walk' is NOT its logical negation. For definite subjects, negate
the main predicate. Avoid double negation and merely different facts.
```

**antonym_substitution** (flip → contradiction)
```
Replace exactly one key adjective, verb, or adverb in the
SOURCE HYPOTHESIS with a direct contextual antonym. Preserve all other facts and do
not use negation. The result must be logically incompatible with the PREMISE, not
merely different or less specific.
```

**adding_contradiction** (flip → contradiction)
```
Copy the complete SOURCE HYPOTHESIS as the first clause,
then append one short clause that contradicts a DIFFERENT fact guaranteed by the
PREMISE but absent from the source hypothesis. Do not contradict the copied clause
itself and do not add merely unmentioned information. The combined statement must
be natural, internally coherent, and false given the premise.
```

**pronoun_substitution** (inv → 同源标签)
```
Replace exactly one unambiguous noun phrase in the SOURCE
HYPOTHESIS with the correct pronoun. The antecedent must remain unambiguous from the
sentence itself. Preserve number, stated gender, grammatical case, all facts, and
the exact truth conditions. If no safe replacement exists, emit no candidate.
```

**voice_switch** (inv → 同源标签)
```
Convert the SOURCE HYPOTHESIS between active and passive
voice only when it has a clear agent-action-patient structure. Preserve agent,
patient, tense, aspect, polarity, and exact truth conditions. If a natural voice
conversion is unavailable, emit no candidate.
```

**synonym_replacement** (inv → 同源标签)
```
Replace exactly one or two common content words in the
SOURCE HYPOTHESIS with natural contextual synonyms. Do not alter named entities,
numbers, quantifiers, roles, locations, or polarity. The result must have identical
truth conditions, not just a similar topic.
```

**uninformative** (neutral → neutral)
```
Copy the complete SOURCE HYPOTHESIS as the first clause,
then append exactly one concrete, plausible detail that the PREMISE neither states
nor rules out. The new detail must make the whole hypothesis neutral. Avoid generic
weather/time boilerplate, modality words, and facts that contradict the premise.
```

**conditional_clause** (neutral → neutral)
```
Copy the complete SOURCE HYPOTHESIS, then append 'only
if' and exactly one concrete necessary condition that the PREMISE neither states nor
rules out. This asserts an unsupported dependency and makes the result neutral.
Preserve every word and the polarity of the source prefix. Do not use may/maybe,
do not put the supported source after 'if', and avoid tautological conditions.
```

**composite_inv** (inv → 同源标签) — *仅用于终审判题*
```
Apply exactly two compatible invariant edits to the
SOURCE HYPOTHESIS, chosen from synonym replacement, unambiguous pronoun substitution,
and active/passive voice conversion. Preserve every fact and exact truth conditions.
Use two genuinely different edit types; if that is unsafe, emit no candidate.
```

**composite_flip** (flip → contradiction) — *仅用于终审判题*
```
First make one conservative invariant edit to the SOURCE
HYPOTHESIS, then make exactly one decisive flip edit chosen from direct negation,
direct-antonym substitution, or a short explicit contradiction. Preserve entities,
roles, counts, and locations. The final result must contradict the PREMISE.
```

**composite_neutral** (neutral → neutral) — *仅用于终审判题*
```
First make one conservative invariant edit to the SOURCE
HYPOTHESIS, then add exactly one unsupported but plausible clause or unresolved
condition. Preserve the original event and entities. The final result must be
neutral relative to the PREMISE, natural, and free of generic boilerplate.
```

---

## 附录 B：源码位置索引

| 组件 | 位置（`augment_snli_all_label_mrs.py`） |
|---|---|
| `PROMPT_VERSION` / `ALLOWED_MODEL` / `LABELS` | L44–46 |
| 词表：否定词 / 停用词 / 强量化词 / 引用词 / 模糊条件词 / 反义词白名单 | L48–83 |
| `GENERATOR_SYSTEM_PROMPT` / `JUDGE_SYSTEM_PROMPT` | L88–103 |
| `OPERATION_SPECS`（11 个 MR） | L105–209 |
| 各类确定性检查（否定范围、语音、反义词白名单） | L225–300 |
| `stratified_sample` | L356–374 |
| `target_label` / `eligible` | L381–408 |
| `build_generation_prompt` | L411–470 |
| `build_judge_prompt` | L473–507 |
| `parse_candidates`（确定性过滤） | L530–699 |
| `parse_judgments` | L702–725 |
| `choose_candidate` | L728–747 |
| `_generate_atomic` | L752–808 |
| `COMPOSITE_STAGE_PLANS` | L810–830 |
| `_generate_composite` | L832–912 |
| `component_mrs_from_log` | L915–929 |
| `synthetic_record`（输出行构造） | L979–1007 |
| `main`（采样 / 主循环 / refill / 报告） | L1086–1420 |
| `audit_augmentation_quality.py`（独立质量审计） | 同目录 |
| `backfill_component_mrs.py`（`component_mrs` 回填） | 同目录 |
| `provenance_schema.py`（共享溯源工具） | 同目录 |
