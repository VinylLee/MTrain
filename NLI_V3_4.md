# 本地 v3.4 减少 NLI 捷径实验

主目标是 **实际新训练文件 → 四个新测试集** 的 hypothesis-only TF-IDF＋逻辑回归
follow-up accuracy 分别 ≤60%，同时通过人工质量检查。试验结果不能代替全量确认。
代码、数据、报告和备份均不推送远程；旧 v3.3/v1.1 输入不覆盖。

## 方法边界

`nli_v3_4/core.py` 定义所有来源共用的生成和同模型 Gemma 评审规则。训练/测试
身份、数据集名称、外部模型评分均不进入提示词、风格选择或候选排名。添加操作
先生成 edit plan，由代码保留原假设并实现连接；必要条件只允许 H⇒C 的实现。
INV 与 flip 继续保留严格操作约束，复合顺序和中间输出记录在日志中。

第二轮修复将 INV 的同模型评审限制为两个 hypothesis，不输入事实 premise、gold
标签或目标标签，以避免把“未被 premise 支持”误判为“不等价”。接受仍要求语法、
操作、等价三个显式布尔值为 true，两个 hypothesis 方向都明确判为 entailment，
并通过确定性的操作检查。INV 输出标签按严格等价关系继承原始 gold，不重新预测
或修改来源标签；其他 MR 仍评审 premise→candidate 标签。每轮仍只有一次生成、
至多一次 Gemma 评审，最多三候选、两轮；无新增外部模型筛选。
代码还修复了“一或两个同义词”的实现限制，并保护数字、文字数量、限定词、
性别／年龄词和大写名字；代词操作必须实际替换 NP，不能仅插入一个代词。
这可能减少部分 MR 覆盖，必须公开，不能用覆盖率代替质量证据。

第三轮把所有INV的生成也限制为原hypothesis，不输入premise、gold、pair ID或
训练／测试身份；风格由hypothesis和统一种子决定。同义词返回明确的单词位置与
替换计划，代码复制其他原文；允许忠实的短语同义词，不允许添加情绪或限定事实。
生成的label_rule必须是inherit_source，不把任意非标签文本改成NLI标签。
评审先判断明确的两组NLI比较，再判断所声称的操作；反例覆盖变宽的类别／动作、
新增工具、名字和时态改变。声音转换的可尝试范围扩展到一般英语，不再限于视觉
动词白名单，仍要求明确的agent/action/patient、原名字、数值、极性及语法／等价通过。
children/kids仅允许作为同年龄类别的候选；若Gemma未批准仍拒绝。

为降低标签与句式的绑定，第三轮在全部来源使用同一固定分布：输出E全部为INV，
N/C各至少80%来自其原Gold标签的合法INV，其余配额用于原有neutral／flip MR。
这是明确的数据分布调整，并非仅改提示词；旧表达对照仍使用原分配策略，解释结果时
必须同时披露此贡献。未达到合法INV数量只能partial，不能用其他MR填满缺口。
所有原始行仍保留，正式增强总数量不变，外部NLI分数不参与选择、过滤或改标签。

生成标签的统一目标是 E/N/C 各约三分之一，以合法来源＋MR 请求调度实现。
原标签不改，原始样本全保留；失败和配额缺口明确记为 partial。每请求最多三候选、
两轮；复合操作新旧均采用一个确定的阶段计划、相同的逐阶段候选预算。最多四轮
补充请求，所有补充都记录，不能借助外部模型分数挑选或删除输出。
各MR均衡分配尝试预算，避免低成功率操作阻塞配额；来源也按已尝试次数轮转。
尝试次数按“目标标签×MR”分别计数，避免填补E配额的INV请求压制N/C来源的
同类INV请求；新旧表达对照使用相同调度策略。生成顺序和所有失败请求保留。
每个逻辑请求的API调用在发送前独立落盘，收到的原始响应也持久化。恢复时按相同
调用序列复用响应；已发送但无持久响应的调用记为interrupted_call并消耗原有槽位，
不会重新发送或猜测标签。改变调用身份时直接报错，不复用旧缓存。此恢复机制不改变
提示词、MR接受条件或新旧臂预算。当前正在运行的旧代码快照未应用这项修复；新代码
身份单独记录，不允许直接更改旧运行元数据来绕过身份检查。
格式兼容还允许按已打开的JSON层次重排末尾关闭括号，但括号种类和数量必须完全
一致，且前缀不变、规范化后能完整解析；缺失内容、错误词索引或先前层次错误仍拒绝。
原始响应保留，规范化方法及哈希写入轨迹。保存的生成候选可用原请求尚未调用的
评审槽位继续检查；已经调用但失败或未知结果的评审不重发。不额外生成候选，不改变
Gemma接受条件、INV分配或提示词轮次。
LM Studio实测已支持当前INV JSON Schema。新工程版本只在INV生成调用中使用
同一格式约束：索引与原词绑定，空数组允许弃权，最多三候选；原提示词、温度、
token上限及评审接受条件不变。此项为显式解码格式干预，不能把差异仅归因于提示词。
评审与旧表达对照不使用此生成格式。旧快照发生中断且响应未落盘时，依据逐请求
回放的原调度定位可能在途的请求，并保守消耗整请求预算；记录实际调用次数未知，
不会伪造响应、接受行或语义复核。新运行保留旧执行代码身份和完整旧输出／日志。
实质性的同义词替换允许规范化句末句号；仅改标点仍拒绝，不改变内部标点范围。

2026-10-15 的人工开发复核已补齐审核人并归档，但四项质量指标均未达到95%。
后续工程修补在送审前将附加分句句首的普通冠词／代词转为句中大小写，保留I、
名字、大写缩写和引号原文；原始模型响应及实现前后文本分别留存。评审和输出使用
同一实现文本；文本改变后禁止复用旧评审。此项不修补原始片段或补写缺失事实。
同义词编辑保护about to等固定短语。voice新增保守的简单及物结构检查，核对
施受角色、词项保留、时态、模态和体；复杂／有歧义结构弃权，仍需Gemma等价评审。
该检查不证明语义正确，且会损失有效复杂句的覆盖，不能靠删旧数据宣称质量改善。
修补保留第3轮全部提示词和调用预算，改变执行契约及代码身份；旧运行不能直接续跑。
52,800条开发生成提示词哈希未变，离线检查未调用模型，也未修改681条旧增强或原审核。
检查与覆盖代价见`local_runs/nli_v3_4_20261015_quality_repair/REPORT_zh.md`。
尚无修补后的独立人工质量结论或全量捷径结果；原追加补齐提案未获批准也未执行。
`run_nli_quality_repair_pilot.py`另提供待批准的小规模质量诊断：在五来源各取10个
未参与上一批人工复核的来源组，五种MR各2请求，固定50请求、至多250次Gemma
调用，无补齐轮次。计划绑定单个输出目录和代码／数据哈希，继续使用第3轮提示词。
prepare／validate不调用模型；run需单独确认新增预算。该诊断保留失败与弃权，
不改变原681条数据或完整实验配额，也不作为最终质量／捷径验收证据。盲审入口
支持显式`preliminary_development_historical_reference`范围，不给历史参考伪造
生成元数据；该范围不能用于冻结。具体清单与执行命令见本地
`local_runs/nli_v3_4_20261015_quality_repair/quality_pilot_proposal_v2/PLAN_zh.md`。
该小试随后获用户批准并完成：50请求、123次Gemma调用、17条候选、33次失败／弃权。
外部模型诊断仅38个输入对，未用于挑选；新人工盲审共29条，已完成并正式归档。INV仅1条
且无历史配对；无配对的质量指标仍报告新版分数和样本数，比较缺失时标记证据不足，
不能漏报或判通过。该17条新候选的人工标签正确14/17、语法11/17、MR合规7/17；
adding_contradiction的MR合规0/6，质量判定不通过，不支持扩量。审核人zhehao li，
输入SHA-256为`2c265d13db3eb28b96b0da91ce7f15f17d3629b4e3ac44687f8919ddbcc54d47`。
最新正式报告见该目录下`execution/human_review_assessment_2c265d13db3e/REPORT_zh.md`；
生成小试报告`execution/REPORT_zh.md`保留当时状态，不代表正式目标完成。

`analysis/design_nli_quality.py` **只允许开发来源和 development 生成记录**。
DeBERTa 与 RoBERTa MNLI 的用途是定位提示词错误并帮助修改通用规则。
评分文件与生成文件分离，只有分数和诊断标记，没有样本接受、过滤或改标签接口。
正式增强入口不导入此工具或 transformers，正式阶段不调用外部 NLI 模型。
完整测试中参与设计的来源另列；冻结后其余来源用于确认，不能反复调参挑最佳结果。

## 固定诊断

`analysis/hypothesis_only_logreg.py` 使用完整训练文件（原始＋增强）拟合词级
unigram/bigram TF-IDF，min_df=2、max_features=30000、sublinear_tf=True。
LR C=1、lbfgs、无类别加权、seed42、tol=1e-4；未收敛则增加迭代上限。
特征输入仅 hypothesis，词表和 IDF 不接触测试。标签用于监督；pair ID 与 MR
只在预测之后用于来源组 bootstrap 和分组报告。主指标只计测试增强行。

```bash
export LD_LIBRARY_PATH=/home/ubuntu/anaconda3/lib
PY=/home/ubuntu/anaconda3/bin/python
$PY analysis/hypothesis_only_logreg.py \
  --train nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data_all_label_mrs_v3_3_full.json \
  --test-root mr_test_data_v3_3_full --output local_runs/baseline_logreg_new_run
```

2026-10-12 完整 v3.3 的该口径基线：SNLI 96.02%、MNLI-m 90.23%、
MNLI-mm 89.78%、SICK 92.74%。当前输出见
`local_runs/nli_v3_4_20261012/baseline_logreg/REPORT_zh.md`。

## 开发、人工复核与冻结

```bash
$PY run_nli_v3_4.py prepare \
  --train nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data_all_label_mrs_v3_3_full.json \
  --test-root mr_test_data_v3_3_full --output local_runs/development_new_run

$PY run_nli_v3_4_development.py \
  --manifest local_runs/development_new_run/manifest.json \
  --output local_runs/pilot_round1 --round 1 --workers 2 \
  --design-python .venv-nli-design/bin/python
```

每个来源固定120组（每原标签40组），五个来源共600组。每来源试验目标180条增强。
三个对照臂：v3_3_original（旧表达＋旧标签比例）、v3_3（旧表达＋均衡）、
v3_4（新表达＋均衡）。这三个臂为重放试验，不等同于历史完整数据。
第三轮v3_4另外固定上述INV分层配额；对照表不能把分布与表达的总差异写成纯提示词效果。
各臂使用相同单请求预算，但语义失败率可能导致实际请求次数不同；请求、MR
覆盖与标签分布全部公开。比较共享来源/MR的配对子集，单列清理和分布调整贡献。

驱动器可用同一命令恢复，生成日志为事实记录；配置、来源或代码变化时拒绝恢复。
它停在人工复核，不自动冻结或启动正式生成。服务连续错误则保留已生成结果并停止。
驱动器的SIGINT／SIGTERM中断会取消排队任务并中断其实际子进程，等待检查点后
退出；不能仅凭停止标记判断进程已结束，需核实进程及其子进程。
第二轮起，即使生成部分完成也会尽量输出开发诊断和临时捷径指标，并明确标记
partial；新提示词不达标时先停止以待修订，不先消耗全部旧表达对照的预算。
临时指标不能通过冻结或最终验收。正式轮次保存代码快照，从快照运行避免后续
工作目录修改影响正在执行的轮次；旧轮次、旧文件和原始生成日志均保留。

```bash
$PY status_nli_v3_4.py --root local_runs/nli_v3_4_20261012/pilot_round1
```

外部诊断环境独立于生成环境。当前隔离环境从已有兼容包复制，Torch2.0.1、
Transformers4.29.2；实际依赖和来源记录在
`local_runs/nli_v3_4_20261012/design_environment.json`。两模型固定本地缓存修订，
使用 safetensors，不下载权重，不静默截断；GPU可用且空闲≥8GiB时选择空闲最多的卡，
否则CPU。开发评分不能被称为质量真值。

人工先读取 `human_review/label_grammar_blind.jsonl`，再读 `operation_blind.jsonl`，
不要读取私有版本键。默认每MR20对，稀有类型全审，证据不足扩大至40对。
由真实审核人填写模板副本，提供审核人标识及人工完成声明；自动意见不能冒充人工。

```bash
$PY analysis/nli_v3_4_review.py assess \
  --review-root local_runs/pilot_round1/human_review \
  --answers local_runs/pilot_round1/human_answers.json \
  --output local_runs/pilot_round1/human_quality.json
$PY run_nli_v3_4.py freeze \
  --manifest local_runs/development_new_run/manifest.json \
  --quality local_runs/pilot_round1/human_quality.json \
  --shortcut local_runs/pilot_round1/v3_4_logreg/metrics.json \
  --round 1 --output local_runs/pilot_round1/frozen.json
```

冻结要求五来源生成完整、四集开发诊断≤60%，及真实人工质量通过：标签、语言、
操作、INV等价率各≥95%，新版本点估计不低于旧版本，配对来源组 bootstrap 差异
95%下界≥−2个百分点。稀有或未覆盖MR的证据量必须披露；不能把缺失证据写成通过。
最多三轮正式提示词开发；早期接口 smoke 单独保存，不混入正式试验。

## 正式数据与最终验收

```bash
$PY run_nli_v3_4.py generate --stage production --dataset train \
  --manifest local_runs/development_new_run/manifest.json \
  --freeze local_runs/pilot_round1/frozen.json \
  --output local_runs/production_v3_4/train.jsonl
```

其余来源分别指定 snli、mnlim、mnlimm、sick。只有冻结后的相同配置可生成正式数据；
正式配额是对应历史完整文件的增强数量，不允许用小样本目标冒充完成。保留所有
2,033训练来源和34,377测试来源。输出兼容现有训练/测试字段，不修改下游训练代码。

生成完后运行主诊断（加 `--development-manifest` 单列非开发来源），分别运行
旧训练→新测试、新训练→旧测试；辅助探针仍高时公开残余捷径。
正式人工盲审可用 `nli_v3_4_review.py package --stage production`，旧基线使用
prepare保存的 baseline_full 目录。所有总体完成声明必须同时具有完整生成、
四集≤60%和人工质量证据；生成报告中的 overall_complete 始终为 false，最终由
独立的 release 验证汇总，不把名义生成完成当成实验完成。

验证命令为 `$PY -m pytest -q tests`。根目录的 new_test.py 导入实验函数 test_nli，
没有对应pytest fixture，故全仓库无参数收集并非有效单元测试入口。
