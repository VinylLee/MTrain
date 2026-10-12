# v3.3 MR 测试生成工具

`generate_mr_test_v3_3.py` 将当前 SNLI 增强管线适配为 MR 测试集生成器。
它直接复用 `augment_snli_all_label_mrs.py` 的提示词、MR 定义、适用条件、
确定性过滤与 composite 阶段计划，输出源样本与 follow-up 合并的 JSONL。

提示词版本为 `snli-pair-aware-all-label-mrs-v3.3`，配置指纹为
`3f7ed9ff3cd82904ebc6f6d5a711f212397a7fb6c25d330a886be42625cf74c0`。
Composite 行的 `component_mrs` 按实际执行顺序记录原子操作。
本次收录工具没有修改核心提示词或既有训练数据。

## 本机数据快照（2026-10-12）

以下数据是本机现有产物，不随本次代码提交上传：

| 数据集 | 原始样本 | 已接受 follow-up | 生成日志覆盖的源样本 | 完成报告 |
|---|---:|---:|---:|---|
| MNLI matched (`mnlim`) | 9,815 | 10,434 | 9,815 | `complete` |
| SICK (`sick`) | 4,906 | 6,019 | 4,906 | `complete` |
| MNLI mismatched (`mnlimm`) | 9,832 | 210 | 220 | 缺失，部分生成 |
| SNLI (`snli`) | 9,824 | 71 | 40 | 缺失，部分生成 |

生成器会将全部选中的源样本写入输出，即使 follow-up 只生成了一部分。
因此，源行数量齐全不能证明全量生成完成。`status=complete` 表示该次生成循环
结束，不能代替语义质量审计，也不保证每个源样本的每个 MR 都生成成功。
当前四个生成进程均未运行。

## 工具与产物

| 文件 | 用途 |
|---|---|
| `prepare_mr_test_v3_3.py` | 按原始数据核对旧版 merged 测试集，保存字节一致的快照并提取源样本 |
| `generate_mr_test_v3_3.py` | 使用当前 MR 管线生成、判定、去重、保存检查点及运行溯源 |
| `verify_mr_test_checkpoint.py` | 检查输出与日志 JSONL、pair_id 连续性及哈希；可显式创建快照 |
| `status_mr_test_v3_3_full.py` | 汇总四个数据集的生成日志覆盖、输出行数与完成报告 |
| `check_mr_test_v3_3_claude.py` | 检查生成进程、GPU、模型及进度；仅写自己的状态，不执行恢复 |
| `run_mr_test_v3_3_full_parallel.sh` | 手动入口：启动四个数据集，跳过已有完成报告或对应 tmux 会话 |
| `run_mr_test_v3_3_sequential.sh` | 当前主模型恢复队列，仅包含 `mnlim` |
| `monitor_mr_test_v3_3_full.sh` | 自动恢复层，仅恢复 `mnlim` 与 `sick` |
| `run_mr_test_v3_3_monitor_loop.sh` | 使用锁与有界单次检查运行周期性恢复 |
| `systemd/mtrain-mr-v33-monitor.*` | 本机恢复监控的 service/timer 配置 |

本机目录分别为：

- `mr_test_sources_v3_3/`：源样本与准备报告。
- `mr_test_data_v3_3_full/`：输出、`.generation_log.jsonl` 与 `.report.json`。
- `mr_test_data_v3_3_smoke/`、`mr_test_data_v3_3_smoke_gpu/`：小样本运行产物。
- `mr_test_data_merged_legacy_*/`：旧版 merged 数据快照。
- `mr_test_checkpoint_backups/`：检查点副本与验证清单。
- `mr_test_runs_v3_3_full/`：stdout、PID、锁及监控状态。

上述目录与 `.bak_*` 副本由 `.gitignore` 排除；忽略不会删除本地文件。
`nli_arg_data/` 中已经跟踪的训练数据及生成日志继续保留在 Git 中。

## 运行与续跑

生成器需要 Python、`openai` 包以及已加载 `gemma-3-4b-it-qat` 的本地
OpenAI 兼容服务。报告中的 generator/judge 模型必须使用该模型；
`--server-model` 可以指定同一模型的本地实例标识。
模型列表存在该标识不代表服务可用，生成器启动时会执行一次推理检查。

源样本准备器默认读取配套 LLMTrain 仓库的旧 merged 数据；
可通过 `--legacy-root`、`--original-root`、`--snapshot-root` 和
`--source-root` 指定路径。快照若已经存在且内容不同，准备器会拒绝覆盖。

使用已准备的源样本进行新目录中的小样本检查：

```bash
python generate_mr_test_v3_3.py \
  --dataset sick \
  --input mr_test_sources_v3_3/sick.jsonl \
  --output mr_test_data_v3_3_smoke/sick.jsonl \
  --sample-size 12 --seed 42 --workers 1
```

示例输出路径必须尚不存在。已有输出默认会拒绝覆盖；
续跑使用 `--resume` 并保留对应的生成日志，使用 `--all` 选择全量源样本。
续跑时应保持输入、MR 列表和提示词配置一致。
日志先于输出保存，可用于重建尚未写入输出的已接受样本；不一致的输出会报错。
请求失败的操作会记录在日志中，续跑不会自动重试已经记录的失败操作；
连续五个源样本出现请求错误时，生成器会中止。

已有产物的只读检查：

```bash
python status_mr_test_v3_3_full.py
python verify_mr_test_checkpoint.py --dataset mnlim
python check_mr_test_v3_3_claude.py --no-persist
```

检查器退出码为 `0=OK`、`1=WARN`、`2=ALERT`、`3=检查器错误`。
检查点验证是结构与哈希检查，不能证明生成完成或语义正确。

## 自动恢复范围

队列与监控的自动恢复范围是 `mnlim` 和 `sick`。
`mnlimm`、`snli` 当前仅报告状态，不自动重启；
四数据集 parallel 脚本是显式手动启动入口，不属于这个恢复限制。
Shell 与 systemd 配置包含 `/home/ubuntu/MTrain`、Conda、LM Studio
及本机模型实例等固定配置，在其他机器使用前需检查并调整。
提交或下载这些文件不会启动任务；应在确认服务和 GPU 资源后手动启动。

## 数据发布

代码仓库中的工具不表示四个测试集都已完成，也不提供本机生成文件下载。
发布数据时应单独整理确定版本的输出、运行报告、SHA-256 清单与源样本来源，
注明完整或部分生成状态、质量审计结果及生成日志的存档位置。
生成日志是 composite 溯源和续跑依据，应独立保存，不能为减小仓库体积而删除。

生成包含随机采样，重新生成得到的是新数据集；代码可用不等于能重新得到
字节一致的既有数据。代码提交与数据版本应分别记录。

## 验证

```bash
python -m pytest -q tests
```

测试使用临时文件与纯函数检查，不启动实际生成、恢复监控或模型服务。
