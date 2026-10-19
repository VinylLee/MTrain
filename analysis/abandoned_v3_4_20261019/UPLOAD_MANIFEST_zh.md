# v3.4 输出上传清单

## 已上传

`v3_4_local_runs.tar.gz` 包含以下七个运行目录中的可分享文件：

- `local_runs/nli_v3_4_20261012/`
- `local_runs/nli_v3_4_20261013_repair/`
- `local_runs/nli_v3_4_20261014_round3/`
- `local_runs/nli_v3_4_20261015_current_prompt_shortcut_check/`
- `local_runs/nli_v3_4_20261015_quality_repair/`
- `local_runs/nli_v3_4_20261015_round4_design/`
- `local_runs/nli_v3_4_20261015_round4b_refinement/`

归档保留提示词与代码快照、各轮生成结果和调用日志、捷径与质量审计、报告、盲审题目及已完成评估。v3.4 当前训练输出仍是部分结果，只有 2,290/3,307 条增强数据；此归档不代表质量通过。

| 项目 | 值 |
|---|---:|
| 压缩文件大小 | 91,355,850 bytes（87.1 MiB） |
| 未压缩普通文件数 | 129,837 |
| 未压缩普通文件总大小 | 482,802,608 bytes |
| tar 目录项总数 | 164,105 |
| SHA-256 | `d5f9ae3ca97ca9b83112a40fe33c50e585d3a34230e6a37973c6b6bf47ad825e` |
| 完整性检查 | gzip/tar 可读；没有发现凭证格式命中 |

压缩包本身小于 GitHub 普通 Git 的 100 MiB 单文件硬限制；超过 50 MiB 会显示警告。归档内原始 `plan.json`（199,756,388 bytes）和训练生成日志（96,525,965 bytes）均随压缩包保存。

## 未上传

| 路径或类别 | 大小 | 原因 |
|---|---:|---|
| `local_runs/nli_v3_4_20261012/wheels/torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl` | 780,460,642 bytes | 环境安装包，不是实验输出；原文件超过 GitHub 100 MiB 限制。 |
| `local_runs/nli_v3_4_20261012/snapshot/mr_test_data_v3_3_full/` | 12 files, 188,361,254 bytes | v3.3 测试数据快照本身不在 v3.4 归档包内；同一根目录数据集已在本次补充上传。 |
| `local_runs/nli_v3_4_20261012/snapshot/nli_arg_data/` | 1 file, 1,370,347 bytes | v3.3 训练基准数据的重复快照；仓库已有正式基准文件。 |
| 六份 `private_key.jsonl` | 125,066 bytes | 对应盲审样本身份/版本的私有映射，不随公开盲审材料发布。 |
| `quality_units.private.jsonl` | 449,367 bytes | 质量抽样私有映射，保留本机以避免暴露隐藏分组信息。 |
| 86 个 `*.pyc` 文件 | 2,182,181 bytes | 可重建的 Python 缓存。 |
| 根目录 `mr_test_data_merged_v1.1/` | 230,884,248 bytes | 当时未装入 v3.4 归档包；已在本次补充上传。 |
| `local_runs/LOCAL_GEMMA_AUTHORIZATION.json` | 本机保留 | 本地授权/凭证文件，不纳入远程备份。 |

以上七个 v3.4 运行目录中排除文件合计 972,948,857 bytes；没有删除或移动这些本地文件。完整运行目录仍在原 `local_runs/` 路径下。

## 2026-10-19 数据集补充上传

为备份，已将以下数据目录直接加入本归档分支；`local_runs/` 下七轮 v3.4 运行材料仍由上面的压缩包保存。共 422 个新增数据/日志文件，工作树中的原始字节数为 4,763,776,832。Git 对逐字节相同的文件共用对象存储，因此 checkpoint 历史副本不会按 3.8 GB 重复占用仓库对象空间。

| 路径 | 文件数 | 原始字节数 |
|---|---:|---:|
| `mr_test_checkpoint_backups/` | 261 | 3,823,989,306 |
| `original_dataset/` | 19 | 273,366,119 |
| `mr_test_data_merged_v1.1/` | 5 | 230,884,248 |
| `mr_test_data_v3_3_full/` | 12 | 188,361,254 |
| `mr_test_data_full/` | 32 | 87,976,377 |
| `mr_test_data_full copy/` | 32 | 87,976,377 |
| `mr_test_data_merged/` | 4 | 22,375,177 |
| `mr_test_data_merged_legacy_20260918/` | 4 | 22,375,177 |
| `mr_test_runs_v3_3_full/` | 10 | 19,297,287 |
| `mr_test_sources_v3_3/` | 5 | 6,845,620 |
| `mr_test_data_v3_3_smoke_gpu/` | 3 | 166,087 |
| `mr_test_data_v3_3_smoke/` | 3 | 74,203 |
| `mr_test_data/` | 32 | 89,600 |

`mr_test_runs_v3_3_full/` 中的日志与状态记录已收录；运行时 `.pid` 和 `.lock` 文件跳过。`nli_arg_data/` 的 70 个文件（44,329,826 bytes）以及根目录 `requirements.txt`（2,492 bytes）已在该分支原有提交中，不需要重复添加。当前仓库中没有其他 requirements/conda/pyproject 环境规格文件。

新增数据文件的逐文件 SHA-256 校验表为 `DATA_SHA256SUMS.txt`。没有纳入缓存、环境安装包、盲审私有映射或本地授权文件。
