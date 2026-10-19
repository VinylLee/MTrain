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
| `local_runs/nli_v3_4_20261012/snapshot/mr_test_data_v3_3_full/` | 12 files, 188,361,254 bytes | v3.3 测试数据快照，不是本轮 v3.4 生成结果；保留本机副本。 |
| `local_runs/nli_v3_4_20261012/snapshot/nli_arg_data/` | 1 file, 1,370,347 bytes | v3.3 训练基准数据的重复快照；仓库已有正式基准文件。 |
| 六份 `private_key.jsonl` | 125,066 bytes | 对应盲审样本身份/版本的私有映射，不随公开盲审材料发布。 |
| `quality_units.private.jsonl` | 449,367 bytes | 质量抽样私有映射，保留本机以避免暴露隐藏分组信息。 |
| 86 个 `*.pyc` 文件 | 2,182,181 bytes | 可重建的 Python 缓存。 |
| 根目录 `mr_test_data_merged_v1.1/` | 本机保留 | 输入测试集，不属于 v3.4 的训练侧输出，未复制到归档包。 |

以上七个 v3.4 运行目录中排除文件合计 972,948,857 bytes；没有删除或移动这些本地文件。完整运行目录仍在原 `local_runs/` 路径下。
