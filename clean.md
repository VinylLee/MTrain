# MetTrain 项目清理分析报告

> 生成日期：2026-06-24

## 1. 项目概述

**MetTrain** 是一个研究原型项目，实现 **面向自然语言推理 (NLI) 任务的 Metamorphic Training（蜕变训练）** 方法。核心思想是利用 LLM 对训练数据施加蜕变关系（如同义词替换、否定翻转、语态转换等），生成增强样本和伪标签，通过半监督多阶段训练提升 NLI 模型在小样本场景下的性能。

训练流程包含三个核心阶段：
1. **阶段1** — 有标签数据上的 CE + R-Drop 训练
2. **阶段2** — 增强数据上的双重一致性训练
3. **阶段3** — 无标签数据上的对比一致性训练

---

## 2. 文件分类总览

共 **48 个 Python 文件**，分为以下 4 类：

| 类别 | 数量 | 占比 |
|------|------|------|
| 🟢 核心文件 (保留) | 16 | 33% |
| 🟡 可选/辅助工具 (视情况保留) | 6 | 13% |
| 🔴 应删除 (调试/实验/备份) | 24 | 50% |
| ⚪ 其他 (数据/配置) | 2 | 4% |

---

## 3. 🟢 核心文件 — 必须保留 (16 个)

这些文件构成了完整的训练→增强→评估→实验可复现流程，删除任何一个都会导致核心功能缺失。

### 3.1 模型与数据层 (5 个)

| 文件 | 行数 | 功能 | 保留理由 |
|------|------|------|----------|
| `NLImodel.py` | 402 | NLI 模型定义：`SequenceClassifierWithCustomLoss` 包装器 + `NLIModel` 容器 | 模型层核心，所有训练脚本依赖 |
| `NLIdataset.py` | 572 | 数据集加载/划分/增强/保存 | 数据层核心，包含从 HuggingFace 下载到本地 JSON 加载的完整逻辑 |
| `NLI_collate_fn.py` | ~100 | DataLoader 的 collate 函数（labeled/augmented/filter） | 训练流程依赖的数据批处理逻辑 |
| `NLItest.py` | 138 | 模型评估：Accuracy/Precision/Recall/F1 | 评估标准实现 |

### 3.2 训练引擎 (3 个)

| 文件 | 行数 | 功能 | 保留理由 |
|------|------|------|----------|
| `NLItrain.py` | 796 | 三大训练主函数：`train_nli`, `train_nli_progressive`, `train_nli_unlabeled_labeled` | 最高层训练编排，论文核心方法实现 |
| `NLI_train_one_epoch.py` | 2538 | 单 epoch 训练函数：CE/RDrop/Dual Consistency/Multi-Signal Joint/Triplet Consistency | 训练原语，被 `NLI_train_stage.py` 调用 |
| `NLI_train_stage.py` | 962 | 训练阶段函数：`train_with_labeled_data`, `train_with_consistency`, `train_with_augment`, `train_with_unlabeled_data` | 连接训练主函数与单 epoch 函数的中间层 |

### 3.3 数据增强 (2 个)

| 文件 | 行数 | 功能 | 保留理由 |
|------|------|------|----------|
| `MRAugment.py` | 2125 | 蜕变关系工具箱：Invariant/Flip/Neutral 三类 MR 的 prompt 工程 + LLM 调用 | 论文最大创新点——蜕变关系定义与生成 |
| `LabeledEnhance.py` | 342 | `DataAugmentor` 类：批量对数据集应用多个 MR，多进程并行 | 增强流程编排器 |

### 3.4 基础设施 (4 个)

| 文件 | 行数 | 功能 | 保留理由 |
|------|------|------|----------|
| `LMstudio.py` | 149 | `OpenAIChat` 类：OpenAI 兼容 API 客户端（支持 Ollama/LM Studio） | LLM 通信的唯一接口 |
| `mylogging.py` | 21 | 全局日志配置 | 所有模块依赖的日志系统 |
| `NLIutils.py` | ~212 | 回译工具 + SBERT 相似度计算（LibreTranslate 集成） | MRAugment 的回译增强依赖 |
| `scoring.py` | 147 | NLI entailment 评分 + 语义相似度 + top-half 筛选 | 数据质量筛选 |

### 3.5 实验入口 (1 个)

| 文件 | 行数 | 功能 | 保留理由 |
|------|------|------|----------|
| `new_run.py` | 1228 | 配置驱动的实验 runner：`MultiSignalExperiment` 类 | **推荐的唯一入口**（README 明确说明），支持 JSON/JSONL，固定配置复现实验 |

### 3.6 工具 (1 个)

| 文件 | 行数 | 功能 | 保留理由 |
|------|------|------|----------|
| `seed_checker.py` | 286 | 随机种子设置 + 可复现性检查 | `new_run.py` 导入使用，论文复现必需 |

---

## 4. 🟡 可选/辅助文件 (6 个)

根据你的实际需求决定是否保留。

| 文件 | 功能 | 保留建议 |
|------|------|----------|
| `new_test.py` (742行) | 多模型多数据集测试脚本，支持命令行参数、分类报告、混淆矩阵 | **建议保留**——比 `NLItest.py` 功能更完善，适合批量评估 |
| `NLImain.py` (399行) | 原始 CLI 入口，argparse 驱动 | **可以删除**——README 明确说已被 `new_run.py` 替代，且硬编码路径多 |
| `testLLMNLI.py` (232行) | 直接评测 LLM 的 NLI 能力（不微调，零样本推理） | **建议保留**——用于评估 LLM 生成质量，论文可能需要消融实验 |
| `metamorphic_training_usage.py` (125行) | 纯文档/示例打印脚本，无实际代码逻辑 | **建议删除**——不是代码，是 usage guide，README 已覆盖 |
| `stress_test.py` (48行) | 批量在一组 checkpoint 上跑 stress test 并输出 CSV | **建议保留**——简单但实用的批量评估脚本 |
| `OurDS7b.py` (155行) | 旧版 OllamaChat 实现 | **可以删除**——已被 `LMstudio.py` 的 `OpenAIChat` 完全替代，无任何文件导入它 |

---

## 5. 🔴 应删除文件 (24 个)

### 5.1 备份/重复文件 (2 个)

| 文件 | 说明 |
|------|------|
| `MRAugment copy.py` | `MRAugment.py` 的完整复本（2125行），备份残留 |
| `requirements-backup.txt` | 可保留作为依赖参考，但非代码 |

### 5.2 调试/修复验证脚本 (19 个)

这些都是开发过程中的临时脚本，用于测试和验证某个特定功能是否正常工作，任务完成后再无价值：

| 文件 | 当初用途 |
|------|----------|
| `test_gpu_simple.py` | 测试 GPU 是否可用 |
| `test_composite_mr_loss.py` | 测试复合 MR 损失函数 |
| `test_data_type_fix.py` | 验证数据类型修复 |
| `test_device_fix.py` | 验证设备选择修复 |
| `test_device_selection.py` | 测试 GPU 设备选择逻辑 |
| `test_gpu_cleanup.py` | 测试 GPU 内存清理 |
| `test_interfaces.py` | 测试模块接口一致性 |
| `test_jsonl_loading.py` | 验证 JSONL 文件加载 |
| `test_mapping_functionality.py` | 测试数据映射功能 |
| `test_metamorphic_functions.py` | 测试蜕变关系函数 |
| `test_metamorphic_validate.py` | 验证蜕变关系正确性 |
| `test_new_training_method.py` | 测试新训练方法 |
| `test_pairwise_training.py` | 测试 pairwise 训练 |
| `test_param_filter.py` | 测试参数过滤 |
| `test_rte_modification.py` | 测试 RTE 数据修改 |
| `test_save_model_feature.py` | 测试模型保存功能 |
| `test_seed.py` | 测试随机种子 |
| `test_validate_fix.py` | 验证 validate 函数修复 |
| `verify_fix.py` | 验证 MR 训练函数签名修复 |

### 5.3 一次性临时脚本 (2 个)

| 文件 | 说明 |
|------|------|
| `temp.py` | 一次性脚本，用于删除 JSONL 中的空行，硬编码路径指向已不存在的目录 |
| `check_project_seed.py` | `seed_checker.py` 的薄封装，仅调用 `print_seed_status()` |

### 5.4 空文件 (1 个)

| 文件 | 说明 |
|------|------|
| `example_unlabeled_first_training.py` | 完全空文件，仅含一个空行 |

---

## 6. ⚪ 数据和配置 (2 个)

| 文件 | 说明 |
|------|------|
| `README.md` | **必须保留**——项目文档 |
| `nli_arg_data/` 目录 | **必须保留**——包含 RTE 示例数据集和数据处理工具 |

### 6.1 nli_arg_data/ 子目录分析

| 文件 | 功能 | 建议 |
|------|------|------|
| `rte_lr0.8_gemma-e-4b-it-qat/` | RTE 示例数据集（labeled/augmented/unlabeled/test/validation.json） | **保留**——README 中用于 Quick Start |
| `Merge.py` | 合并 unlabeled 和 augmented 数据集 | **保留**——数据预处理工具 |
| `MRDissimilarity.py` | 基于 SBERT 计算样本间语义不相似度 | **保留**——数据质量控制 |
| `remove_none_mr_id.py` | 删除 `mr_id="none"` 的条目（交互式） | **可删除**——`_auto` 版本替代 |
| `remove_none_mr_id_auto.py` | 同上，自动执行版本 | **保留**——更实用 |
| `install_dependencies.sh` | 安装 sentence-transformers 等依赖 | **可删除**——README 中已有安装命令 |

---

## 7. 推荐清理操作

### 7.1 立即删除（一步到位）

```bash
cd /home/ubuntu/WoYaoFaA/MetTrain

# 备份副本
rm "MRAugment copy.py"

# 旧的 LLM 接口
rm OurDS7b.py

# 旧入口（已弃用）
rm NLImain.py

# 纯文档脚本
rm metamorphic_training_usage.py

# 空文件
rm example_unlabeled_first_training.py

# 一次性临时脚本
rm temp.py
rm check_project_seed.py

# 19 个调试/测试脚本
rm test_gpu_simple.py
rm test_composite_mr_loss.py
rm test_data_type_fix.py
rm test_device_fix.py
rm test_device_selection.py
rm test_gpu_cleanup.py
rm test_interfaces.py
rm test_jsonl_loading.py
rm test_mapping_functionality.py
rm test_metamorphic_functions.py
rm test_metamorphic_validate.py
rm test_new_training_method.py
rm test_pairwise_training.py
rm test_param_filter.py
rm test_rte_modification.py
rm test_save_model_feature.py
rm test_seed.py
rm test_validate_fix.py
rm verify_fix.py

# nli_arg_data 中不需要的文件
rm nli_arg_data/remove_none_mr_id.py
rm nli_arg_data/install_dependencies.sh
```

### 7.2 删除后的项目结构

```
MetTrain/
├── README.md
├── requirements-backup.txt
├── mylogging.py                  # 日志
├── LMstudio.py                   # LLM 接口
├── seed_checker.py               # 种子管理
├── scoring.py                    # 评分工具
├── NLIutils.py                   # 回译工具
├── NLImodel.py                   # 模型定义
├── NLIdataset.py                 # 数据加载
├── NLI_collate_fn.py             # Collate 函数
├── NLItest.py                    # 评估
├── NLItrain.py                   # 训练主循环
├── NLI_train_one_epoch.py        # 单 epoch 训练
├── NLI_train_stage.py            # 训练阶段
├── MRAugment.py                  # 蜕变关系增强
├── LabeledEnhance.py             # 数据增强器
├── new_run.py                    # 推荐入口
├── new_test.py                   # 测试脚本
├── testLLMNLI.py                 # LLM 零样本评估
├── stress_test.py                # 批量压力测试
├── quick_multi_signal_test.py    # 快速验证
├── bare_jrnl_new_sample4.pdf     # 论文模板
├── nli_arg_data/
│   ├── rte_lr0.8_gemma-e-4b-it-qat/  # 示例数据
│   ├── Merge.py                       # 数据合并
│   ├── MRDissimilarity.py             # 语义去重
│   └── remove_none_mr_id_auto.py      # 数据清洗
└── experiments/                  # 实验输出（运行时生成）
```

最终保留 **22 个 .py 文件**（含 nli_arg_data 中的 3 个工具），从原来的 48 个减少了 **54%**。

---

## 8. 关键依赖关系图

```
new_run.py (入口)
  ├── NLImodel.py → LMstudio.py, mylogging.py, seed_checker.py
  ├── NLIdataset.py → MRAugment.py, LabeledEnhance.py, LMstudio.py, mylogging.py
  ├── NLItrain.py → NLI_train_one_epoch.py, NLI_train_stage.py, NLI_collate_fn.py
  │   ├── NLI_train_one_epoch.py → mylogging.py
  │   └── NLI_train_stage.py → NLI_train_one_epoch.py, mylogging.py
  ├── NLItest.py → NLImodel.py, mylogging.py
  └── scoring.py (独立)
```

---

## 9. 建议的额外清理

1. **硬编码路径**：`NLIdataset.py` 中的 `DEFAULT_AUG_ROOT = "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data"` 和 `NLImodel.py` 中的 `cache_dir` 应改为相对路径或环境变量
2. **已注释代码块**：`NLItrain.py` 和 `NLImodel.py` 中存在大量被注释掉的调试/旧代码段，建议在最终版本中移除以提高可读性
3. **`quick_multi_signal_test.py`**：该文件包含 `NLI_train_one_epoch.py` 某些函数的本地 fallback 实现，如果保留则应与主实现统一
4. **`requirements-backup.txt`**：文件名含 "backup"，建议重命名为 `requirements.txt`
5. **PDF 文件**：`bare_jrnl_new_sample4.pdf` 是论文模板，与代码无关，可放入 `docs/` 目录
