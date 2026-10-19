# 训练侧捷径与 v3.3 质量对照

保留相同 2033 个原始来源。新训练生成是否达到配额：False。四个全量测试集生成仍禁止。

|版本|原始行|增强行|MR 种类|已知新冠词错误|
|---|---:|---:|---:|---:|
|v3_3|2033|3307|11|0|
|v3_4|2033|2290|5|36|

## 训练侧 hypothesis-only 诊断

按原始来源组相同的 80/20 划分；词表、IDF、逻辑回归只在 80% 拟合，评估另外 20% 的增强行。没有用训练内拟合准确率冒充捷径泛化。

|比较口径|seed|版本|增强验证样本|accuracy|
|---|---:|---|---:|---:|
|full_available_training_cohort|42|v3_3|678|95.87%|
|full_available_training_cohort|42|v3_4|434|56.68%|
|common_source_MR_ordinal|42|v3_3|38|86.84%|
|common_source_MR_ordinal|42|v3_4|38|86.84%|
|full_available_training_cohort|43|v3_3|674|96.44%|
|full_available_training_cohort|43|v3_4|485|51.34%|
|common_source_MR_ordinal|43|v3_3|47|91.49%|
|common_source_MR_ordinal|43|v3_4|47|91.49%|
|full_available_training_cohort|44|v3_3|645|94.73%|
|full_available_training_cohort|44|v3_4|472|52.54%|
|common_source_MR_ordinal|44|v3_3|49|87.76%|
|common_source_MR_ordinal|44|v3_4|49|85.71%|

完整队列是主表；共同来源/MR/序号是对照附表，不能用其覆盖之外的结果概括全体。各 MR 数量、类别基线、置信区间及表面词统计见 JSON。不同 MR 配比会影响捷径下降的解释。

独立 DeBERTa/RoBERTa 对照使用预先冻结的代表抽样及单独 MR 补样，共 2672 个模型输入对；分数不回流接受、过滤、重标或补齐。结果见 external_quality/REPORT_zh.md。
盲审包共 130 条，位于 human_review/。模型标签一致性和双向蕴涵是质量代理，不能判断全部语法或 MR 语义，也不能替代真人判断。因此当前不能把“标签捷径下降”直接宣布为“质量不下降”。

所有输入哈希复核通过，全部新接受行原始响应离线回放通过，未增加生成 API 调用。数据未筛选、重标、覆盖；未修改提示词、未推送。下一步等待用户检查，本程序不会解禁全量测试生成。
