# dataset.py
"""
本模块用于定义 NLI 数据集加载及保存相关的类和方法。
支持：
1. 从 HuggingFace Hub 下载原始数据集并按比例划分为 labeled / unlabeled / validation / test。
2. 从本地增广数据目录直接加载数据集，目录规则：
   /home/ubuntu/MTrain/MTrainNLI/nli_aug_data/{数据集名}_lr{labeled_ratio:.3f}/
   ├── labeled_data.json
   ├── unlabeled_data.json
   ├── augmented_data.json
   ├── validation.json
   └── test.json
3. 预留数据增强接口，并可将 (增广后的) 数据再次保存为同样格式，方便下次直接加载。

注意：为了兼容 Python 3.9，这里不再使用新语法的联合类型 ("|")，而是使用 typing.Optional。
SNLI 的 0‑2 标签分别表示 Entailment、Neutral、Contradiction。其他数据集应与snli保持一致。
"""

import os
from tokenize import Triple
from typing import List, Dict, Any, Union, Optional

from datasets import load_dataset, Dataset, concatenate_datasets, IterableDataset

from LabeledEnhance import DataAugmentor
from mylogging import logger

# 默认增广数据根目录
DEFAULT_AUG_ROOT = "/home/ubuntu/MTrain/MTrainNLI/nli_aug_data"

# ---------------- 辅助函数 ----------------


def clean_dataset(dataset: Union[Dataset, IterableDataset]) -> Dataset:
    """移除 label == -1 的无效样本"""
    valid_dataset = dataset.filter(lambda example: example["label"] != -1)
    logger.info(f"清洗后数据集大小: {len(valid_dataset)} (原始大小: {len(dataset)})")
    return valid_dataset


# ---------------- 主类 ----------------


class NLIDataset:
    """
    Initializes the dataset handler for Natural Language Inference (NLI) tasks.

    This class supports loading, splitting, and augmenting NLI datasets. It can 
    handle both original and augmented datasets, with options to specify dataset 
    ratios and directories for caching and augmentation.

    Args:
        dataset_name (str): The name of the dataset to load (default: "snli").
        labeled_ratio (float): The ratio of labeled data in the dataset (default: 0.1).
        unlabeled_ratio (float): The ratio of unlabeled data in the dataset (default: 0.2).
        cache_dir (str): The directory to cache the original dataset 
            (default: "/home/ubuntu/MTrain/MTrainNLI/nli_dataset_cache").
        augmented_dir (Optional[str]): The directory containing augmented data. 
            If None, a default directory is constructed based on the dataset name 
            and labeled ratio (default: None).
        seed (int): The random seed for reproducibility (default: 42).
        use_data_augmentation (bool): Whether to use data augmentation. If True, 
            attempts to load augmented data or performs augmentation if not available 
            (default: True).

    Raises:
        Warning: If the augmented data directory is incomplete or does not exist, 
            the original dataset is loaded and augmented.

    Side Effects:
        - Logs information about the dataset loading and augmentation process.
        - Saves augmented datasets to the specified directory if augmentation is performed.

    Notes:
        - If `use_data_augmentation` is True and `augmented_dir` is not specified, 
            the augmented directory is automatically determined based on the dataset name 
            and labeled ratio.
        - If the augmented data directory is incomplete or missing, the original dataset 
            is loaded, split, and augmented before saving the augmented dataset.
    """
    from LMstudio import MODEL

    def __init__(
        self,
        dataset_name: str = "snli",
        labeled_ratio: float = 0.1,
        unlabeled_ratio: float = 0.2,
        use_data_augmentation: bool = False,
        use_original_val: bool = True,  # 已废弃，为了兼容旧代码
        val_ratio: float = 0.1,  # 已废弃，为了兼容旧代码
        augmented_dir: Optional[str] = None,
        seed: int = 42,
        cache_dir: str = "/home/ubuntu/MTrain/MTrainNLI/nli_dataset_cache",
        llm_model_name: str = MODEL,
    ) -> None:
        self.llm_model_name = llm_model_name
        self.dataset_name = dataset_name
        self.labeled_ratio = labeled_ratio
        self.unlabeled_ratio = unlabeled_ratio
        self.seed = seed
        self.use_data_augmentation = use_data_augmentation
        if augmented_dir is None:
            logger.info("未显式指定 augmented_dir，则按默认规则查找")
            subdir = f"{self.dataset_name}_lr{self.labeled_ratio}_{self.llm_model_name}"
            self.subdir = subdir
            augmented_dir = os.path.join(DEFAULT_AUG_ROOT, subdir)
        self.augmented_dir = augmented_dir
        logger.info(f"数据增强: {use_data_augmentation}")
        if use_data_augmentation:
            loaded, reason = self._try_load_augmented(augmented_dir)
            if not loaded:
                logger.warning(f"增广数据目录 {augmented_dir} 加载失败，尝试数据增强")
                self._load_and_split_original(cache_dir)
                logger.info("原始数据集加载完成，进行数据增强 …")
                self.save_augmented_dataset()
                if "labeled" in reason and "unlabeled" not in reason:
                    logger.warning("label 增广数据集缺失，尝试进行数据增强")
                    self._handle_data_augmentation(augmented_dir, cache_dir)
                elif "unlabeled" in reason:
                    logger.warning("unlabel 无标签数据集缺失，尝试进行无标签数据利用")
                    self._handle_unlabeled_data(augmented_dir, cache_dir)
                else:
                    logger.warning("增广和无标签数据集缺失，尝试进行数据增强和无标签数据利用")
                    self._handle_data_augmentation(augmented_dir, cache_dir)
                    self._handle_unlabeled_data(augmented_dir, cache_dir)

                logger.info("数据增强完成，重新加载增广数据集 …")
                self._try_load_augmented(augmented_dir)
        else:  # 如果不使用数据增强或增强失败，则只加载原始数据
            self._load_and_split_original(cache_dir)

    def _handle_data_augmentation(self, augmented_dir: Optional[str],
                                  cache_dir: str):
        """
        处理数据增强逻辑：先尝试加载已有增强数据，如果不存在则执行增强流程。
        
        参数:
            augmented_dir: 增强数据目录路径，如果为None则使用默认路径
            cache_dir: 原始数据缓存目录
            use_original_val: 是否使用原始验证集
        """
        # 对每个数据新增字段
        self.labeled_dataset = self.labeled_dataset.map(lambda sample, idx: {
            **sample, "pair_id": idx,
            "is_syn": False,
            "weight": 1.0,
            "mr_id": "none",
            "mr_type": "inv"
        },
                                                        with_indices=True)

        # 使用 LLM 进行数据增强
        logger.info(f"使用 LLM 进行数据增强: {self.llm_model_name}")
        from LMstudio import OpenAIChat, HOST, PORT
        llm = OpenAIChat(
            host=HOST,
            port=PORT,
            model=self.llm_model_name,
        )

        # 注册蜕变关系
        from MRAugment import NLIMRTool
        safe_mr_augmentor = DataAugmentor(NLIMRTool(llm=llm))
        for name, func in NLIMRTool.COMPOSITE_MRS_MULTIPLE.items():
            safe_mr_augmentor.register_transformation(name, func)
            logger.info(f"注册蜕变关系: {name}")

        # 对有标签数据集进行增强
        self.augmented_dataset = safe_mr_augmentor.augment_dataset(
            self, num_workers=1)
        logger.info("数据增强完成，保存增广数据集 …")
        self.save_augmented_dataset()

    def _handle_unlabeled_data(self, augmented_dir: Optional[str],
                               cache_dir: str):
        """
        处理无标签数据逻辑：如果增广数据目录存在且完整，则加载 unlabeled 数据集。
        否则，加载原始数据集并划分 unlabeled 数据集。

        参数:
            augmented_dir: 增强数据目录路径
            cache_dir: 原始数据缓存目录
            use_original_val: 是否使用原始验证集
        """
        # 对每个数据新增字段,并且把premise的字段复制给hypothesis，label置为0（蕴含）
        self.unlabeled_dataset = self.unlabeled_dataset.map(
            lambda sample, idx: {
                **sample,
                "hypothesis": sample["premise"],
                "label": 0,
                "pair_id": idx,
                "is_syn": False,
                "mr_id": "none",
                "weight": 0.01,
                "mr_type": "inv",
            },
            with_indices=True)

        # 使用 LLM 进行数据增强
        logger.info(f"使用 LLM 进行数据增强: {self.llm_model_name}")
        from LMstudio import OpenAIChat, HOST, PORT
        llm = OpenAIChat(
            host=HOST,
            port=PORT,
            model=self.llm_model_name,
        )

        # 注册蜕变关系
        from MRAugment import NLIMRTool
        unlabel_mr_augmentor = DataAugmentor(NLIMRTool(llm=llm))
        for name, func in NLIMRTool.COMPOSITE_MRS_MULTIPLE.items():
            unlabel_mr_augmentor.register_transformation(name, func)
            logger.info(f"注册蜕变关系: {name}")

        # 对有标签数据集进行增强
        self.unlabeled_dataset = unlabel_mr_augmentor.augment_dataset(
            self, labeled=False)
        logger.info("数据增强完成，保存增广数据集 …")
        self.save_augmented_dataset()

    def _try_load_augmented(self, aug_dir: Optional[str]) -> tuple[bool, str]:
        """若 aug_dir 完整则加载并打印数据统计"""
        file_map = {
            "labeled": "labeled_data.json",
            "augmented": "augmented_data.json",
            "unlabeled": "unlabeled_data.json",
            "test": "test.json",
        }

        logger.info(f"增广数据目录: {aug_dir}")
        if aug_dir is None:
            logger.warning("增广数据目录不存在")
            return False, "增广数据目录不存在"
        paths = {k: os.path.join(aug_dir, v) for k, v in file_map.items()}
        if not all(os.path.exists(p) for p in paths.values()):
            logger.warning(f"增广目录 {aug_dir} 文件不完整，跳过。")
            return False, f"增广目录 {aug_dir} 文件不完整，跳过。"

        try:
            logger.info(f"检测到增广数据目录 {aug_dir}，开始加载 …")
            self.labeled_dataset = load_dataset(
                "json", data_files={"train": paths["labeled"]})["train"]
            self.unlabeled_dataset = load_dataset(
                "json", data_files={"train": paths["unlabeled"]})["train"]
            self.validation_dataset = load_dataset(
                "json", data_files={"train": paths["test"]})["train"]
            self.test_dataset = load_dataset(
                "json", data_files={"train": paths["test"]})["train"]
            self.augmented_dataset = load_dataset(
                "json", data_files={"train": paths["augmented"]})["train"]
        except Exception as e:
            logger.error(f"加载增广数据集失败: {str(e)}")
            return False, f"加载增广数据集失败: {str(e)}"

        # 检查数据集中是否有异常值
        datasets_to_check = {
            "labeled": self.labeled_dataset,
            "augmented": self.augmented_dataset,
            "validation": self.validation_dataset,
            "test": self.test_dataset,
            "unlabeled": self.unlabeled_dataset
        }

        import math
        for name, dataset in datasets_to_check.items():
            # 检查数据集是否为空
            if len(dataset) == 0:
                logger.warning(f"{name} 数据集为空，可能是数据增强失败")
                return False, name
            # 获取所有字段名
            fields = dataset.column_names
            for field in fields:
                # 检查是否有 None 值
                if dataset.filter(lambda x: x[field] is None).num_rows > 0:
                    logger.warning(
                        f"{name} 数据集中 {field} 字段存在 null 值，可能是数据增强失败")
                    return False, name
                # 检查字符串类型字段是否有空字符串
                if len(dataset) > 0 and isinstance(dataset[0][field], str):
                    if dataset.filter(lambda x: isinstance(x[field], str) and
                                      x[field].strip() == "").num_rows > 0:
                        logger.warning(
                            f"{name} 数据集中 {field} 字段存在空字符串，可能是数据增强失败")
                        return False, name
                # 检查数值类型字段是否有 NaN 或 Infinity
                if len(dataset) > 0 and isinstance(dataset[0][field],
                                                   (int, float)):
                    # 创建一个检查数值是否有效的函数
                    def is_invalid_number(x):
                        try:
                            return isinstance(x[field], float) and (math.isnan(
                                x[field]) or math.isinf(x[field]))
                        except:
                            return False, name

                    if dataset.filter(is_invalid_number).num_rows > 0:
                        logger.warning(
                            f"{name} 数据集中 {field} 字段存在 NaN 或 Infinity 值，可能是数据增强失败"
                        )
                        return False, name

        logger.info(
            f"已加载增广数据集 ({self.dataset_name}, lr={self.labeled_ratio:.3f}): "
            f"labeled={len(self.labeled_dataset)}, "
            f"augmented={len(self.augmented_dataset)}, "
            f"unlabeled={len(self.unlabeled_dataset)}, "
            f"validation={len(self.validation_dataset)}, "
            f"test={len(self.test_dataset)}")
        return True, ""

    def _load_and_split_original(self,
                                 cache_dir: str,
                                 used_original_val: bool = True) -> None:
        """下载/读取原始数据集并作划分, used_original_val已移除"""
        cache_dataset_dir = os.path.join(cache_dir, self.dataset_name)
        if os.path.isdir(cache_dataset_dir):
            logger.info(f"本地缓存存在，读取 {cache_dataset_dir}")
            data_files = {
                "train": os.path.join(cache_dataset_dir, "train.json"),
                "validation": os.path.join(cache_dataset_dir, "test.json"),
                "test": os.path.join(cache_dataset_dir, "test.json"),
            }
            raw = load_dataset("json", data_files=data_files)
        else:
            logger.info(f"本地缓存缺失，HuggingFace Hub 下载 {self.dataset_name}")
            os.makedirs(cache_dataset_dir, exist_ok=True)
            if self.dataset_name == "mnli":
                # MNLI 数据集需要特殊处理，因为它有两个验证集
                raw = load_dataset("glue", "mnli")
                raw["train"].to_json(
                    os.path.join(cache_dataset_dir, "train.json"))
                raw["validation_matched"].to_json(
                    os.path.join(cache_dataset_dir, "validation_matched.json"))
                raw["validation_mismatched"].to_json(
                    os.path.join(cache_dataset_dir,
                                 "validation_mismatched.json"))
            elif self.dataset_name.lower() == "rte":
                # RTE 数据集是 SuperGLUE 的一部分
                raw = load_dataset("super_glue", "rte")
                raw["train"].to_json(
                    os.path.join(cache_dataset_dir, "train.json"))
                raw["validation"].to_json(
                    os.path.join(cache_dataset_dir, "validation.json"))
                if "test" in raw:
                    raw["test"].to_json(
                        os.path.join(cache_dataset_dir, "test.json"))
            else:
                # 其他数据集直接加载
                raw = load_dataset(self.dataset_name)
                raw["train"].to_json(
                    os.path.join(cache_dataset_dir, "train.json"))
                (raw.get("validation")
                 or raw.get("validation_matched")).to_json(
                     os.path.join(cache_dataset_dir, "test.json"))
                (raw.get("test") or raw.get("test_matched")).to_json(
                    os.path.join(cache_dataset_dir, "test.json"))
            logger.info("下载完成并已缓存原始数据")

        # 基本切分
        train_ds = clean_dataset(raw["train"])
        test_ds = clean_dataset(raw.get("test") or raw.get("test_matched"))

        if self.labeled_ratio == 1.0:  # 如果划分比例为 1.0，则直接使用原始数据集
            self.labeled_dataset = train_ds
            self.unlabeled_dataset = Dataset.from_dict({})
            self.validation_dataset = test_ds
            self.test_dataset = test_ds
            self.augmented_dataset = Dataset.from_dict({})
        elif self.labeled_ratio == 0.0:  # 如果划分比例为 0.0，则直接使用 unlabeled 数据集
            # 有标签数据只保留一条
            self.labeled_dataset = Dataset.from_dict({
                "premise": [
                    "Valero Energy Corp., on Monday, said it found \"extensive\" additional damage at its 250,000-barrel-per-day Port Arthur refinery."
                ],
                "hypothesis":
                ["Valero Energy Corp. produces 250,000 barrels per day."],
                "label": [0]
            })
            self.unlabeled_dataset = train_ds
            self.validation_dataset = test_ds
            self.test_dataset = test_ds
            self.augmented_dataset = Dataset.from_dict({
                "premise": [
                    "Valero Energy Corp., on Monday, said it found \"extensive\" additional damage at its 250,000-barrel-per-day Port Arthur refinery."
                ],
                "hypothesis":
                ["Valero Energy Corp. produces 250,000 barrels per day."],
                "label": [0]
            })
        else:  # 划分有标签和无标签数据集
            labeled_split = train_ds.train_test_split(
                test_size=self.labeled_ratio, seed=self.seed)
            if self.unlabeled_ratio == 0.0:
                # 如果 unlabeled_ratio 为 0，则 unlabeled_dataset 为空
                self.unlabeled_dataset = Dataset.from_dict({})
            else:
                unlabeled_split = labeled_split["train"].train_test_split(
                    test_size=self.unlabeled_ratio / (1 - self.labeled_ratio),
                    seed=self.seed)
                self.unlabeled_dataset = unlabeled_split["test"]

            self.labeled_dataset = labeled_split["test"]
            self.validation_dataset = test_ds
            self.test_dataset = test_ds
            self.augmented_dataset = None

        # 处理 unlabeled 数据集，删除其中的hypothesis字段和label字段
        if len(self.unlabeled_dataset) > 0:
            self.unlabeled_dataset = self.unlabeled_dataset.map(
                lambda example: {
                    "hypothesis": None,
                    "label": None
                },
                remove_columns=["hypothesis", "label"])

        logger.info(
            f"原始数据加载完成: \n"
            f"labeled={len(self.labeled_dataset)}, \n"
            f"unlabeled={len(self.unlabeled_dataset)}, \n"
            f"validation={len(self.validation_dataset)}, \n"
            f"augmented={len(self.augmented_dataset) if self.augmented_dataset else 0}, \n"
            f"test={len(self.test_dataset)}")

    # ---------------- 公共接口 ----------------

    def get_labeled_dataset(self) -> Dataset:
        return self.labeled_dataset

    def get_unlabeled_dataset(self) -> Dataset:
        return self.unlabeled_dataset

    def get_validation_dataset(self) -> Dataset:
        return self.validation_dataset

    def get_subdir(self) -> str:
        return self.subdir

    def get_augmented_dataset(self) -> Optional[Dataset]:
        # 如果没有增广数据集，则返回 None
        if self.augmented_dataset is None:
            logger.warning("增广数据集不存在")
            return None
        return self.augmented_dataset

    def get_test_dataset(self) -> Dataset:
        return self.test_dataset

    # --------- 可选的简易增强 & 数据追加 ---------

    # def augment_dataset(self):
    #     """用户自实现的数据增强逻辑"""

    #     # 尝试数据增强
    #     augmentor = DataAugmentor()

    #     from MRAugment import METAMORPHIC_RELATIONS, mr_synonym_replacement, mr_antonym_substitution, mr_hypernym_hyponym, mr_modal_weaken, mr_negation_flip, mr_temporal_shift, mr_voice_switch
    #     for name, func in METAMORPHIC_RELATIONS.items():
    #         augmentor.register_transformation(name, func)
    #         logger.info(f"注册蜕变关系: {name}")
    #     # 对有标签数据集进行增强
    #     self.labeled_dataset = augmentor.augment_dataset(
    #         self.get_labeled_dataset(), self, num_workers=8)

    def add_samples(self, dataset: Dataset,
                    samples: List[Dict[str, Any]]) -> Dataset:
        new_ds = Dataset.from_list(samples)
        return concatenate_datasets([dataset, new_ds])

    # ---------------- 保存函数 ----------------

    def save_augmented_dataset(self, base_dir: str = DEFAULT_AUG_ROOT) -> str:
        """按命名规则保存当前数据集，返回保存目录路径"""
        save_dir = self.augmented_dir
        os.makedirs(save_dir, exist_ok=True)

        try:
            self.labeled_dataset.to_json(
                os.path.join(save_dir, "labeled_data.json"))
        except Exception as e:
            logger.error(f"保存labeled_data.json失败: {str(e)}")

        try:
            self.unlabeled_dataset.to_json(
                os.path.join(save_dir, "unlabeled_data.json"))
        except Exception as e:
            logger.error(f"保存unlabeled_data.json失败: {str(e)}")

        try:
            self.validation_dataset.to_json(
                os.path.join(save_dir, "validation.json"))
        except Exception as e:
            logger.error(f"保存validation.json失败: {str(e)}")

        try:
            self.test_dataset.to_json(os.path.join(save_dir, "test.json"))
        except Exception as e:
            logger.error(f"保存test.json失败: {str(e)}")

        try:
            if self.augmented_dataset is not None:
                self.augmented_dataset.to_json(
                    os.path.join(save_dir, "augmented_data.json"))
        except Exception as e:
            logger.error(f"保存augmented_data.json失败: {str(e)}")

        logger.info(
            f"已将数据集保存至 {save_dir}: "
            f"labeled={len(self.labeled_dataset)}, "
            f"unlabeled={len(self.unlabeled_dataset)}, "
            f"validation={len(self.validation_dataset)}, "
            f"test={len(self.test_dataset)}, "
            f"augmented={len(self.augmented_dataset) if self.augmented_dataset is not None else 0}"
        )
        return save_dir


if __name__ == "__main__":
    # 初始化数据集
    dataset_handler = NLIDataset(dataset_name="mnli",
                                 val_ratio=0.1,
                                 labeled_ratio=0.00001,
                                 unlabeled_ratio=0.1 - 0.005,
                                 use_original_val=True,
                                 use_data_augmentation=False,
                                 llm_model_name="llama-3.2-3b-instruct")

    # 打印部分有标签数据（前5个）
    labeled_data = dataset_handler.get_labeled_dataset()
    logger.info("\n前5条有标签数据:")
    # 展示前5条
    for i in range(min(5, len(labeled_data['premise']))):
        logger.info(f"样本 {i + 1}:")
        logger.info(f"  前提 (premise): {labeled_data['premise'][i]}")
        logger.info(f"  假设 (hypothesis): {labeled_data['hypothesis'][i]}")
        logger.info(f"  标签 (label): {labeled_data['label'][i]}")

    # 添加一条伪样本到有标签数据集中
    new_samples = [{
        "premise": "All dogs bark.",
        "hypothesis": "Dogs make noise.",
        "label": 0  # 0 = entailment
    }]
    updated_dataset = dataset_handler.get_labeled_dataset(), new_samples
    logger.info(f"\n更新后的有标签数据集大小: {len(updated_dataset)}")

    # =================== 手动数据增强 ===================
    # # 尝试数据增强
    # augmentor = DataAugmentor()

    # # 示例：注册一个蜕变关系函数
    # from MRAugment import synonym_replacement

    # # 注册蜕变关系
    # augmentor.register_transformation("synonym_replacement",
    #                                   synonym_replacement)
    # # 对有标签数据集进行增强
    # dataset_handler.labeled_dataset = augmentor.augment_dataset(
    #     dataset_handler.get_labeled_dataset())
    # =================== 手动数据增强 ===================

    # 打印部分标签数据（前5个）
    labeled_data = dataset_handler.get_labeled_dataset()
    logger.info("\n前5条有标签数据:")
    for i in range(min(5, len(labeled_data['premise']))):
        logger.info(f"样本 {i + 1}:")
        logger.info(f"  前提 (premise): {labeled_data['premise'][i]}")
        logger.info(f"  假设 (hypothesis): {labeled_data['hypothesis'][i]}")
        logger.info(f"  标签 (label): {labeled_data['label'][i]}")

    # 保存增强后的数据集
    # save_dir = dataset_handler.save_augmented_dataset()
