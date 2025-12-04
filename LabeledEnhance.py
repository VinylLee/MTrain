from typing import Callable, List, Dict, Any, Optional
from datasets import Dataset
from mylogging import logger
from multiprocessing import Pool

from MRAugment import NLIMRTool

from tqdm import tqdm


class DataAugmentor:
    """
    DataAugmentor类
    ---------------
    用于对NLI数据集进行数据增强。通过注册不同的蜕变关系函数，
    可以对每个数据样本应用多种增强方式，生成增强后的数据集。

    功能：
    - 注册蜕变关系函数
    - 对单个数据样本应用所有蜕变关系
    - 对整个数据集进行批量增强

    方法：
    - register_transformation: 注册蜕变关系函数
    - apply_transformations: 对单个数据样本应用所有蜕变关系
    - augment_dataset: 对整个数据集进行增强
    - augment_unlabeled_dataset: 对未标注数据集进行增强
    """

    def __init__(self, nli_mr_tool: Optional[NLIMRTool] = None):
        """
        初始化DataAugmentor类。
        """
        if nli_mr_tool is None:
            nli_mr_tool = NLIMRTool()
        self.transformations: dict = {}  # 存储蜕变关系函数的字典
        self.nli_mr_tool = nli_mr_tool
        logger.info("DataAugmentor 初始化完成，准备注册蜕变关系函数。")

    def clear_transformations(self):
        """
        清空所有注册的蜕变关系函数。
        """
        self.transformations = {}
        logger.info("所有蜕变关系函数已清空。")

    def register_transformation(self, name: str, func: Callable):
        """
        注册一个蜕变关系函数。

        参数：
        - name (str): 蜕变关系的名称
        - func (Callable): 蜕变关系函数，接受一个数据样本，返回增强后的数据样本

        示例：
        >>> def synonym_replacement(sample):
        >>>     # 实现同义词替换逻辑
        >>>     return sample
        >>> augmentor.register_transformation("synonym_replacement", synonym_replacement)
        """
        self.transformations[name] = func
        logger.info(f"蜕变关系 '{name}' 已注册。")

    def augment_dataset(
            self,
            NLIdataset: "NLIDataset",  # type: ignore[valid-type, misc]
            num_workers: int = 1,
            labeled: bool = True) -> Dataset:
        """
        对整个数据集进行增强，支持并行处理。num_workers已废弃。

        参数：
        - NLIdataset (NLIDataset): NLIDataset对象，包含原始数据集

        返回：
        - Dataset: 增强后的数据集，包含原始数据和增强数据
        """
        logger.info("开始对数据集进行增强...")
        if len(self.transformations) == 0:
            logger.warning("没有注册任何蜕变关系，无法进行数据增强。")
            raise ValueError("没有注册任何蜕变关系，无法进行数据增强。")

        logger.info(f"注册的蜕变关系数量: {len(self.transformations)}")
        logger.info(f"注册的蜕变关系名称: {list(self.transformations.keys())}")
        if labeled:
            dataset = NLIdataset.get_labeled_dataset()
            logger.info(f"原始数据集大小: {len(dataset)}")
        else:
            dataset = NLIdataset.get_unlabeled_dataset()
            logger.info(f"原始未标注数据集大小: {len(dataset)}")

        all_augmented_samples = []

        for sample in tqdm(dataset, desc="增强进度"):
            if labeled:
                result = self.process_sample(sample)
            else:
                result = self.process_unlabeled_sample(sample)
            if result:  # 确保结果不为None
                all_augmented_samples.extend(result)
        if labeled:
            logger.info(f"数据增强完成，共生成 {len(all_augmented_samples)} 条增强样本。")
        else:
            logger.info(f"未标注数据增强完成，共生成 {len(all_augmented_samples)} 条增强样本。")
        augmented_dataset = Dataset.from_list(all_augmented_samples)
        return augmented_dataset

    def process_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        对单个样本进行增强的辅助函数。
        参数：
        - sample (Dict[str, Any]): 单个数据样本，包含 'premise', 'hypothesis', 'label'
        返回：
        - List[Dict[str, Any]]: 增强后的数据样本列表，每个样本对应一种蜕变关系的结果，包含本身
        """
        result_augmented_samples = [sample]  # 先放原始

        augmented_samples = []
        # logger.info(f"开始对样本进行增强：{sample}")

        # NOTE 将数据集中的标签与三种关系进行映射
        # snli中0, 1, 2分别对应entailment, neutral, contradiction
        sample["label"] = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }[sample["label"]]
        for name, func in self.transformations.items():
            try:
                # 调用蜕变关系函数
                the_other = None
                if name in self.nli_mr_tool.DUAL_MRS:
                    augmented_sample, the_other = func(self.nli_mr_tool,
                                                       sample)
                else:
                    augmented_sample = func(self.nli_mr_tool, sample)
                if augmented_sample is None or augmented_sample == "":
                    # logger.info(f"蜕变关系 '{name}' 返回了 None，跳过该蜕变关系。")
                    # breakpoint()
                    pass
                else:
                    # NOTE 将数据集中的标签与三种关系进行反映射
                    # snli中entailment, neutral, contradiction分别对应0, 1, 2
                    if the_other is None:
                        augmented_sample["label"] = {
                            "entailment": 0,
                            "neutral": 1,
                            "contradiction": 2
                        }[augmented_sample["label"]]
                        augmented_sample["is_syn"] = True
                        augmented_sample["mr_id"] = name
                        augmented_sample["mr_type"] = self.nli_mr_tool.MR_TYPE[
                            name]
                        augmented_sample["weight"] = 1.0
                        augmented_sample["pair_id"] = sample["pair_id"]
                        augmented_samples.append(augmented_sample)
                    else:
                        # 如果是双向蜕变关系，添加两个样本
                        augmented_sample["label"] = {
                            "entailment": 0,
                            "neutral": 1,
                            "contradiction": 2
                        }[augmented_sample["label"]]
                        augmented_sample["is_syn"] = True
                        augmented_sample["mr_id"] = name
                        augmented_sample["mr_type"] = self.nli_mr_tool.MR_TYPE[
                            name].split("_")[0]
                        augmented_sample["weight"] = 1.0
                        augmented_sample["pair_id"] = sample["pair_id"]
                        augmented_samples.append(augmented_sample)

                        the_other["label"] = {
                            "entailment": 0,
                            "neutral": 1,
                            "contradiction": 2
                        }[the_other["label"]]
                        the_other["is_syn"] = True
                        the_other["mr_id"] = name
                        the_other["mr_type"] = self.nli_mr_tool.MR_TYPE[
                            name].split("_")[1]
                        the_other["weight"] = 1.0
                        the_other["pair_id"] = sample["pair_id"]
                        augmented_samples.append(the_other)
                    # logger.info(
                    #     f"蜕变关系 '{name}' 应用成功: \n{sample} -> \n{augmented_sample}"
                    # )
            except Exception as e:
                # logger.error(f"蜕变关系 '{name}' 应用失败：{e}")
                pass
        sample["label"] = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }[sample["label"]]

        result_augmented_samples.extend(augmented_samples)
        return result_augmented_samples

    def process_unlabeled_sample(
            self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        对单个未标注样本进行增强的辅助函数。
        
        参数：
        - sample (Dict[str, Any]): 单个未标注数据样本，包含 'premise', 'hypothesis'
        - dataAugmentor: DataAugmentor实例
        
        返回：
        - List[Dict[str, Any]]: 增强后的数据样本列表，包含本身和各种蜕变关系结果
        """
        result_augmented_samples = [sample]  # 先放原始

        augmented_samples = []
        # logger.info(f"开始对样本进行增强：{sample}")

        # NOTE 将数据集中的标签与三种关系进行映射
        # snli中0, 1, 2分别对应entailment, neutral, contradiction
        sample["label"] = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }[sample["label"]]
        for name, func in self.transformations.items():
            try:  # 调用蜕变关系函数
                the_other = None
                # 如果是双向蜕变关系，返回两个样本
                if name in self.nli_mr_tool.DUAL_MRS:
                    augmented_sample, the_other = func(self.nli_mr_tool,
                                                       sample)
                else:
                    augmented_sample = func(self.nli_mr_tool, sample)
                if augmented_sample is None or augmented_sample == "":
                    # logger.info(f"蜕变关系 '{name}' 返回了 None，跳过该蜕变关系。")
                    # breakpoint()
                    pass
                else:
                    # NOTE 将数据集中的标签与三种关系进行反映射
                    # snli中entailment, neutral, contradiction分别对应0, 1, 2
                    if the_other is None:
                        augmented_sample["label"] = {
                            "entailment": 0,
                            "neutral": 1,
                            "contradiction": 2
                        }[augmented_sample["label"]]
                        augmented_sample["is_syn"] = True
                        augmented_sample["mr_id"] = name
                        augmented_sample["mr_type"] = self.nli_mr_tool.MR_TYPE[
                            name]
                        augmented_sample["weight"] = 1.0
                        augmented_sample["pair_id"] = sample["pair_id"]
                        augmented_samples.append(augmented_sample)
                    else:  # 如果是双向蜕变关系，添加两个样本
                        augmented_sample["label"] = {
                            "entailment": 0,
                            "neutral": 1,
                            "contradiction": 2
                        }[augmented_sample["label"]]
                        augmented_sample["is_syn"] = True
                        augmented_sample["mr_id"] = name
                        augmented_sample["mr_type"] = self.nli_mr_tool.MR_TYPE[
                            name].split("_")[0]
                        augmented_sample["weight"] = 1.0
                        augmented_sample["pair_id"] = sample["pair_id"]
                        augmented_samples.append(augmented_sample)

                        the_other["label"] = {
                            "entailment": 0,
                            "neutral": 1,
                            "contradiction": 2
                        }[the_other["label"]]
                        the_other["is_syn"] = True
                        the_other["mr_id"] = name
                        the_other["mr_type"] = self.nli_mr_tool.MR_TYPE[
                            name].split("_")[1]
                        the_other["weight"] = 1.0
                        the_other["pair_id"] = sample["pair_id"]
                        augmented_samples.append(augmented_sample)
                        augmented_samples.append(the_other)
                    # logger.info(
                    #     f"蜕变关系 '{name}' 应用成功: \n{sample} -> \n{augmented_sample}"
                    # )
            except Exception as e:
                # logger.error(f"蜕变关系 '{name}' 应用失败：{e}")
                pass
        sample["label"] = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }[sample["label"]]

        result_augmented_samples.extend(augmented_samples)
        return result_augmented_samples


if __name__ == "__main__":
    from NLIdataset import NLIDataset
    from multiprocessing import Pool
    import time

    # 初始化NLIDataset
    dataset_handler = NLIDataset(dataset_name="snli",
                                 labeled_ratio=0.00001,
                                 unlabeled_ratio=0.1 - 0.005,
                                 use_original_val=True)

    # 获取有标签数据集
    labeled_dataset = dataset_handler.get_labeled_dataset()

    # 初始化DataAugmentor
    augmentor = DataAugmentor()

    # 注册蜕变关系
    augmentor.register_transformation("antonym_substitution",
                                      NLIMRTool.mr_antonym_substitution)

    # 测试并行处理是否有效
    start_time = time.time()

    # 对有标签数据集进行增强，使用并行处理
    augmented_dataset_parallel = augmentor.augment_dataset(dataset_handler,
                                                           num_workers=1)

    parallel_time = time.time() - start_time
    logger.info(f"并行处理耗时: {parallel_time:.2f} 秒")

    # 测试单线程处理耗时
    start_time = time.time()

    # 对有标签数据集进行增强，使用单线程处理
    augmented_dataset_single = augmentor.augment_dataset(labeled_dataset,
                                                         num_workers=1)

    single_time = time.time() - start_time
    logger.info(f"单线程处理耗时: {single_time:.2f} 秒")

    # 比较耗时
    logger.info(f"并行处理比单线程快 {single_time / parallel_time:.2f} 倍")

    # 打印增强后的数据集
    # logger.info("增强后的数据集：")
    # for sample in augmented_dataset:
    #     logger.info(sample)
