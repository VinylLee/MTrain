from __future__ import annotations
"""
NLI 变形增强工具包（英文版）
----------------------------------------------------------------
本模块实现了一组 **变形关系（MRs）**，用于对英文自然语言推理（NLI）数据集进行变形数据增强。
它面向支持显式 *思考* 段落的 `deepseek‑r1:7b` 模型（通过 ``OllamaChat`` 调用）。
因此，prompt 会指示模型将其逐步推理内容包裹在 `<think>` … `</think>` 标签内，并在标签之后给出最终答案。
辅助函数 ``_transform`` 会剥离思考部分，使下游代码仅接收到增强后的句子。

主要特性
~~~~~~~~~~~~
* 针对 SNLI、MNLI 等数据集的英文 prompt
* **基于标签的推理分离**：LLM 返回 `<think>` 段 + 答案，工具包只保留答案部分
* **Prompt 复用**：统一构造器复用
* **自动标签更新**：在必要时自动更新标签
* **注册表模式**：保持公共 API 稳定且易于扩展
"""

from typing import Any, Dict, Callable, Optional
import re, NLIutils
from LMstudio import OpenAIChat, HOST, PORT, MODEL


class NLIMRTool:

    def __init__(self, llm: Optional[OpenAIChat] = None):
        """
        初始化变形增强工具包，传入一个 LLM 用于生成变换句子。

        参数:
            llm: 一个调用 LLM 的函数，接受一个 prompt 并返回响应。
        """
        # 真正需要时才创建
        if llm is None:
            llm = OpenAIChat(MODEL, HOST, PORT)
        self.llm = llm
        self.llm = llm
        self._TRY_LIMIT = 5  # LLM 生成失败重试次数

    def _build_prompt(self, task_instruction: str, sentence: str) -> str:
        """
        构造发送给 LLM 的完整 prompt。
        
        参数:
            task_instruction: 描述任务指令的字符串
            sentence: 需要转换的原始句子

        返回:
            完整格式化的 prompt 字符串
        """
        _BASE_PROMPT_TEMPLATE = """{task}
Instructions:
1. Output ONLY the transformed English sentence, without
   quotation marks or extra commentary.
"""
        return _BASE_PROMPT_TEMPLATE.format(
            task=task_instruction.strip()) + (f"\nSentence:\n{sentence}\n")

    def _call_llm(self, prompt: str) -> str:
        """
        将 prompt 发送给 LLM 并返回原始响应内容（去除多余空格）。

        参数:
            prompt: 构造好的 prompt 文本

        返回:
            LLM 返回的字符串结果（不含推理部分）
        """
        res: str = self.llm.chat(prompt)
        clean = re.sub(r"<think>[\s\S]*?</think>",
                       "",
                       res.strip(),
                       flags=re.IGNORECASE)
        return clean.strip()

    def _transform(self, sentence: str,
                   task_instruction: str) -> Optional[str]:
        """
        执行变换任务，提取并返回句子转换结果（去除 `<think>` 段落）。

        若输出异常（如过长、保留推理过程等），最多尝试 _TRY_LIMIT 次。

        参数:
            sentence: 原始输入句子
            task_instruction: 任务描述，用于生成 prompt

        返回:
            转换后的句子（不含推理部分）

        异常:
            若连续多次失败，抛出 RuntimeError
        """
        for _ in range(self._TRY_LIMIT):
            raw = self._call_llm(self._build_prompt(task_instruction,
                                                    sentence))
            clean = re.sub(r"<think>[\s\S]*?</think>",
                           "",
                           raw,
                           flags=re.IGNORECASE).strip()
            if len(clean) <= len(sentence) * 3:
                return clean
        else:
            return None

    # ---- 不改变标签的 MR --------------------------------------------------------

    def mr_back_translation(
        self,
        sample: Dict[str, Any],
        fields: str = "hypothesis",
    ) -> Dict[str, Any] | None:
        """
        **标签不变** — 对指定字段进行回译增强。

        参数:
            sample: 包含 'premise', 'hypothesis', 'label' 的样本字典。
            fields: 指定要回译的字段，可选值为 "premise", "hypothesis", 或 "both"
            similarity_threshold: 相似度阈值，如果回译结果与原句相似度高于此值，则放弃
            semantic_similarity_threshold: 语义相似度阈值，若回译结果与原句语义相似度低于此值，则放弃

        返回:
            回译增强后的新样本；
            若多次验证失败或相似度检查不通过，则返回 None。
        """
        new_sample = sample.copy()

        # 确定要处理的字段列表
        fields_to_process = []
        if fields == "both":
            fields_to_process = ["premise", "hypothesis"]
        elif fields in ["premise", "hypothesis"]:
            fields_to_process = [fields]
        else:
            raise ValueError(
                "fields must be 'premise', 'hypothesis', or 'both'")

        for field in fields_to_process:
            for _ in range(self._TRY_LIMIT):
                # 使用NLIutils中的回译功能
                translated = NLIutils.augment_back_translate(
                    sample[field], NLIutils.back_translate(sample[field]))
                # 二次验证：判断回译的句子与原句的语义相似性
                check = self._call_llm(
                    self._build_prompt(
                        "Are the following sentence similar to each other? "
                        "Answer with Yes or No.",
                        f"First sentence: {sample[field]}\nSecond sentence: {translated}"
                    ))
                if "yes" in check.strip().lower():
                    new_sample[field] = translated
                    break
            # 检查回译后的句子是否与原句相似度过高
            if new_sample[field] == sample[field]:
                return None

        return new_sample

    def mr_voice_switch(self, sample: Dict[str,
                                           Any]) -> Optional[Dict[str, Any]]:
        """
        **标签不变** — 对 *hypothesis* 进行主动语态 ↔ 被动语态 转换。

        参数:
            sample: 原始样本字典

        返回:
            语态转换后的新样本（双向蕴含验证通过）
        """
        task = (
            "Convert the sentence to the opposite voice (active ↔ passive) "
            "without introducing new agents; use agentless passive if needed.")
        new_sample = sample.copy()

        for _ in range(self._TRY_LIMIT):
            new_premise = self._transform(sample["hypothesis"], task)
            if new_premise is None or new_premise == sample["hypothesis"]:
                continue
            check = self._call_llm(
                self._build_prompt(
                    "Does the following pair mutually entail each other? "
                    "Answer strictly Yes or No.",
                    f"Sentence A: {sample['hypothesis']}\nSentence B: {new_premise}"
                ))

            if "yes" in check.strip().lower():
                new_sample["hypothesis"] = new_premise
                return new_sample
        else:
            return None

    def mr_synonym_replacement(
            self,
            sample: Dict[str, Any],
            fields: str = "hypothesis") -> Optional[Dict[str, Any]]:
        """
        **标签不变** — 使用同义词对 *premise* 和 *hypothesis* 进行释义替换。

        参数:
            sample: 原始样本字典，包含 premise、hypothesis 和 label

        返回:
            同义词替换后的新样本
        """
        fields_to_process = []
        if fields == "both":
            fields_to_process = ["premise", "hypothesis"]
        elif fields in ["premise", "hypothesis"]:
            fields_to_process = [fields]
        else:
            raise ValueError(
                "fields must be 'premise', 'hypothesis', or 'both'")
        task = (
            "Paraphrase the sentence by substituting 1-2 words with appropriate synonyms. "
            "Rules: 1) Only replace common nouns, adjectives, or verbs "
            "2) Keep the EXACT same meaning and semantic relationships "
            "3) Do NOT modify named entities (proper nouns, names, places) "
            "4) Ensure the replacement words are natural and contextually appropriate "
            "5) Maintain grammatical correctness and sentence structure")
        new_sample = sample.copy()

        for field in fields_to_process:
            for _ in range(self._TRY_LIMIT):
                new_sent = self._transform(sample[field], task)
                if new_sent is None or new_sent == sample[field]:
                    continue
                check = self._call_llm(
                    self._build_prompt(
                        "Does the following pair mutually entail each other? "
                        "Answer strictly with Yes or No.",
                        f"Sentence A: {sample[field]}\nSentence B: {new_sent}")
                )
                if "yes" in check.strip().lower():
                    new_sample[field] = new_sent
                    break
            else:
                return None
        return new_sample

    def mr_extracting_snippets(
            self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        无标签MR
        **标签不变/变更为entailment** — 从 premise 中提取片段作为 hypothesis，创建蕴含关系。
        
        此方法通过组合应用多种提取策略到同一个句子上，生成高度压缩的hypothesis。
        
        参数:
            sample: 原始样本字典，包含 premise、hypothesis 和 label
            
        返回:
            提取片段后的新样本（标签为 entailment）
        """
        premise = sample["premise"]
        new_sample = sample.copy()

        # 定义组合策略的层次化应用方法
        combined_strategies = [
            # 策略组合1: 去除修饰语 + 提取核心
            "Remove adjective modifiers, color descriptions, and quantity information while extracting only the subject and main verb.",

            # 策略组合2: 去除状语 + 简化结构
            "Remove location, temporal, and manner information while simplifying to the main clause.",

            # 策略组合3: 提取动作核心
            "Extract only the main action and its direct object, removing all descriptive modifiers.",

            # 策略组合4: 最小化表达
            "Extract the core relationship between main entities, removing all non-essential information.",

            # 策略组合5: 主谓宾结构
            "Simplify to basic subject-verb-object structure, removing all modifiers and subordinate clauses."
        ]

        # 尝试渐进式策略应用
        for _ in range(self._TRY_LIMIT):
            # 随机选择一个组合策略
            import random
            strategy = random.choice(combined_strategies)

            task = (
                f"{strategy} "
                "Ensure the result is a grammatically correct, much shorter sentence "
                "that preserves only the essential meaning.")

            new_hypothesis = self._transform(premise, task)

            # 检查是否成功提取且明显短于原句
            if (new_hypothesis is None or new_hypothesis == premise
                    or len(new_hypothesis) >= len(premise) * 0.8):  # 要求至少压缩20%
                continue

            # 验证premise是否蕴含新的hypothesis
            check = self._call_llm(
                self._build_prompt(
                    "Does the premise entail the hypothesis? "
                    "Answer strictly with Yes or No.",
                    f"Premise: {premise}\nHypothesis: {new_hypothesis}"))

            if "yes" in check.strip().lower():
                new_sample["hypothesis"] = new_hypothesis
                new_sample["label"] = "entailment"
                return new_sample

        # 如果组合策略失败，尝试更激进的片段提取
        for _ in range(self._TRY_LIMIT):
            aggressive_task = (
                "Extract the shortest possible phrase or clause that still makes grammatical sense "
                "and preserves some core meaning from the original sentence. "
                "Remove everything non-essential.")

            new_hypothesis = self._transform(premise, aggressive_task)

            # 更宽松的长度要求，但仍需要显著压缩
            if (new_hypothesis is None or new_hypothesis == premise
                    or len(new_hypothesis) >= len(premise) * 0.9):
                continue

            # 验证蕴含关系
            check = self._call_llm(
                self._build_prompt(
                    "Does the premise entail the hypothesis? "
                    "Answer strictly with Yes or No.",
                    f"Premise: {premise}\nHypothesis: {new_hypothesis}"))

            if "yes" in check.strip().lower():
                new_sample["hypothesis"] = new_hypothesis
                new_sample["label"] = "entailment"
                return new_sample

        return None  # 所有尝试都失败

    def mr_hypernym_substitution(
        self, sample: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        无标签MR
        **超词替换（Hypernym Substitution, HS）** — 将前提(premise)中的具体名词替换为其上位词(hypernym)，
        创建新的蕴含关系样本。这种转换使得原假设能够从新前提中推出，形成entailment关系。
        
        例如：
        - 原前提: "A dog is running in the park"
        - 新前提: "An animal is running in the park" (dog -> animal)
        - 假设保持不变，标签变为entailment
        
        参数:
            sample: 原始样本字典
        返回:
            进行超词替换后的新样本（标签为entailment），副产物（前提和假设对调，标签为中立）
        """
        new_sample = sample.copy()

        for _ in range(self._TRY_LIMIT):
            # 对前提进行超词替换
            task = (
                "Replace ONE specific noun with its hypernym (more general category) "
                "to make the sentence more general while keeping it grammatically correct. "
                "For example, replace 'dog' with 'animal', 'car' with 'vehicle', "
                "'apple' with 'fruit', etc. Make only ONE substitution.")

            new_premise = self._transform(sample["premise"], task)
            if new_premise is None or new_premise == sample["premise"]:
                continue

            new_sample["hypothesis"] = new_premise

            # 二次验证：检查新前提与原假设之间的蕴含关系
            verify_prompt = (
                "Determine the logical relationship between the following premise and hypothesis. "
                "The premise should be more general than the hypothesis, allowing the hypothesis to be inferred. "
                "Respond **ONLY** with one word: entailment, contradiction, or neutral."
            )

            check = self._call_llm(
                self._build_prompt(
                    verify_prompt,
                    f"Premise: {new_premise}\nHypothesis: {sample['hypothesis']}"
                ))

            # 检查是否形成蕴含关系
            if "entailment" in check.strip().lower():
                new_sample["label"] = "entailment"
                # 交换前提和假设，创建中立关系样本
                swapped_sample = {
                    "premise": sample["hypothesis"],
                    "hypothesis": new_premise,
                    "label": "neutral"
                }
                return new_sample, swapped_sample

        return None, None  # 多次尝试后仍未成功

    def mr_pronoun_substitution(
            self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        **代词替换（Pronoun Substitution, PS）** — 将句子中的名词替换为相应的代词，
        保持句子的语义和语法正确性。此变换不改变原始标签。
        
        例如：
        - "The boy is dancing in arena" -> "He is dancing in arena"
        - "The woman walked her dog" -> "She walked her dog"
        
        参数:
            sample: 原始样本字典，包含 premise、hypothesis 和 label
            
        返回:
            进行代词替换后的新样本（标签不变）
        """
        new_sample = sample.copy()

        for _ in range(self._TRY_LIMIT):
            task = (
                "Replace ONE noun with its appropriate pronoun while maintaining "
                "grammatical correctness and the same meaning. For example, replace "
                "'the boy' with 'he', 'the woman' with 'she', 'the car' with 'it', etc. "
                "Make sure the sentence remains grammatically correct and natural."
            )

            new_sent = self._transform(sample["hypothesis"], task)
            if new_sent is None or new_sent == sample["hypothesis"]:
                continue

            # 验证替换后的句子与原句语义等价
            check = self._call_llm(
                self._build_prompt(
                    "Do the following two sentences have the same meaning? "
                    "Answer strictly with Yes or No.",
                    f"Sentence A: {sample['hypothesis']}\nSentence B: {new_sent}"
                ))

            if "yes" in check.strip().lower():
                new_sample["hypothesis"] = new_sent
                break

        # 检查是否至少有一个字段成功替换
        if (new_sample["hypothesis"] == sample["hypothesis"]):
            return None

        # 最终验证：确保整体逻辑关系保持不变
        verify_check = self._call_llm(
            self._build_prompt(
                "What is the relationship between the following premise and hypothesis? "
                "Respond **ONLY** with one word: entailment, contradiction, or neutral.",
                f"Premise: {new_sample['premise']}\nHypothesis: {new_sample['hypothesis']}"
            ))

        # 确保标签保持不变
        if sample["label"].lower() in verify_check.strip().lower():
            return new_sample
        else:
            return None

    def mr_counting(
        self, sample: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        无标签MR
        **计数蕴含（Counting Entailment, CT）** — 从前提中识别具有共同上位词的名词，
        进行计数，并使用模板生成蕴含假设。例如："有{数量}个{上位词}存在"。
        此方法生成正确计数的假设，创建蕴含关系。
        
        例如：
        - 前提: "A motorbike and a car are parked"
        - 假设: "Two automobiles are parked" (数量正确：摩托车和汽车都是交通工具)
        
        参数:
            sample: 原始样本字典
        返回:
            生成计数蕴含假设后的新样本（标签为entailment）与副产物（前提和假设对调，标签为矛盾）
        """
        premise = sample["premise"]
        new_sample = sample.copy()

        # 定义多种计数模板
        counting_templates = [
            "There are {count} {hypernym}s present",
            "There are {count} {hypernym}s", "{count} {hypernym}s are present",
            "{count} {hypernym}s exist", "You can see {count} {hypernym}s",
            "{count} {hypernym}s are there"
        ]

        for _ in range(self._TRY_LIMIT):
            # 要求LLM识别可计数的名词并找到合适的上位词
            analysis_task = (
                "Identify countable nouns in the sentence that share a common hypernym (category). "
                "For example: 'dog' and 'cat' -> 'animal', 'car' and 'bike' -> 'vehicle', "
                "'apple' and 'orange' -> 'fruit'. "
                "Count them and provide the result as: 'COUNT:number HYPERNYM:category'. "
                "If no suitable grouping exists, respond with 'NONE'.")

            analysis_result = self._call_llm(
                self._build_prompt(analysis_task, premise))

            # 解析分析结果
            if "NONE" in analysis_result.upper(
            ) or "COUNT:" not in analysis_result.upper():
                continue

            try:
                # 提取计数和上位词
                import re
                count_match = re.search(r'COUNT:\s*(\d+)',
                                        analysis_result.upper())
                hypernym_match = re.search(r'HYPERNYM:\s*(\w+)',
                                           analysis_result.upper())

                if not count_match or not hypernym_match:
                    continue

                count = count_match.group(1)
                hypernym = hypernym_match.group(1).lower()

                # 随机选择一个模板
                import random
                template = random.choice(counting_templates)

                # 生成假设
                new_hypothesis = template.format(count=count,
                                                 hypernym=hypernym)

                # 验证前提是否蕴含新假设
                verify_prompt = (
                    "Does the premise entail the hypothesis? Consider whether the count "
                    "and category are accurate based on the premise. "
                    "Answer strictly with Yes or No.")

                check = self._call_llm(
                    self._build_prompt(
                        verify_prompt,
                        f"Premise: {premise}\nHypothesis: {new_hypothesis}"))

                if "yes" in check.strip().lower():
                    new_sample["hypothesis"] = new_hypothesis
                    new_sample["label"] = "entailment"
                    # 创建副产物：前提和假设对调，标签为矛盾
                    swapped_sample = {
                        "premise": new_hypothesis,
                        "hypothesis": premise,
                        "label": "contradiction"
                    }
                    return new_sample, swapped_sample

            except Exception:
                continue

        return None, None  # 所有尝试都失败

    # ---- 翻转标签的 MR --------------------------------------------------------

    def mr_negation_flip(self, sample: Dict[str,
                                            Any]) -> Optional[Dict[str, Any]]:
        """
        **标签翻转** — 对 *hypothesis* 取逻辑否定；entailment -> contradiction。
        只对 entailment 标签的样本进行处理，

        参数:
            sample: 原始样本字典

        返回:
            否定后的新样本
        """
        if sample["label"] != "entailment":
            return None

        task = "Rewrite the sentence into its logical negation, preserving grammar."
        new_sample = sample.copy()
        original = sample["hypothesis"]

        for _ in range(self._TRY_LIMIT):
            negated = self._transform(original, task)
            if negated is None or negated == original:
                continue
            check = self._call_llm(
                self._build_prompt(
                    "What is the relationship between the following premise and hypothesis? "
                    "Choose one: entailment, contradiction, or neutral.",
                    f"Premise: {sample['premise']}\nHypothesis: {negated}"))

            if "contradiction" in check.lower():
                new_sample["hypothesis"] = negated
                new_sample["label"] = "contradiction"
                break
        else:
            return None
        return new_sample

    def mr_antonym_substitution(
            self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        **标签翻转** — 将 *hypothesis* 中的一个关键形容词或动词替换为其直接反义词，
        以改变其逻辑含义，从而使原始标签从 entailment 到 contradiction。
        只对 entailment 标签的样本进行处理。

        参数:
            sample: 包含 'premise'、'hypothesis' 和 'label' 的输入样本字典。

        返回:
            修改后的样本字典（'hypothesis' 进行了反义替换）；
            若标签为 neutral 或多次验证失败，则返回 None。
        """
        if sample["label"] != "entailment":
            return None

        task = (
            "Replace ONE key adjective or verb in the sentence with its "
            "direct antonym so that the new meaning is logically opposite.")
        expected_label = "contradiction"
        new_sample = sample.copy()

        for _ in range(self._TRY_LIMIT):
            new_hypothesis = self._transform(sample["hypothesis"], task)
            if new_hypothesis is None or new_hypothesis == sample["hypothesis"]:
                continue
            # 二次验证：使用 LLM 判断修改后的句子与 premise 的逻辑关系
            check = self._call_llm(
                self._build_prompt(
                    "What is the relationship between the following premise and hypothesis? "
                    "Respond **ONLY** with one word: entailment, contradiction, or neutral.",
                    f"Premise: {sample['premise']}\nHypothesis: {new_hypothesis}"
                ))

            if expected_label in check.strip().lower():
                new_sample["hypothesis"] = new_hypothesis
                new_sample["label"] = expected_label
                return new_sample  # 验证成功

        return None  # 多次尝试失败，放弃样本

    def mr_subject_object_swap(
            self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        **标签翻转** — 交换 *hypothesis* 中的主语和宾语，创建矛盾关系。
        将句子中的主语和宾语位置对调，生成逻辑上矛盾的假设。

        例如：
        - "A clock is standing on top of a concrete pillar" 
        -> "A pillar is standing on top of a concrete clock"

        参数:
            sample: 原始样本字典

        返回:
            主语宾语交换后的新样本（标签为contradiction）
        """
        task = (
            "Swap the subject and object in the sentence while keeping the same verb and prepositions. "
            "This is not like converting the sentence to the opposite voice (active to passive)!"
            "For example, change 'A clock is standing on top of a pillar' to "
            "'A pillar is standing on top of a clock'. "
            "Maintain grammatical correctness and ensure the meaning becomes contradictory. "
        )

        # 只对 entailment 标签的样本进行处理
        if sample["label"].lower() != "entailment":
            return None

        new_sample = sample.copy()

        for _ in range(self._TRY_LIMIT):
            swapped_hypothesis = self._transform(sample["hypothesis"], task)
            if swapped_hypothesis is None or swapped_hypothesis == sample[
                    "hypothesis"]:
                continue

            # 验证交换后的句子与原前提是否形成矛盾关系
            check = self._call_llm(
                self._build_prompt(
                    "What is the relationship between the following premise and hypothesis? "
                    "Respond **ONLY** with one word: entailment, contradiction, or neutral.",
                    f"Premise: {sample['premise']}\nHypothesis: {swapped_hypothesis}"
                ))

            if "contradiction" in check.strip().lower():
                new_sample["hypothesis"] = swapped_hypothesis
                new_sample["label"] = "contradiction"
                return new_sample

        return None  # 多次尝试失败

    def mr_number_substitution(
            self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        无标签MR
        **标签翻转** — 替换 *hypothesis* 中的数字，创建矛盾关系。
        只对 entailment 标签的样本进行处理。
        
        例如：
        - "Car has four red lights" -> "Car has seven red lights"
        
        参数:
            sample: 原始样本字典

        返回:
            数字替换后的新样本（标签为contradiction）
        """
        # 只对 entailment 标签的样本进行处理
        if sample["label"].lower() != "entailment":
            return None

        task = (
            "Replace ONE number in the sentence with a different number. "
            "For example, change 'four' to 'seven', '3' to '8', 'A/An' to 'two', etc. "
            "Keep the rest of the sentence exactly the same. ")

        new_sample = sample.copy()

        for _ in range(self._TRY_LIMIT):
            modified_hypothesis = self._transform(sample["hypothesis"], task)
            if modified_hypothesis is None or modified_hypothesis == sample[
                    "hypothesis"]:
                continue

            # 验证修改后的句子与原前提是否形成矛盾关系
            check = self._call_llm(
                self._build_prompt(
                    "What is the relationship between the following premise and hypothesis? "
                    "Respond **ONLY** with one word: entailment, contradiction, or neutral.",
                    f"Premise: {sample['premise']}\nHypothesis: {modified_hypothesis}"
                ))

            if "contradiction" in check.strip().lower():
                new_sample["hypothesis"] = modified_hypothesis
                new_sample["label"] = "contradiction"
                return new_sample

        return None  # 多次尝试失败

    def mr_adding_contradiction(
            self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        **标签翻转** — 在 *hypothesis* 中添加与 *premise* 矛盾的信息，创建矛盾关系。
        此方法通过在假设中加入否定性或对立性信息，使其与前提形成逻辑矛盾。
        只对 entailment 标签的样本进行处理。
        
        例如：
        - 前提: "A man is walking in the park"
        - 原假设: "A man is walking" (entailment)
        - 新假设: "A man is walking and there are no people around" (contradiction)
        
        参数:
            sample: 原始样本字典

        返回:
            添加矛盾信息后的新样本（标签为contradiction）
        """
        # 只对 entailment 标签的样本进行处理
        if sample["label"].lower() != "entailment":
            return None

        new_sample = sample.copy()

        # 定义多种矛盾信息添加策略
        contradiction_strategies = [
            # 策略1: 添加否定性存在信息
            "Add contradictory information about the absence or non-existence of key elements mentioned in the premise.",

            # 策略2: 添加相反的状态描述
            "Add information that describes an opposite state or condition to what is implied in the premise.",

            # 策略3: 添加时间矛盾
            "Add temporal information that contradicts the timing or sequence implied in the premise.",

            # 策略4: 添加空间矛盾
            "Add location or spatial information that contradicts the setting described in the premise.",

            # 策略5: 添加数量矛盾
            "Add quantity information that contradicts the count or amount implied in the premise.",

            # 策略6: 添加属性矛盾
            "Add descriptive attributes that contradict the characteristics mentioned in the premise."
        ]

        # 尝试不同的矛盾添加策略
        for _ in range(self._TRY_LIMIT):
            # 随机选择一个矛盾策略
            import random
            strategy = random.choice(contradiction_strategies)

            task = (
                f"{strategy} "
                "Extend the sentence by adding contradictory information while keeping "
                "the original statement intact. The addition should create a logical "
                "contradiction with the original statement.")

            new_hypothesis = self._transform(sample["hypothesis"], task)

            # 检查是否成功添加且不同于原句
            if (new_hypothesis is None
                    or new_hypothesis == sample["hypothesis"]
                    or len(new_hypothesis) <= len(
                        sample["hypothesis"])):  # 应该变得更长
                continue

            # 验证添加矛盾信息后hypothesis与premise的关系是否变为矛盾
            check = self._call_llm(
                self._build_prompt(
                    "What is the relationship between the following premise and hypothesis? "
                    "The hypothesis should contain information that contradicts the premise. "
                    "Respond **ONLY** with one word: entailment, contradiction, or neutral.",
                    f"Premise: {sample['premise']}\nHypothesis: {new_hypothesis}"
                ))

            if "contradiction" in check.strip().lower():
                new_sample["hypothesis"] = new_hypothesis
                new_sample["label"] = "contradiction"
                return new_sample

        # 如果基本策略失败，尝试更直接的矛盾添加
        for _ in range(self._TRY_LIMIT):
            direct_task = (
                "Add a clause or phrase to the sentence that directly negates or contradicts "
                "a key aspect of the original sentence. Make the contradiction clear and logical. "
                "Use connectors like 'but', 'however', 'although', or 'while' to introduce the contradiction."
            )

            new_hypothesis = self._transform(sample["hypothesis"], direct_task)

            # 检查添加效果
            if (new_hypothesis is None
                    or new_hypothesis == sample["hypothesis"]
                    or len(new_hypothesis)
                    <= len(sample["hypothesis"]) * 1.2):  # 至少增加20%长度
                continue

            # 验证矛盾关系
            check = self._call_llm(
                self._build_prompt(
                    "What is the relationship between the following premise and hypothesis? "
                    "Respond **ONLY** with one word: entailment, contradiction, or neutral.",
                    f"Premise: {sample['premise']}\nHypothesis: {new_hypothesis}"
                ))

            if "contradiction" in check.strip().lower():
                new_sample["hypothesis"] = new_hypothesis
                new_sample["label"] = "contradiction"
                return new_sample

        return None  # 所有尝试都失败

    # ---- 中立标签的 MR --------------------------------------------------------
    def mr_uninformative(self, sample: Dict[str,
                                            Any]) -> Optional[Dict[str, Any]]:
        """
        **标签变更为中立** — 在 *hypothesis* 中添加无关信息，使其与 *premise* 之间的关系变为中立。
        此方法仅适用于原始标签为蕴含(entailment)的样本。

        参数:
            sample: 原始样本字典
        返回:
            添加无关信息后的新样本（标签变为neutral）
        """
        # 检查是否为蕴含关系
        if sample["label"].lower() != "entailment":
            return None  # 仅适用于蕴含关系

        task = (
            "Add a small amount of irrelevant but plausible information to the sentence. "
            "The added information should be unrelated to the logical relationship with the premise, "
            "making the hypothesis neither clearly entailed nor contradicted by the premise. "
            "Keep the addition brief (no more than 5-8 words) and maintain grammatical correctness. "
            "Examples: add weather conditions, time references, or incidental details."
        )
        new_sample = sample.copy()
        # print(sample)
        for _ in range(self._TRY_LIMIT):
            new_hypothesis = self._transform(sample["hypothesis"], task)
            # print(new_hypothesis)
            if new_hypothesis is None or new_hypothesis == sample["hypothesis"]:
                continue
            # 检查关系是否变为中立
            check = self._call_llm(
                self._build_prompt(
                    "Evaluate the logical relationship between premise and hypothesis. "
                    "Consider: Does the premise provide sufficient information to conclude the hypothesis is true? "
                    "Answer with ONLY one word: entailment, contradiction, or neutral. "
                    "Use 'entailment' only if the hypothesis MUST be true given the premise.",
                    f"Premise: {sample['premise']}\nHypothesis: {new_hypothesis}"
                ))

            if "neutral" in check.strip().lower():
                new_sample["hypothesis"] = new_hypothesis
                new_sample["label"] = "neutral"  # 更新标签为中立
                # print(new_sample)
                return new_sample

        return None  # 多次尝试后仍未成功

    # ---- 复合蜕变关系 ------------------------------------------------------------
    def composite_mr_inv(self,
                         sample: Dict[str, Any],
                         unlabeled: bool = False) -> Optional[Dict[str, Any]]:
        """
        **复合蜕变关系** — 随机组合多个蜕变关系，生成新的样本，仅限不改变标签的蜕变关系。

        参数:
            sample: 原始样本字典
        返回:
            复合蜕变后的新样本
        """
        import random

        # 不改变标签的蜕变关系列表
        invariant_mrs = [
            "synonym_replacement",
            "voice_switch",
            "back_translation",
            "pronoun_substitution",
        ]

        if unlabeled:
            invariant_mrs.extend(
                ["hypernym_substitution", "extracting_snippets"])

        num_transforms = random.randint(3, len(invariant_mrs))
        selected_mrs = random.sample(invariant_mrs,
                                     min(num_transforms, len(invariant_mrs)))

        current_sample = sample.copy()

        # 依次应用选中的蜕变关系
        for mr_name in selected_mrs:
            if mr_name in self.METAMORPHIC_RELATIONS:
                mr_func = self.METAMORPHIC_RELATIONS[mr_name]

                # 处理可能返回元组的蜕变关系
                result = mr_func(self, current_sample)
                if isinstance(result, tuple):
                    result = result[0]  # 取主要结果
                if result is not None:
                    current_sample = result
                else:
                    # 如果某个蜕变失败，尝试跳过继续
                    continue

        # print(f"premise: {current_sample['premise']}")
        # print(f"hypothesis: {current_sample['hypothesis']}")
        if current_sample is None:
            return None

        # 验证蕴含关系
        # check = self._call_llm(
        #     self._build_prompt(
        #         "Does the premise entail the hypothesis? "
        #         "Answer strictly with Yes or No.",
        #         f"Premise: {current_sample['premise']}\nHypothesis: {current_sample['hypothesis']}"
        #     ))

        # if "yes" not in check.strip().lower():
        #     return None

        # 验证最终样本是否与原始样本不同且标签保持一致
        if (current_sample != sample
                and current_sample["premise"] != current_sample["hypothesis"]
                and current_sample["hypothesis"] != sample["hypothesis"]
                and current_sample["label"] == sample["label"]):
            return current_sample

        return None

    def composite_mr_flip(self,
                          sample: Dict[str, Any],
                          unlabeled: bool = False) -> Optional[Dict[str, Any]]:
        """
        **复合蜕变关系** — 随机组合多个蜕变关系，生成新的样本，仅限反转标签的蜕变关系。
        可以考虑使用多个不改变标签的蜕变关系进行组合，最后使用一个翻转标签的蜕变关系。

        参数:
            sample: 原始样本字典
        返回:
            复合蜕变后的新样本
        """
        if sample["label"].lower() != "entailment":
            return None
        import random

        # 第一阶段：应用1-2个不改变标签的蜕变关系
        invariant_mrs = [
            "synonym_replacement", "voice_switch", "back_translation",
            "pronoun_substitution"
        ]

        # 第二阶段：选择多个翻转标签的蜕变关系
        flip_mrs = [
            "negation_flip", "antonym_substitution", "subject_object_swap",
            "adding_contradiction"
        ]

        if unlabeled:
            invariant_mrs.extend(
                ["hypernym_substitution", "extracting_snippets"])
            flip_mrs.append("number_substitution")

        current_sample = sample.copy()

        # 第一阶段：随机选择1-2个不改变标签的蜕变关系
        num_inv_transforms = random.randint(1, 3)
        selected_inv_mrs = random.sample(
            invariant_mrs, min(num_inv_transforms, len(invariant_mrs)))

        for mr_name in selected_inv_mrs:
            if mr_name in self.METAMORPHIC_RELATIONS:
                mr_func = self.METAMORPHIC_RELATIONS[mr_name]

                # 处理可能返回元组的蜕变关系
                result = mr_func(self, current_sample)
                if isinstance(result, tuple):
                    result = result[0]
                if result is not None:
                    current_sample = result
                else:
                    # 如果某个蜕变失败，继续下一个
                    continue

        # 第一阶段：随机选择1-2个翻转标签的蜕变关系
        num_flip_transforms = random.randint(1, 2)
        flip_mrs = random.sample(flip_mrs,
                                 min(num_flip_transforms, len(flip_mrs)))

        for mr_name in flip_mrs:
            if mr_name in self.METAMORPHIC_RELATIONS:
                mr_func = self.METAMORPHIC_RELATIONS[mr_name]

            # 处理可能返回元组的蜕变关系
            result = mr_func(self, current_sample)
            if isinstance(result, tuple):
                return None
            if result is not None:
                current_sample = result
            else:
                continue
        final_result = current_sample

        # # 检查关系是否变为矛盾
        # check = self._call_llm(
        #     self._build_prompt(
        #         "What is the relationship between the following premise and hypothesis? "
        #         "Respond **ONLY** with one word: entailment, contradiction, or neutral.",
        #         f"Premise: {sample['premise']}\nHypothesis: {sample['hypothesis']}"
        #     ))

        # if "contradiction" not in check.strip().lower():
        #     return None

        # 验证最终样本标签确实发生了翻转
        if (final_result is not None and final_result != sample
                and final_result["label"] != sample["label"]):
            return final_result

        return None

    def composite_mr_neutral(
            self,
            sample: Dict[str, Any],
            unlabeled: bool = False) -> Optional[Dict[str, Any]]:
        """
        **复合蜕变关系** — 组合多个蜕变关系生成中立标签样本。
        先应用不改变标签的蜕变关系，最后应用生成中立标签的蜕变关系。

        参数:
            sample: 原始样本字典
        返回:
            复合蜕变后的新样本（标签为neutral）
        """
        import random

        # 只对entailment标签的样本进行处理
        if sample["label"].lower() != "entailment":
            return None

        # 第一阶段：应用不改变标签的蜕变关系
        invariant_mrs = [
            "synonym_replacement", "voice_switch", "back_translation",
            "pronoun_substitution"
        ]

        if unlabeled:
            invariant_mrs.extend(
                ["hypernym_substitution", "extracting_snippets"])

        current_sample = sample.copy()

        # 随机选择不改变标签的蜕变关系
        num_inv_transforms = random.randint(2, 4)
        selected_inv_mrs = random.sample(
            invariant_mrs, min(num_inv_transforms, len(invariant_mrs)))

        for mr_name in selected_inv_mrs:
            if mr_name in self.METAMORPHIC_RELATIONS:
                mr_func = self.METAMORPHIC_RELATIONS[mr_name]

                # 处理可能返回元组的蜕变关系
                result = mr_func(self, current_sample)
                if isinstance(result, tuple):
                    result = result[0]  # 取主要结果

                if result is not None:
                    current_sample = result
                else:
                    continue
        # 第二阶段：应用生成中立标签的蜕变关系
        num_netural = random.randint(1, 2)
        # 选择生成中立标签的蜕变关系
        neutral_mrs = ["uninformative"]
        if unlabeled:
            neutral_mrs.append("hypernym_substitution")
        selected_neutral_mrs = random.sample(
            neutral_mrs, min(num_netural, len(neutral_mrs)))
        for mr_name in selected_neutral_mrs:
            if mr_name in self.METAMORPHIC_RELATIONS:
                mr_func = self.METAMORPHIC_RELATIONS[mr_name]

                # 处理可能返回元组的蜕变关系
                result = mr_func(self, current_sample)
                if isinstance(result, tuple):
                    result = result[1]
                if result is not None:
                    current_sample = result
                else:
                    continue

        neutral_result = current_sample
        if neutral_result is None:
            return None

        # 检查关系是否变为中立
        check = self._call_llm(
            self._build_prompt(
                "What is the relationship between the following premise and hypothesis? "
                "Respond **ONLY** with one word: entailment, contradiction, or neutral.",
                f"Premise: {neutral_result['premise']}\nHypothesis: {neutral_result['hypothesis']}"
            ))
        if "neutral" not in check.strip().lower():
            return None

        if (neutral_result != sample and neutral_result["label"] == "neutral"):
            return neutral_result

        return None

    # ---------------------------------------------------------------------------
    # 注册表：公开接口，支持外部调用与扩展
    # ---------------------------------------------------------------------------

    METAMORPHIC_RELATIONS = {
        "synonym_replacement": mr_synonym_replacement,
        "negation_flip": mr_negation_flip,
        "voice_switch": mr_voice_switch,
        "antonym_substitution": mr_antonym_substitution,
        "back_translation": mr_back_translation,
        "uninformative": mr_uninformative,
        "extracting_snippets": mr_extracting_snippets,
        "hypernym_substitution": mr_hypernym_substitution,
        "pronoun_substitution": mr_pronoun_substitution,
        "counting_entailment": mr_counting,
        "subject_object_swap": mr_subject_object_swap,
        "number_substitution": mr_number_substitution,
        "adding_contradiction": mr_adding_contradiction,
        # 添加复合蜕变关系
        "composite_inv": composite_mr_inv,
        "composite_flip": composite_mr_flip,
        "composite_neutral": composite_mr_neutral,
    }

    SAFE_MRS = {
        "voice_switch": mr_voice_switch,
        "synonym_replacement": mr_synonym_replacement,
        "back_translation": mr_back_translation,
        "pronoun_substitution": mr_pronoun_substitution,
        "counting_entailment": mr_counting,
        "negation_flip": mr_negation_flip,
        # 添加安全的复合蜕变关系
        "composite_inv": composite_mr_inv,
    }

    LABEL_MRS = {
        # "negation_flip": mr_negation_flip,
        # "antonym_substitution": mr_antonym_substitution,
        # "subject_object_swap": mr_subject_object_swap,
        # "adding_contradiction": mr_adding_contradiction,
        # "back_translation": mr_back_translation,
        # "hypernym_substitution": mr_hypernym_substitution,
        # "voice_switch": mr_voice_switch,
        # "synonym_replacement": mr_synonym_replacement,
        # "uninformative": mr_uninformative,
        # 添加复合蜕变关系
        "composite_inv": composite_mr_inv,
        "composite_flip": composite_mr_flip,
        "composite_neutral": composite_mr_neutral,
    }

    UNLABEL_MRS = {
        # "synonym_replacement": mr_synonym_replacement,
        # "negation_flip": mr_negation_flip,
        # "voice_switch": mr_voice_switch,
        # "antonym_substitution": mr_antonym_substitution,
        # "back_translation": mr_back_translation,
        # "uninformative": mr_uninformative,
        # "extracting_snippets": mr_extracting_snippets,
        # "hypernym_substitution": mr_hypernym_substitution,
        # "pronoun_substitution": mr_pronoun_substitution,
        # "counting_entailment": mr_counting,
        # "subject_object_swap": mr_subject_object_swap,
        # "number_substitution": mr_number_substitution,
        # "adding_contradiction": mr_adding_contradiction,
        # 添加复合蜕变关系
        "composite_inv": composite_mr_inv,
        "composite_flip": composite_mr_flip,
        "composite_neutral": composite_mr_neutral,
    }

    DUAL_MRS = {
        "hypernym_substitution": mr_hypernym_substitution,
        "counting_entailment": mr_counting,
    }

    # 蜕变关系的类型映射, inv: 反转关系, flip: 翻转关系(E->C), neutral: 中立关系
    MR_TYPE = {
        "synonym_replacement": "inv",
        "negation_flip": "flip",
        "voice_switch": "inv",
        "hypernym_substitution": "inv-neutral",
        "antonym_substitution": "flip",
        "back_translation": "inv",
        "uninformative": "neutral",
        "extracting_snippets": "inv",
        "hypernym_substitution": "inv",
        "pronoun_substitution": "inv",
        "counting_entailment": "inv-flip",
        "subject_object_swap": "flip",
        "number_substitution": "flip",
        "adding_contradiction": "flip",
        # 添加复合蜕变关系类型
        "composite_inv": "inv",
        "composite_flip": "flip",
        "composite_neutral": "neutral",
    }


# ---------------------------------------------------------------------------
# 快速测试（可选，用于开发调试）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 设置随机种子
    import random
    seed = random.randint(0, 100)
    random.seed(seed)

    # Initialize the LLM
    from LMstudio import OpenAIChat, HOST, PORT, MODEL
    llm = OpenAIChat("phi-4", HOST, PORT)

    # Instantiate the NLI tool
    nli_tool = NLIMRTool(llm)

    # Load SNLI data using NLIDataset
    from NLIdataset import NLIDataset
    dataset_handler = NLIDataset(dataset_name="snli",
                                 labeled_ratio=0.01,
                                 unlabeled_ratio=0.01,
                                 seed=seed)

    # Get a sample to test transformations
    samples = list(dataset_handler.get_labeled_dataset())

    # 转换标签格式
    for sample in samples:
        sample['label'] = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }[sample['label']]

    # 获取所有可用的增强方法

    success_count = 0
    attempt_count = 0
    max_attempts = 100  # 最大尝试次数，避免无限循环

    print(f"开始随机选取样本进行增强，目标成功次数：5")
    print("=" * 80)

    while success_count < 5 and attempt_count < max_attempts:
        attempt_count += 1

        # 随机选择一个样本
        sample = random.choice(samples)
        # # 模拟无标签样本
        # sample['hypothesis'] = sample['premise']
        # sample['label'] = "entailment"
        # 随机选择一个增强方法

        # print("原始样本:")
        # print(f"  Premise:    {sample['premise']}")
        # print(f"  Hypothesis: {sample['hypothesis']}")
        # print(f"  Label:      {sample['label']}")

        try:
            # 应用增强方法
            result = nli_tool.mr_adding_contradiction(sample)

            if result:
                success_count += 1
                print(f"\n✅ 增强成功 ({success_count}/5):")
                print(f"  Premise:    {result['premise']}")
                print(f"  Hypothesis: {result['hypothesis']}")
                print(f"  Label:      {result['label']}")
                print("-" * 80)
            else:
                print("❌ 增强失败")

        except Exception as e:
            print(f"❌ 增强过程中出现错误: {e}")

    print(f"\n总结: 成功增强 {success_count} 个样本，总尝试次数 {attempt_count}")
    if success_count < 5:
        print("未能达到目标的5次成功增强")
