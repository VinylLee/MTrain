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

Important Instructions:
1. Output ONLY the transformed English sentence
2. Do NOT include quotation marks, explanations, or commentary
3. Ensure grammatical correctness and natural language flow
4. Preserve the semantic intent of the transformation
5. If transformation is not possible, output the original sentence

Input sentence to transform:
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
        **标签不变** — 对指定字段进行语义等价但形式不同的改写增强。

        使用大模型直接生成语义相同但表达方式不同的句子，取代传统的回译方法。
        通过多样化的表达方式（同义词替换、句式变换、语序调整等）生成高质量增强样本。

        参数:
            sample: 包含 'premise', 'hypothesis', 'label' 的样本字典。
            fields: 指定要改写的字段，可选值为 "premise", "hypothesis", 或 "both"

        返回:
            改写增强后的新样本；
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

        # 改写任务指令
        task_instruction = (
            "Rewrite the following sentence to express the same meaning using different words and structure. "
            "Requirements: "
            "- Preserve the exact semantic meaning and factual content "
            "- Use synonyms, alternative phrases, or different sentence structures "
            "- Ensure the rewritten sentence is natural and grammatically correct "
            "- Avoid changing the core facts, entities, or logical relationships "
            "- Make the expression style different but meaning identical "
            "Examples of acceptable changes: "
            "- Synonym substitution: 'large' → 'big', 'automobile' → 'car' "
            "- Structure variation: 'The cat sat on the mat' → 'On the mat sat the cat' "
            "- Phrase alternatives: 'due to' → 'because of', 'in order to' → 'to' "
            "Provide only the rewritten sentence without explanation.")

        for field in fields_to_process:
            for _ in range(self._TRY_LIMIT):
                # 使用大模型进行改写
                rewritten = self._transform(sample[field], task_instruction)
                if rewritten is None or rewritten == sample[field]:
                    continue

                # 验证改写结果的语义等价性
                verification_prompt = self._get_verification_prompt(
                    "same_meaning")
                verification = self._call_llm(
                    self._build_prompt(
                        verification_prompt,
                        f"Sentence A: {sample[field]}\nSentence B: {rewritten}"
                    ))

                if "yes" in verification.strip().lower():
                    new_sample[field] = rewritten
                    break
            else:
                # 如果该字段改写失败，返回None
                return None

        return new_sample

    def mr_voice_switch(self, sample: Dict[str,
                                           Any]) -> Optional[Dict[str, Any]]:
        """
        **标签不变** — 对 *hypothesis* 进行主动语态 ↔ 被动语态 智能转换。
        
        设计原则：
        1. 只对语法上适合语态转换的句子进行处理
        2. 避免对不及物动词、系动词等进行不合理转换
        3. 检测句子是否具有明确的主谓宾结构
        4. 确保转换后的语态在语境中自然合理
        
        示例：
        - 主动→被动: "The chef prepared the meal" → "The meal was prepared by the chef"
        - 被动→主动: "The window was broken" → "Someone broke the window"
        - 避免转换: "The man exists" (不及物动词，无法转换为被动语态)

        参数:
            sample: 原始样本字典

        返回:
            语态转换后的新样本（双向蕴含验证通过）
        """

        # 预检查：分析句子是否适合语态转换
        def can_switch_voice(sentence: str) -> bool:
            """检查句子是否适合进行语态转换"""

            # 识别不适合转换的动词类型
            intransitive_verbs = [
                "exist", "exists", "existed", "existing", "appear", "appears",
                "appeared", "appearing", "seem", "seems", "seemed", "seeming",
                "happen", "happens", "happened", "happening", "occur",
                "occurs", "occurred", "occurring", "arrive", "arrives",
                "arrived", "arriving", "sleep", "sleeps", "slept", "sleeping",
                "die", "dies", "died", "dying", "live", "lives", "lived",
                "living", "belong", "belongs", "belonged", "belonging"
            ]

            # 系动词（连接动词）
            linking_verbs = [
                "is", "are", "was", "were", "am", "be", "been", "being",
                "become", "becomes", "became", "becoming", "remain", "remains",
                "remained", "remaining", "stay", "stays", "stayed", "staying"
            ]

            sentence_lower = sentence.lower()
            # 检查是否包含不及物动词或系动词
            words = sentence_lower.split()
            for verb in intransitive_verbs + linking_verbs:
                if verb in words:
                    return False

            # 检查是否已经是被动语态（简单检测）
            is_passive = any(
                phrase in sentence_lower for phrase in
                ["was", "were", "is", "are", "am", "been", "being"]) and any(
                    word.endswith("ed") or word.endswith("en")
                    for word in words)

            # 检查是否是主动语态且有明显的动作词
            action_indicators = [
                "make", "create", "build", "write", "read", "eat", "drink",
                "cook", "prepare", "clean", "wash", "drive", "paint", "draw",
                "break", "fix", "repair", "open", "close", "push", "pull",
                "carry", "bring", "take", "send", "give", "show", "tell",
                "teach", "learn", "study", "work", "play", "watch", "see"
            ]

            has_action = any(action in sentence_lower
                             for action in action_indicators)

            return is_passive or has_action

        # 首先检查句子是否适合语态转换
        if not can_switch_voice(sample["hypothesis"]):
            return None

        task = (
            "Convert the sentence between active and passive voice while preserving the exact meaning. "
            "INTELLIGENT CONVERSION GUIDELINES: "
            "1. ONLY convert sentences with clear subject-verb-object relationships "
            "2. For active→passive: identify the direct object and make it the new subject "
            "3. For passive→active: identify the logical agent (by-phrase) or infer a reasonable one "
            "4. Use agentless passive when the agent is unclear, generic, or unimportant "
            "5. Maintain appropriate verb tenses and grammatical aspects "
            "6. Ensure the result sounds natural and grammatically correct "
            "NATURAL EXAMPLES: "
            "✓ 'The chef prepared the meal' → 'The meal was prepared by the chef' "
            "✓ 'Someone broke the window' → 'The window was broken' "
            "✓ 'The committee approved the proposal' → 'The proposal was approved' "
            "✓ 'The book was written by the author' → 'The author wrote the book' "
            "AVOID THESE CONVERSIONS: "
            "✗ Intransitive verbs: 'The man exists' → 'Existence is had by the man' "
            "✗ Linking verbs: 'She seems happy' → 'Happiness is seemed by her' "
            "✗ Stative verbs: 'He owns a car' → 'A car is owned by him' (unless contextually appropriate) "
            "✗ Sentences without clear objects: 'They laughed loudly' "
            "If the sentence cannot be naturally converted, return the original unchanged."
        )

        new_sample = sample.copy()

        for _ in range(self._TRY_LIMIT):
            converted_hypothesis = self._transform(sample["hypothesis"], task)

            # 基本检查：确保有变化
            if converted_hypothesis is None or converted_hypothesis == sample[
                    "hypothesis"]:
                continue

            # 检查是否产生了不自然的语态转换模式
            unnatural_patterns = [
                "is had by", "is seemed by", "is existed by", "is appeared by",
                "is happened by", "is occurred by", "is belonged by",
                "is consisted by", "was had by", "was seemed by",
                "was existed by", "was appeared by", "were had by",
                "were seemed by", "were existed by", "were appeared by",
                "am had by", "are had by", "been had by", "being had by"
            ]

            hypothesis_lower = converted_hypothesis.lower()
            if any(pattern in hypothesis_lower
                   for pattern in unnatural_patterns):
                continue  # 跳过不自然的语态转换

            # 语法正确性检查
            grammar_check = self._call_llm(
                self._build_prompt(
                    "Evaluate this sentence for grammatical correctness and natural English expression. "
                    "Consider: Is the voice conversion appropriate? Does it sound natural? "
                    "Answer with 'Yes' if grammatically correct and natural, 'No' if not.",
                    converted_hypothesis))

            if "no" in grammar_check.strip().lower():
                continue  # 跳过语法不自然的结果

            # 语义等价性验证（双向蕴含）
            semantic_check = self._call_llm(
                self._build_prompt(
                    self._get_verification_prompt("mutual_entailment"),
                    f"Sentence A: {sample['hypothesis']}\nSentence B: {converted_hypothesis}"
                ))

            if "yes" not in semantic_check.strip().lower():
                continue

            # 语境合理性验证
            context_check = self._call_llm(
                self._build_prompt(
                    "Does this voice conversion maintain logical meaning and practical sense? "
                    "Consider whether the sentence structure, agent relationships, and semantic roles are appropriate. "
                    "Answer with 'Yes' if the conversion is logical and maintains meaning, 'No' if not.",
                    f"Original: {sample['hypothesis']}\nConverted: {converted_hypothesis}"
                ))

            if "yes" in context_check.strip().lower():
                new_sample["hypothesis"] = converted_hypothesis
                return new_sample
        return None  # 所有尝试都失败

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
                        self._get_verification_prompt("mutual_entailment"),
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
        优化版本：提升成功率，确保彻底的代词替换。
        
        例如：
        - "The boy is dancing in arena" -> "He is dancing in arena"
        - "The woman walked her dog" -> "She walked her dog"
        - "Two men are walking" -> "They are walking"
        
        参数:
            sample: 原始样本字典，包含 premise、hypothesis 和 label
            
        返回:
            进行代词替换后的新样本（标签不变）
        """
        new_sample = sample.copy()

        for attempt in range(self._TRY_LIMIT):
            # 明确的替换规则指导
            task = (
                "Replace ONE specific noun or noun phrase with the correct pronoun following these STRICT rules: "
                "REPLACEMENT RULES: "
                "- Multiple people/things (plural) → 'they/them' "
                "- Single man/boy/male person → 'he/him' "
                "- Single woman/girl/female person → 'she/her' "
                "- Single animal/object/thing (non-human) → 'it' "
                "GRAMMAR RULES: "
                "- Subject position: use he/she/it/they "
                "- Object position: use him/her/it/them "
                "- Possessive: use his/her/its/their "
                "IMPORTANT: "
                "- Choose the CLEAREST noun to replace (avoid ambiguity) "
                "- Make sure the pronoun refers to something obvious "
                "- Keep sentence natural and grammatically correct "
                "Examples: "
                "  'The boy runs fast' → 'He runs fast' "
                "  'I see the woman' → 'I see her' "
                "  'Two cats are sleeping' → 'They are sleeping' "
                "  'The car is red' → 'It is red' ")

            new_sent = self._transform(sample["hypothesis"], task)
            if new_sent is None or new_sent == sample["hypothesis"]:
                continue

            # 检测是否仍包含完整的先行词 - 如果包含，可能替换不彻底
            contains_antecedent = self._call_llm(
                self._build_prompt(
                    "Does the sentence still contain the complete noun that should have been replaced by the pronoun? "
                    "Look for cases where both the original noun AND its pronoun appear. "
                    "Answer with 'Yes' if both appear, 'No' if only the pronoun remains.",
                    f"Original: {sample['hypothesis']}\nNew: {new_sent}"))

            # 如果仍包含先行词，尝试清理
            if "yes" in contains_antecedent.strip().lower():
                cleanup_task = (
                    "Remove the original noun phrase that was replaced by a pronoun. "
                    "Keep only the sentence with the pronoun, remove redundant noun references. "
                    "Make the sentence concise while keeping it grammatically correct."
                )
                cleaned_sent = self._transform(new_sent, cleanup_task)
                if cleaned_sent and cleaned_sent != new_sent:
                    new_sent = cleaned_sent

            # 基础语法检查
            grammar_issues = [
                " he are ",
                " she are ",
                " it are ",  # 主谓不一致
                " him is ",
                " her is ",
                " them is ",  # 宾格用作主语
                " he him ",
                " she her ",
                " they them ",  # 重复代词
            ]

            sent_lower = " " + new_sent.lower() + " "
            if any(issue in sent_lower for issue in grammar_issues):
                continue

            # 简化验证：只检查语义等价性和语法正确性
            semantic_check = self._call_llm(
                self._build_prompt(
                    "Do these sentences have the same meaning? "
                    "The second sentence uses pronouns instead of nouns, but should mean exactly the same thing. "
                    "Answer with 'Yes' if same meaning, 'No' if different.",
                    f"Original: {sample['hypothesis']}\nPronoun version: {new_sent}"
                ))

            if "yes" in semantic_check.strip().lower():
                # 快速语法验证
                grammar_check = self._call_llm(
                    self._build_prompt(
                        "Is this sentence grammatically correct? "
                        "Check pronoun case, number agreement, and clarity. "
                        "Answer with 'Yes' or 'No' only.", new_sent))

                if "yes" in grammar_check.strip().lower():
                    new_sample["hypothesis"] = new_sent
                    break

        # 检查是否成功替换
        if new_sample["hypothesis"] == sample["hypothesis"]:
            return None

        # 最终验证：确保逻辑关系保持不变
        relationship_check = self._call_llm(
            self._build_prompt(
                "What is the logical relationship between premise and hypothesis? "
                "Answer with one word: entailment, contradiction, or neutral.",
                f"Premise: {new_sample['premise']}\nHypothesis: {new_sample['hypothesis']}"
            ))

        # 确保标签保持不变
        predicted_label = relationship_check.strip().lower()
        if sample["label"].lower() in predicted_label:
            return new_sample
        else:
            return None

    def mr_counting(
        self, sample: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        无标签MR
        **计数蕴含（Counting Entailment, CT）** — 从前提中识别具有共同上位词的可数名词，
        进行计数，并使用模板生成蕴含假设。例如："有{数量}个{上位词}存在"。
        优化版本：简化验证流程，提高成功率。
        
        例如：
        - 前提: "A motorbike and a car are parked"
        - 假设: "Two vehicles are parked" (数量正确：摩托车和汽车都是交通工具)
        
        参数:
            sample: 原始样本字典
        返回:
            生成计数蕴含假设后的新样本（标签为entailment）与副产物（前提和假设对调，标签为矛盾）
        """
        premise = sample["premise"]
        new_sample = sample.copy()

        # 简化的计数模板
        counting_templates = [
            "There are {count} {hypernym}", "{count} {hypernym} are present",
            "{count} {hypernym} exist", "You can see {count} {hypernym}"
        ]

        # 单数形式的计数模板
        singular_templates = [
            "There is {count} {hypernym}", "{count} {hypernym} is present",
            "{count} {hypernym} exists"
        ]

        for _ in range(self._TRY_LIMIT):
            # 简化的分析任务
            analysis_task = (
                "Look at this sentence and count similar things that belong to the same category. "
                "For example: if you see 'dog and cat', count 2 animals. If you see 'car and bike', count 2 vehicles. "
                "Give your answer as: COUNT:number CATEGORY:type "
                "Examples: COUNT:2 CATEGORY:animal, COUNT:3 CATEGORY:person "
                "If you can't find countable similar things, just say 'NONE'.")

            analysis_result = self._call_llm(
                self._build_prompt(analysis_task, premise))

            # 解析分析结果
            if "NONE" in analysis_result.upper(
            ) or "COUNT:" not in analysis_result.upper():
                continue

            try:
                import re
                count_match = re.search(r'COUNT:\s*(\d+)',
                                        analysis_result.upper())
                category_match = re.search(r'CATEGORY:\s*(\w+)',
                                           analysis_result.upper())

                if not count_match or not category_match:
                    continue

                count = int(count_match.group(1))
                hypernym = category_match.group(1).lower()

                # 基本过滤：限制合理范围
                if count < 1 or count > 20:  # 放宽限制
                    continue

                # 过滤明显无意义的词
                if hypernym in ["thing", "item", "object", "stuff"]:
                    continue

                # 根据数量选择模板
                import random
                if count == 1:
                    template = random.choice(singular_templates)
                    count_word = "one"
                    final_hypernym = hypernym
                else:
                    template = random.choice(counting_templates)
                    count_word = str(count)
                    # 简单复数形式处理
                    if hypernym.endswith('y'):
                        final_hypernym = hypernym[:-1] + "ies"
                    elif hypernym.endswith(('s', 'sh', 'ch', 'x')):
                        final_hypernym = hypernym + "es"
                    else:
                        final_hypernym = hypernym + "s"

                # 生成假设
                new_hypothesis = template.format(count=count_word,
                                                 hypernym=final_hypernym)

                # 简化验证：只检查蕴含关系
                check = self._call_llm(
                    self._build_prompt(
                        "Does the first sentence support or prove the second sentence? "
                        "Answer with 'Yes' if the first proves the second, 'No' if not.",
                        f"First: {premise}\nSecond: {new_hypothesis}"))

                if "yes" in check.strip().lower():
                    new_sample["hypothesis"] = new_hypothesis
                    new_sample["label"] = "entailment"
                    # 创建副产物
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
        优化版本：避免双重否定，控制否定范围，提高自然性。
        只对 entailment 标签的样本进行处理。

        参数:
            sample: 原始样本字典

        返回:
            否定后的新样本
        """
        if sample["label"] != "entailment":
            return None

        original = sample["hypothesis"]
        new_sample = sample.copy()

        # 预检查：检测句中是否已有否定词，避免双重否定
        def has_existing_negation(sentence: str) -> bool:
            """检查句子中是否已存在否定词"""
            negation_words = [
                "not", "no", "never", "nobody", "nothing", "nowhere",
                "neither", "nor", "don't", "doesn't", "didn't", "isn't",
                "aren't", "wasn't", "weren't", "won't", "wouldn't", "can't",
                "cannot", "couldn't", "shouldn't", "mustn't", "hardly",
                "barely", "scarcely", "without", "unless", "none", "few",
                "little"
            ]
            sentence_lower = sentence.lower()
            return any(neg_word in sentence_lower
                       for neg_word in negation_words)

        # 如果原句已有否定，跳过处理
        if has_existing_negation(original):
            return None

        # 智能否定策略：根据句子结构选择合适的否定方法
        def get_negation_strategy(sentence: str) -> str:
            """根据句子特征选择最佳否定策略"""
            sentence_lower = sentence.lower()

            # 策略1: 动词否定（适用于有明确动词的句子）
            if any(verb in sentence_lower for verb in [
                    "is", "are", "was", "were", "will", "has", "have", "had",
                    "can", "could", "would", "should"
            ]):
                return "verb_negation"

            # 策略2: 主语否定（适用于简单句型）
            elif sentence.strip().startswith(
                ("A ", "An ", "The ", "Some ", "Many ", "Several ")):
                # 检查主语长度，如果过长使用nobody/no one
                words = sentence.split()
                if len(words) >= 6:  # 句子较长，使用nobody/no one
                    return "nobody_negation"
                else:
                    return "subject_negation"

            # 策略3: 通用动词否定
            else:
                return "verb_negation"

        negation_strategy = get_negation_strategy(original)

        for _ in range(self._TRY_LIMIT):
            if negation_strategy == "verb_negation":
                task = (
                    "Add 'not' to the main verb to negate the sentence. "
                    "Rules: "
                    "- For 'is/are/was/were': add 'not' → 'is not', 'are not', etc. "
                    "- For other verbs: use 'do not/does not/did not' + base verb "
                    "- Keep the sentence structure simple and natural "
                    "- Avoid long or complex negations "
                    "Examples: "
                    "  'A man walks' → 'A man does not walk' "
                    "  'The dog is running' → 'The dog is not running' "
                    "  'They were playing' → 'They were not playing' ")

            elif negation_strategy == "subject_negation":
                task = (
                    "Replace the subject with 'No' + noun to negate the sentence. "
                    "Rules: "
                    "- 'A man walks' → 'No man walks' "
                    "- 'The dog is running' → 'No dog is running' "
                    "- Keep the subject part short (maximum 3 words after 'No') "
                    "- If the subject is too long, use 'Nobody' or 'No one' instead "
                    "Examples: "
                    "  'A person is here' → 'No person is here' "
                    "  'The woman smiles' → 'No woman smiles' ")

            elif negation_strategy == "nobody_negation":
                task = (
                    "Replace the subject with 'Nobody' or 'No one' to negate the sentence. "
                    "Rules: "
                    "- Use 'Nobody' for general statements about people "
                    "- Use 'No one' for more formal contexts "
                    "- Adjust the verb form if needed (nobody + singular verb) "
                    "Examples: "
                    "  'A man in the red shirt is walking' → 'Nobody is walking' "
                    "  'The woman with the bag smiles' → 'No one smiles' ")

            negated = self._transform(original, task)
            if negated is None or negated == original:
                continue

            # 后处理：检查和修正常见问题
            negated_lower = negated.lower()

            # 检查双重否定模式
            double_neg_patterns = [("not", "no "), ("not", "never"),
                                   ("not", "nobody"), ("not", "nothing"),
                                   ("no", "not"), ("never", "not"),
                                   ("nobody", "not"), ("nothing", "not")]

            has_double_neg = False
            for neg1, neg2 in double_neg_patterns:
                if neg1 in negated_lower and neg2 in negated_lower:
                    has_double_neg = True
                    break

            if has_double_neg:
                continue  # 跳过有双重否定的结果

            # 检查"No X"结构的长度限制
            if negated_lower.startswith("no "):
                # 提取"No"后面到第一个动词之前的部分
                words_after_no = negated.split()[1:]  # 去掉"No"
                verb_position = 0
                for i, word in enumerate(words_after_no):
                    if word.lower() in [
                            "is", "are", "was", "were", "has", "have", "had",
                            "can", "could", "will", "would", "does", "do",
                            "did"
                    ]:
                        verb_position = i
                        break

                subject_length = verb_position if verb_position > 0 else len(
                    words_after_no)

                # 如果主语部分超过3个词，改用nobody/no one
                if subject_length > 3:
                    replacement_task = (
                        "Replace the long 'No + subject' with 'Nobody' or 'No one' and adjust the verb. "
                        "Keep the sentence simple and natural. "
                        "Use singular verb form with 'Nobody' or 'No one'.")
                    corrected = self._transform(negated, replacement_task)
                    if corrected and corrected != negated:
                        negated = corrected

            # 简化验证：只检查逻辑关系
            relationship_check = self._call_llm(
                self._build_prompt(
                    "Do these two sentences contradict each other? "
                    "Answer with 'Yes' if they contradict, 'No' if they don't.",
                    f"Sentence 1: {sample['premise']}\nSentence 2: {negated}"))

            if "yes" in relationship_check.strip().lower():
                new_sample["hypothesis"] = negated
                new_sample["label"] = "contradiction"
                return new_sample

        return None  # 所有尝试都失败

    def mr_antonym_substitution(
            self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        **标签翻转** — 将 *hypothesis* 中的一个关键形容词或动词替换为其直接反义词，
        以改变其逻辑含义，从而使原始标签从 entailment 到 contradiction。
        优化版本：简化验证流程，提高成功率。
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
            "Replace exactly ONE key adjective, verb, or adverb with its direct antonym to create a logical opposite meaning. "
            "SIMPLE GUIDELINES: "
            "- Choose the most important word to replace with its opposite "
            "- Use natural antonyms: hot↔cold, big↔small, fast↔slow, up↔down, in↔out "
            "- For verbs: open↔close, start↔stop, rise↔fall, enter↔exit "
            "- Keep grammar correct and meaning clear "
            "- Avoid using 'not' - use real antonyms instead "
            "Examples: "
            "  'The man walks fast' → 'The man walks slowly' "
            "  'The door is open' → 'The door is closed' "
            "  'The ball goes up' → 'The ball goes down' ")

        new_sample = sample.copy()

        for _ in range(self._TRY_LIMIT):
            new_hypothesis = self._transform(sample["hypothesis"], task)
            if new_hypothesis is None or new_hypothesis == sample["hypothesis"]:
                continue

            # 简化检查：只检查是否避免了不当的否定构造
            problematic_patterns = [
                "not ", "don't", "doesn't", "didn't", "isn't", "aren't",
                "wasn't", "weren't"
            ]

            hypothesis_lower = new_hypothesis.lower()
            if any(pattern in hypothesis_lower
                   for pattern in problematic_patterns):
                continue  # 跳过包含否定构造的结果

            # 单步验证：直接检查逻辑关系
            logic_check = self._call_llm(
                self._build_prompt(
                    "Compare these two sentences. Do they contradict each other? "
                    "Answer with 'Yes' if they contradict, 'No' if they don't contradict.",
                    f"Sentence 1: {sample['premise']}\nSentence 2: {new_hypothesis}"
                ))

            if "yes" in logic_check.strip().lower():
                new_sample["hypothesis"] = new_hypothesis
                new_sample["label"] = "contradiction"
                return new_sample

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
            "Swap the subject and object positions in the sentence while keeping the same verb and prepositions. "
            "Important distinctions: "
            "- This is NOT voice conversion (active↔passive) "
            "- Physically exchange the subject and object positions "
            "- Example: 'A clock stands on a pillar' → 'A pillar stands on a clock' "
            "- Maintain all prepositions, articles, and modifiers in their relative positions "
            "- Ensure grammatical correctness while creating logical contradiction "
            "- The result should be semantically contradictory to the original"
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
        **标签变更为矛盾** — 在 *hypothesis* 中精准添加矛盾信息，
        使其与 *premise* 之间的关系变为矛盾。此方法仅适用于原始标签为蕴含(entailment)的样本。
        
        设计原则：
        1. 优先生成简洁明确的矛盾，避免冗长或复杂的表达
        2. 支持分句、删除冗余词汇等策略控制句子长度
        3. 确保矛盾的逻辑性和自然性
        
        示例：
        - 前提: "A man is walking in the park"
        - 原假设: "A man is walking" (entailment)
        - 新假设: "A man is walking but very slowly" (contradiction)
        - 新假设: "A man is walking. However, nobody else is present." (contradiction)
        
        参数:
            sample: 原始样本字典

        返回:
            添加矛盾信息后的新样本（标签为contradiction）
        """
        # 只对 entailment 标签的样本进行处理
        if sample["label"].lower() != "entailment":
            return None

        new_sample = sample.copy()

        # 定义精细化的矛盾信息添加策略，注重质量和简洁性
        contradiction_strategies = [
            # 策略1: 存在性否定（简洁直接）
            "Add a brief contradictory phrase about the absence or non-existence of key elements. "
            "Use concise negations like 'but no one else', 'with nothing visible', 'yet nobody around'. "
            "Limit addition to 3-5 words for maximum impact.",

            # 策略2: 状态/程度反转（控制用词）
            "Add a short contrasting state or degree that opposes the premise's implications. "
            "Use precise adverbs like 'incorrectly', 'unsuccessfully', 'barely', 'accidentally'. "
            "Keep additions under 4 words to maintain clarity.",

            # 策略3: 时间矛盾（精准时间点）
            "Add specific temporal information that creates timing contradiction with the premise. "
            "Use clear time markers: 'too early', 'at the wrong time', 'before it started', 'after it ended'. "
            "Focus on temporal impossibility rather than vague timing.",

            # 策略4: 空间/位置矛盾（明确对比）
            "Add location information that contradicts spatial assumptions in the premise. "
            "Use clear spatial contrasts: 'in the wrong place', 'far from there', 'inside instead', 'elsewhere'. "
            "Create obvious geographical or positional conflicts.",

            # 策略5: 分句式矛盾（双句结构）
            "Transform into two sentences: keep the original statement, then add a contradictory follow-up. "
            "Format: '[Original statement]. However, [specific contradiction in 4-6 words].' "
            "Ensure the second sentence directly challenges the first.",

            # 策略6: 简化后矛盾（先精简再否定）
            "First remove any redundant adjectives or adverbs from the original sentence, "
            "then add a focused contradictory element that creates clear logical conflict. "
            "Prioritize semantic clarity over descriptive detail."
        ]

        # 主要尝试循环：使用策略化方法
        for _ in range(self._TRY_LIMIT):
            # 随机选择一个矛盾策略
            import random
            strategy = random.choice(contradiction_strategies)

            task = (
                f"{strategy} "
                "CRITICAL REQUIREMENTS: "
                "1. Create CLEAR logical contradiction, not just additional information "
                "2. Keep language natural and readable "
                "3. Avoid verbose or overly complex expressions "
                "4. Ensure the contradiction is obvious to readers "
                "5. Maintain grammatical correctness throughout "
                "The goal is producing a logically impossible or inconsistent statement."
            )

            new_hypothesis = self._transform(sample["hypothesis"], task)

            # 基础检查：确保有实际变化
            if (new_hypothesis is None
                    or new_hypothesis == sample["hypothesis"]):
                continue

            # 智能长度控制：动态优化过长句子
            original_length = len(sample["hypothesis"].split())
            new_length = len(new_hypothesis.split())

            # 如果新句子明显过长，尝试简化
            if new_length > original_length + 8 or new_length > 22:
                simplification_task = (
                    "Simplify this sentence while keeping the core contradiction intact. "
                    "Remove unnecessary words, combine clauses efficiently, or split into shorter sentences. "
                    "Maintain the logical contradiction but express it more concisely. "
                    "Aim for clarity and brevity over elaborate description.")

                simplified = self._transform(new_hypothesis,
                                             simplification_task)
                if simplified and len(simplified.split()) < new_length:
                    new_hypothesis = simplified

            # 矛盾关系验证：确保确实产生contradiction
            contradiction_check = self._call_llm(
                self._build_prompt(
                    "Analyze the logical relationship between premise and hypothesis. "
                    "Look for clear contradictions, impossibilities, or mutually exclusive statements. "
                    "Respond with EXACTLY one word: entailment, contradiction, or neutral.",
                    f"Premise: {sample['premise']}\nHypothesis: {new_hypothesis}"
                ))

            if "contradiction" in contradiction_check.strip().lower():
                new_sample["hypothesis"] = new_hypothesis
                new_sample["label"] = "contradiction"
                return new_sample

        # 备用方案：更直接的矛盾注入
        for _ in range(self._TRY_LIMIT):
            direct_contradiction_task = (
                "Add a brief but unmistakable contradictory element using clear negation or opposition. "
                "Options: "
                "1. Use connecting words: 'but not really', 'yet incorrectly', 'however unsuccessfully' "
                "2. Create two-sentence format: '[Original]. But actually, [contradiction].' "
                "3. Add direct negation: 'without actually [key action]', 'despite not [key state]' "
                "Keep the contradiction obvious and the language natural. Maximum 5 additional words."
            )

            backup_hypothesis = self._transform(sample["hypothesis"],
                                                direct_contradiction_task)

            # 验证备用方案
            if (backup_hypothesis is None
                    or backup_hypothesis == sample["hypothesis"]):
                continue

            # 最终矛盾验证
            final_check = self._call_llm(
                self._build_prompt(
                    "Does the hypothesis contradict the premise? Look for logical impossibilities or conflicts. "
                    "Answer with EXACTLY one word: entailment, contradiction, or neutral.",
                    f"Premise: {sample['premise']}\nHypothesis: {backup_hypothesis}"
                ))

            if "contradiction" in final_check.strip().lower():
                new_sample["hypothesis"] = backup_hypothesis
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

    def mr_conditional_clause(
            self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        **添加条件从句生成中立标签** — 在 *hypothesis* 中添加多样化的条件从句，
        主要目标是使逻辑关系变为中立。此方法主要适用于entailment标签的样本。
        
        例如：
        - 原假设: "A man is walking"
        - 新假设: "If it's sunny, then a man is walking" (变为neutral)
        - 或: "A man is walking when he has time" (变为neutral)
        - 或: "Although tired, a man is walking" (变为neutral)
        - 或: "Unless it rains, a man is walking" (变为neutral)
        
        参数:
            sample: 原始样本字典
            
        返回:
            添加条件从句后的新样本（标签主要变为neutral）
        """
        # 主要对entailment标签的样本进行处理，因为这样更容易生成neutral
        if sample["label"].lower() != "entailment":
            return None

        new_sample = sample.copy()

        # 定义多样化的条件从句策略，包含各种连接词
        neutral_conditional_strategies = [
            # IF类条件策略
            {
                "connectors": ["if", "if only"],
                "templates": [
                    "Add an 'if' conditional clause that introduces uncertainty about weather, time, or circumstances. "
                    "Examples: 'If it's sunny, then [original sentence]' or '[original sentence] if the conditions are right'. "
                    "The condition should be plausible but not mentioned or guaranteed in the premise."
                ]
            },

            # WHEN类时间条件策略
            {
                "connectors": ["when", "whenever", "while"],
                "templates": [
                    "Add a 'when/whenever/while' temporal clause that creates time-related uncertainty. "
                    "Examples: 'When the time is right, [original sentence]' or '[original sentence] while conditions permit'. "
                    "Focus on temporal conditions that cannot be confirmed from the premise."
                ]
            },

            # UNLESS类否定条件策略
            {
                "connectors": ["unless", "except when"],
                "templates": [
                    "Add an 'unless' clause that introduces negative conditions or exceptions. "
                    "Examples: 'Unless something prevents it, [original sentence]' or '[original sentence] unless circumstances change'. "
                    "Create dependency on the absence of unspecified obstacles."
                ]
            },

            # ALTHOUGH类让步条件策略
            {
                "connectors": [
                    "although", "though", "even though", "despite",
                    "in spite of"
                ],
                "templates": [
                    "Add an 'although/though/despite' concessive clause that introduces contrasting conditions. "
                    "Examples: 'Although circumstances are uncertain, [original sentence]' or '[original sentence] despite potential obstacles'. "
                    "Focus on acknowledging potential contrary conditions without negating the main statement."
                ]
            },

            # AS LONG AS类持续条件策略
            {
                "connectors":
                ["as long as", "so long as", "provided that", "given that"],
                "templates": [
                    "Add a 'provided that/as long as' clause that sets duration or condition-based requirements. "
                    "Examples: 'As long as conditions allow, [original sentence]' or '[original sentence] provided that everything goes well'. "
                    "Create dependency on continued favorable circumstances."
                ]
            },

            # ASSUMING类假设条件策略
            {
                "connectors": ["assuming", "supposing", "presuming"],
                "templates": [
                    "Add an 'assuming/supposing' clause that introduces hypothetical preconditions. "
                    "Examples: 'Assuming all goes well, [original sentence]' or '[original sentence] supposing conditions are favorable'. "
                    "Focus on hypothetical scenarios that cannot be verified from the premise."
                ]
            },

            # IN CASE类预防条件策略
            {
                "connectors": ["in case", "just in case", "in the event that"],
                "templates": [
                    "Add an 'in case/in the event that' clause that introduces precautionary conditions. "
                    "Examples: 'In case conditions permit, [original sentence]' or '[original sentence] in the event that circumstances allow'. "
                    "Focus on contingency scenarios."
                ]
            },

            # WHETHER类选择条件策略
            {
                "connectors": ["whether", "whether or not"],
                "templates": [
                    "Add a 'whether' clause that introduces alternative possibilities or uncertainty. "
                    "Examples: 'Whether conditions are right or not, [original sentence]' or '[original sentence] whether circumstances permit'. "
                    "Create uncertainty about binary conditions."
                ]
            }
        ]

        for _ in range(self._TRY_LIMIT):
            # 随机选择一个策略类别
            import random
            strategy_group = random.choice(neutral_conditional_strategies)
            strategy_template = random.choice(strategy_group["templates"])
            expected_connectors = strategy_group["connectors"]

            task = (
                f"{strategy_template} "
                "The goal is to make the hypothesis NEUTRAL relative to the premise by introducing "
                "reasonable uncertainty that cannot be resolved from the premise alone. "
                "Avoid conditions that would make the statement contradictory or still entailed. "
                "Keep the addition natural and grammatically correct. "
                "Make sure to use appropriate connecting words for the type of condition being added."
            )

            new_hypothesis = self._transform(sample["hypothesis"], task)

            # 检查是否成功添加条件从句且不同于原句
            if (new_hypothesis is None
                    or new_hypothesis == sample["hypothesis"]
                    or len(new_hypothesis)
                    <= len(sample["hypothesis"]) * 1.05):  # 至少增加5%长度
                continue

            # 扩展的条件标志词检查，包含所有连接词类型
            all_conditional_markers = [
                "if", "unless", "provided", "when", "whenever", "while",
                "then", "assuming", "suppose", "supposing", "presuming",
                "whether", "although", "though", "even though", "despite",
                "in spite of", "as long as", "so long as", "given that",
                "in case", "just in case", "in the event that", "except when",
                "if only"
            ]

            # 检查是否包含任何条件标志词
            hypothesis_lower = new_hypothesis.lower()
            if not any(marker in hypothesis_lower
                       for marker in all_conditional_markers):
                continue

            # 验证新的逻辑关系，专门检查是否为neutral
            check = self._call_llm(
                self._build_prompt(
                    "Determine if the premise provides enough information to confirm or deny the hypothesis. "
                    "The hypothesis contains a conditional or qualifying statement - consider whether the premise "
                    "gives us information about the condition being met or the qualifier being satisfied. "
                    "Answer with ONLY one word: entailment, contradiction, or neutral.",
                    f"Premise: {sample['premise']}\nHypothesis: {new_hypothesis}"
                ))

            check_result = check.strip().lower()

            # 优先接受neutral结果
            if "neutral" in check_result:
                new_sample["hypothesis"] = new_hypothesis
                new_sample["label"] = "neutral"
                return new_sample

            # 如果多次尝试都没有得到neutral，在最后几次尝试中可以接受entailment
            # 但仍然避免contradiction，因为我们的目标是生成neutral

        return None  # 所有尝试都失败

    def mr_named_entity_replacement(
            self,
            sample: Dict[str, Any],
            fields: str = "both") -> Optional[Dict[str, Any]]:
        """
        **命名实体替换** — 随机替换句子中的人名、地点名等命名实体，
        保持句子的语法结构和逻辑关系不变。此变换不改变原始标签。
        
        例如：
        - "John is walking in Paris" -> "Mike is walking in London"
        - "Mary visited the Eiffel Tower" -> "Sarah visited the Statue of Liberty"
        
        参数:
            sample: 原始样本字典
            fields: 指定要处理的字段，可选值为 "premise", "hypothesis", 或 "both"
            
        返回:
            进行命名实体替换后的新样本（标签不变）
        """
        # 确定要处理的字段列表
        fields_to_process = []
        if fields == "both":
            fields_to_process = ["premise", "hypothesis"]
        elif fields in ["premise", "hypothesis"]:
            fields_to_process = [fields]
        else:
            raise ValueError(
                "fields must be 'premise', 'hypothesis', or 'both'")

        new_sample = sample.copy()

        # 定义常见的命名实体替换映射
        person_names = [("John", "Mike"), ("Mary", "Sarah"),
                        ("David", "James"), ("Lisa", "Emma"),
                        ("Michael", "William"), ("Jennifer", "Elizabeth"),
                        ("Robert", "Daniel"), ("Jessica", "Ashley"),
                        ("Christopher", "Matthew"), ("Amanda", "Nicole"),
                        ("Tom", "Ben"),
                        ("Alice", "Grace"), ("Peter", "Andrew"),
                        ("Anna", "Helen"), ("Mark", "Paul"),
                        ("Linda", "Karen"), ("Steve", "Brian"),
                        ("Susan", "Nancy")]

        places = [("Paris", "London"), ("New York", "Tokyo"),
                  ("London", "Berlin"), ("Tokyo", "Sydney"),
                  ("Berlin", "Madrid"), ("Madrid", "Rome"),
                  ("Sydney", "Vancouver"), ("Vancouver", "Chicago"),
                  ("Chicago", "Boston"), ("Boston", "Seattle"),
                  ("Seattle", "Miami"), ("Miami", "Atlanta"),
                  ("Rome", "Amsterdam"), ("Amsterdam", "Vienna"),
                  ("Vienna", "Prague"), ("Prague", "Budapest"),
                  ("Budapest", "Warsaw"), ("Warsaw", "Stockholm")]

        landmarks = [("Eiffel Tower", "Big Ben"),
                     ("Statue of Liberty", "Golden Gate Bridge"),
                     ("Big Ben", "Tower Bridge"),
                     ("Golden Gate Bridge", "Brooklyn Bridge"),
                     ("Empire State Building", "Willis Tower"),
                     ("CN Tower", "Space Needle"),
                     ("London Bridge", "Sydney Harbour Bridge"),
                     ("Times Square", "Piccadilly Circus"),
                     ("Central Park", "Hyde Park"),
                     ("Wall Street", "Fleet Street")]

        for field in fields_to_process:
            original_text = sample[field]
            success = False

            for _ in range(self._TRY_LIMIT):
                # 构造替换指令
                task = (
                    "Replace named entities (person names, place names, landmarks) in the sentence "
                    "with different but similar entities while preserving the sentence structure and meaning. "
                    "Guidelines: "
                    "1) Replace person names with other common names "
                    "2) Replace place names with other well-known places "
                    "3) Replace landmarks with other famous landmarks "
                    "4) Keep the same grammatical structure and relationships "
                    "5) Ensure the replacement maintains semantic coherence "
                    "6) Make only 1-2 replacements to avoid changing too much "
                    "Examples: John→Mike, Paris→London, Eiffel Tower→Big Ben")

                new_text = self._transform(original_text, task)

                if new_text is None or new_text == original_text:
                    continue

                # 简单验证：检查是否真的进行了替换
                # 通过比较词汇差异来判断
                original_words = set(original_text.lower().split())
                new_words = set(new_text.lower().split())

                # 应该有一些词汇发生了变化，但大部分保持不变
                common_words = original_words.intersection(new_words)
                if len(common_words
                       ) >= len(original_words) * 0.7:  # 至少70%的词保持不变
                    # 验证语义等价性
                    semantic_check = self._call_llm(
                        self._build_prompt(
                            self._get_verification_prompt("same_meaning"),
                            f"Sentence A: {original_text}\nSentence B: {new_text}"
                        ))

                    if "yes" in semantic_check.strip().lower():
                        new_sample[field] = new_text
                        success = True
                        break

            # 如果当前字段没有成功替换，尝试下一个字段
            if not success and fields == "both":
                continue  # 对于both模式，允许部分字段失败
            elif not success:
                return None  # 对于单字段模式，失败则返回None

        # 检查是否至少有一个字段成功替换
        if (new_sample["premise"] == sample["premise"]
                and new_sample["hypothesis"] == sample["hypothesis"]):
            return None

        # 最终验证：确保整体逻辑关系保持不变
        final_check = self._call_llm(
            self._build_prompt(
                self._get_verification_prompt("label_relationship"),
                f"Premise: {new_sample['premise']}\nHypothesis: {new_sample['hypothesis']}"
            ))

        # 确保标签保持不变
        if sample["label"].lower() in final_check.strip().lower():
            return new_sample
        else:
            return None

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
            "synonym_replacement",
            "voice_switch",
            "back_translation",
            "pronoun_substitution",
        ]

        # 第二阶段：选择多个翻转标签的蜕变关系
        flip_mrs = [
            "negation_flip", "antonym_substitution", "adding_contradiction"
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
            "synonym_replacement",
            "voice_switch",
            "back_translation",
            "pronoun_substitution",
        ]

        if unlabeled:
            invariant_mrs.extend([
                "hypernym_substitution", "extracting_snippets",
                "conditional_clause"
            ])

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
        # "subject_object_swap": mr_subject_object_swap,
        # "number_substitution": mr_number_substitution,
        "adding_contradiction": mr_adding_contradiction,
        "conditional_clause": mr_conditional_clause,
        "named_entity_replacement": mr_named_entity_replacement,
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
        # "named_entity_replacement": mr_named_entity_replacement,
        # 添加安全的复合蜕变关系
        "composite_inv": composite_mr_inv,
    }

    LABEL_MRS = {
        "negation_flip": mr_negation_flip,
        "antonym_substitution": mr_antonym_substitution,
        "adding_contradiction": mr_adding_contradiction,
        "pronoun_substitution": mr_pronoun_substitution,
        "voice_switch": mr_voice_switch,
        "synonym_replacement": mr_synonym_replacement,
        "uninformative": mr_uninformative,
        "conditional_clause": mr_conditional_clause,
        # 添加复合蜕变关系
        "composite_inv": composite_mr_inv,
        "composite_flip": composite_mr_flip,
        "composite_neutral": composite_mr_neutral,
    }

    UNLABEL_MRS = {
        "synonym_replacement": mr_synonym_replacement,
        "negation_flip": mr_negation_flip,
        "voice_switch": mr_voice_switch,
        "antonym_substitution": mr_antonym_substitution,
        "back_translation": mr_back_translation,
        "uninformative": mr_uninformative,
        "extracting_snippets": mr_extracting_snippets,
        "hypernym_substitution": mr_hypernym_substitution,
        "pronoun_substitution": mr_pronoun_substitution,
        # "counting_entailment": mr_counting,
        "adding_contradiction": mr_adding_contradiction,
        "conditional_clause": mr_conditional_clause,
        # "named_entity_replacement": mr_named_entity_replacement,
        # 添加复合蜕变关系
        "composite_inv": composite_mr_inv,
        "composite_flip": composite_mr_flip,
        "composite_neutral": composite_mr_neutral,
    }

    COMPOSITE_MRS = {
        "composite_inv": composite_mr_inv,
        "composite_flip": composite_mr_flip,
        "composite_neutral": composite_mr_neutral,
    }

    COMPOSITE_MRS_MULTIPLE = {
        "composite_inv": composite_mr_inv,
        "composite_flip": composite_mr_flip,
        "composite_neutral": composite_mr_neutral,
        "composite_inv2": composite_mr_inv,
        "composite_flip2": composite_mr_flip,
        "composite_neutral2": composite_mr_neutral,
        "composite_inv3": composite_mr_inv,
        "composite_flip3": composite_mr_flip,
        "composite_neutral3": composite_mr_neutral,
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
        "conditional_clause": "neutral",
        "named_entity_replacement": "inv",
        # 添加复合蜕变关系类型
        "composite_inv": "inv",
        "composite_flip": "flip",
        "composite_neutral": "neutral",
    }

    def _get_verification_prompt(self, verification_type: str) -> str:
        """
        获取标准化的验证提示词
        
        参数:
            verification_type: 验证类型 ("entailment", "mutual_entailment", "same_meaning", "label_relationship")
        
        返回:
            格式化的验证提示词
        """
        prompts = {
            "entailment":
            ("Analyze the logical relationship: Does the premise provide sufficient evidence "
             "to conclude that the hypothesis must be true? "
             "Answer with ONLY: Yes or No"),
            "mutual_entailment":
            ("Do these two sentences mutually entail each other (have identical meaning)? "
             "Consider if each sentence logically guarantees the truth of the other. "
             "Answer with ONLY: Yes or No"),
            "same_meaning":
            ("Do these two sentences convey exactly the same meaning and semantic content? "
             "Consider nuances, implications, and logical equivalence. "
             "Answer with ONLY: Yes or No"),
            "label_relationship":
            ("Determine the logical relationship between premise and hypothesis. "
             "Consider carefully: "
             "- ENTAILMENT: premise guarantees hypothesis is true "
             "- CONTRADICTION: premise guarantees hypothesis is false "
             "- NEUTRAL: premise neither confirms nor denies hypothesis "
             "Answer with ONLY one word: entailment, contradiction, or neutral"
             )
        }
        return prompts.get(verification_type, prompts["label_relationship"])


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

    target_success = 3  # 目标成功增强样本数
    max_attempts = 45  # 最大尝试次数，避免无限循环

    print("=" * 80)

    for mr in nli_tool.UNLABEL_MRS:
        print(f"尝试增强方法: {mr}")
        success_count = 0
        attempt_count = 0
        for i in range(max_attempts):
            # 随机选择一个样本
            sample = random.choice(samples)
            # 模拟无标签样本
            sample = {
                "premise": sample["premise"],
                "hypothesis": sample["premise"],
                "label": "entailment"
            }
            attempt_count += 1
            if success_count >= target_success:
                print(
                    f"已达到目标成功增强样本数: {target_success}, 失败次数: {attempt_count-success_count}"
                )
                break
            try:
                # 调用蜕变关系方法
                # 考虑两重返回值的情况
                result = nli_tool.METAMORPHIC_RELATIONS[mr](nli_tool, sample)

                # 处理函数可能返回元组或单个结果的情况
                if isinstance(result, tuple):
                    new_sample = result[0]  # 取第一个元素作为主要结果
                else:
                    new_sample = result
                if new_sample is None:
                    continue

                # 检查新样本是否有效
                if (new_sample != sample
                        and new_sample["premise"] != new_sample["hypothesis"]
                        and new_sample["hypothesis"] != sample["hypothesis"]):
                    success_count += 1
                    print(f"原始样本: ")
                    print(f"  Premise:    {sample['premise']}")
                    print(f"  Hypothesis: {sample['hypothesis']}")
                    print(f"  Label:      {sample['label']}")
                    print(f"增强样本: ")
                    print(f"  Premise:    {new_sample['premise']}")
                    print(f"  Hypothesis: {new_sample['hypothesis']}")
                    print(f"  Label:      {new_sample['label']}")

            except Exception as e:
                print(f"增强失败: {e}")
        if success_count == 0:
            print(f"没有成功增强样本，尝试次数: {attempt_count}")
