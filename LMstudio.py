"""
openai_chat.py
~~~~~~~~~~~~~~
封装 OpenAI ChatCompletion 对话的实用类。
"""

from __future__ import annotations

import json
import pathlib
from typing import List, Dict, Any, Optional

from mylogging import logger
from openai import OpenAI


class OpenAIChat:
    """
    与原 OllamaChat 接口保持高度一致，允许
    ``stateless=True`` 实现“每次都是全新对话，不保存历史”。
    """

    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 1234,
        api_key: Optional[str] = "anything",
        history_file: str | pathlib.Path = "chat_history.json",
        options: Optional[Dict[str, Any]] = None,
        stateless: bool = True,
    ) -> None:
        """
        Args:
            model: OpenAI 模型名。
            api_key: API Key（或用环境变量）。
            base_url: 自定义代理地址。
            history_file: 对话历史保存路径（仅在 stateless=False 时有效）。
            options: 透传给 chat.completions.create 的额外参数。
            stateless: 若为 True，则**完全不使用/保存历史**。
        """
        base_url = f"http://{host}:{port}/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.history_file = pathlib.Path(history_file)
        self.options = options or {}
        self.stateless = stateless
        self.history: List[Dict[str, str]] = ([] if stateless else
                                              self._load_history())
        logger.info(f"大模型已初始化: {self.model}")
        # import traceback, sys
        # print(">>> Foo.__init__ 调用栈 <<<")
        # traceback.print_stack(limit=50, file=sys.stdout)

    # ------------------------------------------------------------------ #
    # 公开方法
    # ------------------------------------------------------------------ #
    def chat(
        self,
        content: str,
        role: str = "user",
        save: bool = False,
    ) -> str:
        """
        向 LLM 发送一条消息，返回模型直接生成的文本内容。

        Args:
            content: 输入内容。
            role: 角色，如 "user" 或 "system"。
            save: 是否保存对话（stateless 模式下无效）。

        Returns:
            模型回复的纯文本内容（str）。
        """
        # 1. 构造 messages
        if self.stateless:
            messages = [{"role": role, "content": content}]
        else:
            self.history.append({"role": role, "content": content})
            messages = self.history

        # 2. 调用 OpenAI Chat API
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.options,
            )
            reply_content = resp.choices[0].message.content
        except Exception as e:
            logger.error(f"调用 OpenAI 失败: {e}")
            return "调用失败"

        # 3. 非 stateless 模式保存历史
        if not self.stateless:
            self.history.append({
                "role": "assistant",
                "content": reply_content
            })
            if save:
                self._save_history()

        return reply_content

    def reset_history(self) -> None:
        """清空历史（stateless 时仅删除文件，不影响内存）"""
        self.history.clear()
        if self.history_file.exists():
            self.history_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # 私有工具
    # ------------------------------------------------------------------ #
    def _load_history(self) -> List[Dict[str, str]]:
        if self.history_file.exists():
            try:
                return json.loads(
                    self.history_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save_history(self) -> None:
        try:
            self.history_file.write_text(
                json.dumps(self.history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"无法保存历史文件: {exc}")


# -------------------------------------------------------------------- #
# 示例
# -------------------------------------------------------------------- #
HOST = "127.0.0.1"
PORT = 1234
MODEL = "默认大模型名称"  # 默认大模型，除了import，其他时候不应该加载默认大模型名称，一加载就说明出问题了

if __name__ == "__main__":
    bot = OpenAIChat(
        model=MODEL,
        host=HOST,
        port=PORT,
        stateless=True,  # ← 开启无状态模式
    )
    reply = bot.chat("who are you")
    print(reply)
