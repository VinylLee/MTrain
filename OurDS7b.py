"""
llm_chat.py
~~~~~~~~~~~
封装 Ollama LLM 对话的实用类。
"""

from __future__ import annotations

import json
import pathlib
from typing import List, Dict, Any, Optional
from mylogging import logger
import ollama


class OllamaChat:
    """
    OllamaChat 用于与 Ollama LLM 进行多轮对话并自动保存历史记录。

    典型用法::

        from llm_chat import OllamaChat

        chat_bot = OllamaChat(
            host="221.65.32.4",
            port=11231,
            model="deepseek-r1:7b",
            history_file="nli_chat.json",
            options={"temperature": 0}
        )

        answer = chat_bot.chat("请将下面句子中的实体替换为其同义词：……")
        print(answer["message"]["content"])
    """

    def __init__(
        self,
        host: str,
        port: int,
        model: str,
        history_file: str | pathlib.Path = "chat_history.json",
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        初始化 LLM 客户端。

        Args:
            host: Ollama 服务 IP 或域名。
            port: Ollama 服务监听端口。
            model: 要加载的 Ollama 模型名称。
            history_file: 本地对话历史保存路径，默认为 *chat_history.json*。
            options: 传递给 `client.chat` 的额外选项，如 ``{"temperature": 0}``。
        """
        self.client = ollama.Client(host=f"http://{host}:{port}")
        self.model = model
        self.history_file = pathlib.Path(history_file)
        self.options = options or {}
        self.history: List[Dict[str, str]] = self._load_history()

    # --------------------------------------------------------------------- #
    # 公开方法
    # --------------------------------------------------------------------- #
    def chat(
        self,
        content: str,
        role: str = "user",
        save: bool = False,
    ) -> Dict[str, Any]:
        """
        向 LLM 发送一条消息，并返回模型回复。

        Args:
            content: 用户（或系统）输入的文本。
            role: 消息角色，默认为 ``"user"``，可选 ``"system"``。
            save: 是否写入本地历史记录文件，默认 ``False``。

        Returns:
            原始的 Ollama 返回字典；其中 ``res["message"]["content"]`` 即为回复文本。
        """

        self.history.append({"role": role, "content": content})
        try:
            res = self.client.chat(
                model=self.model,
                messages=self.history,
                options=self.options,
            )
            # logger.info(f"Ollama 返回结果: {res}")
        except Exception as e:
            logger.error(f"调用 Ollama 服务失败: {e}")
            return {"message": {"content": "调用失败"}}

        self.history.append(res["message"])
        if save:
            self._save_history()
        return res

    def reset_history(self) -> None:
        """
        清空当前会话历史并删除历史文件。
        """
        self.history.clear()
        if self.history_file.exists():
            self.history_file.unlink(missing_ok=True)

    # --------------------------------------------------------------------- #
    # 私有工具
    # --------------------------------------------------------------------- #
    def _load_history(self) -> List[Dict[str, str]]:
        """
        尝试从 `history_file` 加载历史记录；如文件不存在则返回空列表。
        """
        if self.history_file.exists():
            try:
                return json.loads(
                    self.history_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass  # 文件损坏或无法读取时自动忽略
        return []

    def _save_history(self) -> None:
        """
        将当前历史写入 `history_file`，采用 UTF‑8 编码保存为 JSON。
        """
        try:
            self.history_file.write_text(
                json.dumps(self.history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            # 在生产代码中可加入日志记录
            print(f"无法保存历史文件: {exc}")


# ------------------------------------------------------------------------- #
# 示例（仅在直接运行该文件时执行）
# ------------------------------------------------------------------------- #
HOST = "localhost"
PORT = 11434
MODEL = "llama3:70b"
if __name__ == "__main__":
    try:
        bot = OllamaChat(
            host=HOST,
            port=PORT,
            model=MODEL,
            options={"temperature": 0.2},
        )
        logger.info("OllamaChat 初始化完成")
        import time
        time.sleep(5)  # 等待 5 秒，确保模型加载完成
        response = bot.chat("你是谁")
        print(response["message"]["content"], flush=True)
    except Exception as e:
        logger.info(f"[ERROR] 程序运行失败: {e}")
