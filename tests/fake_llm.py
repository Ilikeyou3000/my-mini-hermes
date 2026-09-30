"""测试替身:模拟 OpenAI SDK 的 chat.completions.create,不需要 API key。

FakeClient 按脚本顺序返回预设 message,并记录每次请求的 kwargs 供断言。
"""

from __future__ import annotations

import json
from types import SimpleNamespace


def structured_msg(content=None, calls=None):
    """构造结构化模式的 message(带 tool_calls 字段)。

    calls: [(tool_name, args_dict), ...]
    """
    tool_calls = None
    if calls:
        tool_calls = [
            SimpleNamespace(
                id=f"call_{i}",
                type="function",
                function=SimpleNamespace(name=name, arguments=json.dumps(args)),
            )
            for i, (name, args) in enumerate(calls)
        ]
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def text_msg(content):
    """构造文本模式的 message(工具调用藏在正文里)。"""
    return SimpleNamespace(content=content, tool_calls=None)


class FakeClient:
    """脚本化的假 LLM 客户端。

    scripted_messages 里的 message 会被依次弹出返回;
    脚本耗尽后返回兜底文本,避免测试卡死。
    """

    def __init__(self, scripted_messages):
        self._queue = list(scripted_messages)
        self.calls: list[dict] = []  # 每次请求的 kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        msg = self._queue.pop(0) if self._queue else text_msg("(out of script)")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    @property
    def llm_call_count(self) -> int:
        return len(self.calls)

    def last_kwargs(self) -> dict:
        return self.calls[-1] if self.calls else {}