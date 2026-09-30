"""上下文压缩：保留头尾、摘要中间，并在压缩前保存重要记忆。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from openai import OpenAI

from tool_calling import ToolCall

if TYPE_CHECKING:
    from agent import Agent


class ContextCompressor:
    THRESHOLD = 0.5
    TAIL_TOKENS = 20000
    HEAD_MESSAGES = 3
    CHARS_PER_TOKEN = 3.5

    def __init__(self, client: OpenAI, model: str,
                 max_context_tokens: int = 32000,
                 max_tokens: int = 400):
        self.client = client
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_tokens = max_tokens

    def maybe_compress(self, messages: list[dict]) -> list[dict]:
        """达到阈值时保留头尾，对中间消息生成摘要。"""
        estimated = self._estimate_tokens(messages)
        if estimated < self.max_context_tokens * self.THRESHOLD:
            return messages

        head = messages[:self.HEAD_MESSAGES]
        tail = self._get_tail(messages, self.TAIL_TOKENS)
        middle_end = len(messages) - len(tail) if tail else len(messages)
        middle = messages[self.HEAD_MESSAGES:middle_end]
        if not middle:
            return messages

        summary = self._summarize_middle(middle)
        return head + [{
            "role": "system",
            "content": f"[Compressed context summary]\n{summary}",
        }] + tail

    def _estimate_tokens(self, messages: list[dict]) -> int:
        total_chars = sum(len(m.get("content", "") or "") for m in messages)
        return int(total_chars / self.CHARS_PER_TOKEN)

    def _get_tail(self, messages: list[dict], max_tokens: int) -> list[dict]:
        """从最新消息向前保留，遇到超出预算的消息即停止。"""
        tail = []
        total = 0
        for msg in reversed(messages[self.HEAD_MESSAGES:]):
            chars = len(msg.get("content", "") or "")
            tokens = int(chars / self.CHARS_PER_TOKEN)
            if total + tokens > max_tokens:
                break
            tail.insert(0, msg)
            total += tokens
        return tail

    def _summarize_middle(self, messages: list[dict]) -> str:
        transcript = "\n".join(
            f"{m['role']}: {(m.get('content') or '[tool call]')[:200]}"
            for m in messages
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": (
                        "Summarize this conversation segment. Include:\n"
                        "- Questions that were resolved\n"
                        "- Decisions that were made\n"
                        "- Pending work items\n"
                        "- Key facts discovered\n"
                        "Be concise. Under 500 words."
                    ),
                }, {
                    "role": "user",
                    "content": transcript,
                }],
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or "(compression failed)"
        except Exception as e:
            return f"(compression error: {e})"


_FLUSH_PROMPT = (
    "[System: The session is being compressed. "
    "Save anything worth remembering -- prioritize user preferences, "
    "corrections, and recurring patterns over task-specific details.]"
)


def flush_memories(agent: Agent, messages: list[dict], min_turns: int = 0) -> None:
    """追加临时用户消息，仅请求一次 memory 调用，随后清理临时内容。"""
    if not agent.tool_handlers.get("memory"):
        return
    if agent._user_turn_count < min_turns:
        return
    if len(messages) < 3:
        return

    sentinel_idx = len(messages)
    messages.append({"role": "user", "content": _FLUSH_PROMPT, "_flush": True})
    try:
        memory_schema = [
            t for t in agent.tools
            if t.get("function", {}).get("name") == "memory"
        ]
        # 直接请求，避免 _call_llm 再次触发压缩。
        resp = agent.client.chat.completions.create(
            model=agent.model,
            messages=[m for m in messages if "_flush" not in m or m == messages[-1]],
            tools=memory_schema or None,
            max_tokens=agent.max_tokens,
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if tc.function.name == "memory":
                    args = json.loads(tc.function.arguments)
                    agent._execute_tool(ToolCall(name="memory", arguments=args))
    except Exception:
        pass
    while len(messages) > sentinel_idx:
        messages.pop()