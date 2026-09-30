"""工具调用策略层。

Agent 循环本身不关心「工具调用长什么样」,全部交给策略对象:
  - StructuredStrategy: 模型通过 API 的 tool_calls 字段返回(OpenAI/Qwen/Claude)
  - TextStrategy:       模型把工具调用写在正文文本里(Gemma/LLaMA/Phi)

策略负责三件事:
  1. prepare_kwargs        -> 工具怎么告诉模型
  2. parse_response        -> 工具调用怎么解析出来
  3. build_tool_result_msg -> 执行结果怎么回灌
"""

from __future__ import annotations

import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """统一的工具调用表示,屏蔽底层格式差异。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")


class ToolCallingStrategy(ABC):
    @abstractmethod
    def prepare_kwargs(self, kwargs: dict, tools: list[dict]) -> dict: ...

    @abstractmethod
    def parse_response(self, msg: Any) -> tuple[str, list[ToolCall]]: ...

    @abstractmethod
    def build_assistant_msg(self, content: str, tool_calls: list[ToolCall]) -> dict: ...

    @abstractmethod
    def build_tool_result_msg(self, tc: ToolCall, result: str) -> dict: ...


# --------------------------------------------------------------------------- #
# 结构化:模型原生支持 function calling
# --------------------------------------------------------------------------- #
class StructuredStrategy(ToolCallingStrategy):
    def prepare_kwargs(self, kwargs: dict, tools: list[dict]) -> dict:
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    def parse_response(self, msg: Any) -> tuple[str, list[ToolCall]]:
        content = getattr(msg, "content", None) or ""
        raw_calls = getattr(msg, "tool_calls", None) or []
        calls: list[ToolCall] = []
        for rc in raw_calls:
            try:
                args = json.loads(rc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            calls.append(ToolCall(name=rc.function.name, arguments=args, id=rc.id))
        return content, calls

    def build_assistant_msg(self, content: str, tool_calls: list[ToolCall]) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in tool_calls
            ]
        return msg

    def build_tool_result_msg(self, tc: ToolCall, result: str) -> dict:
        return {"role": "tool", "tool_call_id": tc.id, "content": result}


# --------------------------------------------------------------------------- #
# 文本:模型把调用写在正文里,需要自己解析
# --------------------------------------------------------------------------- #
class TextStrategy(ToolCallingStrategy):
    TOOL_PROMPT_HEADER = (
        "\n\n## Available Tools\n"
        "You can call tools by emitting a block exactly like this:\n"
        '<tool_call>{"name": "tool_name", "arguments": {"key": "value"}}</tool_call>\n'
        "Emit nothing else in that turn. Tool results come back as "
        "[Tool Result: name] messages.\n\nTools:\n"
    )

    # 格式 1: <tool_call>{...}</tool_call>
    _RE_XML = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    # 格式 2: ```json {"name": ..., "arguments": ...} ```
    _RE_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    # 格式 3: call:tool_name{...}
    _RE_CALL = re.compile(r"call:\s*([A-Za-z_][\w\-]*)\s*(\{.*?\})", re.DOTALL)

    def prepare_kwargs(self, kwargs: dict, tools: list[dict]) -> dict:
        """不传 tools 参数,而是把工具描述注入 system prompt。"""
        if not tools:
            return kwargs
        messages = list(kwargs.get("messages", []))
        if messages and messages[0].get("role") == "system":
            desc = self.TOOL_PROMPT_HEADER + self._render_tools(tools)
            messages[0] = {
                "role": "system",
                "content": (messages[0].get("content") or "") + desc,
            }
            kwargs["messages"] = messages
        return kwargs

    @staticmethod
    def _render_tools(tools: list[dict]) -> str:
        lines = []
        for t in tools:
            fn = t.get("function", t)
            params = json.dumps(fn.get("parameters", {}), ensure_ascii=False)
            lines.append(f"- {fn.get('name')}: {fn.get('description')}\n  params: {params}")
        return "\n".join(lines)

    def parse_response(self, msg: Any) -> tuple[str, list[ToolCall]]:
        content = getattr(msg, "content", None) or ""
        calls: list[ToolCall] = []
        leftover = content

        for pattern in (self._RE_XML, self._RE_FENCE):
            for m in pattern.finditer(content):
                tc = self._parse_json_call(m.group(1))
                if tc:
                    calls.append(tc)
                    leftover = leftover.replace(m.group(0), "")
            if calls:
                return leftover.strip(), calls

        for m in self._RE_CALL.finditer(content):
            args = self._loose_json(m.group(2))
            if args is not None:
                calls.append(ToolCall(name=m.group(1), arguments=args))
                leftover = leftover.replace(m.group(0), "")

        return leftover.strip(), calls

    def _parse_json_call(self, blob: str) -> ToolCall | None:
        data = self._loose_json(blob)
        if not isinstance(data, dict) or "name" not in data:
            return None
        args = data.get("arguments") or data.get("parameters") or {}
        if isinstance(args, str):
            args = self._loose_json(args) or {}
        return ToolCall(name=data["name"], arguments=args)

    @staticmethod
    def _loose_json(blob: str) -> dict | None:
        """容忍本地模型常见的非标准 JSON(裸 key、单引号)。"""
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass
        fixed = re.sub(r"([{,]\s*)([A-Za-z_]\w*)(\s*:)", r'\1"\2"\3', blob)
        fixed = fixed.replace("'", '"')
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    def build_assistant_msg(self, content: str, tool_calls: list[ToolCall]) -> dict:
        # 文本模式下,助手消息原样保留即可(不存在 tool_calls 字段)
        if tool_calls:
            rendered = "\n".join(
                "<tool_call>"
                + json.dumps({"name": tc.name, "arguments": tc.arguments}, ensure_ascii=False)
                + "</tool_call>"
                for tc in tool_calls
            )
            content = (content + "\n" + rendered).strip()
        return {"role": "assistant", "content": content}

    def build_tool_result_msg(self, tc: ToolCall, result: str) -> dict:
        # 没有 role="tool" 可用,只能伪装成 user 消息喂回去
        return {"role": "user", "content": f"[Tool Result: {tc.name}]\n{result}"}


# --------------------------------------------------------------------------- #
# 选择策略
# --------------------------------------------------------------------------- #
_TEXT_MODEL_HINTS = ("gemma", "llama", "phi", "mistral-7b", "tinyllama")


def strategy_for_model(model: str) -> ToolCallingStrategy:
    """按模型名挑策略。切换模型时重新调用即可。"""
    name = (model or "").lower()
    if any(hint in name for hint in _TEXT_MODEL_HINTS):
        return TextStrategy()
    return StructuredStrategy()