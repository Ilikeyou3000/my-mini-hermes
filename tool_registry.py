"""工具注册表(第 4 章的最小版)。

一个工具 = 名字 + 描述 + JSON Schema + handler + category。
category 既用来控制暴露给模型的工具数量(工具越少决策越准),
也是安全边界(子 agent 只给 read 类工具)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass
class ToolEntry:
    name: str
    description: str
    parameters: dict
    handler: Callable
    category: str = "general"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable,
        category: str = "general",
    ) -> None:
        self._tools[name] = ToolEntry(name, description, parameters, handler, category)

    def tool(self, description: str, parameters: dict, category: str = "general"):
        """装饰器写法,注册更顺手。"""

        def deco(fn: Callable) -> Callable:
            self.register(fn.__name__, description, parameters, fn, category)
            return fn

        return deco

    def _select(self, categories: Iterable[str] | None) -> list[ToolEntry]:
        entries = list(self._tools.values())
        if categories:
            wanted = set(categories)
            entries = [e for e in entries if e.category in wanted]
        return entries

    def get_schemas(self, categories: Iterable[str] | None = None) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": e.name,
                    "description": e.description,
                    "parameters": e.parameters,
                },
            }
            for e in self._select(categories)
        ]

    def get_handlers(self, categories: Iterable[str] | None = None) -> dict[str, Callable]:
        return {e.name: e.handler for e in self._select(categories)}

    def execute(self, name: str, args: dict) -> str:
        entry = self._tools.get(name)
        if not entry:
            raise ValueError(f"Unknown tool: {name}")
        return str(entry.handler(**args))


registry = ToolRegistry()


# --------------------------------------------------------------------------- #
# 内置最小工具集,够验证循环
# --------------------------------------------------------------------------- #
@registry.tool(
    description="Evaluate a simple arithmetic expression, e.g. '2+3*4'.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Arithmetic expression"}},
        "required": ["expression"],
    },
    category="compute",
)
def calc(expression: str) -> str:
    import ast
    import operator as op

    ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.Mod: op.mod,
        ast.USub: op.neg,
    }

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    return str(ev(ast.parse(expression, mode="eval").body))


@registry.tool(
    description="Return the current local time as an ISO-8601 string.",
    parameters={"type": "object", "properties": {}},
    category="general",
)
def now() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")