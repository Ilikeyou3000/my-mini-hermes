"""Prompt caching(第 6 章)。

第 5 章把系统提示词冻结,省下的只是重新组装的那点 CPU。
真正省钱的是让 provider 复用 KV cache:命中的前缀按约 10% 计费。

缓存是**前缀匹配**的 —— 从第一个 token 开始逐字节比对,
遇到第一处差异,后面全部作废。所以:
  - 稳定的内容必须放在最前面(第 5 章的组装顺序就是为此);
  - 断点要打在"到此为止都不会再变"的位置。

策略 system_and_3(最多 4 个断点,Anthropic 的硬上限):
    断点 1     -> system prompt,整个 session 逐字节不变,命中率最高
    断点 2..4  -> 最后 3 条非 system 消息

为什么是"最后 3 条"而不是"前 3 条"?
本轮的完整历史就是下一轮的前缀。把断点压在尾部,下一轮发请求时
整段历史都能命中;打在开头只能命中很短的一截。
留 3 个而不是 1 个是为了容错:窗口滚动、重试、压缩之后,
靠前的断点仍有机会命中,不至于一次失配就全价重发。
"""

from __future__ import annotations

import copy
from typing import Any

# Anthropic 单次请求最多 4 个 cache_control 断点
MAX_CACHE_BREAKPOINTS = 4
# 尾部滚动窗口大小:1(system) + 3 = 4,刚好用满
DEFAULT_RECENT_COUNT = 3

_EPHEMERAL = "ephemeral"


def apply_prompt_caching(
    messages: list[dict[str, Any]],
    cache_ttl: str | None = "5m",
    recent_count: int = DEFAULT_RECENT_COUNT,
) -> list[dict[str, Any]]:
    """返回一份带 cache_control 标记的消息副本。

    不原地修改入参 —— 缓存标记只对"这一次请求"有意义,
    让它渗进 agent 的对话历史,下次序列化/落盘就是脏数据。

    用法(第 14 章的压缩也走同一条路径):
        resp = client.chat.completions.create(
            model=self.model,
            messages=apply_prompt_caching(self.messages),
        )
    """
    marked = copy.deepcopy(messages)

    # 幂等:调用方可能把上一轮的返回值又传回来,先清干净再打,
    # 否则断点会越积越多,直接撞破 4 个的上限。
    strip_cache_control(marked, in_place=True)

    budget = MAX_CACHE_BREAKPOINTS

    # 断点 1:system prompt
    for msg in marked:
        if msg.get("role") == "system":
            if _mark_message(msg, cache_ttl):
                budget -= 1
            break  # 只标第一条,多标一条就少一个尾部断点

    # 断点 2..4:最后 recent_count 条非 system 消息,由旧到新
    if recent_count > 0 and budget > 0:
        recent = [m for m in marked if m.get("role") != "system"]
        for msg in recent[-recent_count:]:
            if budget <= 0:
                break
            if _mark_message(msg, cache_ttl):
                budget -= 1

    return marked


def count_cache_breakpoints(messages: list[dict[str, Any]]) -> int:
    """数一共打了几个断点。超过 4 个 provider 会直接报错,所以值得断言。"""
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            total += sum(
                1
                for block in content
                if isinstance(block, dict) and "cache_control" in block
            )
    return total


def strip_cache_control(
    messages: list[dict[str, Any]],
    in_place: bool = False,
) -> list[dict[str, Any]]:
    """摘掉所有 cache_control。落盘、日志、复算 token 之前都该过一遍。"""
    target = messages if in_place else copy.deepcopy(messages)
    for msg in target:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)
    return target


# --------------------------------------------------------------------------- #
# 内部
# --------------------------------------------------------------------------- #
def _mark_message(message: dict[str, Any], cache_ttl: str | None) -> bool:
    """给消息最后一个 content block 打标记,返回是否真的打上了。

    字符串内容要先升格成 content-block 数组 —— cache_control 是块级字段,
    挂不到裸字符串上。空内容不能当断点(provider 会拒),直接跳过。
    """
    content = message.get("content")

    if isinstance(content, str):
        if not content.strip():
            return False
        content = [{"type": "text", "text": content}]
        message["content"] = content

    if not isinstance(content, list) or not content:
        return False

    last = content[-1]
    if not isinstance(last, dict):
        return False

    last["cache_control"] = _cache_control(cache_ttl)
    return True


def _cache_control(cache_ttl: str | None) -> dict[str, str]:
    """ttl 省略时 provider 默认 5m;1h 需要额外的 beta header。"""
    control = {"type": _EPHEMERAL}
    if cache_ttl:
        control["ttl"] = cache_ttl
    return control


if __name__ == "__main__":
    import json

    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮回答"},
        {"role": "user", "content": "第二轮问题"},
        {"role": "assistant", "content": "第二轮回答"},
        {"role": "user", "content": "第三轮问题"},
    ]
    cached = apply_prompt_caching(history)
    print(json.dumps(cached, ensure_ascii=False, indent=2))
    print(f"\n断点数: {count_cache_breakpoints(cached)} / {MAX_CACHE_BREAKPOINTS}")
    print(f"原始历史未被污染: {count_cache_breakpoints(history) == 0}")