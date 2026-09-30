"""prompt caching 的离线测试(第 6 章)。

跑法:
    pytest tests/
    python3 tests/test_prompt_caching.py

验证点:
  1. system_and_3:system + 最后 3 条非 system 消息各一个断点
  2. 断点总数永不超过 4(Anthropic 硬上限)
  3. 不原地修改入参(缓存标记不该渗进对话历史)
  4. 字符串内容升格成 content-block 数组
  5. 断点打在最后一个 block 上
  6. 空内容不浪费断点
  7. 幂等:重复调用不累积断点
  8. ttl 可配置 / 可省略
  9. strip_cache_control 能清干净
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prompt_caching import (  # noqa: E402
    MAX_CACHE_BREAKPOINTS,
    apply_prompt_caching,
    count_cache_breakpoints,
    strip_cache_control,
)


def history(turns: int = 3) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": "SYSTEM"}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def marked_indices(messages: list[dict]) -> list[int]:
    out = []
    for i, m in enumerate(messages):
        c = m.get("content")
        if isinstance(c, list) and any(
            isinstance(b, dict) and "cache_control" in b for b in c
        ):
            out.append(i)
    return out


# --------------------------------------------------------------------------- #
def test_marks_system_and_last_three():
    """断点位置就是策略本身:system + 尾部滚动窗口。"""
    msgs = history(3)  # 1 system + 6 条对话
    out = apply_prompt_caching(msgs)

    assert marked_indices(out) == [0, 4, 5, 6], "应标 system 和最后 3 条"
    assert count_cache_breakpoints(out) == 4


def test_never_exceeds_provider_limit():
    """超过 4 个断点 provider 直接报错,这是硬约束。"""
    for turns in range(0, 10):
        out = apply_prompt_caching(history(turns))
        assert count_cache_breakpoints(out) <= MAX_CACHE_BREAKPOINTS


def test_recent_count_is_capped_by_budget():
    """就算调用方要标 10 条,总数仍不能破 4。"""
    out = apply_prompt_caching(history(6), recent_count=10)
    assert count_cache_breakpoints(out) == MAX_CACHE_BREAKPOINTS


def test_does_not_mutate_input():
    """标记只对本次请求有效,渗进历史就是脏数据。"""
    msgs = history(3)
    snapshot = [dict(m) for m in msgs]

    apply_prompt_caching(msgs)

    assert count_cache_breakpoints(msgs) == 0
    assert msgs == snapshot, "入参被原地修改了"
    assert all(isinstance(m["content"], str) for m in msgs), "入参内容被升格了"


def test_string_content_is_promoted_to_blocks():
    """cache_control 是块级字段,挂不到裸字符串上。"""
    out = apply_prompt_caching([{"role": "system", "content": "SYSTEM"}])

    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "SYSTEM"
    assert content[0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_marks_last_block_of_multi_block_message():
    """多块消息要标在最后一块,前面的块才算进可缓存前缀。"""
    msgs = [
        {"role": "system", "content": "SYSTEM"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "第一块"},
                {"type": "text", "text": "第二块"},
            ],
        },
    ]
    out = apply_prompt_caching(msgs)

    blocks = out[1]["content"]
    assert "cache_control" not in blocks[0]
    assert "cache_control" in blocks[1]


def test_empty_content_does_not_waste_breakpoint():
    """空内容不能当断点,也不该白占额度。"""
    msgs = [
        {"role": "system", "content": "   "},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "真正的内容"},
    ]
    out = apply_prompt_caching(msgs)

    assert marked_indices(out) == [2]
    assert out[0]["content"] == "   ", "空内容不该被升格"


def test_is_idempotent():
    """把返回值再传回来,断点不能累积。"""
    once = apply_prompt_caching(history(3))
    twice = apply_prompt_caching(once)

    assert count_cache_breakpoints(twice) == MAX_CACHE_BREAKPOINTS
    assert twice == once, "重复应用应收敛到同一结果"


def test_ttl_is_configurable():
    out = apply_prompt_caching(history(1), cache_ttl="1h")
    assert out[0]["content"][0]["cache_control"]["ttl"] == "1h"

    out = apply_prompt_caching(history(1), cache_ttl=None)
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_no_system_message_still_marks_recent():
    """没有 system 时,4 个额度全给尾部。"""
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(5)]
    out = apply_prompt_caching(msgs, recent_count=4)

    assert marked_indices(out) == [1, 2, 3, 4]


def test_strip_cache_control_cleans_everything():
    """落盘前必须清干净。"""
    cached = apply_prompt_caching(history(3))
    cleaned = strip_cache_control(cached)

    assert count_cache_breakpoints(cleaned) == 0
    assert count_cache_breakpoints(cached) == 4, "非 in_place 不该改原对象"
    assert cleaned[0]["content"][0]["text"] == "SYSTEM", "只摘标记,不动正文"


def test_agent_sends_marked_copy_but_keeps_history_clean():
    """接入 Agent 后:发出去的带标记,自己的历史保持干净。"""
    from agent import Agent
    from fake_llm import FakeClient, structured_msg

    client = FakeClient([structured_msg(content="ok"), structured_msg(content="ok2")])
    agent = Agent(
        client=client,
        model="gpt-4o-mini",
        system_prompt="SYSTEM",
        enable_prompt_caching=True,
    )

    agent.run("第一问")
    agent.run("第二问")

    for call in client.calls:
        sent = call["messages"]
        assert 1 <= count_cache_breakpoints(sent) <= MAX_CACHE_BREAKPOINTS
        assert isinstance(sent[0]["content"], list), "system 应已升格并打上标记"

    assert count_cache_breakpoints(agent.messages) == 0, "历史被缓存标记污染了"
    assert agent.messages[0] == {"role": "system", "content": "SYSTEM"}


def test_agent_caching_is_opt_in():
    """默认关闭:纯 OpenAI 接口收到 cache_control 会报错。"""
    from agent import Agent
    from fake_llm import FakeClient, structured_msg

    client = FakeClient([structured_msg(content="ok")])
    agent = Agent(client=client, model="gpt-4o-mini", system_prompt="SYSTEM")
    agent.run("hi")

    assert count_cache_breakpoints(client.calls[0]["messages"]) == 0
    assert client.calls[0]["messages"][0]["content"] == "SYSTEM"


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\n全部通过")