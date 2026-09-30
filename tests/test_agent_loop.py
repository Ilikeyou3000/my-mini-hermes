"""Agent 循环的离线测试(第 3 章)。

跑法:
    pytest tests/           # 推荐
    python3 tests/test_agent_loop.py   # 无 pytest 时也能跑

验证点:
  1. 结构化模式:tool_calls 被解析、执行、以 role=tool 回灌
  2. 文本模式:工具描述注入 system,正则抠出调用,以 role=user 回灌
  3. 未知工具 / 工具抛异常时循环不崩
  4. max_iterations 精确生效
  5. set_model 能热切换策略
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent import Agent  # noqa: E402
from fake_llm import FakeClient, structured_msg, text_msg  # noqa: E402
from tool_calling import StructuredStrategy, TextStrategy  # noqa: E402
from tool_registry import registry  # noqa: E402


def make_agent(client, model="gpt-4o-mini", **kwargs) -> Agent:
    return Agent(
        client=client,
        model=model,
        system_prompt="You are a test agent.",
        tools=registry.get_schemas(),
        tool_handlers=registry.get_handlers(),
        **kwargs,
    )


def roles_of(agent: Agent) -> list[str]:
    return [m["role"] for m in agent.messages]


# --------------------------------------------------------------------------- #


def test_structured_tool_call():
    client = FakeClient([
        structured_msg(calls=[("calc", {"expression": "2+3*4"})]),
        structured_msg(content="结果是 14。"),
    ])
    agent = make_agent(client)

    out = agent.run("2+3*4 等于多少?")

    assert out == "结果是 14。"
    assert "tools" in client.calls[0], "结构化模式应把 tools 作为 API 参数传入"
    assert roles_of(agent) == ["system", "user", "assistant", "tool", "assistant"]
    assert agent.messages[3]["content"] == "14"
    assert agent.messages[3]["tool_call_id"] == agent.messages[2]["tool_calls"][0]["id"]


def test_text_mode_tool_call():
    client = FakeClient([
        text_msg(
            '先算一下。<tool_call>{"name": "calc", '
            '"arguments": {"expression": "10/4"}}</tool_call>'
        ),
        text_msg("答案是 2.5。"),
    ])
    agent = make_agent(client, model="google/gemma-3-27b-it")

    out = agent.run("10 除以 4")

    assert out == "答案是 2.5。"
    assert "tools" not in client.calls[0], "文本模式不应传 tools 参数"
    assert "Available Tools" in client.calls[0]["messages"][0]["content"]
    assert roles_of(agent) == ["system", "user", "assistant", "user", "assistant"]
    assert agent.messages[3]["content"].startswith("[Tool Result: calc]")


def test_text_mode_loose_json():
    """本地模型常吐非标准 JSON(裸 key + 单引号),也要能解析。"""
    client = FakeClient([
        text_msg("call:calc{expression: '7*6'}"),
        text_msg("42"),
    ])
    agent = make_agent(client, model="llama-3.1-8b")

    assert agent.run("7 乘 6") == "42"
    assert "42" in agent.messages[3]["content"]


def test_errors_do_not_break_loop():
    client = FakeClient([
        structured_msg(calls=[("no_such_tool", {})]),
        structured_msg(calls=[("calc", {"expression": "1/0"})]),
        structured_msg(content="都处理完了。"),
    ])
    agent = make_agent(client)

    out = agent.run("试试错误")

    assert out == "都处理完了。"
    assert "unknown tool" in agent.messages[3]["content"]
    assert "Error executing calc" in agent.messages[5]["content"]


def test_bad_arguments_become_text():
    client = FakeClient([
        structured_msg(calls=[("calc", {"wrong_kw": "1+1"})]),
        structured_msg(content="收到。"),
    ])
    agent = make_agent(client)

    assert agent.run("参数写错") == "收到。"
    assert "bad arguments" in agent.messages[3]["content"]


def test_max_iterations():
    client = FakeClient([structured_msg(calls=[("now", {})]) for _ in range(20)])
    agent = make_agent(client, max_iterations=3)

    assert agent.run("一直调工具") == "[Max iterations reached]"
    assert client.llm_call_count == 3


def test_set_model_switches_strategy():
    agent = make_agent(FakeClient([]))
    assert isinstance(agent._strategy, StructuredStrategy)

    agent.set_model("google/gemma-3-27b-it")
    assert isinstance(agent._strategy, TextStrategy)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\n全部通过")