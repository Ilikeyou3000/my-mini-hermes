"""PromptBuilder 的离线测试(第 5 章)。

跑法:
    pytest tests/
    python3 tests/test_prompt_builder.py

验证点:
  1. 章节顺序固定(缓存契约)
  2. 空章节不留空壳标题
  3. 空白归一化:行尾空格 / CRLF / 多余空行都被抹平
  4. 冻结快照:入参变了也不重建,只组装一次
  5. invalidate 后才允许重建(对应压缩事件)
  6. 技能索引只暴露名字+描述(渐进式披露第一层)
  7. 接入 Agent 后,system 消息全程不变
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent import Agent  # noqa: E402
from fake_llm import FakeClient, structured_msg  # noqa: E402
from prompt_builder import PromptBuilder, render_skills_index  # noqa: E402

MEM = "### User Profile\n- 偏好中文回复"
SKILLS = "- git-commit-style: 统一 commit message 格式"
CTX = "Python 3.11, 测试放 tests/"


# --------------------------------------------------------------------------- #


def test_section_order_is_stable():
    """顺序是缓存契约:身份在最前,三段守则在最后。"""
    p = PromptBuilder().build(memory_block=MEM, skills_index=SKILLS, user_context=CTX)

    positions = [
        p.index("You are a helpful AI assistant"),
        p.index(PromptBuilder.MEMORY_HEADING),
        p.index(PromptBuilder.SKILLS_HEADING),
        p.index(PromptBuilder.CONTEXT_HEADING),
        p.index("## Memory Instructions"),
        p.index("## Skill Instructions"),
        p.index("## Tool Use"),
    ]
    assert positions == sorted(positions), "章节顺序被打乱了"
    assert all(x >= 0 for x in positions)


def test_empty_sections_are_skipped():
    """没有记忆/技能时不能留空壳标题,否则模型会误判自己有空技能。"""
    p = PromptBuilder().build()

    assert PromptBuilder.MEMORY_HEADING not in p
    assert PromptBuilder.SKILLS_HEADING not in p
    assert PromptBuilder.CONTEXT_HEADING not in p
    # 守则始终存在
    assert "## Memory Instructions" in p
    assert "## Tool Use" in p
    # 只有空白的内容同样视为空
    assert PromptBuilder.MEMORY_HEADING not in PromptBuilder().build(memory_block="   \n\t ")


def test_whitespace_is_normalized():
    """一个行尾空格就能让 KV cache 失效,拼完必须归一。"""
    p = PromptBuilder().build(
        memory_block="- 行尾有空格   \r\n\r\n\r\n- 还有 CRLF 和三连空行",
        skills_index="  - 前后都有空白  ",
    )

    assert "\r" not in p
    assert "\n\n\n" not in p, "3+ 连续换行应压成 2"
    assert not any(line != line.rstrip() for line in p.split("\n")), "存在行尾空白"
    assert p == p.strip(), "首尾不应有空白"


def test_normalize_makes_equivalent_inputs_identical():
    """脏空白差异不应产生不同的前缀,否则缓存白白失效。

    注意: 归一化只抹平"不可见"差异(行尾空白/CRLF/首尾空白/3+连续换行压到 2),
    不会把空行删掉 —— 段落结构是语义的一部分。
    """
    clean = PromptBuilder().build(memory_block="- a\n- b", skills_index=SKILLS)
    dirty = PromptBuilder().build(
        memory_block="- a   \r\n- b\t\n", skills_index=f"  {SKILLS}   "
    )
    assert clean == dirty

    # 3 连空行和 2 连空行收敛到同一份
    two = PromptBuilder().build(memory_block="- a\n\n- b")
    many = PromptBuilder().build(memory_block="- a\n\n\n\n- b  ")
    assert two == many


def test_frozen_snapshot_ignores_later_inputs():
    """session 中途新写的记忆不该改动已冻结的系统提示词。"""
    b = PromptBuilder()
    first = b.get_system_prompt(memory_block=MEM)

    assert b.is_frozen
    assert b.build_count == 1

    second = b.get_system_prompt(memory_block=MEM + "\n- 新记忆(不该出现)")

    assert second == first, "冻结后不应重建"
    assert "新记忆" not in second
    assert b.build_count == 1, "只该组装一次"


def test_invalidate_allows_rebuild():
    """只有压缩这类事件才 invalidate,之后新内容才进得来。"""
    b = PromptBuilder()
    b.get_system_prompt(memory_block=MEM)

    b.invalidate()
    assert not b.is_frozen

    rebuilt = b.get_system_prompt(memory_block=MEM + "\n- 压缩后的新记忆")

    assert "压缩后的新记忆" in rebuilt
    assert b.build_count == 2


def test_custom_identity_overrides_default():
    b = PromptBuilder(identity="You are a terse code reviewer.")
    p = b.build()

    assert p.startswith("You are a terse code reviewer.")
    assert "persistent memory" not in p.split("## Memory Instructions")[0]


def test_render_skills_index_is_metadata_only():
    """渐进式披露第一层:只出名字和描述,不出正文。"""
    idx = render_skills_index(
        [
            {"name": "git-commit-style", "description": "统一   commit\n格式"},
            {"name": "no-desc"},
            {"name": "  ", "description": "没名字应被丢掉"},
        ]
    )

    assert idx.splitlines() == [
        "- git-commit-style: 统一 commit 格式",  # 内部空白被压成单空格
        "- no-desc",
    ]
    assert "没名字应被丢掉" not in idx


def test_long_description_is_truncated():
    idx = render_skills_index([{"name": "x", "description": "a" * 2000}])
    assert len(idx) == len("- x: ") + 1024


def test_agent_system_message_never_changes():
    """接入 Agent 后,多轮对话里 system 消息必须逐字节不变。"""
    builder = PromptBuilder()
    system_prompt = builder.get_system_prompt(memory_block=MEM, skills_index=SKILLS)
    print(system_prompt)
    client = FakeClient([
        structured_msg(calls=[("calc", {"expression": "1+1"})]),
        structured_msg(content="2"),
        structured_msg(content="好的"),
    ])
    agent = Agent(client=client, model="gpt-4o-mini", system_prompt=system_prompt)
    agent.tool_handlers = {"calc": lambda expression: "2"}

    agent.run("1+1")
    agent.run("再聊一句")

    sent_systems = [c["messages"][0]["content"] for c in client.calls]
    assert len(set(sent_systems)) == 1, "system 消息在多轮间发生了变化,缓存会失效"
    assert sent_systems[0] == system_prompt
    assert agent.messages[0] == {"role": "system", "content": system_prompt}
    assert builder.build_count == 1


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\n全部通过")