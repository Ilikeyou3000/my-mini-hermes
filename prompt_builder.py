"""系统提示词组装(第 5 章)。

一个 session 里系统提示词只组装一次,然后冻结。
不是"每轮重建",因为每轮重建 = prompt cache 全部失效 = 全价重发(第 6 章)。

组装顺序固定,这个顺序本身就是缓存契约:
    identity -> memory -> skills -> project context -> guidance x3
稳定的放前面,易变的放后面;空章节直接跳过,不留空壳标题。
"""

from __future__ import annotations

import re

# 归一化用:行尾空白、3+ 连续换行、全角空格等都会污染 KV cache
_RE_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_RE_MANY_NEWLINES = re.compile(r"\n{3,}")


class PromptBuilder:
    """把身份 + 记忆 + 技能 + 项目上下文 + 行为守则拼成一段系统提示词。

    典型用法(session 启动时一次):
        builder = PromptBuilder()
        system_prompt = builder.get_system_prompt(
            memory_block=persistent.load(),
            skills_index=loader.build_skills_index(),
        )

    压缩事件之后才允许重建:
        builder.invalidate()
        system_prompt = builder.get_system_prompt(...)
    """

    IDENTITY = (
        "You are a helpful AI assistant with persistent memory "
        "and self-improving skills. You remember past conversations and learn "
        "from experience. Use your tools to accomplish tasks."
    )

    MEMORY_GUIDANCE = """## Memory Instructions
After completing tasks, actively decide what's worth remembering:
- User preferences and habits
- Project context and architecture decisions
- Solutions to problems that might recur
Use the memory tool to persist important observations."""

    SKILLS_GUIDANCE = """## Skill Instructions
After difficult or iterative tasks, offer to save as a skill.
Confirm with the user before creating or deleting. Use the
skill_manage tool with action="create" for new skills,
action="patch" (old_string/new_string) to fix existing ones.
Skip for simple one-offs."""

    TOOL_USE_GUIDANCE = """## Tool Use
Take action. Don't just describe what you would do - actually do it.
If the user asks you to write code, write the file. If they ask you
to run something, run it. Prefer action over explanation."""

    # 章节标题集中在一处,测试和后续调整都只看这里
    MEMORY_HEADING = "## What I Remember"
    SKILLS_HEADING = "## Available Skills"
    CONTEXT_HEADING = "## Project Context"

    def __init__(self, identity: str | None = None) -> None:
        if identity is not None:
            self.IDENTITY = identity
        self._cached_prompt: str | None = None
        self.build_count = 0  # 便于断言"真的只组装了一次"

    # ------------------------------------------------------------------ #
    # 纯组装(无缓存),想单独测某一段就用它
    # ------------------------------------------------------------------ #
    def build(
        self,
        memory_block: str = "",
        skills_index: str = "",
        user_context: str = "",
    ) -> str:
        sections: list[str] = [self.IDENTITY]

        # 1. 持久记忆:我知道这个用户什么
        self._add_section(sections, self.MEMORY_HEADING, memory_block)
        # 2. 技能索引:我会哪些流程(只放名字+描述,渐进式披露)
        self._add_section(sections, self.SKILLS_HEADING, skills_index)
        # 3. 项目上下文:这个仓库的专属信息
        self._add_section(sections, self.CONTEXT_HEADING, user_context)

        # 4. 行为守则:自改进行为全靠这三段驱动
        sections.append(self.MEMORY_GUIDANCE)
        sections.append(self.SKILLS_GUIDANCE)
        sections.append(self.TOOL_USE_GUIDANCE)

        self.build_count += 1
        return self.normalize("\n\n".join(sections))

    # ------------------------------------------------------------------ #
    # 冻结快照:整个 session 复用同一份字符串
    # ------------------------------------------------------------------ #
    def get_system_prompt(
        self,
        memory_block: str = "",
        skills_index: str = "",
        user_context: str = "",
    ) -> str:
        """第一次调用组装并缓存,之后无论传什么都返回同一份快照。

        入参变了也不重建 —— 这是刻意的:session 中途写进 MEMORY.md 的新记忆
        落盘即可,不该污染已经被 provider 缓存的前缀。
        """
        if self._cached_prompt is None:
            self._cached_prompt = self.build(
                memory_block=memory_block,
                skills_index=skills_index,
                user_context=user_context,
            )
        return self._cached_prompt

    @property
    def is_frozen(self) -> bool:
        return self._cached_prompt is not None

    def invalidate(self) -> None:
        """丢弃快照。只有上下文压缩这类"前缀已经变了"的事件才该调用。"""
        self._cached_prompt = None

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_section(sections: list[str], heading: str, body: str) -> None:
        """空内容不生成空壳标题 —— 否则模型会以为"我有技能但都是空的"。"""
        body = (body or "").strip()
        if body:
            sections.append(f"{heading}\n{body}")

    @staticmethod
    def normalize(text: str) -> str:
        """空白归一化。

        一个行尾空格就能让 KV cache 全部失效,所以拼接完必须过一遍:
          - 统一换行符
          - 去掉每行行尾空白
          - 3+ 连续换行压成 2
          - 去掉首尾空白
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _RE_TRAILING_WS.sub("", text)
        text = _RE_MANY_NEWLINES.sub("\n\n", text)
        return text.strip()


# --------------------------------------------------------------------------- #
# 技能索引的最小渲染器(第 10 章会换成真正的 skills_list)
# --------------------------------------------------------------------------- #
def render_skills_index(skills: list[dict]) -> str:
    """把 [{name, description}, ...] 渲染成索引行。

    只放名字和描述,不放正文 —— 这就是渐进式披露的第一层:
    技能库长到 100 个,系统提示词的开销也几乎不变。
    """
    lines = []
    for s in skills:
        name = (s.get("name") or "").strip()
        if not name:
            continue
        desc = " ".join((s.get("description") or "").split())[:1024]
        lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    return "\n".join(lines)


if __name__ == "__main__":
    builder = PromptBuilder()
    print(
        builder.get_system_prompt(
            memory_block="### User Profile\n- 偏好中文回复\n\n### Observations\n- 在写 mini-hermes",
            skills_index=render_skills_index(
                [
                    {"name": "git-commit-style", "description": "统一 commit message 格式"},
                    {"name": "csv-to-database", "description": "清洗 CSV 并入库"},
                ]
            ),
            user_context="Python 3.11, 无外部依赖, 测试放 tests/",
        )
    )