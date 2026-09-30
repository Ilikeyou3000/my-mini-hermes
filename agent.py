"""核心 Agent 循环与学习闭环(第 3、13 章)。

循环做的事只有一件:
    调 LLM -> 解析 tool_calls -> 执行工具 -> 结果回灌 -> 重复
直到模型输出纯文本,或者迭代预算耗尽。

注意循环体内没有任何 if "gemma" / if tool_calls 格式判断,
所有格式差异都被 self._strategy 吸收。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from prompt_builder import PromptBuilder
from prompt_caching import apply_prompt_caching
from tool_calling import ToolCall, strategy_for_model

logger = logging.getLogger(__name__)
MAX_TOOL_RESULT_CHARS = 50_000


class Agent:
    def __init__(
        self,
        client: Any,
        model: str,
        system_prompt: str,
        tools: list[dict] | None = None,
        tool_handlers: dict[str, Callable] | None = None,
        max_iterations: int = 15,
        max_tokens: int = 1024,
        verbose: bool = False,
        enable_prompt_caching: bool = False,
        cache_ttl: str | None = "5m",
    ):
        self.client = client
        self.model = model
        self.tools = tools or []
        self.tool_handlers = tool_handlers or {}
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.verbose = verbose
        # 默认关闭:cache_control 是 Anthropic 的字段,
        # 纯 OpenAI 接口收到未知字段会直接报错(第 6 章)
        self.enable_prompt_caching = enable_prompt_caching
        self.cache_ttl = cache_ttl

        self._strategy = strategy_for_model(model)
        # 系统提示词一次性冻结:整个 session 不再重建(为 prompt caching 铺路)
        self.system_prompt = system_prompt
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._memory_nudge_interval = 5
        self._skill_nudge_interval = 8
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        self._user_turn_count = 0
        self._compressor = None

    def configure_learning(self, memory_nudge: int = 5, skill_nudge: int = 8) -> None:
        """配置复盘间隔；0 或负值禁用对应触发器。"""
        self._memory_nudge_interval = memory_nudge
        self._skill_nudge_interval = skill_nudge

    def set_compressor(self, compressor) -> None:
        """注入上下文压缩器；传入 None 禁用压缩。"""
        self._compressor = compressor

    # ------------------------------------------------------------------ #
    # 对外入口
    # ------------------------------------------------------------------ #
    def run(self, user_input: str) -> str:
        self._user_turn_count += 1
        self._turns_since_memory += 1

        # 记忆提醒在本轮开始前锁定，不被当轮主动保存取消。
        review_memory = (
            self._memory_nudge_interval > 0
            and self._turns_since_memory >= self._memory_nudge_interval
        )
        if review_memory:
            self._turns_since_memory = 0
        self.messages.append({"role": "user", "content": user_input})

        for i in range(self.max_iterations):
            msg = self._call_llm()
            content, tool_calls = self._strategy.parse_response(msg)

            self.messages.append(self._strategy.build_assistant_msg(content, tool_calls))

            if not tool_calls:
                break

            # 按有工具调用的迭代计数，而不是按单个工具计数。
            self._iters_since_skill += 1
            self._log(f"[iter {i + 1}] {len(tool_calls)} tool call(s)")
            for tc in tool_calls:
                if tc.name == "memory":
                    self._turns_since_memory = 0
                elif tc.name == "skill_manage":
                    self._iters_since_skill = 0
                result = self._execute_tool(tc)
                self.messages.append(self._strategy.build_tool_result_msg(tc, result))
        else:
            content = "[Max iterations reached]"

        # 技能提醒在本轮结束后判断，主动管理技能会重置计数。
        review_skills = (
            self._skill_nudge_interval > 0
            and self._iters_since_skill >= self._skill_nudge_interval
        )
        if review_skills:
            self._iters_since_skill = 0
        if review_memory or review_skills:
            self._spawn_background_review(review_memory, review_skills)
        return content

    def set_model(self, model: str) -> None:
        """切换模型时必须同步切换策略(CLI 的 /model 命令会用到)。"""
        self.model = model
        self._strategy = strategy_for_model(model)

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _call_llm(self):
        if self._compressor is not None:
            estimated = self._compressor._estimate_tokens(self.messages)
            threshold = self._compressor.max_context_tokens * self._compressor.THRESHOLD
            if estimated >= threshold:
                from compression import flush_memories

                flush_memories(self, self.messages)
                self.messages = self._compressor.maybe_compress(self.messages)

        # 缓存标记只加在"发出去的那份副本"上,self.messages 始终保持干净
        messages = (
            apply_prompt_caching(self.messages, cache_ttl=self.cache_ttl)
            if self.enable_prompt_caching
            else self.messages
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        kwargs = self._strategy.prepare_kwargs(kwargs, self.tools)
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    def _execute_tool(self, tc: ToolCall) -> str:
        handler = self.tool_handlers.get(tc.name)
        if not handler:
            return f"Error: unknown tool '{tc.name}'"
        self._log(f"  -> {tc.name}({tc.arguments})")
        try:
            result = handler(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"
        except Exception as e:  # 工具异常绝不能炸掉循环,必须变成模型可读的文本
            return f"Error executing {tc.name}: {e}"
        text = str(result)
        if len(text) > MAX_TOOL_RESULT_CHARS:
            text = text[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"
        self._log(f"  <- {text[:120]}")
        return text

    _MEMORY_REVIEW = (
        "Review the conversation above. Has the user revealed preferences, "
        "habits, or personal details worth remembering? Has the user expressed "
        "expectations about how you should behave? If so, save using the "
        "memory tool. If nothing is worth saving, say 'Nothing to save.' and stop."
    )
    _SKILL_REVIEW = (
        "Review the conversation above. Was a non-trivial approach used that "
        "required trial and error or course corrections? If a relevant skill "
        "exists, update it. Otherwise, create a new one if reusable. If nothing "
        "is worth saving, say 'Nothing to save.' and stop."
    )
    _COMBINED_REVIEW = (
        "Review the conversation above.\n\n"
        "**Memory**: User preferences, habits, corrections? Save with memory tool.\n\n"
        "**Skills**: Non-trivial approach worth reusing? Create or update a skill.\n\n"
        "Only act if genuinely worth saving. Say 'Nothing to save.' if not."
    )

    def _spawn_background_review(self, review_memory: bool, review_skills: bool):
        """按参考实现浅拷贝消息，在后台线程内部创建复盘 Agent。"""
        if review_memory and review_skills:
            prompt = self._COMBINED_REVIEW
        elif review_memory:
            prompt = self._MEMORY_REVIEW
        else:
            prompt = self._SKILL_REVIEW

        messages_snapshot = [m.copy() for m in self.messages]

        def _run():
            try:
                review_agent = Agent(
                    client=self.client,
                    model=self.model,
                    system_prompt=self.system_prompt,
                    tools=self.tools,
                    max_iterations=6,
                    max_tokens=self.max_tokens,
                )
                review_agent.tool_handlers = self.tool_handlers
                review_agent._memory_nudge_interval = 0
                review_agent._skill_nudge_interval = 0
                review_agent.messages = messages_snapshot
                review_agent.run(prompt)
            except Exception as e:
                logger.debug("Background review failed: %s", e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        logger.info("Background review spawned (memory=%s, skills=%s)",
                    review_memory, review_skills)

    def _log(self, line: str) -> None:
        if self.verbose:
            print(f"\033[90m{line}\033[0m")


def build_default_agent(verbose: bool = True) -> Agent:
    """从环境变量拉配置,连真实 LLM。需要 OPENAI_API_KEY / OPENAI_BASE_URL。"""
    import os

    from openai import OpenAI  # 延迟导入,离线 demo 不依赖它

    from memory import PersistentMemory
    from tool_registry import registry

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    # 持久记忆(第 8 章):启动时读一次,之后是冻结快照的一部分。
    # session 中途写进 MEMORY.md 的新记忆只落盘,不回灌本轮提示词。
    memory = PersistentMemory(os.environ.get("MINI_HERMES_DATA_DIR", "data"))

    # 系统提示词在 session 启动时组装一次并冻结(第 5 章)。
    # 技能索引要等第 10 章才有真实来源,这里先留空。
    builder = PromptBuilder()
    system_prompt = builder.get_system_prompt(
        memory_block=memory.load(),
        user_context=os.environ.get("MINI_HERMES_CONTEXT", ""),
    )

    agent = Agent(
        client=client,
        model=os.environ.get("MINI_HERMES_MODEL", "gpt-4o-mini"),
        system_prompt=system_prompt,
        tools=registry.get_schemas(),
        tool_handlers=registry.get_handlers(),
        verbose=verbose,
        # 走 Anthropic / 兼容 cache_control 的网关时设 MINI_HERMES_PROMPT_CACHE=1
        enable_prompt_caching=os.environ.get("MINI_HERMES_PROMPT_CACHE") == "1",
    )
    # 压缩事件(第 14 章)需要 invalidate 后重建,所以把 builder 挂在 agent 上
    agent.prompt_builder = builder
    # 记忆写入工具(第 10 章的 memory_tool)和后台复盘线程(第 13 章)都要用它
    agent.memory = memory
    return agent


if __name__ == "__main__":
    agent = build_default_agent()
    print("mini-hermes ready (Ctrl-C to quit)\n")
    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ("exit", "quit"):
            break
        if line:
            print(f"\nhermes > {agent.run(line)}\n")