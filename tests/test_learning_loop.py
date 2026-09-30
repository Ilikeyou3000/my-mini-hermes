"""学习闭环离线测试：python3 tests/test_learning_loop.py。"""

import copy
import sys
import threading
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent import Agent
from fake_llm import FakeClient, structured_msg, text_msg


def make_agent(script=(), **kwargs):
    return Agent(FakeClient(script), "test-model", "frozen prompt", **kwargs)


def start_review(agent, memory, skills):
    threads = []
    real_thread = threading.Thread

    def capture_thread(**kwargs):
        thread = real_thread(**kwargs)
        threads.append(thread)
        return thread

    with patch("agent.threading.Thread", side_effect=capture_thread):
        assert agent._spawn_background_review(memory, skills) is None
    return threads[0]


def test_memory_interval_across_turns_and_reset():
    agent = make_agent([text_msg("ok")] * 10)
    with patch.object(agent, "_spawn_background_review") as review:
        for i in range(10):
            assert agent.run("hello") == "ok"
            assert review.call_count == (i + 1) // 5
        assert agent._user_turn_count == 10
        assert agent._turns_since_memory == 0
        assert review.call_args.args == (True, False)


def test_skill_counts_tool_iterations_not_calls():
    tool_turn = structured_msg(calls=[("work", {}), ("work", {})])
    agent = make_agent([tool_turn, text_msg("ok")] * 8)
    agent.configure_learning(memory_nudge=0)
    with patch.object(agent, "_spawn_background_review") as review:
        for i in range(8):
            agent.run("work")
            assert review.call_count == (i + 1) // 8
        assert review.call_args.args == (False, True)
        assert agent._iters_since_skill == 0


def test_active_memory_keeps_due_review_but_skill_cancels():
    agent = make_agent([
        text_msg("ok"),
        structured_msg(calls=[("memory", {}), ("skill_manage", {})]),
        text_msg("done"),
    ], tool_handlers={"memory": lambda: "ok", "skill_manage": lambda: "ok"})
    agent.configure_learning(memory_nudge=2, skill_nudge=1)
    with patch.object(agent, "_spawn_background_review") as review:
        agent.run("first")
        review.assert_not_called()
        agent.run("second")
        review.assert_called_once_with(True, False)
    assert agent._turns_since_memory == agent._iters_since_skill == 0


def test_disabled_reviews_and_plain_text_do_not_count_as_tools():
    agent = make_agent([text_msg("ok")] * 10)
    agent.configure_learning(memory_nudge=0, skill_nudge=-1)
    with patch.object(agent, "_spawn_background_review") as review:
        for _ in range(10):
            agent.run("hello")
        review.assert_not_called()
    assert agent._iters_since_skill == 0


def test_combined_review_after_budget_exhaustion():
    agent = make_agent([structured_msg(calls=[("work", {})])], max_iterations=1)
    agent.configure_learning(memory_nudge=1, skill_nudge=1)
    with patch.object(agent, "_spawn_background_review") as review:
        assert agent.run("hello") == "[Max iterations reached]"
        review.assert_called_once_with(True, True)


def test_background_snapshot_config_and_prompts():
    tools = [{"type": "function", "function": {"name": "work"}}]
    for memory, skills, prompt in (
        (True, False, Agent._MEMORY_REVIEW),
        (False, True, Agent._SKILL_REVIEW),
        (True, True, Agent._COMBINED_REVIEW),
    ):
        agent = make_agent(tools=copy.deepcopy(tools), tool_handlers={"work": lambda: "ok"},
                           verbose=True, enable_prompt_caching=True, cache_ttl="1h")
        agent.messages.append({"role": "user", "content": [{"type": "text", "text": "old"}]})
        captured = []

        def capture_run(child, request):
            captured.append((child, request))

        with patch("agent.threading.Thread") as thread_type, patch.object(Agent, "run", capture_run):
            assert agent._spawn_background_review(memory, skills) is None
            thread_type.return_value.start.assert_called_once()
            assert thread_type.call_args.kwargs["daemon"] is True
            assert captured == []
            snapshot_message = agent.messages[-1]
            agent.messages[-1]["role"] = "assistant"
            agent.messages[-1]["content"][0]["text"] = "new"
            agent.messages.append({"role": "user", "content": "later"})
            agent.tools[0]["function"]["name"] = "changed"
            agent.tool_handlers.clear()
            agent.set_model("gemma")
            thread_type.call_args.kwargs["target"]()

        child, request = captured[0]
        assert request == prompt
        assert child.client is agent.client
        assert child.model == "gemma"
        assert len(child.messages) == 2
        assert child.messages[-1] is not snapshot_message
        assert child.messages[-1]["role"] == "user"
        assert child.messages[-1]["content"] is snapshot_message["content"]
        assert child.messages[-1]["content"][0]["text"] == "new"
        assert child.tools is agent.tools
        assert child.tool_handlers is agent.tool_handlers
        assert child.max_iterations == 6
        assert child.max_tokens == agent.max_tokens
        assert child._memory_nudge_interval == child._skill_nudge_interval == 0
        assert child._compressor is None and not child.verbose
        assert not child.enable_prompt_caching and child.cache_ttl == "5m"
        assert agent.system_prompt == child.system_prompt == "frozen prompt"


def test_background_is_nonblocking_and_history_stays_clean():
    entered, release = threading.Event(), threading.Event()

    def handler():
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test timed out")
        return "saved"

    agent = make_agent([
        structured_msg(calls=[("memory", {})]), text_msg("done"),
    ], tool_handlers={"memory": handler})
    before = copy.deepcopy(agent.messages)
    thread = start_review(agent, True, False)
    try:
        assert entered.wait(2)
        assert thread.is_alive() and thread.daemon
        assert agent.messages == before
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert agent.messages == before
    assert agent._user_turn_count == agent._turns_since_memory == 0


def test_background_budget_does_not_recurse():
    agent = make_agent([structured_msg(calls=[("work", {})])] * 10)
    before = copy.deepcopy(agent.messages)
    with patch.object(Agent, "_spawn_background_review", wraps=agent._spawn_background_review) as spawn:
        thread = start_review(agent, True, True)
        thread.join(3)
        assert not thread.is_alive()
        assert spawn.call_count == 1
    assert agent.client.llm_call_count == 6
    assert agent.messages == before


def test_background_failures_are_isolated():
    agent = make_agent()
    with patch.object(Agent, "run", side_effect=RuntimeError("offline")), patch("agent.logger") as log:
        thread = start_review(agent, True, False)
        thread.join(3)
        assert not thread.is_alive()
        log.debug.assert_called_once()
    with patch("agent.threading.Thread") as thread_type:
        thread_type.return_value.start.side_effect = RuntimeError("no thread")
        try:
            agent._spawn_background_review(True, False)
        except RuntimeError as e:
            assert str(e) == "no thread"
        else:
            raise AssertionError("thread start failure must propagate")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\n全部通过")