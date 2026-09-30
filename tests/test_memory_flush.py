"""压缩前记忆保存离线测试：python3 tests/test_memory_flush.py。"""

import copy
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent import Agent
from compression import flush_memories
from fake_llm import FakeClient, structured_msg, text_msg


def make_agent(script=(), model="test-model", **kwargs):
    saved = []
    tools = [{"type": "function", "function": {"name": name, "parameters": {}}}
             for name in ("memory", "work")]
    agent = Agent(FakeClient(script), model, "frozen", tools=tools,
                  tool_handlers={"memory": lambda **args: saved.append(args)}, **kwargs)
    agent.configure_learning(0, 0)
    agent.messages.extend([{"role": "user", "content": "remember"},
                           {"role": "assistant", "content": "ok"}])
    agent._user_turn_count = 1
    return agent, saved


def test_flush_single_call_memory_only_via_executor():
    agent, saved = make_agent([structured_msg(calls=[
        ("work", {}), ("memory", {"action": "save", "text": "preference"}),
    ])], verbose=True)
    agent.tool_handlers["work"] = lambda: (_ for _ in ()).throw(AssertionError("not allowed"))
    before = copy.deepcopy(agent.messages)
    original = agent.messages
    output = io.StringIO()
    with redirect_stdout(output), patch.object(agent, "_execute_tool", wraps=agent._execute_tool) as execute:
        flush_memories(agent, agent.messages)
    execute.assert_called_once()
    assert execute.call_args.args[0].name == "memory"
    assert "memory" in output.getvalue()
    assert saved == [{"action": "save", "text": "preference"}]
    assert agent.client.llm_call_count == 1
    request = agent.client.calls[0]
    assert [t["function"]["name"] for t in request["tools"]] == ["memory"]
    assert request["messages"][-1]["role"] == "user"
    assert "session is being compressed" in request["messages"][-1]["content"]
    assert request["messages"][-1]["_flush"] is True
    assert request["messages"][0] is agent.messages[0]
    assert agent.messages is original and agent.messages == before


def test_flush_bypasses_text_strategy_and_caching():
    agent, saved = make_agent([
        text_msg('call:memory{"action":"save","text":"preference"}'),
    ], model="gemma", enable_prompt_caching=True)
    before = copy.deepcopy(agent.messages)
    with patch.object(agent._strategy, "prepare_kwargs", side_effect=AssertionError("strategy")):
        flush_memories(agent, agent.messages)
    assert saved == []
    request = agent.client.calls[0]
    assert [t["function"]["name"] for t in request["tools"]] == ["memory"]
    assert request["messages"][0]["content"] == "frozen"
    assert agent.messages == before and agent.system_prompt == "frozen"


def test_flush_guards_skip_api():
    for case in ("handler", "turns", "messages"):
        agent, _ = make_agent()
        min_turns = 0
        if case == "handler":
            agent.tool_handlers.clear()
        elif case == "turns":
            min_turns = 2
        else:
            agent.messages.pop()
        before = copy.deepcopy(agent.messages)
        flush_memories(agent, agent.messages, min_turns=min_turns)
        assert agent.client.llm_call_count == 0
        assert agent.messages == before


def test_flush_without_schema_still_calls_api():
    agent, _ = make_agent([text_msg("nothing to save")])
    agent.tools.clear()
    before = copy.deepcopy(agent.messages)
    flush_memories(agent, agent.messages)
    assert agent.client.llm_call_count == 1
    assert agent.client.calls[0]["tools"] is None
    assert agent.messages == before


def test_flush_api_failure_cleans_appended_artifacts():
    agent, _ = make_agent()
    before = copy.deepcopy(agent.messages)
    original = agent.messages

    def fail(**kwargs):
        assert agent.messages[-1]["_flush"] is True
        assert kwargs["messages"][-1] is agent.messages[-1]
        agent.messages.append({"role": "assistant", "content": "temporary"})
        raise RuntimeError("offline")

    agent.client.chat.completions.create = fail
    flush_memories(agent, agent.messages)
    assert agent.messages is original and agent.messages == before


def test_flush_invalid_json_stops_remaining_calls():
    msg = structured_msg(calls=[("memory", {}), ("memory", {"text": "later"})])
    msg.tool_calls[0].function.arguments = "not json"
    agent, saved = make_agent([msg])
    before = copy.deepcopy(agent.messages)
    flush_memories(agent, agent.messages)
    assert saved == []
    assert agent.messages == before


def test_flush_tool_failure_does_not_prevent_later_save():
    agent, saved = make_agent([structured_msg(calls=[
        ("memory", ["invalid arguments"]), ("memory", {"text": "valid"}),
    ])])
    before = copy.deepcopy(agent.messages)
    flush_memories(agent, agent.messages)
    assert saved == [{"text": "valid"}]
    assert agent.messages == before


def test_flush_no_tools_returned_is_noop():
    agent, saved = make_agent([text_msg("nothing to save")])
    before = copy.deepcopy(agent.messages)
    flush_memories(agent, agent.messages)
    assert not saved and agent.client.llm_call_count == 1
    assert agent.messages == before


class StubCompressor:
    max_context_tokens = 100
    THRESHOLD = 0.5

    def __init__(self, estimated, saved):
        self.estimated = estimated
        self.saved = saved
        self.calls = 0

    def _estimate_tokens(self, messages):
        return self.estimated

    def maybe_compress(self, messages):
        assert self.saved == [{"text": "remembered"}]
        assert not any("session is being compressed" in (m.get("content") or "") for m in messages)
        self.calls += 1
        return [messages[0], {"role": "assistant", "content": "summary"}, messages[-1]]


def test_compression_flushes_before_compressing_without_recursion():
    for estimated in (49, 50):
        script = [text_msg("answer")]
        if estimated == 50:
            script.insert(0, structured_msg(calls=[("memory", {"text": "remembered"})]))
        agent, saved = make_agent(script)
        compressor = StubCompressor(estimated, saved)
        agent.set_compressor(compressor)
        assert agent.run("next") == "answer"
        assert compressor.calls == int(estimated == 50)
        assert agent.client.llm_call_count == 1 + int(estimated == 50)
        assert agent.messages[0] == {"role": "system", "content": "frozen"}
        agent.set_compressor(None)
        with patch.object(compressor, "_estimate_tokens", side_effect=AssertionError("disabled")):
            agent.run("again")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\n全部通过")