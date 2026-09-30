"""上下文压缩离线测试：python3 tests/test_compression.py。"""

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent import Agent
from compression import ContextCompressor
from fake_llm import FakeClient, structured_msg, text_msg


def message(content, role="user"):
    return {"role": role, "content": content}


def test_defaults_and_estimation():
    compressor = ContextCompressor(FakeClient([]), "summary-model")
    assert compressor.max_context_tokens == 32000
    assert compressor.max_tokens == 400
    assert compressor.THRESHOLD == 0.5
    assert compressor.TAIL_TOKENS == 20000
    assert compressor.HEAD_MESSAGES == 3
    assert compressor.CHARS_PER_TOKEN == 3.5
    messages = [message("abc"), message("defg"), message(None), {"role": "assistant"}]
    assert compressor._estimate_tokens(messages) == 2


def test_threshold_and_middle_out():
    for estimated in (49, 50):
        client = FakeClient([text_msg("summary")])
        compressor = ContextCompressor(client, "summary-model", max_context_tokens=100)
        compressor.TAIL_TOKENS = 2
        head = [message("head") for _ in range(3)]
        middle = [message("middle" * 10)]
        tail = [message("1234567")]
        messages = head + middle + tail
        with patch.object(compressor, "_estimate_tokens", return_value=estimated):
            result = compressor.maybe_compress(messages)
        if estimated == 49:
            assert result is messages and client.llm_call_count == 0
        else:
            assert result == head + [message("[Compressed context summary]\nsummary", "system")] + tail
            assert all(result[i] is head[i] for i in range(3))
            assert result[-1] is tail[0]
            assert client.calls[0]["messages"][1]["content"] == "user: " + middle[0]["content"]
        assert messages == head + middle + tail


def test_no_middle_returns_original():
    client = FakeClient([])
    compressor = ContextCompressor(client, "model", max_context_tokens=1)
    for count in (0, 2, 3, 5):
        messages = [message("content") for _ in range(count)]
        assert compressor.maybe_compress(messages) is messages
    assert client.llm_call_count == 0


def test_tail_budget_stops_at_first_oversized_message():
    compressor = ContextCompressor(FakeClient([]), "model")
    head = [message("head") for _ in range(3)]
    messages = head + [message("old"), message("x" * 35), message("1234567")]
    assert compressor._get_tail(messages, 2) == messages[-1:]
    assert compressor._get_tail(messages, 1) == []
    assert compressor._get_tail(messages, 100) == messages[3:]
    assert compressor._get_tail(head, 100) == []


def test_empty_tail_compresses_all_after_head():
    client = FakeClient([text_msg("summary")])
    compressor = ContextCompressor(client, "model", max_context_tokens=1)
    compressor.TAIL_TOKENS = 0
    messages = [message("content") for _ in range(5)]
    result = compressor.maybe_compress(messages)
    assert result == messages[:3] + [message("[Compressed context summary]\nsummary", "system")]
    assert client.calls[0]["messages"][1]["content"] == "user: content\nuser: content"


def test_summary_request_and_transcript():
    client = FakeClient([text_msg("summary")])
    compressor = ContextCompressor(client, "summary-model", max_tokens=123)
    assert compressor._summarize_middle([message("x" * 201), message(None, "assistant")]) == "summary"
    assert client.calls == [{
        "model": "summary-model",
        "max_tokens": 123,
        "messages": [message(
            "Summarize this conversation segment. Include:\n"
            "- Questions that were resolved\n"
            "- Decisions that were made\n"
            "- Pending work items\n"
            "- Key facts discovered\n"
            "Be concise. Under 500 words.", "system"),
            message("user: " + "x" * 200 + "\nassistant: [tool call]")],
    }]


def test_summary_failure_placeholders():
    client = FakeClient([text_msg(None), text_msg("")])
    compressor = ContextCompressor(client, "model", max_context_tokens=1)
    assert compressor._summarize_middle([message("content")]) == "(compression failed)"
    assert compressor._summarize_middle([message("content")]) == "(compression failed)"
    compressor.TAIL_TOKENS = 0
    messages = [message("content") for _ in range(4)]
    with patch.object(client.chat.completions, "create", side_effect=RuntimeError("offline")):
        result = compressor.maybe_compress(messages)
    assert result[-1]["content"] == "[Compressed context summary]\n(compression error: offline)"


def test_agent_flush_summary_then_answer():
    saved = []
    client = FakeClient([
        structured_msg(calls=[("memory", {"text": "preference"})]),
        text_msg("summary"), text_msg("answer"),
    ])
    tools = [{"type": "function", "function": {"name": "memory", "parameters": {}}}]
    agent = Agent(client, "model", "system", tools=tools,
                  tool_handlers={"memory": lambda **args: saved.append(args)})
    agent.configure_learning(0, 0)
    agent.messages.extend([message("old"), message("reply", "assistant"), message("middle")])
    compressor = ContextCompressor(client, "summary-model", max_context_tokens=1)
    compressor.TAIL_TOKENS = 2
    agent.set_compressor(compressor)
    assert agent.run("1234567") == "answer"
    assert saved == [{"text": "preference"}]
    assert [call["model"] for call in client.calls] == ["model", "summary-model", "model"]
    assert client.calls[1]["messages"][1]["content"] == "user: middle"
    assert agent.messages[3]["content"] == "[Compressed context summary]\nsummary"
    assert not any("_flush" in m for m in agent.messages)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\n全部通过")