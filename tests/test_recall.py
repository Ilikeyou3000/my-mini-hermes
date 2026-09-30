"""第 9 章:跨会话召回 SessionRecall + session_search 工具。

召回的价值全在"省钱"和"不添乱"两件事上,所以测试也盯着这两点:

  1. 按会话聚合,而不是按消息 —— 否则 3 个会话的 30 条命中会挤爆上下文
  2. 排除 current_session_id —— 当前对话已经在上下文里,重复摘要等于白烧钱
  3. 每个命中会话只调用一次摘要模型,且带上 topic 和 date
  4. 每一环都可降级:无结果返回空、无 client 用原始摘录、模型报错也不抛异常
  5. 送进模型的 transcript 有上限 —— 贴了整个文件的会话不该撑爆摘要调用

环境没装 pytest,直接 `python3 tests/test_recall.py`。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memory.recall import (  # noqa: E402
    NO_RESULT_MESSAGE,
    SessionRecall,
    register_session_search,
)
from memory.session_db import SessionDB  # noqa: E402
from tool_registry import ToolRegistry  # noqa: E402

TOPIC = "psycopg2"


class CountingClient:
    """记录每次请求的假 LLM,用返回值或异常模拟正常/失败两种路径。"""

    def __init__(self, reply: str = "SUMMARY", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))]
        )


def _db() -> SessionDB:
    # unicode61 会把整句中文当一个 token,trigram 才能按 3 字符滑窗命中
    return SessionDB(":memory:", tokenize="trigram")


def _seed(db: SessionDB, source: str, *messages: tuple[str, str]) -> str:
    sid = db.create_session(source=source)
    for role, content in messages:
        db.append_message(sid, role, content)
    db.end_session(sid)
    return sid


def _seed_topic(db: SessionDB, source: str) -> str:
    """三场都在谈 psycopg2 的会话,用来验证聚合与上限。"""
    return _seed(
        db,
        source,
        ("user", f"{source}: 我们决定用 psycopg2 而不是 SQLAlchemy"),
        ("assistant", f"{source}: 已按 psycopg2 批量插入实现"),
    )


def _recall(db: SessionDB, **kwargs) -> SessionRecall:
    return SessionRecall(db, **kwargs)


# ---------------------------------------------------------------- 检索与聚合 #


def test_no_match_returns_empty_string():
    db = _db()
    _seed(db, "cli", ("user", "hello there"))
    assert _recall(db).recall("kubernetes") == "", "没命中就该返回空串,不是报错"
    db.close()


def test_empty_query_returns_empty_string():
    db = _db()
    _seed_topic(db, "cli")
    assert _recall(db).recall("") == ""
    assert _recall(db).recall("()*") == "", "全是特殊字符,清洗后无可搜词"
    db.close()


def test_groups_by_session_not_by_message():
    db = _db()
    _seed_topic(db, "cli")
    # 单场会话里塞 5 条相关消息:去重后仍只算一场
    extra = db.create_session(source="extra")
    for i in range(5):
        db.append_message(extra, "user", f"psycopg2 note {i}")
    db.end_session(extra)

    out = _recall(db).recall(TOPIC)
    assert out.count("[session ") == 2, "同一场会话的多条命中只能产出 1 个摘要块"
    assert out.count("---") == 1, "两个块之间只该有一条分隔线"
    db.close()


def test_max_sessions_caps_blocks():
    db = _db()
    for i in range(5):
        _seed_topic(db, f"src{i}")
    out = _recall(db).recall(TOPIC, max_sessions=2)
    assert out.count("[session ") == 2
    db.close()


def test_current_session_is_excluded():
    db = _db()
    cur = _seed_topic(db, "current")
    other = _seed_topic(db, "other")

    out = _recall(db, current_session_id=cur).recall(TOPIC)
    assert cur[:8] not in out, "当前会话已在上下文里,不该被摘要一遍"
    assert other[:8] in out
    db.close()


def test_all_matches_in_current_session_returns_empty():
    db = _db()
    cur = _seed_topic(db, "current")
    assert _recall(db, current_session_id=cur).recall(TOPIC) == ""
    db.close()


# ------------------------------------------------------------------ 摘要路径 #


def test_client_called_once_per_session_with_topic_and_model():
    db = _db()
    _seed_topic(db, "cli")
    _seed_topic(db, "telegram")

    client = CountingClient(reply="用户偏好 psycopg2")
    out = _recall(db, client=client, model="fake-mini").recall(TOPIC)

    assert len(client.calls) == 2, "每个命中会话一次摘要调用"
    assert out.count("用户偏好 psycopg2") == 2
    for call in client.calls:
        assert call["model"] == "fake-mini"
        assert call["max_tokens"] == SessionRecall.SUMMARY_MAX_TOKENS
        assert TOPIC in call["messages"][1]["content"], "摘要必须知道要围绕什么主题"
    db.close()


def test_summary_block_carries_session_metadata():
    db = _db()
    _seed_topic(db, "telegram")
    out = _recall(db, client=CountingClient(reply="抽象")).recall(TOPIC)
    head = out.splitlines()[0]
    assert head.startswith("[session ")
    assert "telegram" in head, "来源要标出来,方便模型判断可信度"
    assert "T" in head, "日期要标出来 —— 旧结论可能已经过时"
    db.close()


def test_without_client_falls_back_to_raw_transcript():
    db = _db()
    _seed_topic(db, "cli")
    client = CountingClient()
    out = _recall(db, client=None).recall(TOPIC)

    assert "psycopg2" in out, "降级也要给出内容"
    assert "user:" in out and "assistant:" in out, "降级输出是原始对话摘录"
    assert client.calls == [], "没有 client 就不该有调用"
    db.close()


def test_llm_error_does_not_propagate():
    db = _db()
    _seed_topic(db, "cli")
    client = CountingClient(error=RuntimeError("429 rate limited"))

    out = _recall(db, client=client).recall(TOPIC)
    assert "429 rate limited" in out, "摘要失败要说清原因"
    assert "psycopg2" in out, "失败后仍要退回原始摘录,召回不能整体失效"
    db.close()


def test_empty_summary_falls_back_to_transcript():
    db = _db()
    _seed_topic(db, "cli")
    out = _recall(db, client=CountingClient(reply="   ")).recall(TOPIC)
    assert "user:" in out, "模型返回空内容时不能被当成有效摘要"
    db.close()


# ------------------------------------------------------------------ 转写限流 #


def test_long_message_is_truncated_per_line():
    db = _db()
    sid = db.create_session(source="cli")
    db.append_message(sid, "user", TOPIC + " " + "x" * 3000)
    db.end_session(sid)

    out = _recall(db).recall(TOPIC)
    assert "...[truncated]" in out
    assert len(out) < 2000, "单条超长消息必须先截断"
    db.close()


def test_oversized_transcript_is_capped():
    db = _db()
    sid = db.create_session(source="cli")
    for i in range(120):
        db.append_message(sid, "user", f"psycopg2 记录 {i} " + "y" * 300)
    db.end_session(sid)

    out = _recall(db).recall(TOPIC)
    assert "...[truncated]" in out
    # 抬头 + 分隔线 + 上限 + 收尾提示,远小于 8000
    assert len(out) < SessionRecall.MAX_TRANSCRIPT_CHARS + 500
    db.close()


def test_tool_results_are_labelled_in_transcript():
    db = _db()
    sid = db.create_session(source="cli")
    db.append_message(sid, "user", f"跑一下 {TOPIC} 脚本")
    db.append_message(sid, "tool", "row count = 42", tool_name="terminal")
    db.end_session(sid)

    out = _recall(db).recall(TOPIC)
    assert "tool(terminal):" in out, "工具消息要带名字,否则看不出'做了什么'"
    db.close()


# -------------------------------------------------------------- session_search #


def test_session_search_tool_registered_under_memory_category():
    db = _db()
    _seed_topic(db, "cli")
    reg = ToolRegistry()
    register_session_search(_recall(db), registry=reg)

    names = [s["function"]["name"] for s in reg.get_schemas()]
    assert names == ["session_search"]
    assert "session_search" in reg.get_handlers(categories=["memory"])
    schema = reg.get_schemas(categories=["memory"])[0]["function"]
    assert schema["parameters"]["required"] == ["query"]
    assert "past conversations" in schema["description"]
    db.close()


def test_session_search_tool_returns_message_when_nothing_found():
    db = _db()
    reg = ToolRegistry()
    register_session_search(_recall(db), registry=reg)

    out = reg.execute("session_search", {"query": "nothing here"})
    assert out == NO_RESULT_MESSAGE, "空串对模型是噪音,要换成明确的一句话"
    db.close()


def test_session_search_tool_returns_recall_result():
    db = _db()
    _seed_topic(db, "cli")
    reg = ToolRegistry()
    register_session_search(_recall(db, client=CountingClient(reply="命中摘要")), registry=reg)

    out = reg.execute("session_search", {"query": TOPIC})
    assert "命中摘要" in out
    db.close()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("全部通过")