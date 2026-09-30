"""SessionDB 的离线测试(第 7 章)。

跑法:
    pytest tests/
    python3 tests/test_session_db.py

验证点:
  1. 会话的创建 / 结束 / 计数
  2. 消息按时间顺序落库,limit 取最近 N 条
  3. FTS5 检索命中并带高亮 snippet
  4. 只索引 user/assistant,且空内容不进索引
  5. 查询清洗:特殊字符、FTS5 关键字、空查询都不炸
  6. session_id / roles 过滤
  7. delete_session 同时清掉 fts 索引(不留幽灵命中)
  8. append_openai_message 能拍平 content-block 和 tool_calls
  9. 落盘的库重开后数据还在
 10. trigram 分词能搜中文
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memory.session_db import SessionDB  # noqa: E402


def _db(**kwargs) -> SessionDB:
    return SessionDB(":memory:", **kwargs)


def _seed(db: SessionDB, source: str = "cli") -> str:
    sid = db.create_session(source=source)
    db.append_message(sid, "user", "how do I enable WAL mode in sqlite")
    db.append_message(sid, "assistant", "run PRAGMA journal_mode WAL once")
    db.append_message(sid, "tool", "zzzunique tool output", tool_name="terminal")
    return sid


# ----------------------------------------------------------------- 会话生命周期


def test_create_session_is_readable():
    db = _db()
    sid = db.create_session(source="telegram")
    row = db.get_session(sid)
    assert row is not None
    assert row["source"] == "telegram"
    assert row["started_at"] > 0
    assert row["ended_at"] is None
    assert row["message_count"] == 0


def test_message_count_tracks_appends():
    db = _db()
    sid = _seed(db)
    # tool 消息不进索引,但仍然算一条消息
    assert db.get_session(sid)["message_count"] == 3


def test_end_session_writes_summary_and_keeps_it():
    db = _db()
    sid = _seed(db)
    db.end_session(sid, summary="WAL 问答")
    row = db.get_session(sid)
    assert row["ended_at"] > 0
    assert row["summary"] == "WAL 问答"

    # summary=None 不该把已有摘要抹掉(COALESCE 的意义)
    db.end_session(sid)
    assert db.get_session(sid)["summary"] == "WAL 问答"


def test_list_sessions_returns_all_and_respects_limit():
    db = _db()
    ids = {db.create_session() for _ in range(3)}
    assert {s["id"] for s in db.list_sessions()} == ids
    assert len(db.list_sessions(limit=2)) == 2


# --------------------------------------------------------------------- 消息读写


def test_messages_come_back_in_order():
    db = _db()
    sid = _seed(db)
    msgs = db.get_session_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[2]["tool_name"] == "terminal"
    assert msgs[0]["id"] < msgs[1]["id"] < msgs[2]["id"]


def test_limit_takes_most_recent_but_keeps_chronological_order():
    db = _db()
    sid = db.create_session()
    for i in range(5):
        db.append_message(sid, "user", f"m{i}")
    msgs = db.get_session_messages(sid, limit=2)
    # 要的是最近两条,且仍按时间正序(不是倒序)
    assert [m["content"] for m in msgs] == ["m3", "m4"]


def test_messages_are_scoped_per_session():
    db = _db()
    a, b = db.create_session(), db.create_session()
    db.append_message(a, "user", "alpha")
    db.append_message(b, "user", "beta")
    assert [m["content"] for m in db.get_session_messages(a)] == ["alpha"]


def test_append_openai_message_flattens_blocks_and_tool_calls():
    db = _db()
    sid = db.create_session()
    db.append_openai_message(
        sid,
        {
            "role": "user",
            # 第 6 章打完缓存标记后就是这种 content-block 结构
            "content": [{"type": "text", "text": "hello blocks"}],
        },
    )
    db.append_openai_message(
        sid,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"name": "calc", "arguments": '{"expr": "1+1"}'}}
            ],
        },
    )
    db.append_openai_message(
        sid, {"role": "tool", "content": "2", "tool_call_id": "call_1", "name": "calc"}
    )

    msgs = db.get_session_messages(sid)
    assert msgs[0]["content"] == "hello blocks"
    assert msgs[1]["content"] == '[tool_call] calc({"expr": "1+1"})'
    assert msgs[2]["tool_call_id"] == "call_1"
    assert msgs[2]["tool_name"] == "calc"


# ----------------------------------------------------------------------- 检索


def test_search_hits_and_highlights():
    db = _db()
    sid = _seed(db)
    hits = db.search("WAL sqlite")
    assert len(hits) == 1  # 两个词是 AND,只有 user 那条同时含有
    hit = hits[0]
    assert hit["session_id"] == sid
    assert hit["role"] == "user"
    assert ">>>WAL<<<" in hit["snippet"]
    assert hit["date"] is not None


def test_tool_messages_are_not_indexed():
    db = _db()
    _seed(db)
    # tool 的内容原文还在 messages 里,但不该被检索到
    assert db.search("zzzunique") == []


def test_empty_content_is_not_indexed():
    db = _db()
    sid = db.create_session()
    db.append_message(sid, "user", "")
    db.append_message(sid, "assistant", None)
    assert db.get_session(sid)["message_count"] == 2
    assert db.search("anything") == []


def test_search_across_sessions_and_filters():
    db = _db()
    a = _seed(db, source="cli")
    b = db.create_session(source="telegram")
    db.append_message(b, "assistant", "sqlite is fine for single writer workloads")

    assert len(db.search("sqlite")) == 2
    assert [h["session_id"] for h in db.search("sqlite", session_id=b)] == [b]
    assert [h["role"] for h in db.search("sqlite", roles=("assistant",))] == [
        "assistant"
    ]
    assert len(db.search("sqlite", limit=1)) == 1


def test_malformed_queries_do_not_raise():
    db = _db()
    _seed(db)
    # 裸引号 / 算符 / FTS5 关键字,任何一个没清洗干净都会抛 syntax error
    for bad in ['sqlite" OR 1=1 --', "AND", "*", "NEAR(a b)", "foo:bar", "^x"]:
        assert isinstance(db.search(bad), list)
    # 没有可搜的词就直接空结果,不去打扰 FTS5
    assert db.search("") == []
    assert db.search("   ()  ") == []


def test_delete_session_also_clears_the_index():
    db = _db()
    sid = _seed(db)
    assert db.delete_session(sid) == 3
    assert db.get_session(sid) is None
    assert db.get_session_messages(sid) == []
    # 索引没清干净的话,这里会 JOIN 不到 messages 而静默丢结果甚至报错
    assert db.search("WAL") == []


def test_trigram_tokenizer_searches_chinese():
    db = _db(tokenize="trigram")
    sid = db.create_session()
    db.append_message(sid, "user", "帮我把提示词缓存的断点策略讲清楚")
    assert len(db.search("缓存策略")) == 0  # 不是连续子串
    hits = db.search("断点策略")
    assert len(hits) == 1 and hits[0]["session_id"] == sid


def test_rejects_unknown_tokenizer():
    try:
        SessionDB(":memory:", tokenize="evil'); DROP TABLE messages; --")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 tokenize 应该被拒绝")


def test_file_backed_db_survives_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "state.db")
        with SessionDB(path) as db:
            sid = _seed(db)
            assert db.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

        with SessionDB(path) as db2:
            assert len(db2.get_session_messages(sid)) == 3
            assert len(db2.search("sqlite")) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("全部通过")