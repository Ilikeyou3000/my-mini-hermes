"""会话记忆:SQLite + FTS5(第 7 章)。

前 6 章的 agent 是"失忆"的:进程一退,self.messages 就没了。
这一章给它一本带索引的日记 —— 记录每一轮对话,但**不**把日记
塞回上下文,而是需要时按关键词检索(第 9 章 session_search 会用到)。

为什么是 SQLite + FTS5:
  - 标准库自带,不用额外装服务,数据就一个文件,备份/删除都简单;
  - FTS5 是 SQLite 内置的全文索引,MATCH + rank 就能拿到相关性排序;
  - 上下文开销近似常数:存全部,检索一点,注入更少。

两张表 + 一张虚拟表:
    sessions      一次会话(谁、何时开始、几条消息、摘要)
    messages      每条消息的原文(agent 重放历史时读这张)
    messages_fts  只索引 user/assistant 的 content,rowid 对齐 messages.id

为什么 FTS 只索引 user/assistant:
tool 结果动辄几万字符(agent._execute_tool 截断到 50k),
把它们塞进索引会让噪声淹没真正的对话内容,检索质量反而下降。

关于中文:默认 tokenize='unicode61' 按空白/标点切词,中文整段会被当
成一个 token,几乎搜不到。所以 SessionDB(tokenize="trigram") 提供了
三元组分词 —— 代价是索引更大、查询词至少要 3 个字符。
"""

from __future__ import annotations

import random
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Any

# 只有这两类消息进全文索引(tool 结果太长且噪声大)
INDEXED_ROLES = ("user", "assistant")

DEFAULT_SEARCH_LIMIT = 20
# 单条会话重放时默认最多取多少条消息(第 9 章召回用)
DEFAULT_MESSAGE_LIMIT = 200

# FTS5 的查询语法里这些字符有特殊含义,用户输入必须先抹掉,
# 否则一个裸引号就会让整条 MATCH 报 "fts5: syntax error"
_FTS_SPECIAL_CHARS = '"*+-():^{}[]~,./\\'

# 白名单:tokenize 是拼进 DDL 的,不能让调用方塞任意 SQL
_ALLOWED_TOKENIZERS = ("unicode61", "trigram", "porter", "ascii")

# 写冲突重试:WAL 只允许单写者,多进程(CLI + gateway)并发写会撞锁。
# 抖动退避比固定 sleep 好,避免两个进程同频重试反复互相撞。
_MAX_WRITE_RETRIES = 5
_JITTER_RANGE = (0.02, 0.15)  # 20-150ms


class SessionDB:
    """会话消息的持久化 + 全文检索。"""

    def __init__(self, db_path: str = ":memory:", tokenize: str = "unicode61"):
        if tokenize not in _ALLOWED_TOKENIZERS:
            raise ValueError(f"不支持的 tokenize: {tokenize!r}")
        self.db_path = str(db_path)
        self._tokenize = tokenize
        # check_same_thread=False:第 13 章的后台 review 线程要共用这个连接
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            # WAL:读不阻塞写,写不阻塞读
            self.conn.execute("PRAGMA journal_mode=WAL")
        # 撞锁时先让 SQLite 自己等 5 秒,应用层重试只兜最后的底
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._create_tables()

    # ------------------------------------------------------------------ 建表

    def _create_tables(self) -> None:
        self.conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS sessions (
                id            TEXT PRIMARY KEY,
                source        TEXT DEFAULT 'cli',
                started_at    REAL,
                ended_at      REAL,
                message_count INTEGER DEFAULT 0,
                summary       TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT REFERENCES sessions(id),
                role         TEXT NOT NULL,
                content      TEXT,
                tool_name    TEXT,
                tool_call_id TEXT,
                timestamp    REAL
            );

            -- 按 session 重放历史是最频繁的查询,建索引
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                USING fts5(content, tokenize='{self._tokenize}');
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------ 写入(带重试)

    def _write(self, fn):
        """执行一段写操作,撞锁时抖动退避重试。

        fn 收到 conn,自己发若干条语句;提交/回滚由这里统一负责,
        这样"插消息 + 进索引 + 更新计数"三步要么全成要么全不成。
        """
        last_err: Exception | None = None
        for attempt in range(_MAX_WRITE_RETRIES):
            try:
                with self.conn:  # 正常结束自动 commit,异常自动 rollback
                    return fn(self.conn)
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "locked" not in msg and "busy" not in msg:
                    raise
                last_err = exc
                if attempt == _MAX_WRITE_RETRIES - 1:
                    break
                time.sleep(random.uniform(*_JITTER_RANGE))
        raise sqlite3.OperationalError(
            f"写入重试 {_MAX_WRITE_RETRIES} 次仍失败: {last_err}"
        )

    # ------------------------------------------------------------------ 会话

    def create_session(self, source: str = "cli", session_id: str | None = None) -> str:
        """开一个新会话,返回 session_id。"""
        sid = session_id or str(uuid.uuid4())
        self._write(
            lambda c: c.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                (sid, source, _now()),
            )
        )
        return sid

    def end_session(self, session_id: str, summary: str | None = None) -> None:
        """标记会话结束;summary 供第 9 章召回时先看一眼摘要。"""
        self._write(
            lambda c: c.execute(
                """UPDATE sessions
                      SET ended_at = ?,
                          summary  = COALESCE(?, summary)
                    WHERE id = ?""",
                (_now(), summary, session_id),
            )
        )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _session_to_dict(row) if row else None

    def list_sessions(self, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[str, Any]]:
        """最近的会话在前。"""
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_session_to_dict(r) for r in rows]

    def delete_session(self, session_id: str) -> int:
        """删除一个会话及其消息和索引,返回删掉的消息条数。"""

        def _run(c: sqlite3.Connection) -> int:
            ids = [
                r[0]
                for r in c.execute(
                    "SELECT id FROM messages WHERE session_id = ?", (session_id,)
                )
            ]
            # fts 表没有外键级联,rowid 得手动清,否则检索会命中幽灵消息
            c.executemany(
                "DELETE FROM messages_fts WHERE rowid = ?", [(i,) for i in ids]
            )
            c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return len(ids)

        return self._write(_run)

    # ------------------------------------------------------------------ 消息

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str | None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        timestamp: float | None = None,
    ) -> int:
        """落库一条消息,返回 messages.id。"""
        ts = _now() if timestamp is None else timestamp

        def _run(c: sqlite3.Connection) -> int:
            cur = c.execute(
                """INSERT INTO messages
                       (session_id, role, content, tool_name, tool_call_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, role, content, tool_name, tool_call_id, ts),
            )
            row_id = int(cur.lastrowid)
            if role in INDEXED_ROLES and content:
                # rowid 显式对齐 messages.id,检索时才能 JOIN 回原文
                c.execute(
                    "INSERT INTO messages_fts (rowid, content) VALUES (?, ?)",
                    (row_id, content),
                )
            c.execute(
                "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                (session_id,),
            )
            return row_id

        return self._write(_run)

    def append_openai_message(self, session_id: str, msg: dict[str, Any]) -> int:
        """把 agent.messages 里的一条 OpenAI 格式消息直接落库。

        agent 的消息可能是 content-block 数组(第 6 章打完缓存标记后),
        也可能只有 tool_calls 没有 content,这里统一拍平成一段文本,
        保证日记始终可读、可检索。
        """
        return self.append_message(
            session_id=session_id,
            role=str(msg.get("role", "user")),
            content=_content_to_text(msg),
            tool_name=msg.get("name") or msg.get("tool_name"),
            tool_call_id=msg.get("tool_call_id"),
        )

    def get_session_messages(
        self, session_id: str, limit: int | None = DEFAULT_MESSAGE_LIMIT
    ) -> list[dict[str, Any]]:
        """按时间顺序返回会话消息;limit 时取**最近** N 条(仍按时间正序)。"""
        if limit is None:
            rows = self.conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        else:
            # 先倒序截断再翻回来:要的是"最近的上下文",不是开头的寒暄
            rows = self.conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        return [_message_to_dict(r) for r in rows]

    # ------------------------------------------------------------------ 检索

    def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        session_id: str | None = None,
        roles: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """跨会话全文检索,按 FTS5 的 rank(BM25)排序。"""
        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []

        sql = """
            SELECT m.id, m.session_id, m.role, m.content, m.timestamp,
                   snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                   s.source, s.started_at
              FROM messages_fts
              JOIN messages m ON m.id = messages_fts.rowid
              JOIN sessions s ON s.id = m.session_id
             WHERE messages_fts MATCH ?
        """
        params: list[Any] = [sanitized]
        if session_id:
            sql += " AND m.session_id = ?"
            params.append(session_id)
        if roles:
            sql += f" AND m.role IN ({','.join('?' * len(roles))})"
            params.extend(roles)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # 兜底:清洗漏了某种语法,也不该让 agent 的一次检索崩掉整轮对话
            return []

        return [
            {
                "message_id": r["id"],
                "session_id": r["session_id"],
                "role": r["role"],
                "content": r["content"],
                "snippet": r["snippet"],
                "source": r["source"],
                "date": _to_iso(r["started_at"]),
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """把用户输入降级成纯关键词,避免 FTS5 语法错误和注入。

        返回空串表示"没有可搜的词",调用方直接返回空结果 ——
        比拿 '""' 去 MATCH 更诚实。
        """
        if not query:
            return ""
        cleaned = query
        for ch in _FTS_SPECIAL_CHARS:
            cleaned = cleaned.replace(ch, " ")
        words = [w for w in cleaned.split() if w]
        # AND/OR/NOT/NEAR 是 FTS5 关键字,裸着出现会被当算符,用引号裹成普通词
        return " ".join(f'"{w}"' for w in words)

    # ------------------------------------------------------------------ 杂务

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SessionDB":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------- 模块级工具


def _now() -> float:
    return datetime.now().timestamp()


def _to_iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts).isoformat() if ts else None


def _session_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "source": row["source"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "started_at_iso": _to_iso(row["started_at"]),
        "message_count": row["message_count"],
        "summary": row["summary"],
    }


def _message_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "role": row["role"],
        "content": row["content"],
        "tool_name": row["tool_name"],
        "tool_call_id": row["tool_call_id"],
        "timestamp": row["timestamp"],
        "timestamp_iso": _to_iso(row["timestamp"]),
    }


def _content_to_text(msg: dict[str, Any]) -> str:
    """把 OpenAI 消息的 content 拍平成一段文本。"""
    content = msg.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
    # 只有 tool_calls 没有 content 的 assistant 消息:记下它调了什么,
    # 否则日记里会出现一条空白消息,重放时看不懂发生了什么
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        parts.append(f"[tool_call] {fn.get('name', '?')}({fn.get('arguments', '')})")
    return "\n".join(p for p in parts if p)


if __name__ == "__main__":  # 手动 demo:python3 memory/session_db.py
    db = SessionDB(":memory:")
    sid = db.create_session(source="demo")
    db.append_message(sid, "user", "how do I enable WAL mode in sqlite?")
    db.append_message(sid, "assistant", "run PRAGMA journal_mode=WAL once per db file")
    db.append_message(sid, "tool", "x" * 100, tool_name="terminal")
    db.end_session(sid, summary="sqlite WAL 问答")

    print("session:", db.get_session(sid))
    print("messages:", len(db.get_session_messages(sid)))
    for hit in db.search("WAL sqlite"):
        print(f"  [{hit['role']}] {hit['snippet']}")
    print("tool 消息不进索引:", db.search("xxxxx") == [])
    db.close()