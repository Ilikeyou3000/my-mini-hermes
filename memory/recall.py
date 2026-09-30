"""跨会话召回(第 9 章):FTS5 检索 + LLM 摘要。

第 7 章把每句对话都写进了 SQLite,但"存下来"不等于"用得上":
一个历史会话动辄上万 token,直接塞回上下文会把预算瞬间烧光。

所以召回分四步走:
  1. FTS5 检索       —— 找到命中查询的消息(跨所有会话)
  2. 按会话聚合       —— 取 top N 个**不同会话**,而不是 N 条零散消息
  3. 拉回对话上下文   —— 单条 snippet 没有前因后果,要把周围消息带上
  4. LLM 压缩成摘要   —— 3 个会话 × 1 万 token 压成 3 段 200 字

第 4 步是省钱的关键:一次便宜模型的摘要调用,换掉后续每轮都要重复
传输的几万 token。

降级原则:召回是"锦上添花"的能力,任何一环出问题都只能降级、不能抛异常
把整轮对话搞崩 —— 没检索到就返回空,没有 client 或摘要失败就退回
截断后的原始摘录。

demo: python3 memory/recall.py
"""

from __future__ import annotations

from typing import Any

SUMMARY_SYSTEM_PROMPT = (
    "Summarize this past conversation, focusing on information relevant to "
    "the given topic. Be concise. 200 words max."
)

NO_RESULT_MESSAGE = "No relevant past conversations found."


class SessionRecall:
    """在历史会话里找到"当下用得上的那一段"。"""

    SEARCH_LIMIT = 30  # 先宽召回,再按会话去重收窄
    MAX_SESSIONS = 3
    TRANSCRIPT_MESSAGES = 30  # 每个会话回捞多少条上下文
    MAX_TRANSCRIPT_CHARS = 8000  # 送进摘要模型前的硬上限
    MAX_LINE_CHARS = 500  # 单条消息过长(比如贴了整个文件)先截断
    SUMMARY_MAX_TOKENS = 300
    SEPARATOR = "\n\n---\n\n"

    def __init__(
        self,
        session_db,
        client: Any | None = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = SUMMARY_MAX_TOKENS,
        current_session_id: str | None = None,
    ) -> None:
        """client 允许为 None:此时走"不摘要、只给原始摘录"的降级路径。

        current_session_id 是当前会话,召回时要排除掉 ——
        当前对话已经在上下文里了,再摘要一遍纯属浪费 token。
        """
        self.db = session_db
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.current_session_id = current_session_id

    # ------------------------------------------------------------------ 主流程

    def recall(self, query: str, max_sessions: int | None = None) -> str:
        """检索历史会话并返回摘要好的上下文;没有可用结果时返回空串。"""
        max_sessions = max_sessions or self.MAX_SESSIONS

        # 1. 宽召回
        results = self.db.search(query, limit=self.SEARCH_LIMIT)
        if not results:
            return ""

        # 2. 按会话去重,保留 FTS 的 rank 顺序(先出现 = 更相关)
        sessions = self._top_sessions(results, max_sessions)
        if not sessions:
            return ""

        # 3 + 4. 逐会话拉上下文并压缩
        blocks = []
        for session_id, meta in sessions.items():
            messages = self.db.get_session_messages(
                session_id, limit=self.TRANSCRIPT_MESSAGES
            )
            transcript = self._format_transcript(messages)
            if not transcript:
                continue
            body = self._summarize(query, transcript, meta.get("date"))
            blocks.append(self._render_block(meta, body))

        return self.SEPARATOR.join(blocks)

    def _top_sessions(
        self, results: list[dict[str, Any]], max_sessions: int
    ) -> dict[str, dict[str, Any]]:
        """把消息级命中折叠成会话级命中。

        只保留每个会话**最相关的那条**作为元信息(日期/来源/snippet),
        并在凑满 max_sessions 后停下 —— 后面的会话再相关也装不进预算。
        """
        seen: dict[str, dict[str, Any]] = {}
        for row in results:
            sid = row.get("session_id")
            if not sid or sid == self.current_session_id:
                continue
            if sid not in seen:
                if len(seen) >= max_sessions:
                    break
                seen[sid] = row
        return seen

    # ------------------------------------------------------------------ 转写

    def _format_transcript(self, messages: list[dict[str, Any]]) -> str:
        """把消息列表拍平成 "role: content" 的纯文本对话记录。

        tool 消息不进 FTS 索引,但摘要时是有用的 —— 它们记录了"实际做了什么"。
        这里保留但按 role 标注,并对超长内容做截断,防止一个大文件把预算吃掉。
        """
        lines = []
        for msg in messages:
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if len(content) > self.MAX_LINE_CHARS:
                content = content[: self.MAX_LINE_CHARS] + "...[truncated]"
            role = msg.get("role", "user")
            name = msg.get("tool_name")
            label = f"{role}({name})" if name else role
            lines.append(f"{label}: {content}")

        transcript = "\n".join(lines)
        if len(transcript) > self.MAX_TRANSCRIPT_CHARS:
            # 留尾不留头:会话末尾往往是结论,开头多是寒暄
            transcript = "...[truncated]\n" + transcript[-self.MAX_TRANSCRIPT_CHARS :]
        return transcript

    def _render_block(self, meta: dict[str, Any], body: str) -> str:
        """给每段摘要加上"哪场会话、什么时候"的抬头,方便模型判断时效性。"""
        head = f"[session {str(meta.get('session_id'))[:8]}"
        if meta.get("date"):
            head += f" | {meta['date']}"
        if meta.get("source"):
            head += f" | {meta['source']}"
        head += "]"
        return f"{head}\n{body}"

    # ------------------------------------------------------------------ 摘要

    def _summarize(self, topic: str, transcript: str, date: str | None) -> str:
        """用便宜的模型把一段 transcript 压成焦点摘要。

        没有 client(离线/测试)或调用失败时,退回原始摘录:
        信息密度差一点,但总比召回直接失效好。
        """
        if self.client is None:
            return self._fallback(transcript)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Topic: {topic}\n"
                            f"Date: {date or 'unknown'}\n\n"
                            f"TRANSCRIPT:\n{transcript}"
                        ),
                    },
                ],
                max_tokens=self.max_tokens,
            )
            summary = (resp.choices[0].message.content or "").strip()
            return summary or self._fallback(transcript)
        except Exception as exc:  # noqa: BLE001 - 召回失败不该影响主对话
            return f"[summary unavailable: {exc}]\n{self._fallback(transcript)}"

    def _fallback(self, transcript: str, limit: int = 1200) -> str:
        """降级输出:截断后的原始对话摘录。"""
        if len(transcript) <= limit:
            return transcript
        return transcript[:limit] + "...[truncated]"


# ---------------------------------------------------------------- 工具注册


SESSION_SEARCH_DESCRIPTION = (
    "Search past conversations for relevant context. "
    "Use when you need to recall what was discussed previously, "
    "e.g. the user refers to a decision, file, or approach from an earlier session."
)

SESSION_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search topic or keywords",
        },
    },
    "required": ["query"],
}


def register_session_search(recall: SessionRecall, registry=None) -> None:
    """把 recall 实例挂成 session_search 工具。

    这里用工厂函数而不是模块级 @registry.tool 装饰器:
    SessionRecall 需要数据库和 client,只有运行期(build_default_agent)
    才知道该注入什么 —— 导入即注册会逼所有人都先准备好一个 db。
    """
    if registry is None:
        from tool_registry import registry as default_registry

        registry = default_registry

    def session_search(query: str) -> str:
        # 工具返回值最终要进上下文,空串对模型是噪音,给一句明确的"没找到"
        return recall.recall(query) or NO_RESULT_MESSAGE

    registry.register(
        name="session_search",
        description=SESSION_SEARCH_DESCRIPTION,
        parameters=SESSION_SEARCH_PARAMETERS,
        handler=session_search,
        category="memory",
    )


if __name__ == "__main__":
    # 直跑时 sys.path[0] 就是 memory/,平级 import 即可;
    # 走 `from memory.session_db import ...` 会把整个包再导一遍,没必要。
    from session_db import SessionDB  # type: ignore[import-not-found]

    # 中文要用 trigram 分词,unicode61 会把整句当一个 token
    db = SessionDB(":memory:", tokenize="trigram")

    old = db.create_session(source="cli")
    db.append_message(old, "user", "我们上次决定用 psycopg2 而不是 SQLAlchemy 导入 CSV")
    db.append_message(old, "assistant", "好的,已经按 psycopg2 批量插入写完脚本了")
    db.append_message(old, "user", "记住:导入前一定要先检查表是否存在")
    db.end_session(old)

    other = db.create_session(source="telegram")
    db.append_message(other, "user", "帮我看看 psycopg2 连接池怎么配")
    db.append_message(other, "assistant", "建议用 SimpleConnectionPool,minconn=1")
    db.end_session(other)

    print("=== 降级路径(没有 LLM client,返回原始摘录)===")
    recall = SessionRecall(db)
    print(recall.recall("psycopg2") or "(空)")

    print("\n=== 假 LLM 摘要 ===")

    class _FakeClient:
        """就地捏一个最小 client,证明摘要路径接得上。"""

        def __init__(self):
            from types import SimpleNamespace

            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )
            self.n = 0

        def _create(self, **kwargs):
            from types import SimpleNamespace

            self.n += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=f"摘要 #{self.n}:用户偏好 psycopg2")
                    )
                ]
            )

    client = _FakeClient()
    recall2 = SessionRecall(db, client=client, model="fake-mini")
    print(recall2.recall("psycopg2"))
    print(f"\nLLM 调用次数: {client.n}(每个命中会话一次)")

    print("\n=== 无结果 ===")
    print(repr(recall2.recall("量子计算")))

    db.close()