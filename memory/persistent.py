"""持久记忆:USER.md + MEMORY.md(第 8 章)。

会话记忆(ch7)回答"发生了什么",持久记忆回答"你是谁、什么长期有用"。
它只是两个 markdown 文件,没有数据库、没有网络:

    data/USER.md    用户画像:偏好、习惯、技术栈     上限 1375 字符
    data/MEMORY.md  长期观察:项目上下文、验证过的解法  上限 2200 字符

为什么用文件而不是表:这两个文件是**人的可读资产**,出问题时你直接打开
看一眼就知道 agent 记住了什么,并且能手工删掉错误记忆。用数据库就做不到
"随手 git diff 一下"。

两个刻意的设计:

1. 上限来自 Hermes 的真实默认值(2200 / 1375)。这段文本会在 session 启动
   时被塞进系统提示词,也就是**每一次 API 调用都要重发一遍**。天花板压得
   住,成本才可预测;写不下时淘汰最旧的条目,而不是让提示词无限膨胀。
2. load() 在 session 启动时读一次,之后是**冻结快照**(第 5 章)。session 中途
   新写的观察会落盘,但不会改动已冻结的系统提示词 —— 否则 prompt cache 每
   轮失效,省下的钱全吐回去。

写入用 tempfile + os.replace 做原子替换:后台复盘线程(ch13)和主线程可能同时
在写,读到"写了一半"的文件比丢一条记忆更糟。同进程内再用一把锁串行化,
避免两个线程互相覆盖。
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

MEMORY_FILE = "MEMORY.md"
USER_FILE = "USER.md"

# 条目分隔符:每条观察一行,便于按行淘汰
ENTRY_PREFIX = "- "


class PersistentMemory:
    MEMORY_LIMIT = 2200  # MEMORY.md 字符上限(Hermes 默认)
    USER_LIMIT = 1375  # USER.md 字符上限(Hermes 默认)
    TRUNCATION_MARK = "...[truncated]"  # 单条就超限时的截断标记

    MEMORY_HEADING = "### Observations"
    USER_HEADING = "### User Profile"

    def __init__(self, data_dir: str | os.PathLike = "data"):
        self.data_dir = Path(data_dir)
        self.memory_path = self.data_dir / MEMORY_FILE
        self.user_path = self.data_dir / USER_FILE

        self.data_dir.mkdir(parents=True, exist_ok=True)
        # 文件先建出来,用户打开目录就能看到"这里能写什么"
        self.memory_path.touch(exist_ok=True)
        self.user_path.touch(exist_ok=True)

        # ch13 的后台复盘线程会和主线程并发写,同进程内先加锁
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # 读
    # ------------------------------------------------------------------ #
    def load(self) -> str:
        """把两个文件拼成一段可直接塞进系统提示词的上下文块。

        画像在前、观察在后:先声明"你是谁",再列"我记下了什么",
        模型读起来更像一份简报。两侧都为空时返回空串,让 PromptBuilder
        跳过整个章节,不留空壳标题。
        """
        with self._lock:
            user = self._read_text(self.user_path)
            memory = self._read_text(self.memory_path)

        parts: list[str] = []
        if user:
            parts.append(f"{self.USER_HEADING}\n{user}")
        if memory:
            parts.append(f"{self.MEMORY_HEADING}\n{memory}")
        return "\n\n".join(parts)

    def read_user_profile(self) -> str:
        with self._lock:
            return self._read_text(self.user_path)

    def read_memory(self) -> str:
        with self._lock:
            return self._read_text(self.memory_path)

    def observation_count(self) -> int:
        """当前存了多少条观察(demo 和测试用,不进提示词)。"""
        with self._lock:
            return len(self._entries(self._read_text(self.memory_path)))

    # ------------------------------------------------------------------ #
    # 写
    # ------------------------------------------------------------------ #
    def save_observation(self, text: str) -> bool:
        """追加一条观察,超出上限时从**最旧的**开始淘汰。

        返回 False 表示没有写入(空内容)。淘汰策略是最旧优先:
        越早写下的记忆越可能已经过时,而最近的事实仍在校准期。
        注意观察区是纯条目列表,不要手工在这里写标题,否则标题会在淘汰时被吃掉。
        """
        text = self._normalize_entry(text)
        if not text:
            return False

        entry = ENTRY_PREFIX + text

        # 单条就超过整个上限:只能截断,否则会把之前的记忆全挤掉
        if len(entry) > self.MEMORY_LIMIT:
            room = self.MEMORY_LIMIT - len(ENTRY_PREFIX) - len(self.TRUNCATION_MARK)
            entry = ENTRY_PREFIX + text[:room] + self.TRUNCATION_MARK

        with self._lock:
            entries = self._entries(self._read_text(self.memory_path))
            entries.append(entry)
            trimmed = self._fit(entries, self.MEMORY_LIMIT)
            self._write_text(self.memory_path, trimmed)

        # 提醒调用方:冻结快照只在下次 session 生效,别指望本轮系统提示词变化
        return True

    def update_user_profile(self, text: str) -> None:
        """整体替换用户画像(画像是一份"当前结论",不是流水账)。

        传空串等于清空画像。超过上限直接截断,不淘汰旧内容 ——
        画像是被反复重写的,不存在"最旧一条"的概念。
        """
        cleaned = self._normalize_block(text)
        if len(cleaned) > self.USER_LIMIT:
            cleaned = cleaned[: self.USER_LIMIT].rstrip()
        with self._lock:
            self._write_text(self.user_path, cleaned)

    def clear(self) -> None:
        """清空两份记忆(调试和"从零开始"时用)。"""
        with self._lock:
            self._write_text(self.memory_path, "")
            self._write_text(self.user_path, "")

    # ------------------------------------------------------------------ #
    # 内部:文本处理
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_entry(text: str) -> str:
        """把任意输入压成一行条目:换行和连续空白会被折成单空格。

        为什么必须折:文件是按行记的,一条观察拆成多行会让淘汰逻辑
        只删掉半条记忆,留下语义残缺的残句。
        """
        if not text:
            return ""
        return " ".join(str(text).split())

    @staticmethod
    def _normalize_block(text: str) -> str:
        """块级归一化:去行尾空白、去首尾空行,但保留段落结构。"""
        if not text:
            return ""
        lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").split("\n")]
        return "\n".join(lines).strip()

    @staticmethod
    def _entries(text: str) -> list[str]:
        """按行拆成条目,丢掉空行(空行不占预算也不参与淘汰)。"""
        return [line for line in text.split("\n") if line.strip()]

    @staticmethod
    def _fit(entries: list[str], limit: int) -> str:
        """从头部(最旧)开始丢弃,直到整体不超过 limit。"""
        while entries and len("\n".join(entries)) > limit:
            entries.pop(0)
        return "\n".join(entries)

    # ------------------------------------------------------------------ #
    # 内部:落盘
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    def _write_text(self, path: Path, text: str) -> None:
        """原子写:同目录建临时文件再 os.replace,读者永远看到完整内容。"""
        payload = text.rstrip("\n")
        if payload:
            payload += "\n"  # 文件以换行结尾,git diff 才干净
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".md")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_path, path)  # 同分区 rename,原子生效
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


if __name__ == "__main__":  # 手动 demo:python3 memory/persistent.py
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as tmp:
        mem = PersistentMemory(tmp)

        print("空记忆 load() ->", repr(mem.load()))

        mem.update_user_profile("### 偏好\n- 回复用中文\n- macOS / zsh")
        mem.save_observation("学习 mini-hermes 教程,目录结构对齐教程布局")
        mem.save_observation("环境没装 pytest,测试要能 python3 直跑")

        print("\n--- load() ---")
        print(mem.load())
        print(f"\n观察条数: {mem.observation_count()} / 上限 {PersistentMemory.MEMORY_LIMIT} 字符")

        # 灌到上限之外,验证最旧优先淘汰
        for i in range(300):
            mem.save_observation(f"填充条目 {i}")
        raw = mem.memory_path.read_text()
        print(f"\n灌满后长度: {len(raw.strip())} 字符,条数: {mem.observation_count()}")
        print("最旧的一条还在吗:", "学习 mini-hermes 教程" in raw)
        print("最新的一条在吗:", "填充条目 299" in raw)