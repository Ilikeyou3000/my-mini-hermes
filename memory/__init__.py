"""记忆层(第 7-9 章)。

三层记忆里的前两层都放在这个包:
  - session_db.py  会话记忆:SQLite + FTS5,记录"发生了什么"(ch7)
  - persistent.py  持久记忆:MEMORY.md / USER.md,记录"你是谁"(ch8)
  - recall.py      跨会话召回:FTS5 检索 + LLM 摘要(ch9)

这里做一次再导出,外部只需 `from memory import SessionDB, PersistentMemory`,
后续章节新增模块时不用改调用方的 import。
"""

from .persistent import PersistentMemory
from .session_db import SessionDB

__all__ = ["PersistentMemory", "SessionDB"]