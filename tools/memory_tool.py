"""记忆工具（第 8-9 章），遵循参考项目的初始化和注册方式。

启动时调用 set_memory 注入依赖；导入本模块即注册工具。
本地持久化方法返回布尔值而非提示文本，且画像读取方法名不同，
因此只在这两处适配；写入不更新正在使用的系统提示词快照。
"""

from tool_registry import registry


_persistent_memory = None
_session_recall = None


def set_memory(persistent_memory, session_recall=None):
    """由启动入口注入持久记忆和可选的会话召回依赖。"""
    global _persistent_memory, _session_recall
    _persistent_memory = persistent_memory
    _session_recall = session_recall


def memory(action: str, text: str = "") -> str:
    """统一记忆工具：保存观察、更新画像、读取和搜索。"""
    if _persistent_memory is None:
        return "Error: memory not initialized"

    if action == "save":
        if not text:
            return "Error: text is required for save"
        # 本地返回是否写入；参考项目直接返回成功提示。
        if not _persistent_memory.save_observation(text):
            return "Error: text is required for save"
        return f"Saved to memory: {text[:80]}"
    elif action == "save_user":
        if not text:
            return "Error: text is required for save_user"
        # 本地返回是否截断，False 不是写入失败。
        truncated = _persistent_memory.update_user_profile(text)
        result = f"User profile updated ({len(text)} chars)"
        if truncated:
            result += f"; truncated to {_persistent_memory.USER_LIMIT} chars"
        return result
    elif action == "read":
        mem = _persistent_memory.read_memory() or "(empty)"
        user = _persistent_memory.read_user_profile() or "(empty)"
        return f"## Memory\n{mem}\n\n## User Profile\n{user}"
    elif action == "search":
        if not text:
            return "Error: text (query) is required for search"
        if _session_recall is None:
            return "Error: session recall not available"
        result = _session_recall.recall(text)
        return result if result else "No relevant past sessions found."
    else:
        return f"Error: unknown action '{action}'. Use: save, save_user, read, search"


registry.register(
    name="memory",
    description=(
        "Manage persistent memory across sessions. Actions:\n"
        "- save: Save an observation about the user or project\n"
        "- save_user: Update the user profile\n"
        "- read: Read current memory and user profile\n"
        "- search: Search past conversations for relevant context"
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["save", "save_user", "read", "search"],
                "description": "Action to perform",
            },
            "text": {
                "type": "string",
                "description": "Text to save, or query to search for",
            },
        },
        "required": ["action"],
    },
    handler=memory,
    category="memory",
)