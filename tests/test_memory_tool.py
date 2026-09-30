"""参考记忆工具契约及本地适配测试：python3 tests/test_memory_tool.py。"""

import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.persistent import PersistentMemory
from memory.recall import SessionRecall
from memory.session_db import SessionDB
from prompt_builder import PromptBuilder
from tool_registry import registry
from tools.memory_tool import memory, set_memory


def test_import_registers_without_initialization():
    # 独立进程验证首次导入，无需初始化存储，也不污染其他测试的注册表。
    code = (
        "from tool_registry import registry; "
        "assert 'memory' not in registry.get_handlers(); "
        "from tools.memory_tool import memory; "
        "assert registry.get_handlers(['memory'])['memory'] is memory; "
        "assert memory('read') == 'Error: memory not initialized'"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    schema = next(
        item["function"] for item in registry.get_schemas(["memory"])
        if item["function"]["name"] == "memory"
    )
    assert schema["parameters"] == {
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
    }


def test_uninitialized_and_reinitialization():
    set_memory(None)
    for action in ("save", "save_user", "read", "search", "unknown"):
        assert memory(action, "text") == "Error: memory not initialized"
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        set_memory(PersistentMemory(first), SimpleNamespace(recall=lambda query: "old"))
        assert memory("search", "query") == "old"
        memory("save", "旧依赖的观察")
        set_memory(PersistentMemory(second))
        assert memory("read") == "## Memory\n(empty)\n\n## User Profile\n(empty)"
        assert memory("search", "query") == "Error: session recall not available"


def test_save_read_and_profile_replacement():
    with tempfile.TemporaryDirectory() as tmp:
        persistent = PersistentMemory(tmp)
        set_memory(persistent)
        assert registry.execute("memory", {"action": "save", "text": "喜欢中文"}) == (
            "Saved to memory: 喜欢中文"
        )
        assert memory("save_user", "使用 macOS") == "User profile updated (8 chars)"
        assert memory("save_user", "使用 Linux") == "User profile updated (8 chars)"
        assert memory("read") == "## Memory\n- 喜欢中文\n\n## User Profile\n使用 Linux"
        assert PersistentMemory(tmp).read_user_profile() == "使用 Linux"
        text = "字" * 100
        assert memory("save", text) == f"Saved to memory: {text[:80]}"


def test_required_text_and_exact_action_names():
    with tempfile.TemporaryDirectory() as tmp:
        persistent = PersistentMemory(tmp)
        set_memory(persistent)
        memory("save_user", "保留画像")
        before = persistent.load()
        for action in ("save", "save_user"):
            assert memory(action) == f"Error: text is required for {action}"
            assert memory(action, "") == f"Error: text is required for {action}"
        assert memory("search") == "Error: text (query) is required for search"
        for action in ("forget", " Save ", "SAVE"):
            assert memory(action, "value") == (
                f"Error: unknown action '{action}'. Use: save, save_user, read, search"
            )
        assert memory("save", " \n ") == "Error: text is required for save"
        assert persistent.load() == before


def test_local_profile_truncation_adaptation():
    with tempfile.TemporaryDirectory() as tmp:
        persistent = PersistentMemory(tmp)
        set_memory(persistent)
        assert "truncated" not in memory("save_user", "中文  \r\n偏好  \r\n")
        assert persistent.read_user_profile() == "中文\n偏好"
        assert "truncated" not in memory("save_user", "字" * persistent.USER_LIMIT)
        result = memory("save_user", "字" * (persistent.USER_LIMIT + 1))
        assert result.endswith(f"truncated to {persistent.USER_LIMIT} chars")
        assert len(persistent.read_user_profile()) == persistent.USER_LIMIT


def test_search_results_and_empty_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(":memory:", tokenize="trigram")
        try:
            past = db.create_session(source="cli")
            db.append_message(past, "user", "历史会话选择 psycopg2")
            db.end_session(past)
            current = db.create_session(source="cli")
            db.append_message(current, "user", "当前会话选择 psycopg2")
            recall = SessionRecall(db, current_session_id=current)
            set_memory(PersistentMemory(tmp), recall)
            out = memory("search", "psycopg2")
            assert out == recall.recall("psycopg2")
            assert "历史会话选择" in out and past[:8] in out
            assert "当前会话选择" not in out
            assert memory("search", "不存在的主题") == "No relevant past sessions found."
        finally:
            db.close()


def test_dependency_errors_propagate_to_caller():
    def fail(*args):
        raise OSError("storage unavailable")

    set_memory(SimpleNamespace(
        save_observation=fail, update_user_profile=fail, read_memory=fail,
    ), SimpleNamespace(recall=fail))
    for action in ("save", "save_user", "read", "search"):
        try:
            memory(action, "value")
        except OSError as exc:
            assert str(exc) == "storage unavailable"
        else:
            raise AssertionError(f"{action} should propagate dependency errors")


def test_write_preserves_frozen_prompt_until_rebuild():
    with tempfile.TemporaryDirectory() as tmp:
        persistent = PersistentMemory(tmp)
        set_memory(persistent)
        builder = PromptBuilder()
        frozen = builder.get_system_prompt(memory_block=persistent.load())
        memory("save", "特殊偏好：使用中文")
        assert "特殊偏好" in memory("read")
        assert builder.get_system_prompt(memory_block=persistent.load()) == frozen
        assert "特殊偏好" not in frozen
        builder.invalidate()
        assert "特殊偏好" in builder.get_system_prompt(memory_block=persistent.load())


if __name__ == "__main__":
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                set_memory(None)
                try:
                    fn()
                finally:
                    set_memory(None)
                print(f"PASS  {name}")
    finally:
        set_memory(None)
    print("全部通过")