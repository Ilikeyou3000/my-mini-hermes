"""第 8 章:持久记忆 USER.md / MEMORY.md。

重点不在"能写进去",而在三件事:
  1. 上限真的挡得住 —— 提示词每轮都要重发,超限等于持续烧钱
  2. 淘汰顺序是最旧优先 —— 旧记忆比新记忆更可能过时
  3. 落盘之后系统提示词**不变** —— 冻结快照是 prompt cache 的前提

环境没装 pytest,直接 `python3 tests/test_persistent_memory.py`。
"""

import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memory import PersistentMemory, SessionDB  # noqa: E402
from prompt_builder import PromptBuilder  # noqa: E402


def _fresh() -> PersistentMemory:
    """每个用例一个独立临时目录,避免相互污染。"""
    tmp = tempfile.mkdtemp(prefix="mini-hermes-mem-")
    return PersistentMemory(tmp)


# ---------------------------------------------------------------------- #
# 基础:空、创建、顺序
# ---------------------------------------------------------------------- #
def test_empty_memory_loads_empty_string():
    mem = _fresh()
    assert mem.load() == "", "空记忆不该产出任何文本"
    assert mem.read_memory() == ""
    assert mem.read_user_profile() == ""
    assert mem.observation_count() == 0


def test_files_are_created_on_init():
    mem = _fresh()
    assert mem.memory_path.exists() and mem.user_path.exists()
    # 目录不存在时应自动建出来(首次运行时 data/ 可能还不存在)
    assert mem.memory_path.name == "MEMORY.md"
    assert mem.user_path.name == "USER.md"


def test_load_puts_profile_before_observations():
    mem = _fresh()
    mem.save_observation("观察一")
    mem.update_user_profile("- 偏好中文")

    block = mem.load()
    assert block.index(PersistentMemory.USER_HEADING) < block.index(
        PersistentMemory.MEMORY_HEADING
    ), "画像应排在观察之前"
    assert "- 偏好中文" in block and "观察一" in block


def test_load_omits_empty_sections():
    mem = _fresh()
    mem.save_observation("只有观察,没有画像")
    block = mem.load()
    assert PersistentMemory.MEMORY_HEADING in block
    assert PersistentMemory.USER_HEADING not in block, "空画像不该留空壳标题"


# ---------------------------------------------------------------------- #
# 写入与归一化
# ---------------------------------------------------------------------- #
def test_save_observation_appends_prefixed_line():
    mem = _fresh()
    assert mem.save_observation("第一条") is True
    assert mem.save_observation("第二条") is True

    raw = mem.read_memory()
    assert raw.splitlines() == ["- 第一条", "- 第二条"]
    assert mem.observation_count() == 2


def test_empty_observation_is_rejected():
    mem = _fresh()
    for bad in ("", "   ", "\n\n", "\t \n"):
        assert mem.save_observation(bad) is False, f"{bad!r} 不该被写入"
    assert mem.observation_count() == 0


def test_multiline_observation_collapses_to_one_line():
    mem = _fresh()
    mem.save_observation("第一行\n第二行\t带制表符\n\n第三行  多余空格")

    lines = mem.read_memory().splitlines()
    assert len(lines) == 1, "多行输入必须压成单条,否则淘汰会留下残句"
    assert lines[0] == "- 第一行 第二行 带制表符 第三行 多余空格"


def test_update_user_profile_normalizes_block():
    mem = _fresh()
    mem.update_user_profile("\n\n- 第一行   \n- 第二行\n\n\n")
    assert mem.read_user_profile() == "- 第一行\n- 第二行"


def test_update_user_profile_replaces_not_appends():
    mem = _fresh()
    mem.update_user_profile("旧画像")
    mem.update_user_profile("新画像")
    assert mem.read_user_profile() == "新画像", "画像是当前结论,应整体替换"


def test_clear_empties_both_files():
    mem = _fresh()
    mem.save_observation("将被清掉")
    mem.update_user_profile("也将被清掉")
    mem.clear()
    assert mem.load() == ""


# ---------------------------------------------------------------------- #
# 上限与淘汰
# ---------------------------------------------------------------------- #
def test_oldest_observation_evicted_first():
    mem = _fresh()
    mem.save_observation("我是最老的一条")

    for i in range(200):
        mem.save_observation(f"填充条目 {i} 用来把文件撑到上限之外")

    raw = mem.read_memory()
    assert len(raw.rstrip("\n")) <= PersistentMemory.MEMORY_LIMIT, "总长必须受上限约束"
    assert "我是最老的一条" not in raw, "淘汰应从最旧开始"
    assert "填充条目 199" in raw, "最新一条必须保留"
    # 最新几条都该在,说明不是简单截断尾巴
    assert "填充条目 198" in raw


def test_oversized_single_observation_is_truncated():
    mem = _fresh()
    mem.save_observation("前一条正常记忆")
    mem.save_observation("超" * (PersistentMemory.MEMORY_LIMIT * 2))

    raw = mem.read_memory()
    assert len(raw.rstrip("\n")) <= PersistentMemory.MEMORY_LIMIT
    assert PersistentMemory.TRUNCATION_MARK in raw, "单条超限应留下截断标记"
    assert mem.observation_count() >= 1


def test_user_profile_truncated_to_limit():
    mem = _fresh()
    mem.update_user_profile("画" * (PersistentMemory.USER_LIMIT * 3))
    assert len(mem.read_user_profile()) <= PersistentMemory.USER_LIMIT


def test_repeated_saves_stay_under_limit():
    mem = _fresh()
    for i in range(300):
        mem.save_observation(f"第 {i} 条观察,内容长度大致均匀以便观察淘汰行为")
    assert len(mem.read_memory().rstrip("\n")) <= PersistentMemory.MEMORY_LIMIT


# ---------------------------------------------------------------------- #
# 落盘、并发、与 PromptBuilder 的配合
# ---------------------------------------------------------------------- #
def test_new_instance_reads_persisted_content():
    tmp = tempfile.mkdtemp(prefix="mini-hermes-mem-")
    first = PersistentMemory(tmp)
    first.save_observation("跨实例可见")
    first.update_user_profile("- 中文")

    second = PersistentMemory(tmp)
    assert "跨实例可见" in second.load()
    assert "- 中文" in second.load()


def test_concurrent_writes_do_not_corrupt():
    mem = _fresh()
    errors: list[BaseException] = []

    def worker(tag: int) -> None:
        try:
            for i in range(20):
                mem.save_observation(f"线程 {tag} 的第 {i} 条")
        except BaseException as exc:  # noqa: BLE001 - 测试要如实反映异常
            errors.append(exc)

    # ch13 的后台复盘线程会这样并发写,读到写了一半的文件就全坏了
    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发写抛错: {errors!r}"
    raw = mem.read_memory()
    assert len(raw.rstrip("\n")) <= PersistentMemory.MEMORY_LIMIT
    for line in raw.splitlines():
        assert line.startswith("- "), f"落盘内容被写坏: {line!r}"
    assert mem.observation_count() > 0


def test_memory_does_not_change_frozen_prompt():
    """这是第 8 章最关键的一条:写记忆 ≠ 改提示词。"""
    mem = _fresh()
    mem.update_user_profile("- 偏好中文")
    builder = PromptBuilder()
    first = builder.get_system_prompt(memory_block=mem.load())
    assert builder.is_frozen is True

    # session 中途学到新东西
    mem.save_observation("又学到一条,应落盘但不动本轮提示词")
    second = builder.get_system_prompt(memory_block=mem.load())

    assert second == first, "冻结快照在 session 中途不该被改写"
    assert builder.build_count == 1
    assert "又学到一条" in mem.load(), "内容必须已经落盘,供下次 session 使用"

    builder.invalidate()
    third = builder.get_system_prompt(memory_block=mem.load())
    assert "又学到一条" in third, "invalidate 之后新记忆才进入提示词"


def test_memory_block_shows_up_in_prompt_section():
    mem = _fresh()
    mem.update_user_profile("- 用户: 正在学 mini-hermes")
    builder = PromptBuilder()
    prompt = builder.get_system_prompt(memory_block=mem.load())
    assert builder.MEMORY_HEADING in prompt
    assert "- 用户: 正在学 mini-hermes" in prompt


def test_persistent_memory_coexists_with_session_db():
    """三层记忆是并列的:持久记忆在文件里,会话记忆在 SQLite 里,互不干扰。"""
    mem = _fresh()
    db = SessionDB(":memory:")
    try:
        sid = db.create_session("测试会话")
        db.append_message(sid, "user", "你好")
        mem.save_observation("用户打招呼时习惯用中文")

        assert db.search("你好"), "会话记忆照常工作"
        assert "习惯用中文" in mem.load()
    finally:
        db.close()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("全部通过")