"""技能加载器离线测试：python3 tests/test_skill_loader.py。"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_builder import PromptBuilder
from skills.loader import SkillLoader


def write_skill(root: Path, folder: str, text: str) -> Path:
    path = root / folder / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_empty_directory_is_created():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data" / "skills"
        loader = SkillLoader(root)
        assert root.is_dir()
        assert loader.load_all() == []
        assert loader.build_skills_index() == ""
        assert loader.find_skill("missing") is None
        assert SkillLoader(root).load_all() == []


def test_recursive_discovery_fields_and_stable_order():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_skill(root, "z/nested", "---\nname: nested\n---\n 第二份正文 \n")
        first = write_skill(root, "a", (
            "---\nname: review\ndescription: 中文审查\nversion: 2.0.0\n---\n"
            "\n# 操作步骤\n\n保留正文中的分隔线\n---\n完成\n"
        ))
        (root / "OTHER.md").write_text("---\nname: ignored\n---", encoding="utf-8")
        skills = SkillLoader(root).load_all()
        assert skills == [
            {
                "name": "review", "description": "中文审查", "version": "2.0.0",
                "path": first.parent,
                "body": "# 操作步骤\n\n保留正文中的分隔线\n---\n完成",
            },
            {
                "name": "nested", "description": "", "version": "1.0.0",
                "path": root / "z" / "nested", "body": "第二份正文",
            },
        ]
        assert SkillLoader(root).load_all() == skills


def test_yaml_multiline_and_empty_body():
    with tempfile.TemporaryDirectory() as tmp:
        loader = SkillLoader(Path(tmp))
        meta, body = loader._parse_frontmatter(
            '---\nname: "quoted-name"\ndescription: >-\n  first line\n  second line\n---\n'
        )
        assert meta == {"name": "quoted-name", "description": "first line second line"}
        assert body == ""
        assert loader._parse_frontmatter("---\nname: empty\n---")[1] == ""


def test_invalid_frontmatter_is_ignored_without_losing_text():
    invalid = [
        "正文没有元数据", "---\nname: incomplete", "---\n---\nbody",
        "---\n- list\n---\nbody", "---\nscalar\n---\nbody",
        "---\nname: [broken\n---\nbody",
        "---\n!!python/object/apply:builtins.str [unsafe]\n---\nbody",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        loader = SkillLoader(root)
        for index, text in enumerate(invalid):
            assert loader._parse_frontmatter(text) == ({}, text)
            write_skill(root, str(index), text)
        write_skill(root, "no-name", "---\ndescription: missing name\n---\nbody")
        write_skill(root, "empty-name", "---\nname: ''\n---\nbody")
        write_skill(root, "valid", "---\nname: valid\n---\nbody")
        assert [skill["name"] for skill in loader.load_all()] == ["valid"]


def test_index_contains_only_metadata_and_truncates_description():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_skill(root, "long", (
            "---\nname: long\ndescription: " + "字" * 201 + "\n---\nPRIVATE BODY"
        ))
        write_skill(root, "short", "---\nname: short\n---\nOTHER BODY")
        loader = SkillLoader(root)
        index = loader.build_skills_index()
        assert index == (
            "Available skills (use skill_view to load full content):\n"
            "- **long**: " + "字" * 200 + "\n- **short**: "
        )
        assert loader.find_skill("long")["description"] == "字" * 201
        assert "BODY" not in index
        assert str(root) not in index


def test_find_skill_is_exact_and_reloads_changes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        loader = SkillLoader(root)
        assert loader.find_skill("review") is None
        write_skill(root, "review", "---\nname: review\n---\nold")
        assert loader.find_skill("review")["body"] == "old"
        assert loader.find_skill("REVIEW") is None
        assert loader.find_skill(" review ") is None
        write_skill(root, "review", "---\nname: review\n---\nnew")
        assert loader.find_skill("review")["body"] == "new"


def test_index_preserves_prompt_snapshot_until_invalidation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        loader = SkillLoader(root)
        builder = PromptBuilder()
        first = builder.get_system_prompt(skills_index=loader.build_skills_index())
        write_skill(root, "review", "---\nname: review\ndescription: 审查\n---\n正文")
        index = loader.build_skills_index()
        assert builder.get_system_prompt(skills_index=index) == first
        builder.invalidate()
        rebuilt = builder.get_system_prompt(skills_index=index)
        assert "- **review**: 审查" in rebuilt
        assert "正文" not in rebuilt
        assert rebuilt != first


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\n全部通过")