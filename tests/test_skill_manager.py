"""技能管理器离线测试：python3 tests/test_skill_manager.py。"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_builder import PromptBuilder
from skills import manager
from skills.loader import SkillLoader
from tool_registry import registry


class SkillManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "skills"
        self.loader = SkillLoader(self.root)
        manager.set_skill_loader(self.loader, str(self.root))
        self.addCleanup(manager.set_skill_loader, None, None)

    def create(self, name="demo", body="原文", **kwargs):
        content = f"---\nname: {name}\ndescription: 示例\n---\n{body}"
        self.success(manager.skill_manage("create", name, content=content, **kwargs))
        return content

    def success(self, result):
        payload = json.loads(result)
        self.assertTrue(payload["success"])
        self.assertTrue(payload["message"])

    def error(self, result):
        self.assertTrue(result.startswith("Error:"), result)

    def test_uninitialized_and_reinjection(self):
        manager.set_skill_loader(None, None)
        for result in (manager.skills_list(), manager.skill_view("demo"),
                       manager.skill_manage("delete", "demo")):
            self.assertEqual(result, "Error: skills not initialized")
        manager.set_skill_loader(self.loader, self.root)
        self.create()
        other = self.root / "other"
        manager.set_skill_loader(SkillLoader(other), other)
        self.assertEqual(manager.skills_list(), "No skills found.")
        manager.set_skill_loader(self.loader, self.root)
        self.assertIn("demo", manager.skills_list())

    def test_import_registers_schemas_and_handlers(self):
        code = "\n".join([
            "from tool_registry import registry",
            "assert registry.get_schemas(['skills']) == []",
            "import skills.manager as manager",
            "schemas = {s['function']['name']: s['function'] for s in registry.get_schemas(['skills'])}",
            "assert set(schemas) == {'skills_list', 'skill_view', 'skill_manage'}",
            "assert schemas['skill_view']['parameters']['required'] == ['name']",
            "assert schemas['skill_manage']['parameters']['required'] == ['action', 'name']",
            "assert not schemas['skills_list']['parameters'].get('required')",
            "props = schemas['skill_manage']['parameters']['properties']",
            "assert props['action']['enum'] == ['create', 'patch', 'edit', 'delete', 'write_file', 'remove_file']",
            "assert props['replace_all']['type'] == 'boolean'",
            "assert set(props) == {'action', 'name', 'content', 'category', 'old_string', 'new_string', 'replace_all', 'file_path', 'file_content'}",
            "for name, handler in registry.get_handlers(['skills']).items():",
            "    assert handler is getattr(manager, name)",
            "assert registry.execute('skills_list', {}) == 'Error: skills not initialized'",
        ])
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)

    def test_list_metadata_category_and_full_view(self):
        self.assertEqual(manager.skills_list(), "No skills found.")
        text = self.create(category="coding/python")
        self.create("other", body="不应显示在列表")
        self.assertEqual(json.loads(manager.skills_list("coding")), [
            {"name": "demo", "description": "示例", "version": "1.0.0"},
        ])
        self.assertEqual(manager.skills_list("missing"), "No skills found.")
        self.assertNotIn("不应显示在列表", manager.skills_list())
        self.assertEqual(manager.skill_view("demo"), text)
        updated = text.replace("示例", "字" * 201).replace("---\n原文", "version: 2.0.0\n---\n原文")
        self.success(manager.skill_manage("edit", "demo", content=updated))
        item = json.loads(manager.skills_list("coding"))[0]
        self.assertEqual(item["description"], "字" * 200)
        self.assertEqual(item["version"], "2.0.0")

    def test_create_validation_and_duplicate(self):
        for content in (None, "", "no frontmatter"):
            self.error(manager.skill_manage("create", "demo", content=content))
        original = self.create()
        self.error(manager.skill_manage("create", "demo", content=original))
        self.assertEqual(manager.skill_view("demo"), original)

    def test_patch_single_multiple_empty_and_missing(self):
        self.create(body="repeat repeat\nunique")
        self.error(manager.skill_manage("patch", "demo", old_string="repeat", new_string="x"))
        self.assertIn("repeat repeat", manager.skill_view("demo"))
        self.success(manager.skill_manage("patch", "demo", old_string="repeat", new_string="x", replace_all=True))
        self.assertIn("x x", manager.skill_view("demo"))
        self.success(manager.skill_manage("patch", "demo", old_string="unique", new_string=""))
        self.assertNotIn("unique", manager.skill_view("demo"))
        self.error(manager.skill_manage("patch", "demo", old_string="absent", new_string="x"))
        self.error(manager.skill_manage("patch", "demo", old_string="", new_string="x"))
        self.error(manager.skill_manage("patch", "demo", old_string="x"))

    def test_edit_and_delete_whole_skill(self):
        self.create(category="coding")
        self.success(manager.skill_manage("write_file", "demo", file_path="assets/data.txt", file_content="data"))
        self.error(manager.skill_manage("edit", "demo", content=""))
        updated = "---\nname: demo\n---\n完全重写"
        self.success(manager.skill_manage("edit", "demo", content=updated))
        self.assertEqual(manager.skill_view("demo"), updated)
        self.success(registry.execute("skill_manage", {"action": "delete", "name": "demo"}))
        self.assertFalse((self.root / "coding" / "demo").exists())
        self.assertTrue((self.root / "coding").is_dir())
        self.assertEqual(self.loader.load_all(), [])

    def test_supporting_file_lifecycle_and_allowed_directories(self):
        self.create()
        for folder in ("references", "templates", "scripts", "assets"):
            path = f"{folder}/nested/file.txt"
            self.success(manager.skill_manage("write_file", "demo", file_path=path, file_content="hello"))
            self.assertEqual(manager.skill_view("demo", path), "hello")
            self.success(manager.skill_manage("patch", "demo", file_path=path, old_string="hello", new_string="中文"))
            self.assertEqual(manager.skill_view("demo", path), "中文")
            self.success(manager.skill_manage("write_file", "demo", file_path=path, file_content=""))
            self.assertEqual(manager.skill_view("demo", path), "")
            self.success(manager.skill_manage("remove_file", "demo", file_path=path))
            self.error(manager.skill_view("demo", path))
        for path in ("SKILL.md", "other/file.txt", "references"):
            self.error(manager.skill_manage("write_file", "demo", file_path=path, file_content="x"))

    def test_missing_skill_files_arguments_and_unknown_action(self):
        self.error(manager.skill_view("missing"))
        self.error(manager.skill_manage("delete", "missing"))
        self.create()
        self.error(manager.skill_manage("unknown", "demo"))
        self.error(manager.skill_manage("write_file", "demo", file_path="assets/x"))
        self.error(manager.skill_manage("write_file", "demo", file_content="x"))
        self.error(manager.skill_manage("remove_file", "demo"))
        self.error(manager.skill_manage("remove_file", "demo", file_path="missing"))
        self.error(manager.skill_manage("patch", "demo", file_path="missing", old_string="x", new_string="y"))
        self.error(manager.skill_view("demo", "assets"))

    def test_absolute_parent_and_invalid_name_paths(self):
        self.create()
        text = "---\nname: bad\n---\nbody"
        for name in ("", ".", "..", "../bad", "a/b", "a\\b", "/tmp/bad"):
            self.error(manager.skill_manage("create", name, content=text))
        for category in ("../escape", str(Path(self.temp.name) / "outside")):
            self.error(manager.skill_manage("create", "bad", content=text, category=category))
        for path in ("../SKILL.md", "assets/../../escape", str(self.root / "demo" / "SKILL.md"), "."):
            self.error(manager.skill_view("demo", path))
            for action, kwargs in (("write_file", {"file_content": "x"}),
                                   ("patch", {"old_string": "原文", "new_string": "x"}),
                                   ("remove_file", {})):
                self.error(manager.skill_manage(action, "demo", file_path=path, **kwargs))

    def test_external_symlink_paths_are_rejected(self):
        self.create()
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        victim = outside / "data.txt"
        victim.write_text("untouched", encoding="utf-8")
        (self.root / "demo" / "assets").symlink_to(outside, target_is_directory=True)
        path = "assets/data.txt"
        self.error(manager.skill_view("demo", path))
        for action, kwargs in (("write_file", {"file_content": "x"}),
                               ("patch", {"old_string": "untouched", "new_string": "x"}),
                               ("remove_file", {})):
            self.error(manager.skill_manage(action, "demo", file_path=path, **kwargs))
        (self.root / "linked").symlink_to(outside, target_is_directory=True)
        self.error(manager.skill_manage("create", "bad", content="---\nname: bad\n---", category="linked"))
        self.error(manager.skill_manage("create", "linked", content="---\nname: linked\n---"))
        self.success(manager.skill_manage("delete", "demo"))
        self.assertEqual(victim.read_text(encoding="utf-8"), "untouched")

    def test_remove_internal_symlink_preserves_target(self):
        self.create()
        assets = self.root / "demo" / "assets"
        assets.mkdir()
        target = assets / "target.txt"
        target.write_text("keep", encoding="utf-8")
        link = assets / "link.txt"
        link.symlink_to(target)
        self.success(manager.skill_manage("remove_file", "demo", file_path="assets/link.txt"))
        self.assertFalse(link.is_symlink())
        self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_write_symlink_cannot_overwrite_main_file(self):
        original = self.create()
        assets = self.root / "demo" / "assets"
        assets.mkdir()
        (assets / "alias.txt").symlink_to(self.root / "demo" / "SKILL.md")
        self.error(manager.skill_manage("write_file", "demo", file_path="assets/alias.txt", file_content="bad"))
        self.assertEqual(manager.skill_view("demo"), original)

    def test_changes_do_not_invalidate_frozen_prompt(self):
        builder = PromptBuilder()
        first = builder.get_system_prompt(skills_index=self.loader.build_skills_index())
        self.create()
        index = self.loader.build_skills_index()
        self.assertEqual(builder.get_system_prompt(skills_index=index), first)
        builder.invalidate()
        self.assertIn("- **demo**: 示例", builder.get_system_prompt(skills_index=index))


if __name__ == "__main__":
    unittest.main(verbosity=2)