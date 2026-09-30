"""脚本执行边界测试：python3 tests/test_script_tool.py。"""

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skills import manager
from skills.loader import SkillLoader
from tool_registry import registry
from tools import script_tool


@unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
class ScriptToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.skill = self.root / "demo"
        (self.skill / "scripts").mkdir(parents=True)
        (self.skill / "SKILL.md").write_text("---\nname: demo\ndescription: test\n---\nTest")
        self.old = manager._skill_loader, manager._skills_dir
        self.addCleanup(manager.set_skill_loader, *self.old)
        manager.set_skill_loader(SkillLoader(self.root), self.root)
        self.script = self.skill / "scripts" / "run.py"
        self.script.write_text("print('ok')")

    def run_script(self, **kwargs):
        return json.loads(script_tool.skill_run("demo", "scripts/run.py", **kwargs))

    def test_success_args_cwd_and_stdin(self):
        self.script.write_text(
            "import sys,json,os\n"
            "print(json.dumps([sys.argv[1:],os.getcwd(),sys.stdin.read()]))\n"
            "print('warning',file=sys.stderr)\n"
        )
        args = ["中文 空格", "; echo injected", "$(echo nope)", "--flag"]
        result = self.run_script(args=args)
        self.assertTrue(result["success"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(json.loads(result["stdout"]), [args, str(self.skill.resolve()), ""])
        self.assertEqual(result["stderr"], "warning\n")

    def test_nonzero_exit(self):
        self.script.write_text("import sys\nprint('failed',file=sys.stderr)\nsys.exit(7)")
        result = self.run_script()
        self.assertFalse(result["success"])
        self.assertEqual(result["exit_code"], 7)
        self.assertFalse(result["timed_out"])

    def test_timeout(self):
        self.script.write_text("import time\nprint('started',flush=True)\ntime.sleep(60)")
        start = time.monotonic()
        result = self.run_script(timeout=1)
        self.assertTrue(result["timed_out"])
        self.assertIn("started", result["stdout"])
        self.assertLess(time.monotonic() - start, 5)

    def test_child_holding_pipe_does_not_hang(self):
        self.script.write_text(
            "import subprocess,sys\n"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        )
        start = time.monotonic()
        self.assertTrue(self.run_script(timeout=1)["timed_out"])
        self.assertLess(time.monotonic() - start, 5)

    def test_output_limit_and_invalid_utf8(self):
        self.script.write_text("import os\nos.write(1,b'\\xff')\nprint('x'*100000)")
        result = self.run_script()
        self.assertTrue(result["truncated"])
        self.assertFalse(result["success"])
        self.assertLessEqual(len(result["stdout"]), script_tool.OUTPUT_LIMIT)
        self.assertIn("\ufffd", result["stdout"])

    def test_invalid_paths_and_arguments_never_spawn(self):
        with patch.object(script_tool.subprocess, "Popen") as spawn:
            for path in ("/tmp/run.py", "scripts/../run.py", "run.py", "scripts/a.sh",
                         "scripts/missing.py", "scripts", "", None, "scripts/\x00.py"):
                with self.subTest(path=path):
                    self.assertFalse(json.loads(script_tool.skill_run("demo", path))["success"])
            for kwargs in ({"timeout": 0}, {"timeout": 121}, {"timeout": True},
                           {"args": "hello"}, {"args": [1]}, {"args": ["\x00"]},
                           {"args": ["\ud800"]}, {"args": ["x"] * 101},
                           {"args": ["x" * 32769]}):
                self.assertFalse(self.run_script(**kwargs)["success"])
            spawn.assert_not_called()

    def test_symlink_rejected(self):
        (self.skill / "scripts/link.py").symlink_to(self.script)
        result = json.loads(script_tool.skill_run("demo", "scripts/link.py"))
        self.assertFalse(result["success"])
        (self.skill / "scripts/nested").symlink_to(self.skill / "scripts", target_is_directory=True)
        self.assertFalse(json.loads(script_tool.skill_run("demo", "scripts/nested/run.py"))["success"])

    def test_missing_initialization_and_skill(self):
        self.assertFalse(json.loads(script_tool.skill_run("missing", "scripts/run.py"))["success"])
        manager.set_skill_loader(None, None)
        self.assertFalse(self.run_script()["success"])

    def test_spawn_failure_does_not_leak_exception(self):
        with patch.object(script_tool.subprocess, "Popen", side_effect=OSError("secret")):
            result = self.run_script()
        self.assertFalse(result["success"])
        self.assertNotIn("secret", result["error"])

    def test_registry_category_and_dispatch(self):
        self.assertIn("skill_run", registry.get_handlers(["execution"]))
        self.assertNotIn("skill_run", registry.get_handlers(["skills"]))
        result = json.loads(registry.execute("skill_run", {
            "name": "demo", "file_path": "scripts/run.py",
        }))
        self.assertTrue(result["success"])
        schemas = registry.get_schemas(["execution"])
        schema = next(s["function"] for s in schemas if s["function"]["name"] == "skill_run")
        self.assertEqual(schema["parameters"]["required"], ["name", "file_path"])


if __name__ == "__main__":
    unittest.main()