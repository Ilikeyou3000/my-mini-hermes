"""CLI 离线集成测试：python3 tests/test_cli.py。"""

import io
import sqlite3
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
for path in (str(ROOT), str(ROOT / "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)

import cli
from agent import Agent
from fake_llm import FakeClient, structured_msg
from memory.session_db import SessionDB
from tool_calling import TextStrategy


class CLITests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        (self.root / "config.yaml").write_text(
            "model:\n  api_key: test\n  base_url: http://localhost:1234/v1\n"
            "  model: gpt-4o-mini\nlearning:\n"
            "  memory_nudge_interval: 0\n  skill_nudge_interval: 0\n",
            encoding="utf-8",
        )
        self.client = FakeClient([structured_msg(content="hello back")])
        self.client.models = SimpleNamespace(list=Mock(return_value=SimpleNamespace(
            data=[SimpleNamespace(id="gpt-4o-mini"), SimpleNamespace(id="gemma-3")],
        )))
        self.menu = SimpleNamespace(pick=Mock(return_value=("gemma-3", 1)))
        self.stack.enter_context(patch.dict(sys.modules, {"pick": self.menu}))
        self.stack.enter_context(patch.object(cli, "__file__", str(self.root / "cli.py")))
        self.openai = self.stack.enter_context(patch.object(cli, "OpenAI", return_value=self.client))
        self.stack.enter_context(patch.object(cli, "Agent", side_effect=self.make_agent))
        # 恢复工具模块的全局注入，避免测试间共享已经关闭的数据库。
        for module, names in (
            (cli.tools.memory_tool, ("_persistent_memory", "_session_recall")),
            (cli.skills.manager, ("_skill_loader", "_skills_dir")),
        ):
            for name in names:
                self.stack.enter_context(patch.object(module, name, None))
        self.output = io.StringIO()

    def make_agent(self, **kwargs):
        self.agent = Agent(**kwargs)
        return self.agent

    def run_cli(self, inputs):
        with patch("builtins.input", side_effect=inputs), redirect_stdout(self.output):
            cli.main()
        return self.output.getvalue()

    def saved_rows(self):
        db = SessionDB(self.root / "data" / "state.db")
        try:
            session = db.get_session(self.agent.session_id)
            self.assertIsNotNone(session["ended_at"])
            return db.get_session_messages(self.agent.session_id)
        finally:
            db.close()

    def test_wiring_and_persistence(self):
        output = self.run_cli(["", "hello", "exit"])
        self.assertIn("hello back", output)
        self.assertIn("saved.", output)
        self.openai.assert_called_once_with(api_key="test", base_url="http://localhost:1234/v1")
        self.assertEqual(self.agent.max_tokens, 400)
        self.assertEqual(self.agent.max_iterations, 15)
        self.assertEqual(self.agent._memory_nudge_interval, 0)
        self.assertEqual(self.agent._skill_nudge_interval, 0)
        self.assertEqual(self.agent._compressor.max_context_tokens, 32000)
        self.assertEqual(self.agent._compressor.max_tokens, 400)
        self.assertTrue({"memory", "skills_list", "skill_view", "skill_manage"}.issubset(
            self.agent.tool_handlers,
        ))
        recall = cli.tools.memory_tool._session_recall
        self.assertEqual(recall.current_session_id, self.agent.session_id)
        self.assertEqual(recall.max_tokens, 300)
        self.assertEqual([row["role"] for row in self.saved_rows()], ["system", "user", "assistant"])
        with self.assertRaises(sqlite3.ProgrammingError):
            self.agent.session_db.conn.execute("SELECT 1")

    def test_memory_and_skills_commands(self):
        persistent = cli.PersistentMemory(self.root / "data")
        persistent.save_observation("prefers short answers")
        persistent.update_user_profile("Python developer")
        output = self.run_cli(["/mem", "/skills", "quit"])
        self.assertIn("prefers short answers", output)
        self.assertIn("Python developer", output)
        self.assertIn("No skills yet.", output)
        self.assertIn("prefers short answers", self.agent.system_prompt)
        self.assertEqual(len(self.saved_rows()), 1)
        self.assertEqual(self.client.llm_call_count, 0)

    def test_skill_listing(self):
        skill = {"name": "demo", "description": "example", "body": "private body"}
        with patch.object(cli.SkillLoader, "load_all", return_value=[skill]):
            output = self.run_cli(["/skills", "exit"])
        self.assertIn("demo: example", output)
        self.assertIn("demo", self.agent.system_prompt)
        self.assertNotIn("private body", self.agent.system_prompt)

    def test_model_switch_updates_strategy(self):
        output = self.run_cli(["/model", "hello", "/quit"])
        self.assertIn("Switched to", output)
        self.assertEqual(self.agent.model, "gemma-3")
        self.assertIsInstance(self.agent._strategy, TextStrategy)
        self.assertEqual(self.client.calls[0]["model"], "gemma-3")
        self.assertEqual(self.menu.pick.call_args.kwargs["default_index"], 0)
        self.assertIn("active", self.menu.pick.call_args.args[0][0])

    def test_model_list_error_and_empty_list(self):
        self.client.models.list.side_effect = [RuntimeError("offline"), SimpleNamespace(data=[])]
        output = self.run_cli(["/model", "/model", "exit"])
        self.assertIn("Error listing models: offline", output)
        self.assertIn("No models available.", output)
        self.menu.pick.assert_not_called()

    def test_model_selection_cancel(self):
        self.menu.pick.side_effect = KeyboardInterrupt
        output = self.run_cli(["/model", "exit"])
        self.assertIn("Cancelled.", output)
        self.assertEqual(self.agent.model, "gpt-4o-mini")

    def test_session_search(self):
        with patch.object(cli.SessionRecall, "recall", side_effect=["past context", ""]) as recall:
            output = self.run_cli(["/sessions", "topic", "/sessions", "missing", "/sessions", "", "exit"])
        self.assertIn("past context", output)
        self.assertIn("No results.", output)
        self.assertEqual(recall.call_count, 2)
        self.assertEqual(len(self.saved_rows()), 1)

    def test_exit_aliases_and_interrupts(self):
        for inputs in (["EXIT"], ["quit"], ["/exit"], ["/QUIT"], [EOFError], [KeyboardInterrupt],
                       ["/sessions", EOFError]):
            with self.subTest(inputs=inputs):
                self.run_cli(inputs)
                self.assertEqual(len(self.saved_rows()), 1)

    def test_llm_failure_still_ends_session(self):
        with patch.object(self.client.chat.completions, "create", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                self.run_cli(["hello"])
        self.assertEqual([row["role"] for row in self.saved_rows()], ["system", "user"])

    def test_missing_config_exits_before_client_creation(self):
        with patch.object(cli, "__file__", str(self.root / "missing" / "cli.py")):
            with self.assertRaises(SystemExit) as error:
                self.run_cli([])
        self.assertEqual(error.exception.code, 1)
        self.assertIn("not found", self.output.getvalue())
        self.openai.assert_not_called()

    def test_configured_limits(self):
        (self.root / "config.yaml").write_text(
            "model:\n  api_key: test\n  base_url: http://localhost:1234/v1\n"
            "  model: gpt-4o-mini\n  max_tokens: 800\n"
            "agent:\n  max_iterations: 9\naux_model:\n  max_tokens: 200\n"
            "learning:\n  memory_nudge_interval: 3\n  skill_nudge_interval: 4\n",
            encoding="utf-8",
        )
        self.run_cli(["exit"])
        self.assertEqual(self.agent.max_tokens, 800)
        self.assertEqual(self.agent.max_iterations, 9)
        self.assertEqual(self.agent._memory_nudge_interval, 3)
        self.assertEqual(self.agent._skill_nudge_interval, 4)
        self.assertEqual(cli.tools.memory_tool._session_recall.max_tokens, 200)
        self.assertEqual(self.agent._compressor.max_tokens, 800)


if __name__ == "__main__":
    unittest.main()