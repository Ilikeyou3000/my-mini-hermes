"""离线回归：python3 -m unittest discover -s data/skills/duckduckgo-search/tests。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from lxml.html import fromstring

spec = importlib.util.spec_from_file_location("ddg_search", Path(__file__).parents[1] / "scripts/search.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

RESULT = '''<div class="result results_links">
<h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F">Python <b>文档</b></a></h2>
<a class="result__snippet">官方 <b>说明</b></a></div>'''


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(region=" wt-wt ", safesearch="moderate", timelimit="m", max_results=2)
        self.client = Mock()

    def search(self, body, status=200):
        self.client.request.return_value = SimpleNamespace(status_code=status, text=body)
        return module.search(self.client, fromstring, "Python documentation", self.args)

    def test_get_parameters_and_results(self):
        results = self.search(RESULT)
        self.assertEqual(results, [{"title": "Python 文档", "url": "https://docs.python.org/", "summary": "官方 说明"}])
        self.client.request.assert_called_once_with("GET", "https://html.duckduckgo.com/html/", params={
            "q": "Python documentation", "kl": "wt-wt", "kp": "-1", "df": "m"})

    def test_safesearch_and_optional_time(self):
        for level, value in (("on", "1"), ("moderate", "-1"), ("off", "-2")):
            self.args.safesearch, self.args.timelimit = level, None
            self.search(RESULT)
            params = self.client.request.call_args.kwargs["params"]
            self.assertEqual(params["kp"], value)
            self.assertNotIn("df", params)

    def test_dedup_and_limit(self):
        self.args.max_results = 1
        self.assertEqual(len(self.search(RESULT * 3)), 1)
        self.args.max_results = 20
        self.assertEqual(len(self.search(RESULT * 3)), 1)

    def test_challenge_stops_without_retry(self):
        for status in (200, 202, 403):
            self.client.reset_mock()
            with self.assertRaisesRegex(module.SearchFailure, "CHALLENGE"):
                self.search('<form id="challenge-form" action="/anomaly.js"></form>', status)
            self.client.request.assert_called_once()

    def test_status_and_unexpected_response(self):
        for body, status, code in (("", 429, "RATE_LIMITED"), ("", 503, "HTTP_ERROR"),
                                   ("<html>Forbidden</html>", 403, "HTTP_ERROR"),
                                   ("<html>Pending</html>", 202, "HTTP_ERROR"),
                                   ("<html>Changed layout</html>", 200, "UNEXPECTED_RESPONSE")):
            with self.subTest(status=status, code=code), self.assertRaisesRegex(module.SearchFailure, code):
                self.search(body, status)

    def test_explicit_empty(self):
        self.assertEqual(self.search('<div class="no-results">No results</div>'), [])
        self.assertEqual(self.search('<div class="result result--no-result"></div>'), [])

    def test_result_about_captcha_is_not_challenge(self):
        self.assertEqual(len(self.search(RESULT.replace("文档", "captcha anomaly.js"))), 1)

    def test_non_web_links_rejected(self):
        with self.assertRaisesRegex(module.SearchFailure, "UNEXPECTED_RESPONSE"):
            self.search(RESULT.replace("//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F", "javascript:alert(1)"))

    def test_main_json_and_redacted_errors(self):
        from ddgs.exceptions import TimeoutException
        with patch("ddgs.http_client.HttpClient", return_value=self.client):
            self.client.request.return_value = SimpleNamespace(status_code=200, text=RESULT)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(module.main(["Python", "--json"]), 0)
            self.assertEqual(json.loads(out.getvalue())["results"][0]["url"], "https://docs.python.org/")
            for exc, code in ((RuntimeError("SECRET proxy password"), "REQUEST_OR_PARSE_ERROR"),
                              (TimeoutException("SECRET"), "TIMEOUT")):
                self.client.request.side_effect = exc
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    self.assertEqual(module.main(["Python", "--json"]), 1)
                self.assertEqual(out.getvalue(), "")
                self.assertIn(code, err.getvalue())
                self.assertNotIn("SECRET", err.getvalue())

    def test_invalid_arguments(self):
        for argv in ([" "], ["query", "--max-results", "0"], ["query", "--timeout", "121"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                module.main(argv)
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()