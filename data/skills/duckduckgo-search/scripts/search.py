#!/usr/bin/env python3
"""独立 DuckDuckGo 网页检索；不依赖 mini-hermes Agent。"""

import argparse
import json
import sys
from urllib.parse import parse_qs, urljoin, urlsplit


class SearchFailure(Exception):
    """仅携带可安全展示的固定诊断信息。"""


def search(client, parse_html, query, args):
    # 直接请求 HTML GET 接口，避免 ddgs 聚合层丢失非 200 状态。
    params = {
        "q": query, "kl": args.region.strip(),
        "kp": {"on": "1", "moderate": "-1", "off": "-2"}[args.safesearch],
    }
    if args.timelimit:
        params["df"] = args.timelimit
    response = client.request("GET", "https://html.duckduckgo.com/html/", params=params)
    status = response.status_code
    if status == 429:
        raise SearchFailure("[RATE_LIMITED] HTTP 429：服务限流，请稍后重试。")
    # 202 可能是验证页，但不能只凭状态码断言需要验证码。
    if status not in (200, 202, 403):
        raise SearchFailure(f"[HTTP_ERROR] HTTP {status}：服务未返回正常检索响应。")
    try:
        tree = parse_html(response.text)
    except (ValueError, TypeError):
        raise SearchFailure("[UNEXPECTED_RESPONSE] 返回内容无法解析为检索页面。") from None
    challenge = tree.xpath(
        '//form[@id="challenge-form" or contains(@action, "anomaly.js")]'
        ' | //*[@id="anomaly-modal"]'
    )
    if challenge:
        raise SearchFailure(
            f"[CHALLENGE] HTTP {status}：DuckDuckGo 返回人机验证页；"
            "本次未获得结果，请稍后重试或在浏览器中手动检索，不自动重试或绕过验证。"
        )
    if status != 200:
        raise SearchFailure(f"[HTTP_ERROR] HTTP {status}：服务未返回正常检索响应。")
    results = []
    seen = set()
    for item in tree.xpath('//div[contains(concat(" ", normalize-space(@class), " "), " result ")]'):
        links = item.xpath('.//a[contains(concat(" ", normalize-space(@class), " "), " result__a ")]')
        if not links:
            continue
        link = links[0]
        url = urljoin("https://html.duckduckgo.com", link.get("href", ""))
        parts = urlsplit(url)
        if parts.hostname in ("duckduckgo.com", "html.duckduckgo.com"):
            url = parse_qs(parts.query).get("uddg", [""])[0]
        if urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).netloc or url in seen:
            continue
        title = " ".join(link.text_content().split())
        if not title:
            continue
        snippets = item.xpath('.//*[contains(concat(" ", normalize-space(@class), " "), " result__snippet ")]')
        summary = " ".join(snippets[0].text_content().split()) if snippets else ""
        results.append({"title": title, "url": url, "summary": summary})
        seen.add(url)
        if len(results) >= args.max_results:
            break
    if not results and not tree.xpath(
        '//*[contains(concat(" ", normalize-space(@class), " "), " no-results ")'
        ' or contains(concat(" ", normalize-space(@class), " "), " result--no-result ")]'
    ):
        raise SearchFailure("[UNEXPECTED_RESPONSE] 页面没有可识别的结果或无结果标记，不能认定检索为空。")
    return results


def bounded_int(low, high):
    def parse(value):
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("请输入整数") from None
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"取值范围为 {low} 到 {high}")
        return number
    return parse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="检索关键词；包含空格时使用引号")
    parser.add_argument("--max-results", type=bounded_int(1, 20), default=5)
    parser.add_argument("--region", default="wt-wt", help="地区，如 wt-wt、cn-zh、us-en")
    parser.add_argument("--safesearch", choices=("on", "moderate", "off"), default="moderate")
    parser.add_argument("--timelimit", choices=("d", "w", "m", "y"), help="日/周/月/年")
    parser.add_argument("--timeout", type=bounded_int(1, 120), default=15, help="请求超时秒数")
    parser.add_argument("--json", action="store_true", help="输出 JSON，便于后续处理")
    args = parser.parse_args(argv)
    query = args.query.strip()
    if not query:
        parser.error("检索关键词不能为空")
    if not args.region.strip():
        parser.error("地区不能为空")

    try:
        from ddgs.http_client import HttpClient
        from ddgs.exceptions import TimeoutException
        from lxml.html import fromstring
    except ImportError:
        print("缺少检索依赖，请运行：python3 -m pip install ddgs==9.16.0", file=sys.stderr)
        return 1

    try:
        results = search(HttpClient(timeout=args.timeout), fromstring, query, args)
    except SearchFailure as exc:
        print(f"DuckDuckGo 检索失败：{exc}", file=sys.stderr)
        return 1
    except TimeoutException:
        print("DuckDuckGo 检索失败：[TIMEOUT] 请求超时，请稍后重试。", file=sys.stderr)
        return 1
    except Exception as exc:
        # 不回显底层异常文本，避免泄露代理凭据或请求中的敏感信息。
        print(
            f"DuckDuckGo 检索失败：[REQUEST_OR_PARSE_ERROR] {type(exc).__name__}；"
            "请求或解析失败，无法仅据此判断网络不通或限流。",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2))
    elif not results:
        print("未找到结果，请调整关键词或放宽时间限制。")
    else:
        for index, result in enumerate(results, 1):
            print(f"{index}. {result['title']}\n   {result['url']}\n   {result['summary']}\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("检索已取消。", file=sys.stderr)
        sys.exit(130)
    except BrokenPipeError:
        sys.exit(0)