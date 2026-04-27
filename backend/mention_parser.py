"""
mention_parser.py — M22 Phase 4 @mention 解析

解析 markdown body 中的 @user@example.com pattern。
格式：@<local>@<domain>（完整 email address，前置 @）

回傳：去重後的 email list（僅限格式合法）。
"""

import re
from typing import List

# 比對 @user@domain.tld（允許 +、. 等合法 email local-part 字元）
_MENTION_RE = re.compile(
    r"@([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})"
)


def parse_mentions(body_markdown: str) -> List[str]:
    """
    從 markdown 文字中抽取 @mention email。
    - 去重
    - 保持出現順序（first-seen）
    - 只回傳格式合法的 email（regex 已確保基本格式）

    >>> parse_mentions("Hi @alice@example.com please review @bob@company.org")
    ['alice@example.com', 'bob@company.org']
    >>> parse_mentions("no mention here")
    []
    >>> parse_mentions("@alice@example.com @alice@example.com")
    ['alice@example.com']
    """
    seen: set = set()
    result: List[str] = []
    for m in _MENTION_RE.finditer(body_markdown):
        email = m.group(1).lower()
        if email not in seen:
            seen.add(email)
            result.append(email)
    return result
