"""Automatic checks for the deep-thinking loop.

A small model improves its own answer far more reliably when it is told about CONCRETE problems
than when it is just asked to "look again". These checks give it that feedback. Nothing is ever
executed: Python is parsed (ast) and linted (ruff, real-bug rules only) on a temporary copy, HTML
is parsed, JSON is loaded, and written-out calculations ("287 / 3.5833 = 79.9") are recomputed
from their numbers alone (a tiny + - * / evaluator, never eval).
"""
import ast
import json
import re
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

_FENCE = re.compile(r"```([\w+#.-]*)[^\n]*\n(.*?)```", re.DOTALL)
_LANG_LINE = re.compile(r"\s*(python|py|html|json|javascript|js|css|bash|sh)[ \t]*\n", re.I)
# ruff rules that point at real bugs, not style: syntax/statement errors, bad comparisons, undefined names
_RUFF_RULES = "E9,F63,F7,F82"
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
# closing tags HTML lets you leave out; everything else must be closed
_OPTIONAL_CLOSE = {"p", "li", "dt", "dd", "tr", "td", "th", "thead", "tbody", "tfoot", "option", "html", "head", "body",
                   "colgroup", "caption", "rt", "rp", "optgroup"}
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def code_blocks(text: str) -> list:
    """[(language, code)] for every fenced block; the language is guessed when the fence has none."""
    blocks = []
    for lang, code in _FENCE.findall(text or ""):
        lang = lang.lower()
        if not lang and (m := _LANG_LINE.match(code)):  # "```\npython\n..." - a model quirk
            lang, code = m.group(1).lower(), code[m.end():]
        if not lang:
            head = code.lstrip()[:200].lower()
            if head.startswith(("<!doctype", "<html")) or "<body" in head:
                lang = "html"
            elif re.search(r"^\s*(def |class |import |from \w+ import )", code, re.M):
                lang = "python"
        blocks.append(({"py": "python", "htm": "html", "jsonc": "json"}.get(lang, lang), code))
    return blocks


def check_python(code: str) -> list:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return [f"Python syntax error on line {exc.lineno}: {exc.msg}"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snippet.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "--no-cache", "--output-format", "concise",
                 "--select", _RUFF_RULES, str(path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired):
            return []
    issues = []
    for line in proc.stdout.splitlines():
        m = re.match(r".*?snippet\.py:(\d+):\d+: (\w+) (.*)", line)
        if m:
            issues.append(f"Python line {m.group(1)}: {m.group(3)} ({m.group(2)})")
    return issues[:8]


class _HTMLCheck(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack, self.issues, self.imgs_without_alt = [], [], 0

    def handle_starttag(self, tag, attrs):
        if tag == "img" and "alt" not in dict(attrs):
            self.imgs_without_alt += 1
        if tag not in _VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag, attrs):
        if tag == "img" and "alt" not in dict(attrs):
            self.imgs_without_alt += 1

    def handle_endtag(self, tag):
        if tag in _VOID:
            return
        open_tags = [t for t, _ in self.stack]
        if tag not in open_tags:
            self.issues.append(f"HTML line {self.getpos()[0]}: closing </{tag}> has no matching <{tag}>")
            return
        while self.stack:
            top, line = self.stack.pop()
            if top == tag:
                break
            if top not in _OPTIONAL_CLOSE:
                self.issues.append(f"HTML line {line}: <{top}> is never closed (before </{tag}>)")


def check_html(code: str) -> list:
    checker = _HTMLCheck()
    try:
        checker.feed(code)
        checker.close()
    except Exception as exc:  # malformed beyond repair
        return [f"HTML could not be parsed: {exc}"]
    issues = list(checker.issues)
    issues += [f"HTML line {line}: <{tag}> is never closed" for tag, line in checker.stack if tag not in _OPTIONAL_CLOSE]
    low = code.lower()
    if "<html" in low:  # a full page: check the basics of a good modern page
        if "<!doctype html" not in low:
            issues.append("HTML page has no <!DOCTYPE html> at the top")
        if 'name="viewport"' not in low and "name='viewport'" not in low:
            issues.append('HTML page has no <meta name="viewport" ...>, so it will not be responsive on phones')
        if "<title" not in low:
            issues.append("HTML page has no <title>")
    if checker.imgs_without_alt:
        issues.append(f"{checker.imgs_without_alt} <img> tag(s) without alt text (accessibility)")
    return issues[:8]


def check_json(code: str) -> list:
    try:
        json.loads(code)
    except ValueError as exc:
        return [f"JSON is invalid: {exc}"]
    return []


_CHECKERS = {"python": check_python, "html": check_html, "json": check_json}

# --- arithmetic in prose: "287 / 3.5833 = 79.9" is recomputed exactly ---------------------------------
_NUM = r"\d[\d,]*(?:\.\d+)?"
_UNIT = r"(?:\s+[A-Za-z]{1,7}\b(?!\s*[=≈]))?"  # "287 km / 3.5 hours": unit words between numbers
_OPERAND = rf"\(?\s*{_NUM}\s*\)?{_UNIT}"  # "(287)/(3.5833)" as well as "287 / 3.5833"
_EXPR = rf"(?<![\w.,]){_OPERAND}(?:\s*[-+*/×÷]\s*{_OPERAND})+"
_LATEX_MATH = [(re.compile(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}"), r"(\1)/(\2)"),
               (re.compile(r"\\(?:text|mathrm|boxed|mathbf)\{([^{}]*)\}"), r"\1"),
               (re.compile(r"\\(?:times|cdot)"), "*"), (re.compile(r"\\div"), "/"), (re.compile(r"\\approx"), "≈"),
               (re.compile(r"\$+|\\[()\[\]]"), " ")]


def _plain(text: str) -> str:
    """LaTeX math -> plain operators, so '\\frac{287}{3.5833} = 79.8' is checked like '287 / 3.5833 = 79.8'."""
    for pattern, repl in _LATEX_MATH:
        text = pattern.sub(repl, text)
    return text
_ARITH = re.compile(rf"({_EXPR})\s*\**\s*(=|≈|~=|≅)\s*\**\s*({_EXPR}|{_NUM})(?![\d.,]*\d)")
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")


def _value(expr: str):
    """Evaluate a + - * / expression of plain numbers (units dropped); None if it is anything else."""
    cleaned = re.sub(r"[A-Za-z]+", "", expr).replace("×", "*").replace("÷", "/").replace(",", "")
    try:
        tree = ast.parse(cleaned.strip(), mode="eval")
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Div):
                return a / b if b else None
            return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}[type(node.op)] if None not in (a, b) else None
        raise ValueError
    try:
        return ev(tree.body)
    except (ValueError, TypeError):
        return None


def _fmt(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


_CLOCK = r"(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?"
_SPAN = re.compile(_CLOCK + r"\s*(?:-|–|to|until|till|and)\s*" + _CLOCK, re.I)
_DURATION = re.compile(r"\s*(?:=|is|was|takes?|took|lasts?|of|:|→|,)?\s*(?:exactly|about|approximately|≈|~)?\s*\**\s*"
                       r"(?:(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b(?:\s*(?:and\s*)?(\d+)\s*(minutes?|mins?|m)\b)?"
                       r"|(\d+)\s*(minutes?|mins?)\b)", re.I)


def _minutes(hour: str, minute: str, ampm: str) -> int:
    h = int(hour) % 24
    marker = (ampm or "").lower().replace(".", "")
    if marker == "pm" and h < 12:
        h += 12
    if marker == "am" and h == 12:
        h = 0
    return h * 60 + int(minute)


def check_time_spans(text: str) -> tuple:
    """(spans checked, issues): a duration written right after two clock times must match them.
    '9:40 to 13:15 is 4 h 15 min' -> 3 h 35 min (a real draft got this wrong and its arithmetic was
    otherwise self-consistent, so nothing else could catch it)."""
    prose = _FENCE.sub(" ", text or "")
    checked, issues = 0, []
    for m in _SPAN.finditer(prose):
        a, b = _minutes(*m.group(1, 2, 3)), _minutes(*m.group(4, 5, 6))
        separator = prose[m.end(3) if m.group(3) else m.end(2):m.start(4)]
        if ("-" in separator or "–" in separator) and a > b:
            actual = a - b  # written as end - start: "13:15 - 9:40"
        else:
            actual = (b - a) % 1440  # from start to end, past midnight if needed
        d = _DURATION.match(prose, m.end())
        if not d:
            continue
        if d.group(1):
            stated = float(d.group(1)) * 60 + (int(d.group(3)) if d.group(3) else 0)
        else:
            stated = float(d.group(5))
        checked += 1
        if abs(stated - actual) > 1.5:
            issues.append(f"Time: from {m.group(0).strip()} is {actual // 60} h {actual % 60} min "
                          f"({actual / 60:.4g} hours), not {int(stated) // 60} h {round(stated) % 60} min")
    return checked, issues


def check_arithmetic(text: str) -> tuple:
    """(statements checked, issues) for written-out calculations outside code blocks."""
    text = _plain(text or "")
    span_checked, span_issues = check_time_spans(text)
    prose = _TIME.sub("TIME", _FENCE.sub(" ", text))
    checked, issues = span_checked, list(span_issues)
    for left, sign, right in _ARITH.findall(prose):
        lv, rv = _value(left), _value(right)
        if lv is None or rv is None:
            continue
        checked += 1
        plain = re.fullmatch(r"\s*" + _NUM + r"\s*", right)
        if plain:  # a stated result: allow the rounding its decimals imply (plus a little for "≈")
            decimals = len(right.strip().split(".")[1]) if "." in right else 0
            tol = 0.5 * 10 ** -decimals + abs(lv) * (0.0005 if sign != "=" else 1e-9)
        else:
            tol = abs(lv) * 1e-6 + 1e-9
        if abs(lv - rv) > tol + 1e-9:
            shown = " ".join(left.split())
            issues.append(f"Arithmetic: {shown} is {_fmt(lv)}, not {' '.join(right.split())}")
    return checked, issues[:6]


def run_checks(text: str) -> dict:
    """{"checked": ["python", ...], "issues": [...]} for the code blocks in an answer (or one raw HTML document)."""
    blocks = code_blocks(text)
    stripped = (text or "").lstrip().lower()
    if not blocks and stripped.startswith(("<!doctype", "<html")):
        blocks = [("html", text)]
    checked, issues = [], []
    for i, (lang, code) in enumerate(blocks, 1):
        checker = _CHECKERS.get(lang)
        if not checker:
            continue
        checked.append(lang)
        prefix = f"Code block {i} ({lang}): " if len(blocks) > 1 else ""
        issues += [prefix + issue for issue in checker(code)]
    if "html" not in checked:
        count, wrong = check_arithmetic(text)
        if count:
            checked.append("arithmetic")
            issues += wrong
    return {"checked": checked, "issues": issues[:12]}
