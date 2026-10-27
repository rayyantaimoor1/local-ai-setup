"""Documents the chat creates directly from text: Word, PDF, PowerPoint, Excel, CSV, Markdown, charts.

The model never writes or runs code for this. It passes the CONTENT (simple Markdown) to a
tool, and this module builds the file with officekit. Files only ever go into the outputs
folder, and nothing that looks like a secret is copied into a document (constitution 1.3).

Markdown understood: "#"/"##"/"###" headings (each starts a section, or a slide in PowerPoint),
"- " / "* " / "1. " bullets, "| a | b |" tables (first row = header), "![caption](chart.png)"
images (a chart made earlier in the outputs folder), and plain paragraphs.
"""
import csv
import io
import re
from html import escape
from pathlib import Path

import config
import constitution
import officekit

DOC_KINDS = (".docx", ".pdf", ".pptx", ".xlsx", ".csv", ".md", ".txt", ".html")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")
_IMAGE = re.compile(r"^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


_MAKE_VERB = re.compile(r"\b(create|make|generate|export|write|save|produce|build|prepare|draft|put)\b", re.I)
_INTO = re.compile(r"\b(into|as|in)\s+(?:an?|the)?\s*(?:[\w-]+\s+){0,2}$", re.I)
_READ_VERB = re.compile(r"\b(summari[sz]\w*|read\w*|extract\w*|analy[sz]\w*|describ\w*|explain\w*|review\w*|"
                        r"open\w*|pars\w*|inspect\w*|from)\b", re.I)
_NOUNS = [
    (".docx", re.compile(r"\b(word (?:report|document|doc|file|table)|docx)\b", re.I)),
    (".xlsx", re.compile(r"\b(excel|xlsx|spreadsheet|workbook)\b", re.I)),
    (".pptx", re.compile(r"\b(powerpoint|pptx|slide ?deck|slides|presentation)\b", re.I)),
    (".pdf", re.compile(r"\bpdf\b", re.I)),
    (".png", re.compile(r"\b(chart|plot|graph|png|image)\b", re.I)),
    (".csv", re.compile(r"\bcsv\b", re.I)),
    (".html", re.compile(r"\b(html file|html document)\b|\.html\b", re.I)),
]


def requested_outputs(request: str) -> list:
    """File types the request asks to be CREATED (".docx", ".png", ...).

    Only counted when a "make" verb (or "into/as/in a ...") leads to the noun without a reading verb
    in between: "Create a Word report summarizing the PDF" asks for .docx, not .pdf (the PDF is input).
    """
    wanted = []
    for ext, noun in _NOUNS:
        for match in noun.finditer(request):
            before = request[max(0, match.start() - 160):match.start()]
            verbs = list(_MAKE_VERB.finditer(before))
            if verbs:
                between = before[verbs[-1].end():]
                if len(between.split()) <= 24 and not _READ_VERB.search(between):
                    wanted.append(ext)
                    break
            elif _INTO.search(before):
                wanted.append(ext)
                break
    return wanted


def outputs_dir() -> Path:
    d = Path(config.OUTPUTS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_output(rel_path: str) -> Path:
    """A file inside the outputs folder, or ValueError (the download endpoint is confined to it)."""
    rel = (rel_path or "").strip().replace("\\", "/")
    if not rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        raise ValueError(f"Path must be relative: {rel_path!r}")
    root = outputs_dir().resolve()
    target = (root / rel).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"Path escapes the outputs folder: {rel_path!r}")
    return target


def snapshot() -> dict:
    return {p.name: p.stat().st_mtime_ns for p in outputs_dir().iterdir() if p.is_file()}


def new_files(before: dict) -> list:
    """Files created or changed in the outputs folder since `before`."""
    now = snapshot()
    return sorted(name for name, mtime in now.items() if before.get(name) != mtime)


def _safe_name(filename: str, allowed: tuple, default_ext: str) -> str:
    """A plain file name inside the outputs folder: no folders, no '..', a known extension."""
    name = Path(str(filename or "").replace("\\", "/")).name
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    ext = f".{ext.lower()}" if ext else default_ext
    if ext not in allowed:
        raise ValueError(f"Unsupported file type '{ext}'. Use one of: {', '.join(allowed)}")
    stem = re.sub(r"[^\w\- ]+", "", stem).strip().replace(" ", "_")[:80] or "document"
    return stem + ext


def _unique(name: str) -> Path:
    """Never overwrite an earlier output: report.docx, report-2.docx, report-3.docx..."""
    path = outputs_dir() / name
    stem, ext = path.stem, path.suffix
    n = 2
    while path.exists():
        path = outputs_dir() / f"{stem}-{n}{ext}"
        n += 1
    return path


def _plain(text: str) -> str:
    """Strip inline Markdown the document formats don't need."""
    return re.sub(r"(\*\*|__|`)", "", text).strip()


def _cell(value: str):
    v = value.strip()
    try:
        return int(v) if re.fullmatch(r"-?\d+", v) else float(v) if re.fullmatch(r"-?\d+\.\d+", v) else v
    except ValueError:
        return v


def parse_markdown(content: str) -> list:
    """Markdown -> officekit sections: [(heading, [blocks...]), ...]."""
    sections = [["", []]]
    para, bullets, table = [], [], []

    def flush():
        nonlocal para, bullets, table
        blocks = sections[-1][1]
        if para:
            blocks.append({"text": " ".join(para)})
        if bullets:
            blocks.append({"bullets": list(bullets)})
        if table:
            blocks.append({"table": [list(r) for r in table]})
        para, bullets, table = [], [], []

    for raw in str(content or "").replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        if m := _HEADING.match(line):
            flush()
            sections.append([_plain(m.group(1)), []])
        elif m := _IMAGE.match(line):
            flush()
            img = outputs_dir() / Path(m.group(2).strip().replace("\\", "/")).name  # only charts made in outputs/
            sections[-1][1].append({"image": str(img), "caption": _plain(m.group(1))})
            continue
        elif line.lstrip().startswith("|"):
            if para or bullets:
                flush()
            if not _TABLE_SEP.match(line):
                table.append([_plain(c) for c in line.strip().strip("|").split("|")])
        elif m := _BULLET.match(line):
            if para or table:
                flush()
            bullets.append(_plain(m.group(1)))
        else:
            if bullets or table:
                flush()
            para.append(_plain(line))
    flush()

    # every block is an explicit {"text"|"bullets"|"table"|"image": ...} dict, so officekit never has to guess
    return [(heading, blocks) for heading, blocks in sections if heading or blocks]


def _tables(sections) -> list:
    """[(heading, rows)] for every table in the parsed content."""
    return [(heading, block["table"]) for heading, blocks in sections for block in blocks if "table" in block]


def _refuse_secrets(*texts) -> None:
    if not constitution.enforced("secrets_in_documents"):
        return
    for text in texts:
        kinds = constitution.find_secrets(str(text))
        if kinds:
            constitution.audit("document_blocked", rule="secrets_in_documents", kind=kinds[0])
            raise ValueError(f"Refused by the constitution (data security): the content contains something that "
                             f"looks like a secret ({kinds[0]}), and secrets are never copied into documents.")


_HTML_STYLE = ("body{font:16px/1.6 system-ui,'Segoe UI',sans-serif;max-width:860px;margin:40px auto;padding:0 20px;"
               "color:#1f2328}h1,h2{line-height:1.25}table{border-collapse:collapse;margin:12px 0}"
               "td,th{border:1px solid #d0d7de;padding:6px 10px}th{background:#f6f8fa}img{max-width:100%}")


def _html_page(title: str, content: str, sections: list) -> str:
    """A web page: HTML the model wrote is saved as it is; Markdown becomes a simple, clean page."""
    raw = re.sub(r"^\s*```[\w-]*\n|\n```\s*$", "", str(content).strip())
    if re.match(r"\s*<(!doctype|html|head|body|div|main|section|header|style)\b", raw, re.I):
        return raw + "\n"
    out = [f"<h1>{escape(_plain(title))}</h1>"] if title else []
    for heading, blocks in sections:
        if heading:
            out.append(f"<h2>{escape(heading)}</h2>")
        for block in blocks:
            if "text" in block:
                out.append(f"<p>{escape(block['text'])}</p>")
            elif "bullets" in block:
                out.append("<ul>" + "".join(f"<li>{escape(b)}</li>" for b in block["bullets"]) + "</ul>")
            elif "table" in block:
                rows = block["table"]
                head = "".join(f"<th>{escape(c)}</th>" for c in rows[0])
                body = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            elif "image" in block:
                out.append(f"<figure><figcaption>{escape(block.get('caption', ''))}</figcaption></figure>")
    return ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{escape(_plain(title) or 'Document')}</title>\n<style>{_HTML_STYLE}</style>\n</head>\n<body>\n"
            + "\n".join(out) + "\n</body>\n</html>\n")


def create_document(filename: str, title: str, content: str) -> str:
    """Create a Word (.docx), PDF, PowerPoint (.pptx), Excel (.xlsx), CSV, Markdown, text or web page (.html) file from Markdown content (headings, - bullets, | tables |, ![caption](chart.png)); for .html the content may be a complete HTML page. Saved in the outputs folder."""
    try:
        name = _safe_name(filename, DOC_KINDS, ".docx")
        _refuse_secrets(title, content)
        sections = parse_markdown(content)
        path = _unique(name)
        ext = path.suffix
        if ext == ".docx":
            officekit.make_docx(path, _plain(title), sections)
        elif ext == ".pdf":
            officekit.make_pdf(path, _plain(title), sections)
        elif ext == ".pptx":
            officekit.make_pptx(path, _plain(title), [(h or _plain(title), c) for h, c in sections] or [(_plain(title), "")])
        elif ext in (".xlsx", ".csv"):
            tables = _tables(sections)
            if not tables:  # accept plain CSV lines too
                rows = [r for r in csv.reader(io.StringIO(str(content).strip())) if any(c.strip() for c in r)]
                if len(rows) < 2:
                    raise ValueError("For a spreadsheet or CSV, give the data as a Markdown table (| a | b |) or CSV lines.")
                tables = [("Sheet1", [[c.strip() for c in r] for r in rows])]
            if ext == ".csv":
                officekit.make_csv(path, tables[0][1])
            else:
                sheets, seen = {}, set()
                for i, (heading, rows) in enumerate(tables, 1):
                    sheet = (heading or f"Sheet{i}")[:31]
                    while sheet in seen:
                        sheet = f"{sheet[:28]}-{i}"
                    seen.add(sheet)
                    sheets[sheet] = [[_cell(str(c)) for c in r] for r in rows]
                officekit.make_xlsx(path, sheets)
        elif ext == ".html":
            path.write_text(_html_page(title, content, sections), encoding="utf-8")
        else:  # .md / .txt
            path.write_text((f"# {title}\n\n" if title else "") + str(content).strip() + "\n", encoding="utf-8")
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:  # a library failed on odd input: report it, never crash the chat
        return f"Error: could not create the document: {type(exc).__name__}: {exc}"
    constitution.audit("document_created", file=path.name)
    return f"Created {path.name} ({max(1, path.stat().st_size // 1024)} KB) in the outputs folder."


def create_chart(filename: str, kind: str, labels: list, values, title: str = "", xlabel: str = "", ylabel: str = "") -> str:
    """Create a chart image (.png): kind is bar, barh, line or pie; values is a list of numbers or {"series": [numbers]}. Use it in a document with ![caption](name.png)."""
    try:
        name = _safe_name(filename, (".png",), ".png")
        if kind not in ("bar", "barh", "line", "pie"):
            raise ValueError("kind must be bar, barh, line or pie.")
        series = values if isinstance(values, dict) else {"": values}
        if not labels or any(len(v) != len(labels) for v in series.values()):
            raise ValueError("labels and every list of values must have the same length.")
        _refuse_secrets(title, labels)
        path = _unique(name)
        officekit.make_chart(path, kind, [str(x) for x in labels], values, _plain(title), xlabel, ylabel)
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error: could not create the chart: {type(exc).__name__}: {exc}"
    constitution.audit("document_created", file=path.name)
    return f"Created {path.name} in the outputs folder. Put it in a document with ![caption]({path.name})."
