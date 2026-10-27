"""One-call helpers for building documents from Python scripts.

The coding agent's scripts run with this project on their import path, so a
small model can write ``from officekit import make_docx`` instead of juggling
python-docx, openpyxl, python-pptx, PyMuPDF and matplotlib directly.

Every helper creates parent folders, returns the absolute path it wrote, and
accepts the same "sections" shape:

    sections = [
        ("Summary", "Plain text. A blank line starts a new paragraph."),
        ("Key points", ["bullet one", "bullet two"]),
        ("Numbers", [["Item", "Cost"], ["Pens", 12], ["Paper", 30]]),   # table, first row = header
        ("Chart", {"image": "sales.png", "caption": "Sales by month"}),
    ]
"""
import csv
import html
import os
from pathlib import Path

__all__ = ["make_docx", "make_xlsx", "make_pptx", "make_pdf", "make_chart", "make_csv"]


def _out(path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _blocks(content) -> list:
    """Normalise a section's content to a list of ("text"|"bullets"|"table"|"image", value)."""
    if content is None:
        return []
    if isinstance(content, str):
        return [("text", para.strip()) for para in content.split("\n\n") if para.strip()]
    if isinstance(content, dict):  # explicit, unambiguous blocks: {"text"|"bullets"|"table"|"image": ...}
        if "image" in content:
            return [("image", content)]
        if "table" in content:
            return [("table", [list(r) for r in content["table"]])]
        if "bullets" in content:
            return [("bullets", [str(b) for b in content["bullets"]])]
        if "text" in content:
            return _blocks(str(content["text"]))
        return [("text", str(content))]
    if isinstance(content, (list, tuple)):
        if not content:
            return []
        if all(isinstance(x, dict) for x in content):
            return [b for item in content for b in _blocks(item)]
        if all(isinstance(x, str) for x in content):
            return [("bullets", list(content))]
        if all(isinstance(x, (list, tuple)) for x in content):
            return [("table", [list(r) for r in content])]
        blocks = []
        for item in content:  # a mix of the above
            blocks.extend(_blocks(item))
        return blocks
    return [("text", str(content))]


def make_docx(path, title, sections=(), subtitle=None):
    """Create a Word (.docx) report: title, optional subtitle, then headed sections."""
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    doc.add_heading(str(title), level=0)
    if subtitle:
        doc.add_paragraph(str(subtitle)).italic = True
    for heading, content in sections:
        if heading:
            doc.add_heading(str(heading), level=1)
        for kind, value in _blocks(content):
            if kind == "text":
                doc.add_paragraph(value)
            elif kind == "bullets":
                for item in value:
                    doc.add_paragraph(str(item), style="List Bullet")
            elif kind == "table":
                table = doc.add_table(rows=len(value), cols=max(len(r) for r in value))
                table.style = "Light Grid Accent 1"
                for i, row in enumerate(value):
                    for j, cell in enumerate(row):
                        table.cell(i, j).text = str(cell)
                        if i == 0:
                            for run in table.cell(i, j).paragraphs[0].runs:
                                run.bold = True
                doc.add_paragraph()
            elif kind == "image":
                if Path(value["image"]).is_file():
                    doc.add_picture(str(value["image"]), width=Inches(6))
                if value.get("caption"):
                    doc.add_paragraph(str(value["caption"])).italic = True
    out = _out(path)
    doc.save(out)
    return str(out.resolve())


def make_xlsx(path, sheets):
    """Create an Excel (.xlsx) workbook. `sheets` is {"Sheet name": rows} or just rows;
    rows are lists of cell values and the first row is styled as a header."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    if not isinstance(sheets, dict):
        sheets = {"Sheet1": sheets}
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(str(name)[:31] or "Sheet")
        for row in rows:
            ws.append(list(row))
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="2F5597")
        for col in range(1, ws.max_column + 1):
            width = max((len(str(c.value)) for c in ws[get_column_letter(col)] if c.value is not None), default=8)
            ws.column_dimensions[get_column_letter(col)].width = min(width + 2, 60)
        ws.freeze_panes = "A2"
    out = _out(path)
    wb.save(out)
    return str(out.resolve())


def make_pptx(path, title, slides=(), subtitle=None):
    """Create a PowerPoint (.pptx) deck. `slides` is a list of (slide title, content) where
    content is a string, a list of bullet strings, or {"image": "chart.png"}."""
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    first = prs.slides.add_slide(prs.slide_layouts[0])
    first.shapes.title.text = str(title)
    if subtitle and len(first.placeholders) > 1:
        first.placeholders[1].text = str(subtitle)
    for slide_title, content in slides:
        blocks = _blocks(content)
        image = next((v for k, v in blocks if k == "image" and Path(v["image"]).is_file()), None)
        slide = prs.slides.add_slide(prs.slide_layouts[5 if image else 1])
        slide.shapes.title.text = str(slide_title)
        if image:
            slide.shapes.add_picture(str(image["image"]), Inches(1), Inches(1.5), width=Inches(8))
            continue
        body = slide.placeholders[1].text_frame
        body.clear()
        lines = []
        for kind, value in blocks:
            if kind == "bullets":
                lines.extend(str(v) for v in value)
            elif kind == "table":
                lines.extend(" | ".join(str(c) for c in row) for row in value)
            elif kind == "text":
                lines.append(value)
        for i, line in enumerate(lines or [""]):
            para = body.paragraphs[0] if i == 0 else body.add_paragraph()
            para.text = line
    out = _out(path)
    prs.save(out)
    return str(out.resolve())


def make_pdf(path, title, sections=()):
    """Create a PDF report (A4, paginated) with the same `sections` as make_docx."""
    import pymupdf

    parts = [f"<h1>{html.escape(str(title))}</h1>"]
    for heading, content in sections:
        if heading:
            parts.append(f"<h2>{html.escape(str(heading))}</h2>")
        for kind, value in _blocks(content):
            if kind == "text":
                parts.append(f"<p>{html.escape(value)}</p>")
            elif kind == "bullets":
                parts.append("<ul>" + "".join(f"<li>{html.escape(str(v))}</li>" for v in value) + "</ul>")
            elif kind == "table":
                rows = []
                for i, row in enumerate(value):
                    tag = "th" if i == 0 else "td"
                    rows.append("<tr>" + "".join(
                        f'<{tag} style="border:1px solid #999; padding:3px">{html.escape(str(c))}</{tag}>' for c in row) + "</tr>")
                parts.append('<table style="border-collapse:collapse">' + "".join(rows) + "</table><p></p>")
            elif kind == "image" and Path(value["image"]).is_file():
                parts.append(f'<p><img src="{html.escape(str(value["image"]))}" width="440"></p>')
                if value.get("caption"):
                    parts.append(f"<p><i>{html.escape(str(value['caption']))}</i></p>")
    story = pymupdf.Story(html="".join(parts), archive=pymupdf.Archive(os.getcwd()))
    out = _out(path)
    writer = pymupdf.DocumentWriter(str(out))
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (50, 50, -50, -50)
    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    # The writer embeds full fonts; re-saving with garbage collection + deflate shrinks it a lot.
    with pymupdf.open(str(out)) as doc:
        doc.subset_fonts()
        data = doc.tobytes(garbage=4, deflate=True)
    out.write_bytes(data)
    return str(out.resolve())


def make_chart(path, kind, labels, values, title="", xlabel="", ylabel=""):
    """Create a chart image (.png). kind: "bar", "barh", "line" or "pie".
    `values` is a list of numbers, or {"series name": [numbers]} for several series."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = values if isinstance(values, dict) else {"": list(values)}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if kind == "pie":
        ax.pie(next(iter(series.values())), labels=labels, autopct="%1.0f%%")
    else:
        width = 0.8 / len(series)
        for i, (name, data) in enumerate(series.items()):
            if kind == "line":
                ax.plot(labels, data, marker="o", label=name or None)
            elif kind == "barh":
                ax.barh([j + i * width for j in range(len(labels))], data, height=width, label=name or None)
                ax.set_yticks([j + width * (len(series) - 1) / 2 for j in range(len(labels))], labels)
            else:
                ax.bar([j + i * width for j in range(len(labels))], data, width=width, label=name or None)
                ax.set_xticks([j + width * (len(series) - 1) / 2 for j in range(len(labels))], labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if len(series) > 1 or "" not in series:
            ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    out = _out(path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out.resolve())


def make_csv(path, rows):
    """Write rows (a list of lists) to a CSV file."""
    out = _out(path)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:  # utf-8-sig so Excel opens it cleanly
        csv.writer(fh).writerows(rows)
    return str(out.resolve())
