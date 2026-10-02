"""Render the Markdown subset docs/ uses to PDF, with Korean text and aligned diagrams.

Only reportlab is available here, so the Markdown is parsed directly rather than going
through HTML. The subset is what the documents actually contain: headings, paragraphs,
bullets, fenced code, pipe tables, rules, and inline bold / code / links.

Fonts matter more than usual, in two ways.

Body text is Malgun Gothic, which has Hangul; code blocks are GulimChe, the one monospace
face on Windows that also has Hangul - measured, M and i are both 20.0 at size 10 - and the
ASCII diagrams in docs/MODEL.md only line up in a fixed-width face.

And a glyph the font lacks is dropped SILENTLY by reportlab. That is how this script first
rendered "x-hat = x + dx" as "x = x + dx", a different and wrong statement, and how the
norm bars disappeared out of ||r||. So every character is checked against the font that
will draw it, mapped when there is a sensible stand-in, and REPORTED when there is not.
"""

from __future__ import annotations

import html
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, ListFlowable, ListItem, PageBreak,
                                Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

BODY, BOLD, MONO = "Malgun", "MalgunBold", "GulimChe"

# Malgun Gothic lacks the typographic minus and the solid triangle; GulimChe has both, so
# code blocks and diagrams keep them and only body text is rewritten. The combining marks
# are in neither font and have no stand-in that reads correctly, which is why the source
# documents avoid them - they are listed so that a stray one is dropped loudly.
FALLBACKS = {"\u2212": "-", "\u25ba": ">", "\u2016": "||", "\u1d40": "^T"}
_reported: set[str] = set()


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont(BODY, "C:/Windows/Fonts/malgun.ttf"))
    pdfmetrics.registerFont(TTFont(BOLD, "C:/Windows/Fonts/malgunbd.ttf"))
    pdfmetrics.registerFont(TTFont(MONO, "C:/Windows/Fonts/gulim.ttc", subfontIndex=1))
    pdfmetrics.registerFontFamily(BODY, normal=BODY, bold=BOLD, italic=BODY,
                                  boldItalic=BOLD)


def safe(text: str, font: str) -> str:
    """Replace characters `font` cannot draw; complain about any with no stand-in."""
    cmap = pdfmetrics.getFont(font).face.charToGlyph
    out = []
    for ch in text:
        if ch in "\n\t" or ord(ch) in cmap:
            out.append(ch)
        elif ch in FALLBACKS:
            out.append(FALLBACKS[ch])
        elif ch not in _reported:
            _reported.add(ch)
            print(f"  [warn] {font} has no glyph for U+{ord(ch):04X} and no fallback - "
                  f"dropped", file=sys.stderr)
    return "".join(out)


def styles() -> dict:
    base = ParagraphStyle("body", fontName=BODY, fontSize=9.5, leading=15,
                          alignment=TA_LEFT, spaceAfter=7,
                          textColor=colors.HexColor("#1a1a1a"))
    return {
        "body": base,
        "h1": ParagraphStyle("h1", parent=base, fontName=BOLD, fontSize=19, leading=26,
                             spaceBefore=4, spaceAfter=14),
        "h2": ParagraphStyle("h2", parent=base, fontName=BOLD, fontSize=14, leading=20,
                             spaceBefore=18, spaceAfter=9,
                             textColor=colors.HexColor("#0b3d5c")),
        "h3": ParagraphStyle("h3", parent=base, fontName=BOLD, fontSize=11, leading=17,
                             spaceBefore=13, spaceAfter=6,
                             textColor=colors.HexColor("#17527a")),
        "code": ParagraphStyle("code", parent=base, fontName=MONO, fontSize=8,
                               leading=11.5, spaceBefore=3, spaceAfter=9,
                               leftIndent=7, textColor=colors.HexColor("#17303d")),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.3, leading=12,
                               spaceAfter=0),
        "cellhead": ParagraphStyle("cellhead", parent=base, fontName=BOLD, fontSize=8.3,
                                   leading=12, spaceAfter=0),
    }


def inline(text: str) -> str:
    """Markdown inline syntax to reportlab's mini-HTML. Escapes first, then adds tags."""
    out = html.escape(safe(text, BODY), quote=False)
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)          # links: keep the text
    out = re.sub(r"`([^`]+)`", rf'<font face="{MONO}" size="8.3">\1</font>', out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    return out


def table_flowable(rows: list[list[str]], style: dict, width: float) -> Table:
    header, body = rows[0], rows[1:]
    # Column widths from the longest cell, so a column of numbers does not get the same
    # share as a column of prose. Clamped so one long cell cannot starve the others.
    weights = []
    for column in range(len(header)):
        longest = max((len(r[column]) for r in rows if column < len(r)), default=1)
        weights.append(min(max(longest, 6), 60))
    total = float(sum(weights))
    widths = [width * w / total for w in weights]

    data = [[Paragraph(inline(c), style["cellhead"]) for c in header]]
    data += [[Paragraph(inline(c), style["cell"]) for c in row] for row in body]
    table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f7")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#9fb6c6")),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#dde5eb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def convert(source: str, width: float) -> list:
    style = styles()
    flow: list = []
    paragraph: list[str] = []
    bullets: list[str] = []
    code: list[str] | None = None
    table: list[list[str]] | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            flow.append(Paragraph(inline(" ".join(paragraph)), style["body"]))
            paragraph = []

    def flush_bullets() -> None:
        nonlocal bullets
        if bullets:
            flow.append(ListFlowable(
                [ListItem(Paragraph(inline(b), style["body"]), leftIndent=13)
                 for b in bullets],
                bulletType="bullet", bulletFontName=BODY, bulletFontSize=7,
                leftIndent=13, spaceAfter=7))
            bullets = []

    def flush_table() -> None:
        nonlocal table
        if table:
            flow.append(table_flowable(table, style, width))
            flow.append(Spacer(1, 8))
            table = None

    for raw in source.split("\n"):
        line = raw.rstrip()

        if line.strip().startswith("```"):
            if code is None:
                flush_paragraph(); flush_bullets(); flush_table()
                code = []
            else:
                flow.append(Preformatted(safe("\n".join(code), MONO), style["code"]))
                code = None
            continue
        if code is not None:
            code.append(line)
            continue

        if line.startswith("|"):
            flush_paragraph(); flush_bullets()
            # Split on UNESCAPED pipes only. A table cell containing |S| is written
            # \| in Markdown, and splitting on every pipe turned that row into two extra
            # cells and pushed the rest of the table off the page. The backslash itself
            # must then go: Malgun Gothic draws U+005C as the won sign, which is correct
            # for a Korean font and wrong for a reader.
            body = re.sub(r"^\||\|$", "", line.strip())
            cells = [c.strip().replace("\\|", "|").replace("\\", "")
                     for c in re.split(r"(?<!\\)\|", body)]
            if all(set(c) <= set("-: ") and c for c in cells):   # the |---|---| rule
                continue
            table = (table or []) + [cells]
            continue
        flush_table()

        if not line.strip():
            flush_paragraph(); flush_bullets()
            continue

        if line.startswith("#"):
            flush_paragraph(); flush_bullets()
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("#").strip()
            if level == 1 and flow:
                flow.append(PageBreak())
            flow.append(Paragraph(inline(text), style[f"h{min(level, 3)}"]))
            continue

        if set(line.strip()) == {"-"} and len(line.strip()) >= 3:
            flush_paragraph(); flush_bullets()
            flow.append(Spacer(1, 3))
            flow.append(HRFlowable(width="100%", thickness=0.6,
                                   color=colors.HexColor("#c8d4dd")))
            flow.append(Spacer(1, 7))
            continue

        if line.startswith("- "):
            flush_paragraph()
            bullets.append(line[2:].strip())
            continue
        if bullets and raw.startswith("  "):          # a bullet's continuation line
            bullets[-1] += " " + line.strip()
            continue

        flush_bullets()
        paragraph.append(line.strip())

    flush_paragraph(); flush_bullets(); flush_table()
    return flow


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont(BODY, 7.5)
    canvas.setFillColor(colors.HexColor("#8a9aa6"))
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, str(document.page))
    canvas.restoreState()


def main() -> None:
    register_fonts()
    source_path, out_path = sys.argv[1], sys.argv[2]
    with open(source_path, encoding="utf-8") as handle:
        source = handle.read()

    margin = 20 * mm
    document = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=margin,
                                 rightMargin=margin, topMargin=18 * mm,
                                 bottomMargin=18 * mm, title="scPKR", author="")
    width = A4[0] - 2 * margin
    document.build(convert(source, width), onFirstPage=footer, onLaterPages=footer)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
