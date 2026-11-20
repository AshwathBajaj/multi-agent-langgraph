"""Render docs/PROJECT_EXPLAINER.md into docs/PROJECT_EXPLAINER.pdf."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from fpdf import FPDF
from fpdf.fonts import TextStyle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "PROJECT_EXPLAINER.md"
TARGET = ROOT / "docs" / "PROJECT_EXPLAINER.pdf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

NAVY = "#1f4e79"
INK = "#1c1a17"


class ExplainerPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(10)
        self.set_font("DejaVu", size=9)
        self.set_text_color(90, 98, 108)
        self.cell(0, 6, "Briefwright  |  Interview explainer  |  Ashwath Bajaj")
        self.ln(1.5)
        self.set_draw_color(31, 78, 121)
        self.set_line_width(0.4)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.set_y(22)

    def footer(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_font("DejaVu", size=9)
        self.set_text_color(110, 116, 124)
        self.cell(0, 8, str(self.page_no()), align="C")


def _styles() -> dict[str, TextStyle]:
    return {
        "h1": TextStyle(color=NAVY, font_size_pt=18, t_margin=8, b_margin=3, font_style="B"),
        "h2": TextStyle(color=NAVY, font_size_pt=14, t_margin=6, b_margin=2, font_style="B"),
        "h3": TextStyle(color="#245c8a", font_size_pt=12, t_margin=4, b_margin=1.5, font_style="B"),
        "p": TextStyle(font_size_pt=10.5, t_margin=1, b_margin=1.5),
        "li": TextStyle(font_size_pt=10.5, l_margin=4, t_margin=0.6),
        "blockquote": TextStyle(color="#243140", t_margin=2, b_margin=2),
        "code": TextStyle(font_family="Mono", font_size_pt=9, color=INK),
        "pre": TextStyle(
            font_family="Mono",
            font_size_pt=8,
            t_margin=2,
            b_margin=2,
            fill_color=(244, 241, 234),
        ),
    }


def _cover(pdf: ExplainerPDF) -> None:
    pdf.add_page()
    pdf.set_fill_color(31, 78, 121)
    pdf.rect(0, 0, pdf.w, 8, style="F")
    pdf.set_y(48)
    pdf.set_text_color(31, 78, 121)
    pdf.set_font("DejaVu", "B", 28)
    pdf.multi_cell(0, 14, "Briefwright", align="L")
    pdf.ln(2)
    pdf.set_font("DejaVu", size=16)
    pdf.set_text_color(28, 26, 23)
    pdf.multi_cell(0, 9, "A LangGraph research crew, explained for interviews")
    pdf.ln(8)
    pdf.set_draw_color(31, 78, 121)
    pdf.set_line_width(0.6)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.l_margin + 48, y)
    pdf.ln(8)
    pdf.set_font("DejaVu", size=12)
    pdf.set_text_color(40, 46, 54)
    for line in (
        "Ashwath Bajaj",
        "AI and Data Science",
        "RAG, agentic workflows, and LLM guardrails",
        "November 2026",
    ):
        pdf.cell(0, 8, line)
        pdf.ln(7)
    pdf.ln(16)
    pdf.set_font("DejaVu", size=11)
    pdf.set_text_color(60, 66, 74)
    pdf.multi_cell(
        0,
        6,
        "The graph owns the route. A code guardrail owns pass or fail. "
        "The model only writes language. This note is the walk-through I would "
        "give beside the repository.",
    )


def render(source: Path = SOURCE, target: Path = TARGET) -> Path:
    if not source.exists():
        raise SystemExit(f"Missing {source}")
    html = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["fenced_code", "tables", "sane_lists"],
    )
    # fpdf cannot render tags nested inside a table cell.
    html = re.sub(
        r"<td>(.*?)</td>",
        lambda match: "<td>" + re.sub(r"</?code>", "", match.group(1)) + "</td>",
        html,
        flags=re.DOTALL,
    )
    pdf = ExplainerPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 24, 18)
    pdf.add_font("DejaVu", "", SANS)
    pdf.add_font("DejaVu", "B", SANS_BOLD)
    pdf.add_font("Mono", "", MONO)
    pdf.set_font("DejaVu", size=11)
    _cover(pdf)
    pdf.add_page()
    pdf.set_text_color(28, 26, 23)
    pdf.write_html(html, font_family="DejaVu", tag_styles=_styles())
    target.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(target))
    return target


def main() -> int:
    path = render()
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
