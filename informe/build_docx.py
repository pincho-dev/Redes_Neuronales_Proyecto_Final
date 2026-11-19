"""
Convierte informe.md -> informe.docx (python-docx) para entregar/editar en Word.
Soporta el subconjunto de Markdown usado en el informe: titulos, parrafos con
**negrita** y `codigo`, listas, tablas, imagenes y bloques de codigo.

Uso:  python informe/build_docx.py   (luego: soffice --headless --convert-to pdf informe/informe.docx)
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
SRC = HERE / "informe.md"
OUT = HERE / "informe.docx"

INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|(?<![\w*])\*[^*\s][^*]*\*(?![\w*]))")


def add_inline(par, text, size=None):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            run = par.add_run(part[2:-2].replace("`", "")); run.bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run = par.add_run(part[1:-1]); run.italic = True
        elif part.startswith("`"):
            run = par.add_run(part[1:-1]); run.font.name = "Consolas"
            run.font.size = Pt(9); run.font.color.rgb = RGBColor(0x8B, 0x1A, 0x1A)
        else:
            run = par.add_run(part)
        if size and not part.startswith("`"):
            run.font.size = size


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def add_table(doc, rows):
    header, body = rows[0], rows[2:]
    t = doc.add_table(rows=1 + len(body), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate([header] + body):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = ""
            p = cell.paragraphs[0]
            add_inline(p, val, Pt(9))
            p.paragraph_format.space_after = Pt(0)
            if r == 0:
                for run in p.runs:
                    run.bold = True
                shade(cell, "D9E2F3")
    doc.add_paragraph()


def add_code(doc, lines):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run("\n".join(lines))
    run.font.name = "Consolas"; run.font.size = Pt(8.5)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "F2F2F2")
    pPr.append(shd)


def main() -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(27.94), Cm(21.59)  # carta
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(2.2))
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    st = doc.styles["Normal"]
    st.font.name = "Calibri"; st.font.size = Pt(11)
    st.paragraph_format.space_after = Pt(4)

    lines = SRC.read_text(encoding="utf-8").splitlines()
    i, fig_n = 0, 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            block = []
            i += 1
            while not lines[i].strip().startswith("```"):
                block.append(lines[i]); i += 1
            add_code(doc, block)
        elif s.startswith("# "):
            h = doc.add_heading(s[2:], level=0)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif s.startswith("## "):
            doc.add_heading(s[3:], level=1)
        elif s.startswith("### "):
            doc.add_heading(s[4:], level=2)
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            add_table(doc, rows)
            continue
        elif s.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*)\)", s)
            fig_n += 1
            img = (HERE / m.group(2)).resolve()
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(img), width=Cm(15.5))
            cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cap.add_run(f"Figura {fig_n}. {m.group(1)}"); r.italic = True; r.font.size = Pt(9)
        elif re.match(r"^- ", s):
            add_inline(doc.add_paragraph(style="List Bullet"), s[2:])
        elif re.match(r"^\d+\. ", s):
            add_inline(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", s))
        elif s:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_inline(p, s)
            # Despues de la portada (fecha) pasar a una nueva pagina
            if s.startswith("**Fecha:**"):
                p.add_run().add_break(WD_BREAK.PAGE)
        i += 1
    doc.save(OUT)
    print("Generado", OUT)


if __name__ == "__main__":
    main()
