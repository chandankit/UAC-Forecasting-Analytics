"""Markdown -> professionally formatted Word document.

Usage: python tools/md2docx.py <in.md> <out.docx> [--page A4|Letter] [--body-pt 10.5] [--table-pt 8.5] [--code-pt 7]

Supports headings, paragraphs, bullet and numbered lists (numbering restarts for each list), pipe tables, images, fenced code,
**bold**, *italic* and `code`. Every document gets a "Page X of Y" footer.

Layout rules it applies so a document cannot come out clipped or badly split:
* a paragraph starting "**Table N.**" directly above a table is its caption and stays on the same page as the table;
* `![Figure N. Title](path){width=5.5}` places the image at 5.5 inches (default: full text width), never wider than the text area,
  with the alt text as the caption below it, kept on the same page;
* table columns are sized from their content and never exceed the text width; words never break, rows never split across pages, header rows repeat,
  and tables of up to 18 rows stay on one page;
* a section headed "References" is set as compact hanging-indent entries;
* `<!-- pagebreak -->`, `<!-- landscape -->` and `<!-- portrait -->` control page layout.
* `<!-- toc -->` inserts a table of contents; `<!-- code: path -->` sets a source file verbatim as code at build time.
"""
import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Inches, Pt, RGBColor

INK = RGBColor(0x1F, 0x23, 0x28)
HEAD = RGBColor(0x12, 0x2B, 0x4A)
PAGE = {"A4": (Cm(21.0), Cm(29.7)), "Letter": (Inches(8.5), Inches(11))}
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)")
_VALUE = r"[~<>≥≤+−\-]?\s?\d[\d,.]*\s?(?:%|×|days?|d)?"
NUMERIC = re.compile(rf"^(?:\*\*)?{_VALUE}(?:\s?[–/→]\s?{_VALUE})?(?:\s\([^)]*\))?(?:\*\*)?$")     # a value, a range, or a value with a note
KEEP_WHOLE_MAX_ROWS = 18
COMPACT_STRETCH = 1.3                                         # a table that fits may grow this much at most, then it is centred


class Builder:
    def __init__(self, src, dst, page="A4", body_pt=10.5, table_pt=8.5, code_pt=7.0):
        self.src, self.dst, self.table_pt, self.code_pt = Path(src), Path(dst), table_pt, code_pt
        self.doc = Document()
        self.buf = []
        self.in_refs = False
        self.page = page
        self._num_ids = None
        self._setup(body_pt)

    # ------------------------------------------------------------------ document set-up
    def _setup(self, body_pt):
        d = self.doc
        sec = d.sections[0]
        sec.page_width, sec.page_height = PAGE[self.page]
        sec.left_margin = sec.right_margin = Inches(0.85)
        sec.top_margin, sec.bottom_margin = Inches(0.8), Inches(0.75)
        sec.footer_distance = Inches(0.4)
        normal = d.styles["Normal"]
        normal.font.name, normal.font.size, normal.font.color.rgb = "Calibri", Pt(body_pt), INK
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        normal.paragraph_format.space_after = Pt(5)
        normal.paragraph_format.line_spacing = 1.04
        specs = {"Title": (19, 2, 3), "Heading 1": (13, 11, 4), "Heading 2": (11, 8, 3), "Heading 3": (10.5, 6, 2)}
        for name, (size, before, after) in specs.items():
            st = d.styles[name]
            st.font.name, st.font.size, st.font.bold, st.font.italic, st.font.color.rgb = "Calibri", Pt(size), True, False, HEAD
            rpr = st.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rpr.append(fonts)
            for att in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                fonts.set(qn(att), "Calibri")
            for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                fonts.attrib.pop(qn(att), None)
            st.paragraph_format.space_before, st.paragraph_format.space_after = Pt(before), Pt(after)
            st.paragraph_format.keep_with_next = True
        title = d.styles["Title"].element.pPr
        for b in title.findall(qn("w:pBdr")):                      # the default template draws a rule under the title
            title.remove(b)
        cap = d.styles["Caption"]
        cap.font.name, cap.font.size, cap.font.bold, cap.font.italic, cap.font.color.rgb = "Calibri", Pt(9), False, False, INK
        cap.paragraph_format.space_before, cap.paragraph_format.space_after = Pt(3), Pt(3)
        for name in ("List Bullet", "List Number"):
            d.styles[name].paragraph_format.space_after = Pt(2.5)
        self._footer(sec)
        d.core_properties.author = "Unified Mentor Data Science Internship"

    def _field(self, par, code):
        for kind, text in (("begin", None), (None, code), ("separate", None), ("text", "1"), ("end", None)):
            run = par.add_run()
            run.font.size = Pt(8.5)
            if kind == "text":
                run.text = text
            elif kind:
                fc = OxmlElement("w:fldChar")
                fc.set(qn("w:fldCharType"), kind)
                run._r.append(fc)
            else:
                it = OxmlElement("w:instrText")
                it.set(qn("xml:space"), "preserve")
                it.text = f" {text} "
                run._r.append(it)

    def _footer(self, sec):
        p = sec.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run("Page ").font.size = Pt(8.5)
        self._field(p, "PAGE")
        p.add_run(" of ").font.size = Pt(8.5)
        self._field(p, "NUMPAGES")

    def text_width(self):
        s = self.doc.sections[-1]
        return s.page_width - s.left_margin - s.right_margin

    # ------------------------------------------------------------------ inline text
    def inline(self, par, text, size=None):
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text).replace("\\|", "\x00").replace("\\*", "\x01")
        for part in INLINE.split(text):
            if not part:
                continue
            part = part.replace("\x00", "|")
            if part.startswith("**") and part.endswith("**"):
                run = par.add_run(part[2:-2].replace("\x01", "*"))
                run.bold = True
            elif part.startswith("`") and part.endswith("`"):
                run = par.add_run(part[1:-1].replace("\x01", "*"))
                run.font.name = "Consolas"
                run.font.size = Pt((size or 10.5) - 1)
                continue
            elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                run = par.add_run(part[1:-1].replace("\x01", "*"))
                run.italic = True
            else:
                run = par.add_run(part.replace("\x01", "*"))
            if size:
                run.font.size = Pt(size)

    def flush(self):
        if not self.buf:
            return
        text = " ".join(s.strip() for s in self.buf)
        self.buf = []
        if re.match(r"^\*\*Table [A-Z]?\d+\.\*\*", text):
            p = self.doc.add_paragraph(style="Caption")
            self.inline(p, text, size=9)
            p.paragraph_format.keep_with_next = True
            return
        if self.in_refs:
            self.reference(text)
            return
        self.inline(self.doc.add_paragraph(), text)

    def reference(self, text):
        p = self.doc.add_paragraph()
        p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.22), Inches(-0.22)
        p.paragraph_format.space_after = Pt(1.5)
        p.paragraph_format.line_spacing = 1.0
        self.inline(p, text, size=8.5)

    # ------------------------------------------------------------------ tables
    def col_widths(self, rows, ncol):
        """Column widths from content: no word ever breaks, and a compact table stays compact instead of stretching to the margins."""
        pt = self.table_pt / 72                                   # inches per point of type size
        pad = 0.11                                                # left + right cell padding, inches
        total = self.text_width() / 914400
        plain = [[re.sub(r"[*`]", "", c) for c in r] for r in rows]
        factor = lambda i: 0.56 if i == 0 else 0.50               # conservative character width: bold header / body (Calibri)

        def longest(j, split):
            best = 0.0
            for i, r in enumerate(plain):
                if j < len(r):
                    parts = re.split(split, r[j]) if split else [r[j]]
                    best = max(best, max(len(w) for w in parts) * factor(i) * pt)
            return best + pad

        natural = [longest(j, None) for j in range(ncol)]
        minimum = [longest(j, r"[\s/]+") for j in range(ncol)]
        if natural[0] <= 0.3 * total:                             # row labels stay on one line; long headers and cells wrap first
            minimum[0] = natural[0]
        if sum(natural) <= total:
            scale = min(total / sum(natural), COMPACT_STRETCH)
            widths = [n * scale for n in natural]
        elif sum(minimum) >= total:
            widths = [m * total / sum(minimum) for m in minimum]
        else:
            flex = [n - m for n, m in zip(natural, minimum)]
            room = total - sum(minimum)
            widths = [m + room * f / sum(flex) for m, f in zip(minimum, flex)]
        return [Inches(w) for w in widths]

    def table(self, raw):
        rows = [[c.strip() for c in r.replace("\\|", "\x00").strip().strip("|").split("|")] for r in raw]
        rows = [[c.replace("\x00", "\\|") for c in r] for r in rows if not all(re.fullmatch(r":?-{2,}:?", x.strip()) for x in r)]
        ncol = max(len(r) for r in rows)
        widths = self.col_widths(rows, ncol)
        t = self.doc.add_table(rows=len(rows), cols=ncol)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        tblpr = t._tbl.tblPr
        for tag, val in (("w:tblLayout", {"w:type": "fixed"}),):
            el = OxmlElement(tag)
            for k, v in val.items():
                el.set(qn(k), v)
            tblpr.append(el)
        margins = OxmlElement("w:tblCellMar")
        for side, w in (("left", 55), ("right", 55)):
            m = OxmlElement(f"w:{side}")
            m.set(qn("w:w"), str(w))
            m.set(qn("w:type"), "dxa")
            margins.append(m)
        tblpr.append(margins)
        for j, g in enumerate(t._tbl.tblGrid.findall(qn("w:gridCol"))):
            g.set(qn("w:w"), str(int(widths[j] / 635)))
        def is_numeric(j):
            cells = [r[j] for r in rows[1:] if j < len(r) and r[j].strip() not in ("n/a", "")]    # n/a is neutral
            return j > 0 and bool(cells) and sum(bool(NUMERIC.match(c)) for c in cells) >= 0.6 * len(cells)
        numeric = [is_numeric(j) for j in range(ncol)]
        keep_whole = len(rows) <= KEEP_WHOLE_MAX_ROWS
        for i, row in enumerate(rows):
            tr = t.rows[i]
            trpr = tr._tr.get_or_add_trPr()
            trpr.append(OxmlElement("w:cantSplit"))
            if i == 0:
                trpr.append(OxmlElement("w:tblHeader"))
            for j in range(ncol):
                cell = tr.cells[j]
                cell.width = widths[j]
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                if numeric[j]:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                if keep_whole and i < len(rows) - 1:
                    p.paragraph_format.keep_with_next = True
                self.inline(p, row[j] if j < len(row) else "", size=self.table_pt)
                if i == 0:
                    for r in p.runs:
                        r.bold = True
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), "E8EEF7")
                    cell._tc.get_or_add_tcPr().append(shd)
        spacer = self.doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(0)
        spacer.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        spacer.paragraph_format.line_spacing = Pt(7)

    # ------------------------------------------------------------------ images, code, lists, sections
    def image(self, alt, path, width):
        full = (self.src.parent / path).resolve()
        if not full.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        limit = self.text_width()
        w = min(Inches(width), limit) if width else limit
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.keep_with_next = bool(alt)
        pic = p.add_run().add_picture(str(full), width=Emu(int(w)))
        pic._inline.docPr.set("descr", alt or path)
        if alt:
            c = self.doc.add_paragraph(style="Caption")
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            c.paragraph_format.space_after = Pt(8)
            m = re.match(r"^(Figure \w+\.)\s*(.*)$", alt)
            if m:
                c.add_run(m.group(1)).bold = True
                c.add_run(" " + m.group(2))
            else:
                c.add_run(alt)

    def code(self, lines):
        for line in lines:
            p = self.doc.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.3), Inches(-0.3)
            r = p.add_run(line.rstrip() or " ")
            r.font.name = "Consolas"
            r.font.size = Pt(self.code_pt)

    def contents(self):
        """A table of contents, filled in when Word updates the document's fields (tools/build_paper.py does this)."""
        self.doc.add_paragraph().add_run("Contents").bold = True
        self._field(self.doc.add_paragraph(), r'TOC \o "1-2" \h \z')

    def new_number_list(self):
        """A fresh numbering instance, so each numbered list starts again at 1."""
        numbering = self.doc.part.numbering_part.numbering_definitions._numbering
        base = self.doc.styles["List Number"].element.pPr.numPr.numId.val
        abstract = next(n for n in numbering.findall(qn("w:num")) if n.numId == base).abstractNumId.val
        num = numbering.add_num(abstract)
        lvl = num.add_lvlOverride(ilvl=0)
        lvl.add_startOverride(1)
        return num.numId

    def section(self, landscape):
        s = self.doc.add_section(WD_SECTION.NEW_PAGE)
        wide, tall = max(PAGE[self.page]), min(PAGE[self.page])
        s.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
        s.page_width, s.page_height = (wide, tall) if landscape else (tall, wide)

    # ------------------------------------------------------------------ parser
    def build(self):
        lines = self.src.read_text(encoding="utf-8").splitlines()
        i, title_done, num_id, was_number = 0, False, None, False
        while i < len(lines):
            s = lines[i].strip()
            is_number = bool(re.match(r"^\d+\.\s+", s))
            if not s:
                self.flush()
                i += 1
                continue
            if not is_number:
                was_number = False
            if s.startswith("<!--"):
                self.flush()
                code = re.match(r"<!--\s*code:\s*(.+?)\s*-->", s)
                if code:
                    self.code((self.src.parent / code.group(1)).read_text(encoding="utf-8").splitlines())
                elif "toc" in s:
                    self.contents()
                elif "pagebreak" in s:
                    self.doc.add_page_break()
                elif "landscape" in s:
                    self.section(True)
                elif "portrait" in s:
                    self.section(False)
                i += 1
                continue
            if s.startswith("```"):
                self.flush()
                i += 1
                block = []
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    block.append(lines[i])
                    i += 1
                self.code(block)
                i += 1
                continue
            if s.startswith("|"):
                self.flush()
                raw = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    raw.append(lines[i])
                    i += 1
                self.table(raw)
                continue
            m = re.match(r"^(#{1,4})\s+(.*)", s)
            if m:
                self.flush()
                level, text = len(m.group(1)), m.group(2).replace("**", "")
                if level == 1 and not title_done:
                    self.doc.add_heading(text, level=0)
                    self.doc.core_properties.title = text
                    title_done = True
                else:
                    self.doc.add_heading(text, level=max(1, level - 1))
                self.in_refs = level == 2 and text.strip().lower() == "references"
                i += 1
                continue
            m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)(?:\{width=([\d.]+)\})?", s)
            if m:
                self.flush()
                self.image(m.group(1), m.group(2), float(m.group(3)) if m.group(3) else None)
                i += 1
                continue
            if re.match(r"^[-*]\s+", s):
                self.flush()
                body = re.sub(r"^[-*]\s+", "", s)
                if self.in_refs:
                    self.reference(body)
                else:
                    self.inline(self.doc.add_paragraph(style="List Bullet"), body)
                i += 1
                continue
            if is_number:
                self.flush()
                if not was_number:
                    num_id = self.new_number_list()
                p = self.doc.add_paragraph(style="List Number")
                numpr = p._p.get_or_add_pPr().get_or_add_numPr()
                numpr.get_or_add_numId().val = num_id
                numpr.get_or_add_ilvl().val = 0
                self.inline(p, re.sub(r"^\d+\.\s+", "", s))
                was_number = True
                i += 1
                continue
            if s.startswith(">"):
                self.flush()
                p = self.doc.add_paragraph()
                self.inline(p, s.lstrip("> "))
                p.paragraph_format.left_indent = Inches(0.3)
                for r in p.runs:
                    r.italic = True
                i += 1
                continue
            if s == "---":
                self.flush()
                i += 1
                continue
            self.buf.append(s)
            i += 1
        self.flush()
        self.doc.save(self.dst)
        return self.dst


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--page", default="A4", choices=sorted(PAGE))
    ap.add_argument("--body-pt", type=float, default=10.5)
    ap.add_argument("--table-pt", type=float, default=8.5)
    ap.add_argument("--code-pt", type=float, default=7.0)
    a = ap.parse_args()
    print("wrote", Builder(a.src, a.dst, a.page, a.body_pt, a.table_pt, a.code_pt).build())


if __name__ == "__main__":
    main()
