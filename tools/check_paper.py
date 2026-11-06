"""Quality gate for a research paper built with tools/md2docx.py. It inspects the delivered files, not the intentions behind them.

    python tools/check_paper.py              # prints a table, writes reports/Paper_Quality_Report.md, exit code = number of failed checks

Each check maps to a requirement in the paper's brief, so that the paper cannot quietly drift back to 30 pages, lose a caption or clip a table.
To use it for another paper, build a different Spec and call run_checks(spec).
"""
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE_FILE = re.compile(r"\b[\w./-]+\.(?:py|ipynb|csv|gz|json|txt|xlsx|md)\b")
CODE_LIKE = re.compile(r"`|```|\bimport \w+|\bdef \w+\(|\bpd\.\w+|\bnp\.\w+|\w+\(\)|\w+_\w+\(")
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
ABBREV = re.compile(r"\b(?:et al|vs|e\.g|i\.e|U\.S\.C|No|Fig)\.")


@dataclass
class Spec:
    """What this paper must satisfy. Everything specific to the UAC paper lives here."""
    paper: Path = ROOT / "Research_Paper_UAC_Forecasting"
    appendix: Path = ROOT / "Research_Paper_Appendix"
    max_pages: int = 15
    required_sections: tuple = ("Abstract", "1. Introduction", "2. Data and Data Quality", "3. Exploratory Data Analysis", "4. Feature Engineering",
                                "5. Modelling and Validation", "6. Results", "7. Outlook and Scenarios", "8. Pipeline analytics", "9. Limitations",
                                "10. Actionable Insights and Recommendations", "11. Conclusion", "References", "Appendix")
    methodology_sections: tuple = (2, 3, 4, 5, 6, 7, 8)                # technical sections that need an N.1 Methodology subsection
    methodology_topics: tuple = ("Schema checks", "Change points", "direct multi-horizon", "walk-forward", "Hyperparameter|hyperparameter", "AIC", "MAE, RMSE", "Champions were selected", "Diebold")
    unnumbered: tuple = ("Abstract", "References", "Appendix")
    banned_phrases: tuple = ("Companion", "version 2", "Version 2", "new in version", "this paper does not restate",
                             "policy shock", "policy-break", "policy shift", "policy-driven")   # policy is not in the data, so no finding may be attributed to it
    figure_dpi: tuple = (270, 330)                                     # figures are drawn at print size at 300 dpi, so type stays at its drawn size
    max_blank_band: float = 0.25                                       # largest empty band allowed on any page but the last, as a share of page height
    max_mean_sentence: float = 17.0
    max_long_sentence_share: float = 0.08                              # share of sentences over 30 words
    near_duplicate: float = 0.82
    provenance: tuple = (ROOT / "archive" / "v2.3_full_paper_2026-11-05" / "Research_Paper_UAC_Forecasting.md", ROOT / "notebook_source" / "UAC_Care_Load_Forecasting_v2.py",
                         ROOT / "reports" / "pipeline_analysis.md", ROOT / "artifacts" / "metadata.json")
    provenance_globs: tuple = ("artifacts/*.csv", "reports/pipeline_*.csv")
    key_findings: tuple = ("251 CBP reports", "0.51", "0.90", "2 Friday reports", "Five reports repeat", "about 95%", "about 36 to about 203", "about 3 children a day", "98.6%",
                           "32 children", "43% below", "28.5 children", "−49%", "about 2.5", "all 25 surge events", "**14-day**", "66–71%", "2.88%", "0.52%", "0.44%",
                           "from 1.32 to 2.96 days", "on 44% of current-regime reports", "2,542", "2,486–2,694", "placement capacity", "+323", "Status at the last report (21 Dec 2025): Alert")
    key_caveats: tuple = ("not a measured custody time", "indicative only", "not evidence of non-compliance", "associations, not causes", "development run", "hypothetical assumption",
                          "inferred from stock changes", "rolling 120-day damped-trend ETS", "illustrative", "not independent surges", "Importance shows what a model uses", "as it currently behaves")
    corrected_numbers: dict = field(default_factory=lambda: {"41.9": 41.945})    # printed value -> artifact value; the original paper misprinted this RMSE as 42.0
    n_recommendations: int = 8
    appendix_letters: str = "ABCDEFGH"
    source_files: tuple = ("notebook_source/UAC_Care_Load_Forecasting_v2.py", "uac_forecast_dashboard.py", "src/data_loader.py", "src/kpi_calculations.py",
                           "src/bottleneck_detection.py", "src/visualizations.py", "tools/pipeline_analysis.py", "tools/paper_figures.py")
    extras: dict = field(default_factory=dict)


@dataclass
class Result:
    id: str
    requirement: str
    ok: bool
    detail: str


# ----------------------------------------------------------------------------- reading the delivered files
def body_blocks(docx_path):
    """The document body in order, as (kind, payload) tuples: ('p', Paragraph) or ('t', Table)."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    doc = Document(docx_path)
    blocks = []
    for el in doc.element.body.iterchildren():
        if el.tag.endswith("}p"):
            blocks.append(("p", Paragraph(el, doc)))
        elif el.tag.endswith("}tbl"):
            blocks.append(("t", Table(el, doc)))
    return doc, blocks


def pictures(par):
    return par._p.xpath(".//*[local-name()='inline']")


def main_body(blocks):
    """Blocks before the Appendix pointer heading."""
    for i, (kind, b) in enumerate(blocks):
        if kind == "p" and b.style.name == "Heading 1" and b.text.strip().startswith("Appendix"):
            return blocks[:i]
    return blocks


def md_sections(md):
    """(prose paragraphs and list items, whole text before References) from the Markdown source."""
    head = md.split("\n## References")[0]
    prose = []
    for ln in head.splitlines():
        s = ln.strip()
        if not s or s[0] in "#|!" or s.startswith("**Table") or s.startswith("<!--"):
            continue
        prose.append(re.sub(r"^(?:[-*]|\d+\.)\s+", "", s))
    return prose, head


def sentences(prose):
    out = []
    for p in prose:
        text = re.sub(r"[*`]", "", p)
        text = ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), text)
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(0-9])", text):
            s = s.replace("\x00", ".").strip()
            if s:
                out.append(s)
    return out


def word_count(s):
    return len(re.findall(r"\S+", s))


# ----------------------------------------------------------------------------- checks on the PDF
def check_pdf(spec, pdf_path):
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(pdf_path))
    n = len(pdf)
    results = [Result("C01", f"Paper is at most {spec.max_pages} pages", n <= spec.max_pages, f"{n} pages")]

    missing = [i + 1 for i in range(n) if not re.search(rf"Page\s+{i + 1}\s+of\s+{n}\b", pdf[i].get_textpage().get_text_range())]
    results.append(Result("C02", "Every page shows a visible page number", not missing, f"missing on pages {missing}" if missing else f"'Page i of {n}' on all {n} pages"))

    clipped, blank = [], []
    for i in range(n):
        page = pdf[i]
        w, h = page.get_size()
        for obj in page.get_objects():
            left, bottom, right, top = obj.get_bounds()
            if left < -1 or bottom < -1 or right > w + 1 or top > h + 1:
                clipped.append(i + 1)
                break
        grey = page.render(scale=0.5).to_pil().convert("L")
        rows = [min(grey.crop((0, y, grey.width, y + 1)).getdata()) < 245 for y in range(grey.height)]
        top_margin, bottom_margin = int(0.06 * grey.height), int(0.93 * grey.height)     # footer excluded
        band = best = 0
        for y in range(top_margin, bottom_margin):
            band = 0 if rows[y] else band + 1
            best = max(best, band)
        if i < n - 1 and best / grey.height > spec.max_blank_band:
            blank.append(f"p{i + 1}: {best / grey.height:.0%}")
    results.append(Result("C03", "Nothing is clipped at the page edge", not clipped, f"objects off the page on {sorted(set(clipped))}" if clipped else "all page objects are inside the page"))
    results.append(Result("C04", f"No large blank areas (over {spec.max_blank_band:.0%} of a page, last page excepted)", not blank, "; ".join(blank) if blank else "all pages are well filled"))
    return results


# ----------------------------------------------------------------------------- checks on the Word document
def check_structure(spec, doc, blocks):
    results = []
    heads = [(b.style.name, b.text.strip()) for k, b in blocks if k == "p" and b.style.name in ("Title", "Heading 1", "Heading 2", "Heading 3")]
    problems = []
    if not heads or heads[0][0] != "Title":
        problems.append("the document does not start with its title")
    expected_h1, current, expected_h2 = 1, None, 1
    for style, text in heads[1:]:
        if style == "Heading 1":
            current, expected_h2 = None, 1
            if any(text.startswith(u) for u in spec.unnumbered):
                continue
            m = re.match(r"^(\d+)\.\s", text)
            if not m or int(m.group(1)) != expected_h1:
                problems.append(f"'{text}' should be section {expected_h1}")
            else:
                current = expected_h1
                expected_h1 += 1
        elif style == "Heading 2":
            m = re.match(r"^(\d+)\.(\d+)\s", text)
            if current is None or not m or int(m.group(1)) != current or int(m.group(2)) != expected_h2:
                problems.append(f"'{text}' breaks the numbering under section {current}")
            else:
                expected_h2 += 1
    results.append(Result("C05", "Headings are hierarchical and numbered in sequence", not problems, "; ".join(problems) if problems else f"{len(heads)} headings"))

    titles = [t for s, t in heads if s == "Heading 1"]
    pos, missing = 0, []
    for req in spec.required_sections:
        found = next((i for i in range(pos, len(titles)) if titles[i].startswith(req)), None)
        if found is None:
            missing.append(req)
        else:
            pos = found + 1
    results.append(Result("C06", "All required sections are present, in order, through Actionable Insights", not missing, f"missing or out of order: {missing}" if missing else f"{len(spec.required_sections)} sections in order"))

    no_method, no_intro = [], []
    for n in spec.methodology_sections:
        if not any(s == "Heading 2" and re.match(rf"^{n}\.1 Methodology", t) for s, t in heads):
            no_method.append(n)
    for i, (k, b) in enumerate(blocks):
        if k == "p" and b.style.name == "Heading 1" and re.match(r"^\d+\.", b.text):
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if nxt is None or nxt[0] != "p" or nxt[1].style.name.startswith("Heading") or not nxt[1].text.strip():
                no_intro.append(b.text)
    results.append(Result("C07", "Each technical section has a Methodology subsection", not no_method, f"none in sections {no_method}" if no_method else f"sections {list(spec.methodology_sections)}"))
    results.append(Result("C08", "Each numbered section opens with a short introduction", not no_intro, f"no introduction: {no_intro}" if no_intro else "all numbered sections"))

    text_width = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
    results += check_tables(blocks, text_width) + check_figures(spec, doc, blocks, text_width)
    return results


def check_tables(blocks, text_width, prefix=r"", cid="C09"):
    problems, numbers, seen, wide = [], [], set(), []
    for i, (k, b) in enumerate(blocks):
        if k != "t":
            continue
        prev = blocks[i - 1] if i else None
        cap = prev[1].text.strip() if prev and prev[0] == "p" else ""
        m = re.match(rf"^Table ({prefix}\d+)\.\s*(\S.*)$", cap)
        if not m:
            problems.append(f"table at block {i} has no caption directly above it (found '{cap[:40]}')")
            continue
        numbers.append(m.group(1))
        if m.group(2) in seen:
            problems.append(f"duplicate title '{m.group(2)}'")
        seen.add(m.group(2))
        grid = sum(int(g.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}w") or 0) for g in b._tbl.tblGrid)
        if grid * 635 > text_width * 1.003:
            wide.append(m.group(1))
    if not prefix:
        if [int(x) for x in numbers] != list(range(1, len(numbers) + 1)):
            problems.append(f"numbers are not sequential: {numbers}")
    if wide:
        problems.append(f"wider than the text area: tables {wide}")
    orphan = [b.text[:30] for j, (k, b) in enumerate(blocks) if k == "p" and re.match(rf"^Table {prefix}\d+\.", b.text.strip()) and b.style.name == "Caption" and (j + 1 >= len(blocks) or blocks[j + 1][0] != "t")]
    if orphan:
        problems.append(f"captions without a table: {orphan}")
    return [Result(cid, "Every table has a unique number and title, in order, and fits the page", not problems, "; ".join(problems) if problems else f"{len(numbers)} tables")]


def check_figures(spec, doc, blocks, text_width, prefix=r"", cid="C10", dpi_check=True):
    from PIL import Image
    problems, numbers, off_dpi, no_alt = [], [], [], []
    for i, (k, b) in enumerate(blocks):
        if k != "p":
            continue
        for inl in pictures(b):
            nxt = blocks[i + 1][1].text.strip() if i + 1 < len(blocks) and blocks[i + 1][0] == "p" else ""
            m = re.match(rf"^Figure ({prefix}\d+)\.\s*\S", nxt)
            if not m:
                problems.append(f"image at block {i} has no caption below it")
                continue
            numbers.append(m.group(1))
            cx = int(inl.xpath(".//*[local-name()='extent']")[0].get("cx"))
            if cx > text_width * 1.002:
                problems.append(f"Figure {m.group(1)} is wider than the text area")
            if not inl.xpath(".//*[local-name()='docPr']")[0].get("descr"):
                no_alt.append(m.group(1))
            rid = inl.xpath(".//*[local-name()='blip']")[0].get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
            px = Image.open(BytesIO(doc.part.related_parts[rid].blob)).size[0]
            dpi = px / (cx / 914400)
            if dpi_check and not spec.figure_dpi[0] <= dpi <= spec.figure_dpi[1]:
                off_dpi.append(f"{m.group(1)}: {dpi:.0f} dpi")
    if not prefix and [int(x) for x in numbers] != list(range(1, len(numbers) + 1)):
        problems.append(f"numbers are not sequential: {numbers}")
    if off_dpi:
        problems.append(f"not placed at print size, so labels shrink: {off_dpi}")
    if no_alt:
        problems.append(f"no alt text: {no_alt}")
    return [Result(cid, "Every figure has a unique number and caption, at print size, inside the margins", not problems, "; ".join(problems) if problems else f"{len(numbers)} figures, effective resolution within {spec.figure_dpi} dpi")]


def check_cross_references(blocks):
    body = main_body(blocks)
    tables = [m.group(1) for k, b in body if k == "p" for m in [re.match(r"^Table (\d+)\.", b.text.strip())] if m and b.style.name == "Caption"]
    figures = [m.group(1) for k, b in body if k == "p" for m in [re.match(r"^Figure (\d+)\.", b.text.strip())] if m and b.style.name == "Caption"]
    text = []
    for k, b in body:
        if k == "p" and b.style.name != "Caption":
            text.append(b.text)
        elif k == "t":
            text += [c.text for r in b.rows for c in r.cells]
    joined = "\n".join(text)
    t_refs, f_refs = set(re.findall(r"\bTables? (\d+)", joined)), set(re.findall(r"\bFigures? (\d+)", joined))
    unref = [f"Table {t}" for t in tables if t not in t_refs] + [f"Figure {f}" for f in figures if f not in f_refs]
    broken = [f"Table {t}" for t in t_refs - set(tables)] + [f"Figure {f}" for f in f_refs - set(figures)]
    heads = {m.group(1) for k, b in body if k == "p" and b.style.name.startswith("Heading") for m in [re.match(r"^(\d+(?:\.\d+)?)", b.text.strip())] if m}
    dead = sorted({s for s in re.findall(r"\bSections? (\d+(?:\.\d+)?)", joined) if s not in heads})
    detail = "; ".join(x for x in (f"never cited in the text: {unref}" if unref else "", f"point to nothing: {broken}" if broken else "", f"dead section references: {dead}" if dead else "") if x)
    return [Result("C11", "Every table and figure is cited in the text; every cross-reference resolves", not detail, detail or f"{len(tables)} tables and {len(figures)} figures cited")]


def check_no_code(spec, md, blocks):
    _, head = md_sections(md)
    prose = "\n".join(ln for ln in head.splitlines() if not ln.startswith("!["))
    problems = []
    files = sorted(set(CODE_FILE.findall(prose)) | {m.group(0) for m in CODE_FILE.finditer(prose)})
    if files:
        problems.append(f"file names: {files}")
    code = sorted({m.group(0) for m in CODE_LIKE.finditer(prose)})
    if code:
        problems.append(f"code-like text: {code}")
    consolas = sum(1 for k, b in main_body(blocks) if k == "p" for r in b.runs if r.font.name == "Consolas")
    if consolas:
        problems.append(f"{consolas} runs set in a code font")
    banned = [w for w in spec.banned_phrases if w in head]
    if banned:
        problems.append(f"removed material is back: {banned}")
    return [Result("C12", "No raw code, file names or removed sections in the main body", not problems, "; ".join(problems) if problems else "clean")]


def check_writing(spec, md):
    prose, _ = md_sections(md)
    sents = sentences(prose)
    lengths = [word_count(s) for s in sents]
    mean = sum(lengths) / len(lengths)
    long_share = sum(n > 30 for n in lengths) / len(lengths)
    ok = mean <= spec.max_mean_sentence and long_share <= spec.max_long_sentence_share
    results = [Result("C13", "Sentences are short and simple", ok, f"{len(sents)} sentences, mean {mean:.1f} words (limit {spec.max_mean_sentence}), {long_share:.1%} over 30 words (limit {spec.max_long_sentence_share:.0%})")]

    norm = [(s, re.sub(r"\W+", " ", s.lower()).strip()) for s in sents if word_count(s) >= 8]
    dup = []
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            if abs(len(norm[i][1]) - len(norm[j][1])) < 0.3 * len(norm[i][1]) and SequenceMatcher(None, norm[i][1], norm[j][1]).ratio() >= spec.near_duplicate:
                dup.append(norm[i][0][:70])
    results.append(Result("C14", "No repeated sentences", not dup, f"near-duplicates: {dup}" if dup else f"{len(norm)} sentences compared pairwise"))
    return results


def citations(md):
    head, _, tail = md.partition("\n## References")
    refs_block = tail.split("\n## ")[0]
    refs = {}
    for ln in refs_block.splitlines():
        m = re.match(r"^-\s+(.+?),\s.*?\((\d{4})\)", ln)
        if m:
            refs[(m.group(1).strip(), m.group(2))] = ln
    cited = set()
    names = r"([A-Z][\wèéáí-]+(?: [A-Z][\wèéáí-]+)?)"
    for group in re.findall(r"\(([^()]*\b\d{4}[^()]*)\)", head):
        for part in group.split(";"):
            m = re.match(rf"^\s*{names}(?: et al\.| & [^,]+)?,\s*(\d{{4}})\s*$", part)
            if m:
                cited.add((m.group(1), m.group(2)))
    return cited, set(refs)


def check_citations(md):
    cited, refs = citations(md)
    missing, unused = sorted(cited - refs), sorted(refs - cited)
    detail = "; ".join(x for x in (f"cited, not listed: {missing}" if missing else "", f"listed, never cited: {unused}" if unused else "") if x)
    return [Result("C15", "Citations and references match both ways", not detail, detail or f"{len(refs)} references, all cited")]


def check_provenance(spec, md):
    corpus = set()
    paths = list(spec.provenance)
    for pattern in spec.provenance_globs:
        paths += sorted(ROOT.glob(pattern))
    for p in paths:
        corpus |= {t.replace(",", "") for t in NUMBER.findall(p.read_text(encoding="utf-8", errors="replace"))}
    _, head = md_sections(md)
    head = re.sub(r"\((?:[^()]*\b(?:19|20)\d{2}[^()]*)\)", " ", head)              # citations
    head = re.sub(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?", " ", head)                   # figure captions and paths
    head = re.sub(r"\*\*Table \d+\.\*\*", " ", head)
    head = re.sub(r"\b(?:Tables?|Figures?|Sections?) \d+(?:\.\d+)?", " ", head)
    head = re.sub(r"^#+ .*$", " ", head, flags=re.M)
    head = re.sub(r"^\d+\.\s", " ", head, flags=re.M)
    tokens = {t.replace(",", "").rstrip(".") for t in NUMBER.findall(head)}
    values = {float(t) for t in corpus if t.replace(".", "", 1).isdigit()}
    backed = {t for t, v in spec.corrected_numbers.items() if any(abs(x - v) < 0.001 for x in values) and float(t) == round(v, len(t.split(".")[-1]))}
    unknown = sorted(t for t in tokens if t not in corpus and t not in backed)
    return [Result("C16", "Every number in the paper traces to the original paper, notebook or artifacts", not unknown, f"no source for: {unknown}" if unknown else f"{len(tokens)} distinct numbers traced to {len(paths)} source files")]


def check_retention(spec, md):
    missing = [k for k in spec.key_findings + spec.key_caveats if k not in md]
    m = re.search(r"\*\*Table \d+\.\*\* Insights and recommended actions\n\n\|[^\n]*\n\|[^\n]*\n((?:\|[^\n]*\n)+)", md)
    n_rec = len(m.group(1).strip().splitlines()) if m else 0
    if n_rec != spec.n_recommendations:
        missing.append(f"{spec.n_recommendations} recommendations (found {n_rec})")
    return [Result("C17", "Key findings, caveats and all recommendations are retained", not missing, f"missing: {missing}" if missing else f"{len(spec.key_findings)} findings, {len(spec.key_caveats)} caveats and {n_rec} recommendations present")]


def check_appendix(spec):
    from docx import Document  # noqa: F401
    path, results = spec.appendix, []
    if not (path.with_suffix(".docx").exists() and path.with_suffix(".pdf").exists()):
        return [Result("C18", "Appendix volume holds definitions, logs, outputs and all raw code", False, "Appendix volume not built")]
    doc, blocks = body_blocks(path.with_suffix(".docx"))
    heads = [b.text for k, b in blocks if k == "p" and b.style.name == "Heading 1"]
    letters = "".join(m.group(1) for h in heads for m in [re.match(r"^Appendix ([A-Z])\.", h)] if m)
    problems = [] if letters == spec.appendix_letters else [f"parts found: {letters}"]
    code_text = {b.text.rstrip() for k, b in blocks if k == "p" and any(r.font.name == "Consolas" for r in b.runs)}
    for src in spec.source_files:
        lines = [ln.rstrip() for ln in (ROOT / src).read_text(encoding="utf-8").splitlines() if ln.strip()]
        absent = [ln for ln in lines if ln not in code_text]
        if absent:
            problems.append(f"{src}: {len(absent)} of {len(lines)} lines missing or out of date")
    width = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
    t = check_tables(blocks, width * 1.5, prefix=r"[A-H]", cid="x")[0]                 # landscape sections are wider
    f = check_figures(spec, doc, blocks, width, prefix=r"[A-H]", cid="y", dpi_check=False)[0]
    if not t.ok:
        problems.append(t.detail)
    if not f.ok:
        problems.append(f.detail)
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path.with_suffix(".pdf")))
    n = len(pdf)
    no_number = [i + 1 for i in range(n) if not re.search(rf"Page\s+{i + 1}\s+of\s+{n}\b", pdf[i].get_textpage().get_text_range())]
    if no_number:
        problems.append(f"no page number on pages {no_number[:10]}")
    results.append(Result("C18", "Appendix volume holds definitions, logs, outputs and all raw code", not problems, "; ".join(problems) if problems else f"parts {letters}; {n} pages; {t.detail}; {f.detail}; all {len(spec.source_files)} source files complete and current"))
    return results


def _norm(text):
    """Compare text by letters and digits only, so line breaks, punctuation and ligatures (NFKC) do not matter."""
    import unicodedata
    return re.sub(r"[^0-9a-z]+", "", unicodedata.normalize("NFKC", text).lower())


def _md_units(md):
    """Every heading, paragraph, list item, table cell and figure caption of a Markdown file, as plain text."""
    units, heads = [], []
    for ln in md.splitlines():
        s = ln.strip()
        if not s or s.startswith(("<!--", "```")) or re.fullmatch(r"\|[-:| ]+\|", s) or s == "---":
            continue
        fig = re.match(r"^!\[([^\]]*)\]", s)
        if fig:
            units.append(fig.group(1))
            heads.append(fig.group(1))
            continue
        if s.startswith("|"):
            units += [c for c in re.split(r"(?<!\\)\|", s.strip("|"))]
            continue
        if s.startswith("#"):
            heads.append(s.lstrip("#"))
        units.append(re.sub(r"^(#+|[-*]|\d+\.)\s+", "", s))
    clean = lambda t: re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t).replace("**", "").replace("`", "").replace("*", "")
    return [clean(u) for u in units if _norm(clean(u))], [clean(h) for h in heads]


def in_step(md, docx, pdf=None, root=ROOT):
    """Problems that show a Word or PDF file was not rebuilt from the current Markdown and figures (empty list = in step).

    Content-based on purpose: file timestamps are meaningless after a git clone, which writes every file at checkout time."""
    import hashlib
    problems = []
    doc, blocks = body_blocks(docx)
    parts = []
    for k, b in blocks:
        parts.append(b.text if k == "p" else " ".join(c.text for r in b.rows for c in r.cells))
    word = _norm(" ".join(parts))
    units, heads = _md_units(md)
    missing = [u for u in units if _norm(u) not in word]
    if missing:
        problems.append(f"{len(missing)} of {len(units)} text units are not in the Word file, e.g. '{missing[0][:60]}'")
    blobs = {hashlib.sha256(part.blob).hexdigest() for part in doc.part.related_parts.values() if part.content_type.startswith("image/")}
    for ref in re.findall(r"\]\(([^)]+\.png)\)", md):
        if hashlib.sha256((root / ref).read_bytes()).hexdigest() not in blobs:
            problems.append(f"{ref} differs from the image in the Word file")
    if pdf is not None:
        import pypdfium2 as pdfium
        if not Path(pdf).exists():
            problems.append("no PDF")
        else:
            d = pdfium.PdfDocument(str(pdf))
            text = _norm(" ".join(d[i].get_textpage().get_text_range() for i in range(len(d))))
            absent = [h for h in heads if _norm(h) not in text]
            if absent:
                problems.append(f"{len(absent)} headings or captions are not in the PDF, e.g. '{absent[0][:60]}'")
    return problems


def check_fresh(spec):
    md = spec.paper.with_suffix(".md").read_text(encoding="utf-8")
    problems = in_step(md, spec.paper.with_suffix(".docx"), spec.paper.with_suffix(".pdf"))
    units, heads = _md_units(md)
    return [Result("C19", "Delivered Word and PDF files are in step with their sources", not problems,
                   "; ".join(problems) if problems else f"all {len(units)} text units in the Word file, every figure identical, all {len(heads)} headings and captions in the PDF")]


def run_checks(spec=None):
    spec = spec or Spec()
    md = spec.paper.with_suffix(".md").read_text(encoding="utf-8")
    doc, blocks = body_blocks(spec.paper.with_suffix(".docx"))
    results = check_pdf(spec, spec.paper.with_suffix(".pdf")) + check_structure(spec, doc, blocks) + check_cross_references(blocks) + check_no_code(spec, md, blocks)
    results += check_writing(spec, md) + check_citations(md) + check_provenance(spec, md) + check_retention(spec, md) + check_appendix(spec) + check_fresh(spec)
    return sorted(results, key=lambda r: r.id)


def main():
    results = run_checks()
    lines = ["# Research paper quality report", "", "Generated by `python tools/check_paper.py` from the delivered Word and PDF files.", "", "| Check | Requirement | Result | Detail |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.id} | {r.requirement} | {'PASS' if r.ok else 'FAIL'} | {r.detail.replace('|', '/')} |")
    failed = [r for r in results if not r.ok]
    lines += ["", f"**{len(results) - len(failed)} of {len(results)} checks passed.**"]
    (ROOT / "reports" / "Paper_Quality_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    for r in results:
        print(f"{'PASS' if r.ok else 'FAIL'}  {r.id}  {r.requirement}\n        {r.detail}")
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
