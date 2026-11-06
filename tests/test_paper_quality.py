"""The research paper's quality gate (tools/check_paper.py), tested two ways.

1. The delivered paper passes every check.
2. Negative controls: each check must FAIL on a document with that defect. A gate that cannot fail proves nothing.

Run tools/build_paper.py first if a delivered file is out of date.
"""
import os
import sys

import pandas as pd
import pytest
from conftest import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import check_paper as cp                         # noqa: E402
from md2docx import NUMERIC, Builder             # noqa: E402

PAPER_MD = (ROOT / "Research_Paper_UAC_Forecasting.md").read_text(encoding="utf-8")
CHECK_IDS = [f"C{i:02d}" for i in range(1, 20)]
FIGURE = ROOT / "figures" / "paper" / "fig03_length_of_stay.png"      # 6.4 inches wide at 300 dpi


@pytest.fixture(scope="module")
def results():
    return {r.id: r for r in cp.run_checks()}


# ------------------------------------------------------------------ the delivered paper
@pytest.mark.parametrize("cid", CHECK_IDS)
def test_delivered_paper_passes(results, cid):
    assert results[cid].ok, f"{cid} {results[cid].requirement}: {results[cid].detail}"


def test_every_check_ran(results):
    assert sorted(results) == CHECK_IDS


def test_paper_is_within_the_page_limit_with_room_to_spare(results):
    pages = int(results["C01"].detail.split()[0])
    assert pages <= cp.Spec().max_pages


def test_appendix_narrative_attributes_nothing_to_policy():
    """Policy is not in the data. The Markdown holds the narrative only: code is set from the source files at build time."""
    appendix = (ROOT / "Research_Paper_Appendix.md").read_text(encoding="utf-8")
    assert not [w for w in ("policy shock", "policy-break", "policy shift", "policy-driven") if w in appendix]


def test_every_notebook_chart_is_in_the_appendix_volume():
    appendix = (ROOT / "Research_Paper_Appendix.md").read_text(encoding="utf-8")
    for png in (ROOT / "figures").glob("*.png"):
        assert f"figures/{png.name}" in appendix, png.name


# ------------------------------------------------------------------ the paper's tables against the artifacts
def paper_table(number):
    """Rows of a numbered table in the Markdown, with bold, thousands commas and typographic minus removed."""
    lines = PAPER_MD.split(f"**Table {number}.**")[1].split("\n\n")[1].splitlines()
    clean = lambda c: c.strip().replace("**", "").replace(",", "").replace("−", "-")
    return [[clean(c) for c in ln.strip().strip("|").split("|")] for ln in lines[2:]]


def close(printed, actual, decimals):
    """A printed value must be the rounding of the artifact value (half a unit in the last place, plus float noise)."""
    return abs(float(printed.replace("%", "").split()[0]) - actual) <= 0.5 * 10 ** -decimals + 1e-6


def test_table_9_matches_the_metrics_artifact():
    m = pd.read_csv(ROOT / "artifacts" / "metrics_overall.csv", encoding="utf-8")
    m = m[m["period"] == "test"].assign(key=lambda x: x["model"].str.lower()).set_index(["target", "key"])
    rows = paper_table(9)
    assert len(rows) == 8
    spec = [("hhs_care", "MAE", 1), ("hhs_care", "RMSE", 1), ("hhs_care", "MAPE %", 2), ("hhs_care", "Accuracy %", 1), ("hhs_care", "Bias", 1),
            ("discharged", "MAE", 2), ("discharged", "Accuracy %", 1), ("transferred", "MAE", 2)]
    for row in rows:
        for cell, (target, col, dec) in zip(row[1:], spec):
            assert close(cell, m.at[(target, row[0].lower()), col], dec), (row[0], target, col, cell)


def test_table_12_matches_the_interval_artifact():
    c = pd.read_csv(ROOT / "artifacts" / "interval_method_comparison.csv", encoding="utf-8").set_index(["target", "model", "method"])["coverage80"]
    series = {"Care load SARIMA": ("hhs_care", "SARIMA"), "Discharges Ensemble": ("discharged", "Ensemble"),      # paper_table removes commas
              "Transfers Moving Avg (champion)": ("transferred", "Moving Avg (7d)"), "Transfers Ensemble": ("transferred", "Ensemble")}
    rows = paper_table(12)
    assert len(rows) == len(series)
    for row in rows:
        target, model = series[row[0]]
        for cell, method in zip(row[1:], ("fixed calibration (13 wk)", "rolling origin (26 wk)", "rolling origin (13 wk)")):
            assert close(cell, c[(target, model, method)] * 100, 1), (row[0], method, cell)


def test_table_14_matches_the_early_warning_artifact():
    o = pd.read_csv(ROOT / "artifacts" / "ews_operating_curve.csv", encoding="utf-8")
    o = o.rename(columns={o.columns[0]: "p"})
    rows = paper_table(14)
    assert len(rows) == 5
    for row in rows:
        p = float(row[0].split("%")[0]) / 100
        for offset, source in ((1, "replay"), (4, "test")):
            r = o[(o["p"] == p) & o["source"].str.startswith(source)].iloc[0]
            assert close(row[offset], r["detected %"], 0) and close(row[offset + 1], r["median lead (days)"], 1) and close(row[offset + 2], r["false alarms %"], 0), (row[0], source)


def test_table_15_matches_the_forecast_artifact():
    f = pd.read_csv(ROOT / "artifacts" / "forecast_future.csv", encoding="utf-8")
    care = f[f["target"] == "hhs_care"].set_index(["model", "h"])
    rows = paper_table(15)
    assert len(rows) == 3
    for row in rows:
        h = int(row[0].split()[0])
        s = care.loc[("SARIMA", h)]
        low80, high80 = row[2].split(" – ")
        low95, high95 = row[3].split(" – ")
        assert close(row[1], s["yhat"], 0) and close(low80, s["lo80"], 0) and close(high80, s["hi80"], 0) and close(low95, s["lo95"], 0) and close(high95, s["hi95"], 0), row[0]
        assert close(row[4], care.at[("Ensemble", h), "yhat"], 0), row[0]


def test_control_a_mistyped_table_cell_is_caught():
    assert not close("42.0", 41.945, 1)          # the misprint that the original paper carried
    assert close("41.9", 41.945, 1)


# ------------------------------------------------------------------ the executive summary keeps the same caveats as the paper
SUMMARY_MD = (ROOT / "Executive_Summary_UAC_Forecasting.md").read_text(encoding="utf-8")
SUMMARY_CAVEATS = ("not a measured custody time", "not evidence of non-compliance", "inferred from changes in the census", "simpler model than the one now in use",
                   "not 25 independent surges", "roughly one false alarm in four", "66–71%", "71% of discharges and 52% of transfers", "None of these warning signs identifies a cause",
                   "They are not predictions", "do not measure any child's actual stay")
SUMMARY_OVERSTATED = ("never used to build or choose the models", "policy-driven", "policy shift", "policy regime", "enough to activate", "every one of 25")


@pytest.mark.parametrize("caveat", SUMMARY_CAVEATS)
def test_executive_summary_keeps_the_caveat(caveat):
    assert caveat in SUMMARY_MD


@pytest.mark.parametrize("phrase", SUMMARY_OVERSTATED)
def test_executive_summary_does_not_overstate(phrase):
    assert phrase not in SUMMARY_MD


def test_executive_summary_word_and_pdf_are_in_step_with_its_source():
    base = ROOT / "Executive_Summary_UAC_Forecasting"
    assert cp.in_step(SUMMARY_MD, base.with_suffix(".docx"), base.with_suffix(".pdf")) == [], "rebuild with python tools/build_paper.py"


def test_executive_summary_flow_forecasts_match_the_artifact():
    f = pd.read_csv(ROOT / "artifacts" / "forecast_future.csv", encoding="utf-8")
    weekly = lambda target: f[(f["target"] == target) & (f["model"] == "Ensemble")].pipe(lambda g: g.groupby((g["h"] - 1) // 7)["yhat"].sum())
    assert round(weekly("transferred").mean()) == 58 and "~58 per week" in SUMMARY_MD
    assert round(weekly("discharged").min()) == 87 and round(weekly("discharged").max()) == 91 and "~87–91 per week" in SUMMARY_MD
    assert round(f[(f["target"] == "discharged") & (f["model"] == "Ensemble")]["yhat"].mean(), 1) == 12.7 and "about 12.7 discharges a day" in SUMMARY_MD


# ------------------------------------------------------------------ helpers for negative controls
def build(tmp_path, md, name="doc"):
    src = tmp_path / f"{name}.md"
    src.write_text(md, encoding="utf-8")
    fig = tmp_path / "fig.png"
    fig.write_bytes(FIGURE.read_bytes())
    Builder(src, tmp_path / f"{name}.docx").build()
    return cp.body_blocks(tmp_path / f"{name}.docx")


GOOD = """# Title

## 1. Introduction

Intro text, see Table 1 and Figure 1.

### 1.1 Methodology

Method text.

**Table 1.** A table

| A | B |
|---|---|
| x | 1 |

![Figure 1. A figure](fig.png){width=6.4}
"""


def structure(tmp_path, md, spec=None):
    spec = spec or cp.Spec(required_sections=("1. Introduction",), methodology_sections=(1,), unnumbered=())
    doc, blocks = build(tmp_path, md)
    return {r.id: r for r in cp.check_structure(spec, doc, blocks) + cp.check_cross_references(blocks)}


def test_control_a_good_document_passes(tmp_path):
    res = structure(tmp_path, GOOD)
    assert all(r.ok for r in res.values()), {k: r.detail for k, r in res.items() if not r.ok}


@pytest.mark.parametrize("cid, old, new", [
    ("C05", "### 1.1 Methodology", "### 1.3 Methodology"),                        # numbering jumps
    ("C05", "## 1. Introduction", "## 2. Introduction"),                          # section number wrong
    ("C06", "## 1. Introduction", "## 1. Something else"),                        # required section gone
    ("C07", "### 1.1 Methodology", "### 1.1 Approach"),                           # no Methodology subsection
    ("C08", "Intro text, see Table 1 and Figure 1.\n\n", ""),                     # no introduction
    ("C09", "**Table 1.** A table\n\n", ""),                                      # table without a caption
    ("C10", "![Figure 1. A figure](fig.png){width=6.4}", "![](fig.png){width=6.4}"),      # figure without a caption
    ("C10", "{width=6.4}", "{width=3.0}"),                                        # figure shrunk, so its labels shrink
    ("C11", "see Table 1 and Figure 1", "see Figure 1"),                          # table never cited
    ("C11", "see Table 1 and Figure 1", "see Table 1, Figure 1 and Table 9"),     # cites a table that does not exist
    ("C11", "see Table 1 and Figure 1", "see Table 1, Figure 1 and Section 7.2"),  # dead section reference
])
def test_control_structure_checks_catch_the_defect(tmp_path, cid, old, new):
    assert old in GOOD
    res = structure(tmp_path, GOOD.replace(old, new))
    assert not res[cid].ok, f"{cid} missed: {new[:40]}"


def test_control_table_numbers_must_be_sequential_and_titles_unique(tmp_path):
    table = "| A | B |\n|---|---|\n| x | 1 |\n"
    doc, blocks = build(tmp_path, f"# T\n\n**Table 1.** Same\n\n{table}\n**Table 3.** Other\n\n{table}\n**Table 4.** Same\n\n{table}")
    r = cp.check_tables(blocks, doc.sections[0].page_width)[0]
    assert not r.ok and "not sequential" in r.detail and "duplicate title" in r.detail


@pytest.mark.parametrize("text", ["See `run()` for this.", "It is in metrics_overall.csv today.", "```\nprint(1)\n```", "These are Companion Artefacts.", "It was a policy shock."])
def test_control_no_code_check_catches(tmp_path, text):
    doc, blocks = build(tmp_path, "# T\n\nFine.\n")
    assert cp.check_no_code(cp.Spec(), f"# T\n\n{text}\n\n## References\n", blocks)[0].ok is False


def test_control_code_font_in_the_body_is_caught(tmp_path):
    doc, blocks = build(tmp_path, "# T\n\n```\nprint(1)\n```\n")
    assert cp.check_no_code(cp.Spec(), "# T\n\nFine.\n\n## References\n", blocks)[0].ok is False


def test_control_long_sentences_and_repeats_are_caught():
    spec = cp.Spec()
    long = "This sentence goes on " + "and on " * 30 + "until it ends."
    assert cp.check_writing(spec, f"{long}\n\n## References\n")[0].ok is False
    same = "The care load rose steadily during the whole of the test window."
    assert cp.check_writing(spec, f"{same} Other words here.\n\n{same}\n\n## References\n")[1].ok is False


@pytest.mark.parametrize("old, new", [
    ("(Akaike, 1974)", "(Akaike, 1974; Nobody, 1999)"),                           # cited but not listed
    ("(Akaike, 1974)", ""),                                                       # listed but never cited
])
def test_control_citations_are_checked_both_ways(old, new):
    assert old in PAPER_MD
    assert cp.check_citations(PAPER_MD.replace(old, new))[0].ok is False


def test_control_a_made_up_number_has_no_provenance():
    assert cp.check_provenance(cp.Spec(), PAPER_MD)[0].ok
    assert cp.check_provenance(cp.Spec(), PAPER_MD.replace("2,542 children", "98,765.4 children"))[0].ok is False


@pytest.mark.parametrize("old", ["251 CBP reports", "all 25 surge events", "from 1.32 to 2.96 days", "not a measured custody time", "indicative only"])
def test_control_losing_a_key_finding_is_caught(old):
    assert old in PAPER_MD
    assert cp.check_retention(cp.Spec(), PAPER_MD.replace(old, "something else"))[0].ok is False


def test_control_losing_a_recommendation_is_caught():
    row = next(ln for ln in PAPER_MD.splitlines() if ln.startswith("| **7. Two methods"))
    assert cp.check_retention(cp.Spec(), PAPER_MD.replace(row + "\n", ""))[0].ok is False


def test_control_pdf_checks_catch_length_missing_numbers_blanks_and_clipping(tmp_path):
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    path = tmp_path / "bad.pdf"
    c = canvas.Canvas(str(path), pagesize=(595, 842))
    for i in range(3):
        c.setFont("Helvetica", 11)
        c.drawString(72, 780, "Only one line on this page")              # no page number, mostly blank
        if i == 1:
            c.drawString(560, 400, "This text runs well past the right edge of the page")
        c.showPage()
    c.save()
    res = {r.id: r for r in cp.check_pdf(cp.Spec(max_pages=2), path)}
    assert not res["C01"].ok and not res["C02"].ok and not res["C03"].ok and not res["C04"].ok


def test_control_stale_files_are_caught(tmp_path):
    md = "# T\n\n## 1. Introduction\n\nFirst paragraph.\n\n![Figure 1. A figure](fig.png){width=6.4}\n"
    doc, _ = build(tmp_path, md)
    docx = tmp_path / "doc.docx"
    assert cp.in_step(md, docx, root=tmp_path) == []                                      # built from this source: in step
    assert cp.in_step(md + "\nA paragraph added after the build.\n", docx, root=tmp_path)  # source changed, not rebuilt
    assert cp.in_step(md, docx, tmp_path / "missing.pdf", root=tmp_path) == ["no PDF"]     # Word file fine, PDF never exported
    (tmp_path / "fig.png").write_bytes((ROOT / "figures" / "paper" / "fig01_stock_flow.png").read_bytes())
    assert cp.in_step(md, docx, root=tmp_path) == ["fig.png differs from the image in the Word file"]   # figure redrawn, not rebuilt


def test_control_appendix_must_exist_and_hold_every_source_file(tmp_path):
    assert cp.check_appendix(cp.Spec(appendix=tmp_path / "missing"))[0].ok is False
    assert cp.check_appendix(cp.Spec(source_files=("tests/test_paper_quality.py",)))[0].ok is False    # this file is not in the volume


# ------------------------------------------------------------------ the build tool
def test_table_columns_fit_and_words_never_break(tmp_path):
    b = Builder(tmp_path / "x.md", tmp_path / "x.docx")
    total = b.text_width()
    wide = [["Series and model", "Discharges MAE", "Accuracy %"] + ["Transfers MAE"] * 6, ["Seasonal naive (7d)"] + ["1,234.5"] * 8]
    widths = b.col_widths(wide, 9)
    assert sum(widths) <= total * 1.001
    pt = b.table_pt / 72
    assert widths[1] >= len("Discharges") * 0.5 * pt                       # the longest header word fits on one line
    assert widths[0] >= len("Seasonal naive (7d)") * 0.5 * pt              # row labels stay on one line


def test_compact_tables_stay_compact(tmp_path):
    b = Builder(tmp_path / "x.md", tmp_path / "x.docx")
    assert sum(b.col_widths([["A", "B"], ["1", "2"]], 2)) < b.text_width() * 0.5


def test_a_very_wide_table_is_scaled_to_the_page(tmp_path):
    b = Builder(tmp_path / "x.md", tmp_path / "x.docx")
    rows = [["Averyveryverylongheaderword"] * 12, ["1"] * 12]
    assert sum(b.col_widths(rows, 12)) <= b.text_width() * 1.001


@pytest.mark.parametrize("cell, numeric", [("28.5", True), ("**98.6**", True), ("−94.7%", True), ("2,463 – 2,538", True), ("91.6% / 90.2%", True), ("3.9×", True),
                                           ("6,566 → 8,182 (+1,616)", True), ("18.5 days", True), ("day 24", False), ("Naive", False), ("n/a", False)])
def test_numeric_cells_are_recognised(cell, numeric):
    assert bool(NUMERIC.match(cell)) is numeric


def test_code_directive_sets_a_source_file_verbatim(tmp_path):
    (tmp_path / "s.py").write_text("x = 1\n# ``` a fence inside code is safe\n", encoding="utf-8")
    doc, blocks = build(tmp_path, "# T\n\n<!-- code: s.py -->\n")
    lines = {b.text for k, b in blocks if k == "p"}
    assert {"x = 1", "# ``` a fence inside code is safe"} <= lines
