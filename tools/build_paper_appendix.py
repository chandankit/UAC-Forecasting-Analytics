"""Generate Research_Paper_Appendix.md: everything that supports the research paper but does not belong in its main body.

    python tools/build_paper_appendix.py

Nothing here is typed from memory. Narrative comes word for word from the archived full paper, tables from the artifact CSV files and
the executed notebook, and code from the source files at build time (the `<!-- code: path -->` directive of tools/md2docx.py).
Old section, table and figure numbers are renamed on the way in; the build stops if an old number survives.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive" / "v2.3_full_paper_2026-11-05" / "Research_Paper_UAC_Forecasting.md"
OUT = ROOT / "Research_Paper_Appendix.md"
ART = ROOT / "artifacts"

# Old cross-reference -> new one. Order matters: the most specific patterns come first.
RENAMES = [
    (r"\| Section 5\.2 definition \|", "| Definition in Section 3.3 of the paper |"),
    (r"\(section 5\.2 \"before the break\"\)", "(\"before the break\" in Section 3.3 of the paper)"),
    (r"\(Section 8\)", "(Appendix G)"),
    (r"[Ss]ection 4\.6", "Appendix B.6"),
    (r"Validation rules \(4\.1\)", "Validation rules (Appendix B.2)"),
    (r"Section 5\.1", "Section 3.2 of the paper"),
    (r"[Ss]ection 5\.2", "Section 3.3 of the paper"),
    (r"[Ss]ection 10\.4", "Section 8.4 of the paper"),
    (r"[Ss]ection 10\.3", "Section 8.3 of the paper"),
    (r"[Ss]ection 10", "Section 8 of the paper"),
    (r"Version 2 requires", "The detector requires"),
    (r"which is the policy-break crash", "which is the crash at the early-2025 structural break"),
    (r"Version 2\.1 adds a fast pytest suite in `tests/` \(`python -m pytest`\) for the descriptive pipeline layer of",
     "A fast pytest suite in `tests/` (`python -m pytest`) covers the descriptive pipeline layer of"),
    (r"Seasonal naive \(new\)", "Seasonal naive"),
    (r"and Figure 19\)", "and the notebook-size pipeline chart)"),
]
STALE = re.compile(r"\b[Ss]ection \d+(?:\.\d+)?(?![\d.]| of the paper)|\bFigure \d+\b|\bTable \d+\b")


def md_table(df, digits=3):
    """A DataFrame as a pipe table, keeping integers whole and rounding floats."""
    def cell(v):
        if pd.isna(v):
            return "n/a"
        if isinstance(v, float):
            return f"{v:,.0f}" if v == int(v) and abs(v) >= 1 else f"{round(v, digits):g}"
        return str(v).replace("|", "\\|")
    rows = ["| " + " | ".join(str(c).replace("|", "\\|") for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(rows)


class Volume:
    def __init__(self):
        self.archive = ARCHIVE.read_text(encoding="utf-8").splitlines()
        self.out, self.letter, self.tables, self.figures = [], "", 0, 0

    def add(self, text):
        self.out.append(text)

    def appendix(self, letter, title, intro):
        self.letter, self.tables, self.figures = letter, 0, 0
        self.add(f"## Appendix {letter}. {title}")
        self.add(intro)

    def sub(self, number, title):
        self.add(f"### {self.letter}.{number} {title}")

    def table(self, title, body):
        self.tables += 1
        self.add(f"**Table {self.letter}{self.tables}.** {title}")
        self.add(body)

    def csv(self, title, name, folder=ART, **kwargs):
        self.table(title, md_table(pd.read_csv(folder / f"{name}.csv", encoding="utf-8"), **kwargs))

    def figure(self, title, path):
        self.figures += 1
        self.add(f"![Figure {self.letter}{self.figures}. {title}]({path})")
        return f"Figure {self.letter}{self.figures}"

    def old(self, heading, tables=()):
        """Copy an archived section's body. `tables` gives one caption for each table in it, in order."""
        start = next(i for i, ln in enumerate(self.archive) if ln.strip() == heading)
        level = len(heading) - len(heading.lstrip("#"))
        end = next((j for j in range(start + 1, len(self.archive)) if re.match(rf"^#{{1,{level}}}\s", self.archive[j])), len(self.archive))
        body = self.archive[start + 1:end]
        renumber = {}                                                 # old figure number -> new label, for references in this section
        blocks, buf, titles = [], [], list(tables)
        for ln in body + [""]:
            is_table = ln.startswith("|")
            if buf and (buf[0].startswith("|") != is_table or not ln.strip()):
                blocks.append("\n".join(buf))
                buf = []
            if ln.strip():
                buf.append(ln)
        n_tables = sum(b.startswith("|") for b in blocks)
        if n_tables != len(titles):
            raise SystemExit(f"{heading}: {n_tables} tables but {len(titles)} captions")
        pending = []
        for b in blocks:
            m = re.match(r"^!\[Figure (\d+)\.\s*(.*?)\]\((.*?)\)", b)
            if m:
                renumber[m.group(1)] = self.figure(m.group(2), m.group(3))
                continue
            if b.startswith("|"):
                self.table(titles.pop(0), b)
                pending.append(len(self.out) - 1)
                continue
            self.add(b)
            pending.append(len(self.out) - 1)
        for i in pending:
            self.out[i] = re.sub(r"\bFigure (\d+)\b", lambda k: renumber.get(k.group(1), k.group(0)), self.out[i])

    def text(self):
        doc = "\n\n".join(self.out) + "\n"
        for pat, new in RENAMES:
            doc = re.sub(pat, new, doc)
        stale = sorted({m.group(0) for m in STALE.finditer(doc)})
        if stale:
            raise SystemExit(f"Old cross-references survive in the Appendix: {stale}")
        return doc


def notebook_output(prefix):
    """The saved text output of the executed notebook cell whose source starts with `prefix`."""
    nb = json.loads((ROOT / "UAC_Care_Load_Forecasting.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "".join(cell["source"]).startswith(prefix):
            parts = []
            for out in cell.get("outputs", []):
                t = out.get("text") or out.get("data", {}).get("text/plain")
                if t:
                    parts.append("".join(t).rstrip())
            if parts:
                return "\n".join(parts)
    raise SystemExit(f"No executed notebook output for a cell starting: {prefix}")


def main():
    v = Volume()
    v.add("# Appendix to: Forecasting Care Load and Placement Demand in the HHS Unaccompanied Children Program")
    v.add("This volume supports the research paper. It holds definitions, detailed tables, tuning logs, full model outputs, additional charts and all raw source code. "
          "Section, table and figure numbers without a letter refer to the main paper, and its reference list applies here too.")
    v.add("<!-- toc -->")

    # ---------------------------------------------------------------- A
    v.appendix("A", "KPI Definitions and Length-of-Stay Reconciliation", "This appendix defines every indicator used in the paper and in the dashboard.")
    v.sub(1, "Forecast KPIs")
    start = v.archive.index("## Appendix B: KPI definitions")
    v.table("Forecast KPIs", "\n".join(v.archive[start + 2:start + 9]))
    v.sub(2, "Descriptive pipeline KPIs")
    v.old("### Descriptive pipeline KPIs (section 10)", ["Descriptive pipeline KPIs"])
    v.sub(3, "Alert rules")
    v.old("### Alert rules (section 10.3)", ["Alert rules in full"])
    v.sub(4, "Implied length of stay: two definitions")
    v.old("### Implied length of stay: two definitions", ["Implied length of stay under the two definitions"])

    # ---------------------------------------------------------------- B
    v.appendix("B", "Data-Quality Detail", "This appendix gives the full evidence behind Section 2 of the paper.")
    v.sub(1, "Source fields and the population balance")
    v.old("## 3. Data", ["Source fields and their role in the HHS population balance"])
    v.sub(2, "Input validation")
    v.old("### 4.1 Input validation (new in version 2)")
    v.sub(3, "Reporting cadence")
    v.old("### 4.2 Reporting cadence")
    v.sub(4, "Stale reports")
    v.old("### 4.3 Stale reports")
    v.sub(5, "Copied CBP records")
    v.old("### 4.4 Copied CBP records (the key finding)", ["Stock-flow identity test"])
    v.sub(6, "What the counts mean, and whether interpolating them is justified")
    v.old("### 4.6 What the counts mean, and whether interpolating them is justified")
    v.figure("Continuity with masking: last five weeks of care load", "figures/03_interpolation_mask.png")
    v.sub(7, "Decisions")
    v.old("### 4.5 Decisions", ["Data-quality decisions with their evidence"])

    # ---------------------------------------------------------------- C
    v.appendix("C", "Additional Exploratory Analysis", "This appendix holds exploratory results and notebook charts that the paper summarises.")
    v.sub(1, "Structural break and length of stay")
    v.figure("Care load and daily flows, 2023–2025 (notebook version)", "figures/04_history_overview.png")
    v.figure("Implied average length of stay (notebook version)", "figures/05_length_of_stay.png")
    v.sub(2, "Net pressure")
    v.old("### 5.3 Net pressure as the leading indicator")
    v.sub(3, "Weekly structure and the weekend ratchet")
    v.old('### 5.4 Weekly structure and the "weekend ratchet"', ["Care-load change per day: weekdays against the Thursday-to-Sunday gap"])
    v.figure("Weekday profile of flows", "figures/07_weekday_profile.png")
    v.sub(4, "Monthly pattern, outliers, surge periods and correlations")
    v.old("### 5.5 Monthly pattern, outliers, surge periods and correlations")
    v.sub(5, "Decomposition and stationarity")
    v.old("### 5.6 Decomposition and stationarity")
    v.figure("Autocorrelation and partial autocorrelation", "figures/09_acf_pacf.png")
    v.add("**Stationarity tests (executed notebook output).**")
    v.add("```\n" + notebook_output("# Stationarity: ADF") + "\n```")

    # ---------------------------------------------------------------- D
    v.appendix("D", "Feature and Leakage Audit", "This appendix documents every model input and the tests that show none of them sees the future.")
    v.sub(1, "Time-series preparation")
    v.old("### 6.1 Time-series preparation")
    v.sub(2, "Feature groups, targets and perturbation test")
    v.old("### 6.2 Feature engineering and leakage audit", ["Feature groups in full"])
    v.sub(3, "The month feature")
    v.old("### 6.3 A second root-cause finding: the month feature")
    v.sub(4, "Full feature audit")
    v.add("Each row is one of the 40 features. The last column counts values that changed in the perturbation test.")
    v.csv("Feature leakage audit", "feature_leakage_audit")
    v.figure("Feature importance", "figures/17_feature_importance.png")

    # ---------------------------------------------------------------- E
    v.appendix("E", "Model Specifications and Tuning Logs", "This appendix gives full model settings, the order-selection logs and the design of each experiment. "
               "SARIMA orders were searched by AIC. The machine-learning models use fixed settings, so no grid-search log exists for them.")
    v.sub(1, "Train–test design")
    v.old("### 6.4 Train–test design", ["Data used by each component"])
    v.sub(2, "Model specifications and fallbacks")
    v.old("### 6.5 Models", ["Models, specifications and failure fallbacks"])
    v.sub(3, "SARIMA order-selection log")
    v.add("**Best specification per series, chosen by AIC on data before the calibration window (executed notebook output).**")
    v.add("```\n" + notebook_output("# SARIMA order selection by AIC") + "\n```")
    v.add("**Best non-seasonal against best weekly-seasonal specification (executed notebook output).**")
    v.add("```\n" + notebook_output("# ARIMA vs SARIMA: best non-seasonal") + "\n```")
    v.add("**Selected specification against the other family, walked forward (executed notebook output).**")
    v.add("```\n" + notebook_output("# ARIMA vs SARIMA out of sample") + "\n```")
    v.sub(4, "Evaluation metrics")
    v.old("### 6.6 Evaluation metrics")
    v.sub(5, "Uncertainty quantification")
    v.old("### 6.7 Uncertainty quantification")
    v.sub(6, "Recursive against direct forecasting")
    v.old("### 6.8 Recursive vs direct multi-horizon experiment")
    v.add("**Results.**")
    start = v.archive.index("| Recursive vs direct (target-only GB, MAE) | 1–7 d | 8–14 d | 15–21 d | 22–28 d |")
    v.table("Recursive against direct, target-only Gradient Boosting (MAE)", "\n".join(v.archive[start:start + 5]))
    v.figure("Recursive against direct forecasting on the test window", "figures/11b_recursive_vs_direct.png")
    para = next(ln for ln in v.archive if ln.startswith("With identical inputs, the recursive strategy"))
    v.add(para)
    v.sub(7, "Early-warning design")
    v.old("### 6.9 Early-warning design")
    v.sub(8, "Scenario engine")
    v.old("### 6.10 Scenario engine")

    # ---------------------------------------------------------------- F
    v.appendix("F", "Full Model Outputs", "This appendix holds the complete result tables and the notebook charts behind Sections 6 to 8.")
    v.sub(1, "Accuracy of every model")
    v.csv("Accuracy by series, window and model", "metrics_overall", digits=2)
    v.sub(2, "Error by horizon")
    h = pd.read_csv(ART / "metrics_by_horizon.csv", encoding="utf-8")
    h = h[(h["period"] == "test") & h["h"].isin([1, 7, 14, 21, 28])].pivot_table(index=["target", "model"], columns="h", values="MAE").round(2).reset_index()
    h.columns = ["Series", "Model"] + [f"h = {c}" for c in h.columns[2:]]
    v.table("Test-window MAE by model at selected horizons", md_table(h, digits=2))
    v.add("The file metrics_by_horizon.csv in the artifacts folder has every horizon from 1 to 28.")
    start = v.archive.index("| Scorecard (test) | h = 1 | h = 7 | h = 14 | h = 28 |")
    v.table("Multi-horizon scorecard for the champion models", "\n".join(v.archive[start:start + 7]))
    v.figure("Error by horizon for all eight models (notebook version)", "figures/11_horizon_error.png")
    v.sub(3, "Statistical significance")
    v.old("### 7.4 Statistical significance", ["Diebold–Mariano p-values against naive"])
    v.add("**Full test output (executed notebook output).**")
    v.add("```\n" + notebook_output("# Diebold–Mariano test") + "\n```")
    v.sub(4, "Robustness")
    v.csv("Robustness by series and model", "robustness", digits=2)
    v.sub(5, "Interval coverage")
    v.csv("Interval coverage and width on the test window", "interval_coverage", digits=3)
    v.csv("Coverage by calibration rule", "interval_method_comparison", digits=3)
    v.figure("80% coverage by horizon", "figures/12b_coverage_by_horizon.png")
    v.figure("Care-load forecasts with intervals (notebook version)", "figures/12_fan_chart.png")
    v.sub(6, "Early warning")
    v.csv("Early-warning operating curve", "ews_operating_curve", digits=3)
    v.csv("Lead time of every surge event at the 30% alert level", "surge_lead_times", digits=0)
    v.figure("Early-warning operating curve", "figures/13a_operating_curve.png")
    v.figure("Surge replay (notebook version)", "figures/13_surge_replay.png")
    v.sub(7, "Production forecast and scenarios")
    v.figure("Production forecast from 21 December 2025", "figures/14_production_forecast.png")
    v.figure("Capacity breach probability", "figures/15_breach_probability.png")
    v.figure("Scenario comparison (notebook version)", "figures/16_scenarios.png")
    v.sub(8, "Pipeline analytics")
    reports = ROOT / "reports"
    v.csv("Indicators by period", "pipeline_indicators_by_period", folder=reports, digits=2)
    v.csv("Share of reported days by overall status and year (%)", "pipeline_status_by_year", folder=reports, digits=1)
    v.csv("Every Watch and Alert episode, full history", "pipeline_alert_episodes", folder=reports, digits=2)
    v.figure("Pipeline monitoring (notebook version)", "figures/19_pipeline_monitoring.png")

    # ---------------------------------------------------------------- G
    v.appendix("G", "Software Validation and Reproducibility", "This appendix explains how the work was tested and how to reproduce it.")
    v.sub(1, "Software validation")
    v.old("## 8. Software validation")
    v.add("A separate end-to-end QA run is reported in reports/QA_Test_Execution_Report_v2.3.pdf.")
    v.sub(2, "Where everything is")
    v.old("## Appendix A: Reproducibility", ["Reproducibility: where each result lives"])
    v.table("Rebuilding this paper", "| Item | Command |\n|---|---|\n| Main-body charts | `python tools/paper_figures.py` |\n"
            "| This Appendix volume (Markdown) | `python tools/build_paper_appendix.py` |\n| Word and PDF of both documents, then all quality checks | `python tools/build_paper.py` |\n"
            "| Quality checks only | `python tools/check_paper.py` |")

    # ---------------------------------------------------------------- H
    v.add("<!-- landscape -->")
    v.appendix("H", "Raw Source Code", "This appendix reproduces all source code as it stood when this volume was built. Long lines wrap.")
    for i, (title, path) in enumerate((("Analysis notebook source", "notebook_source/UAC_Care_Load_Forecasting_v2.py"),
                                       ("Streamlit dashboard", "uac_forecast_dashboard.py"),
                                       ("Data loading", "src/data_loader.py"),
                                       ("KPI calculations", "src/kpi_calculations.py"),
                                       ("Bottleneck detection", "src/bottleneck_detection.py"),
                                       ("Chart helpers", "src/visualizations.py"),
                                       ("Pipeline analysis script", "tools/pipeline_analysis.py"),
                                       ("Paper figure script", "tools/paper_figures.py")), start=1):
        v.sub(i, f"{title}: {path}")
        v.add(f"<!-- code: {path} -->")

    OUT.write_text(v.text(), encoding="utf-8")
    print(f"wrote {OUT.name}: {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
