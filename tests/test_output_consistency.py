"""Output consistency: the research paper, executive summary and README quote exactly the values the code produces.

Values are recomputed here from src/ and the artifacts (not read from reports/pipeline_analysis.md), so a stale report,
a hand-edited number or a changed definition all fail the suite.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from src.bottleneck_detection import INDICATORS, Thresholds
from src.kpi_calculations import KPI_DEFINITIONS, filter_period, period_kpis

PAPER = (ROOT / "Research_Paper_UAC_Forecasting.md").read_text(encoding="utf-8")
SUMMARY = (ROOT / "Executive_Summary_UAC_Forecasting.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
META = json.loads((ROOT / "artifacts" / "metadata.json").read_text())
APPENDIX = (ROOT / "Research_Paper_Appendix.md").read_text(encoding="utf-8")
PIPELINE_SECTION = PAPER[PAPER.index("## 8. Pipeline analytics"):PAPER.index("## 9. Limitations")]
KPI_APPENDIX = APPENDIX[APPENDIX.index("## Appendix A."):APPENDIX.index("## Appendix B.")]
PERIODS = {"2023": ("2023-01-12", "2024-01-30"), "before": ("2024-01-31", "2025-01-25"), "regime": ("2025-03-17", "2025-12-21"),
           "last90": ("2025-09-23", "2025-12-21")}


@pytest.fixture(scope="module")
def K(rep):
    return {name: period_kpis(filter_period(rep, a, b)) for name, (a, b) in PERIODS.items()}


def f(v, spec):
    return "n/a" if not np.isfinite(v) else f"{v:{spec}}".replace("-", "−")


def row(label, values):
    return f"| {label} | " + " | ".join(values) + " |"


# ----------------------------------------------------------------------------- research paper, section 10
def test_period_boundaries_match_metadata():
    assert (pd.Timestamp(META["cbp_bad_end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") == PERIODS["before"][0]
    assert (pd.Timestamp(META["break_transfers"]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") == PERIODS["before"][1]
    assert META["regime_start"] == PERIODS["regime"][0] and META["data_end"] == PERIODS["regime"][1]


def test_section10_kpi_table_matches_computed_values(K):
    order = ["2023", "before", "regime", "last90"]
    census = [f"{f(K[p]['hhs_care_start'], ',.0f')} → {f(K[p]['hhs_care_end'], ',.0f')} ({f(K[p]['hhs_care_change'], '+,.0f')})" for p in order]
    expected = [
        row("Reported days", [str(K[p]["reported_days"]) for p in order]),
        row("HHS census: first → last (change)", census),
        row("Children in CBP custody (mean)", [f(K[p]["cbp_custody_mean"], ",.1f") for p in order]),
        row("Discharges per reported day", [f(K[p]["discharged_per_day"], ",.1f" if K[p]["discharged_per_day"] > 100 else ".2f") for p in order]),
        row("HHS discharge rate (% of census per day)", [f(K[p]["discharge_rate_pct"], ".2f") for p in order]),
        row("Implied average stay in HHS care (days)", [f(K[p]["implied_stay_days"], ",.1f") for p in order]),
        row("Implied CBP holding time (days)", [f(K[p]["implied_cbp_hold_days"], ".2f") for p in order]),
        row("CBP transfer rate (% of custody per day)", [f(K[p]["transfer_rate_pct"], ".1f") for p in order]),
        row("Recorded throughput (discharges ÷ transfers)", [f(K[p]["throughput_ratio"], ".2f") for p in order]),
        row("CBP pass-through (transfers ÷ apprehensions)", [f(K[p]["pass_through_ratio"], ".2f") for p in order]),
    ]
    for line in expected:
        assert line in PIPELINE_SECTION, f"not found in paper Section 8:\n{line}"


def test_section10_indicator_shares_match_computed_values(flags):
    def shares(key, a, b):
        lv = filter_period(flags, a, b)[f"{key}_level"].dropna()
        return f"{100 * (lv >= 1).mean():.1f} / {100 * (lv >= 2).mean():.1f}"
    for key in INDICATORS:
        for p in ("before", "regime", "last90"):
            assert shares(key, *PERIODS[p]) in PIPELINE_SECTION, (key, p, shares(key, *PERIODS[p]))
    hold_before = filter_period(flags, *PERIODS["before"])["cbp_hold_days"].max()
    assert f"(max {hold_before:.2f} days)" in PIPELINE_SECTION
    growth_regime = filter_period(flags, *PERIODS["regime"])["census_growth_pct"].max()
    assert f"(max {growth_regime:+.1f}%)" in PIPELINE_SECTION


def test_latest_status_quoted_correctly(flags):
    last = flags.iloc[-1]
    assert last["overall_status"] == "Alert" and "Status at the last report (21 Dec 2025): Alert" in PIPELINE_SECTION
    assert f"CBP holding time was {last['cbp_hold_days']:.2f} days" in PIPELINE_SECTION and f"{last['cbp_hold_days']:.2f} days" in SUMMARY
    assert f"census growth {last['census_growth_pct']:+.1f}%" in PIPELINE_SECTION


def test_length_of_stay_reconciliation(daily, rep):
    los = daily["hhs_care"].rolling(28).mean() / daily["discharged"].rolling(28).mean()
    pre = los.loc[:META["break_discharges"]].tail(180)
    regime = los.loc[META["regime_start"]:]
    assert round(pre.mean()) == 36 and round(regime.mean()) == 203            # Section 3.3's "about 36 → about 203 days"
    k_pre = period_kpis(filter_period(rep, pre.index.min(), META["break_discharges"]))["implied_stay_days"]
    k_reg = period_kpis(filter_period(rep, META["regime_start"], META["data_end"]))["implied_stay_days"]
    for nb, dash in ((pre.mean(), k_pre), (regime.mean(), k_reg)):
        diff = 100 * (dash / nb - 1)
        assert f"| {nb:.1f} days | {dash:.1f} days | {f(diff, '+.1f')}% |" in KPI_APPENDIX


def test_abstract_pipeline_sentence(K, flags):
    abstract = PAPER[PAPER.index("## Abstract"):PAPER.index("## 1. Introduction")]
    for p in ("before", "regime", "last90"):
        assert f"{K[p]['discharge_rate_pct']:.2f}%" in abstract
    assert f"from {K['before']['implied_cbp_hold_days']:.2f} to {K['regime']['implied_cbp_hold_days']:.2f} days" in abstract
    alert_share = (filter_period(flags, *PERIODS["regime"])["cbp_holding_level"].dropna() >= 2).mean() * 100
    assert f"on {alert_share:.0f}% of current-regime reports" in abstract


def test_kpi_appendix_lists_every_kpi_and_the_default_thresholds():
    for label, definition in KPI_DEFINITIONS.values():
        assert f"| {label} | {definition} |" in KPI_APPENDIX, label
    th = Thresholds()
    for text in (f"≥ {th.census_growth_watch_pct:g}%", f"≥ {th.census_growth_alert_pct:g}%", f"≥ {th.discharge_drop_watch_pct:g}%",
                 f"≥ {th.discharge_drop_alert_pct:g}%", f"≥ {th.cbp_hold_watch_days:g} days", f"≥ {th.cbp_hold_alert_days:g} days"):
        assert text in KPI_APPENDIX, text


# ----------------------------------------------------------------------------- executive summary and README
def test_executive_summary_pipeline_table(K):
    r = lambda v, spec: f"{v:{spec}}"
    assert row("Children in CBP custody (average)", [r(K[p]["cbp_custody_mean"], ".0f") for p in ("before", "regime", "last90")]) in SUMMARY
    assert row("Transfers from CBP into HHS per day", [r(K["before"]["transferred_per_day"], ".0f"), r(K["regime"]["transferred_per_day"], ".1f"),
                                                       r(K["last90"]["transferred_per_day"], ".1f")]) in SUMMARY
    assert row("Discharges (sponsor placements) per day", [r(K["before"]["discharged_per_day"], ".0f"), r(K["regime"]["discharged_per_day"], ".1f"),
                                                           r(K["last90"]["discharged_per_day"], ".1f")]) in SUMMARY
    assert row("Share of children in care placed each day", [r(K[p]["discharge_rate_pct"], ".1f") + "%" for p in ("before", "regime", "last90")]) in SUMMARY
    assert row("Implied average wait in CBP custody before transfer", [r(K[p]["implied_cbp_hold_days"], ".1f") + " days" for p in ("before", "regime", "last90")]) in SUMMARY


def test_readme_thresholds_match_code():
    th = Thresholds()
    assert f"| ≥ +{th.census_growth_watch_pct:g}% | ≥ +{th.census_growth_alert_pct:g}% |" in README
    assert f"| ≥ {th.discharge_drop_watch_pct:g}% | ≥ {th.discharge_drop_alert_pct:g}% |" in README
    assert f"| ≥ {th.cbp_hold_watch_days:g} days | ≥ {th.cbp_hold_alert_days:g} days" in README


def test_headline_forecast_numbers_agree_across_documents():
    fc = pd.read_csv(ROOT / "artifacts" / "forecast_future.csv")
    s = fc[(fc.target == "hhs_care") & (fc.model == META["champion"]["hhs_care"]) & (fc.h == META["horizon"])].iloc[0]
    point, band = f"{s.yhat:,.0f}", f"{s.lo80:,.0f}–{s.hi80:,.0f}"
    for doc in (PAPER, SUMMARY, README):
        assert point in doc and band in doc
    acc = META["accuracy_champion"]
    for v in (acc["hhs_care"], acc["discharged"], acc["transferred"]):     # tables put the % in the column header
        assert all(re.search(rf"(?<![\d.]){re.escape(f'{v:.1f}')}(?![\d])", doc) for doc in (PAPER, SUMMARY, README)), v
    rep = next(x for x in META["surge_lead"] if x["source"].startswith("replay"))
    assert all(str(rep["events"]) in doc for doc in (PAPER, SUMMARY, README))
    assert all(f"{rep['median_lead_days']:.0f}-day" in doc or f"{rep['median_lead_days']:.0f} days" in doc for doc in (PAPER, SUMMARY, README))


def test_no_document_says_friday_is_never_reported(raw):
    frame, _ = raw
    assert (frame.index.dayofweek == 4).sum() == 2
    for name, doc in (("paper", PAPER), ("summary", SUMMARY), ("README", README)):
        assert not re.search(r"Friday and Saturday are never reported|Friday/Saturday never reported", doc), name


def test_figures_referenced_exist_and_counts_match_readme():
    refs = re.findall(r"\]\((figures/[^)]+\.png)\)", PAPER)
    assert refs and all((ROOT / r).is_file() for r in refs)
    pngs = list((ROOT / "figures").rglob("*.png"))
    assert f"{len(pngs)} charts ({len(set(refs))} embedded in the research paper)" in README


def test_generated_pipeline_kpis_file_is_current(K):
    """reports/pipeline_kpis.csv must equal a fresh computation (rerun tools/pipeline_analysis.py after any data or code change)."""
    saved = pd.read_csv(ROOT / "reports" / "pipeline_kpis.csv", index_col=0)
    by_label = {label: key for key, (label, _) in KPI_DEFINITIONS.items()}
    for col, p in zip(saved.columns, ("2023", "before", "regime", "last90")):
        for label, key in by_label.items():
            if label in saved.index and key in K[p]:
                v = pd.to_numeric(saved.at[label, col], errors="coerce")
                assert (np.isnan(v) and not np.isfinite(K[p][key])) or v == pytest.approx(K[p][key]), (label, col)
