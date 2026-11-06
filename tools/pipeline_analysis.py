"""Reproduce every number in the research paper's Section 8 (pipeline analytics) from the same src/ functions the dashboard uses.

    python tools/pipeline_analysis.py

Reads artifacts/daily_clean.csv and artifacts/metadata.json. Writes:
    reports/pipeline_kpis.csv              KPIs per analysis period (reported days only)
    reports/pipeline_alert_episodes.csv    every Watch/Alert episode over the full history (default thresholds)
    reports/pipeline_indicators_by_period.csv  per period and indicator: % of reports at Watch/Alert, median and maximum value
    reports/pipeline_status_by_year.csv    share of reported days by overall status and year
    reports/pipeline_los_reconciliation.csv  implied length of stay under the notebook and dashboard definitions
    reports/pipeline_analysis.md           the tables above as Markdown (the paper quotes these values)
    figures/19_pipeline_monitoring.png     notebook-size chart for the Appendix volume (tools/paper_figures.py draws the print-size version)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bottleneck_detection import INDICATORS, LEVELS, Thresholds, alert_episodes, detect_bottlenecks   # noqa: E402
from src.data_loader import load_daily_clean, reported_days                                             # noqa: E402
from src.kpi_calculations import KPI_DEFINITIONS, filter_period, period_kpis                            # noqa: E402

INK2, MUTED, GRID, BASE, SURFACE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"   # notebook figure style
COLORS = {"hhs_care": "#2a78d6", "transferred": "#eb6834", "discharged": "#1baf7a"}
LEVEL_COLORS = {1: "#fab219", 2: "#d03b3b"}


def periods(meta, rep):
    last = rep.index.max()
    return {
        "2023 (CBP masked)": (rep.index.min(), pd.Timestamp(meta["cbp_bad_end"])),
        "Before the break": (pd.Timestamp(meta["cbp_bad_end"]) + pd.Timedelta(days=1), pd.Timestamp(meta["break_transfers"]) - pd.Timedelta(days=1)),
        "Current regime": (pd.Timestamp(meta["regime_start"]), last),
        "Last 90 days": (last - pd.Timedelta(days=89), last),
    }


def kpi_table(rep, per):
    keys = ["reported_days", "hhs_care_start", "hhs_care_end", "hhs_care_change", "hhs_care_change_pct", "hhs_care_mean", "hhs_care_peak",
            "hhs_care_peak_date", "cbp_custody_mean", "cbp_custody_end", "apprehended_per_day", "transferred_per_day", "discharged_per_day",
            "net_intake_per_day", "discharge_rate_pct", "implied_stay_days", "implied_cbp_hold_days", "transfer_rate_pct", "throughput_ratio",
            "pass_through_ratio"]
    cols = {}
    for name, (a, b) in per.items():
        k = period_kpis(filter_period(rep, a, b))
        cols[f"{name} ({a:%d %b %Y} - {b:%d %b %Y})"] = [k[x] for x in keys]
    return pd.DataFrame(cols, index=keys)


def los_reconciliation(daily, rep, meta):
    """Implied stay: notebook (28-day smoothed calendar grid, paper Section 3.3) vs dashboard (ratio of reported-day means)."""
    los = daily["hhs_care"].rolling(28).mean() / daily["discharged"].rolling(28).mean()
    brk_dis, regime = pd.Timestamp(meta["break_discharges"]), pd.Timestamp(meta["regime_start"])
    nb_pre = los.loc[:brk_dis].tail(180)
    windows = {"Paper window before the break (last 180 days to the discharge break)": (nb_pre.index.min(), brk_dis),
               "Current regime": (regime, rep.index.max())}
    rows = []
    for name, (a, b) in windows.items():
        k = period_kpis(filter_period(rep, a, b))
        rows.append({"Window": f"{name}: {a:%d %b %Y} - {b:%d %b %Y}", "Notebook definition (days)": los.loc[a:b].mean(),
                     "Dashboard definition (days)": k["implied_stay_days"]})
    out = pd.DataFrame(rows)
    out["Difference %"] = 100 * (out["Dashboard definition (days)"] / out["Notebook definition (days)"] - 1)
    return out


def status_by_year(flags):
    s = flags["overall_status"].groupby(flags.index.year).value_counts(normalize=True).unstack(fill_value=0) * 100
    cols = [c for c in ["Normal", "Watch", "Alert", "Insufficient data"] if c in s]
    per_ind = {}
    for k in INDICATORS:                                     # share among reports where the indicator has a complete window
        lv = flags[f"{k}_level"].dropna()
        per_ind[INDICATORS[k]["label"] + " at Watch or Alert"] = (lv >= 1).groupby(lv.index.year).mean() * 100
    return s[cols].join(pd.DataFrame(per_ind)).round(1)


def indicator_table(flags, per):
    """Per period and indicator: reports with a complete window, % at Watch or above, % at Alert, median and maximum value."""
    rows = []
    for name, (a, b) in per.items():
        f = filter_period(flags, a, b)
        for k, spec in INDICATORS.items():
            lv, val = f[f"{k}_level"].dropna(), f[spec["value"]].replace(np.inf, np.nan).dropna()
            rows.append({"Period": name, "Indicator": spec["label"], "Reports evaluated": len(lv),
                         "% Watch or Alert": 100 * (lv >= 1).mean() if len(lv) else np.nan, "% Alert": 100 * (lv >= 2).mean() if len(lv) else np.nan,
                         "Median value": val.median() if len(val) else np.nan, "Max value": val.max() if len(val) else np.nan, "Unit": spec["unit"]})
    return pd.DataFrame(rows).set_index("Period")


def figure(rep, flags, th, meta, path):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    start = pd.Timestamp(meta["cbp_bad_end"]) + pd.Timedelta(days=1)
    r, f = rep.loc[start:], flags.loc[start:]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(r.index, r["hhs_care"], color=COLORS["hhs_care"], lw=1.8)
    for lv in (1, 2):                                        # shade days whose census-growth indicator is at Watch / Alert
        on = f["census_growth_level"] == lv
        ax.fill_between(f.index, 0, 1, where=on, transform=ax.get_xaxis_transform(), color=LEVEL_COLORS[lv], alpha=0.18, lw=0,
                        label=f"census growth {LEVELS[lv]} (≥ +{th.census_growth_watch_pct if lv == 1 else th.census_growth_alert_pct:g}% in 28 days)")
    ax.set_title("(a) Children in HHS care, reported days", loc="left"); ax.set_ylabel("children"); ax.legend(loc="upper right", frameon=False)
    ax = axes[1]
    ax.plot(f.index, f["discharge_rate_7d_pct"], color=COLORS["discharged"], lw=1.8, label="7-day discharge rate")
    ax.plot(f.index, f["discharge_baseline_pct"], color=MUTED, lw=1.4, ls=":", label="baseline (median of previous 8 weeks)")
    for lv in (1, 2):
        on = f["discharge_slowdown_level"] == lv
        ax.fill_between(f.index, 0, 1, where=on, transform=ax.get_xaxis_transform(), color=LEVEL_COLORS[lv], alpha=0.18, lw=0,
                        label=f"discharge slowdown {LEVELS[lv]}")
    ax.set_title("(b) HHS discharge rate: % of the census discharged per day", loc="left"); ax.set_ylabel("% per day")
    ax.legend(loc="lower left", frameon=False, ncol=2)
    ax = axes[2]
    hold = f["cbp_hold_days"].replace(np.inf, np.nan)
    ax.plot(f.index, hold, color=COLORS["transferred"], lw=1.8, label="implied CBP holding time (custody ÷ transfers, 7-day)")
    ax.axhline(th.cbp_hold_watch_days, color=LEVEL_COLORS[1], lw=1.2, ls="--", label=f"Watch ≥ {th.cbp_hold_watch_days:g} days")
    ax.axhline(th.cbp_hold_alert_days, color=LEVEL_COLORS[2], lw=1.2, ls="--", label=f"Alert ≥ {th.cbp_hold_alert_days:g} days (72-hour standard)")
    for d, lab, side in ((meta["break_transfers"], "transfer break ", "right"), (meta["regime_start"], " current regime", "left")):
        for a in axes:
            a.axvline(pd.Timestamp(d), color=MUTED, lw=1)
        axes[0].text(pd.Timestamp(d), axes[0].get_ylim()[1] * 0.97, lab, color=INK2, fontsize=9, va="top", ha=side)   # labels on opposite sides
    ax.set_title("(c) Implied CBP holding time (days)", loc="left"); ax.set_ylabel("days"); ax.legend(loc="upper left", frameon=False)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def md(df, floatfmt="{:,.2f}"):
    def cell(v):
        if isinstance(v, pd.Timestamp):
            return f"{v:%d %b %Y}"
        if isinstance(v, (float, np.floating)):
            return "n/a" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    rows = ["| " + " | ".join([str(i)] + [cell(v) for v in r]) + " |" for i, r in zip(df.index, df.itertuples(index=False))]
    return "\n".join([head, "|" + "---|" * (len(df.columns) + 1)] + rows)


def main():
    meta = json.loads((ROOT / "artifacts" / "metadata.json").read_text())
    daily = load_daily_clean(ROOT / "artifacts" / "daily_clean.csv")
    rep, th = reported_days(daily), Thresholds()
    flags = detect_bottlenecks(rep, th)
    out = ROOT / "reports"

    per = periods(meta, rep)
    kpis = kpi_table(rep, per)
    ind = indicator_table(flags, {k: v for k, v in per.items() if k != "2023 (CBP masked)"})
    ind.to_csv(out / "pipeline_indicators_by_period.csv")
    kpis.index = [KPI_DEFINITIONS[k][0] if k in KPI_DEFINITIONS else k for k in kpis.index]
    kpis.index.name = "KPI"
    kpis.to_csv(out / "pipeline_kpis.csv")
    episodes = alert_episodes(flags, min_level=1)
    episodes.to_csv(out / "pipeline_alert_episodes.csv", index=False)
    years = status_by_year(flags)
    years.index.name = "Year"
    years.to_csv(out / "pipeline_status_by_year.csv")
    los = los_reconciliation(daily, rep, meta).set_index("Window")
    los.to_csv(out / "pipeline_los_reconciliation.csv")
    figure(rep, flags, th, meta, ROOT / "figures" / "19_pipeline_monitoring.png")

    alerts = episodes[episodes["Highest level"] == "Alert"].sort_values(["Indicator", "Start"]).reset_index(drop=True)
    alerts.index = alerts.index + 1
    alerts.index.name = "#"
    latest = flags.iloc[-1]
    text = [f"# Pipeline analytics: generated values (do not edit; rerun `python tools/pipeline_analysis.py`)\n",
            f"Data through {rep.index.max():%d %b %Y}; {len(rep)} reported days; default thresholds {th}.\n",
            "## KPIs by period (reported days only)\n", md(kpis), "\n## Indicators by period (reports with a complete window)\n", md(ind),
            "\n## Share of reported days by overall status (%)\n", md(years, "{:.1f}"),
            "\n## Alert-level episodes (full history)\n", md(alerts[["Indicator", "Start", "End", "Reported days", "Calendar days", "Peak value"]]),
            "\n## Implied length of stay: definition reconciliation\n", md(los, "{:,.1f}"),
            f"\n## Latest report ({flags.index[-1]:%d %b %Y})\n",
            f"Overall: {latest['overall_status']}; census growth {latest['census_growth_pct']:+.1f}% ({LEVELS[int(latest['census_growth_level'])]}); "
            f"discharge rate {latest['discharge_drop_pct']:+.1f}% vs baseline ({LEVELS[int(latest['discharge_slowdown_level'])]}); "
            f"CBP holding time {latest['cbp_hold_days']:.2f} days ({LEVELS[int(latest['cbp_holding_level'])]}).\n"]
    (out / "pipeline_analysis.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
