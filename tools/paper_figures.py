"""Draw the research paper's main-body figures at their exact print size, so that axis labels and legends stay readable on the page.

    python tools/paper_figures.py            # writes figures/paper/figNN_*.png, or stops if the data no longer match the paper

Why this script exists: the notebook charts are drawn 12-15 inches wide. Placed on a 6.4-inch text column their labels shrink to 4-5 pt.
These versions use the same data at 6.4 inches and 8 pt type. The notebook charts stay in figures/ and go to the Appendix volume.

Every figure first runs check(): the data it is about to plot must reproduce the numbers printed in the paper. A chart can therefore
never silently disagree with the tables beside it.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bottleneck_detection import Thresholds, detect_bottlenecks   # noqa: E402
from src.data_loader import load_daily_clean, reported_days           # noqa: E402

OUT = ROOT / "figures" / "paper"
WIDTH = 6.4                                                           # inches: the paper's text column is 6.57 inches wide
DPI = 300
INK, INK2, MUTED, GRID, BASE = "#1f2328", "#52514e", "#898781", "#e6e5df", "#a9a89f"
FLOW = {"hhs_care": "#2a78d6", "transferred": "#eb6834", "discharged": "#1baf7a"}
MODEL = {"Naive": "#2a78d6", "Moving Avg (7d)": "#eb6834", "SARIMA": "#1baf7a", "ETS": "#eda100", "Gradient Boosting": "#008300", "Ensemble": "#4a3aa7"}
STYLE = {"Naive": (0, (4, 2)), "Moving Avg (7d)": (0, (1, 1)), "SARIMA": "-", "ETS": (0, (5, 1, 1, 1)), "Gradient Boosting": (0, (2, 1)), "Ensemble": "-"}
SCENARIO = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
WATCH, ALERT, GOOD = "#fab219", "#d03b3b", "#0ca30c"
COMMA = FuncFormatter(lambda v, _: f"{v:,.0f}")
DATE = mdates.DateFormatter("%b %Y")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelsize": 8, "axes.labelcolor": INK2, "axes.edgecolor": BASE, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.2, "legend.columnspacing": 1.2, "legend.labelcolor": INK,
    "lines.linewidth": 1.2, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": DPI, "figure.dpi": DPI,
})


def check(name, actual, expected, tol):
    """Stop the build if plotted data do not reproduce a number the paper prints."""
    if abs(float(actual) - float(expected)) > tol:
        raise SystemExit(f"Figure data do not match the paper: {name} = {actual} (paper says {expected}, tolerance {tol})")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    plt.close(fig)
    print("wrote", (OUT / f"{name}.png").relative_to(ROOT))


def raw_reports():
    """The source file exactly as reported (CBP values unmasked): the stock-flow test needs the suspect values themselves."""
    d = pd.read_csv(ROOT / "HHS_Unaccompanied_Alien_Children_Program.csv", encoding="utf-8-sig").dropna(how="all")
    d.columns = ["date", "apprehended", "cbp_custody", "transferred", "hhs_care", "discharged"]
    d["date"] = pd.to_datetime(d["date"].astype(str).str.strip(), format="%B %d, %Y")
    for c in d.columns[1:]:
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    d = d.sort_values("date", kind="stable").drop_duplicates("date", keep="first").set_index("date")
    stale = (d["hhs_care"].diff() == 0) & (d["discharged"].diff() == 0)   # the five stale HHS reports are masked before the test, as in the notebook
    d.loc[stale, ["hhs_care", "discharged"]] = np.nan
    return d


def fig1_stock_flow(meta):
    df = raw_reports()
    bad_end = pd.Timestamp(meta["cbp_bad_end"])
    gap = df.index.to_series().diff().dt.days
    d = pd.DataFrame({"change": df["hhs_care"].diff(), "net": df["transferred"] - df["discharged"]})
    d["residual"] = d["change"] - d["net"]
    pairs = d[(gap == 1) & d["residual"].notna()]
    groups = [("Copied CBP block (2023 – Jan 2024)", pairs[pairs.index <= bad_end], 192, 0.51, 190, 1), ("Trusted (Feb 2024 onward)", pairs[pairs.index > bad_end], 357, 0.90, 2.6, 0.05)]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.2), constrained_layout=True)
    for ax, (title, g, n, corr, resid, tol) in zip(axes, groups):
        r = g["change"].corr(g["net"])
        check(f"{title} pairs", len(g), n, 0)
        check(f"{title} correlation", r, corr, 0.005)
        check(f"{title} mean residual", g["residual"].mean(), resid, tol)
        ax.scatter(g["net"], g["change"], s=7, color=FLOW["hhs_care"], alpha=0.55, linewidths=0)
        lim = np.nanpercentile(np.abs(np.r_[g["net"], g["change"]]), 99.5) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.9, label="change = net flow")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{title}\nr = {r:.2f}, {len(g)} day pairs", loc="left")
        ax.set_xlabel("recorded transfers − discharges (children)")
        ax.set_ylabel("change in care load (children)")
        ax.legend(loc="lower right", handlelength=1.5)
    save(fig, "fig01_stock_flow")


def fig2_history(daily, meta):
    pre = daily.loc[pd.Timestamp(meta["cbp_bad_end"]):pd.Timestamp(meta["break_transfers"]) - pd.Timedelta(days=1)]
    post = daily.loc[pd.Timestamp(meta["regime_start"]):]
    for col, a, b in (("transferred", 209.9, 9.6), ("discharged", 203.2, 11.7), ("hhs_care", 6751, 2262)):
        tol = 0.5 if col == "hhs_care" else 0.05
        check(f"pre-break mean {col}", pre[col].mean(), a, tol)
        check(f"regime mean {col}", post[col].mean(), b, tol)
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 2.3), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [1, 1]})
    axes[0].plot(daily.index, daily["hhs_care"], color=FLOW["hhs_care"])
    axes[0].set_title("(a) Children in HHS care", loc="left")
    axes[0].set_ylabel("children")
    axes[0].yaxis.set_major_formatter(COMMA)
    axes[0].set_ylim(bottom=0)
    for c, lab in (("transferred", "Transfers into HHS"), ("discharged", "Discharges from HHS")):
        axes[1].plot(daily.index, daily[c].rolling(7, min_periods=3).mean(), color=FLOW[c], label=lab, ls="-" if c == "transferred" else (0, (4, 1.5)))
    axes[1].axvspan(daily.index.min(), pd.Timestamp(meta["cbp_bad_end"]), color="#efeee8", lw=0)
    axes[1].text(pd.Timestamp("2023-07-20"), 30, "transfers masked: copied CBP block", color=INK2, fontsize=7.5, ha="center")
    axes[1].set_title("(b) Daily flows, 7-day mean", loc="left")
    axes[1].set_ylabel("children per day")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=2, borderaxespad=0, fontsize=7)
    for ax in axes:
        ax.axvline(pd.Timestamp(meta["break_transfers"]), color=INK2, lw=0.8)
    axes[0].text(pd.Timestamp(meta["break_transfers"]) + pd.Timedelta(days=8), axes[0].get_ylim()[1] * 0.86, "transfer break\n26 Jan 2025", fontsize=7.5, color=INK2, va="top")
    axes[1].xaxis.set_major_formatter(DATE)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    save(fig, "fig02_history")


def fig3_length_of_stay(daily, meta):
    los = daily["hhs_care"].rolling(28).mean() / daily["discharged"].rolling(28).mean()
    fig, ax = plt.subplots(figsize=(WIDTH, 1.6), constrained_layout=True)
    ax.plot(los.index, los, color=FLOW["discharged"])
    brk = pd.Timestamp(meta["break_discharges"])
    ax.axvline(brk, color=INK2, lw=0.8)
    ax.text(brk - pd.Timedelta(days=8), los.max() * 0.92, "discharge break\n17 Feb 2025", ha="right", va="top", color=INK2, fontsize=7.5)
    ax.set_ylabel("days (28-day smoothing)")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(DATE)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    save(fig, "fig03_length_of_stay")


def fig4_design(daily, meta):
    t_end, cal, test, regime = daily.index.max(), pd.Timestamp(meta["cal_start"]), pd.Timestamp(meta["test_start"]), pd.Timestamp(meta["regime_start"])
    ml_start = pd.Timestamp("2024-02-14")                             # first date with a complete feature window after the copied block (Table 7)
    bars = [("HHS history (exploration, surge replay)", daily.index.min(), t_end, "#a9c9f0", INK),
            ("ML training: 14 Feb 2024 to origin", ml_start, t_end, "#5b9be6", INK),
            ("Statistical training: 17 Mar 2025 to origin", regime, t_end, "#2a78d6", INK),
            ("Calibration: 22 Jun – 20 Sep 2025", cal, test, "#eda100", INK),
            ("Test: 21 Sep – 21 Dec 2025", test, t_end, "#eb6834", INK)]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.3), constrained_layout=True)
    for i, (name, a, b, col, _) in enumerate(bars[::-1]):
        ax.barh(i, (b - a).days, left=mdates.date2num(a), height=0.68, color=col, lw=0)
        ax.text(mdates.date2num(a) - 6, i, name, va="center", ha="right", color=INK, fontsize=7.5) if (a > pd.Timestamp("2024-01-01")) else ax.text(mdates.date2num(a) + 6, i, name, va="center", color=INK, fontsize=7.5)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(DATE)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(mdates.date2num(daily.index.min()) - 5, mdates.date2num(t_end) + 8)
    save(fig, "fig04_design")


def scored_errors():
    s = pd.read_csv(ROOT / "artifacts" / "backtest.csv.gz", parse_dates=["origin", "date"])
    return s.assign(error=s["y"] - s["yhat"])


def fig5_horizon(s):
    test = s[s["period"] == "test"]
    by_h = test.groupby(["target", "model", "h"])["error"].apply(lambda e: e.abs().mean()).reset_index(name="MAE")
    care = test[test["target"] == "hhs_care"].assign(week=lambda x: (x["h"] - 1) // 7)
    paper = {"Naive": (17.9, 44.9, 71.7, 100.5), "SARIMA": (13.4, 28.4, 39.9, 52.0), "ETS": (13.1, 28.0, 38.8, 51.1), "Gradient Boosting": (19.7, 23.0, 32.3, 53.3), "Ensemble": (13.0, 24.0, 33.7, 48.4)}
    for m, vals in paper.items():
        got = care[care["model"] == m].groupby("week")["error"].apply(lambda e: e.abs().mean())
        for w, v in enumerate(vals):
            check(f"Table 10 {m} week {w + 1}", got[w], v, 0.05)
    panels = [("hhs_care", "(a) Children in care", ["Naive", "SARIMA", "ETS", "Gradient Boosting", "Ensemble"]),
              ("discharged", "(b) Discharges", ["Naive", "SARIMA", "Ensemble"]),
              ("transferred", "(c) Transfers", ["Naive", "Moving Avg (7d)", "Ensemble"])]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1, 1]})
    seen = {}
    for ax, (target, title, models) in zip(axes, panels):
        for m in models:
            g = by_h[(by_h["target"] == target) & (by_h["model"] == m)]
            line, = ax.plot(g["h"], g["MAE"], color=MODEL[m], ls=STYLE[m], lw=1.3)
            seen[m] = line
        ax.set_title(title, loc="left")
        ax.set_xticks([1, 7, 14, 21, 28])
        ax.set_xlabel("days ahead")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("mean absolute error (children)")
    order = [m for m in MODEL if m in seen]
    fig.legend([seen[m] for m in order], [m.replace("Avg (7d)", "average (7d)") for m in order], loc="outside lower center", ncol=len(order))
    save(fig, "fig05_horizon_error")


def fig6_fan(s, daily, meta):
    cal = s[(s["period"] == "calibration") & (s["target"] == "hhs_care") & (s["model"] == "SARIMA")]
    test = s[(s["period"] == "test") & (s["target"] == "hhs_care") & (s["model"] == "SARIMA")].copy()
    q = {k: {h: cal.loc[(cal["h"] - h).abs() <= 2, "error"].quantile(p) for h in range(1, 29)} for k, p in {"lo95": .025, "lo80": .10, "hi80": .90, "hi95": .975}.items()}
    for k in q:
        test[k] = test["yhat"] + test["h"].map(q[k])
    check("Table 12 SARIMA fixed-calibration coverage %", test["y"].between(test["lo80"], test["hi80"]).mean() * 100, 91.6, 0.05)
    test_start, end = pd.Timestamp(meta["test_start"]), daily.index.max()
    origins = daily.index[(daily.index >= test_start) & (daily.index < end) & daily["is_reported"]][::18][:4]
    hist = daily.loc[test_start - pd.Timedelta(days=21):]
    hist = hist[hist["is_reported"]]["hhs_care"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.25), constrained_layout=True)
    col = MODEL["SARIMA"]
    for o in origins:
        g = test[test["origin"] == o].sort_values("date")
        ax.fill_between(g["date"], g["lo95"], g["hi95"], color=col, alpha=0.13, lw=0, label="95% interval" if o == origins[0] else None)
        ax.fill_between(g["date"], g["lo80"], g["hi80"], color=col, alpha=0.26, lw=0, label="80% interval" if o == origins[0] else None)
        ax.plot(g["date"], g["yhat"], color=col, lw=1.3, label="SARIMA forecast" if o == origins[0] else None)
        ax.scatter([o], [daily.at[o, "hhs_care"]], s=13, color=col, edgecolor="white", linewidth=0.6, zorder=4, label="forecast origin" if o == origins[0] else None)
    ax.plot(hist.index, hist, color=INK, lw=1.1, label="actual care load", zorder=3)
    ax.yaxis.set_major_formatter(COMMA)
    ax.set_ylabel("children in HHS care")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU, interval=2))
    ax.legend(loc="upper left", ncol=3)
    save(fig, "fig06_fan_chart")


def fig7_replay():
    paths = pd.read_csv(ROOT / "artifacts" / "surge_replay_paths.csv", parse_dates=["origin"], encoding="utf-8").set_index("origin")
    lead = pd.read_csv(ROOT / "artifacts" / "surge_lead_times.csv", parse_dates=["event_date"], encoding="utf-8")
    lead = lead[lead["source"].str.startswith("replay")]
    check("replay surge events", len(lead), 25, 0)
    check("replay events detected", lead["detected"].sum(), 25, 0)
    check("replay median lead days", lead["lead_days"].median(), 14, 0)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.25), constrained_layout=True)
    ax.plot(paths.index, paths["peak"], color=MODEL["ETS"], lw=1.1, label="28-day alert level (P(breach) ≥ 30%)")
    ax.plot(paths.index, paths["actual"], color=INK, lw=1.2, label="actual care load")
    ax.scatter(lead["event_date"], lead["threshold"], s=15, marker="o", color=GOOD, edgecolor="white", linewidth=0.5, zorder=4, label="surge event: warned in advance")
    ax.set_ylim(top=14600)
    top2 = lead[lead["event_date"].dt.year == 2023].nlargest(2, "threshold").sort_values("event_date")
    for r, side in zip(top2.itertuples(), ("right", "left")):
        ax.annotate(f"{r.lead_days}-day warning", (r.event_date, r.threshold), xytext=(r.event_date, 14050), ha=side, va="center", fontsize=7.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=2))
    ax.yaxis.set_major_formatter(COMMA)
    ax.set_ylabel("children in HHS care")
    ax.xaxis.set_major_formatter(DATE)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=7.5)
    save(fig, "fig07_surge_replay")


def run_scenario(care, flow_in, flow_out, in_mult=1.0, out_mult=1.0, start_day=1, extra_in=0.0):
    """The stock-flow identity, as in the notebook and the dashboard."""
    days = np.arange(1, len(care) + 1)
    active = (days >= start_day).astype(float)
    return care + np.cumsum(((flow_in * (in_mult - 1) + extra_in) - flow_out * (out_mult - 1)) * active)


def fig8_scenarios(meta):
    fc = pd.read_csv(ROOT / "artifacts" / "forecast_future.csv", parse_dates=["date"])
    pick = lambda target, model: fc[(fc["target"] == target) & (fc["model"] == model)].sort_values("h")
    care = pick("hhs_care", "Ensemble")
    dates, base = care["date"].to_numpy(), care["yhat"].to_numpy()
    fin = pick("transferred", "Ensemble")["yhat"].to_numpy()
    fout = pick("discharged", "Ensemble")["yhat"].to_numpy()
    pre = meta["pre_break_transfers_per_day"]
    spec = [("Baseline (current trends)", {}, (2533, 2571)),
            ("Discharge slowdown (−30%)", {"out_mult": .7}, (2586, 2677)),
            ("Placement acceleration (+50%)", {"out_mult": 1.5}, (2444, 2393)),
            ("Intake surge (3× from day 8)", {"in_mult": 3.0, "start_day": 8}, (2650, 2917)),
            ("Combined stress (2× in, −25% out)", {"in_mult": 2.0, "out_mult": .75}, (2695, 2892)),
            (f"Return to 2024 intake (~{pre:.0f}/day from day 8)", {"extra_in": pre - fin.mean(), "start_day": 8}, (3913, 6710))]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]})
    handles = {}
    for (name, kw, (d14, d28)), colr in zip(spec, SCENARIO):
        path = run_scenario(base, fin, fout, **kw)
        check(f"Table 17 {name} day 14", path[13], d14, 0.5)
        check(f"Table 17 {name} day 28", path[27], d28, 0.5)
        ax = axes[1] if name.startswith("Return") else axes[0]
        handles[name], = ax.plot(dates, path, color=colr, lw=1.4, ls="-" if name.startswith(("Baseline", "Return")) else (0, (4, 1.2)) if name.startswith(("Discharge", "Combined")) else (0, (1.2, 1)))
    axes[1].plot(dates, base, color=SCENARIO[0], lw=1.4)
    for ax in axes:
        ax.axhline(2650, color=INK2, lw=0.8, ls=(0, (1, 1.5)))
        ax.yaxis.set_major_formatter(COMMA)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].text(dates[0], 2650, "threshold 2,650", color=INK2, fontsize=7.5, va="bottom")
    axes[1].text(dates[-1], 2690, "threshold 2,650", color=INK2, fontsize=7.5, va="bottom", ha="right")
    axes[0].set_title("(a) Operational scenarios", loc="left")
    axes[1].set_title("(b) Return to 2024 intake", loc="left")
    axes[0].set_ylabel("children in HHS care")
    fig.legend(list(handles.values()), list(handles.keys()), loc="outside lower center", ncol=2)
    save(fig, "fig08_scenarios")


def fig9_pipeline(daily, meta):
    rep, th = reported_days(daily), Thresholds()
    flags = detect_bottlenecks(rep, th)
    last = flags.iloc[-1]
    check("CBP holding time at last report", last["cbp_hold_days"], 3.75, 0.005)
    check("census growth at last report", last["census_growth_pct"], 4.6, 0.05)
    start = pd.Timestamp(meta["cbp_bad_end"]) + pd.Timedelta(days=1)
    r, f = rep.loc[start:], flags.loc[start:]
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 3.95), sharex=True, constrained_layout=True)

    def shade(ax, col, label=""):
        for lv, colr, name in ((1, WATCH, "Watch"), (2, ALERT, "Alert")):
            ax.fill_between(f.index, 0, 1, where=f[col] == lv, transform=ax.get_xaxis_transform(), color=colr, alpha=0.22, lw=0, label=f"{label}: {name}" if label else name)

    ax = axes[0]
    ax.plot(r.index, r["hhs_care"], color=FLOW["hhs_care"], label="care load")
    shade(ax, "census_growth_level", "")
    ax.set_title("(a) Children in HHS care", loc="left")
    ax.set_ylabel("children")
    ax.yaxis.set_major_formatter(COMMA)
    ax.set_ylim(bottom=0)
    ax = axes[1]
    ax.plot(f.index, f["discharge_rate_7d_pct"], color=FLOW["discharged"], label="7-day rate")
    ax.plot(f.index, f["discharge_baseline_pct"], color=INK2, lw=1, ls=(0, (1, 1)), label="baseline (prior 8 weeks)")
    shade(ax, "discharge_slowdown_level", "")
    ax.set_title("(b) HHS discharge rate", loc="left")
    ax.set_ylabel("% of census per day")
    ax.set_ylim(bottom=0)
    ax = axes[2]
    ax.plot(f.index, f["cbp_hold_days"].replace(np.inf, np.nan), color=FLOW["transferred"], label="custody ÷ transfers (7-day)")
    ax.axhline(th.cbp_hold_watch_days, color="#b77f00", lw=0.9, ls=(0, (4, 2)), label="Watch ≥ 2 days")
    ax.axhline(th.cbp_hold_alert_days, color=ALERT, lw=0.9, ls=(0, (4, 2)), label="Alert ≥ 3 days")
    ax.set_title("(c) Implied CBP holding time", loc="left")
    ax.set_ylabel("days")
    ax.set_ylim(bottom=0)
    for a in axes:
        a.axvline(pd.Timestamp(meta["break_transfers"]), color=INK2, lw=0.7)
        a.axvline(pd.Timestamp(meta["regime_start"]), color=INK2, lw=0.7, ls=(0, (2, 1)))
        a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=4, borderaxespad=0, handlelength=1.8, columnspacing=1.0, fontsize=7)
    axes[0].text(pd.Timestamp(meta["break_transfers"]) - pd.Timedelta(days=6), 1000, "transfer\nbreak", ha="right", va="center", color=INK2, fontsize=7.5)
    axes[0].text(pd.Timestamp(meta["regime_start"]) + pd.Timedelta(days=6), 5400, "current\nregime", ha="left", va="center", color=INK2, fontsize=7.5)
    axes[2].xaxis.set_major_formatter(DATE)
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    save(fig, "fig09_pipeline")


def main():
    meta = json.loads((ROOT / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
    daily = load_daily_clean(ROOT / "artifacts" / "daily_clean.csv")
    s = scored_errors()
    fig1_stock_flow(meta)
    fig2_history(daily, meta)
    fig3_length_of_stay(daily, meta)
    fig4_design(daily, meta)
    fig5_horizon(s)
    fig6_fan(s, daily, meta)
    fig7_replay()
    fig8_scenarios(meta)
    fig9_pipeline(daily, meta)


if __name__ == "__main__":
    main()
