"""
UAC Care Load & Placement Demand: Streamlit forecasting dashboard (v2.1).

Run from this folder (after executing UAC_Care_Load_Forecasting.ipynb, which writes ./artifacts):
    python -m streamlit run uac_forecast_dashboard.py

v2.1 adds the "Pipeline analytics" tab (descriptive KPIs, bottleneck alerts, data explorer, CSV downloads); its logic
lives in ./src as pure functions with their own tests in ./tests.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UAC Care Load Forecast", page_icon="📈", layout="wide")

try:  # a missing plotting dependency becomes an actionable message instead of a traceback
    import plotly.graph_objects as go
except ImportError as e:
    st.error(f"A required package is missing ({e.name}). Install the pinned dependencies with `pip install -r requirements.txt`, then restart the app.")
    st.stop()

ART = Path(__file__).resolve().parent / "artifacts"
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:            # ./src must be importable whatever the working directory
    sys.path.insert(0, str(APP_DIR))
try:  # descriptive pipeline analytics: a missing ./src disables only that tab, never the forecasts
    from src import visualizations as pviz
    from src.bottleneck_detection import INDICATORS as PIPE_INDICATORS, LEVELS as PIPE_LEVELS, Thresholds, alert_episodes, detect_bottlenecks
    from src.data_loader import DataValidationError, reported_days
    from src.kpi_calculations import KPI_DEFINITIONS, add_daily_metrics, aggregate, filter_period, period_kpis
    PIPELINE_ERROR = None
except ImportError as e:
    PIPELINE_ERROR = (f"The pipeline-analytics module could not be loaded ({e}). Keep the `src/` folder next to `{Path(__file__).name}` "
                      "and install the pinned dependencies with `pip install -r requirements.txt`. The forecast tabs are unaffected.")

# Fixed colour per entity (colour follows the model, never its rank) and reserved status colours
DEFAULT_MODEL_COLORS = {"Naive": "#2a78d6", "Seasonal Naive (7d)": "#8a5a2b", "Moving Avg (7d)": "#eb6834", "SARIMA": "#1baf7a",
                        "ETS": "#eda100", "Random Forest": "#e87ba4", "Gradient Boosting": "#008300", "Ensemble": "#4a3aa7"}
SPARE_COLORS = ["#0f8b8d", "#b8336a", "#5c6bc0", "#7a7a7a"]
SCENARIO_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"good": ("#0ca30c", "✅", "Low risk"), "warning": ("#fab219", "⚠️", "Watch"),
          "serious": ("#ec835a", "🟠", "Elevated"), "critical": ("#d03b3b", "🛑", "Critical")}
UNAVAILABLE = ("#898781", "❔", "Unavailable (no error history for this model)")
ACTUAL = "#898781"
LABELS = {"hhs_care": "Children in HHS care", "discharged": "Discharges (sponsor placements)",
          "transferred": "Transfers into HHS (from CBP)"}


# ----------------------------------------------------------------------------- data
REQUIRED = ["daily_clean.csv", "forecast_future.csv", "metrics_overall.csv", "metrics_by_horizon.csv", "interval_coverage.csv",
            "robustness.csv", "residuals.csv.gz", "surge_lead_times.csv", "surge_replay_paths.csv", "ews_operating_curve.csv", "metadata.json"]
OPTIONAL = {"cov_by_h": "interval_coverage_by_h.csv", "cov_methods": "interval_method_comparison.csv", "validation": "validation_report.csv"}
META_KEYS = ["data_end", "horizon", "champion", "recon_per_day", "alert_probability", "capacity_default", "pre_break_transfers_per_day",
             "accuracy_champion", "surge_lead", "regime_start", "break_transfers", "break_discharges", "cbp_bad_start", "cbp_bad_end",
             "blank_rows", "stale_reports", "copied_cbp_reports"]
FC_COLS = ["target", "model", "h", "date", "yhat", "lo80", "hi80", "lo95", "hi95"]
DAILY_COLS = ["hhs_care", "discharged", "transferred", "is_reported"]      # read by the forecast tabs on every run


def artifact_signature():
    """Changes whenever an artifact is rewritten, so the cache reloads after the notebook is re-run (QA D-UI-04)."""
    if not ART.is_dir():
        return ()
    files = tuple(sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in ART.iterdir() if p.is_file()))
    return (str(ART),) + files if files else ()     # folder path in the key: two artifact folders never share a cache entry


def check_contract(d):
    """Return a readable problem description if the artifacts are internally inconsistent, else None."""
    meta, fc = d["meta"], d["fc"]
    missing_keys = [k for k in META_KEYS if k not in meta]
    if missing_keys:
        return f"metadata.json lacks required key(s): {', '.join(missing_keys)}"
    if not isinstance(meta["horizon"], int) or not 1 <= meta["horizon"] <= 90:
        return f"metadata.json has an unsupported horizon ({meta['horizon']!r}); expected an integer between 1 and 90"
    missing_cols = [c for c in FC_COLS if c not in fc.columns]
    if missing_cols:
        return f"forecast_future.csv lacks column(s): {', '.join(missing_cols)}"
    for tg, champ in meta["champion"].items():
        n = int(((fc["target"] == tg) & (fc["model"] == champ)).sum())
        if n < meta["horizon"]:
            return f"forecast_future.csv has {n} of {meta['horizon']} forecast days for the champion '{champ}' ({tg})"
    if not np.isfinite(fc[["yhat", "lo80", "hi80", "lo95", "hi95"]].to_numpy(float)).all():
        return "forecast_future.csv contains non-finite forecasts or interval bounds"
    return check_daily(d["daily"])


def check_daily(daily):
    """Problems with the cleaned daily table that every tab reads, else None (a damaged table used to crash the app: QA E2E-03)."""
    if len(daily) < 8:
        return f"daily_clean.csv has {len(daily)} row(s); at least 8 days are needed"
    if not isinstance(daily.index, pd.DatetimeIndex):
        return "daily_clean.csv has dates that cannot be parsed"
    missing = [c for c in DAILY_COLS if c not in daily.columns]
    if missing:
        return f"daily_clean.csv lacks column(s): {', '.join(missing)}"
    if daily.index.has_duplicates:
        return f"daily_clean.csv has {int(daily.index.duplicated().sum())} duplicate date(s)"
    if not daily.index.is_monotonic_increasing:
        return "daily_clean.csv is not sorted by date"
    not_numeric = [c for c in DAILY_COLS[:3] if not pd.api.types.is_numeric_dtype(daily[c])]
    if not_numeric or not pd.api.types.is_bool_dtype(daily["is_reported"]):
        return f"daily_clean.csv has values of the wrong type in: {', '.join(not_numeric or ['is_reported'])}"
    gaps = [c for c in ("hhs_care", "discharged") if daily[c].isna().any()]   # the notebook fills every calendar day
    if gaps:
        return f"daily_clean.csv has missing values in: {', '.join(gaps)}"
    return None


@st.cache_data(show_spinner=False)   # spinner shown by the caller on every run, see below
def load_artifacts(signature):
    """Return (artifacts, error message). Failures become a readable message instead of a traceback (QA D-UI-03)."""
    if not signature:
        return None, "No artifacts found. Run `UAC_Care_Load_Forecasting.ipynb` first. It writes the `artifacts/` folder this app reads."
    missing = [f for f in REQUIRED if not (ART / f).exists()]
    if missing:
        return None, f"Missing artifact file(s): {', '.join(missing)}. Re-run `UAC_Care_Load_Forecasting.ipynb` to regenerate `artifacts/`."
    try:
        d = {
            "daily": pd.read_csv(ART / "daily_clean.csv", parse_dates=["date"]).set_index("date"),
            "fc": pd.read_csv(ART / "forecast_future.csv", parse_dates=["date"]),
            "overall": pd.read_csv(ART / "metrics_overall.csv"),
            "by_h": pd.read_csv(ART / "metrics_by_horizon.csv"),
            "coverage": pd.read_csv(ART / "interval_coverage.csv"),
            "robust": pd.read_csv(ART / "robustness.csv"),
            "lead": pd.read_csv(ART / "surge_lead_times.csv", parse_dates=["event_date"]),
            "replay": pd.read_csv(ART / "surge_replay_paths.csv", parse_dates=["origin"]),
            "oc": pd.read_csv(ART / "ews_operating_curve.csv"),
            "meta": json.loads((ART / "metadata.json").read_text()),
        }
        for key, name in OPTIONAL.items():           # newer artifacts: used when present, never required
            d[key] = pd.read_csv(ART / name) if (ART / name).exists() else None
        problem = check_contract(d)
        if problem:
            return None, f"The artifacts are inconsistent: {problem}. Re-run `UAC_Care_Load_Forecasting.ipynb` to regenerate them."
        resid = pd.read_csv(ART / "residuals.csv.gz")
        horizon_max = int(d["meta"]["horizon"])
        # Pre-pool walk-forward errors once per (series, model, horizon h±2): breach probabilities become O(1) lookups (QA D-PERF-01)
        d["err"] = {(tg, m, h): g.loc[(g["h"] - h).abs() <= 2, "error"].to_numpy()
                    for (tg, m), g in resid.groupby(["target", "model"]) for h in range(1, horizon_max + 1)}
    except Exception as e:  # corrupted/partial artifacts
        return None, f"The artifacts could not be read ({type(e).__name__}: {e}). Re-run `UAC_Care_Load_Forecasting.ipynb` to regenerate them."
    return d, None


SIGNATURE = artifact_signature()
# A cache spinner adds a page element only when the cache misses, which moved the tab bar one slot down on the first run
# after a start or data refresh; on the next run the tabs were remounted and the user's tab reset to Overview (QA E2E-01).
# st.spinner around the call adds its element on every run, so every element keeps its position.
with st.spinner("Loading forecast artifacts…"):
    A, load_error = load_artifacts(SIGNATURE)
if A is None:
    st.error(load_error)
    st.stop()

daily, fc, meta, ERR = A["daily"], A["fc"], A["meta"], A["err"]
H_MAX = int(meta["horizon"])
T_END = pd.Timestamp(meta["data_end"])
CHAMP = meta["champion"]
present = set(fc["model"])
MODELS = [m for m in dict.fromkeys(meta.get("models", list(DEFAULT_MODEL_COLORS)) + sorted(present)) if m in present]   # models come from the artifacts
MODEL_COLORS = {**DEFAULT_MODEL_COLORS, **meta.get("model_colors", {})}
for i, m in enumerate(m for m in MODELS if m not in MODEL_COLORS):
    MODEL_COLORS[m] = SPARE_COLORS[i % len(SPARE_COLORS)]


# ----------------------------------------------------------------------------- helpers
def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def style(fig, height=420, yfmt=",.0f", ytitle=None):
    # legend below the plot so long legends never collide with the chart toolbar (QA D-UI-08)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=30, b=8), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.14, x=0, title=None))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(tickformat=yfmt, gridwidth=1, title=ytitle, zeroline=False)
    return fig


def add_band(fig, x, lo, hi, color, name, alpha=0.15):
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=rgba(color, alpha),
                             name=name, hoverinfo="skip"))


def mark_forecast_start(fig):
    """Vertical line at the last report: observed data to the left, model forecast to the right."""
    fig.add_vline(x=T_END, line=dict(color=ACTUAL, width=1, dash="dot"))
    fig.add_annotation(x=T_END, y=1, yref="paper", text="last report → forecast", showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=11, color=ACTUAL))


def errors_for(target, model, h):
    return ERR.get((target, model, h), np.array([]))


def breach_prob(path, target, model, threshold):
    """P(care load > threshold) per day = share of walk-forward errors that would push the path above it; NaN if no errors exist."""
    out = []
    for h in range(1, len(path) + 1):
        e = errors_for(target, model, h)
        out.append(float(np.mean(path[h - 1] + e > threshold)) if len(e) else np.nan)
    return np.array(out)


def risk_status(p):
    if not np.isfinite(p):
        return UNAVAILABLE
    key = "good" if p < 0.10 else "warning" if p < 0.30 else "serious" if p < 0.60 else "critical"
    return STATUS[key]


def model_path(target, model, horizon):
    return fc[(fc["target"] == target) & (fc["model"] == model) & (fc["h"] <= horizon)].sort_values("h")


def run_scenario(base_care, base_in, base_out, in_mult=1.0, out_mult=1.0, start_day=1, extra_in=0.0):
    """Shift the baseline care path by the cumulative change in flows (stock-flow identity)."""
    days = np.arange(1, len(base_care) + 1)
    active = (days >= start_day).astype(float)
    d_in = (np.asarray(base_in) * (in_mult - 1) + extra_in) * active
    d_out = np.asarray(base_out) * (out_mult - 1) * active
    return np.asarray(base_care) + np.cumsum(d_in - d_out)


def pct(p):
    return f"{p:.0%}" if np.isfinite(p) else "n/a"


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    horizon = st.slider("Forecast horizon (days)", 1, H_MAX, H_MAX, step=1,
                        help=f"Any horizon from 1 to {H_MAX} days was validated in the walk-forward backtest; longer horizons are not supported.")
    default_models = [m for m in dict.fromkeys([CHAMP["hhs_care"], "Ensemble"]) if m in MODELS]
    models_sel = st.multiselect("Models to display", MODELS, default=default_models)
    interval = st.radio("Confidence interval", ["80%", "95%", "None"], horizontal=True)
    cap_default = int(np.clip(meta["capacity_default"], 100, 20000))
    capacity = st.number_input("Capacity / stress threshold (children)", min_value=100, max_value=20000, value=cap_default, step=50,
                               help="Enter the program's funded bed capacity or an internal stress trigger. The source data contain no capacity figures, "
                                    "so the default is only an illustrative threshold about 5% above today's load.")
    history_days = st.select_slider("History shown", options=[30, 60, 90, 180, 365], value=90)
    st.caption(f"Data through **{T_END:%d %b %Y}**. Champion models are selected on the calibration window and scored on the untouched test window.")

lo_col, hi_col = {"80%": ("lo80", "hi80"), "95%": ("lo95", "hi95"), "None": (None, None)}[interval]
if not models_sel:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------------- header
st.title("UAC care load & placement demand forecast")
st.caption("HHS Unaccompanied Alien Children Program · forward-looking capacity intelligence · walk-forward validated forecasts with calibrated uncertainty")

tabs = st.tabs(["Overview", "Care load forecast", "Discharge demand", "Model comparison", "Scenario lab", "Early warning & data quality",
                "Pipeline analytics & data explorer"])

# ============================================================================= Overview
with tabs[0]:
    champ = CHAMP["hhs_care"]
    current = float(daily["hhs_care"].iloc[-1])
    path = model_path("hhs_care", champ, horizon)
    end_val = float(path["yhat"].iloc[-1])
    p_breach = breach_prob(path["yhat"].to_numpy(), "hhs_care", champ, capacity)
    p_max = np.nanmax(p_breach) if np.isfinite(p_breach).any() else np.nan
    colr, icon, label = risk_status(p_max)
    fin = model_path("transferred", "Ensemble", horizon)["yhat"].to_numpy()
    fout = model_path("discharged", "Ensemble", horizon)["yhat"].to_numpy()
    acc = meta["accuracy_champion"]
    lead = pd.DataFrame(meta["surge_lead"])
    acc_help = meta.get("accuracy_definition", "Accuracy % = 100 − WAPE on the test window")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Children in HHS care (latest report)", f"{current:,.0f}",
              delta=f"{current - daily['hhs_care'].iloc[-8]:+,.0f} vs 7 days earlier", delta_color="off", border=True,
              chart_data=daily["hhs_care"].iloc[-60:].round().tolist())
    c2.metric(f"Forecast in {horizon} days ({champ})", f"{end_val:,.0f}", delta=f"{end_val - current:+,.0f}", delta_color="inverse", border=True)
    flow_view = (fin - fout).mean() + meta["recon_per_day"]
    c3.metric(f"Change per day ({champ})", f"{(end_val - current) / horizon:+.1f}/day",
              help=f"Direct care-load forecast. Flow-based cross-check (Ensemble transfers − discharges + {meta['recon_per_day']:.1f}/day "
                   f"weekend/non-CBP reconciliation): {flow_view:+.1f}/day. Small daily flow biases compound, so the two routes can differ; see the Discharge demand tab.",
              border=True)
    c4.metric(f"Breach risk > {capacity:,} ({horizon}d)", pct(p_max),
              help="Highest single-day probability, over the horizon, that care load exceeds the threshold (champion model + calibrated walk-forward errors). "
                   "The threshold is your input, not program data.", border=True)

    st.markdown(f"#### {icon} Capacity status: **{label}**")
    st.caption("Status bands: Low risk < 10% · Watch 10–30% · Elevated 30–60% · Critical ≥ 60% probability of exceeding the threshold within the horizon. "
               f"'Elevated' (≥ {meta['alert_probability']:.0%}) is the recommended trigger to start scaling capacity (see Early warning tab).")

    c5, c6, c7 = st.columns(3)
    c5.metric("Forecast accuracy: care load (test)", f"{acc['hhs_care']:.1f}%", help=acc_help, border=True)
    c6.metric("Forecast accuracy: discharges (test)", f"{acc['discharged']:.1f}%", help=acc_help, border=True)
    if not lead.empty:
        rep = lead[lead["source"].str.startswith("replay")]
        if not rep.empty:
            c7.metric("Surge lead time (2023–24 replay, median)", f"{rep['median_lead_days'].iloc[0]:.0f} days",
                      help=f"Detection rate {rep['detection_rate'].iloc[0]:.0%} across {int(rep['events'].iloc[0])} threshold-crossing events.", border=True)

    fig = go.Figure()
    hist = daily["hhs_care"].iloc[-history_days:]
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Observed (Fri/Sat interpolated)", line=dict(color=ACTUAL, width=2)))
    if lo_col:
        add_band(fig, path["date"], path[lo_col], path[hi_col], MODEL_COLORS[champ], f"{champ} {interval} interval")
    fig.add_trace(go.Scatter(x=path["date"], y=path["yhat"], name=f"{champ} forecast", line=dict(color=MODEL_COLORS[champ], width=2)))
    fig.add_hline(y=capacity, line=dict(color=STATUS["critical"][0], width=1), annotation_text=f"threshold {capacity:,} (user input)", annotation_position="top left")
    mark_forecast_start(fig)
    st.plotly_chart(style(fig, 380, ytitle="children"), key="overview_chart")

# ============================================================================= Care load forecast
with tabs[1]:
    st.subheader("Future care load forecast")
    fig = go.Figure()
    hist = daily["hhs_care"].iloc[-history_days:]
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Actual", line=dict(color=ACTUAL, width=2)))
    for m in models_sel:
        p = model_path("hhs_care", m, horizon)
        if lo_col:
            add_band(fig, p["date"], p[lo_col], p[hi_col], MODEL_COLORS[m], f"{m} {interval}", alpha=0.12)
        fig.add_trace(go.Scatter(x=p["date"], y=p["yhat"], name=m, line=dict(color=MODEL_COLORS[m], width=2)))
    fig.add_hline(y=capacity, line=dict(color=STATUS["critical"][0], width=1))
    mark_forecast_start(fig)
    st.plotly_chart(style(fig, 460, ytitle="children in care"), key="care_chart")

    tbl = fc[(fc["target"] == "hhs_care") & fc["model"].isin(models_sel) & (fc["h"] <= horizon)]
    pivot = tbl.pivot_table(index="date", columns="model", values="yhat").round(0)
    pivot.index = pivot.index.strftime("%a %d %b %Y")
    with st.expander("Forecast table (point forecasts)"):
        st.dataframe(pivot, height=320)
        st.download_button("Download these forecasts with 80%/95% intervals (CSV)", tbl.sort_values(["model", "h"]).to_csv(index=False).encode("utf-8"),
                           file_name=f"uac_care_load_forecast_{T_END:%Y%m%d}_h{horizon}.csv", mime="text/csv", on_click="ignore", key="dl_care_fc")
    st.markdown("**Capacity breach probability by day** (threshold from the sidebar)")
    probs = pd.DataFrame({m: breach_prob(model_path("hhs_care", m, horizon)["yhat"].to_numpy(), "hhs_care", m, capacity) for m in models_sel},
                         index=np.arange(1, horizon + 1))
    no_err = [m for m in models_sel if not np.isfinite(probs[m]).any()]
    if no_err:
        st.info(f"No walk-forward error history for {', '.join(no_err)}: breach probability unavailable for these models.")
    fig = go.Figure()
    for m in models_sel:
        fig.add_trace(go.Scatter(x=probs.index, y=probs[m] * 100, name=m, line=dict(color=MODEL_COLORS[m], width=2)))
    fig.update_xaxes(title="days ahead")
    fig.update_yaxes(range=[0, 100])        # fixed 0–100%: a flat 0% line no longer autoscales to duplicated '-1, -1, 0, 1, 1' ticks (QA D-UI-13)
    st.plotly_chart(style(fig, 300, yfmt=".0f", ytitle="probability (%)"), key="breach_chart")

# ============================================================================= Discharge demand
with tabs[2]:
    st.subheader("Discharge (placement) demand & intake balance")
    left, right = st.columns(2)
    for target, col in (("discharged", left), ("transferred", right)):
        fig = go.Figure()
        hist = daily[target].iloc[-history_days:]
        rep = daily.loc[hist.index, "is_reported"].astype(bool)
        fig.add_trace(go.Scatter(x=hist.index[rep], y=hist[rep], mode="markers", name="Reported day",
                                 marker=dict(color=ACTUAL, size=6)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist.rolling(7, min_periods=1).mean(), name="7-day mean", line=dict(color=ACTUAL, width=2)))
        for m in models_sel:
            p = model_path(target, m, horizon)
            if lo_col:
                add_band(fig, p["date"], p[lo_col], p[hi_col], MODEL_COLORS[m], f"{m} {interval}", alpha=0.12)
            fig.add_trace(go.Scatter(x=p["date"], y=p["yhat"], name=m, line=dict(color=MODEL_COLORS[m], width=2)))
        mark_forecast_start(fig)
        col.markdown(f"**{LABELS[target]}**")
        col.plotly_chart(style(fig, 360, ytitle="children / day"), key=f"flow_{target}")

    weeks = pd.DataFrame({"Transfers in": fin, "Discharges out": fout}, index=model_path("discharged", "Ensemble", horizon)["date"])
    week_no = (np.arange(len(weeks)) // 7) + 1
    days_in_week = pd.Series(week_no).value_counts().sort_index()
    weekly = weeks.groupby(week_no).sum().round(0)
    weekly["Net change"] = weekly["Transfers in"] - weekly["Discharges out"] + days_in_week.to_numpy() * meta["recon_per_day"]
    weekly.index = [f"Week {i}" if n == 7 else f"Week {i} (partial: {n} day{'s' if n > 1 else ''})" for i, n in days_in_week.items()]
    c1, c2 = st.columns([1.3, 1])          # the table needs room for partial-week labels (QA D-UI-13)
    c1.markdown("**Weekly totals (Ensemble)**")
    c1.dataframe(weekly.style.format("{:+,.0f}", subset=["Net change"]).format("{:,.0f}", subset=["Transfers in", "Discharges out"]),
                 column_config={"_index": st.column_config.Column("Week", width="medium")})   # partial-week labels stay readable (QA D-UI-13)
    c1.caption(f"Net change adds the measured reconciliation term ({meta['recon_per_day']:+.1f}/day): care-load growth not explained by recorded transfers − discharges "
               "(non-CBP admissions, unreported weekend flows). The file has no 'other exits' column.")
    net = weeks["Transfers in"] - weeks["Discharges out"] + meta["recon_per_day"]
    fig = go.Figure(go.Bar(x=net.index, y=net, marker_color=[STATUS["serious"][0] if v > 0 else STATUS["good"][0] for v in net],
                           name="net pressure", hovertemplate="%{x|%d %b}: %{y:+.1f}/day<extra></extra>"))
    ticks = np.unique(np.round(np.linspace(min(net.min(), 0.0), max(net.max(), 0.0), 5), 1)) + 0.0   # "+ 0.0" turns -0.0 into 0.0
    fig.update_yaxes(tickvals=ticks, ticktext=[f"{v:,.1f}" for v in ticks])                       # no '-0' tick label (QA D-UI-13)
    c2.markdown("**Forecast net pressure (in − out, children/day)**: orange = filling, green = draining")
    c2.plotly_chart(style(fig, 280, yfmt=",.1f"), key="net_chart")   # '+.0f' is rejected by Plotly (QA D-UI-07); one decimal avoids a '-0' tick (QA D-UI-13)

# ============================================================================= Model comparison
with tabs[3]:
    st.subheader("Model selection & comparison (walk-forward backtest)")
    c1, c2 = st.columns(2)
    target = c1.selectbox("Series", list(LABELS), format_func=LABELS.get)
    period = c2.radio("Window", ["test", "calibration"], horizontal=True, help="Champions are chosen on the calibration window; the test window was never used for any choice.")
    board = A["overall"][(A["overall"]["target"] == target) & (A["overall"]["period"] == period)].drop(columns=["target", "period"])
    board = board.sort_values("MAE").set_index("model")
    st.markdown(f"Champion for **{LABELS[target]}**: **{CHAMP[target]}**")
    st.dataframe(board.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "MAPE %": "{:.1f}", "WAPE %": "{:.1f}", "Accuracy %": "{:.1f}", "Bias": "{:+.2f}", "n": "{:,.0f}"}, na_rep="n/a")
                 .highlight_min(subset=["MAE", "RMSE"], color=rgba("#0ca30c", 0.18)))
    st.caption("MAE/RMSE in children (per day for flows). MAPE uses non-zero actuals only ('zero y excluded' counts the rest). "
               f"{meta.get('accuracy_definition', 'Accuracy % = 100 − WAPE')}. Bias = mean(forecast − actual): negative = under-forecast.")
    bh = A["by_h"][(A["by_h"]["target"] == target) & (A["by_h"]["period"] == period) & (A["by_h"]["h"] <= horizon)]
    fig = go.Figure()
    for m in MODELS:
        g = bh[bh["model"] == m]
        fig.add_trace(go.Scatter(x=g["h"], y=g["MAE"], name=m, visible=True if m in models_sel or m == CHAMP[target] else "legendonly",
                                 line=dict(color=MODEL_COLORS[m], width=3 if m == CHAMP[target] else 2)))
    fig.update_xaxes(title="horizon (days ahead)")
    st.markdown("**Horizon error**: MAE by days ahead (click legend items to toggle models)")
    st.plotly_chart(style(fig, 360, yfmt=",.1f", ytitle="MAE (children)"), key="horizon_chart")
    c1, c2 = st.columns(2)
    cov = A["coverage"][A["coverage"]["target"] == target].drop(columns="target").set_index("model")
    c1.markdown("**Interval coverage on test** (target: 0.80 / 0.95), fixed calibration window")
    c1.dataframe(cov.style.format({"coverage80": "{:.0%}", "coverage95": "{:.0%}", "width80": "{:,.1f}"}))
    rob = A["robust"][A["robust"]["target"] == target].drop(columns="target").set_index("model")
    c2.markdown("**Robustness scorecard**")
    c2.dataframe(rob.round(2))
    if A["cov_methods"] is not None:
        cm = A["cov_methods"][A["cov_methods"]["target"] == target].pivot_table(index="model", columns="method", values="coverage80")
        st.markdown("**80% interval coverage by calibration method (test window)**: the production forecast uses the rolling 26-week rule")
        st.dataframe(cm.style.format("{:.0%}", na_rep="n/a"))

# ============================================================================= Scenario lab
with tabs[4]:
    st.subheader("Scenario comparison view")
    st.caption("Only the **baseline** is a model forecast. Every other path is a **hypothetical what-if** built from an assumed change in flows; "
               "it is not a prediction. The threshold line is your input (the source data contain no bed capacity).")
    anchor = st.selectbox("Baseline care-load model", MODELS, index=MODELS.index("Ensemble") if "Ensemble" in MODELS else 0)
    base_care = model_path("hhs_care", anchor, horizon)["yhat"].to_numpy()
    dates = model_path("hhs_care", anchor, horizon)["date"]
    pre_in = meta["pre_break_transfers_per_day"]
    presets = {
        "Baseline (current trends)": dict(),
        "Discharge slowdown (−30%)": dict(out_mult=0.7),
        "Placement acceleration (+50% discharges)": dict(out_mult=1.5),
        "Intake surge (3× transfers from day 8)": dict(in_mult=3.0, start_day=8),
        "Combined stress (2× in, −25% out)": dict(in_mult=2.0, out_mult=0.75),
        f"Return to 2024 intake (~{pre_in:.0f}/day from day 8)": dict(extra_in=pre_in - fin.mean(), start_day=8),
    }
    chosen = st.multiselect("Preset scenarios", list(presets), default=list(presets)[:4])
    with st.expander("Build a custom scenario", expanded=True):
        k1, k2, k3, k4 = st.columns(4)
        in_mult = k1.slider("Transfers multiplier", 0.5, 5.0, 1.0, 0.1)
        out_mult = k2.slider("Discharges multiplier", 0.5, 2.0, 1.0, 0.05)
        extra = k3.slider("Extra daily intake", 0, 300, 0, 5)
        if horizon > 1:
            start = k4.slider("Shock starts on day", 1, horizon, 1)
        else:  # a one-day horizon leaves nothing to choose (a slider needs min < max)
            start = 1
            k4.caption("Shock starts on day 1 (one-day horizon)")
    scen = {name: run_scenario(base_care, fin, fout, **presets[name]) for name in chosen}
    if in_mult != 1 or out_mult != 1 or extra:
        scen["Custom scenario"] = run_scenario(base_care, fin, fout, in_mult, out_mult, start, extra)

    if not scen:  # empty state instead of a KeyError (QA D-UI-01)
        st.info("Select at least one preset scenario, or move a custom-scenario slider, to compare scenarios.")
    else:
        fig = go.Figure()
        for i, (name, path_s) in enumerate(scen.items()):
            fig.add_trace(go.Scatter(x=dates, y=path_s, name=name, line=dict(color=SCENARIO_COLORS[i % len(SCENARIO_COLORS)], width=2)))
        fig.add_hline(y=capacity, line=dict(color=STATUS["critical"][0], width=1), annotation_text=f"threshold {capacity:,}", annotation_position="top left")
        st.plotly_chart(style(fig, 440, ytitle="children in care"), key="scenario_chart")

        rows = []
        for name, path_s in scen.items():
            pb = breach_prob(path_s, "hhs_care", anchor, capacity)
            pb_max = np.nanmax(pb) if np.isfinite(pb).any() else np.nan
            first = next((i + 1 for i, v in enumerate(path_s) if v > capacity), None)
            colr, icon, label = risk_status(pb_max)
            rows.append({"Scenario": name, f"Day {horizon} care load": round(path_s[-1]), "Change vs today": round(path_s[-1] - current),
                         "Peak": round(path_s.max()),
                         "First day above threshold": f"Day {first}" if first else "Not reached",   # one dtype (QA D-UI-05)
                         "Breach probability (if the assumption holds)": pct(pb_max), "Status": f"{icon} {label}",
                         "Type": "model forecast" if name.startswith("Baseline") else "hypothetical"})
        st.dataframe(pd.DataFrame(rows).set_index("Scenario"),
                     column_config={"Change vs today": st.column_config.NumberColumn(format="%+d")})   # explicit sign (QA D-UI-13)

    if scen:  # the translator needs at least one scenario (QA D-UI-01)
        with st.expander("Staffing & bed translator (planning assumptions, edit to match your program)"):
            s1, s2, s3 = st.columns(3)
            ratio = s1.number_input("Children per case manager (assumption)", 1, 50, 10)
            buffer = s2.slider("Surge buffer on beds (%)", 0, 50, 10)
            sc_name = s3.selectbox("Scenario", list(scen))
            peak = scen[sc_name].max()
            e_h = errors_for("hhs_care", anchor, horizon)
            upper = peak + (np.quantile(e_h, 0.9) if len(e_h) else 0.0)
            st.write(f"Peak expected care load: **{peak:,.0f}** · planning upper bound (P90): **{upper:,.0f}**")
            st.write(f"Beds to have available incl. {buffer}% buffer: **{upper * (1 + buffer / 100):,.0f}** · "
                     f"case managers needed at 1:{ratio}: **{np.ceil(upper / ratio):,.0f}** (today: {np.ceil(current / ratio):,.0f})")
            st.caption("Ratios are user assumptions, not program policy. Use them to translate forecasts into staffing and bed conversations.")

# ============================================================================= Early warning & data quality
with tabs[5]:
    p_star = meta["alert_probability"]
    st.subheader("Early warning: historical surge replay (2023–2024)")
    st.caption(f"Alert rule: warn when the probability of exceeding a threshold within {H_MAX} days reaches **{p_star:.0%}** "
               "(the 'Elevated' status). Replayed in real time over the 2023–24 surges with a rolling ETS model.")
    rp, lead_t = A["replay"], A["lead"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rp["origin"], y=rp["actual"], name="Actual care load", line=dict(color=ACTUAL, width=2)))
    fig.add_trace(go.Scatter(x=rp["origin"], y=rp["peak"], name=f"{H_MAX}-day alert level (P ≥ {p_star:.0%})", line=dict(color=MODEL_COLORS["ETS"], width=2)))
    for detected, name, colr in ((True, "Surge event: warned in advance", STATUS["good"][0]), (False, "Surge event: missed", STATUS["critical"][0])):
        ev = lead_t[lead_t["source"].str.startswith("replay") & (lead_t["detected"] == detected)]
        if len(ev):
            fig.add_trace(go.Scatter(x=ev["event_date"], y=ev["threshold"], mode="markers", name=name,
                                     marker=dict(color=colr, size=10, line=dict(color="white", width=2)),
                                     customdata=ev["lead_days"], hovertemplate="%{x|%d %b %Y} · threshold %{y:,}: %{customdata} days warning<extra></extra>"))
    st.plotly_chart(style(fig, 400, ytitle="children in care"), key="replay_chart")
    c1, c2 = st.columns([1, 1.2])
    c1.markdown(f"**Surge lead time at P ≥ {p_star:.0%}**")
    c1.dataframe(pd.DataFrame(meta["surge_lead"]).set_index("source").rename(columns={
        "events": "Events", "detection_rate": "Detection rate", "median_lead_days": "Median lead (days)",
        "mean_lead_days": "Mean lead (days)", "false_alarm_rate": "False alarms"}).style.format(
        {"Detection rate": "{:.0%}", "False alarms": "{:.0%}", "Median lead (days)": "{:.0f}", "Mean lead (days)": "{:.1f}"}))
    c2.markdown("**Operating curve**: earlier warnings cost more false alarms")
    c2.dataframe(A["oc"].set_index(["source", "alert at P(breach) ≥"]))

    st.subheader("Data quality: root-cause findings")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Blank padding rows removed", meta["blank_rows"], border=True)
    q2.metric("Stale HHS reports masked", meta["stale_reports"], border=True)
    q3.metric("Copied CBP reports masked", meta["copied_cbp_reports"], border=True)
    q4.metric("Current regime since", pd.Timestamp(meta["regime_start"]).strftime("%d %b %Y"), border=True)
    if "validation_issues" in meta:
        st.caption(f"Input validation on the last run: **{meta['validation_issues']}** implausible value(s) quarantined or flagged; "
                   f"**{meta.get('fit_failures', 0)}** model fit/prediction failure(s) logged (a failing model falls back to naive persistence).")
    if A["validation"] is not None and len(A["validation"]):
        st.dataframe(A["validation"])
    st.markdown(
        f"""
* **Reporting cadence:** reports arrive Sunday–Thursday; Friday and Saturday are interpolated for continuity but **never scored**, and forecasts are only issued from reported days (no look-ahead).
* **Copied CBP block:** from **{meta['cbp_bad_start']}** to **{meta['cbp_bad_end']}** every CBP record (apprehended, in custody, transferred) duplicates a later date, and the stock-flow identity breaks. CBP columns are masked there; HHS columns reconcile and are kept.
* **Regime shift:** transfers broke on **{meta['break_transfers']}** and discharges on **{meta['break_discharges']}**. Statistical models train on the current regime; ML models use scale-free targets.
""")
    fig = go.Figure()
    for target, colr in (("transferred", "#eb6834"), ("discharged", "#1baf7a")):
        fig.add_trace(go.Scatter(x=daily.index, y=daily[target].rolling(7, min_periods=3).mean(), name=f"{LABELS[target]} (7-day mean)", line=dict(color=colr, width=2)))
    fig.add_vrect(x0=meta["cbp_bad_start"], x1=meta["cbp_bad_end"], fillcolor=rgba("#898781", 0.15), line_width=0,
                  annotation_text="CBP columns masked", annotation_position="top left")
    st.plotly_chart(style(fig, 340, ytitle="children / day"), key="dq_chart")


# ============================================================================= Pipeline analytics (descriptive, recomputed live)
PIPE_STATUS = {0: ("✅", "Normal"), 1: ("⚠️", "Watch"), 2: ("🛑", "Alert")}


@st.cache_data(show_spinner=False)
def pipeline_reported(signature, _daily):
    """Reported days only (no interpolated Fri/Sat values), CBP masked where untrusted; cached per artifact version."""
    return reported_days(_daily)


@st.cache_data(show_spinner=False)   # spinner shown by the caller on every run (stable element positions, QA E2E-01)
def pipeline_flags(signature, th_values, _rep):
    """Indicators and alert levels over the FULL history, so trailing windows at a period's start use earlier reports."""
    return detect_bottlenecks(_rep, Thresholds(*th_values))


def num(v, spec, suffix=""):
    return f"{v:{spec}}{suffix}" if v is not None and np.isfinite(v) else "n/a"


def render_pipeline_tab():
    """Period filter → KPI cards → status banner → charts → episodes → summary table → data explorer and downloads."""
    try:
        rep = pipeline_reported(SIGNATURE, daily)
    except DataValidationError as e:
        st.error(f"The cleaned daily table cannot be used: {e}. Re-run the notebook to regenerate `artifacts/`.")
        return
    if rep.empty:
        st.info("The cleaned daily table contains no reported days.")
        return
    first, last = rep.index.min().date(), rep.index.max().date()
    regime = pd.Timestamp(meta["regime_start"]).date()
    pre_start = (pd.Timestamp(meta["cbp_bad_end"]) + pd.Timedelta(days=1)).date()
    pre_end = (pd.Timestamp(meta["break_transfers"]) - pd.Timedelta(days=1)).date()
    presets = {f"Current regime (since {regime:%d %b %Y})": (regime, last),
               "Last 90 days": (last - pd.Timedelta(days=89).to_pytimedelta(), last),
               "Last 12 months": (last - pd.Timedelta(days=364).to_pytimedelta(), last),
               f"Before the 2025 break ({pre_start:%d %b %Y} – {pre_end:%d %b %Y})": (pre_start, pre_end),
               "Full history": (first, last),
               "Custom range": None}

    c1, c2 = st.columns([1.3, 1])
    choice = c1.selectbox("Analysis period", list(presets), key="pa_period",
                          help="All KPIs, charts, alerts and tables below are recomputed for this period.")
    if presets[choice] is None:
        picked = c2.date_input("Custom date range", value=(max(first, last - pd.Timedelta(days=180).to_pytimedelta()), last),
                               min_value=first, max_value=last, format="DD/MM/YYYY", key="pa_range")
        if not isinstance(picked, (tuple, list)) or len(picked) != 2:   # also what a typed impossible or out-of-range date gives (QA3-UX-01)
            st.info(f"Select both a start and an end date to update the analytics. Dates must lie between {first:%d %b %Y} and "
                    f"{last:%d %b %Y}; type them as DD/MM/YYYY – DD/MM/YYYY or pick them in the calendar.")
            return
        start, end = picked
    else:
        start, end = presets[choice]
        c2.caption(f"**{pd.Timestamp(start):%d %b %Y} – {pd.Timestamp(end):%d %b %Y}** · data available {first:%d %b %Y} – {last:%d %b %Y}")

    with st.expander("Alert thresholds (documented defaults; adjust to your program's standards)"):
        t1, t2, t3 = st.columns(3)
        g_w, g_a = t1.slider("Census growth: Watch / Alert (% in 28 days)", 1.0, 50.0, (5.0, 10.0), 0.5, key="pa_growth")
        d_w, d_a = t2.slider("Discharge slowdown: Watch / Alert (% below 8-week median)", 5.0, 90.0, (15.0, 30.0), 1.0, key="pa_drop")
        h_w, h_a = t3.slider("CBP holding time: Watch / Alert (days)", 0.5, 10.0, (2.0, 3.0), 0.25, key="pa_hold")
        st.caption("Each indicator is evaluated on every reported day from trailing windows only (no look-ahead); a day's status is its worst indicator. "
                   "The 3-day CBP default mirrors the 72-hour transfer standard for unaccompanied children (TVPRA 2008); the census and discharge "
                   "defaults are operating choices, not program policy. Recorded throughput is shown but not used for alerts because recorded "
                   "flows do not close the census balance.")
    th = Thresholds(census_growth_watch_pct=g_w, census_growth_alert_pct=g_a, discharge_drop_watch_pct=d_w, discharge_drop_alert_pct=d_a,
                    cbp_hold_watch_days=h_w, cbp_hold_alert_days=h_a)
    try:
        th.validate()
        with st.spinner("Detecting bottlenecks…"):
            flags = pipeline_flags(SIGNATURE, th.as_tuple(), rep)
        p, fp = filter_period(rep, start, end), filter_period(flags, start, end)
    except ValueError as e:                                          # reversed range or inconsistent thresholds
        st.warning(f"{e}. Adjust the selection to continue.")
        return
    if p.empty:
        st.info(f"No reported days between {pd.Timestamp(start):%d %b %Y} and {pd.Timestamp(end):%d %b %Y} "
                "(reports are published Sunday–Thursday). Choose a wider range.")
        return

    k = period_kpis(p)
    span = (k["end"] - k["start"]).days + 1
    lbl = {key: v[0] for key, v in KPI_DEFINITIONS.items()}
    hlp = {key: v[1] for key, v in KPI_DEFINITIONS.items()}

    st.markdown("##### Populations (stocks)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(lbl["hhs_care_end"], num(k["hhs_care_end"], ",.0f"), border=True, delta_color="off",
              delta=f"{num(k['hhs_care_change'], '+,.0f')} ({num(k['hhs_care_change_pct'], '+.1f', '%')}) vs period start" if np.isfinite(k["hhs_care_change"]) else None,
              help=f"{hlp['hhs_care_end']}. Change: {hlp['hhs_care_change']}.", chart_data=p["hhs_care"].dropna().round().tolist()[-90:])
    m2.metric(lbl["hhs_care_peak"], num(k["hhs_care_peak"], ",.0f"), border=True,
              help=f"{hlp['hhs_care_peak']}: {k['hhs_care_peak_date']:%d %b %Y}." if pd.notna(k["hhs_care_peak_date"]) else hlp["hhs_care_peak"])
    cbp_note = "" if k["cbp_custody_days"] else " No trusted CBP reports in this period (copied records 12 Jan 2023 – 30 Jan 2024 are masked)."
    m3.metric(lbl["cbp_custody_end"], num(k["cbp_custody_end"], ",.0f"), border=True, help=hlp["cbp_custody_end"] + "." + cbp_note,
              delta=f"{num(k['cbp_custody_change'], '+,.0f')} vs period start" if np.isfinite(k["cbp_custody_change"]) else None, delta_color="off")
    m4.metric("Reported days in period", f"{k['reported_days']:,}", border=True,
              help=f"{span:,} calendar days from {k['start']:%d %b %Y} to {k['end']:%d %b %Y}; Friday/Saturday are not published and not imputed here.")

    st.markdown("##### Flows and efficiency")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(lbl["transferred_per_day"], num(k["transferred_per_day"], ",.1f"), border=True, help=hlp["transferred_per_day"] + cbp_note)
    m2.metric(lbl["discharged_per_day"], num(k["discharged_per_day"], ",.1f"), border=True, help=hlp["discharged_per_day"])
    m3.metric(lbl["discharge_rate_pct"], num(k["discharge_rate_pct"], ".2f", "%"), border=True,
              help=f"{hlp['discharge_rate_pct']}. {lbl['implied_stay_days']}: {num(k['implied_stay_days'], ',.0f')} days ({hlp['implied_stay_days']}).")
    m4.metric(lbl["implied_cbp_hold_days"], num(k["implied_cbp_hold_days"], ".2f", " days"), border=True, help=hlp["implied_cbp_hold_days"] + "." + cbp_note)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(lbl["apprehended_per_day"], num(k["apprehended_per_day"], ",.1f"), border=True, help=hlp["apprehended_per_day"] + cbp_note)
    m2.metric(lbl["implied_stay_days"], num(k["implied_stay_days"], ",.0f", " days"), border=True,
              help=hlp["implied_stay_days"] + ". The research paper's 36 → 203-day figure (Section 3.3) averages a 28-day smoothed daily series over "
                   "fixed windows, so it differs slightly; Appendix A of the paper's Appendix volume reconciles the two definitions.")
    m3.metric(lbl["throughput_ratio"], num(k["throughput_ratio"], ".2f"), border=True, help=hlp["throughput_ratio"] + ".")
    m4.metric(lbl["transfer_rate_pct"], num(k["transfer_rate_pct"], ".1f", "%"), border=True, help=hlp["transfer_rate_pct"] + "." + cbp_note)

    latest = fp.iloc[-1]
    if np.isfinite(latest["overall_level"]):
        icon, name = PIPE_STATUS[int(latest["overall_level"])]
    else:
        icon, name = "❔", "Insufficient data"
    st.markdown(f"#### {icon} Bottleneck status on {fp.index[-1]:%d %b %Y} (last report in period): **{name}**")
    parts = []
    for key, spec in PIPE_INDICATORS.items():
        lv, val, unit = latest[f"{key}_level"], latest[spec["value"]], spec["unit"]
        state = PIPE_STATUS[int(lv)][1] if np.isfinite(lv) else "insufficient data"
        if key == "discharge_slowdown" and np.isfinite(val) and val < 0:   # "-19 % below" read as a slowdown (QA E2E-04)
            val, unit = -val, unit.replace("below", "above")
        shown = "∞ (no transfers)" if np.isinf(val) else num(val, {"census_growth": "+.1f", "cbp_holding": ".2f"}.get(key, ".1f"))
        parts.append(f"{spec['label']}: **{state}** ({shown} {unit})")
    st.caption(" · ".join(parts))
    share = fp["overall_level"].value_counts(normalize=True)
    st.caption("Share of reported days in this period: " + " · ".join(f"{PIPE_STATUS[lv][0]} {PIPE_STATUS[lv][1]} {share.get(float(lv), 0):.0%}" for lv in (0, 1, 2))
               + (f" · ❔ insufficient data {fp['overall_level'].isna().mean():.0%}" if fp["overall_level"].isna().any() else ""))

    st.plotly_chart(pviz.population_figure(rep, start, end), key="pa_population")
    st.plotly_chart(pviz.flows_figure(rep, start, end), key="pa_flows")
    left, right = st.columns(2)
    left.plotly_chart(pviz.rates_figure(flags, th, start, end), key="pa_rates")
    right.plotly_chart(pviz.backlog_figure(flags, th, start, end), key="pa_backlog")

    episodes = alert_episodes(fp, min_level=1)
    st.markdown("**Detected pressure episodes** (consecutive reported days at Watch or Alert; episodes are cut at the period boundaries)")
    if fp["overall_level"].isna().all():                   # nothing evaluated is not the same as "all clear" (QA E2E-05)
        st.info("No indicator could be evaluated in this period: the trailing windows need earlier reports. Choose a later or longer period.")
    elif episodes.empty:
        st.success("No indicator reached Watch or Alert in this period.")
    else:
        shown = episodes.assign(**{"Peak value": episodes["Peak value"].map(lambda v: "∞" if np.isinf(v) else f"{v:,.2f}")})
        st.dataframe(shown, hide_index=True, column_config={"Start": st.column_config.DateColumn(format="DD MMM YYYY"),
                                                            "End": st.column_config.DateColumn(format="DD MMM YYYY")})

    gran = st.radio("Summary table", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key="pa_gran",
                    help="Weekly = Sunday–Saturday weeks labelled by the Sunday. Stocks show first/last/mean reports; flows show reported-day totals and means.")
    table = aggregate(p, gran.lower())
    fmt = {c: st.column_config.NumberColumn(format="%.2f") for c in ("Discharge rate %/day", "CBP holding time (days)", "Recorded throughput")}
    fmt.update({c: st.column_config.NumberColumn(format="%.1f") for c in ("Transfers / reported day", "Discharges / reported day",
                                                                          "HHS census (mean)", "CBP custody (mean)")})
    st.dataframe(table, height=320, column_config={"Period": st.column_config.DateColumn(format="DD MMM YYYY"), **fmt})

    explorer = add_daily_metrics(p).join(fp.drop(columns=["indicators_available"]))
    with st.expander(f"Data explorer: {len(explorer):,} reported days with derived metrics and alert levels"):
        st.dataframe(explorer, height=360, column_config={"_index": st.column_config.DateColumn("date", format="DD MMM YYYY")})
    days_row = {"KPI": "Reported days in period", "Value": k["reported_days"],
                "Definition": f"Reports between {k['start']:%d %b %Y} and {k['end']:%d %b %Y}; Friday/Saturday are not published (QA E2E-06)"}
    kpi_rows = [days_row] + [{"KPI": label, "Value": k[key], "Definition": definition} for key, (label, definition) in KPI_DEFINITIONS.items()]
    stamp = f"{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}"
    d1, d2, d3, d4 = st.columns(4)               # four distinct file names: "daily" summary and explorer used to collide (QA E2E-02)
    d1.download_button("KPI summary (CSV)", pd.DataFrame(kpi_rows).to_csv(index=False).encode("utf-8"), f"uac_pipeline_kpis_{stamp}.csv",
                       "text/csv", on_click="ignore", key="pa_dl_kpis")
    d2.download_button(f"{gran} summary (CSV)", table.to_csv().encode("utf-8"), f"uac_pipeline_summary_{gran.lower()}_{stamp}.csv",
                       "text/csv", on_click="ignore", key="pa_dl_table")
    d3.download_button("Explorer data (CSV)", explorer.to_csv(index_label="date").encode("utf-8"), f"uac_pipeline_explorer_{stamp}.csv",
                       "text/csv", on_click="ignore", key="pa_dl_explorer")
    d4.download_button("Alert episodes (CSV)", episodes.to_csv(index=False).encode("utf-8"), f"uac_pipeline_alerts_{stamp}.csv",
                       "text/csv", on_click="ignore", key="pa_dl_episodes")


with tabs[6]:
    st.subheader("Pipeline analytics: CBP custody → HHS care, transfers, discharges and backlog")
    st.caption("Descriptive analytics of the published reports, recomputed for the period you choose. **Reported days only**: Friday/Saturday are never "
               "published and are not imputed here. **Stocks** (children in custody or care) are point-in-time counts described by first, last, mean or "
               "peak values, never summed; **flows** (apprehended, transferred, discharged) are daily counts summed over reported days.")
    if PIPELINE_ERROR:
        st.error(PIPELINE_ERROR)
    else:
        render_pipeline_tab()
