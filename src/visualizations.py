"""Plotly figures for the dashboard's "Pipeline analytics" tab.

Each builder takes the full reported-day history plus the selected period, so rolling windows at the start of a
period still use the preceding reports, and returns a figure containing only dates inside the period.
Colours follow the dashboard's validated palette: one fixed colour per flow (categorical slots 1-3 in pipeline order),
grey for observed stocks, and the reserved status colours (always paired with an icon and a label) for alert levels.
Different units never share an axis: they get separate panels.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .bottleneck_detection import LEVELS
from .kpi_calculations import filter_period

OBSERVED = "#898781"
FLOW_COLORS = {"apprehended": "#2a78d6", "transferred": "#eb6834", "discharged": "#1baf7a"}
FLOW_LABELS = {"apprehended": "Apprehended (CBP intake)", "transferred": "Transferred into HHS", "discharged": "Discharged from HHS"}
LEVEL_STYLE = {0: ("#0ca30c", "✅"), 1: ("#fab219", "⚠️"), 2: ("#d03b3b", "🛑")}
INSUFFICIENT = "#c3c2b7"


def _layout(fig, height, legend=True):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), hovermode="x unified", showlegend=legend,
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, title=None))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    return fig


def _threshold(fig, y, level, text, row=None):
    color, icon = LEVEL_STYLE[level]
    kw = dict(row=row, col=1) if row else {}
    fig.add_hline(y=y, line=dict(color=color, width=1, dash="dash"), annotation_text=f"{icon} {text}",
                  annotation_position="top left", annotation_font=dict(size=11, color="#52514e"), **kw)


def population_figure(rep, start, end):
    """Children in HHS care and in CBP custody (stocks) on reported days: two panels, because the scales differ ~100x."""
    p = filter_period(rep, start, end)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Children in HHS care (reported days)", "Children in CBP custody (reported days)"))
    marker = len(p) <= 120
    for row, col in ((1, "hhs_care"), (2, "cbp_custody")):
        fig.add_trace(go.Scatter(x=p.index, y=p[col], mode="lines+markers" if marker else "lines", name=fig.layout.annotations[row - 1].text,
                                 line=dict(color=OBSERVED, width=2), marker=dict(size=8, color=OBSERVED, line=dict(color="white", width=2)),
                                 connectgaps=False, hovertemplate="%{y:,.0f}<extra></extra>"), row=row, col=1)
        fig.update_yaxes(tickformat=",.0f", title="children", row=row, col=1)
    if len(p) and p["cbp_custody"].isna().all():
        fig.add_annotation(text="CBP values masked in this period (copied records, 12 Jan 2023 - 30 Jan 2024)", xref="x2 domain", yref="y2 domain",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="#52514e"))
    return _layout(fig, 520, legend=False)


def flows_figure(rep, start, end, window_days=7):
    """Daily flows as trailing 7-day means per reported day (computed on the full history, then cut to the period)."""
    fig = go.Figure()
    for col in ("apprehended", "transferred", "discharged"):
        s = rep[col].rolling(f"{window_days}D", min_periods=3).mean()
        s = filter_period(s.to_frame(), start, end)[col]
        fig.add_trace(go.Scatter(x=s.index, y=s, name=FLOW_LABELS[col], line=dict(color=FLOW_COLORS[col], width=2),
                                 connectgaps=False, hovertemplate="%{y:,.1f}/day<extra>" + FLOW_LABELS[col] + "</extra>"))
    fig.update_yaxes(tickformat=",.0f", title="children per reported day")
    fig.update_layout(title=dict(text=f"Flows: {window_days}-day mean per reported day", font=dict(size=14)))
    return _layout(fig, 380)


def rates_figure(flags, th, start, end):
    """Discharge rate vs its 8-week baseline, and implied CBP holding time vs the Watch/Alert thresholds (two panels)."""
    p = filter_period(flags, start, end)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                        subplot_titles=("HHS discharge rate, 7-day (% of census discharged per day)",
                                        "Implied CBP holding time, 7-day (custody ÷ transfers, days)"))
    fig.add_trace(go.Scatter(x=p.index, y=p["discharge_rate_7d_pct"], name="Discharge rate (7-day)", line=dict(color=FLOW_COLORS["discharged"], width=2),
                             hovertemplate="%{y:.2f}%/day<extra>7-day rate</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["discharge_baseline_pct"], name="Baseline (median of previous 8 weeks)",
                             line=dict(color=OBSERVED, width=2, dash="dot"), hovertemplate="%{y:.2f}%/day<extra>baseline</extra>"), row=1, col=1)
    hold = p["cbp_hold_days"].replace(np.inf, np.nan)
    fig.add_trace(go.Scatter(x=p.index, y=hold, name="CBP holding time (7-day)", line=dict(color=FLOW_COLORS["transferred"], width=2),
                             connectgaps=False, hovertemplate="%{y:.1f} days<extra>holding time</extra>"), row=2, col=1)
    _threshold(fig, th.cbp_hold_watch_days, 1, f"Watch ≥ {th.cbp_hold_watch_days:g} days", row=2)
    _threshold(fig, th.cbp_hold_alert_days, 2, f"Alert ≥ {th.cbp_hold_alert_days:g} days (72-hour standard)", row=2)
    fig.update_yaxes(tickformat=".2f", ticksuffix="%", row=1, col=1)
    fig.update_yaxes(tickformat=".1f", title="days", row=2, col=1)
    if np.isinf(p["cbp_hold_days"]).any():
        fig.add_annotation(text=f"{int(np.isinf(p['cbp_hold_days']).sum())} day(s) with no transfers in the window (unbounded holding time) are not drawn",
                           xref="x2 domain", yref="y2 domain", x=0, y=1, showarrow=False, xanchor="left", font=dict(size=11, color="#52514e"))
    return _layout(fig, 560)


def backlog_figure(flags, th, start, end):
    """Census growth vs the Watch/Alert thresholds, with a strip of the daily overall alert status underneath."""
    p = filter_period(flags, start, end)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.78, 0.22],
                        subplot_titles=(f"HHS census growth: % change vs {th.growth_window_days} days earlier", "Overall alert status (worst indicator)"))
    fig.add_trace(go.Scatter(x=p.index, y=p["census_growth_pct"], name="Census growth", line=dict(color=OBSERVED, width=2),
                             connectgaps=False, hovertemplate="%{y:+.1f}%<extra>census growth</extra>"), row=1, col=1)
    _threshold(fig, th.census_growth_watch_pct, 1, f"Watch ≥ +{th.census_growth_watch_pct:g}%", row=1)
    _threshold(fig, th.census_growth_alert_pct, 2, f"Alert ≥ +{th.census_growth_alert_pct:g}%", row=1)
    fig.update_yaxes(tickformat=",.0f", ticksuffix="%", row=1, col=1)
    for level, name in list(LEVELS.items()) + [(None, "Insufficient data")]:
        mask = p["overall_level"].isna() if level is None else p["overall_level"] == level
        if not mask.any():
            continue
        color, icon = (INSUFFICIENT, "❔") if level is None else LEVEL_STYLE[level]
        fig.add_trace(go.Bar(x=p.index[mask], y=np.ones(int(mask.sum())), name=f"{icon} {name}", marker=dict(color=color, line=dict(width=0)),
                             hovertemplate=f"{icon} {name}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_layout(barmode="overlay", bargap=0)
    return _layout(fig, 480)
