"""Rule-based backlog and bottleneck detection on reported days, with documented, adjustable thresholds.

Three indicators are evaluated on every reported day using trailing windows only (no look-ahead), so a historical
alert is exactly what an analyst would have seen on that day:

1. census_growth       Backlog building: % change in children in HHS care versus the last report at least 28 days earlier.
                       The census (a stock) is used because recorded flows do not close the population balance.
2. discharge_slowdown  Placement bottleneck: the 7-day discharge rate (sum discharges / sum census) falls below its
                       median over the previous 8 weeks.
3. cbp_holding         Transfer bottleneck: implied CBP holding time = sum(CBP custody) / sum(transfers) over 7 days.
                       The 3-day alert level mirrors the 72-hour transfer standard for unaccompanied children
                       (TVPRA 2008, 8 U.S.C. 1232(b)(3)); the stock/flow ratio approximates an average, it does not
                       measure any individual child's custody time.

Levels: 0 = Normal, 1 = Watch, 2 = Alert. A day's overall level is the worst indicator level ("worst wins").
Indicators with too few reports in their window are NaN (insufficient data) and are ignored for the overall level.

Recorded throughput (discharges / transfers) was tested as a backlog signal and rejected: over the trusted period
(31 Jan 2024 - 25 Jan 2025) it averaged 0.93, i.e. recorded entries exceeded exits, yet the census fell by about 3,500.
Its month-to-month direction often agrees with the census, but its level is biased because Friday/Saturday flows are
unpublished and recorded flows leave about 3 children a day unexplained (research paper, Section 2.3; Appendix B.6 has the detail).
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

LEVELS = {0: "Normal", 1: "Watch", 2: "Alert"}
INDICATORS = {
    "census_growth": {"label": "HHS census growth (backlog building)", "value": "census_growth_pct", "unit": "% vs 28 days earlier"},
    "discharge_slowdown": {"label": "Discharge slowdown (placement bottleneck)", "value": "discharge_drop_pct", "unit": "% below 8-week median rate"},
    "cbp_holding": {"label": "CBP holding time (transfer bottleneck)", "value": "cbp_hold_days", "unit": "days (custody / transfers, 7-day)"},
}


@dataclass(frozen=True)
class Thresholds:
    """Alert rules. Defaults are operational choices documented in the README and research paper, not fitted to the data."""
    census_growth_watch_pct: float = 5.0      # census up 5% in four weeks
    census_growth_alert_pct: float = 10.0     # census up 10% in four weeks
    growth_window_days: int = 28
    discharge_drop_watch_pct: float = 15.0    # 7-day discharge rate 15% below its recent norm
    discharge_drop_alert_pct: float = 30.0
    rate_window_days: int = 7
    baseline_days: int = 56                   # 8-week trailing baseline, ending where the 7-day window starts
    cbp_hold_watch_days: float = 2.0          # 48 hours
    cbp_hold_alert_days: float = 3.0          # 72-hour transfer standard
    min_reports: int = 3                      # minimum reports in a 7-day window
    min_baseline_reports: int = 15            # minimum 7-day rates in the 8-week baseline

    def validate(self):
        pairs = [("census growth", self.census_growth_watch_pct, self.census_growth_alert_pct),
                 ("discharge slowdown", self.discharge_drop_watch_pct, self.discharge_drop_alert_pct),
                 ("CBP holding time", self.cbp_hold_watch_days, self.cbp_hold_alert_days)]
        for name, watch, alert in pairs:
            if not (np.isfinite(watch) and np.isfinite(alert)) or watch > alert:
                raise ValueError(f"{name}: the Watch threshold ({watch}) must be finite and not above the Alert threshold ({alert})")
        if self.cbp_hold_watch_days <= 0 or self.discharge_drop_watch_pct <= 0:
            raise ValueError("CBP holding and discharge-slowdown thresholds must be positive")
        for name in ("growth_window_days", "rate_window_days", "baseline_days", "min_reports", "min_baseline_reports"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        return self

    def as_tuple(self):
        return tuple(asdict(self).values())


def _window_sums(rep, num, den, days, min_reports):
    """Trailing time-window sums of num and den over reported days where both are observed; count of such days."""
    both = rep[num].notna() & rep[den].notna()
    roll = lambda s: s.rolling(f"{days}D", min_periods=1).sum()
    n, d, k = roll(rep[num].where(both, 0.0)), roll(rep[den].where(both, 0.0)), roll(both.astype(float))
    return n.where(k >= min_reports), d.where(k >= min_reports)


def compute_indicators(rep, th=Thresholds()):
    """Indicator values per reported day (rep must be sorted reported days with the five series)."""
    th.validate()
    out = pd.DataFrame(index=rep.index)
    if rep.empty:
        for c in ("census_growth_pct", "discharge_rate_7d_pct", "discharge_baseline_pct", "discharge_drop_pct", "cbp_hold_days"):
            out[c] = pd.Series(dtype=float)
        return out

    # 1. census growth: last report at or before t - window, accepted only if within 7 days of that date (not stale)
    care = rep["hhs_care"].dropna()
    left = pd.DataFrame({"date": rep.index, "lag_date": rep.index - pd.Timedelta(days=th.growth_window_days)})
    right = pd.DataFrame({"lag_date": care.index, "lag_care": care.to_numpy()})
    lagged = pd.merge_asof(left, right, on="lag_date", direction="backward", tolerance=pd.Timedelta(days=7))
    lag_care = pd.Series(lagged["lag_care"].to_numpy(), index=rep.index)
    out["census_growth_pct"] = 100 * (rep["hhs_care"] / lag_care.where(lag_care > 0) - 1)

    # 2. discharge slowdown: 7-day rate vs the median 7-day rate over the 8 weeks before the current window
    dis, care_sum = _window_sums(rep, "discharged", "hhs_care", th.rate_window_days, th.min_reports)
    rate = 100 * dis / care_sum.where(care_sum > 0)
    out["discharge_rate_7d_pct"] = rate
    shifted = pd.Series(rate.to_numpy(), index=rate.index + pd.Timedelta(days=th.rate_window_days)).dropna()
    base = shifted.rolling(f"{th.baseline_days}D", min_periods=th.min_baseline_reports).median()
    base_at = pd.merge_asof(pd.DataFrame({"date": rep.index}), pd.DataFrame({"date": base.index, "b": base.to_numpy()}),
                            on="date", direction="backward")["b"].to_numpy()
    out["discharge_baseline_pct"] = base_at
    out["discharge_drop_pct"] = 100 * (1 - rate / out["discharge_baseline_pct"].where(out["discharge_baseline_pct"] > 0))

    # 3. CBP holding time: zero transfers while children are held is an unbounded holding time (inf), i.e. an alert
    cbp, tr = _window_sums(rep, "cbp_custody", "transferred", th.rate_window_days, th.min_reports)
    hold = cbp / tr.where(tr > 0)
    out["cbp_hold_days"] = hold.mask((tr == 0) & (cbp > 0), np.inf)
    return out


def _level(values, watch, alert):
    lv = np.where(values >= alert, 2.0, np.where(values >= watch, 1.0, 0.0))
    return pd.Series(np.where(np.isnan(values.to_numpy(dtype=float)), np.nan, lv), index=values.index)


def detect_bottlenecks(rep, th=Thresholds()):
    """Indicator values, per-indicator levels (0/1/2, NaN = insufficient data) and the overall level/status per reported day."""
    ind = compute_indicators(rep, th)
    ind["census_growth_level"] = _level(ind["census_growth_pct"], th.census_growth_watch_pct, th.census_growth_alert_pct)
    ind["discharge_slowdown_level"] = _level(ind["discharge_drop_pct"], th.discharge_drop_watch_pct, th.discharge_drop_alert_pct)
    ind["cbp_holding_level"] = _level(ind["cbp_hold_days"], th.cbp_hold_watch_days, th.cbp_hold_alert_days)
    levels = ind[[f"{k}_level" for k in INDICATORS]]
    ind["indicators_available"] = levels.notna().sum(axis=1)
    ind["overall_level"] = levels.max(axis=1, skipna=True)          # NaN only when no indicator is available
    ind["overall_status"] = ind["overall_level"].map(LEVELS).fillna("Insufficient data")
    return ind


def alert_episodes(flags, min_level=1, max_gap_days=7):
    """Contiguous runs of reported days at or above min_level, per indicator.

    Consecutive reports belong to one episode unless they are more than max_gap_days apart (so the regular
    Thursday -> Sunday gap never splits an episode). Returns one row per episode, longest first within an indicator.
    """
    cols = ["Indicator", "Start", "End", "Reported days", "Calendar days", "Highest level", "Peak value", "Unit"]
    rows = []
    for key, spec in INDICATORS.items():
        lvl = flags.get(f"{key}_level")
        if lvl is None or lvl.empty:
            continue
        on = (lvl >= min_level).to_numpy()
        dates = flags.index
        gap = np.r_[np.inf, np.diff(dates.values).astype("timedelta64[D]").astype(float)]
        start_new = on & (~np.r_[False, on[:-1]] | (gap > max_gap_days))
        episode_id = np.cumsum(start_new) * on
        for eid in np.unique(episode_id[episode_id > 0]):
            idx = np.flatnonzero(episode_id == eid)
            vals = flags[spec["value"]].iloc[idx]
            peak = np.inf if np.isinf(vals).any() else vals.max()
            rows.append({"Indicator": spec["label"], "Start": dates[idx[0]], "End": dates[idx[-1]], "Reported days": len(idx),
                         "Calendar days": int((dates[idx[-1]] - dates[idx[0]]).days) + 1,
                         "Highest level": LEVELS[int(lvl.iloc[idx].max())], "Peak value": peak, "Unit": spec["unit"]})
    return pd.DataFrame(rows, columns=cols)
