"""Stock/flow-correct KPIs for a period of reported days, plus daily/weekly/monthly aggregation.

Rules that keep the numbers honest:
* Stocks (children in CBP custody, children in HHS care) are point-in-time counts. A period is described by its first,
  last, mean and peak report, and its change is last - first. Stocks are never summed over time.
* Flows (apprehended, transferred, discharged) are daily counts. Totals sum reported days only (Friday/Saturday are not
  published, so a "total" is a reported-day total, not a calendar total); per-day values are means over reported days.
* Ratios are ratios of sums over the days where both inputs are observed (robust to single zero days), and are NaN,
  never +/-inf, when the denominator is zero or missing.
"""
import numpy as np
import pandas as pd

from .data_loader import FLOWS, STOCKS

# One definition table, used by the dashboard tooltips, the README and the research paper's Appendix volume (Appendix A)
KPI_DEFINITIONS = {
    "hhs_care_end": ("HHS census at period end", "Children in HHS care on the last reported day of the period (stock)"),
    "hhs_care_change": ("HHS census change", "Last minus first reported HHS census in the period (stock change, not a sum)"),
    "hhs_care_peak": ("Peak HHS census", "Highest reported HHS census in the period, with its date"),
    "cbp_custody_end": ("CBP custody at period end", "Children in CBP custody on the last trusted report of the period (stock)"),
    "apprehended_per_day": ("Apprehensions per reported day", "Mean of daily CBP apprehensions over reported days"),
    "transferred_per_day": ("Transfers into HHS per reported day", "Mean of daily transfers out of CBP into HHS over reported days"),
    "discharged_per_day": ("Discharges per reported day", "Mean of daily HHS discharges (mostly to sponsors) over reported days"),
    "net_intake_per_day": ("Recorded net intake per reported day", "Mean of (transfers - discharges) on days with both reported. "
                           "Recorded flows do not close the census balance (about +3 children/day unexplained), so use the census change for backlog"),
    "transfer_rate_pct": ("CBP transfer rate (%/day)", "100 x sum(transfers) / sum(children in CBP custody): share of the CBP population moved to HHS per day"),
    "discharge_rate_pct": ("HHS discharge rate (%/day)", "100 x sum(discharges) / sum(children in HHS care): share of the HHS population discharged per day"),
    "throughput_ratio": ("Recorded throughput (exits per entry)", "sum(discharges) / sum(transfers): above 1 = recorded exits exceed entries. "
                         "Not used for alerts: it misses unreported Fri/Sat flows and non-CBP admissions"),
    "implied_stay_days": ("Implied average stay in HHS care (days)", "Little's law, ratio of period means: sum(children in HHS care) / sum(discharges) = 100 / discharge rate"),
    "implied_cbp_hold_days": ("Implied CBP holding time (days)", "Little's law: sum(children in CBP custody) / sum(transfers). An approximation of the average time in "
                              "CBP custody before transfer (an upper bound if children also leave CBP by other routes); compare with the 72-hour transfer standard"),
    "pass_through_ratio": ("CBP pass-through (transfers per apprehension)", "sum(transfers) / sum(apprehensions) on days with both reported"),
}


def safe_ratio(num, den):
    """num / den as a float, or NaN when either is missing or the denominator is zero (never +/-inf)."""
    num, den = float(num), float(den)
    return num / den if np.isfinite(num) and np.isfinite(den) and den != 0 else np.nan


def filter_period(frame, start, end):
    """Rows with start <= date <= end (both inclusive, whole days). Raises ValueError for invalid or reversed dates."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Start and end dates are required")
    start, end = start.normalize(), end.normalize()
    if start > end:
        raise ValueError(f"The start date ({start:%d %b %Y}) is after the end date ({end:%d %b %Y})")
    days = frame.index.normalize()
    return frame.loc[(days >= start) & (days <= end)]


def _paired_sums(frame, num, den):
    both = frame[num].notna() & frame[den].notna()
    return frame.loc[both, num].sum(), frame.loc[both, den].sum(), int(both.sum())


def period_kpis(rep):
    """KPIs for one period of reported days (see KPI_DEFINITIONS). Every key is present; values are NaN when undefined."""
    out = {"reported_days": len(rep),
           "start": rep.index.min() if len(rep) else pd.NaT,
           "end": rep.index.max() if len(rep) else pd.NaT}
    for stock in STOCKS:
        s = rep[stock].dropna() if stock in rep else pd.Series(dtype=float)
        first, last = (s.iloc[0], s.iloc[-1]) if len(s) else (np.nan, np.nan)
        out.update({f"{stock}_start": first, f"{stock}_end": last, f"{stock}_change": last - first,
                    f"{stock}_change_pct": 100 * safe_ratio(last - first, first),
                    f"{stock}_mean": s.mean() if len(s) else np.nan,
                    f"{stock}_peak": s.max() if len(s) else np.nan,
                    f"{stock}_peak_date": s.idxmax() if len(s) else pd.NaT,
                    f"{stock}_days": len(s)})
    for flow in FLOWS:
        f = rep[flow].dropna() if flow in rep else pd.Series(dtype=float)
        out.update({f"{flow}_total": f.sum() if len(f) else np.nan,
                    f"{flow}_per_day": f.mean() if len(f) else np.nan,
                    f"{flow}_days": len(f)})

    both = rep["transferred"].notna() & rep["discharged"].notna()
    out["net_intake_per_day"] = (rep.loc[both, "transferred"] - rep.loc[both, "discharged"]).mean() if both.any() else np.nan
    tr, cbp, _ = _paired_sums(rep, "transferred", "cbp_custody")
    dis, care, _ = _paired_sums(rep, "discharged", "hhs_care")
    dis2, tr2, _ = _paired_sums(rep, "discharged", "transferred")
    tr3, app, _ = _paired_sums(rep, "transferred", "apprehended")
    out["transfer_rate_pct"] = 100 * safe_ratio(tr, cbp)
    out["discharge_rate_pct"] = 100 * safe_ratio(dis, care)
    out["throughput_ratio"] = safe_ratio(dis2, tr2)
    out["implied_stay_days"] = safe_ratio(care, dis)
    out["implied_cbp_hold_days"] = safe_ratio(cbp, tr)
    out["pass_through_ratio"] = safe_ratio(tr3, app)
    return out


def add_daily_metrics(rep):
    """Per-reported-day derived columns: net intake, rates, census change since the previous report and the gap in days."""
    out = rep.copy()
    out["net_intake"] = out["transferred"] - out["discharged"]
    out["discharge_rate_pct"] = 100 * out["discharged"] / out["hhs_care"].where(out["hhs_care"] > 0)
    out["transfer_rate_pct"] = 100 * out["transferred"] / out["cbp_custody"].where(out["cbp_custody"] > 0)
    out["hhs_care_change"] = out["hhs_care"].diff()
    out["days_since_prev_report"] = out.index.to_series().diff().dt.days.to_numpy()
    return out


AGG_COLUMNS = {  # period_kpis key -> table column
    "reported_days": "Reported days",
    "hhs_care_start": "HHS census (first)", "hhs_care_end": "HHS census (last)", "hhs_care_mean": "HHS census (mean)",
    "hhs_care_change": "HHS census change",
    "cbp_custody_end": "CBP custody (last)", "cbp_custody_mean": "CBP custody (mean)",
    "apprehended_total": "Apprehended (reported-day total)", "transferred_total": "Transferred (reported-day total)",
    "discharged_total": "Discharged (reported-day total)",
    "transferred_per_day": "Transfers / reported day", "discharged_per_day": "Discharges / reported day",
    "discharge_rate_pct": "Discharge rate %/day", "implied_stay_days": "Implied stay (days)",
    "implied_cbp_hold_days": "CBP holding time (days)", "throughput_ratio": "Recorded throughput",
}


def aggregate(rep, freq):
    """Summarise reported days by 'daily', 'weekly' (Sunday-Saturday, labelled by the Sunday) or 'monthly'.

    Vectorised, with exactly the period_kpis definitions (tests/test_kpi_calculations.py checks the two agree):
    stocks use their first/last/mean report, flows their reported-day sums and means, ratios the sums over days where
    both inputs are observed. A period with fewer reported days than usual simply reports fewer days.
    """
    if freq not in ("daily", "weekly", "monthly"):
        raise ValueError(f"Unknown frequency '{freq}'; use daily, weekly or monthly")
    if rep.empty:
        return pd.DataFrame(columns=list(AGG_COLUMNS.values()), index=pd.DatetimeIndex([], name="Period"))
    key = rep.index if freq == "daily" else rep.index.to_period("W-SAT" if freq == "weekly" else "M").start_time
    g = rep.groupby(key, sort=True)
    paired = {}
    for num_col, den_col in (("discharged", "hhs_care"), ("cbp_custody", "transferred"), ("discharged", "transferred")):
        both = rep[num_col].notna() & rep[den_col].notna()
        sums = rep[[num_col, den_col]].where(both, axis=0).groupby(key, sort=True).sum(min_count=1)
        paired[(num_col, den_col)] = (sums[num_col], sums[den_col])

    def ratio(num, den):
        return num / den.where(den != 0)

    care_first, care_last = g["hhs_care"].first(), g["hhs_care"].last()
    out = pd.DataFrame({
        "reported_days": g.size(),
        "hhs_care_start": care_first, "hhs_care_end": care_last, "hhs_care_mean": g["hhs_care"].mean(),
        "hhs_care_change": care_last - care_first,
        "cbp_custody_end": g["cbp_custody"].last(), "cbp_custody_mean": g["cbp_custody"].mean(),
        "apprehended_total": g["apprehended"].sum(min_count=1), "transferred_total": g["transferred"].sum(min_count=1),
        "discharged_total": g["discharged"].sum(min_count=1),
        "transferred_per_day": g["transferred"].mean(), "discharged_per_day": g["discharged"].mean(),
        "discharge_rate_pct": 100 * ratio(*paired[("discharged", "hhs_care")]),
        "implied_stay_days": ratio(*reversed(paired[("discharged", "hhs_care")])),
        "implied_cbp_hold_days": ratio(*paired[("cbp_custody", "transferred")]),
        "throughput_ratio": ratio(*paired[("discharged", "transferred")]),
    })
    out.index = pd.DatetimeIndex(out.index, name="Period")
    return out[list(AGG_COLUMNS)].rename(columns=AGG_COLUMNS)
