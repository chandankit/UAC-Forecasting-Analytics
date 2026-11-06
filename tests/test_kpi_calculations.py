"""KPI calculations, date filtering and aggregation: hand-computed examples, edge cases and real-data cross-checks."""
import numpy as np
import pandas as pd
import pytest

from conftest import RAW_CSV, make_rep
from src.kpi_calculations import (AGG_COLUMNS, KPI_DEFINITIONS, add_daily_metrics, aggregate, filter_period, period_kpis, safe_ratio)


@pytest.fixture
def toy():
    """Three reported days with hand-checkable values (Sun 1 - Tue 3 June 2025)."""
    return make_rep(pd.to_datetime(["2025-06-01", "2025-06-02", "2025-06-03"]),
                    app=[10, 20, 30], cbp=[40, 50, 60], tr=[20, 10, 30], care=[1000, 1010, 990], dis=[10, 20, 0])


# ----------------------------------------------------------------------------- safe_ratio
@pytest.mark.parametrize("num, den, expected", [(6, 3, 2.0), (0, 5, 0.0), (1, 0, np.nan), (1, -0.0, np.nan), (1, np.nan, np.nan),
                                                (np.nan, 5, np.nan), (1, np.inf, np.nan)])
def test_safe_ratio(num, den, expected):
    got = safe_ratio(num, den)
    assert (np.isnan(got) and np.isnan(expected)) or got == expected


# ----------------------------------------------------------------------------- period_kpis
def test_period_kpis_hand_computed(toy):
    k = period_kpis(toy)
    assert k["reported_days"] == 3 and k["start"] == pd.Timestamp("2025-06-01") and k["end"] == pd.Timestamp("2025-06-03")
    # stock: first / last / change / mean / peak, never a sum
    assert (k["hhs_care_start"], k["hhs_care_end"], k["hhs_care_change"]) == (1000, 990, -10)
    assert k["hhs_care_change_pct"] == pytest.approx(-1.0)
    assert k["hhs_care_mean"] == pytest.approx(1000) and k["hhs_care_peak"] == 1010 and k["hhs_care_peak_date"] == pd.Timestamp("2025-06-02")
    assert (k["cbp_custody_end"], k["cbp_custody_change"]) == (60, 20)
    # flows: reported-day totals and per-day means
    assert (k["transferred_total"], k["transferred_per_day"]) == (60, 20)
    assert (k["discharged_total"], k["discharged_per_day"]) == (30, 10)
    assert k["apprehended_per_day"] == 20
    assert k["net_intake_per_day"] == pytest.approx(10)        # mean of (10, -10, 30)
    # ratios of sums
    assert k["transfer_rate_pct"] == pytest.approx(40.0)       # 60 / 150
    assert k["discharge_rate_pct"] == pytest.approx(1.0)       # 30 / 3000
    assert k["throughput_ratio"] == pytest.approx(0.5)         # 30 / 60
    assert k["implied_stay_days"] == pytest.approx(100.0)      # 3000 / 30 = 100 / discharge rate
    assert k["implied_cbp_hold_days"] == pytest.approx(2.5)    # 150 / 60
    assert k["pass_through_ratio"] == pytest.approx(1.0)       # 60 / 60


def test_stock_change_is_not_a_sum(toy):
    k = period_kpis(toy)
    assert k["hhs_care_change"] != toy["hhs_care"].sum() and k["hhs_care_change"] == toy["hhs_care"].iloc[-1] - toy["hhs_care"].iloc[0]


def test_zero_denominators_give_nan_not_inf(toy):
    t = toy.assign(transferred=0.0, discharged=0.0)
    k = period_kpis(t)
    assert k["transfer_rate_pct"] == 0 and k["discharge_rate_pct"] == 0 and k["pass_through_ratio"] == 0   # zero numerators are real zeros
    for key in ("throughput_ratio", "implied_stay_days", "implied_cbp_hold_days"):                          # zero denominators
        assert np.isnan(k[key]), key
    assert np.isnan(period_kpis(toy.assign(apprehended=0.0))["pass_through_ratio"])
    t2 = toy.assign(hhs_care=0.0)
    assert np.isnan(period_kpis(t2)["hhs_care_change_pct"]) and np.isnan(period_kpis(t2)["discharge_rate_pct"])


def test_missing_values_are_paired_not_zero_filled(toy):
    t = toy.copy()
    t.iloc[1, t.columns.get_loc("cbp_custody")] = np.nan
    k = period_kpis(t)
    assert k["transfer_rate_pct"] == pytest.approx(50.0)       # (20 + 30) / (40 + 60): day 2 dropped from both sums
    assert k["implied_cbp_hold_days"] == pytest.approx(2.0)
    assert k["transferred_per_day"] == 20                      # the flow itself still uses all three days


def test_masked_cbp_period_gives_nan_cbp_kpis(toy):
    t = toy.assign(apprehended=np.nan, cbp_custody=np.nan, transferred=np.nan)
    k = period_kpis(t)
    for key in ("cbp_custody_end", "transferred_per_day", "transfer_rate_pct", "implied_cbp_hold_days", "throughput_ratio", "net_intake_per_day"):
        assert np.isnan(k[key]), key
    assert k["cbp_custody_days"] == 0 and k["discharge_rate_pct"] == pytest.approx(1.0)   # HHS KPIs unaffected


def test_empty_period_returns_all_keys_as_nan(toy):
    k_full, k_empty = period_kpis(toy), period_kpis(toy.iloc[0:0])
    assert set(k_full) == set(k_empty) and k_empty["reported_days"] == 0 and pd.isna(k_empty["start"])
    assert all(pd.isna(v) for key, v in k_empty.items() if key not in ("reported_days",) and not key.endswith("_days"))


def test_every_documented_kpi_is_computed(toy):
    k = period_kpis(toy)
    assert set(KPI_DEFINITIONS) <= set(k)


# ----------------------------------------------------------------------------- filter_period
def test_filter_period_is_inclusive_and_ignores_time_of_day(toy):
    assert len(filter_period(toy, "2025-06-01", "2025-06-03")) == 3
    assert list(filter_period(toy, "2025-06-02", "2025-06-02 08:30").index) == [pd.Timestamp("2025-06-02")]
    assert filter_period(toy, "2025-07-01", "2025-07-31").empty


def test_filter_period_rejects_invalid_ranges(toy):
    with pytest.raises(ValueError, match="after the end date"):
        filter_period(toy, "2025-06-03", "2025-06-01")
    with pytest.raises(ValueError, match="required"):
        filter_period(toy, None, "2025-06-01")
    with pytest.raises(ValueError):
        filter_period(toy, "not a date", "2025-06-01")


def test_date_filter_matches_independent_raw_computation(rep):
    """Current-regime KPIs equal a direct computation on the raw export (stale reports excluded), not just the cleaned table."""
    from src.data_loader import load_raw_csv
    raw, _ = load_raw_csv(RAW_CSV)
    raw = raw.loc[raw.index.isin(rep.index)].loc["2025-03-17":"2025-12-21"]
    k = period_kpis(filter_period(rep, "2025-03-17", "2025-12-21"))
    assert k["reported_days"] == len(raw) == 185
    assert k["hhs_care_end"] == 2484 and k["hhs_care_peak"] == raw["hhs_care"].max()
    assert k["transferred_per_day"] == pytest.approx(raw["transferred"].mean())
    assert k["discharge_rate_pct"] == pytest.approx(100 * raw["discharged"].sum() / raw["hhs_care"].sum())
    assert k["implied_cbp_hold_days"] == pytest.approx(raw["cbp_custody"].sum() / raw["transferred"].sum())


# ----------------------------------------------------------------------------- aggregation
def test_weekly_weeks_run_sunday_to_saturday_labelled_by_sunday():
    dates = pd.to_datetime(["2025-06-01", "2025-06-05", "2025-06-07", "2025-06-08"])     # Sun, Thu, Sat | Sun
    t = make_rep(dates, care=[100, 110, 120, 130], tr=[1, 2, 3, 4], dis=[1, 1, 1, 1])
    w = aggregate(t, "weekly")
    assert list(w.index) == [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-08")]
    first = w.iloc[0]
    assert first["Reported days"] == 3 and first["HHS census (first)"] == 100 and first["HHS census (last)"] == 120
    assert first["HHS census change"] == 20 and first["Transferred (reported-day total)"] == 6
    assert first["Transfers / reported day"] == pytest.approx(2.0)


def test_monthly_uses_last_report_for_stocks_and_sums_for_flows():
    dates = pd.to_datetime(["2025-05-29", "2025-06-01", "2025-06-30"])
    m = aggregate(make_rep(dates, care=[500, 600, 700], dis=[5, 6, 7]), "monthly")
    assert list(m.index) == [pd.Timestamp("2025-05-01"), pd.Timestamp("2025-06-01")]
    assert m.loc["2025-06-01", "HHS census (last)"] == 700 and m.loc["2025-06-01", "Discharged (reported-day total)"] == 13


def test_aggregate_matches_period_kpis_on_real_data(rep):
    for freq in ("daily", "weekly", "monthly"):
        got = aggregate(rep, freq)
        key = rep.index if freq == "daily" else rep.index.to_period("W-SAT" if freq == "weekly" else "M").start_time
        ref = pd.DataFrame([{"Period": k, **period_kpis(g)} for k, g in rep.groupby(key)]).set_index("Period")
        ref = ref[list(AGG_COLUMNS)].rename(columns=AGG_COLUMNS)
        assert got.shape == ref.shape
        np.testing.assert_allclose(got.to_numpy(float), ref.to_numpy(float), rtol=1e-12, equal_nan=True, err_msg=freq)


def test_aggregate_edge_cases(toy):
    with pytest.raises(ValueError, match="Unknown frequency"):
        aggregate(toy, "hourly")
    empty = aggregate(toy.iloc[0:0], "weekly")
    assert empty.empty and list(empty.columns) == list(AGG_COLUMNS.values())


def test_add_daily_metrics(toy):
    d = add_daily_metrics(toy.assign(hhs_care=[1000, 0, 990]))
    assert d["net_intake"].tolist() == [10, -10, 30]
    assert d["discharge_rate_pct"].iloc[0] == pytest.approx(1.0) and np.isnan(d["discharge_rate_pct"].iloc[1])   # zero census -> NaN
    assert d["transfer_rate_pct"].iloc[0] == pytest.approx(50.0)
    thu_sun = add_daily_metrics(make_rep(pd.to_datetime(["2025-06-05", "2025-06-08"])))
    assert thu_sun["days_since_prev_report"].iloc[1] == 3
