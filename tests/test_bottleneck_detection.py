"""Bottleneck detection: documented thresholds, boundaries, no look-ahead, edge cases and the documented historical events."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_rep, reported_calendar
from src.bottleneck_detection import INDICATORS, LEVELS, Thresholds, alert_episodes, compute_indicators, detect_bottlenecks

TH = Thresholds()


def test_documented_default_thresholds():
    """The README and research paper quote these defaults; changing them must be a deliberate, documented decision."""
    assert (TH.census_growth_watch_pct, TH.census_growth_alert_pct, TH.growth_window_days) == (5.0, 10.0, 28)
    assert (TH.discharge_drop_watch_pct, TH.discharge_drop_alert_pct, TH.rate_window_days, TH.baseline_days) == (15.0, 30.0, 7, 56)
    assert (TH.cbp_hold_watch_days, TH.cbp_hold_alert_days) == (2.0, 3.0)
    assert Thresholds(*TH.as_tuple()) == TH


@pytest.mark.parametrize("kwargs", [dict(census_growth_watch_pct=12.0), dict(cbp_hold_watch_days=4.0), dict(discharge_drop_alert_pct=10.0),
                                    dict(cbp_hold_watch_days=0.0, cbp_hold_alert_days=1.0), dict(rate_window_days=0), dict(census_growth_alert_pct=np.inf)])
def test_invalid_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Thresholds(**kwargs).validate()


def test_steady_state_is_normal_after_warm_up():
    f = detect_bottlenecks(make_rep(reported_calendar(weeks=16)))      # constant flows: hold = 10/10 = 1 day, no growth, no slowdown
    assert f["overall_status"].iloc[0] == "Insufficient data"            # one report: no window is complete yet
    assert (f["overall_level"].iloc[2:] == 0).all()
    assert f["census_growth_pct"].dropna().eq(0).all() and f["cbp_hold_days"].dropna().eq(1).all()


@pytest.mark.parametrize("new_care, level", [(1049.9, 0), (1050.0, 1), (1099.9, 1), (1100.0, 2)])
def test_census_growth_boundaries(new_care, level):
    dates = reported_calendar(weeks=10)
    care = np.where(dates >= dates[40], new_care, 1000.0)
    f = detect_bottlenecks(make_rep(dates, care=care))
    assert f["census_growth_level"].iloc[40] == level                   # watch at >= 5%, alert at >= 10% (inclusive)


def test_census_growth_ignores_a_stale_lag():
    dates = reported_calendar(weeks=10)
    dates = dates[(dates < pd.Timestamp("2025-01-20")) | (dates > pd.Timestamp("2025-02-10"))]   # three-week reporting gap
    ind = compute_indicators(make_rep(dates))
    t = pd.Timestamp("2025-02-27")                                       # t - 28 days = 30 Jan, inside the gap; last report 16 Jan > 7 days earlier
    assert np.isnan(ind.at[t, "census_growth_pct"])


def test_discharge_slowdown_detected_after_a_drop():
    dates = reported_calendar(weeks=20)
    change = dates[60]
    f = detect_bottlenecks(make_rep(dates, dis=np.where(dates >= change, 6.0, 10.0)))   # 40% fewer discharges
    before = f.loc[: change - pd.Timedelta(days=1), "discharge_slowdown_level"].dropna()
    assert (before == 0).all()
    after = f.loc[change: change + pd.Timedelta(days=14)]
    assert (after["discharge_slowdown_level"] == 2).any()
    assert after["discharge_drop_pct"].max() == pytest.approx(40.0, abs=0.5)


@pytest.mark.parametrize("cbp, level", [(19.9, 0), (20.0, 1), (29.9, 1), (30.0, 2)])
def test_cbp_holding_time_boundaries(cbp, level):
    f = detect_bottlenecks(make_rep(reported_calendar(weeks=4), cbp=cbp, tr=10.0))
    assert (f["cbp_holding_level"].iloc[3:] == level).all()              # 2.0 days = watch, 3.0 days (72 h) = alert


def test_zero_transfers_with_children_held_is_an_unbounded_alert():
    dates = reported_calendar(weeks=4)
    f = detect_bottlenecks(make_rep(dates, tr=np.where(dates >= dates[8], 0.0, 10.0), cbp=15.0))
    assert np.isinf(f["cbp_hold_days"].iloc[-1]) and f["cbp_holding_level"].iloc[-1] == 2
    ep = alert_episodes(f, min_level=2)
    assert np.isinf(ep.loc[ep["Indicator"] == INDICATORS["cbp_holding"]["label"], "Peak value"]).all()


def test_zero_transfers_and_empty_custody_is_not_an_alert():
    f = detect_bottlenecks(make_rep(reported_calendar(weeks=3), tr=0.0, cbp=0.0))
    assert f["cbp_hold_days"].isna().all() and f["cbp_holding_level"].isna().all()


def test_masked_cbp_gives_insufficient_data_but_other_indicators_still_work():
    f = detect_bottlenecks(make_rep(reported_calendar(weeks=12), cbp=np.nan, tr=np.nan, app=np.nan))
    assert f["cbp_holding_level"].isna().all()
    assert (f["overall_level"].iloc[-10:] == 0).all() and (f["indicators_available"].iloc[-10:] == 2).all()


def test_overall_level_is_the_worst_indicator():
    dates = reported_calendar(weeks=10)
    f = detect_bottlenecks(make_rep(dates, care=np.where(dates >= dates[40], 1060.0, 1000.0), cbp=35.0, tr=10.0))
    row = f.iloc[40]
    assert (row["census_growth_level"], row["cbp_holding_level"], row["overall_level"], row["overall_status"]) == (1, 2, 2, "Alert")


def test_empty_input():
    f = detect_bottlenecks(make_rep(pd.DatetimeIndex([])))
    assert f.empty and {"overall_level", "overall_status", "census_growth_pct"} <= set(f.columns)
    assert alert_episodes(f).empty and list(alert_episodes(f).columns)[0] == "Indicator"


def test_no_look_ahead_on_synthetic_data():
    dates = reported_calendar(weeks=20)
    rng = np.random.default_rng(0)
    r = make_rep(dates, care=1000 + rng.normal(0, 30, len(dates)).cumsum(), dis=rng.poisson(10, len(dates)), tr=rng.poisson(10, len(dates)),
                 cbp=rng.poisson(25, len(dates)))
    full, cut = detect_bottlenecks(r), detect_bottlenecks(r.iloc[:70])
    pd.testing.assert_frame_equal(full.iloc[:70], cut)


# ----------------------------------------------------------------------------- episodes
def test_episodes_merge_across_the_weekend_and_split_on_long_gaps():
    dates = pd.to_datetime(["2025-06-04", "2025-06-05", "2025-06-08", "2025-06-09", "2025-06-30", "2025-07-01"])   # Wed Thu | Sun Mon | 3-week gap
    f = pd.DataFrame({"census_growth_level": [1, 2, 1, 0, 1, 1], "census_growth_pct": [6, 12, 7, 1, 8, 9],
                      "discharge_slowdown_level": np.nan, "discharge_drop_pct": np.nan, "cbp_holding_level": np.nan, "cbp_hold_days": np.nan}, index=dates)
    ep = alert_episodes(f, min_level=1)
    assert len(ep) == 2
    first = ep.iloc[0]
    assert (first["Start"], first["End"], first["Reported days"], first["Calendar days"], first["Highest level"], first["Peak value"]) == \
        (pd.Timestamp("2025-06-04"), pd.Timestamp("2025-06-08"), 3, 5, "Alert", 12)
    assert len(alert_episodes(f, min_level=2)) == 1


# ----------------------------------------------------------------------------- the real data
def overlaps(ep, label, start, end):
    e = ep[ep["Indicator"] == label]
    return ((e["Start"] <= pd.Timestamp(end)) & (e["End"] >= pd.Timestamp(start))).any()


def test_real_data_documented_events_are_detected(flags):
    alerts = alert_episodes(flags, min_level=2)
    assert overlaps(alerts, INDICATORS["census_growth"]["label"], "2023-07-15", "2023-09-30")          # the 2023 summer surge
    assert overlaps(alerts, INDICATORS["discharge_slowdown"]["label"], "2025-02-17", "2025-03-31")     # discharge break, 17 Feb 2025
    assert flags.loc[:"2024-01-30", "cbp_holding_level"].isna().all()                                  # no CBP verdicts on copied records


def test_real_data_no_look_ahead(rep, flags):
    cut = rep.loc[:"2025-06-30"]
    pd.testing.assert_frame_equal(flags.loc[:"2025-06-30"], detect_bottlenecks(cut))


def test_real_data_statuses_are_valid(flags):
    assert set(flags["overall_status"]) <= set(LEVELS.values()) | {"Insufficient data"}
    assert flags["overall_status"].eq("Insufficient data").sum() < 20      # only the first weeks of 2023 lack a complete window
