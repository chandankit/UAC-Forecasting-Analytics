"""Shared fixtures: the real dataset and cleaned table, and builders for small synthetic reported-day frames."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bottleneck_detection import detect_bottlenecks          # noqa: E402
from src.data_loader import load_daily_clean, load_raw_csv, reported_days   # noqa: E402

RAW_CSV = ROOT / "HHS_Unaccompanied_Alien_Children_Program.csv"
DAILY_CSV = ROOT / "artifacts" / "daily_clean.csv"
APP = ROOT / "uac_forecast_dashboard.py"
HEADER = ("Date,Children apprehended and placed in CBP custody*,Children in CBP custody,Children transferred out of CBP custody,"
          "Children in HHS Care,Children discharged from HHS Care")


@pytest.fixture(scope="session")
def raw():
    frame, report = load_raw_csv(RAW_CSV)
    return frame, report


@pytest.fixture(scope="session")
def daily():
    return load_daily_clean(DAILY_CSV)


@pytest.fixture(scope="session")
def rep(daily):
    return reported_days(daily)


@pytest.fixture(scope="session")
def flags(rep):
    return detect_bottlenecks(rep)


def reported_calendar(start="2025-01-05", weeks=20):
    """Sunday-Thursday report dates, like the real feed (2025-01-05 is a Sunday)."""
    days = pd.date_range(start, periods=weeks * 7, freq="D")
    return days[days.dayofweek.isin([6, 0, 1, 2, 3])]


def make_rep(dates, care=1000.0, dis=10.0, tr=10.0, cbp=10.0, app=10.0):
    """Reported-day frame with the five series (scalars broadcast, arrays used as given) and trusted CBP data."""
    n = len(dates)
    col = lambda v: np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()
    return pd.DataFrame({"apprehended": col(app), "cbp_custody": col(cbp), "transferred": col(tr),
                         "hhs_care": col(care), "discharged": col(dis), "cbp_trusted": True}, index=pd.DatetimeIndex(dates, name="date"))


def write_csv(path, rows, header=HEADER):
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path
