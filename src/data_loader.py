"""Load and validate the UAC dataset for descriptive pipeline analytics.

Two inputs, one contract:
* the raw HHS export (`HHS_Unaccompanied_Alien_Children_Program.csv`): validated here and used to check the cleaned table;
* the notebook's cleaned daily table (`artifacts/daily_clean.csv`): the analytical base, because it already carries the
  data-quality masks (5 stale HHS reports removed; CBP columns masked in the copied block 12 Jan 2023 - 30 Jan 2024).

Descriptive KPIs use reported days only. Friday/Saturday rows in the daily table are interpolated for the forecasting
models and never enter a descriptive KPI.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Raw header (position, normalised text) -> short name used everywhere (same names as the notebook)
RAW_HEADERS = [("date", "date"),
               ("children apprehended and placed in cbp custody", "apprehended"),
               ("children in cbp custody", "cbp_custody"),
               ("children transferred out of cbp custody", "transferred"),
               ("children in hhs care", "hhs_care"),
               ("children discharged from hhs care", "discharged")]
SERIES = [short for _, short in RAW_HEADERS[1:]]
STOCKS = ["cbp_custody", "hhs_care"]                   # point-in-time counts: never summed over time
FLOWS = ["apprehended", "transferred", "discharged"]   # daily counts: summed over reported days
CBP_COLS = ["apprehended", "cbp_custody", "transferred"]
HHS_COLS = ["hhs_care", "discharged"]
DAILY_REQUIRED = SERIES + ["is_reported", "cbp_trusted"]
MISSING_TOKENS = {"", "n/a", "na", "nan", "null", "-"}


class DataValidationError(ValueError):
    """Raised when an input cannot be used; the message names the column, file line and value."""


def _norm(text):
    return " ".join(str(text).replace("*", " ").lower().split())


def _examples(lines, values, k=5):
    return ", ".join(f"line {ln}: {v!r}" for ln, v in list(zip(lines, values))[:k])


def load_raw_csv(path):
    """Read and validate the raw HHS export.

    Returns (frame, report). `frame` is indexed by report date (ascending) with the five series as floats.
    `report` records what was repaired: blank rows dropped, ISO dates accepted, duplicate dates (first record in file
    order kept, as in the notebook), and negative counts quarantined to missing (never zero-filled).
    Raises FileNotFoundError or DataValidationError with a readable message.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, skip_blank_lines=False)
    except pd.errors.EmptyDataError as e:
        raise DataValidationError(f"{path.name} is empty") from e
    except (UnicodeDecodeError, pd.errors.ParserError) as e:   # a workbook, archive or non-UTF-8 export renamed .csv (QA3-ING-01)
        raise DataValidationError(f"{path.name} is not a readable UTF-8 CSV file ({type(e).__name__}). It may be an Excel workbook, "
                                  "a zip archive or a file saved in another encoding: export it again as 'CSV UTF-8'") from e
    if frame.shape[1] != len(RAW_HEADERS):
        raise DataValidationError(f"Expected {len(RAW_HEADERS)} columns, found {frame.shape[1]}: {list(frame.columns)}")
    for i, (expected, _) in enumerate(RAW_HEADERS):
        if _norm(frame.columns[i]) != expected:
            raise DataValidationError(f"Column {i + 1} is '{frame.columns[i]}'; expected '{expected}'")
    frame.columns = [short for _, short in RAW_HEADERS]
    frame["file_line"] = np.arange(len(frame)) + 2                       # header is line 1

    text = frame[[short for _, short in RAW_HEADERS]].apply(lambda s: s.str.strip())
    blank = (text == "").all(axis=1)
    frame, text = frame.loc[~blank], text.loc[~blank]
    report = {"blank_rows": int(blank.sum()), "iso_dates": 0, "duplicates": [], "quarantined": []}
    if frame.empty:
        raise DataValidationError("The file has a header but no data rows")

    dates = pd.to_datetime(text["date"], format="%B %d, %Y", errors="coerce")
    iso = dates.isna() & text["date"].ne("")
    dates[iso] = pd.to_datetime(text.loc[iso, "date"], format="%Y-%m-%d", errors="coerce")
    report["iso_dates"] = int((iso & dates.notna()).sum())
    if dates.isna().any():
        bad = dates.isna()
        raise DataValidationError(f"{int(bad.sum())} unparseable date(s): " + _examples(frame.loc[bad, "file_line"], text.loc[bad, "date"]))

    out = pd.DataFrame({"date": dates, "file_line": frame["file_line"]})
    for col in SERIES:
        raw = text[col].str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
        missing = raw.str.lower().isin(MISSING_TOKENS)
        num = pd.to_numeric(raw.where(~missing), errors="coerce")
        bad = num.isna() & ~missing
        if bad.any():
            raise DataValidationError(f"Column '{col}' has {int(bad.sum())} non-numeric value(s): "
                                      + _examples(frame.loc[bad, "file_line"], text.loc[bad, col]))
        neg = num < 0
        for ln, d, v in zip(frame.loc[neg, "file_line"], dates[neg], num[neg]):
            report["quarantined"].append({"file_line": int(ln), "date": d, "column": col, "value": float(v)})
        out[col] = num.mask(neg).astype(float)

    dup = out["date"].duplicated(keep="first")              # file lists the newest publication first: keep that one
    for d in out.loc[dup, "date"].unique():
        rows = out[out["date"] == d]
        conflict = bool(rows[SERIES].nunique(dropna=False).gt(1).any())
        report["duplicates"].append({"date": d, "file_lines": rows["file_line"].tolist(), "conflicting": conflict})
    out = out.loc[~dup].sort_values("date", kind="stable").set_index("date").drop(columns="file_line")
    return out, report


def validate_daily(daily):
    """Check the cleaned daily table (DatetimeIndex + required columns); raise DataValidationError if unusable."""
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise DataValidationError("The daily table must be indexed by date")
    missing = [c for c in DAILY_REQUIRED if c not in daily.columns]
    if missing:
        raise DataValidationError(f"The daily table lacks column(s): {', '.join(missing)}")
    if daily.index.has_duplicates:
        raise DataValidationError(f"The daily table has {int(daily.index.duplicated().sum())} duplicate date(s)")
    for col in SERIES:                                   # text in a count column would otherwise fail later with a bare ValueError
        bad = pd.to_numeric(daily[col], errors="coerce").isna() & daily[col].notna()
        if bad.any():
            raise DataValidationError(f"The daily table has non-numeric value(s) in '{col}', e.g. {daily.loc[bad, col].iloc[0]!r}")
    return daily


def load_daily_clean(path):
    """Read `artifacts/daily_clean.csv` (written by the notebook) and validate it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned daily table not found: {path}. Run the notebook to regenerate artifacts/.")
    daily = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return validate_daily(daily)


def reported_days(daily):
    """Rows with a genuine (non-stale) HHS report, the five series and the CBP trust flag; CBP values masked where untrusted."""
    validate_daily(daily)
    rep = daily.loc[daily["is_reported"].astype(bool), SERIES + ["cbp_trusted"]].copy()
    rep["cbp_trusted"] = rep["cbp_trusted"].astype(bool)
    rep.loc[~rep["cbp_trusted"], CBP_COLS] = np.nan       # defensive: the notebook already masks the copied block
    rep[SERIES] = rep[SERIES].astype(float)
    return rep.sort_index()


def reconcile(raw, rep):
    """Compare the cleaned reported days with the raw export.

    Returns a frame of cell mismatches (date, column, raw, clean). HHS columns must equal the raw values on every
    reported day; CBP columns must equal them where trusted and be missing where masked.
    """
    rows = []
    for d in rep.index:
        if d not in raw.index:
            rows.append({"date": d, "column": "(row)", "raw": np.nan, "clean": np.nan})
            continue
        for col in SERIES:
            r, c = raw.at[d, col], rep.at[d, col]
            expected_missing = col in CBP_COLS and not rep.at[d, "cbp_trusted"]
            same = np.isnan(c) if expected_missing else (np.isnan(r) and np.isnan(c)) or (not np.isnan(r) and not np.isnan(c) and np.isclose(r, c))
            if not same:
                rows.append({"date": d, "column": col, "raw": r, "clean": c})
    return pd.DataFrame(rows, columns=["date", "column", "raw", "clean"])
