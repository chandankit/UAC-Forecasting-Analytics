"""Data ingestion and validation: the real dataset loads correctly, and malformed files fail with readable messages."""
import numpy as np
import pandas as pd
import pytest

from conftest import HEADER, write_csv
from src.data_loader import (CBP_COLS, SERIES, DataValidationError, load_raw_csv, reconcile, reported_days, validate_daily)


# ----------------------------------------------------------------------------- the real dataset
def test_raw_dataset_loads_with_expected_shape(raw):
    frame, report = raw
    assert frame.shape == (720, 5)
    assert list(frame.columns) == SERIES
    assert frame.index.min() == pd.Timestamp("2023-01-12") and frame.index.max() == pd.Timestamp("2025-12-21")
    assert frame.index.is_monotonic_increasing and frame.index.is_unique
    assert report["blank_rows"] == 450 and report["duplicates"] == [] and report["quarantined"] == [] and report["iso_dates"] == 0


def test_thousands_separators_are_parsed(raw):
    frame, _ = raw
    assert frame.at[pd.Timestamp("2025-12-21"), "hhs_care"] == 2484      # file text: "2,484"
    assert frame.at[pd.Timestamp("2023-01-12"), "hhs_care"] == 6566      # oldest record, last data line of the file


def test_no_negative_or_missing_values_in_raw(raw):
    frame, _ = raw
    assert frame.notna().all().all() and (frame >= 0).all().all()


def test_reporting_cadence_has_no_saturday_reports(raw):
    frame, _ = raw
    weekdays = frame.index.dayofweek.value_counts()
    assert weekdays.get(5, 0) == 0                    # Saturday never reported
    assert weekdays.get(4, 0) <= 5                    # Friday almost never (2 in the file)


def test_cleaned_table_reconciles_with_raw(raw, rep):
    frame, _ = raw
    assert len(rep) == 715
    assert reconcile(frame, rep).empty
    dropped = frame.index.difference(rep.index)       # the 5 stale reports: HHS values repeated from the previous report
    assert len(dropped) == 5
    prev = frame.shift(1)
    assert ((frame.loc[dropped, "hhs_care"] == prev.loc[dropped, "hhs_care"]) & (frame.loc[dropped, "discharged"] == prev.loc[dropped, "discharged"])).all()


def test_cbp_columns_masked_in_copied_block(rep):
    assert rep.loc[:"2024-01-30", CBP_COLS].isna().all().all()
    assert rep.loc["2024-01-31":, CBP_COLS].notna().all().all()
    assert rep[["hhs_care", "discharged"]].notna().all().all()


# ----------------------------------------------------------------------------- malformed inputs
def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_csv(tmp_path / "nope.csv")


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError, match="empty"):
        load_raw_csv(p)


def test_header_only_raises(tmp_path):
    p = write_csv(tmp_path / "h.csv", [",,,,,", ",,,,,"])
    with pytest.raises(DataValidationError, match="no data rows"):
        load_raw_csv(p)


def test_missing_column_raises(tmp_path):
    header = HEADER.rsplit(",", 1)[0]
    p = write_csv(tmp_path / "m.csv", ['"December 21, 2025",6,18,11,"2,484"'], header=header)
    with pytest.raises(DataValidationError, match="Expected 6 columns"):
        load_raw_csv(p)


def test_renamed_column_raises_and_names_it(tmp_path):
    p = write_csv(tmp_path / "r.csv", ['"December 21, 2025",6,18,11,"2,484",14'], header=HEADER.replace("Children in HHS Care", "Kids in care"))
    with pytest.raises(DataValidationError, match="Column 5 is 'Kids in care'"):
        load_raw_csv(p)


def test_unparseable_date_reports_file_line(tmp_path):
    p = write_csv(tmp_path / "d.csv", ['"December 21, 2025",6,18,11,"2,484",14', "yesterday,1,2,3,4,5"])
    with pytest.raises(DataValidationError, match=r"line 3: 'yesterday'"):
        load_raw_csv(p)


def test_non_numeric_value_reports_column_and_line(tmp_path):
    p = write_csv(tmp_path / "n.csv", ['"December 21, 2025",6,18,eleven,"2,484",14'])
    with pytest.raises(DataValidationError, match=r"Column 'transferred'.*line 2: 'eleven'"):
        load_raw_csv(p)


def test_iso_dates_and_missing_tokens_accepted(tmp_path):
    p = write_csv(tmp_path / "i.csv", ["2025-12-21,6,N/A,11,2484,14", '"December 18, 2025",11,50,6,"2,472",16'])
    frame, report = load_raw_csv(p)
    assert report["iso_dates"] == 1
    assert np.isnan(frame.at[pd.Timestamp("2025-12-21"), "cbp_custody"])
    assert list(frame.index) == [pd.Timestamp("2025-12-18"), pd.Timestamp("2025-12-21")]    # sorted ascending


def test_negative_counts_are_quarantined_not_zero_filled(tmp_path):
    p = write_csv(tmp_path / "neg.csv", ['"December 21, 2025",6,18,-3,"2,484",14'])
    frame, report = load_raw_csv(p)
    assert np.isnan(frame.iloc[0]["transferred"])
    assert report["quarantined"] == [{"file_line": 2, "date": pd.Timestamp("2025-12-21"), "column": "transferred", "value": -3.0}]


def test_duplicate_dates_keep_first_record_in_file_order(tmp_path):
    p = write_csv(tmp_path / "dup.csv", ['"December 21, 2025",6,18,11,"2,484",14', '"December 21, 2025",9,18,11,"2,400",14'])
    frame, report = load_raw_csv(p)
    assert len(frame) == 1 and frame.iloc[0]["hhs_care"] == 2484
    assert report["duplicates"][0]["conflicting"] is True and report["duplicates"][0]["file_lines"] == [2, 3]


@pytest.mark.parametrize("name, data", [
    ("garbage.csv", bytes(range(256)) * 8),                                           # binary noise
    ("workbook.csv", b"PK\x03\x04\x14\x00\x00\x00\x08\x00" + b"\xff\xfe\x80\x81" * 40),  # zip / .xlsx header + bytes never valid in UTF-8
    ("ansi.csv", (HEADER.replace("HHS Care", "HHS Care é") + "\n").encode("cp1252")),   # Windows-1252 export
])
def test_non_utf8_or_binary_file_is_a_readable_error(tmp_path, name, data):
    """Regression for QA3-ING-01: these used to escape as a bare UnicodeDecodeError instead of a DataValidationError."""
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises(DataValidationError, match="not a readable UTF-8 CSV file.*CSV UTF-8"):
        load_raw_csv(path)


def test_daily_table_validation(daily):
    with pytest.raises(DataValidationError, match="is_reported"):
        validate_daily(daily.drop(columns="is_reported"))
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_daily(pd.concat([daily.iloc[:3], daily.iloc[:1]]))
    with pytest.raises(DataValidationError, match="indexed by date"):
        validate_daily(daily.reset_index())


def test_text_in_a_count_column_is_a_validation_error_not_a_crash(daily):
    """Regression for QA E2E-03: 'abc' in a CBP column used to escape as a bare ValueError and crash the pipeline tab."""
    d = daily.copy()
    d["cbp_custody"] = d["cbp_custody"].astype(object)
    d.iloc[-1, d.columns.get_loc("cbp_custody")] = "abc"
    with pytest.raises(DataValidationError, match="non-numeric value.*cbp_custody.*'abc'"):
        reported_days(d)
    validate_daily(daily)                                       # the real table (with NaN in masked CBP days) still passes


def test_reported_days_defensively_masks_untrusted_cbp(daily):
    d = daily.copy()
    d.loc["2023-03-01":"2023-03-31", "cbp_custody"] = 999.0     # pretend the mask was lost
    assert reported_days(d).loc["2023-03-01":"2023-03-31", "cbp_custody"].isna().all()
