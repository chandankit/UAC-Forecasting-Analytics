"""Dashboard integration: the real app script (Streamlit AppTest) wired to the analytical functions, its controls,
edge cases, file handling, and a real server start-up."""
import base64
import datetime as dt
import gc
import io
import json
import shutil
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest

from conftest import APP, ROOT
from src.bottleneck_detection import detect_bottlenecks
from src.kpi_calculations import filter_period, period_kpis

LAST = dt.date(2025, 12, 21)


def app():
    return AppTest.from_file(str(APP), default_timeout=180).run()


def metrics(at):
    return {m.label: m.value for m in at.metric}


def no_errors(at):
    return not at.exception and not at.error


def xs(trace):
    """Plotly 6 may serialise arrays as typed base64 blocks; dates arrive as ISO strings."""
    x = trace.get("x", [])
    if isinstance(x, dict) and "bdata" in x:
        x = np.frombuffer(base64.b64decode(x["bdata"]), dtype=np.dtype(x["dtype"]))
    return pd.to_datetime(pd.Series(list(x)))


def pipeline_specs(at):
    """Specs of the four pipeline charts (they are the last four Plotly charts in the script)."""
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")[-4:]]


def expected(rep, start, end):
    return period_kpis(filter_period(rep, start, end))


def fmt(v, spec):
    return f"{v:{spec}}"


# ----------------------------------------------------------------------------- start-up and wiring
def test_app_starts_with_pipeline_tab_and_no_errors():
    at = app()
    assert no_errors(at)
    assert [t.label for t in at.tabs][-1] == "Pipeline analytics & data explorer" and len(at.tabs) == 7


def test_default_period_kpis_come_from_the_analytical_functions(rep):
    """Default = current regime (17 Mar 2025 onward); every tile equals the module's value for that period (no hard-coded numbers)."""
    at, k = app(), expected(rep, "2025-03-17", LAST)
    m = metrics(at)
    assert m["HHS census at period end"] == fmt(k["hhs_care_end"], ",.0f") == "2,484"
    assert m["Peak HHS census"] == fmt(k["hhs_care_peak"], ",.0f")
    assert m["Reported days in period"] == f"{k['reported_days']:,}"
    assert m["Transfers into HHS per reported day"] == fmt(k["transferred_per_day"], ",.1f")
    assert m["Discharges per reported day"] == fmt(k["discharged_per_day"], ",.1f")
    assert m["HHS discharge rate (%/day)"] == fmt(k["discharge_rate_pct"], ".2f") + "%"
    assert m["Implied CBP holding time (days)"] == fmt(k["implied_cbp_hold_days"], ".2f") + " days"
    assert m["Implied average stay in HHS care (days)"] == fmt(k["implied_stay_days"], ",.0f") + " days"
    assert m["Recorded throughput (exits per entry)"] == fmt(k["throughput_ratio"], ".2f")


def test_page_layout_is_identical_on_cold_and_warm_cache_runs():
    """Regression for QA E2E-01: a cache spinner added a page element only on a cache miss, so the tab bar moved one slot
    between the first and second run after a start; the browser then remounted the tabs and the user's tab reset to Overview."""
    def layout(at):
        return ([getattr(c, "type", type(c).__name__) for c in at.main.children.values()],
                [getattr(c, "type", type(c).__name__) for c in at.tabs[6].children.values()])
    st.cache_data.clear()
    at = AppTest.from_file(str(APP), default_timeout=180)
    cold = layout(at.run())
    warm = layout(at.run())
    at.slider(key="pa_hold").set_value((2.25, 3.0)).run()                   # new thresholds: the bottleneck cache misses
    assert cold == warm == layout(at) and no_errors(at)


def test_forecast_tabs_still_show_artifact_values():
    fc = pd.read_csv(ROOT / "artifacts" / "forecast_future.csv")
    at = app()
    m = metrics(at)
    sarima28 = fc[(fc.target == "hhs_care") & (fc.model == "SARIMA") & (fc.h == 28)].yhat.iloc[0]
    assert m["Children in HHS care (latest report)"] == "2,484"
    assert m["Forecast in 28 days (SARIMA)"] == f"{sarima28:,.0f}"


# ----------------------------------------------------------------------------- interactive controls
@pytest.mark.parametrize("preset, start", [("Last 90 days", LAST - dt.timedelta(days=89)), ("Full history", dt.date(2023, 1, 12)),
                                           ("Last 12 months", LAST - dt.timedelta(days=364))])
def test_period_presets_update_kpis_and_charts(rep, preset, start):
    at = app()
    at.selectbox(key="pa_period").set_value(preset).run()
    assert no_errors(at)
    k, m = expected(rep, start, LAST), metrics(at)
    assert m["Reported days in period"] == f"{k['reported_days']:,}"
    assert m["Peak HHS census"] == fmt(k["hhs_care_peak"], ",.0f")
    for spec in pipeline_specs(at):                           # every chart is cut to the period
        for trace in spec["data"]:
            x = xs(trace)
            if len(x):
                assert x.min() >= pd.Timestamp(start) and x.max() <= pd.Timestamp(LAST), (preset, trace.get("name"))


def test_custom_date_range_updates_kpis(rep):
    at = app()
    at.selectbox(key="pa_period").set_value("Custom range").run()
    at.date_input(key="pa_range").set_value((dt.date(2024, 6, 1), dt.date(2024, 8, 31))).run()
    assert no_errors(at)
    k, m = expected(rep, "2024-06-01", "2024-08-31"), metrics(at)
    assert m["Reported days in period"] == f"{k['reported_days']:,}"
    assert m["HHS census at period end"] == fmt(k["hhs_care_end"], ",.0f")
    assert m["CBP custody at period end"] == fmt(k["cbp_custody_end"], ",.0f")
    assert m["Implied CBP holding time (days)"] == fmt(k["implied_cbp_hold_days"], ".2f") + " days"


def test_summary_granularity_changes_the_table():
    at = app()
    sizes = {}
    for g in ("Daily", "Weekly", "Monthly"):
        at.radio(key="pa_gran").set_value(g).run()
        table = next(d.value for d in at.dataframe if "Discharge rate %/day" in d.value.columns)
        sizes[g] = len(table)
    assert sizes["Daily"] == 185 and sizes["Daily"] > sizes["Weekly"] > sizes["Monthly"] >= 9


def test_threshold_change_updates_the_status(rep):
    at = app()
    at.selectbox(key="pa_period").set_value("Last 90 days").run()
    status = lambda: next(x.value for x in at.markdown if "Bottleneck status" in x.value)
    flags = detect_bottlenecks(rep)
    assert flags["overall_status"].iloc[-1] in status()                  # banner = module verdict for the last report
    at.slider(key="pa_hold").set_value((5.0, 8.0)).run()
    at.slider(key="pa_growth").set_value((20.0, 40.0)).run()
    assert "Normal" in status() and no_errors(at)


# ----------------------------------------------------------------------------- edge cases
def test_empty_range_shows_a_message_not_an_error():
    at = app()
    at.selectbox(key="pa_period").set_value("Custom range").run()
    at.date_input(key="pa_range").set_value((dt.date(2025, 12, 19), dt.date(2025, 12, 20))).run()     # Friday-Saturday: never reported
    assert no_errors(at)
    assert any("No reported days between 19 Dec 2025 and 20 Dec 2025" in i.value for i in at.info)
    assert "HHS census at period end" not in metrics(at)


def test_half_selected_range_waits_for_the_end_date():
    at = app()
    at.selectbox(key="pa_period").set_value("Custom range").run()
    at.date_input(key="pa_range").set_value((dt.date(2025, 12, 1),)).run()
    assert no_errors(at) and any("Select both a start and an end date" in i.value for i in at.info)
    # QA3-UX-01: the browser drops a typed impossible or out-of-range date the same way, so the message names the allowed range
    assert any("between 12 Jan 2023 and 21 Dec 2025" in i.value and "DD/MM/YYYY" in i.value for i in at.info)


def test_masked_cbp_period_shows_na_not_copied_numbers():
    at = app()
    at.selectbox(key="pa_period").set_value("Custom range").run()
    at.date_input(key="pa_range").set_value((dt.date(2023, 3, 1), dt.date(2023, 6, 30))).run()
    m = metrics(at)
    assert no_errors(at)
    assert m["CBP custody at period end"] == m["Implied CBP holding time (days)"] == m["Transfers into HHS per reported day"] == "n/a"
    assert m["HHS census at period end"] != "n/a"


def test_download_buttons_are_offered():
    labels = [d.proto.label for d in app().get("download_button")]
    assert labels == ["Download these forecasts with 80%/95% intervals (CSV)", "KPI summary (CSV)", "Weekly summary (CSV)",
                      "Explorer data (CSV)", "Alert episodes (CSV)"]


def downloads(at):
    """{label: (file name, bytes)} for every download button. AppTest keeps the files in an in-memory media store
    (a private attribute of the pinned Streamlit 1.52.1; the file id is the last part of the button's URL)."""
    files = {fid: f for store in gc.get_objects() if isinstance(store, MemoryMediaFileStorage) for fid, f in store._files_by_id.items()}
    return {d.proto.label: (files[d.proto.url.rsplit("/", 1)[-1].split(".")[0]].filename,
                            files[d.proto.url.rsplit("/", 1)[-1].split(".")[0]].content) for d in at.get("download_button")}


@pytest.mark.parametrize("gran", ["Daily", "Weekly", "Monthly"])
def test_pipeline_download_file_names_are_distinct(gran):
    """Regression for QA E2E-02: with 'Daily' selected, the summary and the explorer were both uac_pipeline_daily_<dates>.csv."""
    at = app()
    at.radio(key="pa_gran").set_value(gran).run()
    names = [name for label, (name, _) in downloads(at).items() if not label.startswith("Download these forecasts")]
    assert len(names) == 4 and len(set(names)) == 4, names
    assert f"uac_pipeline_summary_{gran.lower()}_20250317_20251221.csv" in names and "uac_pipeline_explorer_20250317_20251221.csv" in names


def test_kpi_download_labels_every_row_and_matches_the_cards(rep):
    """Regression for QA E2E-06: the first row was the raw key 'reported_days' with no definition."""
    at = app()
    kpis = pd.read_csv(io.BytesIO(downloads(at)["KPI summary (CSV)"][1]))
    k = expected(rep, "2025-03-17", LAST)
    assert kpis.iloc[0]["KPI"] == "Reported days in period" and int(kpis.iloc[0]["Value"]) == k["reported_days"] == 185
    assert kpis["Definition"].notna().all() and not kpis["KPI"].str.contains("_").any()
    assert np.isclose(kpis.set_index("KPI").loc["Implied CBP holding time (days)", "Value"], k["implied_cbp_hold_days"])


def test_negative_discharge_drop_is_described_as_above_the_baseline():
    """Regression for QA E2E-04: a 7-day rate 19% ABOVE its baseline was shown as '-19.0 % below 8-week median rate'."""
    caption = next(c.value for c in app().caption if "Discharge slowdown (placement bottleneck)" in c.value)
    assert "19.0 % above 8-week median rate" in caption and "-19.0" not in caption


def test_period_without_evaluable_indicators_is_not_reported_as_all_clear():
    """Regression for QA E2E-05: the first weeks of data (no trailing history) showed a green 'no indicator reached Watch'."""
    at = app()
    at.selectbox(key="pa_period").set_value("Custom range").run()
    at.date_input(key="pa_range").set_value((dt.date(2023, 1, 12), dt.date(2023, 1, 31))).run()
    assert no_errors(at) and not at.success
    assert any("No indicator could be evaluated in this period" in i.value for i in at.info)


# ----------------------------------------------------------------------------- file handling (fresh processes, so ./src is not cached)
RUNNER = textwrap.dedent("""
    import json, sys
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(sys.argv[1], default_timeout=180).run()
    print("QA_JSON " + json.dumps({"exceptions": [e.value[:200] for e in at.exception], "errors": [e.value[:300] for e in at.error],
                                   "tabs": len(at.tabs), "metrics": {m.label: m.value for m in at.metric}}))
""")


def run_isolated(app_path, cwd):
    r = subprocess.run([sys.executable, "-c", RUNNER, str(app_path)], cwd=cwd, capture_output=True, text=True, timeout=300)
    line = next((l for l in r.stdout.splitlines() if l.startswith("QA_JSON ")), None)
    assert line, r.stderr[-800:]
    return json.loads(line[8:])


@pytest.mark.slow
def test_app_runs_from_another_working_directory(tmp_path):
    out = run_isolated(APP, cwd=tmp_path)
    assert not out["exceptions"] and not out["errors"] and out["tabs"] == 7
    assert out["metrics"]["HHS census at period end"] == "2,484"


@pytest.mark.slow
def test_missing_src_package_disables_only_the_pipeline_tab(tmp_path):
    shutil.copy(APP, tmp_path / APP.name)
    shutil.copytree(ROOT / "artifacts", tmp_path / "artifacts")
    out = run_isolated(tmp_path / APP.name, cwd=tmp_path)
    assert not out["exceptions"]
    assert any("pipeline-analytics module could not be loaded" in e for e in out["errors"])
    assert out["metrics"]["Children in HHS care (latest report)"] == "2,484"       # forecasts unaffected


def _last_report(col, value):
    def damage(d):
        d[col] = d[col].astype(object)
        d.loc[d.index[d["is_reported"]][-1], col] = value
        return d
    return damage


DAMAGED_DAILY = {   # regression for QA E2E-03: each of these used to end in Streamlit's generic "This app has encountered an error"
    "missing is_reported column": (lambda d: d.drop(columns="is_reported"), "lacks column(s): is_reported"),
    "missing hhs_care column": (lambda d: d.drop(columns="hhs_care"), "lacks column(s): hhs_care"),
    "header only": (lambda d: d.iloc[0:0], "0 row(s)"),
    "duplicate date": (lambda d: pd.concat([d, d.iloc[[-1]]]), "1 duplicate date(s)"),
    "blank census on the last report": (_last_report("hhs_care", np.nan), "missing values in: hhs_care"),
    "text in a forecast column": (_last_report("discharged", "abc"), "wrong type in: discharged"),
    "unparseable date": (lambda d: d.assign(date=d["date"].where(d.index != d.index[-30], "not-a-date")), "dates that cannot be parsed"),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", list(DAMAGED_DAILY))
def test_damaged_daily_table_gives_a_readable_message(tmp_path, case):
    damage, message = DAMAGED_DAILY[case]
    shutil.copy(APP, tmp_path / APP.name)
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copytree(ROOT / "artifacts", tmp_path / "artifacts")
    path = tmp_path / "artifacts" / "daily_clean.csv"
    damage(pd.read_csv(path)).to_csv(path, index=False)
    out = run_isolated(tmp_path / APP.name, cwd=tmp_path)
    assert not out["exceptions"], out["exceptions"]
    assert any(message in e and "daily_clean.csv" in e for e in out["errors"]), out["errors"]


@pytest.mark.slow
def test_text_in_a_cbp_column_disables_only_the_pipeline_tab(tmp_path):
    """CBP columns are not read by the forecast tabs, so the forecasts stay up and the pipeline tab explains the problem."""
    shutil.copy(APP, tmp_path / APP.name)
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copytree(ROOT / "artifacts", tmp_path / "artifacts")
    path = tmp_path / "artifacts" / "daily_clean.csv"
    _last_report("cbp_custody", "abc")(pd.read_csv(path)).to_csv(path, index=False)
    out = run_isolated(tmp_path / APP.name, cwd=tmp_path)
    assert not out["exceptions"]
    assert any("cleaned daily table cannot be used" in e and "cbp_custody" in e for e in out["errors"]), out["errors"]
    assert out["metrics"]["Children in HHS care (latest report)"] == "2,484"


# ----------------------------------------------------------------------------- application start-up (real server)
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
def test_streamlit_server_starts_and_serves_the_app():
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true", "--server.port", str(port),
                             "--browser.gatherUsageStats", "false"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline, health = time.time() + 90, None
        while time.time() < deadline and health != "ok":
            assert proc.poll() is None, proc.stdout.read()[-1500:]
            try:
                health = urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=3).read().decode()
            except OSError:
                time.sleep(1)
        assert health == "ok"
        page = urllib.request.urlopen(f"http://localhost:{port}/", timeout=10)
        assert page.status == 200 and b"<html" in page.read().lower()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
