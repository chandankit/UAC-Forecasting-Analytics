# Change log

The public repository holds the deliverables only. Folders mentioned below such as `QA_Testing/`, `reports/` test reports and `archive/` builds are kept in the local working copy.

## Version 2.5 — 6 November 2026 (published on GitHub and Streamlit Community Cloud)

**Why Streamlit said "The app's code is not connected to a remote GitHub repository".** The local Deploy button looks for a git repository whose current branch tracks a GitHub remote. The project folder had never been a git repository, so there was nothing to connect. Git 2.41 and GitPython were installed, so neither was the cause. Running the app locally never publishes it; the hosted copy is a separate environment that runs from the GitHub repository.

**What was done.**

- The project folder became a git repository whose `.gitignore` is an allowlist: only named deliverables are published (96 files, about 22 MB). Personal and course documents (`intern.pdf`, `unified_details.png`, `Project_Details.txt`), the QA workbench, archived builds, build logs and local settings stay local. A scan of every published file, including inside the notebook, Word and PDF files, found no credentials.
- The entry point stays at the repository root (`uac_forecast_dashboard.py`). Streamlit Community Cloud supports a root entry point, and moving it would have broken the tested paths for no benefit.
- Deployment fixes: `.streamlit/config.toml` no longer pins `server.address = "localhost"`, because a pinned address can stop a host from serving the app; hiding tracebacks from visitors is kept. `requirements.txt` lost a byte-order mark that can break `uv`, which Community Cloud uses to install requirements.
- Privacy: Word recorded the Windows user name as "last modified by" when it saved the Appendix. `tools/build_paper.py` now restores the document's own author after Word saves, then exports the PDF in a second, read-only pass.
- `tools/build_paper.py` now also builds the executive summary and its PDF.
- The paper's freshness check (C19) now compares content, not file times, because a git clone writes every file at checkout time. It checks every text unit in the Word file, every embedded figure byte for byte, and every heading and caption in the PDF.
- New `tests/test_repository.py`: README links point to published files, the run command matches the entry point, the Streamlit config does not pin the server, `requirements.txt` parses cleanly and covers every app import, the app uses relative paths only, and nothing personal or local is published. New `requirements-dev.txt` lists the test packages.
- The README was rewritten for GitHub visitors, with the caveats corrected in v2.4.
- Spec change: QA case TC-UI-034 now requires that the server address and port are not pinned. D-SEC-01's localhost rule is superseded by the decision to publish the dashboard.
- Commits use the author name `ankitchand` with the account's GitHub noreply address. The account keeps its email private, so a personal address could have been rejected on push or exposed.

## Version 2.4 — 5 November 2026 (research paper shortened to 15 pages at most, with a quality gate)

The paper is now 14 pages, down from 30. Everything else moved to a separate Appendix volume of 107 pages. The full 30-page paper is kept in `archive/v2.3_full_paper_2026-11-05/`.

**Why the paper had grown (measured).** Versions 1, 2 and 2.1 each added text and nothing removed any. Implementation detail sat in the main body (file names, version history, test descriptions). Findings were repeated: holding time was mentioned 15 times. No page budget or automated check existed, so nothing stopped the growth. Charts were drawn 12–15 inches wide for the notebook, so on the page their labels shrank to about 4–5 pt.

| Measure (same counting method) | Before | After |
|---|---|---|
| Pages | 30 | 14 |
| Words in the source | 11,461 | 7,443 |
| Tables / figures in the main body | 24 / 19 | 21 / 9 |
| Mean sentence length (words) | 17.9 | 14.3 |
| Sentences over 30 words | 14.0% | 3.1% |
| File names and code terms in the body | 31 | 0 |

**Decision.** The brief asks for 15 pages in total and also for all raw code in the Appendix. About 3,400 lines of code cannot fit in 15 pages, so the Appendix is a separate volume (`Research_Paper_Appendix.md/.docx/.pdf`). The main paper ends with a short pointer to it.

**Where the old sections went.**

| Old section | Now |
|---|---|
| 1–2 Introduction, background | Section 1; method notes moved into each section's Methodology subsection |
| 3–4 Data, data quality | Section 2; full evidence in Appendix B |
| 5 Exploratory analysis | Section 3; extra detail and charts in Appendix C |
| 6 Methodology | Sections 4 and 5; specifications, order-selection logs and experiment designs in Appendices D and E |
| 7 Results | Section 6; all result tables and notebook charts in Appendix F |
| 8 Software validation, Appendix A | Appendix G |
| 9 Outlook and scenarios | Section 7 |
| 10 Pipeline analytics | Section 8 |
| 11–13 Limitations, recommendations, conclusion | Sections 9–11 |
| Appendix B KPI definitions | Appendix A |
| (raw code, previously outside the paper) | Appendix H |

**Durable fix: the paper is now built and checked by tools.**

- `tools/check_paper.py` runs 19 checks on the delivered Word and PDF files: page limit, page numbers, clipping, blank areas, heading order, required sections, Methodology subsections, table and figure captions, print-size figures, cross-references, no code in the body, sentence length, repeated sentences, citations both ways, number provenance, retained findings and caveats, Appendix completeness, and file freshness. It writes `reports/Paper_Quality_Report.md`.
- `tools/paper_figures.py` redraws the nine main charts at print size. Each chart first checks that its data reproduce the numbers printed in the paper, and the build stops if they do not. This caught an error in the redraw itself (194 day pairs against 192, because the stale-report mask had been left out).
- `tools/build_paper_appendix.py` generates the Appendix volume from the archived text, the artifacts and the executed notebook. Code is read from the source files at build time, so it cannot go stale.
- `tools/build_paper.py` runs the whole build through Word and then the checks.
- `tools/md2docx.py` was rewritten: page-number footers, table captions kept with tables, column widths from content, right-aligned numbers, contents page, code directive.
- `tests/test_paper_quality.py` runs the gate on the delivered files and includes negative controls: each check must fail on a document with that defect. It also compares Tables 9, 12, 14 and 15 with the artifacts.

**An independent review of the shortened text found problems the checks could not see.** Each was verified against the sources before it was fixed.

- Shortening had dropped caveats or overstated claims. Restored: the replay used a simpler ETS model, not the champion; 66–71% coverage applies to the champion, and SARIMA reached the target under two of three rules; coverage falls with lead time; the weekend ratchet is inferred from stock changes; surge events at nearby thresholds overlap; holding time is not a measured custody time and other exit routes would overstate it; feature importance is not causal; test results played no part in specific choices rather than "any choice".
- Other wording fixed: "weekends" became "Fridays and Saturdays", because Sundays are reported; single-horizon sample size is 26–55, not 42–55; a ranking of recommendations that the original never made was removed.
- Key caveats are now part of check C17, so they cannot be dropped silently.

**One correction to a printed result.** Gradient Boosting's test RMSE is 41.9 (`artifacts/metrics_overall.csv`: 41.945). The original paper printed 42.0. No model was re-run and no other result changed.

**Test changes (specification changes, not weakened assertions).**

- `tests/test_output_consistency.py`: the same assertions now read Section 8 of the paper and Appendix A of the Appendix volume. `test_appendix_b_lists_every_kpi_and_the_default_thresholds` was renamed `test_kpi_appendix_lists_every_kpi_and_the_default_thresholds`. The chart count now includes `figures/paper/`.
- QA case TC-DOC-004: one marker follows the reworded heading "Statistical vs machine learning, and recursive vs direct".

**Executive summary brought into line with the corrected paper.** The same caveats were applied: the replay's simpler model and its false alarms, overlapping surge events, the weekend pattern as an inference, holding time as an indirect estimate, range reliability by model and lead time, implied stays, and which results the test window did not influence. Three further fixes: "policy-driven placement break" became "structural break in placements", because policy is not in the data and no indicator identifies a cause; the weekly flows are now labelled as ensemble forecasts, and the difference between 2,542 and 2,571 is explained; and placements are stated as the swing factor (12.7 forecast against 9.4 recent discharges a day). The stale "version 2.1" label was removed. No number changed. The previous version is in `archive/v2.3_full_paper_2026-11-05/`, and the Word file was rebuilt with the new `tools/md2docx.py`. `tests/test_paper_quality.py` now holds these caveats in place.

**Unsupported policy attributions removed.** "Policy shock" and "policy-break crash" in the paper and the Appendix narrative now read "structural break", because policy is not in the data and no indicator identifies a cause. Check C12 now rejects these phrases. The notebook's own comments still use the older wording and are reproduced unchanged in Appendix H, since that section is the raw source.

**Verification.** Quality gate: 19 of 19 checks pass on the delivered files. Fast suite: 210 of 210 pass (`reports/pytest_v2.4_console.txt`). QA suite: 217 of 222 pass, and the 5 failures are the same known ones as in v2.3 (TC-PREP-015, TC-MET-010, TC-MOD-008, TC-API-012, TC-PERF-001; `QA_Testing/results/pytest_v2.4_qa_console.txt`). All 14 pages were reviewed by eye, plus a sample of 8 Appendix pages. The older QA register in `QA_Testing/tests/` was not re-run as a whole: TC-DOC-004's three markers were checked directly in the Markdown and Word files.

Cross-references updated: dashboard tooltip, `src/bottleneck_detection.py`, `src/kpi_calculations.py`, `tools/pipeline_analysis.py`, README, execution guide. The tooltip text is the only change to the dashboard's behaviour. `pypdfium2` was added to the test requirements.

## Version 2.3 — 30 October 2026 (aggressive QA round and PDF test report)

A third test round tried to break the app: hostile and malformed input in every field, workflow abuse (rapid changes, double clicks, refresh mid-run, back/forward, two sessions, server stop and restart), HTTP and security probes, and 19 file edge cases on the raw-CSV loader. 85 cases: 72 pass (8 of them after the fixes below), 2 fail (a Streamlit input quirk and the known D-SEC-02), 2 blocked by machine load, 2 not executed, 7 not applicable (the app has no LLM, agent, RAG, MCP, database or external API). Report: `reports/QA_Test_Execution_Report_v2.3.pdf`; register: `QA_Testing/e2e/qa3_test_cases.csv`.

| ID | Severity | Problem | Root cause | Fix | Verified by |
|---|---|---|---|---|---|
| QA3-ING-01 | Low | A binary file, a zip or Excel workbook renamed `.csv`, or a Windows-1252 export made `load_raw_csv` raise a bare `UnicodeDecodeError` | Only `EmptyDataError` was translated into `DataValidationError` | `UnicodeDecodeError` and `ParserError` now raise a readable `DataValidationError` ("export it again as CSV UTF-8") | `ingestion_edge_cases.py` 19/19; `test_non_utf8_or_binary_file_is_a_readable_error` (×3) |
| QA3-UX-01 | Low | Typing an impossible, reversed or out-of-range date showed only "Select both a start and an end date" | The browser drops such dates, so the app cannot tell "missing" from "invalid" | The message now names the accepted range and format | Browser retest; stronger assertion in `test_half_selected_range_waits_for_the_end_date` |

Files changed: `src/data_loader.py`, `uac_forecast_dashboard.py` (message text), `tests/test_data_validation.py`, `tests/test_dashboard_integration.py`. New: `QA_Testing/e2e/http_security_checks.py`, `ingestion_edge_cases.py`, `build_qa3_register.py`, `tools/build_qa_report_pdf.py`.

**Re-run of the blocked and not-run cases (30 Oct, later).** The machine was still saturated by an unrelated job, so performance was measured as CPU seconds the dashboard itself uses per run (`QA_Testing/e2e/cpu_cost_profile.py`), which does not depend on other load: cold start 6.7 CPU-s, each interaction 1.4–1.7 CPU-s; in the browser 1.3–2.4 s per change and a 4.2 s first render after a fresh start. Four simultaneous sessions stayed correct and isolated; the artifact-refresh path kept the user's tab in a real browser (E2E-01 retest); an idle session resumed correctly. New observation QA3-OBS-02 (Low): a new browser session costs about 4.4 server CPU-seconds against 1.5 for a rerun (one script run confirmed in a debug log; serving the front-end files costs only 0.22 s), and all sessions share one Python process, so several people opening the app at the same moment wait in turn.

## Version 2.2 — 29 October 2026 (end-to-end browser test of the running dashboard)

The running app was tested as a user would: real clicks, typing, scrolling, screenshots and downloads in the Claude desktop app's built-in browser, with every on-screen number compared against an independent calculation from the raw CSV (`QA_Testing/e2e/oracle.py`, which does not import `src/`). 101 cases: 88 passed on the first run, 13 failed; after the fixes below 100 pass and 1 Low cosmetic item stays open. Full report: `reports/E2E_Test_Report_v2.2.md`; case register: `QA_Testing/e2e/e2e_test_cases.csv`.

### Defects found and fixed

| ID | Severity | Problem | Root cause | Fix | Verified by |
|---|---|---|---|---|---|
| E2E-01 | Medium | After a server start (or an artifact refresh), the first change on any tab sent the user back to Overview | `st.cache_data(show_spinner=…)` adds a page element only when the cache misses. On the first, cold run the tab bar was element 3 of the page; on every later run it was element 2, so the browser treated it as a new widget and reset it. The v2.1 assessment had wrongly ruled this out as a scripting artefact | Both cached loaders use `show_spinner=False`; the call sites are wrapped in `st.spinner(…)`, which adds its element on every run, so positions never move | `test_page_layout_is_identical_on_cold_and_warm_cache_runs`; browser: fresh start, first change kept the Pipeline tab; a threshold change kept the tab and the open expander |
| E2E-02 | Medium | With "Daily" selected, the summary download and the explorer download had the same file name (`uac_pipeline_daily_<dates>.csv`) | Explorer file named "daily"; summary file named after the granularity | `uac_pipeline_summary_<granularity>_<dates>.csv` and `uac_pipeline_explorer_<dates>.csv` | `test_pipeline_download_file_names_are_distinct` (×3); browser: four distinct names |
| E2E-03 | Medium | A damaged `artifacts/daily_clean.csv` (missing column, header only, duplicate date, blank census, text in a number column) ended in Streamlit's generic "This app has encountered an error" in 6 of 10 scenarios | The artifact check covered `metadata.json` and the forecast file but not the daily table every tab reads; the pipeline's `validate_daily` did not check types | `check_daily()` added to the artifact check (rows, dates, columns, duplicates, order, types, gaps); `validate_daily` rejects non-numeric counts with `DataValidationError` | `test_damaged_daily_table_gives_a_readable_message` (×7), `test_text_in_a_cbp_column_disables_only_the_pipeline_tab`, `test_text_in_a_count_column_is_a_validation_error_not_a_crash`; `QA_Testing/e2e/resilience_scan.py` 10/10; browser: duplicate-date copy shows the message |
| E2E-04 | Low | Status caption read "Discharge slowdown: Normal (-19.0 % below 8-week median rate)" | A negative drop printed with the fixed word "below" | Negative values are shown as "19.0 % above 8-week median rate" | `test_negative_discharge_drop_is_described_as_above_the_baseline`; browser |
| E2E-05 | Low | A period with no evaluable indicator (first weeks of data) showed a green "No indicator reached Watch or Alert" | The empty-episodes branch did not distinguish "nothing evaluated" from "all clear" | Info message "No indicator could be evaluated in this period…" | `test_period_without_evaluable_indicators_is_not_reported_as_all_clear`; browser |
| E2E-06 | Low | The KPI download's first row was the raw key `reported_days` with no definition | Row built from the key list without a label | "Reported days in period" with the period dates as its definition | `test_kpi_download_labels_every_row_and_matches_the_cards`; `validate_downloads.py` |
| E2E-07 | Low | Summary table showed census and custody means unrounded (2273.1304) | No column format for the two mean columns | One decimal in the table (downloads keep full precision) | Browser |

### Files changed

| File | Change |
|---|---|
| `uac_forecast_dashboard.py` | E2E-01 to E2E-07 above (`check_daily`, `DAILY_COLS`, spinner placement, caption wording, info message, KPI file row, column formats, file names) |
| `src/data_loader.py` | `validate_daily` rejects non-numeric values in the five count columns |
| `tests/test_dashboard_integration.py`, `tests/test_data_validation.py` | 16 regression tests (fast suite: 97 → 113) |
| `QA_Testing/tests/test_dashboard.py` | The helper-extraction list also loads `check_daily` and `DAILY_COLS`, which `check_contract` now calls (expected results unchanged) |
| `QA_Testing/e2e/` (new) | `oracle.py`, `validate_downloads.py`, `resilience_scan.py`, case register, screenshots, downloaded files, damaged app copies for the browser checks |
| `.claude/launch.json` | Two launch entries for the damaged copies (ports 8504 and 8505) |
| `README.md`, `EXECUTION_GUIDE.md` | Test counts, end-to-end checks, three new known limitations, two troubleshooting rows |
| `reports/02_Review_Assessment_v2.1.md` | Correction note: the tab reset ruled out in v2.1 was real (E2E-01) |

## Version 2.1 — 29 October 2026 (review against the deliverable brief, live analytics, validation)

A review against the three-deliverable brief (research paper, live Streamlit analytics, executive summary) found the forecasting stack sound but four gaps in the descriptive "live analytics" requirements:

- no date-range filter;
- CBP custody never plotted;
- no transfer or discharge efficiency KPIs, and no backlog or bottleneck detection;
- no data explorer or downloads.

It also found 12 small numerical or wording errors in the documents (independent audit of 210 claims: 181 verified, 12 mismatches, 6 unverifiable, 11 design parameters). Full assessment: `reports/02_Review_Assessment_v2.1.md`.

### Files created

| File | Purpose |
|---|---|
| `src/data_loader.py` | Raw-CSV validation (`DataValidationError` naming column, file line and value; blank rows; ISO dates; duplicates kept in file order; negatives quarantined), cleaned-table checks, reported-day selection, raw-vs-clean reconciliation |
| `src/kpi_calculations.py` | `KPI_DEFINITIONS` (single source for the dashboard, README and paper), inclusive period filter, stock/flow-correct `period_kpis`, vectorised daily/weekly (Sun–Sat)/monthly `aggregate` |
| `src/bottleneck_detection.py` | `Thresholds` (validated), three no-look-ahead indicators (census growth, discharge slowdown, CBP holding time), Normal/Watch/Alert levels, alert episodes |
| `src/visualizations.py` | Plotly figures for the new tab (separate panels for different units, palette-consistent, status colours with icons) |
| `tests/` (5 modules, 97 tests) + `pytest.ini` | Data validation, KPI calculations, bottleneck detection, dashboard integration (AppTest, other working directory, missing `src/`, real server start-up), output consistency |
| `tools/pipeline_analysis.py` | Regenerates every number in paper section 10, `reports/pipeline_*.csv`, `reports/pipeline_analysis.md`, `figures/19_pipeline_monitoring.png` |
| `reports/02_Review_Assessment_v2.1.md`, `reports/Test_Report_v2.1.md` | Review, root-cause analysis, status and test evidence |

### Files modified

| File | Change |
|---|---|
| `uac_forecast_dashboard.py` | New 7th tab, "Pipeline analytics & data explorer", appended last so that every existing element keeps its position. It has: period presets and a custom date range; 12 KPI cards with definitions; a bottleneck status banner; 4 charts; detected episodes; a daily/weekly/monthly table; a data explorer; 4 CSV downloads. A forecast CSV download was added to the Care-load tab. `src/` is imported with graceful degradation (forecast tabs keep working without it). Empty, half-selected and reversed ranges show messages instead of errors. The module-level `ART` line is unchanged because the QA harness executes it in isolation. |
| `Research_Paper_UAC_Forecasting.md/.docx` | New section 10 (transfers, discharges and backlog) with Figure 19; old sections 10–12 renumbered 11–13. Abstract, limitation, recommendation 8 and conclusion sentences added. Appendix B gains the pipeline KPIs, alert rules and a length-of-stay definition reconciliation. Audit corrections: Friday wording (2 Friday reports exist), file line count, Q2–Q4 2024 transfer ratio, weekend-gap period label, 2024 month-end range, recursive-vs-direct week, gap decomposition rounding, per-horizon sample size (26–55), statsmodels citation, development-run figures labelled as not reproducible. Section 5.5 surge note: 13 Aug 2023 is interpolated (no reports 13–15 Aug). |
| `Executive_Summary_UAC_Forecasting.md/.docx` | New sections "Pipeline trends" and "Periods of operational pressure"; bottom-line point 5 (CBP wait, to verify); recommendation 6; apprehension-coverage and recorded-flow caveats; Friday wording; dashboard usage |
| `README.md` | Rewritten to the required structure (overview, folder tree, versions, dataset configuration, install and launch, tests, behaviour, alert rules, limitations, troubleshooting); fixes the "22 charts" and Friday statements |
| `EXECUTION_GUIDE.md` | Fast suite, pipeline analysis, port-conflict note |
| `QA_Testing/tests/test_dashboard.py`, `test_edge_cases_v2.py`, `measure_dashboard_coverage.py` | Isolated-copy helpers also copy `src/` (the app's new dependency; expected results unchanged). TC-UI-001 expects 7 tabs (spec change). |
| `QA_Testing/tests/test_regression_docs.py` | TC-RR-004 compares the 22 notebook figures (Figure 19 comes from `tools/`). TC-DOC-004 and TC-DOC-005 use the corrected sentence plus v2.1-only markers, so they now also prove the Word files were regenerated. TC-DOC-006 treats first-party packages (`src`) as local, which removes a false positive, and now also scans `src/*.py` for third-party imports. |

### Defects found and fixed during v2.1

| ID | Problem | Root cause | Fix | Verified by |
|---|---|---|---|---|
| D-PA-01 | Daily summary rerun took 3.99 s (> 3 s target) | `aggregate()` called `period_kpis` once per group (up to 715 Python calls) | Vectorised groupby with identical definitions (15–21 ms) | `test_aggregate_matches_period_kpis_on_real_data`; warm reruns 0.8–1.0 s |
| D-PA-02 | Isolated QA runs would fail to import the new tab's code | QA helpers copy only the app file and `artifacts/` | Helpers copy `src/`; app degrades gracefully if `src/` is missing | `test_missing_src_package_disables_only_the_pipeline_tab` |
| D-DOC-03 | 12 numerical or wording mismatches in paper, summary, README | Rounding and labels carried from earlier drafts | Corrected (see above) | `tests/test_output_consistency.py`, re-audit |

No analytical definition of the forecasting stack, model, artifact or notebook cell was changed.

## Version 2.0 — 28 October 2026 (investigation, validation, testing and quality improvement)

### Preservation of the original

Every v1 source and output was copied before any change to `archive/v1_baseline_2026-10-05/` (notebook, dashboard, paper and summary in .md/.docx, README, requirements, QA PDF, `artifacts/`, `figures/`, `.streamlit/`) with `SHA256_MANIFEST.txt` (42 files). The v1 notebook is also kept as an editable script in `notebook_source/UAC_Care_Load_Forecasting_v1.py`.

### Files inspected

`UAC_Care_Load_Forecasting.ipynb` (161 cells, AST-scanned and diffed output by output), `uac_forecast_dashboard.py`, `HHS_Unaccompanied_Alien_Children_Program.csv`, `artifacts/*` (12 files), `figures/*` (19), `requirements.txt`, `.streamlit/config.toml`, `.claude/launch.json`, `README.md`, `Research_Paper_UAC_Forecasting.md/.docx`, `Executive_Summary_UAC_Forecasting.md/.docx`, `QA_Testing/tests/*` (6 suites, framework, registry, report builder) and the v1 QA results. Findings: `reports/01_Investigation_Report.md`, `reports/investigation_findings.json`.

### Files modified

| File | Change |
|---|---|
| `UAC_Care_Load_Forecasting.ipynb` | Rebuilt from `notebook_source/UAC_Care_Load_Forecasting_v2.py` and executed top to bottom in a clean kernel (205 cells: 91 code, 114 markdown; 0 errors). Details below. |
| `uac_forecast_dashboard.py` | v2 dashboard (details below). |
| `Research_Paper_UAC_Forecasting.md` | Rewritten for v2: problem statement, methodological background with 22 references, validation, leakage audit, new experiments and results, limitations. |
| `Research_Paper_UAC_Forecasting.docx` | Regenerated from the v2 Markdown (first written to a temporary `_v2.docx` while Word held the file open; regenerated in place after Word was closed and the temporary copy removed, D-DOC-02). |
| `Executive_Summary_UAC_Forecasting.md/.docx` | Updated: problem, historical patterns, benefits, capacity risks, intake-uncertainty warning, deployment conditions. |
| `README.md` | v2 structure, commands and documents. |
| `QA_Testing/tests/test_data_pipeline.py` | TC-ING-009/011/012/013 and TC-FN-014 implement their stated expected results (v1 code returned *fail* unconditionally whatever happened, so a fix could never pass); expected results unchanged. |
| `QA_Testing/tests/test_artifacts_models.py` | 8 models, 17 artifacts, 672 forecast rows, 48 metric groups (the v2 specification). |
| `QA_Testing/tests/test_dashboard.py` | 8 models / 25 traces; new overview trace label; extraction of the new loader helpers. |
| `QA_Testing/tests/test_regression_docs.py` | Code-cell and figure counts for v2; wall-clock filter; TC-NBQ-005 now *executes* path resolution from a sub-folder and via `UAC_BASE_DIR` (it previously only searched the source text); TC-DOC-004 checks a v2-only marker. |
| `QA_Testing/tests/apptest_isolated.py` | Richer snapshot (info, warnings, model list, metrics) and a `no_plotly` scenario. |

### Files created

| File | Purpose |
|---|---|
| `reports/01_Investigation_Report.md`, `reports/investigation_findings.json` | Phase-1 findings with severity, evidence, impact, remediation and the dependency map |
| `notebook_source/UAC_Care_Load_Forecasting_v1.py`, `..._v2.py` | Editable percent-format sources (v1 export, v2 source of truth) |
| `tools/nb_export.py`, `tools/nb_build.py`, `tools/nb_sync_markdown.py`, `tools/md2docx.py`, `tools/investigate.py` | Export, clean-kernel build, markdown-only sync (refuses if any code cell differs), Markdown→Word, static investigation |
| `artifacts/validation_report.csv`, `model_fit_log.csv`, `feature_leakage_audit.csv`, `interval_coverage_by_h.csv`, `interval_method_comparison.csv` | New v2 artifacts |
| `figures/07c_monthly_pattern.png`, `11b_recursive_vs_direct.png`, `12b_coverage_by_horizon.png` | New v2 figures (22 in total) |
| `QA_Testing/tests/test_edge_cases_v2.py`, `synthetic.py` | 80 new Phase-10 edge-case tests on synthetic fixtures and the real app (222 registry tests in total) |
| `QA_Testing/tests/defects_v2.py`, `build_test_report_v2.py` | Defect register and report builder (counts computed from records) |
| `QA_Testing/pytest_suite/` | pytest layer: 8 category modules, `qa_bridge.py`, `conftest.py`, `pytest.ini` |
| `QA_Testing/requirements-test.txt` | Pinned test-only dependencies |
| `reports/Test_Report_v2.md`, `test_cases_v2.csv`, `defects_v2.csv`, `Test_Report_v2.xlsx` | Deliverable D |
| `QA_Testing/evidence/v2/` | Screenshots and `screenshot_map.csv` (Deliverable E) |
| `CHANGELOG.md`, `EXECUTION_GUIDE.md` | Deliverable H |

### Notebook changes (finding / defect IDs in brackets)

**Data validation and preparation**
- Portable paths: `UAC_BASE_DIR` or the first folder (working directory or a parent) containing the CSV; clear error if the file is missing [INV-11, D-CFG-02].
- §2.1 schema validation with `DataValidationError` messages naming column, file line and value; ISO dates accepted with a note [D-ING-01].
- Stable sort for duplicate dates; conflicting records listed; first record in file order kept [INV-06, D-PIPE-02].
- §3.1 plausibility validation: negative counts and stock spikes quarantined to missing and listed, never zero-filled; fractional counts and flow spikes flagged [D-ING-02]. Real data: 0 issues.
- Copy detection: no-block branch and a 14-report minimum run [D-PIPE-03]; origin-time check that the block and the regime breaks are identical with data up to the first forecast origin [INV-04/INV-07, D-PIPE-01 mitigation].
- `single_break` returns `None` on short series and the caller raises a clear error [D-FN-01].
- Per-series `OBSERVED` mask for scored and training targets [D-PIPE-05].
- Domain-definition table and accounting identity; recorded flows leave ~3 children/day unexplained (reconciliation term, not invented flows) [INV-10].

**EDA and features**
- Interpolation evidence for count variables (Sunday counts are single-day counts; weekend reconciliation within 0.45/day in the current regime; 14-day means differ 3–5% from reported-only means) [INV-09].
- Monthly pattern, outliers and surge periods, correlations (labelled as associations).
- §7.1 leakage audit: executed future-perturbation test (0 changed values, negative controls changed) and a documented definition for all 40 features; excluded calendar features recorded [INV-17].
- `training_table` keeps only finite rows [D-MOD-03].

**Models, validation and uncertainty**
- Seasonal naive (7d) baseline; model cards; minimum-history checks; `safe_forecast` fallback with `FIT_LOG` → `model_fit_log.csv` [INV-03, D-MOD-02]. Real data: 0 failures, 0 fallbacks.
- ARIMA vs SARIMA: AIC comparison table and out-of-sample walk-forward of the other family.
- Recursive vs direct multi-horizon experiment (same learner and inputs) with chart.
- `score()` returns NaN (never ±inf) for undefined ratios and reports excluded zero actuals; `diebold_mariano()` guards [INV-16, D-FN-02].
- Rolling-origin recalibration evaluated in real time (26- and 13-week windows), coverage by horizon, multi-horizon scorecard (1/7/14/28 days with bias and coverage), replay interval coverage in high-demand periods, production interval sanity checks and zero-floor counts [D-MOD-01 retest, INV-14].
- Design-decision evidence restated on calibration/replay data only (ML training window, month feature, alert trigger) [INV-04, D-NB-01].
- RISK table computed once per threshold (removed 4× duplicate computation) [INV-12].
- Scenarios explicitly labelled as hypothetical assumptions; capacity threshold labelled illustrative.
- Exports 5 new artifacts and new metadata keys (`models`, `model_colors`, `interval_method`, `accuracy_definition`, `capacity_note`, `validation_issues`, `fit_failures`, `coverage80_rolling_26wk`, `notebook_version`).
- All 20 new outputs carry an Observation written from the executed output (two were added after TC-NBQ-003 flagged them); 6 existing Observations were corrected where v2 numbers changed (timings, row counts, the seasonal-naive rows, stability ranking, artifact/figure counts). All other numeric outputs are identical to v1 (verified by an output-by-output diff), and two v2 runs are identical apart from wall-clock lines.

### Dashboard changes

- Horizon 1–28 days in daily steps; partial-week labels and day-weighted reconciliation in the weekly table; no shock-day slider when the horizon is 1 day [D-UI-09].
- Models derived from the artifacts (spare colours for unknown models); artifact contract check with readable messages for missing champion rows, unsupported horizons, missing metadata keys and non-finite values [D-UI-10].
- Missing plotting dependency → actionable error [D-UI-11].
- "Last report → forecast" marker on the overview, care-load and flow charts; observed series labelled; scenario `Type` column (model forecast / hypothetical); threshold labelled as user input; accuracy definition in tooltips and captions [D-UI-12].
- Breach probability shown as "n/a" with an explanation when a model has no error history (no false "Critical" status).
- Model-comparison tab shows the interval-method comparison; data-quality tab shows validation and model-failure counts.

### Known issues after v2

- D-MOD-01 (Medium, open): transfer 80% intervals under-cover (66–71%); use the 95% band for intake.
- D-PIPE-01 (Low, accepted): copy detection is retrospective; no effect on any evaluated origin.
- D-SEC-02 (Low, deferred): no authentication or security headers; localhost only.
- D-PERF-02 (Low, environment-dependent): the notebook took 8.4–14.0 minutes in three clean runs but 25.5 minutes in the isolated re-run under load, above the 20-minute target.
- Empirical 95% interval bounds are not forced to widen monotonically with the horizon (e.g. SARIMA lower bound 2,279 at day 21 vs 2,354 at day 28); documented, not changed.

## Version 1.0 — 5 October 2026

Initial notebook, dashboard, research paper, executive summary and QA cycle (see `archive/v1_baseline_2026-10-05/` and `QA_Testing_Report_UAC_Forecasting.pdf`, which describes v1).
