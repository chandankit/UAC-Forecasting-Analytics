# UAC Care Load & Placement Demand Forecasting

Forecasting and early warning for the U.S. Department of Health and Human Services (HHS) Unaccompanied Alien Children (UAC) Program: how many children will be in care over the next four weeks, how many placements to expect, and how much warning planners get before a capacity threshold is reached.

**Live dashboard:** deployment in progress; the public link is added here once it has been verified.

| Read | File |
|---|---|
| Research paper (14 pages: data, EDA, methods, results, insights and recommendations) | [PDF](Research_Paper_UAC_Forecasting.pdf) · [Word](Research_Paper_UAC_Forecasting.docx) · [Markdown](Research_Paper_UAC_Forecasting.md) |
| Executive summary for government stakeholders (4 pages) | [PDF](Executive_Summary_UAC_Forecasting.pdf) · [Word](Executive_Summary_UAC_Forecasting.docx) · [Markdown](Executive_Summary_UAC_Forecasting.md) |
| Appendix volume (KPI definitions, tuning logs, full model outputs, all source code) | [PDF](Research_Paper_Appendix.pdf) · [Word](Research_Paper_Appendix.docx) · [Markdown](Research_Paper_Appendix.md) |
| Executed analysis notebook | [UAC_Care_Load_Forecasting.ipynb](UAC_Care_Load_Forecasting.ipynb) |

Unified Mentor Data Science Internship, October 2026.

## Project overview

**What it is.** Children apprehended at the border pass from U.S. Customs and Border Protection (CBP) custody into HHS care, where they stay until they can be released to a sponsor. Shelter beds, case managers and clinicians take weeks to add, but the number of children in care can change within days.

**The problem.** Daily reports show what happened yesterday; they do not show what is coming. Using only the published daily reports, this project forecasts the number of children in HHS care, daily discharges and transfers, and the imbalance between them, 1–28 days ahead. It quantifies the uncertainty, provides an early-warning rule, and shows by out-of-sample testing how reliable each forecast is.

**Who it is for.** HHS and Office of Refugee Resettlement (ORR) program leadership, capacity planners and analysts.

## Key findings

- **The data needed repair before use.** A block of 251 CBP reports (12 Jan 2023 – 30 Jan 2024) copies values from later dates. Fridays and Saturdays are almost never reported, and five reports are stale repeats. All were detected and handled before any model was built.
- **The program changed regime in early 2025.** Intake fell about 95%, and the implied average stay in care rose from about 36 to about 203 days. Placement capacity now decides the care load.
- **Care load can be forecast closely.** The champion model (SARIMA) is 98.6% accurate on the untouched test window (mean absolute error 32 children, 43% below a naive forecast). The champions for discharges and transfers reach 64.6% and 58.8%.
- **Early warning gives about two weeks' notice of gradual build-ups.** In a real-time replay of 2023–24 with a simpler model, a 30% breach-probability alert flagged all 25 surge events, with a median 14-day warning and roughly one false alarm in four. Events at nearby thresholds overlap, so they are not 25 independent surges.
- **Outlook.** About 2,542 children by 18 January 2026 (80% interval 2,486–2,694), up from 2,484 on 21 December 2025.
- **Known limitation.** The champion's intake (transfer) 80% intervals cover only 66–71% of outcomes, so plan intake on the 95% band.

## Research and methodology

**Dataset.** 720 daily HHS reports from 12 January 2023 to 21 December 2025 ([HHS_Unaccompanied_Alien_Children_Program.csv](HHS_Unaccompanied_Alien_Children_Program.csv), 36 KB): children apprehended and placed in CBP custody, children in CBP custody, children transferred out of CBP custody, children in HHS care, and children discharged from HHS care. Stocks are point-in-time levels; flows are daily counts.

**Preprocessing.** Schema and plausibility validation run before any analysis. The data are placed on a daily calendar with time interpolation and a mask of genuine reports. Stale reports are set to missing, and the copied CBP block is masked. Every forecast is issued from a reported day and scored only on reported days, so interpolated values never leak future information.

**Exploratory analysis.** A stock-flow identity test proves the copied block (the daily change in care should equal transfers in minus discharges out). Change-point detection dates the structural break (26 Jan 2025 for transfers, 17 Feb 2025 for discharges). Little's law gives the implied length of stay. STL decomposition, ADF and KPSS tests, weekday profiles and correlations complete the analysis.

**Forecasting.** Eight approaches were compared: naive, seasonal naive and moving-average baselines; SARIMA and ETS; Random Forest and Gradient Boosting on 40 leakage-audited features with a direct multi-horizon design; and an equal-weight ensemble. Statistical models train on the current regime, and machine-learning models train on the trusted period with recency weights.

**Evaluation.** Walk-forward validation from 124 forecast origins at horizons of 1–28 days (52,464 scored forecasts). Champions were chosen on a calibration window (22 Jun – 20 Sep 2025) and reported on an untouched test window (21 Sep – 21 Dec 2025). Metrics are MAE, RMSE, MAPE, bias, accuracy (100 − WAPE) and Diebold–Mariano tests. Intervals come from walk-forward error quantiles, and early warning is evaluated by surge lead time and false-alarm rate.

The research paper gives the full method and results, and the Appendix volume has every table and log.

## Dashboard features

The Streamlit dashboard ([uac_forecast_dashboard.py](uac_forecast_dashboard.py)) has seven tabs and a sidebar for the forecast horizon (1–28 days), the models shown, the interval (80% or 95%), a capacity threshold and the history shown.

| Tab | What it shows |
|---|---|
| Overview | Latest census, forecast at the chosen horizon, breach risk against your threshold, accuracy, surge lead time |
| Care load forecast | Up to eight models with 80% and 95% intervals, a forecast table with CSV download, breach probability by day |
| Discharge demand | Discharge and transfer forecasts, weekly totals, net pressure |
| Model comparison | Walk-forward leaderboards, error by horizon, interval coverage, robustness |
| Scenario lab | Baseline forecast against labelled hypothetical scenarios, and a staffing and bed translator |
| Early warning & data quality | 2023–24 surge replay, alert operating curve, data-quality findings |
| Pipeline analytics & data explorer | CBP custody, HHS care, transfers and discharges for any date range: KPI cards, bottleneck alerts, charts, a data explorer and four CSV downloads |

Pipeline KPIs use reported days only. Stocks are never summed, and the alert rules use trailing windows only, so a past flag is exactly what an analyst would have seen on that day. The default alert rules can be adjusted in the tab, and a day's status is its worst indicator:

| Indicator | Watch | Alert |
|---|---|---|
| HHS census growth against 28 days earlier | ≥ +5% | ≥ +10% |
| 7-day discharge rate below its 8-week median | ≥ 15% | ≥ 30% |
| Implied CBP holding time (custody ÷ transfers, 7 days) | ≥ 2 days | ≥ 3 days (mirrors the 72-hour transfer standard) |

## Insights and recommendations

1. **Treat placement capacity as the primary lever.** The discharge rate fell to less than a fifth of its pre-break level. In the scenario model, 50% more placements turns growth into decline within a month.
2. **Add weekend placement and discharge processing.** In the current regime, net growth in the census has come across the unreported weekend gaps.
3. **Scale up on a probability trigger.** Begin scaling beds and staff when the 28-day breach probability for real capacity reaches 30%.
4. **Keep standby capacity that can be activated within days**, because abrupt intake jumps can leave as little as one day's warning.
5. **Fix the data feed.** Publish Friday and Saturday flows, non-CBP admissions, other exits and bed capacity, and correct the 2023 CBP records.
6. **Re-estimate every two weeks** and monitor interval coverage. Use the 95% band for intake.
7. **Verify the implied CBP holding time** (about 3 days since March 2025, against 1.3 before the break) with CBP custody-duration records before acting. It is an indirect estimate, not a measured custody time.

## Project structure

```text
UAC-Forecasting-Analytics/
├── uac_forecast_dashboard.py          Streamlit app (entry point)
├── src/                               analytics package used by the app's Pipeline analytics tab
│   ├── data_loader.py                 raw-CSV validation, cleaned-table checks, reported-day selection
│   ├── kpi_calculations.py            period filter, KPI definitions, daily/weekly/monthly aggregation
│   ├── bottleneck_detection.py        alert thresholds, indicators, alert levels and episodes
│   └── visualizations.py              Plotly figures for the pipeline tab
├── artifacts/                         model outputs the dashboard reads (cleaned data, forecasts, metrics, backtest)
├── HHS_Unaccompanied_Alien_Children_Program.csv   source data
├── UAC_Care_Load_Forecasting.ipynb    executed analysis notebook (205 cells)
├── notebook_source/                   editable percent-format source of the notebook
├── Research_Paper_UAC_Forecasting.*   research paper (.pdf, .docx, .md)
├── Research_Paper_Appendix.*          Appendix volume (.pdf, .docx, .md)
├── Executive_Summary_UAC_Forecasting.*   executive summary (.pdf, .docx, .md)
├── figures/                           32 charts (9 embedded in the research paper): figures/paper/ holds those 9 at print size; the 23 notebook charts are in the Appendix volume
├── reports/                           generated pipeline tables quoted in the paper, and the paper's quality report
├── archive/                           the original long paper, the source of the Appendix text
├── tools/                             notebook build, pipeline analysis, paper figures, paper build and quality checks
├── tests/                             pytest suite: data, KPIs, alert rules, dashboard, documents and repository
├── requirements.txt                   pinned runtime dependencies (used by Streamlit Community Cloud)
├── requirements-dev.txt               extra packages for the tests
├── .streamlit/config.toml             Streamlit settings (no tracebacks shown to visitors)
├── pytest.ini                         test configuration
└── CHANGELOG.md                       version history
```

## Run locally

Requires Python 3.13; it was tested with 3.13.14 on Windows 11.

```bash
git clone https://github.com/chandankit/UAC-Forecasting-Analytics.git
cd UAC-Forecasting-Analytics
python -m venv venv
```

Activate the environment:

| Platform | Command |
|---|---|
| Windows PowerShell | `.\venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `venv\Scripts\activate.bat` |
| macOS or Linux | `source venv/bin/activate` |

Then install the dependencies and start the dashboard:

```bash
pip install -r requirements.txt
python -m streamlit run uac_forecast_dashboard.py
```

Streamlit prints `Local URL: http://localhost:8501`. Open it in a browser and stop the app with Ctrl+C. To keep a local run on your machine only, add `--server.address localhost`. If port 8501 is busy, add `--server.port 8502`.

The artifacts are already in the repository, so the notebook does not need to run first. To regenerate them, run `python tools/nb_build.py notebook_source/UAC_Care_Load_Forecasting_v2.py UAC_Care_Load_Forecasting.ipynb` (8–26 minutes).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite covers data ingestion and validation, KPI definitions, the alert rules, dashboard integration including a real server start-up, consistency between the documents and the code, the research paper's quality checks, and the deployment settings. Add `-m "not slow"` to skip the server and subprocess tests.

Rebuilding the Word and PDF documents (`python tools/build_paper.py`) needs Windows with Microsoft Word. The built documents are in the repository, so this is only needed after editing them.

## Deployment

The dashboard runs on Streamlit Community Cloud with these settings: repository `chandankit/UAC-Forecasting-Analytics`, branch `main`, main file path `uac_forecast_dashboard.py`, and Python 3.13. Community Cloud installs `requirements.txt` and reads `.streamlit/config.toml`. The app uses only relative paths and the files in this repository, and it needs no secrets.

## Data source and licence

The data are daily aggregate counts from the HHS Unaccompanied Alien Children Program reports, supplied with the internship brief. They contain no personal information. Works of the U.S. federal government are generally in the public domain (17 U.S.C. § 105); check the publisher's terms before reusing the data elsewhere. No licence has been chosen yet for the code in this repository.

## Limitations

- The current regime has about nine months of data, so every model needs re-validation after the next major change.
- The data contain no bed-capacity figures, so the capacity threshold is a user input.
- Recorded flows leave about 3 children a day unexplained.
- The implied stays and holding times are Little's-law ratios. They approximate averages and do not measure any child's actual stay.
- Border events, policy announcements and sponsor-vetting rules are not in the data, so forecasts describe the system as it currently behaves.
