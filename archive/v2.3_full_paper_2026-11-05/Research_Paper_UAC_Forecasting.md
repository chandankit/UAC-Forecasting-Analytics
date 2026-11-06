# Forecasting Care Load and Placement Demand in the HHS Unaccompanied Children Program

## A regime-aware, walk-forward-validated early-warning approach (version 2.1)

**Unified Mentor Data Science Internship, October 2026**

**Data:** HHS Unaccompanied Alien Children (UAC) Program daily reports, 12 January 2023 to 21 December 2025 (file supplied with the internship brief)

**Companion artefacts:** `UAC_Care_Load_Forecasting.ipynb` (full executed analysis, v2), `uac_forecast_dashboard.py` (Streamlit dashboard), `src/` (descriptive pipeline analytics, v2.1), `tools/pipeline_analysis.py` (reproduces every number in section 10), `artifacts/` (all model outputs), `reports/` (investigation and test reports), `QA_Testing/` (automated tests and evidence). Version 1 of every file is preserved in `archive/v1_baseline_2026-10-05/`; `CHANGELOG.md` lists every change.

---

## Abstract

The Office of Refugee Resettlement (ORR) cares for unaccompanied children transferred from U.S. Customs and Border Protection (CBP) until they can be released to a sponsor. Planners need to know how many children will be in care in the coming weeks, whether placements will keep pace with arrivals, and when to scale shelters and staff. This study builds and evaluates a forecasting and early-warning system on three years of daily program data.

A root-cause audit came first, and it found that the dataset is not the "high-quality daily series" it appears to be. Every CBP record between 12 January 2023 and 30 January 2024 (251 reports) is a copy of a later date's values, which a stock-flow reconciliation test confirms (correlation 0.51 in the copied block vs 0.90 afterwards). Saturday is never reported and Friday almost never (2 of 720 reports), five reports are stale carry-forwards, and the program changed regime in early 2025: intake fell about 95% and the implied length of stay rose from about 36 to about 203 days. Even on consecutive reported days, care load rises about 3 children a day more than recorded transfers minus discharges, so the published flows do not close the population balance.

The pipeline therefore validates every input (schema and plausibility rules), interpolates **with masking**, uses regime-aware training windows and scale-free direct multi-horizon machine-learning targets, and evaluates strictly by walk-forward. An executed perturbation test found no feature leakage, and the data-quality masks are identical when derived only from data available at the first forecast origin. Champions are chosen on a 13-week calibration window and reported on an untouched 13-week test window. Eight approaches are compared: naive, seasonal naive, moving average, SARIMA, ETS, Random Forest, Gradient Boosting and an equal-weight ensemble.

On the test window:

- **Care load:** the champion (SARIMA) is 98.6% accurate with a mean absolute error of 32 children, 43% below naive persistence. The ensemble reaches 28.5 children (−49%), and its 4-week error is half that of naive.
- **Discharges:** the champion's error is about 3.5 placements a day, about one child a day above the noise floor implied by Poisson randomness.
- **Early warning:** alerting when the probability of breaching a capacity threshold reaches 30% detected all 25 historical surges in a real-time 2023–24 replay, with a median **14-day** warning.
- **Uncertainty:** care-load and discharge intervals are close to or above nominal coverage, but intake (transfer) 80% intervals cover only 66–71% of outcomes even with rolling recalibration, an open limitation.

A descriptive monitoring layer (version 2.1) complements the forecasts. Recomputed for any period from reported days only, it tracks CBP custody, HHS care, transfers and discharges with stock/flow-correct KPIs and flags backlog and bottlenecks with three documented threshold rules that use no future data. The HHS discharge rate fell from 2.88% of the census per day before the break to 0.52% in the current regime (0.44% in the last 90 days). The implied CBP holding time (custody ÷ transfers) rose from 1.32 to 2.96 days and was at or above the 3-day level, which mirrors the 72-hour transfer standard, on 44% of current-regime reports. This is an indicative signal that needs confirmation from CBP custody-duration records, not a measured custody time.

The outlook from 21 December 2025 is continued slow growth: 2,542 children by 18 January 2026 (80% interval 2,486–2,694). Placement capacity, especially at weekends, is the main lever for planners.

---

## 1. Introduction and problem statement

The UAC Program operates under high uncertainty. Border activity, enforcement policy and humanitarian crises can change the number of children entering federal care within weeks, and capacity (licensed beds, case managers, clinicians) takes weeks to scale. Descriptive reporting tells planners what happened yesterday; it does not tell them what is coming.

**Problem statement.** Given only the program's published daily reports, produce forecasts of (a) the number of children in HHS care, (b) daily discharges and transfers, and (c) the imbalance between them, for 1–28 days ahead, with calibrated uncertainty and an early-warning rule, and show by honest out-of-sample evaluation how reliable each is.

**Objectives.**

1. Forecast the number of children in HHS care 1–28 days ahead.
2. Estimate the future imbalance between intake (transfers in) and exits (discharges to sponsors).
3. Predict short-term discharge (placement) demand.
4. Provide early warnings with quantified uncertainty.
5. Compare statistical and machine-learning approaches, and evaluate reliability across horizons.

**Contribution.** Beyond the forecasts themselves, the study shows that **data-quality root-cause analysis and leakage control change the conclusions**. A naive pipeline on this file would have trained on fabricated intake records, interpolated future values into its features, and (as Section 6.3 shows) learned a policy shock as "January seasonality".

## 2. Methodological background

- **Statistical forecasting.** ARIMA and seasonal ARIMA models follow Box et al. (2015); orders are chosen by Akaike's information criterion (Akaike, 1974). Exponential smoothing is used in its innovations state-space form (Hyndman et al., 2008), and the general workflow follows Hyndman & Athanasopoulos (2021). Seasonal decomposition uses STL (Cleveland et al., 1990); stationarity is checked with the ADF (Dickey & Fuller, 1979) and KPSS (Kwiatkowski et al., 1992) tests, whose null hypotheses are opposite and therefore complementary.
- **Machine learning.** Random Forests (Breiman, 2001) and gradient boosting (Friedman, 2001) are used through scikit-learn (Pedregosa et al., 2011); the boosting loss is Huber's robust loss (Huber, 1964). Multi-step strategies (direct vs recursive) are compared following the taxonomy of Ben Taieb et al. (2012).
- **Evaluation.** Rolling-origin (walk-forward) evaluation follows Tashman (2000). Scale-dependent and percentage error measures, and the problems of percentage errors near zero, are discussed by Hyndman & Koehler (2006). Forecast-accuracy differences are tested with the Diebold–Mariano test (Diebold & Mariano, 1995) with the small-sample correction of Harvey et al. (1997) and a Bartlett-kernel long-run variance (Newey & West, 1987).
- **Uncertainty.** Intervals are built from empirical out-of-sample errors, in the spirit of split-conformal prediction (Vovk et al., 2005); adaptive conformal inference under distribution shift (Gibbs & Candès, 2021) is identified as the next step for intake.
- **Change points and queues.** The regime break is located by single-split least-squares segmentation (binary segmentation; see Truong et al., 2020). Average length of stay is derived from Little's law (Little, 1961).

## 3. Data

The source file contains 1,170 rows below its header line, of which 720 are daily reports and 450 are blank padding rows. Each report has six fields:

| Field | Meaning | Type | Role in the HHS population balance |
|---|---|---|---|
| Date | Reporting date (text, newest first) | index | |
| Children apprehended and placed in CBP custody | CBP intake | daily count | upstream, not part of the HHS balance |
| Children in CBP custody | CBP care load | stock (snapshot) | upstream pipeline |
| Children transferred out of CBP custody | Flow into HHS | daily count | admissions from CBP (one movement, counted once) |
| Children in HHS care | HHS care load | stock (snapshot) | closing population |
| Children discharged from HHS care | Sponsor placements | daily count | exits |

The distinction between **stocks** (levels) and **flows** (daily counts) matters throughout: a stock must reconcile with its flows, and that identity becomes the main data-quality test. The testable balance is *closing = opening + transfers in − discharges + (unrecorded other admissions − other exits)*. The file contains no columns for non-CBP admissions, other exits or bed capacity; these gaps are measured and reported (Section 4.6) rather than invented, and every capacity threshold in this study is a user input or an illustration.

## 4. Data quality: validation and root-cause analysis

### 4.1 Input validation (new in version 2)

Two validation layers run before any analysis. **Schema validation** stops a malformed file with a message naming the column, file line and value: six columns in the documented order, at least one data row and at least 365 reports, dates in the documented format (ISO dates accepted and reported; time-stamped or other text rejected), numbers with optional thousands separators (blank and `N/A` mean missing; other text is rejected), no empty or constant target columns, and a daily cadence (median gap ≤ 3 days). **Plausibility validation** quarantines impossible values (negative counts; a care load more than 1.5× or less than 1/1.5× the median of the three reports on either side) by setting them to missing and listing them (never replacing them with zero), and flags unusual but possible values (non-integer counts; flows more than 5× their local median) for review.

On the real file every check passes and nothing is quarantined or flagged; the largest care-load deviation from neighbouring reports is ×1.11 and the largest flow ratio 3.2×, far inside the limits. The negative cases are exercised on synthetic files in the automated test suite (Section 8).

### 4.2 Reporting cadence

Reports arrive Sunday to Thursday. In three years there are only 2 Friday reports and none on a Saturday (Figure 1). Gaps of 3 days (Thursday→Sunday) are the second most common interval, and about a third of calendar days (355 of 1,075) have no report. Missing days are therefore **structural, not random**, and Friday/Saturday flows are never observed.

![Figure 1. Reports by weekday and gaps between reports](figures/01_reporting_cadence.png)

### 4.3 Stale reports

Five reports repeat the previous report's care load *and* discharges exactly, while their CBP fields change: 8 Jan 2024, 26 Feb 2024, 25 Jun 2024, 27 Mar 2025 and 14 May 2025. These are carry-forwards. They were set to missing, re-estimated by interpolation, and never used as a scored target.

### 4.4 Copied CBP records (the key finding)

A CBP report is a triple (apprehended, in custody, transferred), and identical triples on different dates should be rare. In this file, 374 of 720 rows share their triple with another date. The earlier copies form one **contiguous block of 251 reports from 12 Jan 2023 to 30 Jan 2024**; 100% of 2023 CBP rows are copies. For example, the triple (56, 82, 68) appears on 1 Jun 2023, 15 Oct 2023, 27 Nov 2023, 26 Dec 2023 and 9 Jan 2025, each time alongside a completely different HHS care load. (Version 2 requires a run of at least 14 consecutive copies before declaring a corrupted block, so isolated chance repeats at low counts on a future refresh are not masked.)

**Proof via the stock-flow identity.** On consecutive days, the change in HHS care should equal transfers in minus discharges out. Figure 2 tests this:

| Period | Day pairs | corr(ΔHHS, net flow) | Mean residual per day |
|---|---|---|---|
| Copied CBP block (2023 – Jan 2024) | 192 | **0.51** | **+190 children** |
| Trusted (Feb 2024 →) | 357 | **0.90** | +2.6 children |

![Figure 2. Stock-flow identity: copied block vs trusted period](figures/02_stock_flow_identity.png)

Admissions implied by the HHS columns alone (ΔHHS + discharges) ran at 280–400 per day in 2023, while recorded transfers captured only 34–57% of that. In Q2–Q4 2024 recorded transfers match implied admissions (97–100%); at the very low 2025 volumes the ratio is 67–92%, consistent with the roughly 3 children a day that enter care outside recorded transfers (section 4.6). **Conclusion:** the CBP columns for 2023 – January 2024 are corrupted; the HHS columns reconcile wherever they can be checked (February 2024 onward) and are kept for the whole history, although they cannot be cross-checked inside the copied block itself.

**Origin-time check.** Copy detection is retrospective: a copy is recognised only once its later original exists. All originals of the copied rows are dated 26 June 2024 – 12 February 2025, before the first forecast origin (22 June 2025). Re-running the detection, and the change-point search of Section 5.1, with data up to that origin only gives the identical block and identical break dates, so the masks and training windows used in the backtest would have been available in real time.

### 4.5 Decisions

| Issue | Evidence | Treatment |
|---|---|---|
| 450 blank rows; text dates; comma numbers; newest-first order | Raw audit | Dropped, parsed, typed, sorted (stable sort) |
| Duplicate dates (none in this file) | Stable sort + conflict listing | Identical repeats dropped; for conflicting repeats the first record in file order is kept and printed (the file carries no revision timestamp) |
| Malformed or implausible values on a refresh (none in this file) | Validation rules (4.1) | Stop with a clear message, or quarantine to missing and list |
| Saturday never reported, Friday almost never (2 reports) | Weekday counts | Daily calendar + time interpolation, with an `is_reported` mask |
| 5 stale HHS reports | Exact repeats | Set to missing, interpolated, never scored |
| CBP block copied (Jan 2023 – Jan 2024) | Duplicate triples + broken identity | CBP fields masked; ML trained on the trusted period only |
| HHS columns | Reconcile where CBP is trusted | Kept for the full history (EDA, surge replay) |

**Leakage guard.** An interpolated Friday value is computed from the following Sunday. A forecast issued "as of Friday" would therefore secretly know Sunday. Every forecast is issued from a reported day and scored only on reported days with a genuine value for that series, so interpolated values fill rolling-window features *between* reports but never carry future information.

### 4.6 What the counts mean, and whether interpolating them is justified

Interpolating a stock between snapshots is natural; interpolating daily **counts** needs evidence. Three checks were run (current regime unless stated):

1. **Sunday counts are single-day counts, not 3-day totals.** Sunday discharges average 7.2 against 12.8 on Monday–Thursday; a weekend total would be about three times a weekday figure.
2. **Interpolated weekend flows reconcile with the stock.** The care-load change not explained by recorded flows is 2.97 children/day between consecutive weekday reports and 3.42/day across Thursday→Sunday gaps with interpolated Friday/Saturday flows, a difference of about 0.45/day. Across the trusted period from February 2024 onward, which the high-volume 2024 reports dominate, the weekend gap was large (−18.8/day vs +2.6/day), so 2024 weekend flow features are biased; they enter only the ML training rows, which a 180-day recency half-life down-weights.
3. **The choice barely changes model inputs.** 14-day flow means from the interpolated grid differ by 3.2% (transfers) and 4.8% (discharges) from means built on reported days only, with correlation ≥ 0.999.

Zero-filling was rejected because the stock keeps moving over weekends. The constant ~3 children/day that recorded flows never explain is consistent with admissions from outside CBP or a definitional difference; it is carried as a measured reconciliation term (+3.35 children/day over the last 60 days), not as a modelled flow.

## 5. Exploratory analysis

### 5.1 A structural break in early 2025

Single change-point detection on log flows places the break in **transfers on 26 Jan 2025** and in **discharges on 17 Feb 2025**. The current regime is dated from 17 Mar 2025, four weeks later, allowing the backlog to drain.

| Average per day | Feb 2024 – break | Current regime | Change |
|---|---|---|---|
| Apprehended (CBP intake) | 146.0 | 7.8 | −94.7% |
| Transfers into HHS | 209.9 | 9.6 | −95.4% |
| Discharges from HHS | 203.2 | 11.7 | −94.3% |
| Children in HHS care | 6,751 | 2,262 | −66.5% |

![Figure 3. Care load and daily flows, 2023–2025](figures/04_history_overview.png)

### 5.2 The bottleneck moved from intake to placement

By Little's law, average length of stay ≈ children in care ÷ daily discharges. It rose from about **36 days** before the break to about **203 days** in the current regime (Figure 4). The program shrank, but placements shrank *faster than the caseload*. Care load now responds mainly to the pace of discharges, which makes **placement capacity the primary planning lever**.

![Figure 4. Implied average length of stay](figures/05_length_of_stay.png)

### 5.3 Net pressure as the leading indicator

Net pressure (transfers − discharges) is the flow signal that drives the stock. In the current regime it explains the shape of the care-load curve (Figure 5): net outflow (down to −19 children/day in mid-July) drove the June–August 2025 drawdown (about 2,500 → 1,972). Small recorded net inflows since September (about +1/day) coincide with the climb, but explain only part of it; the reconciliation term and the weekend pattern below account for the rest.

![Figure 5. Current-regime care load and net pressure](figures/06_net_pressure.png)

### 5.4 Weekly structure and the "weekend ratchet"

On reported days in the current regime, discharges peak mid-week (Thursday 14.9, Wednesday 13.5) and are lowest on Sundays (7.2). Transfers are flatter (8.7–10.8). Comparing care-load changes on weekday transitions with changes across the unreported Thursday→Sunday gap reveals a striking pattern:

| Period | Weekday change / day | Weekend (Thu→Sun) change / day |
|---|---|---|
| 2023 – Jan 2024 | +66.8 | −76.6 |
| Feb 2024 – break | +19.4 | −64.5 |
| **Current regime** | **−0.3** | **+3.4** |

Over the current regime, care load rose by 373. Weekday transitions netted **−40**, Thursday→Sunday gaps **+323** and other multi-day reporting gaps (holidays, a June outage and removed stale reports) **+90**. In other words, **net growth has come across weekends**, when arrivals continue but placements slow. The pattern is the reverse of the high-volume era, which is another reason to model the current regime separately.

![Figure 6. The weekend ratchet](figures/07b_weekend_ratchet.png)

### 5.5 Monthly pattern, outliers, surge periods and correlations

- **No stable month-of-year pattern.** Month-end care load followed a different path every year (peak 11,516 on 20 Dec 2023; month-end values of about 6,000–8,700 through 2024; collapse in early 2025), and the month-to-month direction agreed between 2023 and 2024 in only 5 of 11 months (Figure 7). Each calendar month occurs at most three times, once on each side of the break, so month effects cannot be estimated.
- **Outliers.** The largest robust deviations (MAD z-scores against a centred 29-day median) occur at the low volumes of 2025: 29 transfers on 11 Sep 2025 against a local median of 11 (z = 12.1) and 39 discharges on 10 Jul 2025 against 17 (z = 4.9). They pass the plausibility rules and are kept.
- **Surge periods.** The fastest 28-day rises in care load were all in 2023: +2,806 (+43%) to 13 Aug 2023 (on the interpolated grid, because no reports exist for 13–15 Aug 2023; on reported days the largest rise is +37.6% to 16 Aug 2023, section 10.4), +1,342 (+20%) to 9 Feb 2023 and +1,695 (+17%) to the 11,516 peak on 20 Dec 2023.
- **Correlations (associations, not causal effects).** In the current regime the daily care-load change rises with transfers (+0.47) and falls with discharges (−0.61). Transfers correlate 0.38 with same-day apprehensions but 0.59 and 0.58 with apprehensions one and two days earlier, so CBP intake leads HHS transfers by one to two days.

![Figure 7. Monthly pattern by year](figures/07c_monthly_pattern.png)

### 5.6 Decomposition and stationarity

STL decomposition (weekly period, robust fitting) gives a weekly seasonal strength of 0.42 for discharges but only 0.14 for care load (Figure 8). With fewer than ten months in the current regime and an irregular reporting cadence, the decomposition is descriptive; no annual component can be estimated.

- **Care load.** ADF and KPSS are inconclusive on the short regime window (low test power). Because a stock is integrated by construction (care_t = care_t−1 + in − out) and shows multi-month trends, care load is modelled with one difference (d = 1).
- **Flows.** Transfers test stationary. Discharges test non-stationary in level because of a temporary mid-2025 placement push, and are modelled in logs with a constant (d = 0), so forecasts revert to the regime's typical level instead of extrapolating a temporary surge.

![Figure 8. STL decomposition, current regime](figures/08_stl_decomposition.png)

## 6. Methodology

### 6.1 Time-series preparation

1. Re-index onto a continuous daily calendar (1,075 days).
2. Time-weighted linear interpolation of interior gaps (justified in Section 4.6).
3. An `is_reported` mask that restricts forecast origins to genuine, non-stale reports, and a per-series "observed" mask that restricts scored and training targets to genuine values of that series.
4. CBP fields masked in the copied block.

### 6.2 Feature engineering and leakage audit

All features describe information available at the end of origin day *t*. The ML models use a direct multi-horizon design: one row per (origin, horizon h = 1…28), so a single model learns all horizons and never needs predicted or actual future values as inputs.

| Group | Features |
|---|---|
| Lags (t−1, t−7, t−14) | care-load log-changes; flow lags relative to their 14-day mean |
| Rolling mean & variance (7 / 14 days) | care-load deviation from rolling mean, rolling volatility; flow rolling means and log-variances |
| Flow-based signals | net pressure per 100 children (origin day, 7- and 14-day means); CBP pipeline per 100 children; flow rates per 100 children |
| Calendar | target weekday, US federal-holiday window (±1 day), origin weekday, horizon |
| Seasonal anchor | same weekday last week relative to today's level |

**Scale-free targets.** The 2024 and 2025 regimes differ twentyfold in volume. ML targets are therefore log-ratios: log(care_t+h / care_t) for the stock, and log1p(flow_t+h) − log1p(14-day mean) for flows.

**Leakage audit.** Each of the 40 features is documented with its definition, source columns, look-back window, availability at the origin and horizon suitability (`artifacts/feature_leakage_audit.csv`). An executed perturbation test then scaled every report after a cut-off (18 Oct 2025) by 1.5, rebuilt the whole preparation and all features, and compared values at the 410 reported origins up to the cut-off: **0** of the feature values changed (40 features × 3 series × horizons 1/7/14/28). Negative controls confirm that the test detects look-ahead: the target changed in 99 rows, and 2 of the 3 interpolated days just before the cut-off changed. The net-pressure feature is labelled as an incomplete balance (Section 4.6).

### 6.3 A second root-cause finding: the month feature

*The figures in the first two paragraphs of this subsection come from a development run that is not kept in the delivered notebook, so they cannot be reproduced from the delivered outputs. The exceptions are the post-removal test errors (38 and 31), which are.* The first full run included the target month as a calendar feature. It became the **single most important care-load feature** (about 30% of Random Forest importance), and the production forecast showed Random Forest falling from 2,525 on 31 December to 2,282 on 1 January, a **10% cliff at the month boundary**. The trusted history contains only one January, January 2025, which is the policy-break crash. The model had learned a regime change disguised as seasonality.

The month feature was removed because of that cliff, not because of a test score. For transparency, removing it raised Random Forest's test error (MAE 25 → 38) but lowered Gradient Boosting's (40 → 31), and the ensemble stayed the best test model. Week of year has the same flaw and was never added. The holiday-window feature changes forecasts by at most about 4 children and was kept.

### 6.4 Train–test design

| Component | Data used |
|---|---|
| Statistical models (SARIMA, ETS) | current regime, 17 Mar 2025 → origin |
| ML models (RF, GB) | trusted period, 14 Feb 2024 → origin, recency-weighted (180-day half-life) |
| Calibration window (champion selection, interval calibration) | 22 Jun – 20 Sep 2025 |
| Test window (reporting only) | 21 Sep – 21 Dec 2025 |

![Figure 9. Train–test design](figures/10_train_test_design.png)

Forecasts are issued from **every reported day** (124 origins) for horizons 1–28. Models are re-estimated every 14 days and absorb each new observation in between, mirroring production use. The backtest produced 83,328 forecasts (7 models plus the ensemble), of which 52,464 had reported targets. A model whose fit or prediction fails is logged and replaced by naive persistence for that origin (flagged); in the delivered run no fit failed. The 28-day horizon covers four complete weekly cycles; 30-day forecasts were not validated.

**Training-window ablation.** A development experiment, not re-run in the delivered notebook, compared three ML training designs: the trusted window with recency weights, the trusted window without weights, and the current regime only. The decision rests on the **calibration** window alone: regime-only training was far worse early in the regime (calibration MAE 125 vs 89 for Random Forest, 126 vs 97 for Gradient Boosting, 5.8 vs 4.2 for discharges), when only about three months of history existed, so the recency-weighted trusted window was adopted.

### 6.5 Models

The statistical models and the STL, ADF and KPSS diagnostics use statsmodels (Seabold & Perktold, 2010); the machine-learning models use scikit-learn (Pedregosa et al., 2011).

| Family | Model | Specification | Failure condition → fallback |
|---|---|---|---|
| Baseline | Naive persistence | ŷ_t+h = y_t | none |
| Baseline | Seasonal naive (new) | ŷ_t+h = y at the same weekday of the last week (≤ t) | none |
| Baseline | Moving average | 7-day mean, flat | none |
| Statistical | SARIMA | orders by AIC before the calibration window: care (1,1,1); discharges (1,0,1)(1,0,1)₇ + constant on log1p; transfers (1,0,0) + constant on log1p | < 56 days of history or non-convergence → previous fit, else naive |
| Statistical | ETS | care: additive error, damped additive trend, weekly season; flows: log1p, weekly season | < 28 days → previous fit, else naive |
| ML | Random Forest | 200 trees, min leaf 5, 50% features, 70% bootstrap sample | < 200 training rows → previous fit, else naive |
| ML | Gradient Boosting | 150 trees, depth 3, learning rate 0.07, Huber loss | as RF |
| Hybrid | Ensemble | equal-weight mean of SARIMA, ETS, RF, GB | components that fail contribute their flagged fallback |

**ARIMA vs SARIMA.** For each series the best non-seasonal and the best weekly-seasonal specification were compared (18 specifications fitted per series on pre-calibration data). Weekly terms earn their place only for discharges (ΔAIC −11.5: 74.9 vs 86.4); for care load and transfers the seasonal versions are worse (+3.4 and +3.7). Walk-forward errors on the calibration window confirm every AIC choice (care 55.1 vs 55.3, discharges 4.09 vs 4.76, transfers 4.47 vs 4.60, selected vs other family).

### 6.6 Evaluation metrics

MAE, RMSE, MAPE (non-zero actuals only; the number of excluded zero actuals is reported, e.g. 18 discharge forecasts on test), WAPE = Σ|error| / Σ|actual|, and **Forecast Accuracy = 100 − WAPE** = 100 × (1 − Σ|y − ŷ| / Σ|y|). Accuracy is a volume-weighted complement of relative error, not a transformation of MAE or MAPE; it is 100 for a perfect forecast, can be negative when errors exceed volumes, and is undefined (reported as missing, never as a number) when all actuals are zero. Bias = mean(ŷ − y). Errors are analysed by horizon (1–28 days, weekly buckets, and a scorecard at 1/7/14/28 days). Diebold–Mariano tests with Bartlett variance and the Harvey correction test whether gains over naive are significant.

### 6.7 Uncertainty quantification

For each model, series and horizon, intervals add the empirical quantiles of out-of-sample errors (pooled over h ± 2) to the point forecast. Two calibration rules are evaluated on the test window: the **fixed calibration window** (split conformal), and the **rolling-origin rule** that the production forecast uses: at each origin, the error quantiles of all forecasts from the previous 26 weeks of origins whose target date has already been observed (a 13-week variant as sensitivity check). The rolling rule was fixed before its test results were inspected. Flow bounds are floored at zero; flooring cannot change coverage for non-negative actuals but is a truncation, not a count model.

### 6.8 Recursive vs direct multi-horizon experiment

To isolate the multi-step strategy, two Gradient Boosting models (100 trees, depth 3, Huber loss, recency weights) were given identical target-only inputs (the last 14 transformed values, their 7- and 14-day means, and the target weekday) and walked forward over the same 124 origins (refit every 28 days): a **direct** stacked model told the horizon, and a **recursive** one-step model fed its own predictions. The recursive model must step through every calendar day, so it trains on interpolated Friday/Saturday values; both are scored on reported days only.

### 6.9 Early-warning design

A **surge event** is the first reported day care load crosses threshold *C* after at least 14 days below it. An **alert** fires at origin *t* when the probability of exceeding *C* within 28 days reaches *p*\*. **Lead time** is the length of the unbroken alert streak before the event. Sweeping *C* over many thresholds produces many events. The system is evaluated on the 2025 test window (champion and its calibrated errors) and in a **real-time replay of 2023–24** with a rolling 120-day damped-trend ETS model and analytic intervals (only the HHS columns are trustworthy then, so the replay model is univariate). The operating point p\* = 30% was chosen on the replay curve; the test window played no part in the choice.

### 6.10 Scenario engine

Scenarios modify forecast flows and propagate the change through the stock-flow identity:

care^scenario_t+h = care^baseline_t+h + Σ (Δtransfers − Δdischarges)

The baseline path is a model forecast; every other path is a **hypothetical assumption**, not a prediction, and carries no probability of its own. The same engine runs in the dashboard's Scenario lab.

## 7. Results

### 7.1 Accuracy on the untouched test window

**Children in HHS care** (champion chosen on calibration: SARIMA)

| Model | MAE | RMSE | MAPE % | Accuracy % | Bias |
|---|---|---|---|---|---|
| **Ensemble** | **28.5** | 42.0 | 1.23 | **98.8** | −17.1 |
| Gradient Boosting | 30.9 | 42.0 | 1.34 | 98.7 | −13.9 |
| ETS | 31.3 | 45.3 | 1.34 | 98.7 | −13.8 |
| SARIMA (champion) | 32.0 | 46.0 | 1.37 | 98.6 | −15.9 |
| Random Forest | 37.8 | 60.5 | 1.65 | 98.4 | −24.7 |
| Naive | 55.7 | 66.9 | 2.38 | 97.6 | −54.7 |
| Seasonal naive (7d) | 67.1 | 76.5 | 2.87 | 97.1 | −66.5 |
| Moving Avg (7d) | 67.6 | 77.2 | 2.89 | 97.1 | −67.2 |

**Discharges per day** (champion: Ensemble)

| Model | MAE | RMSE | Accuracy % |
|---|---|---|---|
| **SARIMA** | **3.33** | 4.01 | **65.9** |
| Ensemble (champion) | 3.46 | 4.19 | 64.6 |
| Moving Avg (7d) | 3.63 | 4.44 | 62.9 |
| ETS | 3.69 | 4.60 | 62.3 |
| Gradient Boosting | 3.78 | 4.59 | 61.4 |
| Random Forest | 3.80 | 4.61 | 61.2 |
| Seasonal naive (7d) | 4.08 | 5.06 | 58.3 |
| Naive | 4.60 | 5.73 | 53.0 |

**Transfers per day** (champion: Moving Avg (7d)): ETS 3.98, SARIMA 4.04, Ensemble 4.07, Moving Avg 4.23, RF 4.44, GB 4.71, seasonal naive 5.09, Naive 5.12.

Every fitted model beats naive persistence on care load; the flat and weekly baselines do not, because a rising stock has no weekly repeat. Seasonal naive helps only for discharges (−11% vs naive), the one series with a strong weekly cycle. All modelled care-load forecasts are **biased low** (−14 to −25 children) because the test window was a steady climb. Under-forecasting is the dangerous direction for capacity, which is one reason decisions should use the upper interval and breach probabilities rather than the point forecast.

Discharge errors of about 3.3–3.5 per day look large in percentage terms. At about 10 placements a day, however, pure Poisson randomness alone implies a mean absolute error of about 2.5, so the models are within about one child a day of the irreducible noise. Summing the reported (Sunday–Thursday) days within each week cuts SARIMA's relative error on discharges from 34% to 16% (naive: 37%), so **weekly totals** are the better planning unit.

### 7.2 Horizon error and the multi-horizon scorecard

![Figure 10. MAE by forecast horizon (test window)](figures/11_horizon_error.png)

| Care load MAE | 1–7 days | 8–14 days | 15–21 days | 22–28 days |
|---|---|---|---|---|
| Naive | 17.9 | 44.9 | 71.7 | 100.5 |
| SARIMA | 13.4 | 28.4 | 39.9 | 52.0 |
| ETS | 13.1 | 28.0 | 38.8 | 51.1 |
| Gradient Boosting | 19.7 | 23.0 | 32.3 | 53.3 |
| Ensemble | 13.0 | 24.0 | 33.7 | **48.4** |

| Scorecard (test) | h = 1 | h = 7 | h = 14 | h = 28 |
|---|---|---|---|---|
| Care load, SARIMA: MAE / bias | 5.8 / +1.6 | 19.2 / −3.0 | 32.5 / −13.8 | 58.6 / −44.0 |
| Care load, Naive: MAE | 5.7 | 29.0 | 55.5 | 111.2 |
| Discharges, Ensemble: MAE / bias | 3.02 / +0.31 | 3.01 / +0.27 | 3.66 / +1.12 | 3.70 / +1.78 |
| Transfers, Moving Avg: MAE / bias | 3.53 / +0.45 | 4.13 / +0.71 | 4.38 / +0.94 | 4.63 / +1.65 |
| 80% coverage (rolling 26-week rule): care / discharges / transfers | 94 / 89 / 77% | 91 / 85 / 71% | 90 / 66 / 70% | 88 / 71 / 52% |

Naive error grows about 5.6× from week 1 to week 4; the modelled forecasts grow 2.7–4.0×. At four weeks the ensemble's error is **half** of naive, and SARIMA is 34% below naive at 7 days and 41–47% below at 14–28 days. Flow errors are almost flat across horizons. Each single-horizon figure rests on 42–55 forecasts, so single-horizon differences are noisy.

### 7.3 Statistical vs machine-learning, and recursive vs direct

| Best of family (test MAE) | 1–7 d | 8–14 d | 15–21 d | 22–28 d |
|---|---|---|---|---|
| Care load: Statistical | 13.1 | 28.0 | 38.8 | 51.1 |
| Care load: Machine learning | 16.9 | **23.0** | **32.3** | 53.3 |
| Care load: Ensemble | **13.0** | 24.0 | 33.7 | **48.4** |
| Discharges: Statistical | 3.3 | **3.3** | **3.4** | **3.3** |
| Discharges: Machine learning | **3.1** | 3.7 | 4.2 | 4.1 |
| Transfers: Statistical | **3.8** | **4.0** | **4.0** | **4.0** |
| Transfers: Machine learning | 4.1 | 4.3 | 4.6 | 4.8 |

Statistical models are the better choice for the **flows** at almost every horizon. ML adds value for the **stock** at medium horizons (8–21 days). The **ensemble is the most robust** all-rounder: it beat naive in 100% of test weeks for care load.

| Recursive vs direct (target-only GB, MAE) | 1–7 d | 8–14 d | 15–21 d | 22–28 d |
|---|---|---|---|---|
| Care load, test: direct / recursive | 24.3 / **13.0** | 27.0 / **23.9** | 41.1 / **33.1** | 64.7 / **40.6** |
| Care load, calibration: direct / recursive | 41.7 / **21.5** | 67.8 / **58.6** | 108.0 / **102.3** | 144.1 / **142.5** |
| Discharges, test: direct / recursive | 3.47 / **3.16** | **3.64** / 3.77 | **3.61** / 4.13 | **3.68** / 4.17 |

![Figure 11. Recursive vs direct multi-horizon forecasting (test window)](figures/11b_recursive_vs_direct.png)

With identical inputs, the recursive strategy is better for the smooth care-load stock at every horizon in both windows, while for the noisy discharge flow it is better only in week 1 and worse from week 2 on the test window (from week 3 on the calibration window) as its own errors feed back. On the calibration window (the decision window) neither variant approaches the SARIMA champion (week averages of about 81 vs 55 children), although the recursive model would have beaten it on the test window (about 28 vs 32). The production design is unchanged; a recursive care-load learner is a candidate for the next model review.

### 7.4 Statistical significance

| Series | h = 1 | h = 7 | h = 14 | h = 28 |
|---|---|---|---|---|
| Care load (SARIMA) | p = 0.89 | p = 0.052 | **p = 0.045** | **p < 0.001** |
| Discharges (Ensemble) | **p = 0.008** | **p = 0.003** | p = 0.072 | p = 0.29 |
| Transfers (Moving Avg) | **p = 0.035** | p = 0.12 | p = 0.059 | p = 0.58 |

For the care load, persistence is unbeatable at one day (tomorrow ≈ today), but the gains become significant at two and four weeks, the horizons that matter for capacity. For flows the gains are significant at short horizons.

### 7.5 Uncertainty calibration

| 80% coverage on test | Fixed calibration (13 wk) | Rolling origin (26 wk, production rule) | Rolling origin (13 wk) |
|---|---|---|---|
| Care load, SARIMA (mean width) | 91.6% (166) | 90.2% (150) | 81.7% (108) |
| Discharges, Ensemble | 77.2% | 78.7% | 77.9% |
| Transfers, Moving Avg (champion) | 66.0% | 68.4% | 71.1% |
| Transfers, Ensemble | 68.4% | 71.9% | 69.3% |

Care-load intervals are conservative under the fixed window and become almost exactly calibrated, and a third narrower, with a 13-week rolling window. Discharge intervals are close to target. **Intake intervals remain too narrow under every rule** (open defect D-MOD-01). The cause is not higher volatility (transfers were steadier in the test window, SD 4.7 vs 5.0 a day) but a bias shift: most models under-forecast transfers by 1–2 a day during calibration and not during the test, which empirical windows absorb only slowly. Coverage also falls with lead time for the flows (discharge Ensemble 84–91% in week 1 vs 70–76% in week 4; transfers 72–77% vs 61–66%; Figure 13). Planners should use the 95% band for intake until an adaptive method has been validated on a new window.

During high demand the replay's analytic 80% ETS intervals held up: 80.4% coverage at the 153 origins where care load had risen at least 5% in the previous 28 days (84.3% otherwise), with misses mostly *below* the band (14.7%) rather than above it (4.8%), the safe direction for capacity. All 672 production forecasts and bounds are finite and correctly ordered, with no negative point forecast; the zero floor binds for 73 discharge and 116 transfer lower 95% bounds (of 224 each) but rarely at 80% (8 and 12).

![Figure 12. Champion care-load forecasts with calibrated 80% and 95% intervals](figures/12_fan_chart.png)

![Figure 13. Interval coverage by horizon on the test window](figures/12b_coverage_by_horizon.png)

### 7.6 Key performance indicators

| KPI | Care load | Discharges | Transfers |
|---|---|---|---|
| Forecast accuracy % (champion, test) | 98.6 | 64.6 | 58.8 |
| Forecast Stability Index (champion) | 25.2 | 86.5 | 84.3 |
| Win-rate vs naive across test weeks | 85.7% (Ensemble 100%) | 85.7% | 78.6% |
| Error growth week 4 / week 1 | 3.9× (naive 5.6×) | 1.2× | 1.1× |
| 80% interval coverage (fixed / rolling 26 wk) | 91.6% / 90.2% | 77.2% / 78.7% | 66.0% / 68.4% |

The **Forecast Stability Index** = 100 / (1 + mean revision ÷ mean actual daily change) shows a real trade-off. Flow forecasts are far steadier than naive (FSI 74–96 vs 50). Trend-following care-load models revise *more* than the data's own daily movement (23–33 vs 51; the flat baselines are steadiest at 58.5–63.5), because each new report can tilt the whole 28-day path. Planners should therefore read intervals and breach probabilities and judge trends over several days.

### 7.7 Early warning: surge lead time

| Alert at P(breach) ≥ | Replay 2023–24: detected | median lead | false alarms | Test 2025: detected | median lead | false alarms |
|---|---|---|---|---|---|---|
| 50% (point forecast) | 80% | 1 day | 12% | 100% | 18.5 days | 16% |
| 40% | 100% | 6 days | 15% | 100% | 23.5 days | 21% |
| **30% (recommended)** | **100%** | **14 days** | **26%** | **100%** | **28 days** | **22%** |
| 20% | 100% | 27 days | 36% | 100% | 28 days | 24% |
| 10% | 100% | 28 days | 43% | 100% | 28 days | 25% |

![Figure 14. Early-warning operating curves](figures/13a_operating_curve.png)

The 2023–24 replay (25 surge events across thresholds from 6,000 to 11,500) is the real stress test. Alerting on the point forecast warns only about one day ahead. Alerting at a **30% breach probability detects every surge with a median two-week warning** (14.6 days on average), at the cost of roughly one false alarm in four. In the steady 2025 test window, lead times saturate at the 28-day horizon.

![Figure 15. Real-time surge replay, 2023–2024](figures/13_surge_replay.png)

## 8. Software validation

The notebook executes top to bottom in a clean kernel (205 cells, 0 errors, 8–26 minutes depending on machine load). An automated suite, organised into eight pytest categories (data validation, preprocessing and features, model units, forecast and metrics, integration, Streamlit application, end-to-end workflow, performance and reliability), executes the notebook's own code and the real dashboard script, including edge cases on synthetic files (empty, malformed, constant, short, weekly, out-of-range inputs; failing and invalid models; missing dependencies; corrupted artifacts). The Streamlit dashboard was also exercised in a real browser. Pass/fail counts, open defects and evidence are in `reports/Test_Report_v2.md`; this paper does not restate them, to keep a single source of truth.

Version 2.1 adds a fast pytest suite in `tests/` (`python -m pytest`) for the descriptive pipeline layer of section 10:

- data ingestion and validation, including a cell-by-cell reconciliation of the cleaned table with the raw export;
- KPI and aggregation definitions, checked against hand-computed examples and an independent recomputation from the raw file;
- the alert rules, at their threshold boundaries, with a no-look-ahead truncation test;
- the dashboard's controls, empty and invalid ranges, file handling and a real server start-up;
- the consistency of this paper, the executive summary and the README with the generated values.

Results are in `reports/Test_Report_v2.1.md`.

## 9. Outlook and scenarios (from 21 December 2025)

### 9.1 28-day forecast

| Horizon | SARIMA (champion) | 80% interval | 95% interval | Ensemble |
|---|---|---|---|---|
| 7 days (28 Dec) | 2,503 | 2,463 – 2,538 | 2,436 – 2,561 | 2,509 |
| 14 days (4 Jan) | 2,519 | 2,456 – 2,595 | 2,381 – 2,622 | 2,533 |
| 28 days (18 Jan) | **2,542** | **2,486 – 2,694** | 2,354 – 2,731 | 2,571 |

![Figure 16. 28-day outlook for care load, discharges and transfers](figures/14_production_forecast.png)

Expected flows are about 58 transfers and 87–91 discharges per week (ensemble).

The ensemble's direct care-load forecast (2,571) and the flow-implied route (forecast flows + the +3.3/day reconciliation term → 2,455 on day 28) differ by about 115 children by day 28; the gap to SARIMA's 2,542 is 87. Most of it (about 90 children, or roughly 80%) comes from the discharge forecast: the ensemble expects 12.7 discharges a day against a recent average of 9.4. The slightly lower transfer forecast (8.3 vs 8.7 a day) adds about 11–12 more. Placements are therefore the swing factor in the outlook. The flow-implied value sits inside the champion's 95% interval.

### 9.2 Capacity breach probability

| Threshold | Day 7 | Day 14 | Day 21 | Day 28 |
|---|---|---|---|---|
| 2,550 | 5% | 33% | 55% | 76% |
| 2,600 | 0% | 8% | 28% | **44%** |
| 2,650 | 0% | 0% | 6% | **27%** |
| 2,700 | 0% | 0% | 0% | 7% |

The file contains no official capacity figure, so thresholds are illustrative. Using the dashboard's status bands (Watch 10–30%, Elevated 30–60%, Critical ≥ 60%): a stress trigger at 2,600 reaches 44% by day 28, "Elevated" and above the recommended 30% alert trigger. At 2,650 it reaches 27%, "Watch".

![Figure 17. Breach probability over the horizon](figures/15_breach_probability.png)

### 9.3 Scenarios (hypothetical assumptions, not forecasts)

| Scenario (ensemble baseline) | Day 14 | Day 28 | Change vs today | First day > 2,650 |
|---|---|---|---|---|
| Baseline (current trends, model forecast) | 2,533 | 2,571 | +87 | n/a |
| Discharge slowdown (−30%) | 2,586 | 2,677 | +193 | day 24 |
| Placement acceleration (+50% discharges) | 2,444 | 2,393 | **−91** | n/a |
| Intake surge (3× transfers from day 8) | 2,650 | 2,917 | +433 | day 15 |
| Combined stress (2× in, −25% out) | 2,695 | 2,892 | +408 | day 12 |
| Return to 2024 intake (~205/day from day 8) | 3,913 | 6,710 | **+4,226** | day 8 |

![Figure 18. Scenario comparison](figures/16_scenarios.png)

These paths show what the stock-flow identity implies *if* each assumption held; they are not predictions. They indicate that the system is far more sensitive to intake shocks than its current size suggests, and that a 50% increase in placements would reverse the current growth within a month.

## 10. Pipeline analytics: transfers, discharges and backlog

The forecasts answer *what comes next*. This descriptive layer, added in version 2.1, answers *where the CBP → HHS pipeline is under pressure now, and since when*. Its functions live in `src/` and power the dashboard tab "Pipeline analytics & data explorer". Every number in this section is generated by `python tools/pipeline_analysis.py` (`reports/pipeline_analysis.md` and its CSV files), and `tests/test_output_consistency.py` checks the values quoted here against that output.

### 10.1 Design: reported days, stocks and flows

- **Reported days only.** All 715 genuine reports are used (720 minus the 5 stale ones), and no interpolated Friday or Saturday value enters a KPI. Values therefore differ slightly from section 5.1, which averages the interpolated calendar grid. The largest difference is in discharges before the break: 194.2 a day on reported days vs 203.2 on the grid, because the grid weights interpolated days like real ones.
- **Stocks are never summed.** Children in CBP custody and in HHS care are point-in-time counts. A period is described by its first, last, mean and peak report, and its change is last minus first.
- **Flows are summed over reported days.** Apprehensions, transfers and discharges are daily counts. A "total" is therefore a reported-day total, and per-day values are means over reported days.
- **Ratios are ratios of sums.** Each ratio sums its two inputs over the days where both are observed. It is undefined (shown as n/a), never infinite, when the denominator is zero. CBP values from the copied block (12 Jan 2023 – 30 Jan 2024) stay masked, so CBP KPIs are n/a for 2023.

KPI definitions are in Appendix B.

### 10.2 Results by period

| KPI (reported days) | 2023 (CBP masked) | Before the break (31 Jan 2024 – 25 Jan 2025) | Current regime (17 Mar – 21 Dec 2025) | Last 90 days (23 Sep – 21 Dec 2025) |
|---|---|---|---|---|
| Reported days | 250 | 245 | 185 | 61 |
| HHS census: first → last (change) | 6,566 → 8,182 (+1,616) | 8,153 → 4,621 (−3,532) | 2,111 → 2,484 (+373) | 2,145 → 2,484 (+339) |
| Peak HHS census (date) | 11,516 (20 Dec 2023) | 8,902 (7 Mar 2024) | 2,528 (1 Jun 2025) | 2,484 (21 Dec 2025) |
| Children in CBP custody (mean) | n/a | 274.4 | 28.1 | 34.9 |
| Apprehensions per reported day | n/a | 147.1 | 7.65 | 7.54 |
| Transfers into HHS per reported day | n/a | 208.3 | 9.49 | 10.48 |
| Discharges per reported day | 288.4 | 194.2 | 11.71 | 10.23 |
| HHS discharge rate (% of census per day) | 3.33 | 2.88 | 0.52 | 0.44 |
| Implied average stay in HHS care (days) | 30.0 | 34.8 | 192.3 | 225.3 |
| Implied CBP holding time (days) | n/a | 1.32 | 2.96 | 3.33 |
| CBP transfer rate (% of custody per day) | n/a | 75.9 | 33.8 | 30.0 |
| Recorded net intake per reported day | n/a | +14.0 | −2.2 | +0.25 |
| Recorded throughput (discharges ÷ transfers) | n/a | 0.93 | 1.23 | 0.98 |
| CBP pass-through (transfers ÷ apprehensions) | n/a | 1.42 | 1.24 | 1.39 |

**Findings verified on the data:**

1. **Placement throughput fell far more than the census.** The HHS discharge rate fell from 2.88% of the census per day before the break to 0.52% in the current regime, and to 0.44% in the last 90 days. The implied stay therefore rose from 34.8 to 192.3 and then 225.3 days. This is the same bottleneck shift as section 5.2, measured with a period-level definition; Appendix B reconciles the two.
2. **The census grew 17.7% across the current regime** (2,111 → 2,484, +373), the same +373 that the weekend-ratchet analysis decomposes in section 5.4.
3. **Recorded flows misstate the backlog.** Before the break, recorded net intake was +14.0 children per reported day, yet the census fell by 3,532. In the current regime it was −2.2 per day while the census rose by 373. Backlog detection therefore uses the census (a stock), and recorded throughput is reported for transparency only.
4. **Transfers out of CBP slowed relative to custody.** CBP custody averaged 274 children before the break and 28 in the current regime. The share of the CBP population transferred per day fell from 75.9% to 33.8%, so the implied holding time rose from 1.32 to 2.96 days, and to 3.33 days in the last 90 days.
5. **The apprehension column does not cover all CBP intake.** Transfers exceed recorded apprehensions in every period (1.24–1.42 per apprehension). The source header marks this column with an asterisk whose footnote is not in the file. No alert rule uses it.

### 10.3 Bottleneck detection rules

Each rule is evaluated on every reported day using trailing windows only, so a historical flag is exactly what an analyst would have seen on that day. A test verifies that truncating the data never changes earlier flags. A day's overall status is its worst indicator; an indicator without a complete window is "insufficient data" and is ignored. All thresholds can be changed in the dashboard.

| Indicator | Signal (trailing windows) | Watch | Alert | Rationale |
|---|---|---|---|---|
| HHS census growth (backlog building) | % change in the census vs the last report at least 28 days earlier (that report must be within 7 days of the 28-day mark) | ≥ +5% | ≥ +10% | The census is the only balance-consistent measure of backlog |
| Discharge slowdown (placement bottleneck) | 7-day discharge rate vs the median 7-day rate over the 8 weeks before that window | ≥ 15% below | ≥ 30% below | Placements are the binding constraint (section 5.2) |
| CBP holding time (transfer bottleneck) | Σ custody ÷ Σ transfers over 7 days (no transfers while children are held = unbounded) | ≥ 2 days | ≥ 3 days | 3 days mirrors the 72-hour transfer standard for unaccompanied children (TVPRA 2008, 8 U.S.C. § 1232(b)(3)) |

Recorded throughput was tested as a backlog rule and rejected (finding 3). The census and discharge thresholds are operating choices, not program policy.

### 10.4 Periods of potential operational pressure

| Indicator | Before the break: % of reports at Watch or Alert / at Alert | Current regime | Last 90 days | Largest episodes reaching Alert (consecutive Watch-or-Alert reports) |
|---|---|---|---|---|
| Census growth | 21.6 / 6.5 | 42.2 / 0.0 (max +8.7%) | 63.9 / 0.0 | 10 Jul – 28 Sep 2023 (55 reports, peak +37.6%); 23 Nov – 28 Dec 2023 (+17.5%); 9 – 30 Dec 2024 (+16.4%) |
| Discharge slowdown | 10.6 / 0.4 | 42.6 / 22.4 | 44.3 / 21.3 | 17 Feb – 20 Apr 2025 (45 reports, peak 87.4% below baseline: the discharge break); 13 Nov – 4 Dec 2025 (peak 46.8%) |
| CBP holding time | 0.0 / 0.0 (max 1.85 days) | 95.6 / 43.7 | 100.0 / 60.7 | 2 Sep – 21 Dec 2025 (76 reports, peak 6.57 days); 22 Jun – 28 Aug 2025 (peak 5.38 days) |

![Figure 19. Pipeline monitoring indicators, February 2024 – December 2025 (shading: Watch / Alert days)](figures/19_pipeline_monitoring.png)

- **The 2023 surges are the clearest backlog episodes.** The census rose up to 37.6% in four weeks during July–September 2023. Alert episodes also cover the other two section 5.5 surge dates (9 Feb and 20 Dec 2023).
- **The current regime shows slow, sustained growth rather than surges.** Census growth reached Watch on 42% of current-regime reports and on 64% of the last 90 days, but never Alert (maximum +8.7% in four weeks).
- **The placement bottleneck is visible as it formed.** The discharge-slowdown rule fired from 17 Feb 2025, the date the change-point analysis dates the discharge break, and short slowdowns recurred from August to December 2025.
- **A possible transfer bottleneck.** The implied CBP holding time never reached 2 days before the break. In the current regime (since 17 March 2025) it was at or above 2 days on 96% of reports and at or above 3 days on 44%; in the last 90 days these shares are 100% and 61%.
- **Status at the last report (21 Dec 2025): Alert.** CBP holding time was 3.75 days (Alert), census growth +4.6% (Normal, just below Watch), and the discharge rate was 19% above its baseline (Normal).

### 10.5 Interpretation and caveats

The census-growth and discharge indicators rest on the HHS columns, which reconcile wherever they can be checked (section 4.4). The CBP holding time is **indicative**:

- Little's law approximates an *average* holding time under a roughly steady state. It does not measure any child's custody time and is not evidence of non-compliance with the 72-hour standard.
- At about 10 transfers a day, a few long stays move the ratio noticeably.
- If children leave CBP custody by routes other than transfer to HHS, the ratio overstates holding time.
- The time of day of the custody snapshot is not documented.

The signal should be confirmed against CBP book-in and book-out durations before any operational conclusion. None of these indicators identifies a cause. Policy, border and processing changes are not in the data.

## 11. Limitations

- **Short regime.** The current regime has about nine months of data. Every model will need re-validation after the next policy change.
- **Incomplete balance and no capacity data.** Recorded flows leave about 3 children a day unexplained, and the file has no other-admission, other-exit or capacity fields. The reconciliation term is an empirical average, and breach probabilities are computed against user-entered thresholds.
- **Intake intervals.** Transfer 80% intervals under-cover (66–71% depending on the calibration rule; D-MOD-01 open). Flow lower bounds are floored at zero, a truncation rather than a count model.
- **Test window diversity.** September–December 2025 was a steady rise. The 2023–24 replay supplies the surge stress test, but with a univariate model because CBP data are corrupted in that period.
- **Small samples per horizon.** Each single-horizon test metric rests on 26–55 forecasts (42–55 at the scorecard horizons of 1, 7, 14 and 28 days).
- **Descriptive pipeline indicators (section 10).** Implied holding times and stays are Little's-law ratios of stocks to flows. They approximate averages under a roughly steady state; they do not measure individual stays, and at about 10 transfers a day a few long CBP stays move the ratio noticeably. The recorded apprehension column does not cover all CBP intake (transfers exceed it in every period). Alert thresholds are operating defaults, not program policy, and the trigger frequencies reported depend on them.
- **Retrospective data cleaning.** Copy detection needs the later original; it was shown to be stable at the first forecast origin, but a real-time system would detect copies late.
- **Interpolated weekend flows** reconcile with the stock in the current regime (within about 0.45/day) but were biased in 2024; the weekend ratchet is inferred from stock changes.
- **ML extrapolation.** Tree models cannot forecast beyond their training ranges; large shocks are handled by the statistical models and the scenario engine.
- **Unobserved drivers.** Border events, policy announcements and sponsor-vetting rules are not in the data. Forecasts describe the system *as it currently behaves*. Correlations reported here are associations, not causal effects.
- **Horizon.** 28 days was validated; 30 days was not.

## 12. Recommendations

1. **Treat discharge capacity as the primary lever.** Expand sponsor-vetting and case-management throughput; in the stock-flow scenario, +50% placements turns growth into decline within a month.
2. **Stop the weekend ratchet.** Add weekend placement and discharge processing. Weekend gaps are where most of the current regime's net growth accumulated.
3. **Adopt a probability-based scaling trigger.** Begin scaling beds and staff when the 28-day breach probability for the real capacity reaches 30% (dashboard status "Elevated"). In the 2023–24 replay this gave a median two-week warning for gradual build-ups.
4. **Keep surge contingency warm.** Warnings averaged two weeks for gradual build-ups but were as short as one day for an abrupt jump (25 April 2023), so standby capacity should be activatable within days, and the early-warning system should run on every new report.
5. **Fix the data at the source.** Publish Friday/Saturday flows (or weekend totals), non-CBP admissions, other exits and bed capacity; keep the ingestion validation built here in front of every refresh; correct the 2023 CBP records.
6. **Govern the models.** Re-estimate every two weeks; monitor interval coverage by horizon (use the 95% band for intake); check the model-fit log after every run; change champions only through the calibration procedure, never on test results; review every calendar feature for regime leakage.
7. **Next model review.** Evaluate a recursive care-load learner and adaptive conformal intervals for intake on a *new* evaluation window with the selection rule fixed in advance.
8. **Verify, then monitor, the transfer pipeline.** The implied CBP holding time has been at or above 2 days on 96% of current-regime reports and at or above 3 days on 44% (section 10.4). Compare it with CBP custody-duration records before acting on it. If it is confirmed, review the CBP-to-HHS referral and placement process. Review the dashboard's pipeline status at every data refresh, alongside the forecast.

## 13. Conclusion

Careful root-cause analysis turned an apparently clean dataset into a trustworthy one. It exposed a year of copied intake records, a structural break, a reporting cadence that would have leaked future values into a careless model, and flows that do not close the population balance. Version 2 adds input validation, an executed leakage audit, origin-time checks of every data-quality decision, a seasonal baseline, ARIMA/SARIMA and recursive/direct comparisons, and a real-time evaluation of the production interval rule. On that foundation, a regime-aware, walk-forward-validated system forecasts care load with an average error of about 1.2–1.4% across the four-week horizon (about 2.5% at four weeks, half of naive persistence), explains why discharges cannot be forecast much more precisely day by day, and converts forecasts into probabilities that support action. A 30% breach-probability trigger would have warned of every 2023–24 surge, with a median of two weeks' notice. Intake uncertainty is still understated and must be read from the 95% band. The descriptive monitoring layer adds a period-by-period view of the CBP → HHS pipeline. It shows discharge rates at less than a fifth of their pre-break level, and an implied CBP holding time that more than doubled and now warrants verification. For HHS planners the message is clear: in the current regime, **placement capacity, including weekend placements, decides the care load**, and probability-based early warning gives enough lead time to act before shelters fill.

---

## References

- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723.
- Ben Taieb, S., Bontempi, G., Atiya, A. F., & Sorjamaa, A. (2012). A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition. *Expert Systems with Applications*, 39(8), 7067–7083.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on Loess. *Journal of Official Statistics*, 6(1), 3–73.
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366), 427–431.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift. *Advances in Neural Information Processing Systems*, 34.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- Huber, P. J. (1964). Robust estimation of a location parameter. *Annals of Mathematical Statistics*, 35(1), 73–101.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*, 54(1–3), 159–178.
- Little, J. D. C. (1961). A proof for the queuing formula L = λW. *Operations Research*, 9(3), 383–387.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference*, 92–96.
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting*, 16(4), 437–450.
- Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*, 167, 107299.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.

---

## Appendix A: Reproducibility

| Item | Location |
|---|---|
| Full executed analysis | `UAC_Care_Load_Forecasting.ipynb` (8–26 minutes in a clean kernel; deterministic seeds; editable source `notebook_source/UAC_Care_Load_Forecasting_v2.py`) |
| Interactive dashboard | `python -m streamlit run uac_forecast_dashboard.py` |
| Clean daily data with masks | `artifacts/daily_clean.csv` |
| Walk-forward forecasts (scored) | `artifacts/backtest.csv.gz` |
| Metrics, coverage, robustness | `artifacts/metrics_overall.csv`, `metrics_by_horizon.csv`, `interval_coverage.csv`, `interval_coverage_by_h.csv`, `interval_method_comparison.csv`, `robustness.csv` |
| Validation and audit | `artifacts/validation_report.csv`, `model_fit_log.csv`, `feature_leakage_audit.csv` |
| Production forecasts with intervals | `artifacts/forecast_future.csv` |
| Early-warning results | `artifacts/surge_lead_times.csv`, `ews_operating_curve.csv`, `surge_replay_paths.csv` |
| Run metadata | `artifacts/metadata.json` |
| Tests and test report | `QA_Testing/pytest_suite/`, `reports/Test_Report_v2.md` |
| Pipeline analytics (section 10) | `src/` (functions), `python tools/pipeline_analysis.py` (regenerates `reports/pipeline_*.csv`, `reports/pipeline_analysis.md` and Figure 19) |
| Pipeline and consistency tests | `python -m pytest` (suite in `tests/`), results in `reports/Test_Report_v2.1.md` |

## Appendix B: KPI definitions

| KPI | Definition |
|---|---|
| Forecast Accuracy (%) | 100 − WAPE, where WAPE = Σ\|error\| / Σ\|actual\| on the test window; undefined when all actuals are zero |
| Surge Lead Time | Days between the start of the unbroken alert streak and the threshold crossing |
| Capacity Breach Probability | P(care load > threshold), from the point forecast plus walk-forward errors for that horizon (last 26 weeks of observable errors); threshold supplied by the user |
| Forecast Stability Index | 100 / (1 + mean revision for the same target date between consecutive origins ÷ mean actual daily change) |
| Model robustness | Win-rate vs naive across test weeks; error growth from week 1 to week 4; interval coverage |

### Descriptive pipeline KPIs (section 10)

All computed on reported days only. Stocks: children in CBP custody, children in HHS care. Flows: apprehensions, transfers, discharges. These definitions are the ones shown in the dashboard tooltips (`src/kpi_calculations.py`, `KPI_DEFINITIONS`).

| KPI | Definition |
|---|---|
| HHS census at period end | Children in HHS care on the last reported day of the period (stock) |
| HHS census change | Last minus first reported HHS census in the period (stock change, not a sum) |
| Peak HHS census | Highest reported HHS census in the period, with its date |
| CBP custody at period end | Children in CBP custody on the last trusted report of the period (stock) |
| Apprehensions per reported day | Mean of daily CBP apprehensions over reported days |
| Transfers into HHS per reported day | Mean of daily transfers out of CBP into HHS over reported days |
| Discharges per reported day | Mean of daily HHS discharges (mostly to sponsors) over reported days |
| Recorded net intake per reported day | Mean of (transfers - discharges) on days with both reported. Recorded flows do not close the census balance (about +3 children/day unexplained), so use the census change for backlog |
| CBP transfer rate (%/day) | 100 x sum(transfers) / sum(children in CBP custody): share of the CBP population moved to HHS per day |
| HHS discharge rate (%/day) | 100 x sum(discharges) / sum(children in HHS care): share of the HHS population discharged per day |
| Recorded throughput (exits per entry) | sum(discharges) / sum(transfers): above 1 = recorded exits exceed entries. Not used for alerts: it misses unreported Fri/Sat flows and non-CBP admissions |
| Implied average stay in HHS care (days) | Little's law, ratio of period means: sum(children in HHS care) / sum(discharges) = 100 / discharge rate |
| Implied CBP holding time (days) | Little's law: sum(children in CBP custody) / sum(transfers). An approximation of the average time in CBP custody before transfer (an upper bound if children also leave CBP by other routes); compare with the 72-hour transfer standard |
| CBP pass-through (transfers per apprehension) | sum(transfers) / sum(apprehensions) on days with both reported |

### Alert rules (section 10.3)

| Indicator | Value | Watch | Alert | Window and data requirement |
|---|---|---|---|---|
| HHS census growth | 100 × (census / census at the last report ≥ 28 days earlier − 1) | ≥ 5% | ≥ 10% | Lagged report must be within 7 days of the 28-day mark |
| Discharge slowdown | 100 × (1 − 7-day discharge rate / median 7-day rate of the previous 8 weeks) | ≥ 15% | ≥ 30% | ≥ 3 reports in the 7-day window; ≥ 15 rates in the baseline |
| CBP holding time | Σ custody / Σ transfers over 7 days (∞ if no transfers while children are held) | ≥ 2 days | ≥ 3 days | ≥ 3 reports with both values; trusted CBP data only |

Overall status = worst indicator level (Normal, Watch, Alert); indicators without a complete window are ignored ("insufficient data" if none is available).

### Implied length of stay: two definitions

Section 5.2 averages a 28-day smoothed daily ratio (calendar grid) over fixed windows. The dashboard and section 10 use the ratio of period means over reported days, so the value equals 100 ÷ the discharge rate for any selected period. On the same windows the two definitions differ by 4–5%:

| Window | Section 5.2 definition | Ratio of reported-day means | Difference |
|---|---|---|---|
| 22 Aug 2024 – 17 Feb 2025 (section 5.2 "before the break") | 35.9 days | 37.2 days | +3.7% |
| 17 Mar – 21 Dec 2025 (current regime) | 203.2 days | 192.3 days | −5.4% |
