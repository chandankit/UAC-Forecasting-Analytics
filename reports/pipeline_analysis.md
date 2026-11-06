# Pipeline analytics: generated values (do not edit; rerun `python tools/pipeline_analysis.py`)

Data through 21 Dec 2025; 715 reported days; default thresholds Thresholds(census_growth_watch_pct=5.0, census_growth_alert_pct=10.0, growth_window_days=28, discharge_drop_watch_pct=15.0, discharge_drop_alert_pct=30.0, rate_window_days=7, baseline_days=56, cbp_hold_watch_days=2.0, cbp_hold_alert_days=3.0, min_reports=3, min_baseline_reports=15).

## KPIs by period (reported days only)

| KPI | 2023 (CBP masked) (12 Jan 2023 - 30 Jan 2024) | Before the break (31 Jan 2024 - 25 Jan 2025) | Current regime (17 Mar 2025 - 21 Dec 2025) | Last 90 days (23 Sep 2025 - 21 Dec 2025) |
|---|---|---|---|---|
| reported_days | 250 | 245 | 185 | 61 |
| hhs_care_start | 6,566.00 | 8,153.00 | 2,111.00 | 2,145.00 |
| HHS census at period end | 8,182.00 | 4,621.00 | 2,484.00 | 2,484.00 |
| HHS census change | 1,616.00 | -3,532.00 | 373.00 | 339.00 |
| hhs_care_change_pct | 24.61 | -43.32 | 17.67 | 15.80 |
| hhs_care_mean | 8,661.99 | 6,754.76 | 2,251.33 | 2,304.77 |
| Peak HHS census | 11,516.00 | 8,902.00 | 2,528.00 | 2,484.00 |
| hhs_care_peak_date | 20 Dec 2023 | 07 Mar 2024 | 01 Jun 2025 | 21 Dec 2025 |
| cbp_custody_mean | n/a | 274.44 | 28.06 | 34.92 |
| CBP custody at period end | n/a | 75.00 | 18.00 | 18.00 |
| Apprehensions per reported day | n/a | 147.10 | 7.65 | 7.54 |
| Transfers into HHS per reported day | n/a | 208.26 | 9.49 | 10.48 |
| Discharges per reported day | 288.43 | 194.24 | 11.71 | 10.23 |
| Recorded net intake per reported day | n/a | 14.01 | -2.22 | 0.25 |
| HHS discharge rate (%/day) | 3.33 | 2.88 | 0.52 | 0.44 |
| Implied average stay in HHS care (days) | 30.03 | 34.77 | 192.29 | 225.31 |
| Implied CBP holding time (days) | n/a | 1.32 | 2.96 | 3.33 |
| CBP transfer rate (%/day) | n/a | 75.88 | 33.82 | 30.00 |
| Recorded throughput (exits per entry) | n/a | 0.93 | 1.23 | 0.98 |
| CBP pass-through (transfers per apprehension) | n/a | 1.42 | 1.24 | 1.39 |

## Indicators by period (reports with a complete window)

| Period | Indicator | Reports evaluated | % Watch or Alert | % Alert | Median value | Max value | Unit |
|---|---|---|---|---|---|---|---|
| Before the break | HHS census growth (backlog building) | 245 | 21.63 | 6.53 | -0.86 | 16.44 | % vs 28 days earlier |
| Before the break | Discharge slowdown (placement bottleneck) | 245 | 10.61 | 0.41 | 2.21 | 33.62 | % below 8-week median rate |
| Before the break | CBP holding time (transfer bottleneck) | 243 | 0.00 | 0.00 | 1.31 | 1.85 | days (custody / transfers, 7-day) |
| Current regime | HHS census growth (backlog building) | 185 | 42.16 | 0.00 | 4.46 | 8.68 | % vs 28 days earlier |
| Current regime | Discharge slowdown (placement bottleneck) | 183 | 42.62 | 22.40 | 10.01 | 87.41 | % below 8-week median rate |
| Current regime | CBP holding time (transfer bottleneck) | 183 | 95.63 | 43.72 | 2.88 | 6.57 | days (custody / transfers, 7-day) |
| Last 90 days | HHS census growth (backlog building) | 61 | 63.93 | 0.00 | 5.88 | 8.25 | % vs 28 days earlier |
| Last 90 days | Discharge slowdown (placement bottleneck) | 61 | 44.26 | 21.31 | 13.28 | 46.79 | % below 8-week median rate |
| Last 90 days | CBP holding time (transfer bottleneck) | 61 | 100.00 | 60.66 | 3.19 | 6.57 | days (custody / transfers, 7-day) |

## Share of reported days by overall status (%)

| Year | Normal | Watch | Alert | Insufficient data | HHS census growth (backlog building) at Watch or Alert | Discharge slowdown (placement bottleneck) at Watch or Alert | CBP holding time (transfer bottleneck) at Watch or Alert |
|---|---|---|---|---|---|---|---|
| 2023 | 45.7 | 14.3 | 33.9 | 6.1 | 50.0 | 5.3 | n/a |
| 2024 | 70.6 | 22.6 | 6.9 | 0.0 | 21.4 | 10.5 | 0.0 |
| 2025 | 16.0 | 30.4 | 53.6 | 0.0 | 32.9 | 41.7 | 79.6 |

## Alert-level episodes (full history)

| # | Indicator | Start | End | Reported days | Calendar days | Peak value |
|---|---|---|---|---|---|---|
| 1 | CBP holding time (transfer bottleneck) | 02 Mar 2025 | 24 Mar 2025 | 17 | 23 | 3.80 |
| 2 | CBP holding time (transfer bottleneck) | 31 Mar 2025 | 08 May 2025 | 30 | 39 | 3.11 |
| 3 | CBP holding time (transfer bottleneck) | 15 May 2025 | 08 Jun 2025 | 14 | 25 | 3.30 |
| 4 | CBP holding time (transfer bottleneck) | 22 Jun 2025 | 28 Aug 2025 | 48 | 68 | 5.38 |
| 5 | CBP holding time (transfer bottleneck) | 02 Sep 2025 | 21 Dec 2025 | 76 | 111 | 6.57 |
| 6 | Discharge slowdown (placement bottleneck) | 27 Aug 2024 | 10 Sep 2024 | 10 | 15 | 33.62 |
| 7 | Discharge slowdown (placement bottleneck) | 17 Feb 2025 | 20 Apr 2025 | 45 | 63 | 87.41 |
| 8 | Discharge slowdown (placement bottleneck) | 17 Aug 2025 | 25 Aug 2025 | 7 | 9 | 32.78 |
| 9 | Discharge slowdown (placement bottleneck) | 03 Sep 2025 | 09 Sep 2025 | 5 | 7 | 36.41 |
| 10 | Discharge slowdown (placement bottleneck) | 15 Oct 2025 | 23 Oct 2025 | 7 | 9 | 31.24 |
| 11 | Discharge slowdown (placement bottleneck) | 13 Nov 2025 | 04 Dec 2025 | 15 | 22 | 46.79 |
| 12 | HHS census growth (backlog building) | 09 Feb 2023 | 22 Feb 2023 | 9 | 14 | 20.44 |
| 13 | HHS census growth (backlog building) | 01 May 2023 | 18 May 2023 | 14 | 18 | 16.71 |
| 14 | HHS census growth (backlog building) | 10 Jul 2023 | 28 Sep 2023 | 55 | 81 | 37.61 |
| 15 | HHS census growth (backlog building) | 23 Nov 2023 | 28 Dec 2023 | 25 | 36 | 17.46 |
| 16 | HHS census growth (backlog building) | 27 Feb 2024 | 12 Mar 2024 | 11 | 15 | 10.01 |
| 17 | HHS census growth (backlog building) | 05 May 2024 | 21 May 2024 | 13 | 17 | 12.88 |
| 18 | HHS census growth (backlog building) | 09 Dec 2024 | 30 Dec 2024 | 16 | 22 | 16.44 |

## Implied length of stay: definition reconciliation

| Window | Notebook definition (days) | Dashboard definition (days) | Difference % |
|---|---|---|---|
| Paper window before the break (last 180 days to the discharge break): 22 Aug 2024 - 17 Feb 2025 | 35.9 | 37.2 | 3.7 |
| Current regime: 17 Mar 2025 - 21 Dec 2025 | 203.2 | 192.3 | -5.4 |

## Latest report (21 Dec 2025)

Overall: Alert; census growth +4.6% (Normal); discharge rate -19.0% vs baseline (Normal); CBP holding time 3.75 days (Alert).
