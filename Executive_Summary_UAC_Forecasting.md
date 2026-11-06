# Executive Summary: Forecasting UAC Care Load & Placement Demand

**For:** HHS / ORR program leadership and capacity planners

**Data through:** 21 December 2025 · **Forecast horizon:** 28 days (to 18 January 2026)

**Prepared under:** Unified Mentor Data Science Internship, October 2026 (includes pipeline monitoring of CBP custody, transfers, discharges and backlog)

---

## The problem

Shelter beds, case managers and clinicians take weeks to add, but the number of children needing care can change within days. Daily reports say what happened yesterday. This work turns them into a forward view: how many children will be in care over the next four weeks, how many placements and arrivals to expect, and how much warning planners get before a capacity threshold is reached.

## The bottom line

1. **Care load is rising slowly and steadily:** from 1,972 children (21 Aug 2025) to **2,484** (21 Dec 2025). The forecast is **about 2,540 by 18 January 2026** (80% range 2,486–2,694).
2. **Placements, not arrivals, now drive the numbers.** Since the structural break in early 2025, intake fell about 95%, but the implied average stay in care rose from about **36 days to about 203 days**. How fast children are released to sponsors now decides how many remain in care.
3. **Net growth has come mostly across weekends.** Since March 2025, weekday changes netted −40 children, while weekends added +323 (other reporting gaps a further +90). This is inferred from changes in the census, because Friday and Saturday flows are almost never reported. It suggests that arrivals continue at weekends while placements slow.
4. **We can warn about two weeks ahead of gradual build-ups.** In a replay of 2023–24, using a simpler model than the one now in use, the early-warning rule flagged **all 25 surge events**, with a median **14 days' notice**, at the cost of roughly one false alarm in four. Events at nearby thresholds can overlap, so they are not 25 independent surges.
5. **Children may now be waiting longer in CBP custody before transfer, which needs checking.** Comparing the number in CBP custody with daily transfers suggests an average wait of about **3 days** since March 2025, against about **1.3 days** in 2024. This is an indirect estimate, not a measured custody time, and it is not evidence of non-compliance with the 72-hour transfer standard. It should be confirmed with CBP custody-time records before any action.

## What planners should expect (next 4 weeks)

| Measure | Forecast | Planning range |
|---|---|---|
| Children in HHS care on 18 Jan 2026 | **2,542** | 2,486 – 2,694 (80%) |
| Discharges (sponsor placements) | **~87–91 per week** | daily counts vary widely; plan weekly |
| Transfers in from CBP | **~58 per week** | intake is the least predictable flow; **use the 95% range** |
| Chance care load exceeds 2,600 by 18 Jan | **44%** | status: Elevated |
| Chance care load exceeds 2,650 by 18 Jan | **27%** | status: Watch |

*There is no official capacity figure in the data; 2,600 and 2,650 are illustrative. The dashboard lets planners enter actual funded capacity. Status bands: Watch 10–30%, Elevated 30–60%, Critical ≥ 60% probability of exceeding the threshold. The care-load forecast is from SARIMA, the model chosen for care load; the weekly flows are from the ensemble, which combines four models.*

## Key historical patterns

- **The data needed repair before use.** One year of CBP intake records (January 2023 – January 2024, 251 reports) copies later dates; Saturday is never reported and Friday almost never; five reports are stale repeats. All were detected and handled before any model was built. The CBP "apprehended" count is also smaller than the number transferred in every period, so it does not capture all children entering CBP custody.
- **A regime change in early 2025.** Transfers fell from about 210 to 10 a day and discharges from about 200 to 12, while the census fell only two-thirds, so implied stays are far longer.
- **The published flows do not fully add up.** Even on ordinary weekdays the census grows about 3 children a day more than recorded transfers minus discharges. The data contain no figures for other admissions, other exits or capacity. Before the break, recorded arrivals exceeded recorded exits by 14 a day, yet the census fell by over 3,500. The census itself is therefore the reliable measure of backlog.

## Pipeline trends: CBP custody, HHS care, transfers and discharges

*Averages over reported days. "Before the break" = 31 Jan 2024 – 25 Jan 2025; "current" = 17 Mar – 21 Dec 2025.*

| Measure | Before the break | Current regime | Last 90 days |
|---|---|---|---|
| Children in CBP custody (average) | 274 | 28 | 35 |
| Transfers from CBP into HHS per day | 208 | 9.5 | 10.5 |
| Children in HHS care (start → end of period) | 8,153 → 4,621 | 2,111 → 2,484 | 2,145 → 2,484 |
| Discharges (sponsor placements) per day | 194 | 11.7 | 10.2 |
| Share of children in care placed each day | 2.9% | 0.5% | 0.4% |
| Implied average wait in CBP custody before transfer | 1.3 days | 3.0 days | 3.3 days |

- **HHS care:** after the early-2025 fall, the census has grown about 18% since March 2025, slowly rather than in surges.
- **Placements:** the share of children placed each day is now less than a fifth of its 2024 level. This is why implied stays are long and why placement capacity matters most.
- **CBP custody:** far fewer children are held, but the implied wait before transfer has more than doubled (see "Periods of operational pressure").

## Periods of operational pressure

The dashboard checks three warning signs on every report, using only information available on that day:

- **Census growth** above 5% (watch) or 10% (alert) in four weeks;
- **Slower placements:** a discharge rate 15% or 30% below its recent norm;
- **Longer CBP waits:** an implied wait of 2 days or more (watch), or 3 days, which mirrors the 72-hour transfer standard (alert).

| Period | What the warning signs showed |
|---|---|
| July – September 2023 | Census rose up to 38% in four weeks: the largest backlog build-up in the data |
| November – December 2023 and December 2024 | Further census surges of about 16–17% in four weeks |
| 17 February – 20 April 2025 | Placements fell up to 87% below their recent norm: the structural break in placements |
| March – December 2025 | Implied CBP wait at 2 days or more on 96% of reports, and 3 days or more on 44% (61% in the last 90 days) |
| August – December 2025 | Repeated short placement slowdowns (up to 47% below normal); census growth at watch level on 64% of reports in the last 90 days |

**Status at the last report (21 Dec 2025): Alert**, driven by the implied CBP wait (3.75 days). Census growth (+4.6% in four weeks) and placements were within normal range.

*These thresholds are operating defaults that planners can change in the dashboard; they are not program policy. None of these warning signs identifies a cause.*

## How reliable are these forecasts?

Every model was tested the way it would be used: forecasting forward from each past day, scored on a final 13-week window. The chosen models, the rule that sets the ranges and the alert level were all fixed without using that window's results. The window was a steady rise, so the evidence on surges comes from the 2023–24 replay. Eight approaches were compared, from simple rules ("no change", "same as last week") to statistical and machine-learning models.

| Series | Accuracy (test) | Typical error | vs "no change" baseline |
|---|---|---|---|
| Children in care | **98.6%** | ±32 children | **43% smaller error** (49% for the ensemble) |
| Daily discharges | 64.6% | ±3.5 per day | 25% smaller (close to the natural day-to-day randomness) |
| Daily transfers | 58.8% | ±4.2 per day | 17% smaller |

At four weeks ahead, the care-load error is **half** that of assuming no change. *Accuracy here means 100% minus the total forecast error as a share of the total actual volume.*

**Uncertainty ranges:** for care load they are reliable to slightly cautious (about 90% of outcomes fell inside the 80% range), which is the safe side for capacity. For **daily transfers (intake) they are too narrow**: for the chosen model, only about two-thirds of outcomes fell inside the 80% range (66–71%, depending on the calibration rule), even after recalibrating on the most recent data. Ranges for the flows also get less reliable further ahead: at four weeks, the 80% range covered about 71% of discharges and 52% of transfers. Until this is fixed, plan intake on the 95% range.

## Expected operational benefits

- **About two weeks' notice** of gradual surges in the 2023–24 replay, giving time to start activating standby beds and staff.
- **Planning in ranges and probabilities** instead of single numbers, tied to the program's own capacity figure.
- **What-if testing** of placement, intake and staffing decisions before committing resources.
- **Automatic data checks** on every refresh: a broken or implausible data file is stopped or flagged instead of silently changing the forecast.

## What-if scenarios (28 days)

*These show what would follow if each assumption held. They are not predictions. The baseline is the ensemble forecast (2,571), which is slightly above the SARIMA forecast quoted above (2,542).*

| If… | Care load on day 28 | Crosses 2,650 on |
|---|---|---|
| Current trends continue (ensemble forecast) | 2,571 | n/a |
| Placements slow by 30% | 2,677 | day 24 |
| Transfers triple from next week | 2,917 | day 15 |
| Transfers double and placements fall 25% | 2,892 | day 12 |
| Intake returns to 2024 levels (~205/day) | **6,710** | **day 8** |
| **Placements increase by 50%** | **2,393 (falls by ~90)** | n/a |

Intake shocks move the numbers fastest: a return to 2024-level intake would add over 4,000 children in three weeks, more than doubling today's census. Faster placements are the only scenario shown that brings the census down.

## Potential capacity risks

- A **sudden intake jump** can give as little as one day's warning (as on 25 April 2023), far less than the typical two weeks.
- **Placements are the swing factor.** The weekly forecast above implies about 12.7 discharges a day, against a recent average of 9.4. This explains most of the roughly 115-child gap between the ensemble's care-load forecast (2,571) and a route built from its forecast flows (2,455). A 30% slowdown in placements would cross 2,650 in about three and a half weeks.
- **Intake uncertainty is understated** by the 80% range (see above).

## Recommendations

1. **Invest in placement throughput.** Sponsor vetting and case management are the main lever; in the scenario model, +50% placements reverses the current growth within a month.
2. **Add weekend placement processing** to address the weekend build-up, where the census data indicate the current regime's net growth occurs.
3. **Use a 30% breach probability as the trigger to scale up.** When the dashboard shows "Elevated" for your real capacity, start activating standby beds and surge staff. In the 2023–24 replay this gave about two weeks' notice for gradual build-ups.
4. **Keep surge contingency ready.** Warnings can be as short as one day when intake jumps abruptly, so standby beds should be activatable within days. Run the early-warning system on every new report.
5. **Fix the data feed.** Correct the duplicated 2023 CBP records at source; publish Friday/Saturday flows, other admissions, other exits and funded capacity; keep the automatic validation checks (already built) running on every new report.
6. **Check the CBP-to-HHS transfer pipeline.** Compare the implied CBP wait (about 3 days since March 2025) with CBP's own custody-time records. If it is confirmed, review referral and shelter-placement processing. Review the dashboard's pipeline status with every refresh, next to the forecast.

## Conditions before operational deployment

The system is ready for **decision support by analysts**, not yet for unattended operational use. Before relying on it operationally:

1. Enter the program's **real funded capacity**; all thresholds in this summary are illustrative.
2. Fix or replace the **intake uncertainty ranges** (currently too narrow) and re-check them on a new period.
3. Put the dashboard **behind organisational sign-in** with standard web security headers; today it runs on the analyst's own computer only.
4. Agree an **owner and schedule**: refresh and re-check the models every two weeks and after any policy change, and review the model-failure and data-validation logs after each run.
5. Re-validate on a period that includes a **surge under the current regime** when one occurs.

## How to use the tools

- **Dashboard** (`uac_forecast_dashboard.py`): choose any horizon from 1 to 28 days, switch models on and off, show 80% or 95% ranges, enter real capacity to get a live breach probability and status, and compare scenarios side by side (each clearly marked as forecast or hypothetical), including a custom scenario builder. The **Pipeline analytics** tab recalculates CBP custody, HHS care, transfer and discharge measures and the three warning signs for any date range, with adjustable thresholds, a data explorer and CSV downloads.
- **Full analysis, research paper, Appendix volume and test report:** `UAC_Care_Load_Forecasting.ipynb`, `Research_Paper_UAC_Forecasting.md`, `Research_Paper_Appendix.md`, `reports/Test_Report_v2.md`.

*Caveats: the current regime has only about nine months of history; external events (border, policy, vetting rules) are not in the data, so forecasts assume the system keeps behaving as it does now and should be re-validated after any major policy change. Implied stays and waits are ratios of census to flows: they approximate averages and do not measure any child's actual stay.*
