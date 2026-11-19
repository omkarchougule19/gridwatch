# Trends in the data and the news behind them

The TA asked us (2026-11-18) to stick with the well-tuned models, without adding regions, and to look for interesting trends and the real-world stories behind them. This file collects them.
- **Figures:** `docs/visuals/10_spikes_and_news.png` and `docs/visuals/11_load_vs_spike_hours_by_year.png`.
- **Numbers:** computed from `data/processed/*.parquet` with the p99 spike labels.
- **Model checks:** use the MISO pick, logreg-no_rel.

## 1. ERCOT: demand keeps rising, yet spikes collapsed after 2023
| Year | Average load (GW) | Spike hours (≥ $329) | Mean RT price ($/MWh) |
|---|---|---|---|
| 2022 | 49.2 | 100 | 62 |
| 2023 | 51.0 | 132 | 48 |
| 2024 | 52.8 | 24 | 27 |
| 2025 | 55.7 | 8 | 32 |
| 2026 (Jan–Oct) | 59.7 | 25 | 32 |

- **The puzzle:** load grew about 17% from 2023 to 2026, but spike hours fell by more than 80%. Normally more demand means more stress.
- **The story: a battery boom.**
  - ERCOT's grid-scale battery fleet nearly doubled in 2025 alone, from 7.8 GW to 13.9 GW (Modo Energy), after already growing fast in 2024 (Yes Energy: it "nearly tripled" 2023–2025). Batteries charge on cheap midday solar and discharge in the evening ramp, exactly the hours that used to spike.
  - ERCOT's own market monitor (Potomac Economics) found that summer 2025 price spikes were less frequent than in 2024, crediting the added storage with better ramp management.
  - Ascend Analytics: scarcity revenue (prices above $500/MWh) was over $180/kW-year in 2023 and fell to single digits by 2025. Batteries supplied about 1% of 2025 energy but set the price 23% of the time.
  - IEEFA: despite record summer demand, solar plus storage kept ERCOT prices low.
- **Weather contributed too.** 2025 lacked the extreme heat of 2022–2023 (Modo Energy).
- **What 2022–2023 looked like:**
  - **July 13, 2022:** ERCOT asked Texans to conserve power during record demand (about 78 GW), with low wind and thermal plant outages. This is our largest 2022 episode: $5,087/MWh.
  - **August–September 2023:** a long heat wave. On **Sept 6, 2023**, ERCOT entered **EEA2**, its first emergency since Winter Storm Uri, when solar faded at sunset, wind was low and reserves dropped to about 2.1 GW. Several of our top episodes come from August–September 2023, at $2,000–5,000/MWh.
- **Why this matters for the model:** ERCOT's training data (2021–mid-2024) comes from a "scarcity era" that no longer exists. That is concept drift with a known physical cause, and it's why ERCOT has only 11 test events.

## 2. MISO: spikes have grown every year
| Year | Average load (GW) | Spike hours (≥ $111) | Hours where the DA price was already ≥ $111 | Mean RT price |
|---|---|---|---|---|
| 2023 | 73.2 | 83 | 16 | 28 |
| 2024 | 73.6 | 102 | 74 | 27 |
| 2025 | 75.8 | 233 | 150 | 37 |
| 2026 (Jan–Oct) | 78.4 | 365 | 404 | 44 |

- **The story: a tightening supply margin.**
  - MISO's summer surplus capacity shrank from about 6.5 GW in 2023 to 4.6 GW in 2024 and 2.6 GW in 2025, as coal and gas units retired (3.3 GW slated for 2025/26) while demand forecasts rose with data centers and electrification (Voltus, Enel).
  - The 2025/26 capacity auction cleared at **$666.50/MW-day for summer, up from $30** the year before.
  - Gas prices roughly doubled from 2024's historic lows (Potomac Economics, MISO's market monitor), which raises every hour's price, pushing more hours over a fixed $111 line.
- **The named events behind the biggest episodes:**
  - **Jan 21, 2025, arctic blast:** 21 spike hours, temperatures down to −17 °F in the forecast.
  - **June 23–24, 2025, early heat wave:** MISO declared a Max Gen Event and then a Max Gen Warning, with load near 120 GW. RT prices reached $1,114/MWh.
  - **Jan 23–27, 2026, Winter Storm Fern:** our single largest episode (69 spike hours, up to $1,956/MWh at the Illinois hub).
    - MISO declared **Max Gen emergencies** on Jan 24 as temperatures fell to double digits below zero.
    - Midwest gas reached more than $66/MMBtu as gas liquidity dried up (Potomac Economics).
    - MISO's systems failed to post ex-post prices for 13 hours.
  - **Sept 2, 2026, heat dome:** MISO set its highest-ever September peak (about 123 GW), entered **EEA2**, and prices hit the $10,000/MWh cap somewhere in the footprint as evening solar collapsed and imports weakened (RTO Insider). Our Illinois hub reached $4,109/MWh.
  - **Sept 17, 2026:** a second scarcity event with the same mechanism (Enverus); $7,485/MWh at our hub, the highest price in our MISO data.
- **Why this matters for the model:** the day-ahead market increasingly "sees" scarcity coming. The number of hours where the DA price was already above the spike line went from 16 (2023) to 404 (2026). That is why tomorrow's DA price became the dominant feature, and why the tree models, trained on lower DA prices, fail on test (see `DECISIONS.md` §20–21).

## 3. The model catches the headline events; it misses the surprises
Test and val hours caught at the val-tuned alert threshold (logreg-no_rel):

| Event | Split | Spike hours | Caught |
|---|---|---|---|
| MISO, Winter Storm Fern (Jan 2026) | test | 69 | **69 (100%)** |
| MISO, Sept 2026 heat dome (EEA2) | test | 15 | **15 (100%)** |
| MISO, Sept 17, 2026 scarcity | test | 8 | 6 (75%) |
| MISO, June 2025 heat wave | val | 28 | 26 (93%) |
| MISO, Jan 2025 arctic blast | val | 21 | 21 (100%) |
| ERCOT, Winter Storm Fern (Jan 2026) | test | 8 | 8 (100%) |

Split by whether the day-ahead market had already priced the spike (test year):

| | Spike hours where the DA price was already ≥ threshold | "Surprise" spike hours (DA price below threshold) |
|---|---|---|
| MISO | 211, **97% caught** | 262, **62% caught** |
| ERCOT | 11, **100% caught** | 15, **20% caught** |

- **Takeaway:** big, forecastable weather events are easy, because the DA market has priced them in by 2 PM the day before. The hard part, and the honest limit of a day-ahead model, is **surprise spikes**: sudden plant outages, wind or solar falling short of forecast, or evening ramps the DA market didn't anticipate.
- **In the report:** this is a good way to frame the "real-time spikes beyond what the DA market shows" discussion (TA question 10).

## Sources
- Modo Energy, ERCOT battery buildout 2025 annual report: https://modoenergy.com/research/en/ercot-battery-buildout-2025-annual-report
- Modo Energy, why ERCOT battery revenues were low in 2025: https://modoenergy.com/research/why-were-ercot-battery-revenues-so-low-in-2025-weather-energy-arbitrage-builodout
- Yes Energy, ERCOT's battery storage boom: https://www.yesenergy.com/blog/ercots-battery-storage-boom
- ERCOT IMM (Potomac Economics) report, Sept 2025: https://www.ercot.com/files/docs/2025/09/15/8.1-Independent-Market-Monitor-IMM-Report.pdf
- Ascend Analytics, ERCOT scarcity pricing and load growth: https://www.ascendanalytics.com/blog/ercot-market-outlook-scarcity-pricing-load-growth
- IEEFA, solar and storage kept ERCOT prices low: https://ieefa.org/resources/despite-record-summer-demand-solar-storage-kept-ercot-prices-low
- ERCOT, July 13, 2022 conservation appeal: https://www.ercot.com/news/release/2022-07-13-ercot-issues-conservation
- ERCOT, Sept 6, 2023 EEA2: https://www.ercot.com/news/release/2023-09-06-ercot-has-initiated
- Yes Energy, ERCOT EEA2 analysis: https://www.yesenergy.com/blog/ercot-south-wind-contributed-to-eea-2-event
- ERCOT, Winter Storm Fern post-event report: https://www.ercot.com/files/docs/2026/01/28/ERCOT-Post-Event-Report-Winter-Storm-Fern.pdf
- Voltus, MISO 2025 capacity auction: https://www.voltus.co/blog/miso-2025-pra
- Enel, MISO 2025 capacity auction results: https://www.enelnorthamerica.com/insights/blogs/miso-2025-capacity-auction-results
- MISO IMM (Potomac Economics), Spring 2025 quarterly: https://www.potomaceconomics.com/wp-content/uploads/2025/06/IMM-Quarterly-Report_Spring-2025-MSC.pdf
- MISO IMM, Summer 2025 quarterly (June heat wave): https://www.potomaceconomics.com/wp-content/uploads/2025/11/IMM-Quarterly-Report_Summer-2025-MSC.pdf
- MISO, Winter Storm Fern overview: https://cdn.misoenergy.org/20260217%20RSC%20Item%2005%20Winter%20Storm%20Fern%20Report741721.pdf
- MISO IMM, Q1 2026 quarterly (Fern gas prices): https://cdn.misoenergy.org/20260324%20Markets%20Committee%20of%20the%20BOD%20Item%2009%20IMM%20Quarterly%20Report746764.pdf
- RTO Insider, MISO Max Gen emergency in the arctic blast: https://www.rtoinsider.com/124096-miso-enters-max-gen-emergency-in-arctic-blast/
- RTO Insider, MISO emergency in early-September 2026 heat: https://www.rtoinsider.com/142336-miso-enters-emergency-in-early-sept-heat/
- Enverus, PJM and MISO 2026 winter outlook (Sept 17 event): https://www.enverus.com/blog/pjm-and-miso-2026-winter-outlook-thin-margin/

Caveat: most trend explanations come from market monitors and industry analysts. They are correlational, consistent with our data, but we did not test causation (for example, battery MW is not a model feature).
