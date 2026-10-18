# PART A — Drop-in data files: what to put in `data/`, exact formats, fallbacks, and what is still missing

*(Added 2026-10. Parts 0–5 below are the older provenance notes and remain valid. `docs/DATASETS_REQUIREMENTS.md` is the shorter
status table.)*  Everything in this part is implemented in `pifsrl/env/datafiles.py` / `sessions.py` and verified by
`python scripts/check_data.py` (prints OK / MISSING / PARTIAL / INVALID per file, plus the fallback that will be used).
**Rule: a missing or malformed optional file never crashes a run; the code prints one labelled line and uses the fallback.
Every number you report must carry the label of the fallback that produced it.**

## A.1 Tick-off: what we have, what we do not, what is urgent

| # | Item | File | Status | Fallback in code (label to print in the paper) | A\*-urgency |
|---|---|---|---|---|---|
| 1 | Caltech EV sessions (ACN-Data, 2018-04-25..2021-09-13) | `data/caltech_local_sessions.csv` | ☑ HAVE (1,084 days) | none needed | — |
| 2 | JPL EV sessions (2018-09-05..2021-09-13) | `data/jpl_local_sessions.csv` | ☑ HAVE (953 days; 912 shared with Caltech) | none needed | — |
| 3 | Historical **retail tariff + demand charge**, Caltech | `data/tariff_caltech.csv` | ☐ MISSING | "SCE TOU-D-PRIME **2026 residential stand-in**" | **URGENT** (every $ result depends on it) |
| 4 | Historical retail tariff + demand charge, JPL (serving utility may differ — verify) | `data/tariff_jpl.csv` | ☐ MISSING | same stand-in | **URGENT** |
| 5 | CAISO SP15 LMP, real, 2018-04..2021-09 | `data/caiso_sp15_lmp.csv` | ☐ MISSING for 2018–2021 (the free OASIS API only serves recent months — see A.4) | "synthetic CAISO-like duck curve"; or `--lmp-analog` replay (labelled PROXY) | **HIGH** (only if a wholesale-price claim is made) |
| 6 | Real feeder / system load shape | `data/feeder_load.csv` | ☐ MISSING | archetype base-load generator | MEDIUM |
| 7 | EV battery-capacity distribution | `data/battery_models.csv` | ☐ MISSING | no battery assumption; `--battery-default` = ASSUMED {40,60,75} kWh | MEDIUM (needed only for any V2G claim; ACN has no battery data) |
| 8 | Third ACN site (e.g. office001) | `data/<site>_local_sessions.csv` | ☐ MISSING (optional) | 2-client federated study | MEDIUM — a 2-client FL result is weak evidence |
| 9 | Real IEEE-123 / OpenDSS case | — | ☐ not needed | theory-only (tree lemma + `scripts/bench_tree_qp.py`) | LOW |

**What is urgently needed for an A\*-grade paper, in priority order** (data *and* non-data; honest):
1. **Items 3 + 4** (historical tariffs). Without them the cost numbers are on a stand-in tariff and a reviewer can reject every $ claim.
2. **More held-out days.** With 4 test days per site no policy difference is certifiable (the suite's Hoeffding gate needs
   n ≈ 2 ln(1/δ)·(B/Δ)² days, i.e. >1000 days for Δ=$20, B=$300). Run `--days 60` or more (we have 900+ days per site) and
   report paired bootstrap CIs; use an empirical-Bernstein / bootstrap gate in the paper rather than plain Hoeffding.
3. **Item 5 or an honest "price-shape replay"** — pick one and label it; run the headline result under ≥2 price sources.
4. **The headline must survive the baselines**: tuned EDF *and* MPC *and* safe-uncontrolled (at JPL-stress the safe-uncontrolled
   baseline is often the strongest). Do not claim RL beats a heuristic unless the paired CI excludes 0.
5. Item 8 only if the federated claim is kept in the main text.

## A.2 File formats (copy a template from `data/templates/`, replace the numbers with REAL ones)

All files: UTF-8 CSV with a header row, `.` decimal point, ISO dates. Timestamps with a timezone (`Z`, `-00:00`, `+00:00`) are
converted to `America/Los_Angeles`; timestamps without one are treated as UTC.

### A.2.1 `data/tariff_<site>.csv` — `<site>` is `caltech` or `jpl` (one file per site; the two sites may differ)
Columns (all required except `site`, `months`, `daytype`, `effective_to`, `demand_usd_per_kw_month`):

| column | meaning |
|---|---|
| `site` | `caltech` / `jpl` / `all` (rows for other sites are ignored) |
| `effective_from`, `effective_to` | ISO dates the rate row applied (inclusive); blank `effective_to` = open-ended |
| `months` | `all` · `6-9` (range, wraps: `10-5` = Oct–May) · `6,7,8,9` (list) |
| `daytype` | `weekday` · `weekend` · `all` |
| `start_hour`, `end_hour` | hour of day `[start,end)`; `0` to `24`; wrap-around (`22` to `6`) allowed |
| `energy_usd_per_kwh` | **$/kWh** energy rate in that window |
| `demand_usd_per_kw_month` | **$/kW-month** demand charge (converted to $/kW-day = value/30.4 when `--demand-from-tariff`) |

Sample (`data/templates/tariff_caltech.csv`):
```
site,effective_from,effective_to,months,daytype,start_hour,end_hour,energy_usd_per_kwh,demand_usd_per_kw_month
caltech,2018-01-01,2021-12-31,6-9,weekday,12,18,0.30,15.0
caltech,2018-01-01,2021-12-31,6-9,weekday,0,12,0.12,15.0
```
The 24 h must be fully covered by the rows matching a date (otherwise that date falls back, once-warned).
**Fallback:** `sce_tou_prime` (SCE TOU-D-PRIME, 2026 published residential rates) — label "SCE-2026 stand-in".
Use: `--auto-data` (picks `tariff_<site>.csv` if present) or `--tariff-csv path`; add `--demand-from-tariff` to take the demand charge from it.

### A.2.2 `data/caiso_sp15_lmp.csv` — wholesale prices, node `TH_SP15_GEN-APND`
Raw CAISO OASIS exports are accepted unchanged (columns `INTERVALSTARTTIME_GMT, MARKET_RUN_ID, LMP_TYPE, NODE, MW`; only rows
with `LMP_TYPE=LMP` are used; `MARKET_RUN_ID` DAM preferred unless `--lmp-market RTM`). A simple 2-column file also works:
`timestamp,price` (price in **$/MWh**). Price used = `LMP/1000 + 0.12 $/kWh` retail adder (`price_adder`, assumption; sweep 0.05/0.12/0.20).
```
INTERVALSTARTTIME_GMT,MARKET_RUN_ID,LMP_TYPE,NODE,MW
2019-07-01T07:00:00-00:00,DAM,LMP,TH_SP15_GEN-APND,28.41
```
**Fallbacks, in order:** (1) exact date from the file → real; (2) if the date is absent and `--lmp-analog month` is given, the nearest
same-daytype day of the **same month** in the file (seasonal analog, PROXY); `--lmp-analog any` = nearest same-daytype day of
**any** month (price-shape replay, PROXY); (3) otherwise the labelled SYNTHETIC CAISO-like duck curve. Use `--price-source caiso_rtp --lmp-csv data/caiso_sp15_lmp.csv`
(or `--auto-data --price-source caiso_rtp`).

### A.2.3 `data/feeder_load.csv` — exogenous feeder base-load *shape*
`timestamp,load` (any unit; hourly or finer; one value per row). Each day is normalised to its own peak (level comes from the
IEEE-33 case). Fallback: the archetype load generator. Wired by `--auto-data` / `--load-csv`.

### A.2.4 `data/battery_models.csv` — V2G what-if only
`model,capacity_kwh,weight` (`weight` optional → uniform; weights are normalised). Applied per session: requested energy is kept,
capacity sampled, arrival SoC = 1 − E/C. Fallback: no battery model (capacity = requested energy, i.e. V2G is nearly inert).
`--battery-default` uses the ASSUMED {40,60,75} kWh × {0.4,0.4,0.2} — a modelling assumption, say so.

### A.2.5 Session CSVs (have)
ACN-Data export columns `connectionTime, disconnectTime, doneChargingTime, kWhDelivered, stationID, userInputs, timezone`.
Another ACN site can be added the same way; it needs an entry in `ACN_SITE_INFO` (EVSE count, nameplate kW).

## A.3 Checking and using the files
```
python scripts/check_data.py            # OK / MISSING / PARTIAL / INVALID + the fallback in force, per file
python scripts/check_data.py --strict   # non-zero exit if any optional file is missing/invalid (for CI)
python sample.py ... --auto-data        # wire every file found in data/ ; missing ones fall back by themselves
python sample.py ... --tariff-csv X --lmp-csv Y --load-csv Z --battery-csv W --price-source caiso_rtp --lmp-analog any
```
JPL and Caltech use **the same pipeline and the same flags**; `--auto-data` looks for `tariff_<site>.csv` per site, the other three
files are shared. For a tariff *shift* study give the sites different price sources (`cross_site_fed.py --caltech-price … --jpl-price …`).

## A.4 CAISO download note (from your downloader's banner)
The banner says the free OASIS API only serves data from about **2021-09-14 onward**; April 2018 – Sep 2021 needs the CAISO
bulk portal (or a `gridstatus` export). The ACN sessions end 2021-09-13, so with the free API alone the **exact-date overlap is
~0 days**. Options, honest ranking: (a) get the 2018–2021 history (bulk portal / gridstatus) → real, exact; (b) use the free
recent LMP with `--lmp-analog any` → "real wholesale price *shapes* replayed on other dates" — acceptable as a scenario, **not** as a
historical claim; (c) synthetic. Do not mix: report which one each table used.


---

# PARTS 0–5 (older provenance notes)

## 0. The one framing point that matters most

A reviewer will ask: *"if the RL agent proposes a charging action nobody in history
actually took, where does the resulting voltage come from?"* The honest answer is: **from
a physics simulation** (a power-flow solve), not from a measurement — because a
measurement of a hypothetical action that was never taken cannot exist, for anyone,
in any paper.

This is not a weakness specific to this project. The exact reference paper your professor
sent, Orfanoudakis, Oliehoek, Palensky & Vergara (2025), *"Physics-Informed Reinforcement
Learning for Large-Scale EV Smart Charging Considering Distribution Network Voltage
Constraints,"* Applied Energy (arXiv:2510.12335), benchmarks on the **IEEE 34-bus and
123-bus reference feeders** — published topologies, not live telemetry — and computes
voltage with a **differentiable power-flow model**, i.e. a simulator, for exactly this
reason.

So "use real data, not synthetic" is applied here as: **use real, recorded, cited data
for everything that has a historical record (EV sessions, tariffs, grid topology,
infrastructure ratings), and simulate only the one thing that cannot have a historical
record for a hypothetical action (the resulting power flow)** — which is standard,
reviewable methodology, not a compromise.

---

## 1. What is REAL, with exact provenance

### 1.1 EV charging sessions — Caltech & JPL, ACN-Data

**Files:** `caltech_local_sessions.csv` (31,424 sessions), `jpl_local_sessions.csv`
(33,638 sessions). These are the actual files provided, inspected directly (not assumed):

```
columns: _id, userInputs, sessionID, stationID, spaceID, siteID, clusterID,
        connectionTime, disconnectTime, kWhDelivered, doneChargingTime,
        timezone, userID, site_location
date range: 2018-04-25 to 2021-09-14 (both sites, continuous, ~3.5 years)
```

**Citation:**
> Lee, Z. J., Li, T., & Low, S. H. (2019). ACN-Data: Analysis and Applications of an Open
> EV Charging Dataset. *Proceedings of the Tenth ACM International Conference on Future
> Energy Systems* (e-Energy '19), 139–149. https://doi.org/10.1145/3307772.3328313

**Infrastructure citation** (station ratings, total site capacity):
> Lee, Z. J., Sharma, S., Zhang, D., Low, S. H. et al. Adaptive Charging Networks: A
> Framework for Smart Electric Vehicle Charging. *IEEE Transactions on Smart Grid*,
> 12(5), 4339–4350, 2021.

**What we verified directly from the files (not assumed):**

| Quantity | Caltech | JPL | How verified |
|---|---|---|---|
| Sessions | 31,424 | 33,638 | `len(df)` |
| Distinct stations | 55 | 52 | `df.stationID.nunique()` |
| kWh delivered, mean/median | 9.25 / 6.72 | 14.05 / 11.21 | `df.kWhDelivered.describe()` |
| Session dwell, mean/median (h) | 5.65 / 4.73 | 6.86 / 7.58 | `disconnectTime − connectionTime` |
| Missing `doneChargingTime` | 6.5% | 6.1% | `df.doneChargingTime.isna().mean()` |
| Negative-duration or ≤0 kWh rows | 0 | 0 | explicit check |

**A timestamp bug we caught and fixed, not assumed away:** the `connectionTime` /
`disconnectTime` strings carry a `GMT` suffix. We verified empirically — not by
assumption — that these are genuinely UTC (not a mislabeled local time): converting to
`America/Los_Angeles` produces the expected workday arrival curve (peak at 08:00–09:00,
trough overnight), while treating the raw hour as already-local produces a physically
implausible double-peaked shape. **`load_acn_sessions_real()` performs this timezone
conversion explicitly**; any analysis that skips it will silently report every arrival
time 7–8 hours wrong.

**Energy target reconstruction** (documented, not hidden): ACN-Data records what was
*delivered* (`kWhDelivered`, always present) and, for roughly half of Caltech sessions and
94% of JPL sessions, what the driver *requested* (`kWhRequested`, inside the `userInputs`
field, a Python-literal list of dicts — we take the driver's **last** entry, i.e. their
final, most up-to-date stated request). Where a stated request exists, we use it. Where it
doesn't, `kWhDelivered` is used as a **documented lower-bound proxy** — this likely
understates true demand for those sessions, and that limitation is stated here rather than
corrected for, because correcting it would require inventing a number ACN-Data does not
contain.

**A finding directly useful for your paper, from the real data, not a simulation:** among
sessions with a stated request, **delivered energy was less than requested energy in
~88–90% of sessions** (mean requested ≈ 22 kWh vs. mean delivered ≈ 13 kWh at Caltech).
The historical, largely-uncontrolled charging system already under-served the majority of
stated demand. This is a real, citable baseline for the "service level" metric — your
RL+safety policy's service level should be compared against this real historical number,
not against a synthetic 100%-service assumption.

**Real, revealed-behaviour proxy labels** (used to calibrate the VPW willingness weights,
Section 5.2 of the proposal — ACN-Data has no willingness label, so *some* proxy is
required in any approach, including a purely rule-based one):
- `opt_in` = whether the session has a stated `userInputs` request at all — a real,
  binary, observed behavioural signal (did this driver bother to specify a request).
- `early_leave` = whether the car finished charging (`doneChargingTime`) more than 15
  minutes before the driver actually disconnected — a real signal that the driver was
  *not* in a rush to leave immediately after charging finished.

Both are computed directly from real timestamps in the file — **no synthetic Bernoulli
draw is used for real sessions** (contrast with the synthetic archetypes below, where no
such real signal exists and a documented synthetic proxy remains necessary).

**Infrastructure constants used as real, cited values (not fitted or assumed):**

| Constant | Caltech | JPL | Source |
|---|---|---|---|
| EVSE count (= `k_max`) | 54 | 52 | Lee et al. 2019 |
| Nameplate rating (majority of stations) | 6.7 kW | 6.6 kW | Lee et al. 2021 |
| Faster-rated subset | 14×13.3 kW, 10×16.6 kW | (not separately reported) | Lee et al. 2021 |
| Total feeder capacity | 150 kW | 195 kW | Lee et al. 2021 |

**A fourth integration hazard, caught while wiring up federated learning across both real
sites (not just single-site use):** the hub thermal limit (`s_max`) was, before this
fix, built from a synthetic "22 kW × cluster_size per slot" placeholder with no connection
to either site's real capacity. `recommended_env_overrides_for_acn()` now sets
`real_total_capacity_kw` (150 kW / 195 kW) to override it directly — see
`V2GEnv.__init__` in `pifsrl/env/v2g_env.py` for exactly where this plugs in, and
`tests/test_real_data.py::test_real_total_capacity_overrides_synthetic_nameplate` for the
regression test.

**A fifth, structural point specific to federated learning across the two real sites:**
FedProx/FedAvg average the shared network weights directly, which requires every client's
observation dimension (a function of `k_max`) to match. Caltech (54 EVSE) and JPL (52
EVSE) differ, so `--real-federated` uses a **shared `k_max = max(54, 52) = 54`** for both
clients' action-slot array sizing. This is architecturally necessary, not a modeling
distortion: JPL's real session data never has more than 52 concurrent sessions (a
historical physical fact, not a constraint being relaxed), so the 2 extra slots simply
never get used. Each site's real total capacity still independently governs its own
thermal constraint via `real_total_capacity_kw`, which does not depend on `k_max`.

A methodological note we want to be explicit about: the **action bound** (max kW a
session may be charged at) is set to the site's cited **nameplate rating**, not to the
session's own historically-realized average rate (`kWhDelivered / active_hours`). We
computed that historical rate too (median ≈ 3.1–3.2 kW, 95th percentile ≈ 6.6 kW — fully
consistent with the nameplate spec above) and considered using it directly as the bound,
but rejected that: a session's historical average rate reflects how the *old* control
scheme (ACN's own scheduler, or a slow onboard charger) happened to use the station, not a
physical limit on what our agent is allowed to request. Using it as a hard cap would
silently forbid our agent from charging faster than history did, even on sessions where
faster charging was physically available — precisely the kind of unstated assumption this
document exists to rule out. We *do* let the observed rate raise the bound above nameplate
when it provides clear evidence of a faster-rated station (2.5% of sampled sessions showed
this, consistent with the minority of 13.3/16.6 kW units reported above), but never lower
it. See `load_acn_sessions_real()` in `pifsrl/env/sessions.py` for the exact logic, and
`_RATE_OUTLIER_KW = 20.0` for the documented data-quality filter (a handful of sessions —
12 at Caltech, 22 at JPL — have a `doneChargingTime` that implies a >20 kW rate, ~3x the
fastest real EVSE at either site; these are kept with the nameplate default rather than
dropped, since only the rate *estimate*, not the session itself, is suspect).

### 1.2 Grid topology — IEEE-33 bus feeder

**Already real in this codebase; no change was needed here.**

**Citation:**
> Baran, M. E., & Wu, F. F. (1989). Network reconfiguration in distribution systems for
> loss reduction and load balancing. *IEEE Transactions on Power Delivery*, 4(2),
> 1401–1407.

Hard-coded topology, impedances, and nominal loads in `pifsrl/physics/feeder.py`, validated
in `tests/test_physics.py::test_ieee33_matches_baran_wu` against the paper's published
V_min = 0.9131 p.u. and losses = 202.68 kW (matches to <0.2%). The DistFlow recursion
itself (Eqs. 1–6 of the proposal) is standard radial power-flow theory; Nagrath & Kothari's
*Power System Engineering* is a fine textbook citation for the equations alongside Baran &
Wu for the specific test feeder.

### 1.3 Electricity tariff — SCE TOU-D-PRIME

**Citation:**
> Southern California Edison. "Electric Vehicle (EV) Rate Plans — TOU-D-PRIME."
> https://www.sce.com/save-money/rates-financing/electric-vehicle-plan (accessed
> September 2026; rates subject to change — re-verify the exact figures against SCE's
> current published rate sheet before final submission, and record the access date in
> your paper, since utility tariffs are revised periodically).

This is SCE's rate plan explicitly marketed for EV owners. Implemented in
`sce_tou_prime_price()` in `pifsrl/env/sessions.py`:

| Period | Summer (Jun–Sep) | Winter (Oct–May) |
|---|---|---|
| On-peak (16:00–21:00) | $0.59/kWh | $0.51/kWh |
| Off-peak (all other hours) | $0.26/kWh | $0.24/kWh |

This replaces the previous representative-but-synthetic 3-tier tariff as the **default**
for real-data runs (`EnvConfig.price_source = "sce_tou_prime"`); the synthetic schedule
(`price_source = "synthetic_tou"`) remains available for sensitivity analysis against an
arbitrary tariff shape, clearly labeled as such if you report it.

---

## 2. What remains SYNTHETIC, and exactly why

Being direct about this is what makes the "real" claims above credible.

### 2.1 Background (non-EV) grid load and PV generation

**Status: synthetic** — a parameterized daily shape (`base_load_profile()`,
`pv_profile()` in `pifsrl/env/sessions.py`), not a measured feed.

**Why not fixed yet:** a real substitute exists and is specifically recommended —
**CAISO OASIS** publishes real hourly system demand and real day-ahead LMP prices via a
public, no-auth-required API (`http://oasis.caiso.com/oasisapi/`). This sandbox's network
egress is restricted and cannot reach caiso.com directly, so this integration could not be
completed and verified here. **This is the single highest-value remaining upgrade** if you
want to strengthen the "real data" claim further. A loader stub compatible with a
downloaded CAISO CSV is described in §4 below — it needs your machine's network access to
fetch the file, then the same "inspect real columns before writing the parser" discipline
applied to the ACN files in §1.1.

**Impact of leaving this synthetic:** it affects the *severity* of the background grid
stress (how close to the voltage/thermal limits the network already is before any EV
acts), not the EV session data, not the topology, and not the safety-layer mathematics.
Every physics and safety-layer validation in `tests/` is independent of this choice.

### 2.2 IEEE-123 bus network — removed from scope

At the supervisor's request, the 123-bus network and the QP-scalability benchmark
that depended on it were removed (commented out in the code, kept for reference).
All experiments use the real IEEE-33 feeder only. The earlier synthetic
"123-bus-like" network was never the published IEEE-123 feeder, so no 123-bus
claim should appear in the paper.

### 2.3 The three archetypes with no ACN counterpart

`office`, `residential`, and `fleet_depot` in `pifsrl/env/sessions.py::ARCHETYPES` remain
**synthetic but calibrated** — ACN-Data is workplace-charging-only (Caltech, JPL), so there
is no real, open, per-session dataset for overnight residential or fleet-depot charging
behind these archetypes' assumed distributions. If your paper's scope allows dropping to
just the two archetypes with real data (`workplace_caltech`, `workplace_jpl`), that
removes this caveat entirely; if the federated-learning story needs heterogeneous sites,
these three remain the honest, clearly-labeled synthetic clients, contrasted against the
two real ones.

### 2.4 VPW willingness — synthetic archetypes only

For the synthetic archetypes (§2.3), `_attach_proxy_labels()` in `sessions.py` still
generates `opt_in`/`early_leave` from a latent, synthetic `vpw_prior` variable plus noise,
exactly as before — because no real willingness signal exists for those archetypes either.
**This is unavoidable for any EV dataset in existence**, not a shortcut specific to this
project: no public EV charging dataset records a driver's true price/inconvenience
willingness, so every paper in this literature that models it (including VPW-style
prioritization) calibrates against *some* proxy. For the two real ACN archetypes, that
proxy is now the real behavioral signal described in §1.1 — a genuine improvement over
using a synthetic proxy where a real one is available.

---

## 3. Summary table

| Component | Status | Source | File |
|---|---|---|---|
| EV sessions (Caltech, JPL) | **Real** | Lee, Li & Low (2019) | `data/*_local_sessions.csv` (not bundled — see §5) |
| Session→request reconstruction | Real where stated; documented proxy otherwise | same file, `userInputs` field | `sessions.py::load_acn_sessions_real` |
| Willingness proxy (Caltech/JPL) | Real behavioral signal | same file | `sessions.py::load_acn_sessions_real` |
| Charger ratings, site capacity | **Real** | Lee et al. (2021) | `sessions.py::ACN_SITE_INFO` |
| Grid topology (33-bus) | **Real** | Baran & Wu (1989) | `physics/feeder.py::ieee33` |
| Electricity tariff | **Real** | SCE TOU-D-PRIME | `sessions.py::sce_tou_prime_price` |
| Power flow (given an action) | Simulated (standard practice — see §0) | Baran & Wu DistFlow eqs. | `physics/powerflow.py` |
| Background grid load / PV | Synthetic | — | `sessions.py::base_load_profile`, `pv_profile` |
| Grid topology (123-bus) | Removed from scope | — | (commented out) |
| Office / residential / fleet_depot sessions | Synthetic, calibrated | NHTS-style literature values | `sessions.py::ARCHETYPES` |
| Willingness proxy (synthetic archetypes) | Synthetic | — | `sessions.py::_attach_proxy_labels` |

---

## 4. How to run with real data

```python
from pifsrl.env.v2g_env import EnvConfig, V2GEnv
from pifsrl.env.sessions import recommended_env_overrides_for_acn, list_acn_dates

# see what real calendar dates exist
dates = list_acn_dates("data/caltech_local_sessions.csv", "caltech")
print(dates[0], "to", dates[-1], f"({len(dates)} days)")

# recommended_env_overrides_for_acn fixes three real integration hazards
# (soc_max, cluster_size, k_max) documented in sessions.py -- always use it
overrides = recommended_env_overrides_for_acn("caltech")

cfg = EnvConfig(
    feeder="ieee33", horizon=192, dt_h=0.25, seed=0,
    sessions_source="acn_csv",
    acn_csv_file="data/caltech_local_sessions.csv",
    acn_site="caltech",
    acn_date=dates[500],                 # any real date from the list above
    price_source="sce_tou_prime",        # real SCE tariff instead of synthetic
    **overrides,
)
env = V2GEnv(cfg)
obs = env.reset(0)
```

Or via the CLI (once you've placed the CSVs in `data/`, see §5):
```bash
# Single real day (the default). Screen shows important messages only; the full
# output (and any warnings/errors) is in logs/<mode>_<timestamp>.log
python sample.py --mode single --real-data caltech --acn-date 2019-09-07 --device cuda

# Multiple real days -- the real-data equivalent of running multiple seeds:
# reports mean +/- std across N evenly-spaced real calendar dates rather than
# one (potentially unrepresentative) day.
python sample.py --mode single --real-data caltech --n-real-days 10 --episodes 300 --device cuda

# Real federated learning: Caltech and JPL as the two clients, not synthetic
# archetypes. Genuinely Non-IID by construction (different real capacity,
# different real usage), same real calendar date at both sites by default.
python sample.py --mode federated --real-federated --rounds 60 --compare --device cuda
```

For a genuine train/test split rather than a single day, use `list_acn_dates()` to
partition by calendar date (e.g. train on 2018–2020, held-out test on 2021) — a real
temporal split, not a random shuffle, which is the right way to validate a control policy
against non-stationary real-world behavior.

## 5. Why the CSV files are not bundled in this repository

ACN-Data's licensing is not unambiguous across sources we checked (the original e-Energy
'19 paper carries an ACM "no commercial reproduction/distribution" notice; a public
mirror lists CC BY-NC-SA 4.0). Rather than resolve that ambiguity by assumption, this
repository does not redistribute the CSVs. You already have legitimate access to them
(via your ACN-Data account) — place your own copies at:
```
data/caltech_local_sessions.csv
data/jpl_local_sessions.csv
```
and every path above (loader, tests, CLI) will find them automatically. `tests/test_real_data.py`
is written to skip itself automatically if these files are absent, so their absence from
the zip does not break `sample.py --quicktest` or the main test suite for anyone who
receives the code without the data.

## 6. Verification checklist for this document

Everything stated as "real" above was checked against either the actual uploaded CSV
files (schema, ranges, distributions — reproducible via the commands in `tests/test_real_data.py`)
or a primary source located and read directly (not recalled from training data) during
this integration. The one place a citation could not be independently confirmed — the
third PDF your professor shared ("Dynamic Distribution Network Reconfiguration
Considering V2G-Enabled EV Aggregation Stations...") — is flagged as unconfirmed rather
than guessed; upload that PDF and its exact citation can be extracted directly from the
file.
