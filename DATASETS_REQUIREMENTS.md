# Dataset requirements for the PIFSRL-V2G study (what we have, what we still need, how the code degrades)

Status legend: **HAVE** = in the repo/workspace and used; **NEED** = must be obtained by the team (the cloud
sandbox cannot reach the source); **ASSUMED** = a modelling assumption that must be stated as such in the paper;
**FALLBACK** = what the code does when the item is missing (always loud, never silent).

## 1. EV sessions (the demand side)

| Item | Requirement | Status | Notes |
|---|---|---|---|
| Caltech sessions | ACN-Data CSV, columns `connectionTime, disconnectTime, doneChargingTime, kWhDelivered, stationID, userInputs, timezone`; 2018-04-25 .. 2021-09-14 | **HAVE** `data/caltech_local_sessions.csv` (31,424 sessions, 1,084 distinct days) | Lee, Li, Low, e-Energy 2019 |
| JPL sessions | same schema | **HAVE** `data/jpl_local_sessions.csv` (33,638 sessions, 953 days from 2018-09-05, 912 days overlapping Caltech) | JPL is a closed NASA site: cite the ACN-Data licence/terms; redistribute nothing |
| Third ACN site (office001) | optional for a 3-site federated study | **NEED** (optional) | Would test heterogeneity beyond two sites |
| Battery capacity & arrival SoC | per session | **ASSUMED** (ACN has neither) | `userInputs.kWhRequested` gives requested energy only. Every V2G conclusion therefore depends on an assumed battery distribution; say so. Sensitivity sweep over capacity in {40, 60, 80} kWh is a required ablation |
| V2G-capable hardware flag | per charger | **ASSUMED** | ACN chargers are unidirectional; V2G results are "what-if" |
| Willingness (VPW) labels | per user | **ASSUMED / proxy** (`_attach_proxy_labels`) | Not measured; keep the VPW claims out of the main results |

Hygiene rules already enforced in `load_acn_sessions_real`: sessions with implied rate > 20 kW are kept with the
nameplate rating (data-quality artefact), `k_max` is set to the real EVSE count (54 / 52), JPL is padded to 54 slots only
for weight sharing.

## 2. Prices and tariffs (the price side — currently the weakest data link)

| Item | Requirement | Status | FALLBACK / notes |
|---|---|---|---|
| Retail tariff, Caltech | The tariff the site actually paid **in 2018-2021** (commercial TOU with demand charge, e.g. SCE TOU-GS-2/GS-3 or Pasadena Water & Power schedule if that is the serving utility) | **NEED** (verify serving utility with Caltech facilities) | Code uses `sce_tou_prime`: a *residential* EV plan at *2026* published rates applied to 2018-2021 days. This is a stand-in; the paper must say so |
| Retail tariff, JPL | Same, for JPL's serving utility (do not assume it equals Caltech's) | **NEED** | `price_source` is per client/site (`--caltech-price`, `--jpl-price`); a different tariff per site is a first-class setting |
| Demand charge | $/kW-month and the billing window of the above tariffs | **NEED** | We use a *daily* $1.5/kW-day demand charge as a scenario parameter, not a measurement; give a sweep {0, 0.5, 1.5, 3} |
| Wholesale prices (CAISO) | Hourly DAM LMP and/or 5-min RTM LMP, node `TH_SP15_GEN-APND`, 2018-04-01 .. 2021-09-30, UTC timestamps, $/MWh | **NEED**: CAISO OASIS `PRC_LMP` (DAM) / `PRC_INTVL_LMP` (RTM), monthly zips; or `gridstatus`/`gridstatus.io` export. Not reachable from the sandbox | `price_source="caiso_rtp"` + `--lmp-csv`. Columns accepted: `INTERVALSTARTTIME_GMT|timestamp|time` and `MW|LMP|price`. **FALLBACK:** if the file or the date is missing the env prints one warning and uses the labelled *synthetic* duck curve (`caiso_rtp_synthetic_price`) — results with it must be labelled "synthetic CAISO-like" |
| Retail adder for indexed pricing | $/kWh on top of LMP | **ASSUMED** 0.12 (`--price_adder`) | Sweep {0.05, 0.12, 0.20} |
| Ancillary / frequency-regulation prices | only if V2G regulation is studied | **NEED** (out of scope for now) | not used |

## 3. Network and exogenous grid data

| Item | Requirement | Status | Notes |
|---|---|---|---|
| IEEE-33 feeder | Baran & Wu 1989 branch/load tables | **HAVE** (`pifsrl/physics/feeder.py`) | Branch ratings are *derived* (1.6x nominal flow), not published: state it |
| Feeder base load / PV shapes | per-bus time series | **ASSUMED** (shape generators) | A real load profile (e.g. CAISO system load or a utility feeder profile) would strengthen the stress claims |
| Hub siting | which buses host Caltech/JPL | **ASSUMED** (weakest buses) | Both sites are mapped onto the same 33-bus feeder for the federated study; they are independent feeders in the experiments |
| IEEE-123 | real IEEE-123 test-feeder case (OpenDSS/pandapower), multi-phase | **NEED** only if the scale-up is ever coded | Not coded: a synthetic 123-bus graph is disabled on purpose; §7 of the proposal treats scale-up theoretically |

## 4. What the paper must disclose (data section checklist)

1. Which of {tariff, load, PV, battery capacity, VPW, hub siting} are real vs assumed (table above).
2. The tariff year mismatch (2026 published rates vs 2018-2021 sessions) unless a historical tariff is obtained.
3. All price results labelled "SCE-2026 stand-in", "synthetic CAISO-like" or "real CAISO SP15" exactly.
4. Day selection protocol: 20 evenly spaced weekdays per site, split 12/4/4 with a seeded permutation; never tuned on test days.
5. Both regimes (normal, stress) reported side by side; stress = 2x charger power and 2/3 of nameplate hub capacity.

## 5. Minimal acquisition checklist (in order of scientific value)

1. Historical commercial tariff + demand-charge structure for Caltech and for JPL (facilities / utility rate book).
2. CAISO SP15 DAM LMP 2018-2021 CSV (one download; unlocks the real wholesale-price arm).
3. Real feeder load profile for the exogenous load (replaces the shape generator).
4. A battery-capacity distribution source for the EV models seen at the sites (e.g. fleet statistics) for the V2G what-if.
