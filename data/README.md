# `data/` — Data Files & Drop-in Specification

Full schema details, source citations, and assumptions are documented in [`datasets.md`](../datasets.md).

---

## 1. Real ACN-Data Session CSVs (Optional / User-Supplied)

Due to licensing restrictions (Lee, Li, Low, e-Energy 2019), raw ACN session exports are not redistributed directly in this repository. You can obtain them freely from the official portal:
* **Portal:** [ev.caltech.edu/dataset](https://ev.caltech.edu/dataset)
* **Target Files:**
  * `caltech_local_sessions.csv` (Caltech site, 54 EVSE slots)
  * `jpl_local_sessions.csv` (JPL NASA-JPL site, 52 EVSE slots)

> **Important Reproducibility Note:**
> If these CSVs are not present in `data/`, the entire pipeline **automatically falls back** to the high-fidelity calibrated synthetic session generator. All unit tests (`python sample.py --tests`), quick verification (`python sample.py --quicktest`), and standard training scripts run without errors.

---

## 2. Drop-in Tariff and Exogenous Data Files

Format templates are provided in [`data/templates/`](templates/). Each file has an automatic, labeled fallback if omitted:

| File | Description | Automatic Fallback if Omitted |
| :--- | :--- | :--- |
| `tariff_caltech.csv` | Caltech retail TOU + demand charges | Published SCE TOU-D-PRIME rate schedule |
| `tariff_jpl.csv` | JPL retail TOU + demand charges | Published SCE TOU-D-PRIME rate schedule |
| `caiso_sp15_lmp.csv` | CAISO SP15 wholesale Locational Marginal Prices | Synthetic CAISO-like duck curve model |
| `feeder_load.csv` | Exogenous non-EV base feeder load shape | IEEE-33 archetype load generator |
| `battery_models.csv` | EV battery capacity distribution | Standard distribution {40, 60, 75} kWh |

---

## 3. Data Diagnostics

Verify what data files are recognized and see active fallbacks by running:
```bash
python scripts/check_data.py
```
To automatically load all recognized real data files across experiments:
```bash
python sample.py --auto-data --mode single
```
