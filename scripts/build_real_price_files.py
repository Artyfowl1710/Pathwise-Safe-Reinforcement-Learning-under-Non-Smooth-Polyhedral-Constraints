"""Build the two drop-in price files from FREE public sources (provenance printed and written next to the files).
 (1) data/tariff_<site>.csv  <- SCE TOU-EV-4 (March-2019 schedule) as shipped in the open-source `acnportal` package (pip install acnportal).
     Applied to 2018-2021 and to BOTH sites: an ASSUMPTION (the schedule changed over time; JPL's real rate is not published in acnportal).
 (2) data/caiso_sp15_lmp.csv <- LBNL/Berkeley Lab Wholesale Electricity Prices (WEP) tool, CAISO hourly series, 2018-2021 (Millstein, O'Shaughnessy, Wiser 2026).
     This is the WEP CAISO hourly price, NOT the TH_SP15_GEN-APND node -> labelled NODE=WEP_CAISO in the file.
usage: python scripts/build_real_price_files.py --wep path/to/wep_download_data.xlsx"""
import argparse, glob, json, os
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--wep", default=""); ap.add_argument("--data-dir", default="data"); a = ap.parse_args()
import acnportal
sch = json.load(open(glob.glob(os.path.join(os.path.dirname(acnportal.__file__), "signals/tariffs/tariff_schedules/sce_tou_ev_4_march_2019.json"))[0]))
rows = []
for site in ("caltech", "jpl"):
    for blk in sch["schedule"]:
        months = "6-9" if blk["id"].startswith("Summer") else "10-5"; dtp = "weekday" if "Weekday" in blk["id"] else "weekend"
        t = blk["times"] + [24]
        for i, pr in enumerate(blk["tariffs"]):
            rows.append([site, "2018-01-01", "2021-12-31", months, dtp, t[i], t[i + 1], pr, blk["demand_charge"]])
cols = ["site", "effective_from", "effective_to", "months", "daytype", "start_hour", "end_hour", "energy_usd_per_kwh", "demand_usd_per_kw_month"]
df = pd.DataFrame(rows, columns=cols)
for site in ("caltech", "jpl"):
    df[df.site == site].to_csv(os.path.join(a.data_dir, f"tariff_{site}.csv"), index=False)
print("wrote tariff_caltech.csv / tariff_jpl.csv from", sch["name"], "effective", sch["effective"], sch["link"], "| demand", rows[0][-1], "$/kW-month")
if a.wep:
    d = pd.read_excel(a.wep, "8760 prices (heat map)"); d = d[(d.iso == "CAISO") & d.year.between(2018, 2021)].copy()
    ts = pd.to_datetime(d["date"]) + pd.to_timedelta(d["hour"], unit="h")
    ts = ts.dt.tz_localize("US/Pacific", ambiguous="NaT", nonexistent="NaT"); ok = ts.notna()
    out = pd.DataFrame({"INTERVALSTARTTIME_GMT": ts[ok].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S-00:00"), "MARKET_RUN_ID": "WEP", "LMP_TYPE": "LMP", "NODE": "WEP_CAISO", "MW": d.loc[ok, "price"].values})
    out.to_csv(os.path.join(a.data_dir, "caiso_sp15_lmp.csv"), index=False); print("wrote caiso_sp15_lmp.csv rows", len(out), "(WEP CAISO hourly, NOT the SP15 node)")
open(os.path.join(a.data_dir, "PRICE_PROVENANCE.txt"), "w").write("tariff_*.csv: SCE TOU-EV-4 Mar-2019 schedule from acnportal, applied to 2018-2021 and both sites (assumption).\ncaiso_sp15_lmp.csv: LBNL WEP tool CAISO hourly 2018-2021 (Millstein, O'Shaughnessy, Wiser 2026); NOT the SP15 node.\n")
