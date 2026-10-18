# data/  — drop-in files (see ../datasets.md Part A for the full spec)

Required (not redistributed; copy your ACN-Data exports here):
    caltech_local_sessions.csv      jpl_local_sessions.csv

Optional (each has a labelled fallback; copy a template from data/templates/ and fill it with REAL numbers):
    tariff_caltech.csv  tariff_jpl.csv   historical retail TOU (+demand charge)  -> else SCE 2026 stand-in
    caiso_sp15_lmp.csv                    CAISO SP15 LMP (raw OASIS format OK)    -> else synthetic duck curve
    feeder_load.csv                       exogenous feeder load shape            -> else generator
    battery_models.csv                    EV battery capacities (V2G what-if)    -> else none / --battery-default

Check what the code sees:   python scripts/check_data.py
Use everything found:       add --auto-data to sample.py / cross_site_fed.py / run_suite.py
