import importlib.util, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_small():
    spec = importlib.util.spec_from_file_location("acs", os.path.join(ROOT, "scripts", "adaptive_conformal_shift.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    from types import SimpleNamespace
    from pifsrl.physics.feeder import ieee33
    f = ieee33(); hubs = [f.n_bus - 1, f.n_bus - 8, f.n_bus // 2]
    a = SimpleNamespace(seeds=3, alpha=0.05, gamma=0.01, l_cal=0.75, n_cal=1000, n_test=2000)
    r = m.one_shift(f, hubs, a, 1.25, False)
    return {"fixed": round(float(np.mean(r["fixed"])), 6), "aci": round(float(np.mean(r["aci"])), 6)}
