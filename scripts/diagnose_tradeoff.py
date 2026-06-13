"""Diagnose why fitted guccione_C correlates with afterload (Rsys).

Hypothesis: guccione_C is determined where the passive stress dominates
(end-diastole), so it depends on the operating point P_ed, the stretch reference
volume0 (= ESV per cycle) and prestress (= P_min per cycle). volume0 and
prestress are read from the data and shift with loading (Rsys), leaking into
guccione_C. Test by re-fitting with volume0/prestress held at population
constants and checking whether guccione_C decouples from Rsys.
"""
import glob
import os
import numpy as np
from scipy.optimize import least_squares

import calibrate_yale as cy

X = np.loadtxt(os.path.expanduser(
    "~/Downloads/simulations_data_yale/data/X.txt"))  # a_ventricles, EDP_lv, Rsys
FREE_IDX = [1, 2, 5, 6, 7, 8]  # guccione_C, gamma_sigma_max, tau_1, tau_2, m1, m2


def fit(t, P, V, b_f, b_t, volume0, prestress):
    scale = max(np.max(np.abs(P)), 1e3)

    def expand(xf):
        th = np.empty(len(cy.PARAM_NAMES))
        th[FREE_IDX] = xf
        th[0] = volume0; th[3] = prestress; th[4] = 0.0; th[9] = 1.0
        return th

    def resid(xf):
        th = expand(xf)
        return (cy.model_tau(th, t) - cy.reconstruct_tau(th, P, V, b_f, b_t)) / scale
    x0 = np.array([2e3, 3e4, 0.08, 0.18, 8.0, 8.0])
    lb = np.array([0.0, 0.0, 0.02, 0.05, 1.0, 1.0])
    ub = np.array([1e6, 1e7, 0.4, 0.6, 40.0, 40.0])
    r = least_squares(resid, x0, bounds=(lb, ub), method="trf", x_scale="jac",
                      max_nfev=2000, ftol=1e-12)
    return r.x[0], r.x[1]  # guccione_C, gamma_sigma_max


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def main():
    paths = sorted(glob.glob(os.path.join(cy.DATA_DIR, "*", "cav.LV.csv")),
                   key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    rows = []
    for p in paths:
        idx = int(os.path.basename(os.path.dirname(p)).split("_")[1])
        t, P, V = cy.load_cycle(p)
        b_f, b_t = cy.read_b(p)
        rows.append((idx, t, P, V, b_f, b_t, V.min(), P.min()))
    av = np.array([X[r[0], 0] for r in rows])
    rsys = np.array([X[r[0], 2] for r in rows])
    vol0_med = np.median([r[6] for r in rows])
    pre_med = np.median([r[7] for r in rows])

    print(f"data-derived references vs Rsys: "
          f"corr(volume0=ESV, Rsys)={corr([r[6] for r in rows], rsys):+.2f}, "
          f"corr(prestress=Pmin, Rsys)={corr([r[7] for r in rows], rsys):+.2f}\n")

    for label, getrefs in [
        ("A: per-cycle volume0=ESV, prestress=Pmin (current)",
         lambda r: (r[6], r[7])),
        ("B: volume0, prestress fixed at population medians",
         lambda r: (vol0_med, pre_med)),
        ("C: volume0 fixed, prestress per-cycle",
         lambda r: (vol0_med, r[7])),
        ("D: volume0 per-cycle, prestress fixed",
         lambda r: (r[6], pre_med)),
    ]:
        gC, gsm = [], []
        for r in rows:
            v0, pre = getrefs(r)
            c, s = fit(r[1], r[2], r[3], r[4], r[5], v0, pre)
            gC.append(c); gsm.append(s)
        gC, gsm = np.array(gC), np.array(gsm)
        print(f"{label}")
        print(f"   corr(guccione_C, Rsys)         = {corr(gC, rsys):+.2f}")
        print(f"   corr(guccione_C, a_ventricles) = {corr(gC, av):+.2f}  "
              f"(ideal: high; this is the true material a)")
        print(f"   corr(gamma_sigma_max, Rsys)    = {corr(gsm, rsys):+.2f}\n")


if __name__ == "__main__":
    main()
