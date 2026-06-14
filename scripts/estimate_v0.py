"""Estimate the unloaded volume v0 (and the passive scaling) robustly from the
load phase, and correlate with the inputs.

The 'load' phase of cav.LV.csv is a pure passive inflation from the unloaded mesh
(active stress = 0). So the load-phase (P, V) points are the Guccione EDPVR:
    P(V) = passive_guccione(stretch) / stretch,  stretch = (V / v0)^(1/3),
with prestress = 0 at the unloaded state. Fitting v0 and guccione_C to the whole
loading curve gives a multi-point estimate (vs the single P=0 row) and an
independent passive-material estimate that can be cross-checked against the
cardiac-cycle calibration.
"""
import glob
import os
import numpy as np
from scipy.optimize import least_squares

import calibrate_yale as cy

X = np.loadtxt(os.path.expanduser(
    "~/Downloads/simulations_data_yale/data/X.txt"))  # a_ventricles, EDP_lv, Rsys


def read_load_phase(path):
    """Load-phase (P, V) in SI: rows with t < 0 (passive inflation, P from 0)."""
    t, P, V = [], [], []
    with open(path) as fh:
        fh.readline(); fh.readline()
        for line in fh:
            q = line.split(",")
            try:
                ti = float(q[0])
                if ti < 0:
                    t.append(ti); P.append(float(q[1])); V.append(float(q[2]))
            except (ValueError, IndexError):
                pass
    return (np.array(P) * cy.MMHG_TO_PA, np.array(V) * cy.ML_TO_M3)


def fit_passive(Pl, Vl, b_f, b_t):
    """Fit v0, guccione_C to the passive inflation curve (prestress = 0)."""
    scale = max(Pl.max(), 1e2)

    def model(v0, C):
        stretch = (Vl / v0) ** (1.0 / 3.0)
        return cy.passive_guccione(stretch, C, b_f, b_t) / stretch

    def resid(x):
        return (model(x[0], x[1]) - Pl) / scale
    x0 = np.array([Vl.min() * 0.98, 1.0e3])
    lb = np.array([Vl.min() * 0.5, 0.0])
    ub = np.array([Vl.min() * 1.001, 1e6])
    r = least_squares(resid, np.clip(x0, lb + 1e-12, ub - 1e-12), bounds=(lb, ub),
                      method="trf", x_scale="jac", max_nfev=2000, ftol=1e-12)
    rms = np.sqrt(np.mean((r.fun * scale) ** 2)) / max(Pl.max() - Pl.min(), 1.0)
    return r.x[0], r.x[1], rms


def c(a, b):
    return np.corrcoef(a, b)[0, 1]


def main():
    paths = sorted(glob.glob(os.path.join(cy.DATA_DIR, "*", "cav.LV.csv")),
                   key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    idx = [int(os.path.basename(os.path.dirname(p)).split("_")[1]) for p in paths]
    av, edp, rsys = X[idx, 0], X[idx, 1], X[idx, 2]

    v0_fit, C_load, v0_row, C_card, rms = [], [], [], [], []
    for p in paths:
        b_f, b_t = cy.read_b(p)
        Pl, Vl = read_load_phase(p)
        v0, C, e = fit_passive(Pl, Vl, b_f, b_t)
        v0_fit.append(v0); C_load.append(C); rms.append(e)
        v0_row.append(cy.read_unloaded(p))
        t, P, V = cy.load_cycle(p)
        th, *_ = cy.calibrate_cycle(t, P, V, b_f, b_t, v0)  # cardiac fit using this v0
        C_card.append(th[1])
    v0_fit = np.array(v0_fit); C_load = np.array(C_load); rms = np.array(rms)
    v0_row = np.array(v0_row); C_card = np.array(C_card)

    print(f"load-phase passive fit ({len(paths)} cycles), median P RMS {np.median(rms):.2%}\n")
    print(f"v0 (load-phase fit) vs v0 (single P=0 row): "
          f"r = {c(v0_fit, v0_row):.3f}, "
          f"median |diff| = {np.median(np.abs(v0_fit-v0_row))*1e6:.2f} mL "
          f"(robustness: should agree)\n")
    print(f"{'quantity':<24}{'a_ventricles':>14}{'EDP_lv':>10}{'Rsys':>10}")
    for nm, arr in [("v0  (load-phase fit)", v0_fit),
                    ("v0  (P=0 row)", v0_row),
                    ("guccione_C (load only)", C_load),
                    ("guccione_C (cardiac)", C_card)]:
        print(f"{nm:<24}{c(arr,av):>14.2f}{c(arr,edp):>10.2f}{c(arr,rsys):>10.2f}")
    print(f"\nguccione_C: load-phase vs cardiac-cycle estimate: r = {c(C_load, C_card):.2f} "
          f"(median load {np.median(C_load):.0f} Pa, cardiac {np.median(C_card):.0f} Pa)")


if __name__ == "__main__":
    main()
