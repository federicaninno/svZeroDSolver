"""Try estimating the Guccione b parameters from the data instead of taking them
from the 3D parameters.par.

Result: the load-phase EDPVR spans only a small stretch range (the near-linear
foot of the exponential), so b is not identifiable -- a 4-parameter fit
(volume0, guccione_C, b_f, b_t) reproduces the load phase essentially perfectly
but with biased b (~4.6, 0.5 instead of 8, 3), trading off against guccione_C.
The b values only matter at high stretch, which the low-pressure load phase never
reaches. So b must be taken from the 3D model (or literature).
"""
import glob
import os
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calibrate_yale as cy

BF_3D, BT_3D = 8.0, 3.0


def fit_load(Pl, Vl, free_b):
    scale = max(Pl.max(), 1e2)

    def model(x):
        v0, C = x[0], x[1]
        bf, bt = (x[2], x[3]) if free_b else (BF_3D, BT_3D)
        stretch = (Vl / v0) ** (1.0 / 3.0)
        return cy.passive_guccione(stretch, C, bf, bt) / stretch
    n = 4 if free_b else 2
    lb = np.array([Vl.min() * 0.5, 0.0, 0.1, 0.1])[:n]
    ub = np.array([Vl.min() * 1.001, 1e7, 60.0, 60.0])[:n]
    x0 = np.clip([Vl.min() * 0.98, 1e3, 5.0, 5.0][:n], lb + 1e-9, ub - 1e-9)
    r = least_squares(lambda x: (model(x) - Pl) / scale, x0, bounds=(lb, ub),
                      method="trf", x_scale="jac", max_nfev=4000, ftol=1e-14)
    rms = np.sqrt(np.mean(((model(r.x) - Pl)) ** 2)) / max(Pl.max() - Pl.min(), 1.0)
    return r.x, rms


def main():
    paths = sorted(glob.glob(os.path.join(cy.DATA_DIR, "*", "cav.LV.csv")),
                   key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    bf, bt, rms4, rms2 = [], [], [], []
    for p in paths:
        Pl, Vl = cy.read_load_phase(p)
        x4, e4 = fit_load(Pl, Vl, True)
        _, e2 = fit_load(Pl, Vl, False)
        bf.append(x4[2]); bt.append(x4[3]); rms4.append(e4); rms2.append(e2)
    bf, bt = np.array(bf), np.array(bt)
    print(f"fitting b from the load phase ({len(paths)} cycles):")
    print(f"  fitted b_f = {np.median(bf):.2f} [p10 {np.percentile(bf,10):.2f}, "
          f"p90 {np.percentile(bf,90):.2f}]   (true 8.0)")
    print(f"  fitted b_t = {np.median(bt):.2f} [p10 {np.percentile(bt,10):.2f}, "
          f"p90 {np.percentile(bt,90):.2f}]   (true 3.0)")
    print(f"  load-phase fit RMS: free-b {np.median(rms4):.2%}, fixed-b(8,3) {np.median(rms2):.2%}"
          f"  -> both fit equally well, so b is not constrained")

    # demonstration on one cycle: both fits overlap over the load range but
    # diverge when extrapolated to high stretch
    Pl, Vl = cy.read_load_phase(paths[0])
    _, _, V = cy.load_cycle(paths[0])
    x4, _ = fit_load(Pl, Vl, True)
    x2, _ = fit_load(Pl, Vl, False)
    v0 = x4[0]
    Vg = np.linspace(v0 * 1.001, V.max() * 1.6, 300)

    def curve(C, bf, bt):
        s = (Vg / v0) ** (1.0 / 3.0)
        return cy.passive_guccione(s, C, bf, bt) / s / cy.MMHG_TO_PA

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(Vl * 1e6, Pl / cy.MMHG_TO_PA, "ko", ms=4, label="load-phase data")
    ax[0].plot(Vg * 1e6, curve(x2[1], BF_3D, BT_3D), "g-", lw=1.8,
               label=f"b from 3D (8, 3)")
    ax[0].plot(Vg * 1e6, curve(x4[1], x4[2], x4[3]), "r--", lw=1.8,
               label=f"b fitted ({x4[2]:.1f}, {x4[3]:.1f})")
    ax[0].axvspan(Vl.min() * 1e6, Vl.max() * 1e6, color="#eee", label="load-phase range")
    ax[0].axvline(V.max() * 1e6, color="#888", ls=":", lw=1)
    ax[0].text(V.max() * 1e6, 2, " EDV", fontsize=8, color="#555")
    ax[0].set(xlabel="volume [mL]", ylabel="passive pressure [mmHg]", ylim=(-1, 40),
              title="cycle_0: both b fit the load phase, diverge when extrapolated")
    ax[0].legend(fontsize=8)
    ax[1].scatter(bf, bt, s=22, color="#367", edgecolor="k", lw=0.3, label="fitted")
    ax[1].scatter([BF_3D], [BT_3D], s=120, marker="*", color="r", label="true (3D)")
    ax[1].set(xlabel="fitted b_f", ylabel="fitted b_t",
              title="Fitted b vs true (load phase cannot recover them)")
    ax[1].legend()
    fig.tight_layout()
    out = os.path.join(cy.OUT_DIR, "viz_b_estimation.png")
    fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
