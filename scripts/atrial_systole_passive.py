"""Estimate the passive material from atrial systole (the a-wave of the last
cardiac cycle) instead of the load phase, and compare.

Findings:
 - volume0 (unloaded reference) cannot be fit from the loaded cardiac cycle (no
   P=0 state): a full-fill fit rails it to ESV. It must come from the mesh.
 - guccione_C CAN be fit from atrial systole (a-wave, the cleanest passive
   segment: LV fully relaxed, highest diastolic pressure) with volume0 supplied:
   median 869 Pa (vs 848 from the load phase), corr with the true material a =
   0.62 -- usable, but noisier than the load phase (0.99) because the a-wave
   spans only ~1.5 mmHg.
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calibrate_yale as cy

X = np.loadtxt(os.path.expanduser(
    "~/Downloads/simulations_data_yale/data/X.txt"))


def read_full(path):
    t, P, V, S = [], [], [], []
    with open(path) as fh:
        fh.readline(); fh.readline()
        for line in fh:
            q = [x.strip() for x in line.split(",")]
            try:
                t.append(float(q[0])); P.append(float(q[1]))
                V.append(float(q[2])); S.append(q[3])
            except (ValueError, IndexError):
                pass
    return map(np.array, (t, P, V, S))


def phases(path):
    """Last-beat fill phase split into E-wave, diastasis, a-wave (atrial systole)."""
    t, P, V, S = read_full(path); tmax = t.max()
    m = (t >= tmax - 1000) & (t < tmax) & (S == "fill")
    tf, Pf, Vf = t[m], P[m], V[m]
    dV = np.gradient(Vf, tf); ipk = int(np.argmax(dV))
    awave = tf >= tf[-1] - 150          # last 150 ms = atrial systole / a-wave
    return tf, Pf, Vf, ipk, awave


def fit_C(Pl, Vl, v0, b_f, b_t):        # volume0 fixed, fit only guccione_C
    s = (Vl / v0) ** (1.0 / 3.0)
    base = cy.passive_guccione(s, 1.0, b_f, b_t) / s
    return float(np.sum(base * Pl) / np.sum(base * base))


def main():
    paths = sorted(glob.glob(os.path.join(cy.DATA_DIR, "*", "cav.LV.csv")),
                   key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    av = np.array([X[int(os.path.basename(os.path.dirname(p)).split("_")[1]), 0]
                   for p in paths])
    C_load, C_atr = [], []
    for p in paths:
        b_f, b_t = cy.read_b(p)
        v0, C = cy.estimate_passive(p, b_f, b_t)   # load phase (reference)
        C_load.append(C)
        tf, Pf, Vf, ipk, aw = phases(p)
        C_atr.append(fit_C(Pf[aw] * cy.MMHG_TO_PA, Vf[aw] * cy.ML_TO_M3, v0, b_f, b_t))
    C_load, C_atr = np.array(C_load), np.array(C_atr)
    r_load = np.corrcoef(C_load, av)[0, 1]
    r_atr = np.corrcoef(C_atr, av)[0, 1]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # panel 1: the fill phase split, one cycle
    tf, Pf, Vf, ipk, aw = phases(paths[0])
    ax[0].plot(Vf, Pf, color="#bbb", lw=1, label="diastasis")
    ax[0].plot(Vf[:ipk + 1], Pf[:ipk + 1], color="#39a", lw=2, label="E-wave (early filling)")
    ax[0].plot(Vf[aw], Pf[aw], color="#c44", lw=3, label="atrial systole (a-wave) -> passive fit")
    ax[0].set(xlabel="volume [mL]", ylabel="pressure [mmHg]",
              title="cycle_0 diastolic filling: atrial systole = passive segment")
    ax[0].legend(fontsize=8)
    # panel 2: guccione_C recovery, load phase vs atrial systole
    ax[1].scatter(av, C_load, s=20, color="#37a", label=f"load phase (r={r_load:.2f})")
    ax[1].scatter(av, C_atr, s=20, color="#c44", marker="x", label=f"atrial systole (r={r_atr:.2f})")
    ax[1].set(xlabel="3D Guccione a  (a_ventricles)", ylabel="fitted guccione_C [Pa]",
              title="Passive material: load phase vs atrial systole")
    ax[1].legend()
    fig.tight_layout()
    out = os.path.join(cy.OUT_DIR, "viz_atrial_systole.png")
    fig.savefig(out, dpi=120); print("wrote", out)
    print(f"guccione_C: load-phase median {np.median(C_load):.0f} Pa (r={r_load:.2f}); "
          f"atrial-systole median {np.median(C_atr):.0f} Pa (r={r_atr:.2f})")


if __name__ == "__main__":
    main()
