"""Big grid of all 124 LV P-V loops: data vs fitted ChamberSphere.

The fitted loop is the model pressure at each observed volume,
    P_model(t) = (tau_pred(t) + passive(V)) / stretch(V),
where passive = 4*gamma_W1*(1 - CG^-3) + prestress and tau_pred is the active
stress from integrating the calibrated activation ODE. The data loop is the raw
(V, P). Their gap is the active-stress fit error propagated into pressure.
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import calibrate_yale as cy

OUT = cy.OUT_DIR
MMHG = cy.MMHG_TO_PA


def model_pressure(theta, t, P, V):
    volume0, gamma_W1, _, prestress = theta[:4]
    stretch = (V / volume0) ** (1.0 / 3.0)
    CG = stretch ** 2
    passive = 4.0 * gamma_W1 * (1.0 - CG ** (-3)) + prestress
    tau_obs = cy.reconstruct_tau(theta, P, V)
    tau_pred = cy.integrate_tau(theta, t, tau_obs[0])
    return (tau_pred + passive) / stretch  # Pa


def main():
    paths = sorted(glob.glob(os.path.join(cy.DATA_DIR, "*", "cav.LV.csv")),
                   key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    results = []
    for path in paths:
        name = os.path.basename(os.path.dirname(path))
        try:
            t, P, V = cy.load_cycle(path)
            theta, rms, amp, _, _ = cy.calibrate_cycle(t, P, V)
            Pm = model_pressure(theta, t, P, V)
            results.append((int(name.split("_")[1]), V * 1e6, P / MMHG, Pm / MMHG,
                            rms / max(amp, 1.0)))
        except Exception as e:
            print(f"  {name}: {e}")
    print(f"plotting {len(results)} loops")

    ncol, nrow = 12, 11
    fig, axes = plt.subplots(nrow, ncol, figsize=(26, 24))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (cyc, V, Pd, Pm, err) in zip(axes.flat, results):
        ax.axis("on")
        ax.plot(V, Pd, color="#222", lw=1.4)
        ax.plot(V, Pm, color="#d83", lw=1.2, ls="--")
        ax.set_title(f"cycle_{cyc}  ({err*100:.1f}%)", fontsize=8, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)

    fig.legend(handles=[Line2D([], [], color="#222", lw=1.6, label="data"),
                        Line2D([], [], color="#d83", lw=1.6, ls="--", label="fitted ChamberSphere")],
               loc="upper center", ncol=2, fontsize=14, frameon=False,
               bbox_to_anchor=(0.5, 0.997))
    fig.suptitle("All 124 LV pressure-volume loops: data vs fitted ChamberSphere "
                 "(x = volume [mL], y = pressure [mmHg]; % = active-stress fit error)",
                 fontsize=15, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p = os.path.join(OUT, "viz_all_pv_loops.png")
    fig.savefig(p, dpi=95); print("wrote", p)


if __name__ == "__main__":
    main()
