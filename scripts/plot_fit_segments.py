"""Show which segments of the data feed which fit (one representative cycle)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calibrate_yale as cy

PERIOD_MS = 1000.0


def main():
    path = os.path.join(cy.DATA_DIR, "cycle_0", "cav.LV.csv")
    t, P, V = [], [], []
    with open(path) as fh:
        fh.readline(); fh.readline()
        for line in fh:
            q = line.split(",")
            try:
                t.append(float(q[0])); P.append(float(q[1])); V.append(float(q[2]))
            except (ValueError, IndexError):
                pass
    t, P, V = np.array(t), np.array(P), np.array(V)
    tmax = t.max()
    last0 = tmax - PERIOD_MS                       # start of the last beat

    # the unloaded point (first row, t<0, P=0): supplies volume0 directly
    t_un, V_un = t[0], V[0]

    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    for a, y, lab in [(ax[0], P, "pressure [mmHg]"), (ax[1], V, "volume [mL]")]:
        a.plot(t, y, color="#222", lw=1.0)
        a.axvspan(t.min(), 0, color="#ccc", alpha=0.5)               # load phase
        a.axvspan(0, last0, color="#ddd", alpha=0.7)                 # init beats
        a.axvspan(last0, tmax, color="#f0a64a", alpha=0.5)           # last beat
        a.set_ylabel(lab, fontsize=10)
    ax[1].set_xlabel("time [ms]", fontsize=10)
    # mark the unloaded point that supplies volume0 (on the volume axis)
    ax[1].scatter([t_un], [V_un], s=55, color="#2a7", edgecolor="k", zorder=5)
    ax[1].annotate(" volume0\n (unloaded, P=0)", (t_un, V_un), color="#176",
                   fontsize=8.5, va="center", weight="bold")
    # end-diastole (max V) of the last beat -> where the cycle is rolled to t=0
    mb = t >= last0
    ted = t[mb][np.argmax(V[mb])]
    ax[1].axvline(ted, color="r", ls=":", lw=1.2)
    ax[1].text(ted, V.max(), " EDV (roll to t=0)", color="r", fontsize=8, va="top")

    ax[0].text(t.min() + 5, P.max() * 0.7,
               "load phase\n-> volume0 only\n(unloaded volume)", fontsize=9, color="#176")
    ax[0].text(last0 - 1950, P.max() * 0.7,
               "init beats\n(not used)", fontsize=9, color="#777", ha="left")
    ax[0].text(last0 + 30, P.max() * 0.6,
               "LAST CARDIAC CYCLE\n-> fitted parameters\n(guccione_C,\n gamma_sigma_max,\n tau_1, tau_2, m1, m2)",
               fontsize=9, color="#a60", weight="bold")
    ax[0].set_title("Which data feeds the fit: volume0 from the unloaded point, "
                    "the rest from the last cardiac cycle", fontsize=12)
    fig.tight_layout()
    out = os.path.join(cy.OUT_DIR, "viz_fit_segments.png")
    fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
