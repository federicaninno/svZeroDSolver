"""Visualize the Yale ChamberSphere calibration results in several ways."""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import calibrate_yale as cy

OUT = cy.OUT_DIR
UNITS = ["m^3", "Pa", "Pa", "Pa", "1/s", "1/s", "s", "s", "s"]
NAMES = cy.PARAM_NAMES


def run_all():
    """Re-run the calibration, collecting parameters, identifiability and the
    raw P-V loops."""
    cycles = sorted(glob.glob(os.path.join(cy.DATA_DIR, "*", "cav.LV.csv")))
    thetas, relses, bounds, relfit, loops = [], [], [], [], []
    for path in cycles:
        try:
            t, P, V = cy.load_cycle(path)
            b_f, b_t = cy.read_b(path)
            theta, rms, amp, rel_se, at_bound = cy.calibrate_cycle(t, P, V, b_f, b_t)
            thetas.append(theta); relses.append(rel_se); bounds.append(at_bound)
            relfit.append(rms / max(amp, 1.0))
            loops.append((V * 1e6, P / cy.MMHG_TO_PA, theta))  # mL, mmHg
        except Exception:
            pass
    return (np.array(thetas), np.array(relses), np.array(bounds),
            np.array(relfit), loops)


def fig_distributions(thetas, bounds):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for i, ax in enumerate(axes.flat):
        col = thetas[:, i]
        pct_bound = 100.0 * np.mean(bounds[:, i])
        fixed = NAMES[i] == "steepness"
        from_data = NAMES[i] in cy.DATA_DERIVED
        poor = pct_bound > 50
        color = "#888" if fixed else ("#8a6" if from_data else
                                      ("#c44" if poor else "#39a"))
        ax.hist(col, bins=24, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(np.median(col), color="k", ls="--", lw=1)
        tag = ("fixed (not calibrated)" if fixed else "read from data"
               if from_data else "NOT identifiable" if poor else "identifiable")
        ax.set_title(f"{NAMES[i]}  [{UNITS[i]}]\nmedian={np.median(col):.3g}  ({tag})",
                     fontsize=9, color=color if (poor or fixed or from_data) else "black")
        ax.tick_params(labelsize=8)
    fig.suptitle("Parameter distributions across 124 cardiac cycles", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, "viz_param_distributions.png")
    fig.savefig(p, dpi=110); print("wrote", p)


def fig_fit_and_identifiability(thetas, relses, bounds, relfit):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    # (1) fit-error histogram
    ax[0].hist(relfit * 100, bins=25, color="#3a7", edgecolor="white")
    ax[0].axvline(np.median(relfit) * 100, color="k", ls="--",
                  label=f"median {np.median(relfit)*100:.2f}%")
    ax[0].set(xlabel="relative fit error [%]", ylabel="# cycles",
              title="Fit quality (active-stress match)")
    ax[0].legend(fontsize=9)

    # (2) identifiability: median rel.SE (log) with %@bound annotation
    med_se = np.median(relses, axis=0)
    pct_b = 100.0 * np.mean(bounds, axis=0)
    y = np.arange(len(NAMES))
    colors = ["#c44" if pct_b[i] > 50 else "#39a" for i in range(len(NAMES))]
    ax[1].barh(y, np.maximum(med_se, 1e-9), color=colors)
    ax[1].set_xscale("log")
    ax[1].set_yticks(y); ax[1].set_yticklabels(NAMES, fontsize=9)
    ax[1].invert_yaxis()
    ax[1].axvline(0.1, color="gray", ls=":", lw=1)
    for i in range(len(NAMES)):
        if pct_b[i] > 50:
            ax[1].text(med_se[i], i, f"  {pct_b[i]:.0f}%@bound",
                       va="center", fontsize=8, color="#c44")
    ax[1].set(xlabel="median relative std. error (log)",
              title="Identifiability per parameter")

    # (3) parameter correlation heatmap (non-railed parameters)
    keep = [i for i in range(len(NAMES)) if np.std(thetas[:, i]) > 1e-12
            and np.mean(bounds[:, i]) < 0.5]
    sub = thetas[:, keep]
    C = np.corrcoef(sub.T)
    im = ax[2].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[2].set_xticks(range(len(keep))); ax[2].set_yticks(range(len(keep)))
    lbl = [NAMES[i] for i in keep]
    ax[2].set_xticklabels(lbl, rotation=45, ha="right", fontsize=8)
    ax[2].set_yticklabels(lbl, fontsize=8)
    ax[2].set_title("Parameter correlations\n(identifiable params, across cycles)")
    for a in range(len(keep)):
        for b in range(len(keep)):
            ax[2].text(b, a, f"{C[a,b]:.2f}", ha="center", va="center",
                       fontsize=7, color="k" if abs(C[a, b]) < 0.6 else "w")
    fig.colorbar(im, ax=ax[2], fraction=0.046)

    fig.tight_layout()
    p = os.path.join(OUT, "viz_fit_identifiability.png")
    fig.savefig(p, dpi=110); print("wrote", p)


def fig_pv_loops(loops, thetas):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    gsm = thetas[:, 2]  # gamma_sigma_max
    norm = plt.Normalize(gsm.min(), gsm.max())
    for (V, P, th) in loops:
        ax[0].plot(V, P, color=cm.viridis(norm(th[2])), lw=0.6, alpha=0.6)
    ax[0].set(xlabel="volume [mL]", ylabel="pressure [mmHg]",
              title="All 124 LV P-V loops (color = active stress gamma_sigma_max)")
    sm = cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    fig.colorbar(sm, ax=ax[0], label="gamma_sigma_max [Pa]")

    # active stress amplitude vs systolic timing window, sized by stiffness
    win = thetas[:, 7] - thetas[:, 6]  # tdias - tsys
    sc = ax[1].scatter(gsm, thetas[:, 1], c=win, s=30, cmap="plasma",
                       edgecolor="k", lw=0.3)
    ax[1].set(xlabel="gamma_sigma_max  [Pa]  (active stress level)",
              ylabel="guccione_C  [Pa]  (passive scaling)",
              title="Active vs passive material parameters")
    fig.colorbar(sc, ax=ax[1], label="systole duration tdias-tsys [s]")

    fig.tight_layout()
    p = os.path.join(OUT, "viz_pv_loops.png")
    fig.savefig(p, dpi=110); print("wrote", p)


if __name__ == "__main__":
    thetas, relses, bounds, relfit, loops = run_all()
    print(f"loaded {len(thetas)} cycles")
    fig_distributions(thetas, bounds)
    fig_fit_and_identifiability(thetas, relses, bounds, relfit)
    fig_pv_loops(loops, thetas)
