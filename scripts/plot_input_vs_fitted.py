"""Relate the generating (input) parameters of the 124 Yale P-V loops to the
fitted ChamberSphere parameters.

The 125 cycles are a 5x5x5 full-factorial sweep over three high-fidelity inputs
(data/X.txt, columns from data/xlabels.txt):
    a_ventricles  - ventricular contractility scaling
    EDP_lv        - LV end-diastolic pressure (preload)
    Rsys          - systemic resistance (afterload)

This script aligns each cycle's inputs with its fitted parameters and shows how
they relate: a correlation heatmap and scatter plots of the strongest pairs.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/Downloads/simulations_data_yale")
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "chamber_sphere_calibration_results.csv")

IN_LABELS = ["a_ventricles", "EDP_lv", "Rsys"]
IN_NICE = ["contractility\n(a_ventricles)", "preload\n(EDP_lv)", "afterload\n(Rsys)"]
# fitted parameters to relate (drop alpha_max: railed constant; steepness: fixed)
FIT_KEYS = ["volume0", "guccione_C", "gamma_sigma_max",
            "tau_1", "tau_2", "m1", "m2"]


def load():
    X = np.loadtxt(os.path.join(DATA, "X.txt"))  # [125, 3], row i = cycle_i
    fitted, inputs = [], []
    with open(RESULTS) as f:
        for r in csv.DictReader(f):
            idx = int(r["cycle"].split("_")[1])
            inputs.append(X[idx])
            fitted.append([float(r[k]) for k in FIT_KEYS])
    return np.array(inputs), np.array(fitted)


def fig_heatmap(inputs, fitted):
    C = np.zeros((len(IN_LABELS), len(FIT_KEYS)))
    for i in range(len(IN_LABELS)):
        for j in range(len(FIT_KEYS)):
            C[i, j] = np.corrcoef(inputs[:, i], fitted[:, j])[0, 1]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FIT_KEYS)))
    ax.set_xticklabels(FIT_KEYS, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(IN_LABELS)))
    ax.set_yticklabels(IN_NICE, fontsize=9)
    for i in range(len(IN_LABELS)):
        for j in range(len(FIT_KEYS)):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="w" if abs(C[i, j]) > 0.6 else "k")
    ax.set_title("How generating parameters drive the fitted ChamberSphere parameters\n"
                 "(Pearson r across 124 cycles)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, label="correlation")
    fig.tight_layout()
    p = os.path.join(BASE, "viz_input_vs_fitted_heatmap.png")
    fig.savefig(p, dpi=120); print("wrote", p)
    return C


def fig_scatters(inputs, fitted, C):
    # six strongest |r| input-fitted pairs
    pairs = sorted(((abs(C[i, j]), i, j) for i in range(len(IN_LABELS))
                    for j in range(len(FIT_KEYS))), reverse=True)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (_, i, j) in zip(axes.flat, pairs):
        x, y = inputs[:, i], fitted[:, j]
        # color by the contractility input to expose the factorial structure
        sc = ax.scatter(x, y, c=inputs[:, 0], cmap="viridis", s=28,
                        edgecolor="k", lw=0.3)
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b * xs + a, "r--", lw=1.2)
        ax.set_xlabel(IN_LABELS[i], fontsize=9)
        ax.set_ylabel(FIT_KEYS[j], fontsize=9)
        ax.set_title(f"r = {C[i, j]:+.2f}", fontsize=10)
        ax.tick_params(labelsize=8)
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("a_ventricles (contractility)", fontsize=9)
    fig.suptitle("Strongest generating-parameter -> fitted-parameter relationships",
                 fontsize=12)
    p = os.path.join(BASE, "viz_input_vs_fitted_scatters.png")
    fig.savefig(p, dpi=120); print("wrote", p)


def fig_factorial(inputs, fitted):
    """Active stress vs contractility, split by preload and afterload, to show
    the 5x5x5 grid structure explicitly."""
    gsm = fitted[:, FIT_KEYS.index("gamma_sigma_max")]
    av, edp, rsys = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    rsys_lvls = np.unique(rsys)
    fig, axes = plt.subplots(1, len(rsys_lvls), figsize=(15, 3.6), sharey=True)
    norm = plt.Normalize(edp.min(), edp.max())
    for ax, rl in zip(axes, rsys_lvls):
        m = rsys == rl
        sc = ax.scatter(av[m], gsm[m], c=edp[m], cmap="plasma", norm=norm,
                        s=30, edgecolor="k", lw=0.3)
        ax.set_title(f"Rsys = {rl:.2f}", fontsize=10)
        ax.set_xlabel("a_ventricles", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("fitted gamma_sigma_max [Pa]", fontsize=9)
    cb = fig.colorbar(sc, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("EDP_lv (preload)", fontsize=9)
    fig.suptitle("Fitted active stress across the contractility x preload x afterload grid",
                 fontsize=12)
    p = os.path.join(BASE, "viz_input_vs_fitted_factorial.png")
    fig.savefig(p, dpi=120); print("wrote", p)


if __name__ == "__main__":
    inputs, fitted = load()
    print(f"aligned {len(inputs)} cycles; inputs {inputs.shape}, fitted {fitted.shape}")
    C = fig_heatmap(inputs, fitted)
    fig_scatters(inputs, fitted, C)
    fig_factorial(inputs, fitted)
