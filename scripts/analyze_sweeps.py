"""Marginal analysis: how each fitted parameter changes along the three input
sweeps (5x5x5 factorial over a_ventricles, EDP_lv, Rsys)."""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/Downloads/simulations_data_yale")
X = np.loadtxt(os.path.join(BASE, "data", "X.txt"))  # [125,3]: a_ventricles, EDP_lv, Rsys
IN_NAMES = ["a_ventricles", "EDP_lv", "Rsys"]
FIT_KEYS = ["guccione_C", "gamma_sigma_max", "prestress", "volume0",
            "alpha_max", "alpha_min", "tsys", "tdias"]
UNITS = {"guccione_C": "Pa", "gamma_sigma_max": "Pa", "prestress": "Pa",
         "volume0": "mL", "alpha_max": "1/s", "alpha_min": "1/s",
         "tsys": "s", "tdias": "s"}


def load():
    inp, fit = [], []
    with open(os.path.join(BASE, "chamber_sphere_calibration_results.csv")) as f:
        for r in csv.DictReader(f):
            idx = int(r["cycle"].split("_")[1])
            inp.append(X[idx])
            row = [float(r[k]) for k in FIT_KEYS]
            fit.append(row)
    inp, fit = np.array(inp), np.array(fit)
    fit[:, FIT_KEYS.index("volume0")] *= 1e6  # m^3 -> mL
    return inp, fit


def main():
    inp, fit = load()
    # marginal means: for each input dim, average each fitted param at each level
    print(f"\nMarginal response of fitted parameters to each input sweep")
    print(f"(mean over the other two inputs; arrow = trend low->high input level)\n")
    for j, fk in enumerate(FIT_KEYS):
        print(f"{fk} [{UNITS[fk]}]:")
        for i, inm in enumerate(IN_NAMES):
            levels = np.unique(inp[:, i])
            means = [fit[np.isclose(inp[:, i], lv), j].mean() for lv in levels]
            lo, hi = means[0], means[-1]
            arrow = "increases" if hi > lo else "decreases"
            rel = (hi - lo) / (abs(np.mean(means)) + 1e-30) * 100
            print(f"   vs {inm:13s}: {means[0]:>9.3g} -> {means[-1]:>9.3g}"
                  f"  ({arrow} {abs(rel):.0f}%)")
        print()

    # marginal-effects figure: rows = fitted params, cols = inputs
    fig, axes = plt.subplots(len(FIT_KEYS), 3, figsize=(11, 2.0 * len(FIT_KEYS)))
    for j, fk in enumerate(FIT_KEYS):
        for i, inm in enumerate(IN_NAMES):
            ax = axes[j, i]
            levels = np.unique(inp[:, i])
            means = np.array([fit[np.isclose(inp[:, i], lv), j].mean() for lv in levels])
            stds = np.array([fit[np.isclose(inp[:, i], lv), j].std() for lv in levels])
            ax.errorbar(levels, means, yerr=stds, marker="o", ms=4, lw=1.2,
                        capsize=2, color="#367")
            if j == len(FIT_KEYS) - 1:
                ax.set_xlabel(inm, fontsize=9)
            if i == 0:
                ax.set_ylabel(f"{fk}\n[{UNITS[fk]}]", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("How each fitted parameter changes along the three input sweeps\n"
                 "(marginal mean +/- SD over the other two inputs)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(BASE, "viz_sweep_marginals.png")
    fig.savefig(p, dpi=115); print("wrote", p)


if __name__ == "__main__":
    main()
