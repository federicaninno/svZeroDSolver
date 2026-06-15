"""Compare 0D and 3D parameters in two categories:

  (1) COUPLED -- the fitted passive stiffness guccione_C maps to the 3D Guccione a
      (a_ventricles). With volume0 fixed to the 3D unloaded cavity volume (taken
      directly from the data, not fitted), guccione_C no longer absorbs the
      stretch reference, so the coupling is clean (corr ~0.99, slope = the
      thin-wall factor gamma = guccione_C / a, ~constant across the sweep -- this
      is how you get the 0D stiffness from the 3D material). volume0 itself is an
      input here (= 3D unloaded volume by construction), not a fitted 0D output.
  (2) CONSTANT -- the active-twitch timing (tau_1, tau_2, m1, m2). The 3D
      electrophysiology / calcium transient is the same for every cycle, so these
      should be (nearly) constant across the sweep -- unlike the passive/active
      magnitude parameters, which vary by design.
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calibrate_yale as cy

X = np.loadtxt(os.path.expanduser(
    "~/Downloads/simulations_data_yale/data/X.txt"))  # a_ventricles[kPa], EDP_lv, Rsys
ML = 1e6


def collect():
    paths = sorted(glob.glob(os.path.join(cy.DATA_DIR, "*", "cav.LV.csv")),
                   key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    idx = [int(os.path.basename(os.path.dirname(p)).split("_")[1]) for p in paths]
    inputs = X[idx]                       # a_ventricles, EDP_lv, Rsys
    fit = {k: [] for k in cy.PARAM_NAMES}
    unloaded = []
    for p in paths:
        b_f, b_t = cy.read_b(p)
        t, P, V = cy.load_cycle(p)
        v0 = cy.read_unloaded(p)            # volume0 from the 3D data (an input)
        theta, *_ = cy.calibrate_cycle(t, P, V, b_f, b_t, v0)
        for i, k in enumerate(cy.PARAM_NAMES):
            fit[k].append(theta[i])
        unloaded.append(v0)
    return inputs, {k: np.array(v) for k, v in fit.items()}, np.array(unloaded)


def main():
    inputs, fit, unloaded = collect()
    a3d = inputs[:, 0]                     # 3D Guccione a [kPa]

    # ---- Figure 1: coupled parameters (0D vs 3D) ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    gC = fit["guccione_C"]
    r1 = np.corrcoef(a3d, gC)[0, 1]
    k = np.sum(a3d * 1e3 * gC) / np.sum((a3d * 1e3) ** 2)   # gC = k * a  (a in Pa)
    ax[0].scatter(a3d, gC, s=22, color="#367", edgecolor="k", lw=0.3)
    xs = np.linspace(a3d.min(), a3d.max(), 50)
    ax[0].plot(xs, k * xs * 1e3, "r--", lw=1.4, label=f"gC = {k*1e3:.0f}·a  (r={r1:.2f})")
    ax[0].set(xlabel="3D Guccione a  [kPa]", ylabel="0D guccione_C  [Pa]",
              title="COUPLED: passive stiffness"); ax[0].legend(fontsize=9)

    # conversion factor gamma = guccione_C / a (a in Pa): how to get the 0D
    # stiffness from the 3D material. It should be ~constant (the thin-wall
    # geometric factor), independent of the sweep, now that volume0 is fixed.
    gamma = gC / (a3d * 1e3)
    gmed = np.median(gamma)
    cv = np.std(gamma) / np.mean(gamma)
    ax[1].hist(gamma, bins=18, color="#963", edgecolor="k", lw=0.3)
    ax[1].axvline(gmed, color="r", ls="--", lw=1.6,
                  label=f"median γ = {gmed:.3f}\nCV {cv*100:.0f}%")
    ax[1].set(xlabel="γ = guccione_C / a  [-]", ylabel="count",
              title="COUPLED: conversion factor γ (≈const)")
    ax[1].legend(fontsize=9)
    fig.suptitle("0D vs 3D COUPLED: guccione_C = γ·a, with γ the thin-wall factor "
                 "(volume0 taken from 3D)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = os.path.join(cy.OUT_DIR, "viz_0d3d_coupled.png")
    fig.savefig(p1, dpi=120); print("wrote", p1)

    # ---- Figure 2: constant (active timing) vs varying (passive/loading) ----
    timing = ["tau_1", "tau_2", "m1", "m2"]
    varying = ["guccione_C", "gamma_sigma_max"]  # the two fitted magnitudes
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    def strip(a, names, title):
        for i, nm in enumerate(names):
            col = fit[nm]
            norm = col / np.median(col)
            cv = np.std(col) / np.mean(col)
            # strongest absolute correlation with any of the 3 sweep inputs
            mc = max(abs(np.corrcoef(inputs[:, j], col)[0, 1]) for j in range(3))
            a.scatter(np.full_like(norm, i) + np.random.default_rng(i).normal(0, 0.04, len(norm)),
                      norm, s=10, alpha=0.6, color="#367")
            a.text(i, 1.32, f"CV {cv*100:.0f}%\n|r|≤{mc:.2f}", ha="center", fontsize=8)
        a.axhline(1.0, color="#999", lw=0.8, ls=":")
        a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=20, fontsize=9)
        a.set_ylim(0.55, 1.45); a.set_title(title, fontsize=12)

    strip(ax[0], timing, "CONSTANT: active twitch timing\n(3D electrophysiology fixed)")
    strip(ax[1], varying, "VARY: passive + loading params\n(coupled to the 3D sweep)")
    ax[0].set_ylabel("parameter / its median")
    fig.suptitle("0D vs 3D: timing params are constant; passive/loading params vary",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p2 = os.path.join(cy.OUT_DIR, "viz_0d3d_constant.png")
    fig.savefig(p2, dpi=120); print("wrote", p2)

    print(f"\ncoupled: guccione_C vs a  r={r1:.2f};  "
          f"conversion γ=guccione_C/a  median={gmed:.3f} CV={cv:.0%}")
    print("volume0 taken directly from 3D unloaded volume (input, not fitted)")
    print("timing-param CV across 124 cycles:",
          {k: f"{np.std(fit[k])/np.mean(fit[k]):.0%}" for k in timing})


if __name__ == "__main__":
    main()
