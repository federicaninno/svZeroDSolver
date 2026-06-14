"""Relate the passive stiffness scaling in the two models.

3D high-fidelity: Guccione W = (a/2)(exp(Q)-1), a = a_ventricles (kPa), b_f=8, b_t=3.
0D ChamberSphere: the fitted scaling guccione_C [Pa], same b_f, b_t.

The 0D model absorbs the thin-wall factor gamma = t0/r0 (wall thickness / radius)
into the gamma-scaled stress, so
    guccione_C = gamma * a    (with a in the same stress units).
For a thin sphere with (roughly constant) wall volume V_wall,
    gamma = t0/r0 = V_wall / (3 * V0),
so guccione_C should scale as a / V0. We test both forms and report the
conversion factor.
"""
import glob
import os
import numpy as np

import calibrate_yale as cy

X = np.loadtxt(os.path.expanduser(
    "~/Downloads/simulations_data_yale/data/X.txt"))  # a_ventricles[kPa], EDP_lv, Rsys
KPA = 1.0e3


def main():
    paths = sorted(glob.glob(os.path.join(cy.DATA_DIR, "*", "cav.LV.csv")),
                   key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    gC, a, V0 = [], [], []
    for p in paths:
        i = int(os.path.basename(os.path.dirname(p)).split("_")[1])
        b_f, b_t = cy.read_b(p)
        v0, c = cy.estimate_passive(p, b_f, b_t)
        gC.append(c); a.append(X[i, 0] * KPA); V0.append(v0)
    gC = np.array(gC); a = np.array(a); V0 = np.array(V0)  # Pa, Pa, m^3
    r0 = (3.0 * V0 / (4.0 * np.pi)) ** (1.0 / 3.0)         # sphere radius (m)
    gamma = gC / a                                         # dimensionless t0/r0

    def fit_through_origin(x, y):
        k = np.sum(x * y) / np.sum(x * x)
        r2 = 1 - np.sum((y - k * x) ** 2) / np.sum((y - y.mean()) ** 2)
        return k, r2

    print(f"passive stiffness, {len(paths)} cycles (a in Pa, guccione_C in Pa)\n")
    print("Model 1:  guccione_C = gamma * a  (gamma constant)")
    k1, r2_1 = fit_through_origin(a, gC)
    print(f"   gamma = {k1:.3f}   (R^2 = {r2_1:.3f});  per-cycle gamma = "
          f"{np.median(gamma):.3f} +/- {np.std(gamma):.3f} (CV {np.std(gamma)/np.mean(gamma):.0%})")

    print("\nModel 2:  guccione_C = (V_wall/3) * a / V0  (wall volume constant)")
    k2, r2_2 = fit_through_origin(a / V0, gC)
    Vwall = 3.0 * k2
    print(f"   V_wall = {Vwall*1e6:.1f} mL   (R^2 = {r2_2:.3f})")

    t0 = gamma * r0
    print(f"\nimplied geometry: r0 = {np.median(r0)*1e2:.2f} cm, "
          f"wall thickness t0 = gamma*r0 = {np.median(t0)*1e3:.1f} mm "
          f"(t0/r0 = {np.median(gamma):.2f})")
    print(f"correlation gamma vs V0: r = {np.corrcoef(gamma, V0)[0,1]:+.2f} "
          f"(thin-wall theory predicts gamma ~ 1/V0)")

    print("\n--- conversion ---")
    print(f"  3D -> 0D:  guccione_C [Pa]  ~  {np.median(gamma):.2f} * a [Pa]"
          f"  =  {np.median(gamma)*KPA:.0f} * a [kPa]")
    print(f"  0D -> 3D:  a [kPa]          ~  guccione_C [Pa] / {np.median(gamma)*KPA:.0f}")
    print(f"  geometric: guccione_C = (t0/r0) * a,  t0/r0 = V_wall/(3 V0)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(a / KPA, gC, s=22, color="#367", edgecolor="k", lw=0.3)
    xs = np.linspace(a.min(), a.max(), 50)
    ax[0].plot(xs / KPA, k1 * xs, "r--", lw=1.4, label=f"guccione_C = {k1:.2f}·a  (R²={r2_1:.3f})")
    ax[0].set(xlabel="3D Guccione a  [kPa]", ylabel="0D guccione_C  [Pa]",
              title="Passive scaling: 0D vs 3D"); ax[0].legend()
    ax[1].scatter(V0 * 1e6, gamma, s=22, color="#963", edgecolor="k", lw=0.3)
    ax[1].set(xlabel="unloaded volume V0 [mL]", ylabel="gamma = guccione_C / a  (= t0/r0)",
              title="Thin-wall factor gamma vs V0")
    fig.tight_layout()
    out = os.path.join(cy.OUT_DIR, "viz_passive_stiffness.png")
    fig.savefig(out, dpi=120); print("\nwrote", out)


if __name__ == "__main__":
    main()
