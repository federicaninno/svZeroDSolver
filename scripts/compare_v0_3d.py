"""Compare the 0D estimated volume0 (Stage-1 load-phase fit) to the 3D cavity
volumes: the unloaded volume (V at P=0) and the operating range (ESV-EDV)."""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calibrate_yale as cy

ML = 1e6  # m^3 -> mL


def main():
    paths = sorted(glob.glob(os.path.join(cy.DATA_DIR, "*", "cav.LV.csv")),
                   key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    v0_est, v_unl, esv, edv = [], [], [], []
    for p in paths:
        b_f, b_t = cy.read_b(p)
        v0_est.append(cy.estimate_passive(p, b_f, b_t)[0])  # 0D Guccione-sphere fit
        v_unl.append(cy.read_unloaded(p))                   # 3D unloaded (V at P=0)
        _, _, V = cy.load_cycle(p)
        esv.append(V.min()); edv.append(V.max())
    v0_est = np.array(v0_est) * ML; v_unl = np.array(v_unl) * ML
    esv = np.array(esv) * ML; edv = np.array(edv) * ML

    r = np.corrcoef(v0_est, v_unl)[0, 1]
    diff = v0_est - v_unl
    print(f"estimated volume0 vs 3D unloaded volume (V at P=0), {len(paths)} cycles:")
    print(f"  r = {r:.4f}, mean diff {diff.mean():+.2f} mL, "
          f"RMS diff {np.sqrt((diff**2).mean()):.2f} mL, max |diff| {np.abs(diff).max():.2f} mL")
    frac = v0_est / edv
    print(f"  unloaded volume0 is {np.median(frac):.0%} of EDV (median); "
          f"sits between ESV (median {np.median(esv):.0f} mL) and "
          f"EDV (median {np.median(edv):.0f} mL)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # panel 1: agreement scatter
    lim = [min(v0_est.min(), v_unl.min()) - 3, max(v0_est.max(), v_unl.max()) + 3]
    ax[0].plot(lim, lim, "k--", lw=1, label="identity")
    ax[0].scatter(v_unl, v0_est, s=22, color="#367", edgecolor="k", lw=0.3)
    ax[0].set(xlabel="3D unloaded cavity volume (V at P=0) [mL]",
              ylabel="0D estimated volume0 [mL]", xlim=lim, ylim=lim,
              title=f"Estimated vs 3D unloaded volume (r = {r:.3f})")
    ax[0].legend()

    # panel 2: volume0 in the operating-range context, cycles sorted by EDV
    o = np.argsort(edv)
    x = np.arange(len(o))
    ax[1].fill_between(x, esv[o], edv[o], color="#cdd", label="operating range (ESV-EDV)")
    ax[1].plot(x, v0_est[o], color="#c44", lw=1.6, label="unloaded volume0 (estimated)")
    ax[1].plot(x, esv[o], color="#666", lw=0.8)
    ax[1].plot(x, edv[o], color="#666", lw=0.8)
    ax[1].set(xlabel="cycle (sorted by EDV)", ylabel="volume [mL]",
              title="Unloaded volume0 vs operating range")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(cy.OUT_DIR, "viz_volume0_vs_3d.png")
    fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
