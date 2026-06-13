"""Make an example fit figure and a markdown report from the Yale calibration."""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calibrate_yale as cy

OUT = cy.OUT_DIR


def example_figure(cycle="cycle_0"):
    path = os.path.join(cy.DATA_DIR, cycle, "cav.LV.csv")
    t, P, V = cy.load_cycle(path)
    b_f, b_t = cy.read_b(path)
    theta, rms, amp, rel_se, at_bound = cy.calibrate_cycle(t, P, V, b_f, b_t)
    tau_obs = cy.reconstruct_tau(theta, P, V, b_f, b_t)
    tau_pred = cy.integrate_tau(theta, t, tau_obs[0])
    act, act_plus = cy.activation(t, *theta[4:9])

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(V * 1e6, P / cy.MMHG_TO_PA, "k")
    ax[0].set(xlabel="volume [mL]", ylabel="pressure [mmHg]",
              title=f"{cycle}: LV P-V loop (data)")
    ax[1].plot(t, tau_obs / 1e3, label="tau reconstructed (from P,V)")
    ax[1].plot(t, tau_pred / 1e3, "--", label="tau ODE fit")
    ax[1].set(xlabel="time [s]", ylabel="active stress tau [kPa]",
              title=f"active-stress fit (rel err {rms/max(amp,1):.1%})")
    ax[1].legend(fontsize=8)
    ax[2].plot(t, act_plus, label="act_plus (systole gate)")
    ax[2].axvline(theta[6], color="g", ls=":", label="tsys")
    ax[2].axvline(theta[7], color="r", ls=":", label="tdias")
    ax[2].set(xlabel="time [s]", ylabel="activation", title="calibrated activation")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT, "chamber_sphere_fit_example.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    example_figure()
