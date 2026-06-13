"""Forward-fit calibration: instead of reconstructing tau from (P, V), iteratively
run the whole 0D ChamberSphere forward model and minimize the pressure mismatch.

For each cycle the chamber is driven with the data's volume (prescribed inlet flow
Q = dV/dt, initial volume set) and the actual svZeroD solver produces P(t). The
six free parameters (guccione_C, gamma_sigma_max, tau_1, tau_2, m1, m2) are fit by
least_squares on (P_sim - P_data), warm-started from the reconstruction fit.
volume0, prestress, b_f, b_t are taken from the data (as before); t_shift = 0.
"""
import os
import glob
import numpy as np
import pysvzerod
from scipy.optimize import least_squares

import calibrate_yale as cy

N_CYCLES = 2
FREE_NAMES = ["guccione_C", "gamma_sigma_max", "tau_1", "tau_2", "m1", "m2"]


def run_forward(vals, t, V, dVdt):
    """Drive the chamber with the data volume; return simulated pressure on t."""
    tw = t.tolist() + [cy.PERIOD]
    Qw = dVdt.tolist() + [dVdt[0]]
    cfg = {
        "simulation_parameters": {
            "number_of_cardiac_cycles": N_CYCLES,
            "number_of_time_pts_per_cardiac_cycle": len(t),
            "output_variable_based": True, "output_all_cycles": True,
            "steady_initial": False, "absolute_tolerance": 1e-9,
        },
        "boundary_conditions": [
            {"bc_name": "IN", "bc_type": "FLOW", "bc_values": {"Q": Qw, "t": tw}},
            {"bc_name": "OUT", "bc_type": "FLOW",
             "bc_values": {"Q": [0.0, 0.0], "t": [0.0, cy.PERIOD]}},
        ],
        "vessels": [{
            "vessel_id": 0, "vessel_length": 1.0, "vessel_name": "ventricle",
            "zero_d_element_type": "ChamberSphere",
            "boundary_conditions": {"inlet": "IN", "outlet": "OUT"},
            "zero_d_element_values": vals,
        }],
        "initial_condition": {"volume:ventricle": float(V[0] - vals["volume0"])},
    }
    res = pysvzerod.simulate(cfg)
    sub = res[res["name"] == "pressure:ventricle:OUT"].sort_values("time")
    ts, Ps = sub["time"].to_numpy(), sub["y"].to_numpy()
    ts = ts - (ts.max() - cy.PERIOD)
    return np.interp(t, ts, Ps, period=cy.PERIOD)


def base_vals(theta_free, fixed, b_f, b_t):
    v = dict(zip(FREE_NAMES, theta_free))
    v.update(volume0=fixed["volume0"], prestress=fixed["prestress"],
             t_shift=0.0, b_f=float(b_f), b_t=float(b_t))
    return v


def fit_cycle(path):
    t, P, V = cy.load_cycle(path)
    b_f, b_t = cy.read_b(path)
    dVdt = np.gradient(V, t, edge_order=2)
    fixed = cy.data_fixed(t, P, V)

    # warm start from the reconstruction fit
    theta0_full, *_ = cy.calibrate_cycle(t, P, V, b_f, b_t)
    x0 = np.array([theta0_full[cy.PARAM_NAMES.index(n)] for n in FREE_NAMES])
    scale = max(P.max() - P.min(), 1e3)

    def resid(x):
        return (run_forward(base_vals(x, fixed, b_f, b_t), t, V, dVdt) - P) / scale

    # pressure RMS of the warm start (reconstruction fit), then forward-fit
    rms_recon = np.sqrt(np.mean((resid(x0) * scale) ** 2)) / (P.max() - P.min())
    lb = np.array([0.0, 0.0, 0.02, 0.05, 1.0, 1.0])
    ub = np.array([1e6, 1e7, 0.4, 0.6, 40.0, 40.0])
    r = least_squares(resid, np.clip(x0, lb + 1e-9, ub - 1e-9), bounds=(lb, ub),
                      method="trf", x_scale="jac", max_nfev=200, ftol=1e-8, xtol=1e-8)
    rms_fwd = np.sqrt(np.mean((r.fun * scale) ** 2)) / (P.max() - P.min())
    return rms_recon, rms_fwd, x0, r.x


def cold_start_check(path):
    """Forward-fit from a deliberately-wrong start, to confirm the optimum is
    real and not just the warm start."""
    t, P, V = cy.load_cycle(path)
    b_f, b_t = cy.read_b(path)
    dVdt = np.gradient(V, t, edge_order=2)
    fixed = cy.data_fixed(t, P, V)
    scale = max(P.max() - P.min(), 1e3)

    def resid(x):
        return (run_forward(base_vals(x, fixed, b_f, b_t), t, V, dVdt) - P) / scale
    x_cold = np.array([5.0e2, 3.0e4, 0.10, 0.30, 5.0, 5.0])
    lb = np.array([0.0, 0.0, 0.02, 0.05, 1.0, 1.0])
    ub = np.array([1e6, 1e7, 0.4, 0.6, 40.0, 40.0])
    r = least_squares(resid, x_cold, bounds=(lb, ub), method="trf", x_scale="jac",
                      max_nfev=400, ftol=1e-8, xtol=1e-8)
    return np.sqrt(np.mean((r.fun * scale) ** 2)) / (P.max() - P.min())


if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = sorted(glob.glob(os.path.join(cy.DATA_DIR, "*", "cav.LV.csv")),
                   key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))

    cold = cold_start_check(paths[0])
    print(f"cold-start forward-fit P RMS on cycle_0: {cold:.2%}")

    t0 = time.time()
    rr, rf = [], []
    for path in paths:
        try:
            a, b, *_ = fit_cycle(path)
            rr.append(a); rf.append(b)
        except Exception as e:
            print("  ", os.path.basename(os.path.dirname(path)), e)
    rr, rf = np.array(rr), np.array(rf)
    print(f"\n{len(rr)} cycles in {time.time()-t0:.0f}s")
    print(f"reconstruction-fit  P RMS: median {np.median(rr):.2%}, p90 {np.percentile(rr,90):.2%}")
    print(f"forward-fit         P RMS: median {np.median(rf):.2%}, p90 {np.percentile(rf,90):.2%}")
    print(f"median improvement: {np.median(rr - rf)*100:.3f} pp "
          f"({np.median((rr-rf)/rr)*100:.1f}% relative)")

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(rr * 100, rf * 100, s=20, color="#367", edgecolor="k", lw=0.3)
    lim = [0, max(rr.max(), rf.max()) * 105]
    ax.plot(lim, lim, "k--", lw=1, label="equal")
    ax.set(xlabel="reconstruction-fit P RMS [%]", ylabel="forward-fit P RMS [%]",
           title="Forward-fit vs reconstruction-fit (per cycle)", xlim=lim, ylim=lim)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(cy.OUT_DIR, "viz_forward_vs_recon.png")
    fig.savefig(p, dpi=120); print("wrote", p)
