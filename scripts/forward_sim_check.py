"""Forward-simulation sanity check for the calibrated ChamberSphere.

For each cycle, drive the calibrated chamber with the data's volume trajectory
(prescribe the net inlet flow Q = dV/dt and the initial volume), run the actual
svZeroD forward solver, and compare the simulated pressure to the data. This
closes the loop: the calibration works in active-stress space, so reproducing the
measured P-V loop with the real C++ block validates the whole pipeline.
"""
import os
import json
import numpy as np
import pysvzerod

import calibrate_yale as cy

N_CYCLES = 4  # run a few cycles; compare the converged last one


def forward_pv(path):
    t, P, V = cy.load_cycle(path)            # SI, last beat, rolled to EDV at t=0
    b_f, b_t = cy.read_b(path)
    theta, _, _, _, _ = cy.calibrate_cycle(t, P, V, b_f, b_t)
    names = cy.PARAM_NAMES
    vals = {names[i]: float(theta[i]) for i in range(len(names))}
    vals["b_f"], vals["b_t"] = float(b_f), float(b_t)

    # net inflow that reproduces the volume trajectory: Q = dV/dt (Qout = 0)
    dVdt = np.gradient(V, t, edge_order=2)
    tw = t.tolist() + [cy.PERIOD]
    Qw = dVdt.tolist() + [dVdt[0]]            # periodic wrap

    volume0 = vals["volume0"]
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
        # initial volume state = V(0) - volume0 (data starts at end-diastole)
        "initial_condition": {"volume:ventricle": float(V[0] - volume0)},
    }
    res = pysvzerod.simulate(cfg)

    # last cycle of the simulated pressure, aligned to the data time grid
    sub = res[res["name"] == "pressure:ventricle:OUT"].sort_values("time")
    ts, Ps = sub["time"].to_numpy(), sub["y"].to_numpy()
    m = ts >= ts.max() - cy.PERIOD - 1e-9
    ts, Ps = ts[m] - (ts.max() - cy.PERIOD), Ps[m]
    Psim = np.interp(t, ts, Ps, period=cy.PERIOD)
    return t, V, P, Psim


def main():
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    paths = sorted(glob.glob(os.path.join(cy.DATA_DIR, "*", "cav.LV.csv")),
                   key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    results = []
    for path in paths:
        cyc = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        try:
            t, V, P, Psim = forward_pv(path)
            err = np.sqrt(np.mean((Psim - P) ** 2)) / (P.max() - P.min())
            results.append((cyc, V * 1e6, P / cy.MMHG_TO_PA, Psim / cy.MMHG_TO_PA, err))
        except Exception as e:
            print(f"  cycle_{cyc}: {e}")
    errs = np.array([r[4] for r in results])
    print(f"forward-sim vs data over {len(results)} cycles: "
          f"median rel RMS {np.median(errs):.2%}, p90 {np.percentile(errs,90):.2%}, "
          f"worst {errs.max():.2%}")

    # big grid: data vs forward-simulated P-V loops
    ncol, nrow = 12, 11
    fig, axes = plt.subplots(nrow, ncol, figsize=(26, 24))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (cyc, V, Pd, Ps, err) in zip(axes.flat, results):
        ax.axis("on")
        ax.plot(V, Pd, color="#222", lw=1.4)
        ax.plot(V, Ps, color="#2a8", lw=1.2, ls="--")
        ax.set_title(f"cycle_{cyc}  ({err*100:.1f}%)", fontsize=8, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    fig.legend(handles=[Line2D([], [], color="#222", lw=1.6, label="data"),
                        Line2D([], [], color="#2a8", lw=1.6, ls="--",
                               label="forward simulation (calibrated ChamberSphere)")],
               loc="upper center", ncol=2, fontsize=14, frameon=False,
               bbox_to_anchor=(0.5, 0.997))
    fig.suptitle("Sanity check: data vs FORWARD-SIMULATED P-V loops, all 124 cycles "
                 "(x = volume [mL], y = pressure [mmHg]; % = pressure rel RMS)",
                 fontsize=15, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p1 = os.path.join(cy.OUT_DIR, "viz_forward_sim_all.png")
    fig.savefig(p1, dpi=95); print("wrote", p1)

    # error histogram
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errs * 100, bins=24, color="#2a8", edgecolor="white")
    ax.axvline(np.median(errs) * 100, color="k", ls="--",
               label=f"median {np.median(errs)*100:.1f}%")
    ax.set(xlabel="forward-sim pressure rel RMS [%]", ylabel="# cycles",
           title="Forward-simulation sanity check across 124 cycles")
    ax.legend()
    fig2.tight_layout()
    p2 = os.path.join(cy.OUT_DIR, "viz_forward_sim_error.png")
    fig2.savefig(p2, dpi=110); print("wrote", p2)


if __name__ == "__main__":
    main()
