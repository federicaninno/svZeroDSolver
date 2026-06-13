"""Generate the ChamberSphere calibration test fixture and run an identifiability
check.

Strategy (adapted from the ChamberSphere calibration work on
claude/gallant-yonath-faecaa, ported to the simplified n=1, eta=0 block):

1. Run a *forward simulation* of the chamber to obtain a realistic volume(t)
   and tau(t) trajectory over one cardiac cycle.
2. From those, reconstruct the *exact* full-state observation set
   (Pin, Qin, Pout, Qout, stress, tau, volume and the derivatives dtau_dt,
   dvolume_dt that actually enter the residuals) so that every ChamberSphere
   residual vanishes identically at the true parameters.
3. Write a point-wise calibration fixture with a 20%-perturbed start and run the
   calibrator. Recovering every parameter from full-state data demonstrates that
   all nine parameters are identifiable.

The only derivatives appearing in the residuals/Jacobian are dtau_dt (active
stress ODE) and dvolume_dt (mass conservation); both are computed exactly here,
so the fixture is consistent to machine precision regardless of solver
tolerance.
"""

import os
import json
import copy
import numpy as np

import pysvzerod

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.dirname(THIS_DIR)
OUT = os.path.join(REPO, "tests", "cases", "chamber_sphere_calibration.json")

# Ground-truth parameters (all nonzero so the relative recovery check is
# meaningful). Order matches the ParamId enum in ChamberSphere.h.
TRUE = {
    "volume0": 1.0e-4,
    "guccione_C": 1.0e3,
    "gamma_sigma_max": 1.85e5,
    "prestress": 1.0e3,
    "alpha_max": 30.0,
    "alpha_min": -30.0,
    "tsys": 0.17,
    "tdias": 0.484,
    "steepness": 0.05,
    "b_f": 8.0,
    "b_t": 3.0,
}


def passive_guccione(lam, C, b_f, b_t):
    """Guccione passive spherical wall stress (matches ChamberSphere)."""
    Ep = 0.5 * (lam ** 2 - 1.0)
    Er = 0.5 * (lam ** (-4) - 1.0)
    Q = (b_f + b_t) * Ep ** 2 + b_t * Er ** 2
    return C * np.exp(Q) * (
        0.5 * (b_f + b_t) * lam ** 2 * Ep - b_t * lam ** (-4) * Er)
PERIOD = 1.0
NUM_OBS = 200
PERTURB = 1.20  # 20% perturbed start


def activation(t, p):
    """Active-stress activation, identical to ChamberSphere::get_elastance_values
    and to the symbolic definition in scripts/ChamberSphere.yaml."""
    tc = np.mod(t, PERIOD)
    s_plus = 0.5 * (1.0 + np.tanh((tc - p["tsys"]) / p["steepness"]))
    s_minus = 0.5 * (1.0 - np.tanh((tc - p["tdias"]) / p["steepness"]))
    f = s_plus * s_minus
    act_t = p["alpha_max"] * f + p["alpha_min"] * (1.0 - f)
    return np.abs(act_t), np.maximum(act_t, 0.0)


def forward_sim():
    """Run the chamber forward simulation with the true parameters and return
    the ventricle volume(t) and tau(t) over the last cardiac cycle."""
    cfg = json.load(open(os.path.join(REPO, "tests", "cases", "chamber_sphere.json")))
    for v in cfg["vessels"]:
        if v["zero_d_element_type"] == "ChamberSphere":
            v["zero_d_element_values"] = dict(TRUE)
    cfg["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] = 2000
    res = pysvzerod.simulate(cfg)

    def series(name):
        sub = res[res["name"] == name].sort_values("time")
        return sub["time"].to_numpy(), sub["y"].to_numpy()

    tv, vol = series("volume:ventricle")
    tt, tau = series("tau:ventricle")
    # restrict to the last cycle and shift to [0, PERIOD)
    t0 = tv.max() - PERIOD
    mask = tv >= t0 - 1e-12
    return tv[mask] - t0, vol[mask], tau[mask]


def main():
    tv, vol_t, tau_t = forward_sim()

    # Sample one cycle uniformly.
    t = np.linspace(0.0, PERIOD, NUM_OBS, endpoint=False)
    volume = np.interp(t, tv, vol_t, period=PERIOD)
    tau = np.interp(t, tv, tau_t, period=PERIOD)

    p = TRUE
    V0 = p["volume0"]
    stretch = ((volume + V0) / V0) ** (1.0 / 3.0)
    CG = stretch ** 2

    # Reconstruct the consistent full state (every residual vanishes exactly).
    act, act_plus = activation(t, p)
    dtau_dt = -act * tau + p["gamma_sigma_max"] * act_plus          # residual 2
    stress = tau + passive_guccione(stretch, p["guccione_C"], p["b_f"], p["b_t"]) \
        + p["prestress"]                                            # residual 1
    Pout = stress / stretch                                          # residual 0
    Pin = Pout                                                       # residual 4

    # dvolume_dt (only constrains residual 3, which we satisfy by construction).
    dvolume_dt = np.gradient(volume, t, edge_order=2)
    Qin = dvolume_dt
    Qout = np.zeros_like(t)

    zeros = np.zeros_like(t)
    y = {
        "pressure:IN:ventricle": Pin,
        "flow:IN:ventricle": Qin,
        "pressure:ventricle:OUT": Pout,
        "flow:ventricle:OUT": Qout,
        "stress:ventricle": stress,
        "tau:ventricle": tau,
        "volume:ventricle": volume,
    }
    dy = {
        "pressure:IN:ventricle": zeros,
        "flow:IN:ventricle": zeros,
        "pressure:ventricle:OUT": zeros,
        "flow:ventricle:OUT": zeros,
        "stress:ventricle": zeros,
        "tau:ventricle": dtau_dt,
        "volume:ventricle": dvolume_dt,
    }

    names = list(TRUE.keys())
    start = {k: TRUE[k] * PERTURB for k in names}

    fixture = {
        "_comment": (
            "Full-state ChamberSphere (n=1, eta=0) calibration through the "
            "point-wise calibrator. y/dy carry every state and a 't' vector "
            "provides the observation time, so the activation/timing parameters "
            "are identifiable alongside the time-independent ones. y/dy are "
            "reconstructed from a forward simulation so all residuals vanish at "
            "_true_values. Generated by scripts/make_chamber_sphere_calibration.py."
        ),
        "_true_values": TRUE,
        "_calibrate_subset": names,
        "vessels": [
            {
                "boundary_conditions": {"inlet": "IN", "outlet": "OUT"},
                "vessel_id": 0,
                "vessel_length": 1.0,
                "vessel_name": "ventricle",
                "zero_d_element_type": "ChamberSphere",
                "zero_d_element_values": start,
                "calibrate": names,
            }
        ],
        "junctions": [],
        "y": {k: v.tolist() for k, v in y.items()},
        "dy": {k: v.tolist() for k, v in dy.items()},
        "t": t.tolist(),
        "calibration_parameters": {
            "tolerance_gradient": 1e-9,
            "tolerance_increment": 1e-13,
            "maximum_iterations": 200,
            "initial_damping_factor": 1.0,
            "cardiac_cycle_period": PERIOD,
        },
    }

    with open(OUT, "w") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {OUT} ({NUM_OBS} observations)")

    # Identifiability check: calibrate from the perturbed start and report the
    # recovered value and relative error for every parameter.
    result = pysvzerod.calibrate(copy.deepcopy(fixture))
    calibrated = result["vessels"][0]["zero_d_element_values"]
    print(f"\n{'parameter':<18}{'true':>14}{'start':>14}{'calibrated':>16}{'rel.err':>12}")
    ok = True
    for k in names:
        rel = abs(calibrated[k] - TRUE[k]) / abs(TRUE[k])
        ok = ok and rel < 1e-6
        print(f"{k:<18}{TRUE[k]:>14.6g}{start[k]:>14.6g}{calibrated[k]:>16.8g}{rel:>12.2e}")
    print(f"\nAll {len(names)} parameters recovered to rtol<1e-6: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
