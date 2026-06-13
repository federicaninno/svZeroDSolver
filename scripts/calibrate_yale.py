"""Calibrate the simplified ChamberSphere (n=1, eta=0) to the Yale LV data.

The Yale ``cav.LV.csv`` traces provide only pressure and volume; the chamber's
active stress ``tau`` is a hidden state. Following the point-wise calibration
strategy (claude/gallant-yonath-faecaa) adapted to partial observations, the
full state is reconstructed from (P, V) *as a function of the parameters*:

    stretch = (V_total / volume0)^(1/3),   CG = stretch^2          (geometry)
    stress  = P * stretch                                          (residual 0)
    tau     = stress - 4*gamma_W1*(1 - CG^-3) - prestress          (residual 1)

The only equation not satisfied by construction is the active-stress ODE
(residual 2), which becomes the calibration residual:

    r(t) = d(tau)/dt + act*tau - gamma_sigma_max*act_plus

with the activation act/act_plus identical to ChamberSphere::get_elastance_values.
All nine parameters are fit per cycle by minimizing sum_t r(t)^2 with
Levenberg-Marquardt (scipy.optimize.least_squares).

Units: the data is mmHg / mL / ms; everything is converted to SI (Pa, m^3, s).
"""

import os
import glob
import json
import numpy as np
from scipy.optimize import least_squares

MMHG_TO_PA = 133.322
ML_TO_M3 = 1.0e-6
PERIOD = 1.0  # s (SA_CL = 1000 ms)

DATA_DIR = os.path.expanduser("~/Downloads/simulations_data_yale/simulations_yale")
OUT_DIR = os.path.expanduser("~/Downloads/simulations_data_yale")

PARAM_NAMES = [
    "volume0", "guccione_C", "gamma_sigma_max", "prestress",
    "alpha_max", "alpha_min", "tsys", "tdias", "steepness",
]


def read_b(cav_path):
    """Read the Guccione exponential parameters b_f, b_t from the cycle's
    parameters.par (b_fs only multiplies shear terms, which vanish for the
    equibiaxial spherical deformation, so it is not needed)."""
    import re
    par = os.path.join(os.path.dirname(cav_path), "parameters.par")
    txt = open(par).read()
    b_f = float(re.search(r"b_f=([0-9.eE+-]+)", txt).group(1))
    b_t = float(re.search(r"b_t=([0-9.eE+-]+)", txt).group(1))
    return b_f, b_t


def passive_guccione(lam, C, b_f, b_t):
    """Passive spherical wall stress for a thin-walled incompressible sphere with
    a Guccione strain-energy W = (C/2)(exp(Q) - 1).

    Equibiaxial in-plane stretch lam (fiber f and sheet s, both tangential) and
    radial stretch lam^-2 (normal n). With Green-Lagrange strains
    E_p = (lam^2 - 1)/2 (in-plane) and E_r = (lam^-4 - 1)/2 (radial) and no shear,
        Q = (b_f + b_t) E_p^2 + b_t E_r^2.
    The mean in-plane Cauchy stress (with sigma_radial = 0 for the thin wall) is
        S = C exp(Q) [ (b_f + b_t)/2 lam^2 E_p - b_t lam^-4 E_r ].
    Only the scaling C is fitted; b_f, b_t come from the data."""
    Ep = 0.5 * (lam ** 2 - 1.0)
    Er = 0.5 * (lam ** (-4) - 1.0)
    Q = (b_f + b_t) * Ep ** 2 + b_t * Er ** 2
    return C * np.exp(Q) * (
        0.5 * (b_f + b_t) * lam ** 2 * Ep - b_t * lam ** (-4) * Er)


def load_cycle(path):
    """Return (t, P, V) in SI for the last full beat, rolled so t=0 is at
    end-diastole (maximum volume)."""
    t, P, V = [], [], []
    with open(path) as fh:
        fh.readline()  # header
        fh.readline()  # units
        for line in fh:
            p = [x.strip() for x in line.split(",")]
            try:
                t.append(float(p[0])); P.append(float(p[1])); V.append(float(p[2]))
            except (ValueError, IndexError):
                pass
    t = np.array(t); P = np.array(P); V = np.array(V)

    # last full beat [tmax-1000, tmax) ms; exclude the wrap endpoint (same phase
    # as the start) so the time vector has no duplicated point.
    tmax = t.max()
    m = (t >= tmax - 1000.0 - 1e-9) & (t < tmax - 1e-9)
    t, P, V = t[m], P[m], V[m]
    if len(t) < 50:
        raise RuntimeError(f"too few points in last beat for {path}")

    # roll so t=0 at end-diastole (max V), drop the duplicated wrap point
    i0 = int(np.argmax(V))
    t = np.concatenate([t[i0:], t[:i0] + 1000.0]) - t[i0]
    P = np.concatenate([P[i0:], P[:i0]])
    V = np.concatenate([V[i0:], V[:i0]])

    return t / 1000.0, P * MMHG_TO_PA, V * ML_TO_M3


def activation(t, alpha_max, alpha_min, tsys, tdias, steepness):
    tc = np.mod(t, PERIOD)
    s_plus = 0.5 * (1.0 + np.tanh((tc - tsys) / steepness))
    s_minus = 0.5 * (1.0 - np.tanh((tc - tdias) / steepness))
    f = s_plus * s_minus
    act_t = alpha_max * f + alpha_min * (1.0 - f)
    return np.abs(act_t), np.maximum(act_t, 0.0)


def reconstruct_tau(theta, P, V, b_f, b_t):
    """Active stress tau reconstructed from (P, V) via residuals 0 and 1, with a
    Guccione passive law (scaling C = theta[1], b parameters from the data)."""
    volume0, C, _, prestress = theta[:4]
    stretch = (V / volume0) ** (1.0 / 3.0)
    stress = P * stretch
    tau = stress - passive_guccione(stretch, C, b_f, b_t) - prestress
    return tau


def integrate_tau(theta, t, tau0):
    """Predict tau by integrating the active-stress ODE (residual 2),
    dtau/dt = -act*tau + gamma_sigma_max*act_plus, with a stable backward-Euler
    step on the uniform time grid."""
    gamma_sigma_max = theta[2]
    act, act_plus = activation(t, *theta[4:9])
    dt = np.diff(t)
    tau = np.empty_like(t)
    tau[0] = tau0
    for i in range(len(t) - 1):
        h = dt[i]
        tau[i + 1] = (tau[i] + h * gamma_sigma_max * act_plus[i + 1]) / (
            1.0 + h * act[i + 1]
        )
    return tau


def residual(theta, t, P, V, scale, b_f, b_t):
    # tau reconstructed from the data (depends on the material parameters) ...
    tau_obs = reconstruct_tau(theta, P, V, b_f, b_t)
    # ... must be consistent with the active-stress ODE (activation parameters).
    tau_pred = integrate_tau(theta, t, tau_obs[0])
    return (tau_pred - tau_obs) / scale


# Parameters not calibrated:
#  - steepness: numerical smoothing of the activation indicator, kept constant at
#    the model's canonical value (tests/cases/chamber_sphere.json).
#  - volume0, prestress: the unloaded reference volume and the resting passive
#    stress are not identifiable from a single loaded P-V loop (a profile scan
#    over volume0 is monotonic). They are instead read directly from the data:
#    volume0 = minimum volume (ESV, so stretch >= 1 everywhere), prestress =
#    minimum transmural pressure (the resting stress).
STEEPNESS = 0.005
NOT_CALIBRATED = ("volume0", "prestress", "steepness")
DATA_DERIVED = ("volume0", "prestress")
FREE = [i for i, nm in enumerate(PARAM_NAMES) if nm not in NOT_CALIBRATED]  # 6


def data_fixed(t, P, V):
    """Pick volume0 and prestress directly from the data (plus constant steepness)."""
    return {"volume0": float(V.min()), "prestress": float(P.min()),
            "steepness": STEEPNESS}


def _expand(theta_free, fixed):
    """Build the full 9-parameter vector from the free params and the fixed ones."""
    theta = np.empty(len(PARAM_NAMES))
    theta[FREE] = theta_free
    for nm, val in fixed.items():
        theta[PARAM_NAMES.index(nm)] = val
    return theta


def residual_free(theta_free, t, P, V, scale, fixed, b_f, b_t):
    return residual(_expand(theta_free, fixed), t, P, V, scale, b_f, b_t)


def calibrate_cycle(t, P, V, b_f, b_t):
    fixed = data_fixed(t, P, V)
    # start guess (SI) for the 6 free params: guccione_C, gamma_sigma_max,
    # alpha_max, alpha_min, tsys, tdias
    theta0 = np.array([2.0e3, 2.0e4, 25.0, -25.0, 0.03, 0.32])
    lb = np.array([0.0, 0.0, 1.0, -200.0, 0.0, 0.0])
    ub = np.array([1e6, 1e7, 200.0, -1.0, 0.6, 0.9])
    theta0 = np.clip(theta0, lb + 1e-12, ub - 1e-12)

    # residual scale ~ characteristic stress rate, keeps the cost well-scaled
    scale = max(np.max(np.abs(P)) * 1.0, 1.0e3)

    res_free = least_squares(
        residual_free, theta0, args=(t, P, V, scale, fixed, b_f, b_t),
        bounds=(lb, ub), method="trf", x_scale="jac", max_nfev=2000,
        ftol=1e-12, xtol=1e-12,
    )

    full_x = _expand(res_free.x, fixed)
    tau_obs = reconstruct_tau(full_x, P, V, b_f, b_t)
    tau_pred = integrate_tau(full_x, t, tau_obs[0])
    rms = np.sqrt(np.mean((tau_pred - tau_obs) ** 2))  # Pa
    tau_amp = np.ptp(tau_obs)

    # Per-parameter relative standard error from the Jacobian covariance, a
    # local identifiability measure (large -> the data does not constrain that
    # parameter). A parameter pinned at a bound is flagged separately. These are
    # computed for the free parameters and scattered back into a 9-vector;
    # the not-calibrated parameters get NaN.
    res = res_free
    m, n = res.jac.shape
    dof = max(m - n, 1)
    cov = np.linalg.pinv(res.jac.T @ res.jac) * (2.0 * res.cost / dof)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))  # free-parameter units
    rel_se = np.full(len(PARAM_NAMES), np.nan)
    rel_se[FREE] = se / np.maximum(np.abs(res_free.x), 1e-30)
    at_bound = np.zeros(len(PARAM_NAMES), bool)
    at_bound[FREE] = (res_free.x <= lb + 1e-6 * (ub - lb)) | (
        res_free.x >= ub - 1e-6 * (ub - lb))
    return full_x, rms, tau_amp, rel_se, at_bound


def main():
    cycles = sorted(glob.glob(os.path.join(DATA_DIR, "*", "cav.LV.csv")))
    print(f"found {len(cycles)} cycles")
    names, thetas, rmss, amps, relses, bounds = [], [], [], [], [], []
    for path in cycles:
        name = os.path.basename(os.path.dirname(path))
        try:
            t, P, V = load_cycle(path)
            b_f, b_t = read_b(path)
            theta, rms, tau_amp, rel_se, at_bound = calibrate_cycle(t, P, V, b_f, b_t)
            names.append(name); thetas.append(theta); rmss.append(rms)
            amps.append(tau_amp); relses.append(rel_se); bounds.append(at_bound)
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    thetas = np.array(thetas); rmss = np.array(rmss); amps = np.array(amps)
    relses = np.array(relses); bounds = np.array(bounds)
    rel_fit = rmss / np.maximum(amps, 1.0)

    # write per-cycle CSV
    import csv
    out_csv = os.path.join(OUT_DIR, "chamber_sphere_calibration_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cycle", *PARAM_NAMES, "rms_resid_Pa", "tau_amplitude_Pa",
                    "rel_fit_error"])
        for i, nm in enumerate(names):
            w.writerow([nm] + [f"{x:.6g}" for x in thetas[i]]
                       + [f"{rmss[i]:.6g}", f"{amps[i]:.6g}", f"{rel_fit[i]:.4g}"])
    print(f"wrote {out_csv} ({len(names)} cycles)")

    # structured summary: value distribution + identifiability per parameter
    units = ["m^3", "Pa", "Pa", "Pa", "1/s", "1/s", "s", "s", "s"]
    print(f"\n{'parameter':<16}{'unit':>6}{'median':>13}{'p10':>13}{'p90':>13}"
          f"{'rel.SE':>10}{'%@bound':>9}")
    print("-" * 80)
    for i, nm in enumerate(PARAM_NAMES):
        col = thetas[:, i]
        if nm in DATA_DERIVED:
            print(f"{nm:<16}{units[i]:>6}{np.median(col):>13.4g}"
                  f"{np.percentile(col,10):>13.4g}{np.percentile(col,90):>13.4g}"
                  f"{'-':>10}{'-':>9}  from data")
            continue
        if nm not in [PARAM_NAMES[k] for k in FREE]:
            print(f"{nm:<16}{units[i]:>6}{np.median(col):>13.4g}"
                  f"{'-':>13}{'-':>13}{'-':>10}{'-':>9}  fixed")
            continue
        med_relse = np.nanmedian(relses[:, i])
        pct_bound = 100.0 * np.mean(bounds[:, i])
        ident = "well" if med_relse < 0.1 and pct_bound < 20 else (
            "weak" if med_relse < 1.0 and pct_bound < 60 else "poor")
        print(f"{nm:<16}{units[i]:>6}{np.median(col):>13.4g}"
              f"{np.percentile(col,10):>13.4g}{np.percentile(col,90):>13.4g}"
              f"{med_relse:>10.2g}{pct_bound:>8.0f}%  {ident}")
    print("-" * 80)
    print(f"fit quality (tau_pred vs tau_obs over the cycle):")
    print(f"  median relative error {np.median(rel_fit):.2%}, "
          f"p90 {np.percentile(rel_fit,90):.2%}, "
          f"worst {rel_fit.max():.2%}")
    print(f"  median RMS {np.median(rmss):.4g} Pa, "
          f"median tau amplitude {np.median(amps):.4g} Pa")
    print("\nidentifiability key: rel.SE = median relative std error of the "
          "parameter\n  (Jacobian covariance); %@bound = fraction of cycles "
          "pinned at a bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
