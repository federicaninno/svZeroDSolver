"""Profile the calibration fit vs the shape factor n: for each fixed n, fit the
other parameters and report the median fit error. Shows whether n is a real
shape parameter (interior optimum) or a nuisance that just flattens the geometry
(monotonic improvement)."""
import glob
import os
import numpy as np
from scipy.optimize import least_squares

import calibrate_yale as cy

FREE_IDX = [1, 2, 5, 6, 7, 8]  # guccione_C, gamma_sigma_max, tau_1, tau_2, m1, m2


def fit_fixed_n(t, P, V, b_f, b_t, n_val, volume0):
    fixed = cy.data_fixed(t, P, V, volume0)
    scale = max(np.max(np.abs(P)), 1e3)

    def expand(xf):
        th = np.empty(len(cy.PARAM_NAMES))
        th[FREE_IDX] = xf
        th[0] = fixed["volume0"]; th[3] = fixed["prestress"]
        th[4] = 0.0; th[9] = n_val
        return th

    def resid(xf):
        th = expand(xf)
        return (cy.model_tau(th, t) - cy.reconstruct_tau(th, P, V, b_f, b_t)) / scale

    x0 = np.array([2e3, 3e4, 0.08, 0.18, 8.0, 8.0])
    lb = np.array([0.0, 0.0, 0.02, 0.05, 1.0, 1.0])
    ub = np.array([1e6, 1e7, 0.4, 0.6, 40.0, 40.0])
    r = least_squares(resid, x0, bounds=(lb, ub), method="trf", x_scale="jac",
                      max_nfev=2000, ftol=1e-12)
    th = expand(r.x)
    tau_obs = cy.reconstruct_tau(th, P, V, b_f, b_t)
    rms = np.sqrt(np.mean((cy.model_tau(th, t) - tau_obs) ** 2))
    return rms / max(np.ptp(tau_obs), 1.0)


def main():
    paths = sorted(glob.glob(os.path.join(cy.DATA_DIR, "*", "cav.LV.csv")))
    data = [(cy.load_cycle(p), cy.read_b(p), cy.read_unloaded(p)) for p in paths]
    print(f"{'n':>6}  {'median fit err':>15}  {'p90':>8}")
    for n_val in [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]:
        errs = []
        for (t, P, V), (b_f, b_t), vol0 in data:
            try:
                errs.append(fit_fixed_n(t, P, V, b_f, b_t, n_val, vol0))
            except Exception:
                pass
        errs = np.array(errs)
        print(f"{n_val:>6.2f}  {np.median(errs):>14.2%}  {np.percentile(errs,90):>7.2%}")


if __name__ == "__main__":
    main()
