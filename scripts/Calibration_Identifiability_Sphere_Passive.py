"""
Calibration and, optionally, identifiability analysis for the passive
spherical chamber model of Caruel et al. (2014), "Dimensional reductions
of a cardiac model for effective validation and calibration," Biomech
Model Mechanobiol. At its core this is a joint least-squares fit of the
model below; when RUN_IDENTIFIABILITY is on, it goes further and
characterizes how well each fitted parameter is actually constrained by
the data, via profile likelihood and a local Fisher Information Matrix.

Model (spherical chamber, inertia dropped, no active tension, tau=0):

    lam    = (V/V0)^(1/(3n)),  C = lam^2
    stress = 4*(1-C^-3)*(dWe_dJ1 + C*dWe_dJ2) + 2*dWe_dJ4 + visc(eta, Cdot)
    P_model(t) = gamma * lam * stress / C

NH:  dWe_dJ1 = W1, dWe_dJ2 = W2, dWe_dJ4 = 0
HO:  dWe_dJ1 = 0.5*a*exp(b*(J1-3)), dWe_dJ4 from fiber/sheet terms (a4f,b4f,a4s,b4s)

-----------------------------------------------------------------------------
WHAT YOU CAN CONFIGURE (see "USER SETTINGS" below)
-----------------------------------------------------------------------------
FREE_PARAMS is a list of strings. Each entry is EXACTLY ONE optimization
variable (one profile-likelihood curve), and is either:
  - a single raw parameter name, e.g. "n", "b"; or
  - a tied product "gamma*<partner>", where <partner> is "W1" or "eta" (NH),
    or "a" or "eta" (HO) -- see ALLOWED below.

Every raw parameter that does not appear in ANY entry stays FIXED at its
BASE_PARAMS value. Nothing is decomposed automatically: FREE_PARAMS =
["gamma*a", "n"] gives exactly two profile curves -- "gamma*a" and "n" --
never separate "gamma" and "a" curves (list them separately instead, e.g.
FREE_PARAMS = ["gamma", "a", "n"], a different, 3-DOF analysis).

Only "gamma*W1" / "gamma*eta" (NH) and "gamma*a" / "gamma*eta" (HO) are
accepted as tied products -- anything else (e.g. "b*n", "a*b") raises an
error at startup. This is deliberate: gamma is a pure linear prefactor of
the whole stress, and W1/a/eta are each the pure linear coefficient of the
one additive term they multiply, so "gamma*<that coefficient>" is the one
kind of product for which "only the product matters" is a clean statement
about the model. Any other pairing sits inside a nonlinear term (b, b4f,
b4s are inside exp(); n is inside an exponent of V/V0 itself) and would be
numerically computable but wouldn't mean what "identifiability of the
product" is supposed to mean.

ONE tied product active (e.g. just "gamma*a"): gamma and its partner move
TOGETHER, each scaled by the same multiplicative step from their own
reference values (BASE_PARAMS, or an INIT_OVERRIDES entry). gamma is still
a real, separately reconstructed number, so it correctly keeps scaling any
OTHER fixed term it also multiplies (fixed nonzero fiber/viscous terms).

TWO tied products active at once (e.g. "gamma*W1" and "gamma*eta" for NH,
matching optimization_HO_products_...py's product-mode calibration): gamma
can no longer be reconstructed as one standalone number shared by both, so
it is eliminated from the model entirely -- each active product directly
supplies the coefficient of its own additive term (isotropic / viscous).
This requires every OTHER term gamma would also touch to be fixed at
exactly zero (HO fiber terms a4f/a4s; NH's W2) -- validated at startup,
since there would otherwise be no way to know what gamma should multiply
them by.

Examples:
    FREE_PARAMS = ["gamma", "n"]              # original geometry-only analysis
    FREE_PARAMS = ["gamma*a", "n"]             # HO: one product; b, fibers, eta fixed
    FREE_PARAMS = ["gamma*a", "gamma*eta", "n"]  # HO: two products at once (fibers must be 0)
    FREE_PARAMS = ["gamma", "a", "b", "n"]     # HO: gamma, a, b, n each profiled separately
    FREE_PARAMS = ["gamma*W1", "gamma*eta", "n"]  # NH: matches optimization_HO_products_...py

-----------------------------------------------------------------------------
METHOD
-----------------------------------------------------------------------------
Profile likelihood (the standard, rigorous definition -- e.g. Raue et al.
2009): for each FREE_PARAMS entry, fix it at a grid of values and
re-optimize (least_squares) over every OTHER entry (the nuisance parameters)
to obtain the conditional minimum RMSE. This is used uniformly for any
number of free entries (no combinatorial N-D grid), with warm-starting from
the previous grid point for stability/speed. A "confidence region" is
RMSE <= best + DELTA_MMHG; an entry is tagged IDENTIFIABLE if that region
closes strictly inside the scanned grid.

A local Fisher Information Matrix (Gauss-Newton J^T J) at the joint optimum
gives eigenvalues/condition number over the full set of FREE_PARAMS entries.

Run:            python Calibration_Identifiability_Sphere_Passive.py
Re-plot only:   python Calibration_Identifiability_Sphere_Passive.py --plot-only
(re-plot reads the CSVs saved by a previous full run -- no data loading or
optimization -- and regenerates the clean "profiles only" figure.)
"""

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

MATERIAL_MODEL = "HO"  # "NH" or "HO"

# JSON_PATH = (
#     "/Users/Postdoc/Reduced_order_modeling/svZeroDSolver_Sphere_opt/svZeroDSolver/"
#     "build/chamber_sphere_NH_calibr_E_100kPa_4calibration_notaunovelonostressnoradius.json"
# )

JSON_PATH = (
    "/Users/Postdoc/Reduced_order_modeling/svZeroDSolver_Sphere_opt/svZeroDSolver/"
    "build/chamber_sphere_HO_passive_calibr_realfibers_correctParams_Nikou.json"
)

OUT_DIR = "/Users/Postdoc/Reduced_order_modeling/svZeroDSolver_Sphere_opt/svZeroDSolver/scripts/identifiability_passive_HO_gammaxa_n_bfixed_fibers0_eta0"

# Raw parameter vocabulary per material model, and their fixed/reference
# values. Every raw parameter must appear here regardless of whether it ends
# up free or fixed -- FREE_PARAMS below decides which ones are optimized.
NH_BASE_PARAMS = {"gamma": 0.5, "n": 1.0, "W1": 1.69e5, "W2": 0.0, "eta": 0.0}
HO_BASE_PARAMS = {
    "gamma": 0.5, "n": 1.0,
    "a": 14720.0, "b": 33.39,
    "a4f": 0.0, "b4f": 0.0,
    "a4s": 0.0, "b4s": 0.0,
    "eta": 0.0,
}

BASE_PARAMS = dict(NH_BASE_PARAMS if MATERIAL_MODEL == "NH" else HO_BASE_PARAMS)

# What to run identifiability on. Each entry is ONE optimization variable and
# ONE profile-likelihood curve: a single raw parameter name, or a tied
# product "gamma*<partner>" (see docstring above for which partners are
# supported, and what freeing TWO such products at once requires). Anything
# not named here (in any entry) stays fixed at its BASE_PARAMS value.
FREE_PARAMS = ["gamma*a", "n"]

# Reference/initial value for a free parameter, when it must differ from its
# BASE_PARAMS entry (log-space optimization needs a strictly positive start,
# so a parameter whose passive default is 0, e.g. eta, needs an override here
# if you free it). Parameters not listed here just use BASE_PARAMS.
INIT_OVERRIDES = {
    "eta": 500.0,  # gives gamma*eta an initial guess of ~250, order-of-magnitude
                   # sensitive here (product mode's cost surface is quite sloppy) --
                   # see optimization_HO_products_...py's own NH_OPT_INIT for reference
}

# True: run the full identifiability analysis (profile likelihood + FIM +
# confidence intervals) for every FREE_PARAMS entry, as usual.
# False: only run the joint calibration (least_squares best fit), print/plot
# it, and stop there -- skips all profiling, FIM, and per-parameter CSVs.
RUN_IDENTIFIABILITY = True

# confidence threshold: RMSE within (best + DELTA_MMHG)
DELTA_MMHG = 0.5

# half-decade scan range around the optimum, in each direction (log10 units)
GRID_HALF_WIDTH_LOG10 = 0.6  # ~ factor of 4 either side
GRID_N = 80  # per-entry scan resolution

# Tag identifying this run's config in output filenames (auto if None)
RUN_TAG = None

# ============================================================
# RAW PARAMETER REGISTRY / GROUP PARSING / VALIDATION
# ============================================================

RAW_PARAM_REGISTRY = {
    "NH": ["gamma", "n", "W1", "W2", "eta"],
    "HO": ["gamma", "n", "a", "b", "a4f", "b4f", "a4s", "b4s", "eta"],
}

# Which raw parameter is the linear coefficient of which additive stress
# term, per model -- these are the only partners "gamma*<partner>" may tie
# to (see module docstring for why).
GAMMA_TERM_PARTNERS = {
    "NH": {"W1": "isotropic", "eta": "viscous"},
    "HO": {"a": "isotropic", "eta": "viscous"},
}

MMHG_TO_BARYE = 1333.22

# FREE_PARAMS, parsed once: one raw-parameter-name list ("group") per entry.
FREE_GROUPS = [entry.split("*") for entry in FREE_PARAMS]

# Number of tied gamma products ("gamma*<partner>" entries) requested at
# once. With 2 or more, gamma cannot be reconstructed as one standalone
# number shared by all of them -- see model_pressure_product_mode().
N_GAMMA_PRODUCTS = sum(1 for g in FREE_GROUPS if len(g) == 2 and "gamma" in g)
PRODUCT_MODE = N_GAMMA_PRODUCTS >= 2

# {term_key: theta index}, only meaningful when PRODUCT_MODE is True.
GAMMA_TERM_MAP = {}
if PRODUCT_MODE:
    _partners = GAMMA_TERM_PARTNERS[MATERIAL_MODEL]
    for _idx, _group in enumerate(FREE_GROUPS):
        if len(_group) == 2 and "gamma" in _group:
            _partner = _group[0] if _group[1] == "gamma" else _group[1]
            GAMMA_TERM_MAP[_partners[_partner]] = _idx


def effective_base(p):
    """Reference value a free parameter starts from (and is scaled relative
    to); overridable for parameters whose fixed default is 0."""
    return INIT_OVERRIDES.get(p, BASE_PARAMS[p])


def validate_config():
    if MATERIAL_MODEL not in RAW_PARAM_REGISTRY:
        raise ValueError(f"MATERIAL_MODEL must be 'NH' or 'HO', got '{MATERIAL_MODEL}'.")
    valid = RAW_PARAM_REGISTRY[MATERIAL_MODEL]
    partners = GAMMA_TERM_PARTNERS[MATERIAL_MODEL]

    missing = [p for p in valid if p not in BASE_PARAMS]
    if missing:
        raise ValueError(f"BASE_PARAMS is missing required {MATERIAL_MODEL} parameters: {missing}")

    if len(set(FREE_PARAMS)) != len(FREE_PARAMS):
        raise ValueError(f"FREE_PARAMS has duplicate entries: {FREE_PARAMS}")

    seen = set()  # raw params (other than "gamma" itself) allowed in at most one entry
    for label, group in zip(FREE_PARAMS, FREE_GROUPS):
        if len(group) > 1:
            if len(group) != 2 or "gamma" not in group:
                raise ValueError(
                    f"FREE_PARAMS entry '{label}' is not supported: only 'gamma*<partner>' products "
                    f"are allowed, never a product of two non-gamma parameters."
                )
            partner = group[0] if group[1] == "gamma" else group[1]
            if partner not in partners:
                allowed = ", ".join(f"gamma*{p}" for p in partners)
                raise ValueError(
                    f"FREE_PARAMS entry '{label}' is not a supported tied product for {MATERIAL_MODEL}. "
                    f"Allowed products: {allowed} (see module docstring for why); "
                    f"list parameters separately (e.g. {list(group)}) if you want each profiled on its own."
                )
            if partner in seen:
                raise ValueError(f"Raw parameter '{partner}' appears in more than one FREE_PARAMS entry.")
            seen.add(partner)
            if effective_base(partner) <= 0.0:
                raise ValueError(
                    f"Reference value for free parameter '{partner}' (in '{label}') must be > 0 "
                    f"(log-space optimization). Add an INIT_OVERRIDES entry for it."
                )
        else:
            p = group[0]
            if p not in valid:
                raise ValueError(f"FREE_PARAMS entry '{label}' references '{p}', not valid for {MATERIAL_MODEL}.")
            if p in seen:
                raise ValueError(f"Raw parameter '{p}' appears in more than one FREE_PARAMS entry.")
            seen.add(p)
            if effective_base(p) <= 0.0:
                raise ValueError(
                    f"Reference value for free parameter '{p}' (in '{label}') must be > 0 "
                    f"(log-space optimization). Add an INIT_OVERRIDES entry for it."
                )

    if N_GAMMA_PRODUCTS > 0 and ["gamma"] in FREE_GROUPS:
        raise ValueError(
            "FREE_PARAMS cannot include a bare 'gamma' entry together with a tied 'gamma*<partner>' "
            "product -- gamma is not a standalone value once any product ties it to another parameter."
        )

    if PRODUCT_MODE:
        if MATERIAL_MODEL == "HO" and (BASE_PARAMS["a4f"] != 0.0 or BASE_PARAMS["a4s"] != 0.0):
            raise ValueError(
                "Two or more tied gamma products (e.g. gamma*a and gamma*eta) are free at once, so "
                "gamma is never reconstructed as a standalone number -- but the fiber terms (a4f, a4s) "
                "are still fixed at a nonzero value, and gamma would need to multiply them too. Set "
                "a4f = a4s = 0.0 in HO_BASE_PARAMS, or free only one gamma product at a time."
            )
        if MATERIAL_MODEL == "NH" and BASE_PARAMS["W2"] != 0.0:
            raise ValueError(
                "gamma*W1 and gamma*eta are both free at once, so gamma is never reconstructed as a "
                "standalone number -- but W2 is still fixed at a nonzero value, and gamma would need "
                "to multiply it too. Set W2 = 0.0 in NH_BASE_PARAMS."
            )


def run_tag():
    if RUN_TAG:
        return RUN_TAG
    return "_".join(FREE_PARAMS).replace("*", "x")


# ============================================================
# PHYSICS (Caruel et al. 2014 spherical chamber model, passive: tau=0)
# ============================================================

def safe_exp(z, zmax=80.0):
    return np.exp(np.clip(z, -zmax, zmax))


def material_response(C, Cdot, model, params):
    if model == "NH":
        dWe_dJ1 = params["W1"]
        dWe_dJ2 = params["W2"]
        dWe_dJ4 = np.zeros_like(C)
        visc = 2.0 * params["eta"] * Cdot * (1.0 - 2.0 * C ** -6)
    elif model == "HO":
        J1 = (1.0 / C ** 2) + 2.0 * C
        dWe_dJ1 = 0.5 * params["a"] * safe_exp(params["b"] * (J1 - 3.0))
        dWe_dJ2 = np.zeros_like(C)
        macaulay = np.maximum(C - 1.0, 0.0)
        dWe_dJ4 = (
            params["a4f"] * macaulay * safe_exp(params["b4f"] * macaulay ** 2)
            + params["a4s"] * macaulay * safe_exp(params["b4s"] * macaulay ** 2)
        )
        visc = params["eta"] * Cdot * (1.0 + 2.0 * C ** -6)
    else:
        raise ValueError(model)
    return dWe_dJ1, dWe_dJ2, dWe_dJ4, visc


def spherical_stress(C, Cdot, model, params):
    dWe_dJ1, dWe_dJ2, dWe_dJ4, visc = material_response(C, Cdot, model, params)
    return 4.0 * (1.0 - C ** -3) * (dWe_dJ1 + C * dWe_dJ2) + 2.0 * dWe_dJ4 + visc


def model_pressure(params, model, case):
    """P_model(t) = traction / C. Standalone-gamma path (0 or 1 tied gamma
    product active): gamma is a real reconstructed number."""
    n = params["n"]
    lam = case["Vratio"] ** (1.0 / (3.0 * n))
    C = lam ** 2
    lam_dot = lam * (1.0 / (3.0 * n)) * (case["dvolume"] / case["V_safe"])
    Cdot = 2.0 * lam * lam_dot
    stress = spherical_stress(C, Cdot, model, params)
    traction = params["gamma"] * lam * stress
    return traction / C


def model_pressure_product_mode(theta, model, case, singleton_params):
    """P_model(t) when 2+ tied gamma products are free simultaneously: gamma
    is never reconstructed. Each active product directly supplies the
    coefficient of its own additive term (isotropic / viscous); any other
    term gamma would also touch must be exactly 0 (validate_config checks
    this)."""
    n = singleton_params["n"]
    lam = case["Vratio"] ** (1.0 / (3.0 * n))
    C = lam ** 2
    lam_dot = lam * (1.0 / (3.0 * n)) * (case["dvolume"] / case["V_safe"])
    Cdot = 2.0 * lam * lam_dot

    def term_value(term_key, base_partner):
        if term_key in GAMMA_TERM_MAP:
            return float(safe_exp(theta[GAMMA_TERM_MAP[term_key]]))
        return BASE_PARAMS["gamma"] * BASE_PARAMS[base_partner]

    g_viscous = term_value("viscous", "eta")
    if model == "NH":
        g_isotropic = term_value("isotropic", "W1")
        stress_over_lam = (
            4.0 * (1.0 - C ** -3) * g_isotropic
            + 2.0 * g_viscous * Cdot * (1.0 - 2.0 * C ** -6)
        )
    elif model == "HO":
        J1 = (1.0 / C ** 2) + 2.0 * C
        g_isotropic = term_value("isotropic", "a")
        stress_over_lam = (
            4.0 * (1.0 - C ** -3) * 0.5 * g_isotropic * safe_exp(singleton_params["b"] * (J1 - 3.0))
            + g_viscous * Cdot * (1.0 + 2.0 * C ** -6)
        )
    else:
        raise ValueError(model)

    traction = lam * stress_over_lam
    return traction / C


def load_case(json_path):
    d = json.load(open(json_path))["y"]
    time = np.asarray(d["time:ventricle"], float)
    volume = np.asarray(d["volume:ventricle"], float)
    Pout = None
    for k in ["pressure:ventricle:OUT", "pressure:IN:ventricle", "pressure:ventricle"]:
        if k in d:
            Pout = np.asarray(d[k], float)
            break
    if Pout is None:
        raise KeyError(f"No pressure key found in {json_path}")
    dy = d.get("dy", d.get("ydot", None))
    if dy is not None and "volume:ventricle" in dy:
        dvolume = np.asarray(dy["volume:ventricle"], float)
    else:
        dvolume = np.gradient(volume, time)
    V_safe = np.maximum(volume, 1e-12)
    V0 = V_safe[0]
    Vratio = V_safe / V0
    return dict(time=time, Pout=Pout, volume=volume, dvolume=dvolume, V_safe=V_safe, V0=V0, Vratio=Vratio)


# ============================================================
# THETA (one entry per FREE_PARAMS group) <-> FULL RAW PARAMS
#
# Uniform convention: theta[k] = log(reported axis value) for EVERY entry,
# whether it's a plain parameter, a single tied product, or a product-mode
# term. For a singleton ["p"], the axis value IS the parameter. For a tied
# pair ["gamma","X"] (only meaningful when PRODUCT_MODE is False, i.e. this
# is the sole active gamma product), the axis value is gamma*X; it is split
# back into individual gamma, X values via a symmetric multiplicative step
# from their own references, so gamma stays a real usable number.
# ============================================================

def singleton_params_from_theta(theta):
    """Raw params coming from non-tied (singleton) FREE_GROUPS entries,
    plus everything else fixed at BASE_PARAMS. This is all PRODUCT_MODE
    needs (n, and b for HO); it's also the starting point for the
    standalone-gamma path below."""
    params = dict(BASE_PARAMS)
    for group, t in zip(FREE_GROUPS, theta):
        if len(group) == 1:
            params[group[0]] = float(safe_exp(t))
    return params


def params_from_theta(theta):
    """Full raw-parameter dict for the standalone-gamma path (0 or 1 tied
    gamma product active)."""
    params = singleton_params_from_theta(theta)
    for group, t in zip(FREE_GROUPS, theta):
        if len(group) == 2:
            value = float(safe_exp(t))
            base_prod = float(np.prod([effective_base(p) for p in group]))
            mult = (value / base_prod) ** (1.0 / len(group))
            for p in group:
                params[p] = effective_base(p) * mult
    return params


def model_pressure_from_theta(theta, model, case):
    if PRODUCT_MODE:
        return model_pressure_product_mode(theta, model, case, singleton_params_from_theta(theta))
    return model_pressure(params_from_theta(theta), model, case)


def axis_value(theta_k):
    """The reported value of any FREE_PARAMS entry is simply exp(theta_k)
    under the uniform convention above."""
    return float(safe_exp(theta_k))


def residual(theta, model, case):
    P_model = model_pressure_from_theta(theta, model, case)
    return (P_model - case["Pout"]) / MMHG_TO_BARYE


def interval(grid, prof, thr):
    ok = grid[prof <= thr]
    return (float(ok.min()), float(ok.max())) if ok.size else (np.nan, np.nan)


def classify_profile(grid, prof, thr, rmse_hat):
    """Confidence interval plus a verdict that tells apart two very
    different reasons the interval can come back empty: the parameter is
    genuinely unconstrained (RMSE stays above thr everywhere scanned), vs.
    it's constrained so tightly that its peak is narrower than a single
    grid step -- rmse_hat itself is comfortably below thr, but no discrete
    grid point happens to land close enough to the optimum to also be. The
    latter is NOT "not constrained"; if anything it's the opposite."""
    lo, hi = interval(grid, prof, thr)
    if np.isfinite(lo):
        tag = "IDENTIFIABLE" if (hi < grid[-1] and lo > grid[0]) else "NOT constrained within grid"
        return lo, hi, tag
    if rmse_hat < thr:
        return lo, hi, "IDENTIFIABLE (narrower than grid resolution)"
    return lo, hi, "NOT constrained within grid"


# ============================================================
# PROFILE LIKELIHOOD -- one curve per FREE_PARAMS entry
# ============================================================

def profile_one_entry(k, theta_hat, model, case):
    """Rigorous profile likelihood for FREE_GROUPS[k]: scan its value,
    re-optimizing over every other entry (nuisance) at each point."""
    n_entries = len(FREE_GROUPS)
    nuisance_idx = [i for i in range(n_entries) if i != k]

    axis_best = axis_value(theta_hat[k])
    axis_grid = np.geomspace(axis_best * 10 ** -GRID_HALF_WIDTH_LOG10,
                              axis_best * 10 ** GRID_HALF_WIDTH_LOG10, GRID_N)

    prof_rmse = np.empty(GRID_N)
    samples = [None] * GRID_N

    theta_hat_nuisance = theta_hat[nuisance_idx].copy()

    def solve_at(i, theta_nuisance_guess):
        theta_val = float(np.log(axis_grid[i]))

        def resid_nuisance(tn, theta_val=theta_val):
            theta_full = np.empty(n_entries)
            theta_full[k] = theta_val
            if nuisance_idx:
                theta_full[nuisance_idx] = tn
            return residual(theta_full, model, case)

        theta_full = np.empty(n_entries)
        theta_full[k] = theta_val
        if nuisance_idx:
            # Try both the running warm-start and a fresh restart from the
            # joint optimum's own nuisance values, keep whichever is better.
            # Warm-starting alone can get stuck at a bad local sub-solve
            # partway through the sweep (not just at the first point of a
            # direction) -- and once stuck, every later point in the chain
            # inherits that bad guess. Giving every single point a chance to
            # reset to a point already known to fit well prevents that.
            guesses = [theta_nuisance_guess]
            if not np.allclose(theta_nuisance_guess, theta_hat_nuisance):
                guesses.append(theta_hat_nuisance)
            best_sol = None
            for guess in guesses:
                sol = least_squares(resid_nuisance, guess, method="lm", max_nfev=2000)
                if best_sol is None or np.linalg.norm(sol.fun) < np.linalg.norm(best_sol.fun):
                    best_sol = sol
            theta_full[nuisance_idx] = best_sol.x
            r, next_guess = best_sol.fun, best_sol.x
        else:
            r, next_guess = resid_nuisance(np.array([])), theta_nuisance_guess
        return theta_full, float(np.sqrt(np.mean(r ** 2))), next_guess

    # Sweep outward from the grid point closest to the true joint optimum, in
    # both directions, each restarted from theta_hat's own nuisance values.
    center = int(np.argmin(np.abs(np.log(axis_grid / axis_best))))
    for start, stop, step in [(center, GRID_N, 1), (center - 1, -1, -1)]:
        theta_nuisance = theta_hat_nuisance.copy()
        for i in range(start, stop, step):
            theta_full, prof_rmse[i], theta_nuisance = solve_at(i, theta_nuisance)
            samples[i] = (theta_full.copy(), prof_rmse[i])

    return axis_grid, prof_rmse, samples


# ============================================================
# FISHER INFORMATION MATRIX (local, at joint optimum)
# ============================================================

def local_fim(theta_hat, model, case, sigma2):
    h = 1e-4
    N = len(theta_hat)
    J = np.zeros((len(case["time"]), N))
    for k in range(N):
        tp = theta_hat.copy(); tp[k] += h
        tm = theta_hat.copy(); tm[k] -= h
        J[:, k] = (residual(tp, model, case) - residual(tm, model, case)) / (2 * h)
    FIM = J.T @ J / sigma2
    eigvals, eigvecs = np.linalg.eigh(FIM)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis():
    validate_config()
    tag = run_tag()
    model = MATERIAL_MODEL

    print(f"\n{'='*70}\n{model} passive identifiability -- free: {FREE_PARAMS}\n{'='*70}")
    if PRODUCT_MODE:
        print(f"PRODUCT_MODE: {N_GAMMA_PRODUCTS} tied gamma products free at once -- "
              f"gamma is not reconstructed as a standalone number.")
    case = load_case(JSON_PATH)
    p_range_mmHg = float((case["Pout"].max() - case["Pout"].min()) / MMHG_TO_BARYE)
    flat_free = [p for g in FREE_GROUPS for p in g]
    fixed_params = {p: v for p, v in BASE_PARAMS.items() if p not in flat_free}
    print(f"Loaded {JSON_PATH.split('/')[-1]}: N={len(case['time'])}, P range={p_range_mmHg:.2f} mmHg")
    print(f"Free parameters (one DOF each): {FREE_PARAMS}")
    print(f"Fixed parameters: {fixed_params}")

    # theta0[k] = log(reported axis value at the reference point): the
    # parameter's own base value (singleton), or the product of both
    # partners' base values (tied pair) -- consistent whether or not
    # PRODUCT_MODE ends up splitting that pair back into individual values.
    theta0 = np.array([np.log(np.prod([effective_base(p) for p in g])) for g in FREE_GROUPS])
    sol = least_squares(residual, theta0, args=(model, case), method="lm", max_nfev=4000)
    theta_hat = sol.x
    rmse_hat = float(np.sqrt(np.mean(sol.fun ** 2)))
    print(f"\nBest fit ({100*rmse_hat/p_range_mmHg:.2f}% of range, RMSE={rmse_hat:.4f} mmHg):")
    for k, (label, group) in enumerate(zip(FREE_PARAMS, FREE_GROUPS)):
        value = axis_value(theta_hat[k])
        if len(group) == 1:
            print(f"  {label:>10s} = {value:.6g}")
        elif PRODUCT_MODE:
            print(f"  {label:>10s} = {value:.6g}  (gamma not separately reconstructed)")
        else:
            base_prod = float(np.prod([effective_base(p) for p in group]))
            mult = (value / base_prod) ** (1.0 / len(group))
            parts = ", ".join(f"{p}={effective_base(p) * mult:.6g}" for p in group)
            print(f"  {label:>10s} = {value:.6g}  ({parts})")

    if not RUN_IDENTIFIABILITY:
        print("\nRUN_IDENTIFIABILITY = False -- skipping profile likelihood, FIM, and CSVs.")
        make_fit_only_figure(model, tag, case, theta_hat)
        return

    thr = rmse_hat + DELTA_MMHG

    # ---------------- profile likelihood, one curve per entry ----------------
    profiles = {}
    summary_rows = []
    all_samples = []

    for k, label in enumerate(FREE_PARAMS):
        grid, prof, samples = profile_one_entry(k, theta_hat, model, case)
        all_samples.extend(samples)
        best = axis_value(theta_hat[k])
        lo, hi, tag_k = classify_profile(grid, prof, thr, rmse_hat)
        pct = 100 * (hi - lo) / 2 / best if np.isfinite(lo) else np.nan
        profiles[label] = dict(grid=grid, prof=prof, best=best, lo=lo, hi=hi, tag=tag_k)
        summary_rows.append(dict(quantity=label, best=best, lo=lo, hi=hi, pct=pct, tag=tag_k))
        print(f"  {label:>10s} in [{lo:.4g}, {hi:.4g}]  (+/-{pct:.0f}%) -> {tag_k}")

        safe_label = label.replace("*", "_")
        pd.DataFrame({label: grid, "rmse_mmHg": prof}).to_csv(
            f"{OUT_DIR}/identifiability_{model}_{tag}_profile_{safe_label}.csv", index=False)

    # ---------------- local FIM ----------------
    eigvals, _ = local_fim(theta_hat, model, case, rmse_hat ** 2)
    cond_number = float(eigvals[0] / max(eigvals[-1], 1e-300))
    print(f"\nLocal FIM in log-space of {FREE_PARAMS}:")
    print(f"  eigenvalues = {eigvals}")
    print(f"  condition number = {cond_number:.3g}")
    for i, ev in enumerate(eigvals):
        summary_rows.append(dict(quantity=f"fim_eigval_{i+1}", best=ev, lo=np.nan, hi=np.nan, pct=np.nan, tag=""))
    summary_rows.append(dict(quantity="fim_cond_number", best=cond_number, lo=np.nan, hi=np.nan, pct=np.nan, tag=""))
    summary_rows.append(dict(quantity="rmse_mmHg", best=rmse_hat, lo=np.nan, hi=np.nan, pct=np.nan, tag=""))
    summary_rows.append(dict(quantity="p_range_mmHg", best=p_range_mmHg, lo=np.nan, hi=np.nan, pct=np.nan, tag=""))

    pd.DataFrame(summary_rows).to_csv(f"{OUT_DIR}/identifiability_{model}_{tag}_summary.csv", index=False)

    # ---------------- figures ----------------
    make_diagnostic_figure(model, tag, case, profiles, theta_hat, all_samples, thr, rmse_hat)
    make_clean_profiles_figure(model, tag, profiles, rmse_hat, p_range_mmHg)


# ============================================================
# PLOTTING
# ============================================================

def make_fit_only_figure(model, tag, case, theta_hat):
    """RUN_IDENTIFIABILITY = False: just the best fit against the data, no
    profile/FIM machinery."""
    order_idx = np.argsort(case["time"])
    P_best = model_pressure_from_theta(theta_hat, model, case)[order_idx] / MMHG_TO_BARYE

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(case["volume"][order_idx], P_best, "b-", lw=2, label="best fit")
    ax.plot(case["volume"], case["Pout"] / MMHG_TO_BARYE, "k.", ms=3, label="data")
    ax.set_xlabel("Volume (mL)"); ax.set_ylabel("Pressure (mmHg)")
    ax.set_title(f"Calibration only: {model}, free = {FREE_PARAMS}", fontsize=11)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = f"{OUT_DIR}/identifiability_{model}_{tag}_fit_only.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved {out_png}")


def make_diagnostic_figure(model, tag, case, profiles, theta_hat, all_samples, thr, rmse_hat):
    n_panels = len(profiles) + 1  # +1 for the curve bundle
    ncols = min(3, n_panels)
    nrows = -(-n_panels // ncols)

    disp_cap = max(thr * 6.0, 5.0 * rmse_hat)

    fig = plt.figure(figsize=(4.5 * ncols, 4 * nrows))
    for i, (label, prof) in enumerate(profiles.items()):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.semilogx(prof["grid"], np.minimum(prof["prof"], disp_cap), "-", lw=2)
        ax.axhline(thr, color="r", ls="--", label=f"best + {DELTA_MMHG} mmHg")
        if np.isfinite(prof["lo"]):
            ax.axvspan(prof["lo"], prof["hi"], color="orange", alpha=0.2, label="conf. interval")
        ax.axvline(prof["best"], color="k", ls=":", lw=1)
        ax.set_ylim(0, disp_cap)
        ax.set_xlabel(label)
        ax.set_ylabel("profile RMSE (mmHg)")
        ax.set_title(f"{label} ({prof['tag']})", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # curve bundle: all accepted (within-threshold) sampled fits
    ax = fig.add_subplot(nrows, ncols, n_panels)
    order_idx = np.argsort(case["time"])
    V_line = case["volume"][order_idx]
    accepted = [s for s in all_samples if s[1] <= thr]
    if len(accepted) > 300:
        rng = np.random.default_rng(0)
        accepted = [accepted[i] for i in rng.choice(len(accepted), 300, replace=False)]
    for theta_full, _ in accepted:
        P_c = model_pressure_from_theta(theta_full, model, case)[order_idx] / MMHG_TO_BARYE
        ax.plot(V_line, P_c, "-", color="0.7", lw=0.6, alpha=0.5)
    P_best = model_pressure_from_theta(theta_hat, model, case)[order_idx] / MMHG_TO_BARYE
    ax.plot(V_line, P_best, "b-", lw=2, label="best fit")
    ax.plot(case["volume"], case["Pout"] / MMHG_TO_BARYE, "k.", ms=3, label="data")
    ax.set_xlabel("Volume (mL)"); ax.set_ylabel("Pressure (mmHg)")
    ax.set_title("conf.-region curve bundle\n(tight = identified)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Passive identifiability: {model}, free = {FREE_PARAMS}", fontsize=13)
    fig.tight_layout()
    out_png = f"{OUT_DIR}/identifiability_{model}_{tag}.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved {out_png}")


def make_clean_profiles_figure(model, tag, profiles, rmse_hat, p_range_mmHg):
    """Minimal re-render (large fonts, % of P-range, no titles/legends), matching
    the style of the original *_PLOT.py companions."""
    n = len(profiles)
    if n == 0:
        return
    to_pct = 100.0 / p_range_mmHg
    thr = rmse_hat + DELTA_MMHG
    disp_cap = max(thr * to_pct * 6.0, 5.0 * rmse_hat * to_pct)

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5))
    axes = np.atleast_1d(axes)
    for ax, (label, prof) in zip(axes, profiles.items()):
        prof_pct = prof["prof"] * to_pct
        ax.plot(prof["grid"], np.minimum(prof_pct, disp_cap), "-", color="b", lw=2)
        ax.axvline(prof["best"], color="k", ls=":", lw=1)
        ax.set_ylim(0, disp_cap)
        ax.set_xlabel(label, fontsize=20)
        ax.set_ylabel("RMSE (% of P range)", fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    fig.tight_layout()
    out_png = f"{OUT_DIR}/identifiability_{model}_{tag}_profiles_only.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png}")


# ============================================================
# --plot-only : regenerate the clean figure from saved CSVs, no recompute
# ============================================================

def replot_from_saved():
    validate_config()
    if not RUN_IDENTIFIABILITY:
        raise RuntimeError(
            "RUN_IDENTIFIABILITY is False, so there is nothing to re-plot -- a "
            "calibration-only run (RUN_IDENTIFIABILITY=False) never saves profile/summary "
            "CSVs. Set RUN_IDENTIFIABILITY = True and run a full analysis first."
        )
    tag = run_tag()
    model = MATERIAL_MODEL

    summary = pd.read_csv(f"{OUT_DIR}/identifiability_{model}_{tag}_summary.csv").set_index("quantity")
    rmse_hat = float(summary.loc["rmse_mmHg", "best"])
    p_range_mmHg = float(summary.loc["p_range_mmHg", "best"])

    profiles = {}
    for label in FREE_PARAMS:
        safe_label = label.replace("*", "_")
        df = pd.read_csv(f"{OUT_DIR}/identifiability_{model}_{tag}_profile_{safe_label}.csv")
        row = summary.loc[label]
        profiles[label] = dict(grid=df[label].to_numpy(), prof=df["rmse_mmHg"].to_numpy(),
                                best=float(row["best"]), lo=float(row["lo"]), hi=float(row["hi"]),
                                tag=row["tag"])

    make_clean_profiles_figure(model, tag, profiles, rmse_hat, p_range_mmHg)


# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="Re-render the clean profiles-only figure from previously saved CSVs; no recompute.")
    args = ap.parse_args()

    if args.plot_only:
        replot_from_saved()
    else:
        run_analysis()
