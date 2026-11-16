"""
ENGINE: calibration and, optionally, identifiability analysis for the
ACTIVE spherical chamber model with a Guccione passive law and a two-hill
active twitch (ChamberSphere block of the Guccione svZeroDSolver build).
This is the active-contraction counterpart of
Calibration_Identifiability_Sphere_Passive.py: same FREE_PARAMS /
tied-product conventions, same profile likelihood + FIM machinery -- but
the model is no longer evaluated in closed form. Every residual evaluation
calls the forward solver (pysvzerod.simulate), driven by the measured
volume trace of one cardiac cycle (volume-driven: Q_in = dV/dt of the
data, Q_out = 0), and compares the simulated chamber pressure with the
measured one.

You normally do NOT edit or run this file directly: copy a wrapper such as
Run_Calibration_Identifiability_Sphere_Active_Guccione.py (or the _TonToff
variant), adapt its settings to your dataset, and
run that. The wrapper passes its settings to run() below. (Running this
file directly uses the defaults in DEFAULT_SETTINGS: cycle_124.)

Model (spherical chamber, shape-correction factor n, volume-driven):

    lam    = (V/V0)^(1/(3n)),  C_G = lam^2
    Ep     = (lam^2 - 1)/2,  Er = (lam^-4 - 1)/2
    Q      = (b_f + b_t) Ep^2 + b_t Er^2
    S_pas  = C * [ (b_f+b_t)/2 * lam^2 Ep - b_t lam^-4 Er ] * exp(Q)
    tau(t) = sigma_max * A(t),  A = g1/(1+g1) * 1/(1+g2),
             g1 = ((t-t_shift)/tau_1)^m1,  g2 = ((t-t_shift)/tau_2)^m2
    P_model(t) = gamma * (S_pas + tau) / lam

-----------------------------------------------------------------------------
HOW gamma AND n REACH THE SOLVER (no C++ changes needed)
-----------------------------------------------------------------------------
The Guccione ChamberSphere block only takes the products
    guccione_C      = gamma * C
    gamma_sigma_max = gamma * sigma_max
and has n hard-wired to 1 (P = S * (V/V0)^(-1/3)). Both limitations are
lifted here on the Python side, exactly:

  * gamma, C, sigma_max: this script works with the raw parameters and
    hands the solver the two products above.

  * n: S and P depend on V only through lam. The solver computes
    lam' = (V'/V0)^(1/3); driving it with the REMAPPED volume trace
        V'(t) = V0 * (V(t)/V0)^(1/n)
    gives lam' = (V/V0)^(1/(3n)) = lam, i.e. exactly the n-model
    (checked against the closed form through the real solver: agreement
    ~1e-2 mmHg RMSE, the solver's own discretization level at n = 1).

-----------------------------------------------------------------------------
FREE_PARAMS (see the wrapper for the full list of settings)
-----------------------------------------------------------------------------
Each entry is EXACTLY ONE optimization variable (one profile curve):
  - a raw parameter: "gamma", "n", "C", "sigma_max", "t_shift", "tau_1",
    "tau_2", "m1", "m2", "b_f", "b_t"; or
  - a tied product "gamma*C" or "gamma*sigma_max" (only these: gamma times
    the linear coefficient of one additive stress term).
Everything else stays fixed at BASE_PARAMS; volume0 is set from the data.

ONE tied product: gamma and its partner move together by the same
multiplicative step; gamma is reconstructed and keeps scaling the other
(fixed) term. BOTH products: gamma is eliminated, each product becomes the
solver's guccione_C / gamma_sigma_max directly.

    FREE_PARAMS = ["gamma*C", "gamma*sigma_max", "n"]   # compound
    FREE_PARAMS = ["gamma", "C", "sigma_max", "n"]      # individual
In the individual version only the products reach the model, so gamma/C/
sigma_max are STRUCTURALLY non-identifiable (flat profiles, one ~zero FIM
eigenvalue along d(log gamma) = -d(log C) = -d(log sigma_max)).

FREE_PARAMS is both the calibrated set and the set whose identifiability
is analyzed: every profile re-optimizes all the other FREE_PARAMS entries.
To assess the identifiability of the calibrated model, list everything
that is calibrated (e.g. the timing/shape parameters too):
    FREE_PARAMS = ["gamma*C", "gamma*sigma_max", "n",
                   "t_shift", "tau_1", "tau_2", "m1", "m2"]
Parameters left out are fixed at BASE_PARAMS and are neither fitted nor
analyzed.

-----------------------------------------------------------------------------
TWITCH PARAMETERIZATION ("TWITCH_PARAMETERIZATION")
-----------------------------------------------------------------------------
"tau_m" (default): the solver's own parameters (t_shift, tau_1, tau_2, m1,
m2). With steep hills these are degenerate: t_shift can move while the
tau's and m's compensate, leaving the twitch almost unchanged.

"ton_toff_w": t_shift is FIXED (the clock origin, e.g. the time of
electrical activation) and the twitch is calibrated through
    t_on  = t_shift + tau_1        half-activation time
    t_off = t_shift + tau_2        half-relaxation time
    w1, w2                         10-90 % contraction / relaxation durations,
                                   w = 2 tau sinh(ln 9 / m)
converted to (tau_1, tau_2, m1, m2) before every solver call. All times
are measured from the first data sample; BASE_PARAMS needs t_shift, t_on,
t_off, w1, w2 (and no tau/m values), e.g.
    FREE_PARAMS = ["gamma*C", "gamma*sigma_max", "t_on", "t_off", "w1", "w2"]

-----------------------------------------------------------------------------
METHOD / COST
-----------------------------------------------------------------------------
Joint least-squares fit (bounded 'trf', finite-difference Jacobian), then
for each FREE_PARAMS entry a rigorous profile likelihood (fix it on a grid,
re-optimize every other entry, warm-started along the sweep), confidence
region RMSE <= best + DELTA_MMHG, and a local Gauss-Newton FIM.
Every evaluation is one solver run (~0.2-1 s), so profiles run in parallel
(one process per entry) and each is saved to CSV as soon as it finishes.

Requires the Guccione build of pysvzerod (my-dev-branch).
"""

import argparse
import contextlib
import copy
import glob
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr

import pysvzerod

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# DEFAULT SETTINGS (overridden by the wrapper; see run())
# ============================================================

DEFAULT_SETTINGS = {
    # ---- data: where it is and how it is laid out ----
    "DATA": {
        # one file, a list of files, or a glob pattern (batch: same layout for all)
        "path": "/Users/Postdoc/Imperial_PV_loops/simulations_data_yale/simulations_yale/cycle_124/cav.LV.csv",
        # case name for titles/output folders; "{parent}" = containing folder
        # name, "{stem}" = file name without extension (required for batches)
        "label": "{parent}",
        "label_remove": "",            # text removed from each label, e.g. "average_over_time_"
        "sep": ",",
        "skip_rows": 2,                # header lines before the numbers start
        "time_col": 0,                 # 0-based column indices
        "pressure_col": 1,
        "volume_col": 2,
        "state_col": 3,                # phase labels (IVC/ejec/IVR/fill), or None
        "time_unit": "ms",             # "s" | "ms"
        "pressure_unit": "mmHg",       # "mmHg" | "kPa" | "Pa" | "Barye"
        "volume_unit": "mL",           # "mL" | "cm3" | "L" | "m3" | "uL"
        "cycle": "last_full_from_state",  # "all" | (t0, t1) in file time units | "last_full_from_state"
        "cycle_start_state": "IVC",    # state label that starts a cycle ("last_full_from_state")
        "V0": "min_pressure",          # "min_pressure" | "min_volume" | a number in mL
        "data_legend": "3D",
        "model_legend": "Sphere",
    },
    "OUT_DIR": "/Users/Postdoc/Reduced_order_modeling/svZeroDSolver_Sphere_opt/svZeroDSolver/scripts/identifiability_active_Guccione",
    # ---- parameters ----
    "BASE_PARAMS": {
        "gamma": 1.0, "n": 1.0,
        "C": 4297.245, "sigma_max": 135416.94,
        "t_shift": -0.486404, "tau_1": 0.547537, "tau_2": 0.738695,
        "m1": 49.9994, "m2": 22.5291,
        "b_f": 8.0, "b_t": 3.0,
    },
    "FREE_PARAMS": ["gamma*C", "gamma*sigma_max", "n"],
    "INIT_OVERRIDES": {},
    "BOUNDS_OVERRIDES": {},
    # "tau_m": two-hill timing as (t_shift, tau_1, tau_2, m1, m2)
    # "ton_toff_w": as (t_on, t_off, w1, w2) with t_shift fixed (see TWITCH PARAMETERIZATION)
    "TWITCH_PARAMETERIZATION": "tau_m",
    # ---- analysis ----
    "RUN_IDENTIFIABILITY": True,
    "DELTA_MMHG": 0.5,
    "GRID_N": 15,
    "GRID_HALF_WIDTH_LOG10": 0.6,
    "GRID_HALF_WIDTH_LOG10_OVERRIDES": {"gamma*sigma_max": 0.05, "gamma*C": 0.4},
    "N_WORKERS": None,                 # None -> one per FREE_PARAMS entry (capped by CPU count)
    "REUSE_SAVED_PROFILES": False,
    "RUN_TAG": None,
    # ---- batches (several files in DATA["path"]) ----
    "CASE_WORKERS": 1,                 # >1: cases in parallel (profiles then run serially per case)
    "SKIP_COMPLETED_CASES": False,     # skip cases whose summary (or best-fit) CSV already exists
    "BATCH_PER_PAGE": 20,              # cases per page of the batch PV-loop figure / parameter table
    # units of the batch calibrated-parameter table (CSV + image), per FREE_PARAMS
    # entry, e.g. {"gamma*C": "kPa", "t_shift": "ms"}; entries not listed keep the
    # native unit (Ba / s). Choices: TABLE_UNIT_FACTORS.
    "TABLE_UNITS": {},
    # calibrated parameters vs exercise state + correlation with covariates (batch only).
    # "csv": one row per case, in exercise order, with a "case" column (the case
    # labels), the "state_col" column (x-axis tick labels) and the "covariates"
    # columns ({column: axis label}). None -> no exercise plots.
    # "labels_csv" (optional, default: "csv"): csv with the same "case" and
    # "state_col" columns, used to name the cases in the batch PV-loop and
    # active-stress figures -- e.g. the full list, when "csv" leaves some
    # states out of the correlations. Cases not in it keep their case name.
    "EXERCISE": {"csv": None, "state_col": "state", "covariates": {}, "labels_csv": None},
    # ---- optimizer / solver ----
    "FD_STEP": 1.0e-3,
    "FIT_MAX_NFEV": 600,
    "PROFILE_MAX_NFEV": 150,
    "FIM_FD_STEP_LOG": 0.05,
    "FIM_FD_STEP_LIN": 0.005,
    "WARMUP_CYCLES": 4,
    "N_TIME_PTS_PER_CYCLE": 1000,
    "SOLVER_ABS_TOL": 1e-9,
    "SILENCE_SOLVER": True,
}

# ============================================================
# FIXED REGISTRY
# ============================================================

MMHG_TO_BARYE = 1333.22
FAILED_RESIDUAL = 1e6  # residual returned when the solver fails or the twitch is invalid

TIME_TO_S = {"s": 1.0, "ms": 1.0e-3}
PRESSURE_TO_BARYE = {"mmHg": MMHG_TO_BARYE, "kPa": 1.0e4, "Pa": 10.0, "Barye": 1.0}
VOLUME_TO_ML = {"mL": 1.0, "cm3": 1.0, "L": 1.0e3, "m3": 1.0e6, "uL": 1.0e-3}
# table display units: {native unit: {display unit: size of one display unit in native units}}
TABLE_UNIT_FACTORS = {
    "Ba": {"Ba": 1.0, "dyn/cm2": 1.0, "Pa": 10.0, "kPa": 1.0e4, "mmHg": MMHG_TO_BARYE},
    "s": {"s": 1.0, "ms": 1.0e-3},
}

# transform: "log" (theta = log value) or "lin" (theta = value).
# bounds "tshift" -> (-T_cycle, -1e-4): t - t_shift must stay > 0 over the
# cycle, or the two-hill power law (and its Jacobian's log) is undefined.
DEFAULT_RAW_SPEC = {
    "gamma":     {"transform": "log", "bounds": (1.0e-2, 1.0e2)},
    "n":         {"transform": "log", "bounds": (0.3, 20.0)},
    "C":         {"transform": "log", "bounds": (1.0e0, 1.0e6)},
    "sigma_max": {"transform": "log", "bounds": (1.0e2, 1.0e8)},
    "t_shift":   {"transform": "lin", "bounds": "tshift", "profile_halfwidth": 0.05},
    "tau_1":     {"transform": "log", "bounds": (0.01, 2.0)},
    "tau_2":     {"transform": "log", "bounds": (0.01, 2.0)},
    "m1":        {"transform": "log", "bounds": (1.0, 50.0)},
    "m2":        {"transform": "log", "bounds": (1.0, 50.0)},
    "b_f":       {"transform": "log", "bounds": (0.1, 50.0)},
    "b_t":       {"transform": "log", "bounds": (0.1, 50.0)},
    # "ton_toff_w" parameterization only (bounds "t_on"/"t_off" -> (t_shift, T)):
    "t_on":      {"transform": "lin", "bounds": "t_on", "profile_halfwidth": 0.03},
    "t_off":     {"transform": "lin", "bounds": "t_off", "profile_halfwidth": 0.03},
    "w1":        {"transform": "log", "bounds": (0.005, 0.6)},
    "w2":        {"transform": "log", "bounds": (0.005, 0.6)},
}

# Raw parameters of each twitch parameterization (the rest is common).
COMMON_PARAMS = ["gamma", "n", "C", "sigma_max", "b_f", "b_t"]
TWITCH_PARAMS = {
    "tau_m": ["t_shift", "tau_1", "tau_2", "m1", "m2"],
    "ton_toff_w": ["t_shift", "t_on", "t_off", "w1", "w2"],
}

# Partners gamma may be tied to, and which solver coefficient each product is.
GAMMA_TERM_PARTNERS = {"C": "guccione_C", "sigma_max": "gamma_sigma_max"}

# ============================================================
# ACTIVE CONFIGURATION
#
# The functions below read module-level globals (as in the passive
# script). configure() sets them from a settings dict and recomputes the
# derived quantities; worker processes call it too (via the pool
# initializer), since they re-import this module and would otherwise only
# ever see DEFAULT_SETTINGS.
# ============================================================

_SETTINGS = {}


def configure(settings):
    global _SETTINGS, RAW_SPEC, FREE_GROUPS, N_GAMMA_PRODUCTS, PRODUCT_MODE, GAMMA_TERM_MAP
    unknown = set(settings) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown setting(s): {sorted(unknown)}. Known: {sorted(DEFAULT_SETTINGS)}")
    s = copy.deepcopy(DEFAULT_SETTINGS)
    for key, value in copy.deepcopy(settings).items():
        if key == "DATA":
            s["DATA"].update(value)  # data keys not given fall back to the defaults' layout
        else:
            s[key] = value
    if s["N_WORKERS"] is None:
        s["N_WORKERS"] = max(1, min(len(s["FREE_PARAMS"]), os.cpu_count() or 1))
    _SETTINGS = s
    globals().update(s)

    RAW_SPEC = copy.deepcopy(DEFAULT_RAW_SPEC)
    for p, b in s["BOUNDS_OVERRIDES"].items():
        if p not in RAW_SPEC:
            raise ValueError(f"BOUNDS_OVERRIDES: unknown parameter '{p}'")
        RAW_SPEC[p]["bounds"] = tuple(b)

    FREE_GROUPS = [entry.split("*") for entry in s["FREE_PARAMS"]]
    N_GAMMA_PRODUCTS = sum(1 for g in FREE_GROUPS if len(g) == 2 and "gamma" in g)
    PRODUCT_MODE = N_GAMMA_PRODUCTS >= 2
    GAMMA_TERM_MAP = {}  # {solver coefficient name: theta index}, PRODUCT_MODE only
    if PRODUCT_MODE:
        for idx, group in enumerate(FREE_GROUPS):
            if len(group) == 2 and "gamma" in group:
                partner = group[0] if group[1] == "gamma" else group[1]
                GAMMA_TERM_MAP[GAMMA_TERM_PARTNERS[partner]] = idx
    validate_config()


def _worker_settings():
    """What a worker needs: everything but DATA (the data is passed to it
    already loaded, and DATA may hold things that don't pickle)."""
    return {k: v for k, v in _SETTINGS.items() if k != "DATA"}


def effective_base(p):
    return INIT_OVERRIDES.get(p, BASE_PARAMS[p])


def model_params():
    """Raw parameters of the active twitch parameterization."""
    return COMMON_PARAMS + TWITCH_PARAMS[TWITCH_PARAMETERIZATION]


def validate_config():
    if TWITCH_PARAMETERIZATION not in TWITCH_PARAMS:
        raise ValueError(f"TWITCH_PARAMETERIZATION must be one of {list(TWITCH_PARAMS)}, "
                         f"got '{TWITCH_PARAMETERIZATION}'.")
    missing = [p for p in model_params() if p not in BASE_PARAMS]
    if missing:
        raise ValueError(f"BASE_PARAMS is missing required parameters: {missing}")
    if len(set(FREE_PARAMS)) != len(FREE_PARAMS):
        raise ValueError(f"FREE_PARAMS has duplicate entries: {FREE_PARAMS}")
    for entry, unit in TABLE_UNITS.items():
        if entry not in FREE_PARAMS:
            raise ValueError(f"TABLE_UNITS: '{entry}' is not a FREE_PARAMS entry {FREE_PARAMS}")
        native = native_unit(entry)
        if unit not in TABLE_UNIT_FACTORS.get(native, {}):
            allowed = list(TABLE_UNIT_FACTORS.get(native, {}))
            raise ValueError(f"TABLE_UNITS: '{entry}' (native unit '{native or 'dimensionless'}') "
                             f"cannot be shown in '{unit}'; allowed: {allowed or 'none'}")
    for key, needed in (("csv", ["case", EXERCISE.get("state_col"), *EXERCISE.get("covariates", {})]),
                        ("labels_csv", ["case", EXERCISE.get("state_col")])):
        path = EXERCISE.get(key)
        if not path:
            continue
        if not os.path.exists(path):
            raise ValueError(f"EXERCISE: {key} '{path}' does not exist")
        columns = pd.read_csv(path, nrows=0).columns
        missing = [c for c in needed if c not in columns]
        if missing:
            raise ValueError(f"EXERCISE: columns {missing} not in '{path}' (has {list(columns)})")
    if BASE_PARAMS["t_shift"] >= 0.0:
        raise ValueError("BASE_PARAMS['t_shift'] must be negative (t - t_shift > 0 over the cycle).")
    if TWITCH_PARAMETERIZATION == "ton_toff_w":
        if "t_shift" in FREE_PARAMS:
            raise ValueError("With TWITCH_PARAMETERIZATION = 'ton_toff_w', t_shift is fixed (it only sets "
                             "the twitch clock origin); remove it from FREE_PARAMS.")
        if not BASE_PARAMS["t_shift"] < BASE_PARAMS["t_on"] < BASE_PARAMS["t_off"]:
            raise ValueError("BASE_PARAMS must satisfy t_shift < t_on < t_off.")

    seen = set()
    for label, group in zip(FREE_PARAMS, FREE_GROUPS):
        if len(group) > 1:
            if len(group) != 2 or "gamma" not in group:
                raise ValueError(
                    f"FREE_PARAMS entry '{label}' is not supported: only 'gamma*<partner>' products "
                    f"are allowed, never a product of two non-gamma parameters."
                )
            partner = group[0] if group[1] == "gamma" else group[1]
            if partner not in GAMMA_TERM_PARTNERS:
                allowed = ", ".join(f"gamma*{p}" for p in GAMMA_TERM_PARTNERS)
                raise ValueError(
                    f"FREE_PARAMS entry '{label}' is not a supported tied product. Allowed products: "
                    f"{allowed}; list parameters separately (e.g. {list(group)}) to profile each on its own."
                )
            members = [partner]
        else:
            if group[0] not in model_params():
                raise ValueError(f"FREE_PARAMS entry '{label}' is not a parameter of the "
                                 f"'{TWITCH_PARAMETERIZATION}' parameterization: {model_params()}")
            members = group
        for p in members:
            if p in seen:
                raise ValueError(f"Raw parameter '{p}' appears in more than one FREE_PARAMS entry.")
            seen.add(p)
            if RAW_SPEC[p]["transform"] == "log" and effective_base(p) <= 0.0:
                raise ValueError(
                    f"Reference value for free parameter '{p}' (in '{label}') must be > 0 "
                    f"(log-space optimization). Add an INIT_OVERRIDES entry for it."
                )

    if N_GAMMA_PRODUCTS > 0 and ["gamma"] in FREE_GROUPS:
        raise ValueError(
            "FREE_PARAMS cannot include a bare 'gamma' entry together with a tied 'gamma*<partner>' "
            "product -- gamma is not a standalone value once any product ties it to another parameter."
        )


def run_tag():
    if RUN_TAG:
        return RUN_TAG
    return "_".join(FREE_PARAMS).replace("*", "x")


def case_dir():
    return os.path.join(OUT_DIR, DATA["label"])


def output_prefix():
    return f"{case_dir()}/{DATA['label']}_Guccione_active_{run_tag()}"


def structural_warning():
    singles = {g[0] for g in FREE_GROUPS if len(g) == 1}
    if {"gamma", "C", "sigma_max"} <= singles:
        return ("NOTE: gamma, C and sigma_max are all free individually -- only gamma*C and "
                "gamma*sigma_max reach the model, so expect flat profiles for all three and a "
                "~zero FIM eigenvalue (structural non-identifiability).")
    return None


# ============================================================
# DATA
# ============================================================

def load_case(data):
    """Read one cardiac cycle as described by the DATA settings and return
    it on a uniform, strictly periodic grid: N unique samples at
    t = 0, dt, ..., T - dt (a closing sample equal to the first, as in
    files that repeat t = T, is dropped -- the solver BC appends it back)."""
    df = pd.read_csv(data["path"], sep=data["sep"], header=None, skiprows=data["skip_rows"],
                     skipinitialspace=True)
    t_raw = pd.to_numeric(df.iloc[:, data["time_col"]], errors="coerce").to_numpy()
    P_raw = pd.to_numeric(df.iloc[:, data["pressure_col"]], errors="coerce").to_numpy()
    V_raw = pd.to_numeric(df.iloc[:, data["volume_col"]], errors="coerce").to_numpy()
    ok = np.isfinite(t_raw) & np.isfinite(P_raw) & np.isfinite(V_raw)
    state = (df.iloc[:, data["state_col"]].astype(str).str.strip().to_numpy()
             if data["state_col"] is not None else np.array([""] * len(df)))
    t_raw, P_raw, V_raw, state = t_raw[ok], P_raw[ok], V_raw[ok], state[ok]

    cycle = data["cycle"]
    if cycle == "all":
        m = np.ones(len(t_raw), bool)
    elif cycle == "last_full_from_state":
        if data["state_col"] is None:
            raise ValueError("DATA['cycle'] = 'last_full_from_state' needs a state_col.")
        s0 = data["cycle_start_state"]
        starts = [t_raw[i] for i in range(1, len(state)) if state[i] == s0 and state[i - 1] != s0]
        if len(starts) < 2:
            raise RuntimeError(f"Could not find two '{s0}' onsets in the state column.")
        m = (t_raw >= starts[-2]) & (t_raw < starts[-1])
    else:
        t0, t1 = cycle
        m = (t_raw >= t0) & (t_raw < t1)
    if m.sum() < 10:
        raise RuntimeError("Selected cycle has too few points.")

    t = t_raw[m] * TIME_TO_S[data["time_unit"]]
    P = P_raw[m] * PRESSURE_TO_BARYE[data["pressure_unit"]]
    V = V_raw[m] * VOLUME_TO_ML[data["volume_unit"]]
    state = state[m]
    t = t - t[0]

    dt_all = np.diff(t)
    dt = float(np.median(dt_all))
    if np.max(np.abs(dt_all - dt)) > 0.01 * dt:
        print("WARNING: time samples are not uniformly spaced; the inflow Q = dV/dt "
              "(periodic central differences) assumes uniform spacing.")
    closing = (np.isclose(V[-1], V[0], rtol=1e-9, atol=1e-12)
               and np.isclose(P[-1], P[0], rtol=1e-9, atol=1e-12))
    if closing:
        t, P, V, state = t[:-1], P[:-1], V[:-1], state[:-1]
        print(f"Note: the last sample repeats the first (t = {t[-1] + dt:.6g} s = one period) -- "
              f"dropped; the period is {t[-1] + dt:.6g} s.")
    T = float(t[-1] + dt)

    if data["V0"] == "min_pressure":
        V0 = float(V[int(np.argmin(P))])
    elif data["V0"] == "min_volume":
        V0 = float(V.min())
    else:
        V0 = float(data["V0"])
    return dict(time=t, T=T, Pout=P, volume=V, state=state, V0=V0)


# ============================================================
# FORWARD SOLVER (volume-driven ChamberSphere via pysvzerod)
# ============================================================

@contextlib.contextmanager
def _silenced_stdout():
    """The debug build prints ~30 lines per simulate() call from C++, which
    Python-level redirection can't catch -- redirect file descriptor 1."""
    if not SILENCE_SOLVER:
        yield
        return
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def periodic_derivative(V, T):
    dt = T / len(V)
    dVdt = (np.roll(V, -1) - np.roll(V, 1)) / (2.0 * dt)
    return dVdt - dVdt.mean()  # exactly periodic volume after integration


def build_config(case, n, solver_params):
    """Solver JSON for one cycle. The data volume is remapped to
    V' = V0 (V/V0)^(1/n) so the n = 1 solver reproduces the n-model (see
    module docstring). The BC arrays get a closing sample at t = T: the
    solver takes the cycle period as t_bc[-1] - t_bc[0]."""
    V0 = case["V0"]
    V_remap = V0 * (case["volume"] / V0) ** (1.0 / n)
    Q = periodic_derivative(V_remap, case["T"])
    t_bc = np.append(case["time"], case["T"]).tolist()
    Q_bc = np.append(Q, Q[0]).tolist()
    chamber = {"volume0": V0}
    for key in ["guccione_C", "gamma_sigma_max", "t_shift", "tau_1", "tau_2", "m1", "m2", "b_f", "b_t"]:
        chamber[key] = float(solver_params[key])
    return {
        "boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"Q": Q_bc, "t": t_bc}},
            {"bc_name": "OUTFLOW", "bc_type": "FLOW", "bc_values": {"Q": [0.0] * len(t_bc), "t": t_bc}},
        ],
        "simulation_parameters": {
            "number_of_cardiac_cycles": WARMUP_CYCLES + 1,
            "number_of_time_pts_per_cardiac_cycle": N_TIME_PTS_PER_CYCLE,
            "output_all_cycles": False,
            "output_variable_based": True,
            "steady_initial": False,
            "absolute_tolerance": SOLVER_ABS_TOL,
        },
        "initial_condition": {"volume:ventricle": float(V_remap[0] - V0)},
        "vessels": [
            {"boundary_conditions": {"inlet": "INFLOW", "outlet": "OUTFLOW"},
             "vessel_id": 0, "vessel_length": 1.0, "vessel_name": "ventricle",
             "zero_d_element_type": "ChamberSphere",
             "zero_d_element_values": chamber},
        ],
        "junctions": [],
    }


def simulate(case, solver_params):
    """Run the solver; return (P, V) interpolated onto the data time points,
    with V mapped back from the remapped volume to physical volume."""
    n = solver_params["n"]
    with _silenced_stdout():
        df = pysvzerod.simulate(build_config(case, n, solver_params))

    def series(name):
        s = df[df["name"] == name].sort_values("time")
        return s["time"].to_numpy(), s["y"].to_numpy()

    t_sim, dV = series("volume:ventricle")
    _, P_sim = series("pressure:ventricle:OUTFLOW")
    t_sim = t_sim - t_sim[0]
    V0 = case["V0"]
    V_sim = V0 * ((dV + V0) / V0) ** n

    def on_data(y):
        return interp1d(t_sim, y, kind="linear", bounds_error=False,
                        fill_value=(y[0], y[-1]))(case["time"])

    return on_data(P_sim), on_data(V_sim)


# ============================================================
# THETA (one entry per FREE_PARAMS group) <-> RAW / SOLVER PARAMS
#
# theta[k] = log(axis value) for log-transformed entries (every tied
# product, and every singleton except t_shift), theta[k] = axis value for
# t_shift. A tied pair ["gamma","X"] is split back into gamma and X by a
# symmetric multiplicative step from their references (non-PRODUCT_MODE
# only), as in the passive script.
# ============================================================

def entry_is_log(group):
    return len(group) == 2 or RAW_SPEC[group[0]]["transform"] == "log"


def raw_bounds(p, T):
    b = RAW_SPEC[p]["bounds"]
    if b == "tshift":
        return (-T, -1e-4)
    if b in ("t_on", "t_off"):  # half-points between the (fixed) clock origin and the end of the cycle
        return (BASE_PARAMS["t_shift"] + 1e-3, T)
    return b


def entry_bounds(group, T):
    """Axis-value bounds of one entry (product bounds for tied pairs)."""
    los, his = zip(*(raw_bounds(p, T) for p in group))
    return float(np.prod(los)), float(np.prod(his))


def entry_theta_bounds(T):
    lo, hi = [], []
    for g in FREE_GROUPS:
        blo, bhi = entry_bounds(g, T)
        if entry_is_log(g):
            blo, bhi = np.log(blo), np.log(bhi)
        lo.append(blo); hi.append(bhi)
    return np.array(lo), np.array(hi)


def axis_value(k, theta_k):
    return float(np.exp(theta_k)) if entry_is_log(FREE_GROUPS[k]) else float(theta_k)


def axis_to_theta(k, value):
    return float(np.log(value)) if entry_is_log(FREE_GROUPS[k]) else float(value)


def theta_reference():
    return np.array([
        axis_to_theta(k, float(np.prod([effective_base(p) for p in g])))
        for k, g in enumerate(FREE_GROUPS)
    ])


def raw_params_from_theta(theta):
    """Full raw-parameter dict (0 or 1 tied gamma product active). In
    PRODUCT_MODE the tied entries are left at BASE_PARAMS here -- they are
    not meaningful individually; solver_params_from_theta handles them."""
    params = dict(BASE_PARAMS)
    for k, (group, t) in enumerate(zip(FREE_GROUPS, theta)):
        value = axis_value(k, t)
        if len(group) == 1:
            params[group[0]] = value
        elif not PRODUCT_MODE:
            base_prod = float(np.prod([effective_base(p) for p in group]))
            mult = (value / base_prod) ** (1.0 / len(group))
            for p in group:
                params[p] = effective_base(p) * mult
    return params


def twitch_from_ton_toff_w(t_shift, t_on, t_off, w1, w2):
    """(t_on, t_off, w1, w2) -> the solver's (tau_1, tau_2, m1, m2), from
    t_on = t_shift + tau_1, t_off = t_shift + tau_2 and the exact 10-90 %
    width of each hill, w = 2 tau sinh(ln 9 / m)."""
    tau_1, tau_2 = t_on - t_shift, t_off - t_shift
    m1 = np.log(9.0) / np.arcsinh(w1 / (2.0 * tau_1))
    m2 = np.log(9.0) / np.arcsinh(w2 / (2.0 * tau_2))
    return dict(tau_1=float(tau_1), tau_2=float(tau_2), m1=float(m1), m2=float(m2))


def solver_params_from_theta(theta):
    raw = raw_params_from_theta(theta)
    if TWITCH_PARAMETERIZATION == "ton_toff_w":
        raw.update(twitch_from_ton_toff_w(raw["t_shift"], raw["t_on"], raw["t_off"], raw["w1"], raw["w2"]))
    sp = {k: raw[k] for k in ["n", "t_shift", "tau_1", "tau_2", "m1", "m2", "b_f", "b_t"]}
    if PRODUCT_MODE:
        for coeff, k in GAMMA_TERM_MAP.items():
            sp[coeff] = axis_value(k, theta[k])
    else:
        sp["guccione_C"] = raw["gamma"] * raw["C"]
        sp["gamma_sigma_max"] = raw["gamma"] * raw["sigma_max"]
    return sp


def residual(theta, case):
    if TWITCH_PARAMETERIZATION == "ton_toff_w":
        raw = raw_params_from_theta(theta)
        if raw["t_off"] <= raw["t_on"]:  # relaxation must come after contraction
            return np.full(len(case["time"]), FAILED_RESIDUAL)
    sp = solver_params_from_theta(theta)
    if sp["tau_2"] <= 0.5 * sp["tau_1"]:
        return np.full(len(case["time"]), FAILED_RESIDUAL)
    try:
        P_model, _ = simulate(case, sp)
    except Exception:
        return np.full(len(case["time"]), FAILED_RESIDUAL)
    if not np.all(np.isfinite(P_model)):
        return np.full(len(case["time"]), FAILED_RESIDUAL)
    return (P_model - case["Pout"]) / MMHG_TO_BARYE


def rmse(r):
    return float(np.sqrt(np.mean(r ** 2)))


def fit(fun, x0, lo, hi, max_nfev):
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    return least_squares(fun, x0, bounds=(lo, hi), method="trf", x_scale="jac",
                         jac="2-point", diff_step=FD_STEP, max_nfev=max_nfev)


def interval(grid, prof, thr):
    ok = grid[prof <= thr]
    return (float(ok.min()), float(ok.max())) if ok.size else (np.nan, np.nan)


def classify_profile(grid, prof, thr, rmse_hat):
    """Same verdicts as the passive script (see there)."""
    lo, hi = interval(grid, prof, thr)
    if np.isfinite(lo):
        tag = "IDENTIFIABLE" if (hi < grid[-1] and lo > grid[0]) else "NOT constrained within grid"
        return lo, hi, tag
    if rmse_hat < thr:
        return lo, hi, "IDENTIFIABLE (narrower than grid resolution)"
    return lo, hi, "NOT constrained within grid"


# ============================================================
# ACTIVE-STRESS METRICS
#
# Features of the calibrated active stress tau_act(t) = gamma*sigma_max * A(t)
# (stress level, in pressure units; its contribution to the chamber
# pressure is tau_act / lambda). Times are from the first data sample.
# ============================================================

ACTIVE_METRICS = {  # key: (table header, CSV column, table number format)
    "act_peak": ("Peak (mmHg)", "act_peak_mmHg", "{:.1f}"),
    "act_t_peak": ("Time to peak (s)", "act_t_peak_s", "{:.3f}"),
    "act_max_rise": ("Max rise (mmHg/s)", "act_max_rise_mmHg_s", "{:.1f}"),
    "act_max_fall": ("Max fall (mmHg/s)", "act_max_fall_mmHg_s", "{:.1f}"),
}


def active_stress_curve(sp, T, n_pts=20000, t_start=0.0):
    """tau_act(t) in mmHg on a fine grid over one period [t_start, t_start + T).
    The twitch time is wrapped into [0, T), as in the solver, so the curve is
    periodic: an upstroke starting before t = 0 (t_shift < 0) sits at the end
    of the cycle, and t_start < 0 shows it before the first data sample."""
    t = np.linspace(t_start, t_start + T, n_pts, endpoint=False)
    ts = np.mod(t - sp["t_shift"], T)
    g1 = (ts / sp["tau_1"]) ** sp["m1"]
    g2 = (ts / sp["tau_2"]) ** sp["m2"]
    return t, sp["gamma_sigma_max"] * g1 / (1 + g1) / (1 + g2) / MMHG_TO_BARYE


def active_stress_metrics(sp, T):
    """Peak, time to peak and max rise/fall rates of tau_act over one cycle.
    The rates are taken along the twitch time s = mod(t - t_shift, T), on
    which the twitch is smooth: this skips the restart point t = T + t_shift,
    where any not-yet-relaxed tail steps down to the new upstroke."""
    t, tau = active_stress_curve(sp, T)
    order = np.argsort(np.mod(t - sp["t_shift"], T))
    dtau = np.empty_like(tau)
    dtau[order] = np.gradient(tau[order], t[1] - t[0])
    return {
        ACTIVE_METRICS["act_peak"][1]: float(tau.max()),
        ACTIVE_METRICS["act_t_peak"][1]: float(t[np.argmax(tau)]),
        ACTIVE_METRICS["act_max_rise"][1]: float(dtau.max()),
        ACTIVE_METRICS["act_max_fall"][1]: float(dtau.min()),
    }


# ============================================================
# JOINT CALIBRATION
# ============================================================

def describe_entry(k, value):
    group = FREE_GROUPS[k]
    label = FREE_PARAMS[k]
    if len(group) == 1:
        return f"  {label:>16s} = {value:.6g}"
    if PRODUCT_MODE:
        return f"  {label:>16s} = {value:.6g}  (gamma not separately reconstructed)"
    base_prod = float(np.prod([effective_base(p) for p in group]))
    mult = (value / base_prod) ** 0.5
    parts = ", ".join(f"{p}={effective_base(p) * mult:.6g}" for p in group)
    return f"  {label:>16s} = {value:.6g}  ({parts})"


def calibrate(case, title):
    """Joint least-squares fit over the current FREE_PARAMS. Returns
    (theta_hat, rmse_hat, cost_log)."""
    p_range_mmHg = float((case["Pout"].max() - case["Pout"].min()) / MMHG_TO_BARYE)
    lo, hi = entry_theta_bounds(case["T"])
    theta0 = theta_reference()
    cost_log = []

    def logged_residual(th):
        r = residual(th, case)
        cost_log.append(rmse(r))
        return r

    print(f"\n{title}: fitting {FREE_PARAMS} (reference-point RMSE = "
          f"{rmse(residual(theta0, case)):.4f} mmHg)...")
    t_fit = time.time()
    sol = fit(logged_residual, theta0, lo, hi, FIT_MAX_NFEV)
    theta_hat = sol.x
    rmse_hat = rmse(sol.fun)
    print(f"Best fit ({100*rmse_hat/p_range_mmHg:.2f}% of range, RMSE={rmse_hat:.4f} mmHg, "
          f"{len(cost_log)} solver runs, {(time.time() - t_fit)/60:.1f} min, status {sol.status}):")
    for k in range(len(FREE_PARAMS)):
        print(describe_entry(k, axis_value(k, theta_hat[k])))
        if np.isclose(theta_hat[k], lo[k], atol=1e-6) or np.isclose(theta_hat[k], hi[k], atol=1e-6):
            print(f"  {'':>16s}   ^ at its bound -- consider widening it (BOUNDS_OVERRIDES)")
    sp = solver_params_from_theta(theta_hat)
    print(f"  solver-side: guccione_C={sp['guccione_C']:.6g}, "
          f"gamma_sigma_max={sp['gamma_sigma_max']:.6g}, n={sp['n']:.6g}")
    return theta_hat, rmse_hat, cost_log


# ============================================================
# PROFILE LIKELIHOOD -- one curve per FREE_PARAMS entry
# ============================================================

def profile_grid(k, theta_hat, T):
    group = FREE_GROUPS[k]
    best = axis_value(k, theta_hat[k])
    blo, bhi = entry_bounds(group, T)
    if entry_is_log(group):
        hw = GRID_HALF_WIDTH_LOG10_OVERRIDES.get(FREE_PARAMS[k], GRID_HALF_WIDTH_LOG10)
        lo = max(best * 10 ** -hw, blo)
        hi = min(best * 10 ** hw, bhi)
        return np.geomspace(lo, hi, GRID_N)
    half = RAW_SPEC[group[0]]["profile_halfwidth"]
    return np.linspace(max(best - half, blo), min(best + half, bhi - 1e-6), GRID_N)


def profile_one_entry(k, theta_hat, case, thr):
    """Rigorous profile likelihood for FREE_GROUPS[k]: scan its value,
    re-optimizing over every other entry (nuisance) at each point. Sweeps
    outward from the optimum in both directions with warm starts; a point
    whose warm-started refit lands above the threshold is retried from the
    joint optimum's nuisance values (a stuck warm start there is exactly
    what would corrupt the confidence interval)."""
    t_start = time.time()
    n_entries = len(FREE_GROUPS)
    nuisance_idx = [i for i in range(n_entries) if i != k]
    lo_all, hi_all = entry_theta_bounds(case["T"])
    lo_n, hi_n = lo_all[nuisance_idx], hi_all[nuisance_idx]

    grid = profile_grid(k, theta_hat, case["T"])
    best = axis_value(k, theta_hat[k])
    prof_rmse = np.empty(GRID_N)
    samples = [None] * GRID_N
    theta_hat_nuisance = theta_hat[nuisance_idx].copy()

    def solve_at(i, guess):
        theta_full = np.empty(n_entries)
        theta_full[k] = axis_to_theta(k, grid[i])

        def resid_nuisance(tn):
            tf = theta_full.copy()
            tf[nuisance_idx] = tn
            return residual(tf, case)

        if not nuisance_idx:
            return theta_full, residual(theta_full, case), guess
        sol = fit(resid_nuisance, guess, lo_n, hi_n, PROFILE_MAX_NFEV)
        if rmse(sol.fun) > thr and not np.allclose(guess, theta_hat_nuisance):
            sol2 = fit(resid_nuisance, theta_hat_nuisance, lo_n, hi_n, PROFILE_MAX_NFEV)
            if rmse(sol2.fun) < rmse(sol.fun):
                sol = sol2
        theta_full[nuisance_idx] = sol.x
        return theta_full, sol.fun, sol.x

    dist = np.abs(np.log(grid / best)) if entry_is_log(FREE_GROUPS[k]) else np.abs(grid - best)
    center = int(np.argmin(dist))
    for start, stop, step in [(center, GRID_N, 1), (center - 1, -1, -1)]:
        guess = theta_hat_nuisance.copy()
        for i in range(start, stop, step):
            theta_full, r, guess = solve_at(i, guess)
            if np.all(r == FAILED_RESIDUAL):
                # solver failed (or tau_2 <= tau_1/2) everywhere the refit
                # looked: no usable fit at this grid point -- leave a gap
                prof_rmse[i] = np.nan
                print(f"    [{FREE_PARAMS[k]}] {i + 1:>3d}/{GRID_N}: {grid[i]:.5g} -> "
                      f"solver failed, recorded as NaN", flush=True)
                continue
            prof_rmse[i] = rmse(r)
            P_curve = case["Pout"] + r * MMHG_TO_BARYE
            samples[i] = (theta_full.copy(), prof_rmse[i], P_curve)
            print(f"    [{FREE_PARAMS[k]}] {i + 1:>3d}/{GRID_N}: {grid[i]:.5g} -> "
                  f"RMSE {prof_rmse[i]:.4f} mmHg", flush=True)

    print(f"  profile '{FREE_PARAMS[k]}' done in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return grid, prof_rmse, samples


def profile_csv_path(label):
    return f"{output_prefix()}_profile_{label.replace('*', '_')}.csv"


# ============================================================
# FISHER INFORMATION MATRIX (local, at joint optimum)
# ============================================================

def local_fim(theta_hat, case, sigma2):
    N = len(theta_hat)
    J = np.zeros((len(case["time"]), N))
    for k in range(N):
        h = FIM_FD_STEP_LOG if entry_is_log(FREE_GROUPS[k]) else FIM_FD_STEP_LIN
        tp = theta_hat.copy(); tp[k] += h
        tm = theta_hat.copy(); tm[k] -= h
        J[:, k] = (residual(tp, case) - residual(tm, case)) / (2 * h)
    FIM = J.T @ J / sigma2
    eigvals, eigvecs = np.linalg.eigh(FIM)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


# ============================================================
# MAIN ANALYSIS
# ============================================================

def list_cases(data):
    """[(path, label)] for DATA['path'] = one file, a list, or a glob."""
    spec = data["path"]
    if isinstance(spec, (list, tuple)):
        paths = list(spec)
    elif any(c in spec for c in "*?["):
        paths = sorted(glob.glob(spec))
    else:
        paths = [spec]
    if not paths:
        raise FileNotFoundError(f"DATA['path'] matched no files: {spec}")
    if len(paths) > 1 and "{" not in data["label"]:
        raise ValueError("Several input files but DATA['label'] has no {stem}/{parent} placeholder -- "
                         "every case would write to the same output folder.")
    cases = []
    for p in paths:
        label = data["label"].format(stem=os.path.splitext(os.path.basename(p))[0],
                                     parent=os.path.basename(os.path.dirname(os.path.abspath(p))))
        if data["label_remove"]:
            label = label.replace(data["label_remove"], "")
        cases.append((p, label))
    labels = [lab for _, lab in cases]
    if len(set(labels)) != len(labels):
        raise ValueError(f"DATA['label'] gives duplicate case names: {labels}")
    return cases


def case_settings(settings, path, label):
    s = copy.deepcopy(settings)
    s["DATA"] = {**s.get("DATA", {}), "path": path, "label": label}
    return s


def run_case(settings, path, label, log_to_file=False):
    """One dataset, isolated: configure, run, return its batch-summary row
    (a failure is reported in the row instead of stopping the batch)."""
    configure(case_settings(settings, path, label))
    os.makedirs(case_dir(), exist_ok=True)
    done_marker = (f"{output_prefix()}_summary.csv" if RUN_IDENTIFIABILITY
                   else f"{output_prefix()}_bestfit_solver_params.csv")
    if SKIP_COMPLETED_CASES and os.path.exists(done_marker):
        print(f"[{label}] already completed ({done_marker}) -- skipped")
        return dict(case=label, path=path, status="skipped (completed earlier)")
    log = open(f"{output_prefix()}_log.txt", "w") if log_to_file else None
    try:
        with contextlib.redirect_stdout(log) if log else contextlib.nullcontext():
            row = run_analysis()
        return dict(case=label, path=path, status="ok", **row)
    except Exception as exc:
        print(f"[{label}] FAILED: {type(exc).__name__}: {exc}")
        return dict(case=label, path=path, status=f"failed: {type(exc).__name__}: {exc}")
    finally:
        if log:
            log.close()


def run(settings=None, plot_only=False):
    """Entry point for wrappers: configure from `settings` (anything not
    given falls back to DEFAULT_SETTINGS), then run every case in
    DATA['path'], or only re-render the profiles figures from saved CSVs
    (plot_only)."""
    settings = settings or {}
    configure(settings)  # validates the settings once, before any case runs
    cases = list_cases(DATA)
    os.makedirs(OUT_DIR, exist_ok=True)
    if plot_only:
        for path, label in cases:
            configure(case_settings(settings, path, label))
            if RUN_IDENTIFIABILITY:
                replot_from_saved()
        if len(cases) > 1:
            make_batch_outputs(settings, cases)
        return

    batch = len(cases) > 1
    summary_csv = f"{OUT_DIR}/batch_summary_Guccione_active_{run_tag()}.csv"
    rows = []

    def record(row):
        rows.append(row)
        if batch:
            pd.DataFrame(rows).to_csv(summary_csv, index=False)  # rewritten after every case

    if batch and CASE_WORKERS > 1:
        # cases in parallel: one process each, profiles serial inside, output to per-case logs
        per_case = {**settings, "N_WORKERS": 1}
        print(f"Running {len(cases)} cases, {CASE_WORKERS} at a time (logs: <case>/..._log.txt)")
        with ProcessPoolExecutor(max_workers=CASE_WORKERS) as pool:
            futures = {pool.submit(run_case, per_case, p, lab, True): lab for p, lab in cases}
            for fut in as_completed(futures):
                row = fut.result()
                print(f"  [{row['case']}] {row['status']}"
                      + (f", RMSE {row['rmse_pct']:.2f}% of range" if "rmse_pct" in row else ""))
                record(row)
    else:
        for i, (p, lab) in enumerate(cases):
            if batch:
                print(f"\n##### case {i + 1}/{len(cases)}: {lab} #####")
            record(run_case(settings, p, lab))
    if batch:
        n_ok = sum(r["status"] == "ok" for r in rows)
        print(f"\nBatch done: {n_ok}/{len(rows)} ok. Summary: {summary_csv}")
        make_batch_outputs(settings, cases)


def run_analysis():
    """Full analysis for the currently configured single case; returns the
    case's batch-summary row."""
    label = DATA["label"]
    os.makedirs(case_dir(), exist_ok=True)
    print(f"\n{'='*70}\nGuccione active identifiability ({label}) -- free: {FREE_PARAMS}\n{'='*70}")
    case = load_case(DATA)
    p_range_mmHg = float((case["Pout"].max() - case["Pout"].min()) / MMHG_TO_BARYE)
    print(f"Loaded {DATA['path']}: T={case['T']:.4f} s, N={len(case['time'])}, "
          f"V0={case['V0']:.3f} mL ({DATA['V0']}), "
          f"P range={p_range_mmHg:.2f} mmHg, V range=[{case['volume'].min():.2f}, "
          f"{case['volume'].max():.2f}] mL")

    if PRODUCT_MODE:
        print("PRODUCT_MODE: gamma*C and gamma*sigma_max both free -- gamma is not reconstructed.")
    warn = structural_warning()
    if warn:
        print(warn)
    flat_free = [p for g in FREE_GROUPS for p in g]
    print(f"Fixed parameters: { {p: v for p, v in BASE_PARAMS.items() if p not in flat_free} }")

    # ---------------- joint calibration ----------------
    theta_hat, rmse_hat, cost_log = calibrate(case, "CALIBRATION")
    sp_hat = solver_params_from_theta(theta_hat)
    calibrated = {entry: axis_value(k, theta_hat[k]) for k, entry in enumerate(FREE_PARAMS)}
    metrics_hat = active_stress_metrics(sp_hat, case["T"])
    pd.DataFrame([{**sp_hat, **calibrated, "volume0": case["V0"], "T": case["T"], "rmse_mmHg": rmse_hat,
                   "rmse_pct": 100 * rmse_hat / p_range_mmHg, **metrics_hat}]).to_csv(
        f"{output_prefix()}_bestfit_solver_params.csv", index=False)
    print("Active-stress metrics: " + ", ".join(f"{k} = {v:.4g}" for k, v in metrics_hat.items()))
    make_fit_figures(case, theta_hat, rmse_hat, p_range_mmHg, cost_log)
    row = dict(rmse_mmHg=rmse_hat, rmse_pct=100 * rmse_hat / p_range_mmHg, p_range_mmHg=p_range_mmHg,
               V0=case["V0"], T=case["T"], **{f"fit_{k}": v for k, v in sp_hat.items()}, **metrics_hat)

    if not RUN_IDENTIFIABILITY:
        print("\nRUN_IDENTIFIABILITY = False -- skipping profile likelihood, FIM, and CSVs.")
        return row

    thr = rmse_hat + DELTA_MMHG

    # ---------------- profile likelihood, one curve per entry ----------------
    results = {}
    todo = []
    for k, entry in enumerate(FREE_PARAMS):
        path = profile_csv_path(entry)
        if REUSE_SAVED_PROFILES and os.path.exists(path):
            df = pd.read_csv(path)
            results[k] = (df[entry].to_numpy(), df["rmse_mmHg"].to_numpy(), [])
            print(f"  reusing saved profile for '{entry}' ({path})")
        else:
            todo.append(k)

    n_workers = min(N_WORKERS, len(todo)) if todo else 1
    print(f"\nProfiling {[FREE_PARAMS[k] for k in todo]} ({GRID_N} points each, "
          f"up to {PROFILE_MAX_NFEV} solver calls per refit, {n_workers} worker(s))...")

    def store(k, out):
        results[k] = out
        grid, prof, _ = out
        pd.DataFrame({FREE_PARAMS[k]: grid, "rmse_mmHg": prof}).to_csv(
            profile_csv_path(FREE_PARAMS[k]), index=False)

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=configure,
                                 initargs=(_worker_settings(),)) as pool:
            futures = {pool.submit(profile_one_entry, k, theta_hat, case, thr): k for k in todo}
            for fut in as_completed(futures):
                store(futures[fut], fut.result())
    else:
        for k in todo:
            store(k, profile_one_entry(k, theta_hat, case, thr))

    profiles = {}
    summary_rows = []
    all_samples = []
    print()
    for k, entry in enumerate(FREE_PARAMS):
        grid, prof, samples = results[k]
        all_samples.extend(s for s in samples if s is not None)
        best = axis_value(k, theta_hat[k])
        lo_k, hi_k, tag_k = classify_profile(grid, prof, thr, rmse_hat)
        pct = 100 * (hi_k - lo_k) / 2 / abs(best) if np.isfinite(lo_k) else np.nan
        profiles[entry] = dict(grid=grid, prof=prof, best=best, lo=lo_k, hi=hi_k, tag=tag_k,
                               log=entry_is_log(FREE_GROUPS[k]))
        summary_rows.append(dict(quantity=entry, best=best, lo=lo_k, hi=hi_k, pct=pct, tag=tag_k))
        print(f"  {entry:>16s} in [{lo_k:.4g}, {hi_k:.4g}]  (+/-{pct:.0f}%) -> {tag_k}")
        row.update({f"{entry}_best": best, f"{entry}_lo": lo_k, f"{entry}_hi": hi_k, f"{entry}_tag": tag_k})

    # ---------------- local FIM ----------------
    eigvals, eigvecs = local_fim(theta_hat, case, rmse_hat ** 2)
    cond_number = float(eigvals[0] / max(eigvals[-1], 1e-300))
    print(f"\nLocal FIM in theta-space of {FREE_PARAMS} (log, except t_shift):")
    print(f"  eigenvalues = {eigvals}")
    print(f"  condition number = {cond_number:.3g}")
    print("  sloppiest direction: " + ", ".join(
        f"{entry}={eigvecs[i, -1]:+.3f}" for i, entry in enumerate(FREE_PARAMS)))
    pd.DataFrame(eigvecs, index=FREE_PARAMS,
                 columns=[f"eig{i+1}" for i in range(len(eigvals))]).to_csv(
        f"{output_prefix()}_fim_eigvecs.csv")
    for i, ev in enumerate(eigvals):
        summary_rows.append(dict(quantity=f"fim_eigval_{i+1}", best=ev, lo=np.nan, hi=np.nan, pct=np.nan, tag=""))
    summary_rows.append(dict(quantity="fim_cond_number", best=cond_number, lo=np.nan, hi=np.nan, pct=np.nan, tag=""))
    summary_rows.append(dict(quantity="rmse_mmHg", best=rmse_hat, lo=np.nan, hi=np.nan, pct=np.nan, tag=""))
    summary_rows.append(dict(quantity="p_range_mmHg", best=p_range_mmHg, lo=np.nan, hi=np.nan, pct=np.nan, tag=""))

    # ---------------- active-stress metrics: range over all accepted profile fits ----------------
    accepted = [solver_params_from_theta(th) for th, r_s, _ in all_samples if r_s <= thr]
    if accepted:
        vals = pd.DataFrame([active_stress_metrics(sp, case["T"]) for sp in accepted])
        print(f"\nActive-stress metrics over {len(accepted)} fits within best + {DELTA_MMHG} mmHg:")
        for col, best in metrics_hat.items():
            lo_m, hi_m = float(min(vals[col].min(), best)), float(max(vals[col].max(), best))
            pct = 100 * (hi_m - lo_m) / 2 / abs(best) if best else np.nan
            summary_rows.append(dict(quantity=col, best=best, lo=lo_m, hi=hi_m, pct=pct, tag="profile-sample range"))
            row.update({f"{col}_lo": lo_m, f"{col}_hi": hi_m})
            print(f"  {col:>22s} = {best:.4g}  in [{lo_m:.4g}, {hi_m:.4g}]  (+/-{pct:.0f}%)")
    pd.DataFrame(summary_rows).to_csv(f"{output_prefix()}_summary.csv", index=False)

    # ---------------- figures ----------------
    make_diagnostic_figure(case, profiles, theta_hat, all_samples, thr, rmse_hat)
    make_clean_profiles_figure(profiles, rmse_hat, p_range_mmHg)
    row["fim_cond_number"] = cond_number
    return row


# ============================================================
# PLOTTING
# ============================================================

def closed(y):
    """Append the first sample at the end, so a cycle's curves close (the
    fit itself uses each instant once: t = 0 and t = T are the same point)."""
    return np.append(y, y[0])


def make_fit_figures(case, theta_hat, rmse_hat, p_range_mmHg, cost_log):
    """PV-loop comparison, pressure vs time and convergence, in the style of
    the APR29002 figures (optimization_active_contraction_Guccione_multiparameter_4Katrin.py)."""
    label = DATA["label"]
    sp = solver_params_from_theta(theta_hat)
    P_model, V_model = simulate(case, sp)
    mmHg = 1.0 / MMHG_TO_BARYE
    rmse_pct = 100.0 * rmse_hat / p_range_mmHg
    prefix = output_prefix()

    pd.DataFrame({"time_s": case["time"], "volume_data_mL": case["volume"],
                  "pressure_data_mmHg": case["Pout"] * mmHg, "volume_model_mL": V_model,
                  "pressure_model_mmHg": P_model * mmHg}).to_csv(f"{prefix}_fit_curves.csv", index=False)

    fig = plt.figure(figsize=(7, 6))
    plt.plot(closed(case["volume"]), closed(case["Pout"]) * mmHg, "-", lw=3, label=DATA["data_legend"])
    plt.plot(closed(V_model), closed(P_model) * mmHg, "-", lw=3, label=DATA["model_legend"])
    plt.xlabel("Volume (mL)", fontsize=20)
    plt.ylabel("Pressure (mmHg)", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.title(f"{label} - PV loop (RMSE = {rmse_pct:.2f}%)", fontsize=20)
    plt.legend(loc="upper right", fontsize=16)
    plt.tight_layout()
    fig.savefig(f"{prefix}_PV_loop_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    t_closed = np.append(case["time"], case["T"])
    ax.plot(t_closed, closed(case["Pout"]) * mmHg, "-", lw=2, label=f"{DATA['data_legend']} (target)")
    ax.plot(t_closed, closed(P_model) * mmHg, "--", lw=2, label="0D model")
    if TWITCH_PARAMETERIZATION == "ton_toff_w":
        for key, col in (("tau_1", "C2"), ("tau_2", "C3")):
            t_half = sp["t_shift"] + sp[key]
            name = "on" if key == "tau_1" else "off"
            ax.axvline(t_half, color=col, ls="--", lw=1.2, label=f"$t_{{{name}}}$ = {t_half:.3f}s")
    else:
        ax.axvline(np.mod(sp["t_shift"], case["T"]), color="C2", ls="--", lw=1.2,  # onset, wrapped into the cycle
                   label=f"$t_{{shift}}$ = {sp['t_shift']:.3f}s")
    ax.set_xlabel("time in cycle (s)"); ax.set_ylabel("Pressure (mmHg)")
    ax.set_title(f"{label} - pressure over the cardiac cycle")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{prefix}_pressure_vs_time.png", dpi=200)
    plt.close(fig)

    if cost_log:
        fig = plt.figure(figsize=(7, 4))
        plt.semilogy(cost_log, ".-")
        plt.xlabel("residual evaluation"); plt.ylabel("pressure RMSE (mmHg)")
        plt.title(f"Calibration convergence (Guccione, {label})")
        plt.tight_layout()
        fig.savefig(f"{prefix}_convergence.png", dpi=200)
        plt.close(fig)
    print(f"Saved {prefix}_PV_loop_comparison.png / _pressure_vs_time.png / _convergence.png")


def make_diagnostic_figure(case, profiles, theta_hat, all_samples, thr, rmse_hat):
    n_panels = len(profiles) + 1  # +1 for the curve bundle
    ncols = min(3, n_panels)
    nrows = -(-n_panels // ncols)
    disp_cap = max(thr * 6.0, 5.0 * rmse_hat)

    fig = plt.figure(figsize=(4.5 * ncols, 4 * nrows))
    for i, (entry, prof) in enumerate(profiles.items()):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        plot = ax.semilogx if prof["log"] else ax.plot
        plot(prof["grid"], np.minimum(prof["prof"], disp_cap), "-o", lw=2, ms=3)
        ax.axhline(thr, color="r", ls="--", label=f"best + {DELTA_MMHG} mmHg")
        if np.isfinite(prof["lo"]):
            ax.axvspan(prof["lo"], prof["hi"], color="orange", alpha=0.2, label="conf. interval")
        ax.axvline(prof["best"], color="k", ls=":", lw=1)
        ax.set_ylim(0, disp_cap)
        ax.set_xlabel(entry)
        ax.set_ylabel("profile RMSE (mmHg)")
        ax.set_title(f"{entry} ({prof['tag']})", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # curve bundle: all accepted (within-threshold) profile fits
    ax = fig.add_subplot(nrows, ncols, n_panels)
    mmHg = 1.0 / MMHG_TO_BARYE
    for _, rmse_s, P_curve in all_samples:
        if rmse_s <= thr:
            ax.plot(closed(case["volume"]), closed(P_curve) * mmHg, "-", color="0.7", lw=0.6, alpha=0.5)
    P_best, V_best = simulate(case, solver_params_from_theta(theta_hat))
    ax.plot(closed(V_best), closed(P_best) * mmHg, "b-", lw=2, label="best fit")
    ax.plot(case["volume"], case["Pout"] * mmHg, "k.", ms=3, label="data")
    ax.set_xlabel("Volume (mL)"); ax.set_ylabel("Pressure (mmHg)")
    ax.set_title("conf.-region curve bundle\n(tight = identified)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Active identifiability: Guccione, {DATA['label']}, free = {FREE_PARAMS}", fontsize=13)
    fig.tight_layout()
    out_png = f"{output_prefix()}_identifiability.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved {out_png}")


def make_clean_profiles_figure(profiles, rmse_hat, p_range_mmHg):
    """Minimal re-render (large fonts, % of P-range, no titles/legends),
    matching the passive script's profiles-only figure."""
    n = len(profiles)
    if n == 0:
        return
    to_pct = 100.0 / p_range_mmHg
    thr = rmse_hat + DELTA_MMHG
    disp_cap = max(thr * to_pct * 6.0, 5.0 * rmse_hat * to_pct)

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5))
    axes = np.atleast_1d(axes)
    for ax, (entry, prof) in zip(axes, profiles.items()):
        prof_pct = prof["prof"] * to_pct
        ax.plot(prof["grid"], np.minimum(prof_pct, disp_cap), "-", color="b", lw=2)
        ax.axvline(prof["best"], color="k", ls=":", lw=1)
        ax.set_ylim(0, disp_cap)
        ax.set_xlabel(entry, fontsize=20)
        ax.set_ylabel("RMSE (% of P range)", fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    fig.tight_layout()
    out_png = f"{output_prefix()}_profiles_only.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png}")


# ============================================================
# BATCH OVERVIEW: all PV loops + table of calibrated parameters
# (built from each case's saved CSVs, so it also covers skipped / parallel
# cases and --plot-only)
# ============================================================

RAW_DISPLAY = {
    "gamma": (r"$\gamma$", ""), "n": (r"$n$", ""), "C": (r"$C$", "Ba"),
    "sigma_max": (r"$\sigma_{max}$", "Ba"), "t_shift": (r"$t_{shift}$", "s"),
    "tau_1": (r"$\tau_1$", "s"), "tau_2": (r"$\tau_2$", "s"), "m1": (r"$m_1$", ""),
    "m2": (r"$m_2$", ""), "b_f": (r"$b_f$", ""), "b_t": (r"$b_t$", ""),
    "t_on": (r"$t_{on}$", "s"), "t_off": (r"$t_{off}$", "s"), "w1": (r"$w_1$", "s"), "w2": (r"$w_2$", "s"),
}


def native_unit(entry):
    """Unit a FREE_PARAMS entry is computed in ('Ba', 's' or '' if dimensionless)."""
    return next((RAW_DISPLAY[p][1] for p in entry.split("*") if RAW_DISPLAY[p][1]), "")


def table_unit(entry):
    """Unit a FREE_PARAMS entry is shown in in the batch table (TABLE_UNITS or native)."""
    return TABLE_UNITS.get(entry, native_unit(entry))


def to_table_unit(entry, value):
    """Value of a FREE_PARAMS entry (native units) converted to its table unit."""
    native = native_unit(entry)
    return value / TABLE_UNIT_FACTORS[native][table_unit(entry)] if native else value


def with_unit(name, unit):
    return f"{name} ({unit})" if unit else name


def display_name(entry):
    """Table header for a FREE_PARAMS entry, e.g. 'gamma*C' -> gamma C (kPa)."""
    symbol = r"$\cdot$".join(RAW_DISPLAY[p][0] for p in entry.split("*"))
    return with_unit(symbol, table_unit(entry))


def fmt_value(v):
    if not np.isfinite(v):
        return "--"
    return f"{v:.4g}" if 1e-3 <= abs(v) < 1e5 else f"{v:.3e}"


def pages(items):
    return [items[i:i + BATCH_PER_PAGE] for i in range(0, len(items), BATCH_PER_PAGE)]


def page_path(base, i, n_pages):
    return f"{base}.png" if n_pages == 1 else f"{base}_page{i + 1}.png"


def _text_len(text):
    """Approximate printed width (characters) of a cell text: mathtext commands
    such as \\sigma or \\cdot count as one character, markup as none."""
    text = re.sub(r"\\[A-Za-z]+", "x", text)
    return len(re.sub(r"[${}_^]", "", text))


def bold_unit(header):
    """Header with its trailing '(unit)' in mathtext \\mathbf, so only the unit
    is bold in a regular-weight header cell."""
    return re.sub(r"\(([^()$]*)\)$",
                  lambda m: r"$\mathbf{(" + m.group(1).replace(" ", r"\ ") + r")}$", header)


def save_table_images(headers, cells, base, split_units=True, bold_units_only=False):
    """Presentable table image(s), BATCH_PER_PAGE rows per page. With
    split_units, headers are split into name / (unit) on two lines; with
    bold_units_only, only the header units are bold (else the whole header).
    Each column is as wide as its longest line of text."""
    if split_units:
        headers = [h[::-1].replace("( ", "(\n", 1)[::-1] if " (" in h else h for h in headers]
    bold_whole = [True] * len(headers)
    if bold_units_only:  # headers without a unit (e.g. "Case") stay bold in full
        bold_whole = [bold_unit(h) == h for h in headers]
        headers = [bold_unit(h) for h in headers]
    two_line_header = any("\n" in h for h in headers)
    widths = []
    for c, h in enumerate(headers):
        longest = max([_text_len(line) for line in h.split("\n")] + [_text_len(row[c]) for row in cells])
        widths.append(longest + 3)  # padding
    total = float(sum(widths))
    table_pages = pages(cells)
    for i, page in enumerate(table_pages):
        fig_w = max(6.0, 0.13 * total)
        fig, ax = plt.subplots(figsize=(fig_w, 0.42 * (len(page) + 2) + 0.4))
        ax.axis("off")
        tab = ax.table(cellText=page, colLabels=headers, loc="center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(10)
        tab.scale(1.0, 1.5)
        for (r, c), cell in tab.get_celld().items():
            cell.set_width(widths[c] / total)
            cell.set_edgecolor("#bbbbbb")
            if r == 0:
                if two_line_header:
                    cell.set_height(cell.get_height() * 1.9)  # room for two header lines
                cell.set_facecolor("#dde3ea")
                cell.set_text_props(weight="bold" if bold_whole[c] else "normal")
            elif r % 2 == 0:
                cell.set_facecolor("#f4f6f8")
        fig.tight_layout()
        out = page_path(base, i, len(table_pages))
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


def state_labels():
    """{case: exercise state} for naming cases in the batch figures
    (EXERCISE labels_csv, else csv); empty if neither is set."""
    path = EXERCISE.get("labels_csv") or EXERCISE.get("csv")
    if not path:
        return {}
    ex = pd.read_csv(path)
    return dict(zip(ex["case"].astype(str), ex[EXERCISE["state_col"]].astype(str)))


def make_exercise_plots(rows, base, metric_rows=(), metric_spec=None):
    """Calibrated FREE_PARAMS (table units) and active-stress metrics vs
    exercise state, and the parameters' correlation with each EXERCISE
    covariate (scatter + linear fit, Pearson / Spearman). Cases are ordered
    as in the EXERCISE csv."""
    ex = pd.read_csv(EXERCISE["csv"])
    ex["case"] = ex["case"].astype(str)
    fitted = pd.DataFrame(rows).set_index("case")
    missing = [c for c in ex["case"] if c not in fitted.index]
    if missing:
        print(f"  EXERCISE: no calibrated fit for {missing} -- left out of the exercise plots")
    ex = ex[ex["case"].isin(fitted.index)].reset_index(drop=True)
    if len(ex) < 3:
        print("  EXERCISE: fewer than 3 cases with a fit -- no exercise plots")
        return
    values = fitted.loc[ex["case"]]
    states = ex[EXERCISE["state_col"]].astype(str).tolist()
    # each case keeps its color from the active-stress curves (spread over ALL
    # fitted cases), also when the EXERCISE csv lists only some of them
    all_colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(fitted)))
    colors = [all_colors[fitted.index.get_loc(c)] for c in ex["case"]]
    ncols = min(3, len(FREE_PARAMS))
    nrows = -(-len(FREE_PARAMS) // ncols)

    def plot_vs_state(table, columns, titles, name):
        """One panel per column of `table` (indexed by case) vs exercise state."""
        x = np.arange(len(ex))
        nc = 2 if len(columns) == 4 else min(3, len(columns))  # 4 panels: 2 x 2
        nr = -(-len(columns) // nc)
        fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False)
        for ax, col, title in zip(axes.ravel(), columns, titles):
            y = table.loc[ex["case"], col].to_numpy(dtype=float)
            ax.plot(x, y, "-", color="#999999", lw=1.2, zorder=1)
            ax.scatter(x, y, c=colors, s=45, edgecolors="k", linewidths=0.5, zorder=2)
            ax.set_title(title, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(states, rotation=45, ha="right", fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
        for ax in axes.ravel()[len(columns):]:
            ax.axis("off")
        fig.supxlabel("Exercise state", fontsize=13)
        fig.tight_layout()
        out = base.format(name) + ".png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved {out}")

    # ---- parameters vs exercise state ----
    plot_vs_state(fitted, FREE_PARAMS, [display_name(e) for e in FREE_PARAMS], "parameters_vs_exercise")

    # ---- active-stress metrics vs exercise state (units as in the metrics table) ----
    if len(metric_rows):
        spec = metric_spec or ACTIVE_METRICS
        metrics = pd.DataFrame(list(metric_rows)).set_index("case")
        plot_vs_state(metrics, [col for _, col, _ in spec.values()],
                      [head for head, _, _ in spec.values()], "active_stress_metrics_vs_exercise")

    # ---- correlation with each covariate ----
    corr_rows = []
    for cov, cov_label in EXERCISE["covariates"].items():
        c = ex[cov].to_numpy(dtype=float)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows + 0.6), squeeze=False)
        for ax, entry in zip(axes.ravel(), FREE_PARAMS):
            v = values[entry].to_numpy(dtype=float)
            ok = np.isfinite(c) & np.isfinite(v)
            r, p = pearsonr(c[ok], v[ok])
            rho, p_rho = spearmanr(c[ok], v[ok])
            corr_rows.append({"parameter": with_unit(entry, table_unit(entry)), "covariate": cov,
                              "n": int(ok.sum()), "pearson_r": r, "pearson_p": p,
                              "spearman_rho": rho, "spearman_p": p_rho})
            slope, intercept = np.polyfit(c[ok], v[ok], 1)
            c_line = np.linspace(c[ok].min(), c[ok].max(), 50)
            ax.plot(c_line, slope * c_line + intercept, "--", color="#555555", lw=1.2, zorder=1)
            ax.scatter(c, v, c=colors, s=45, edgecolors="k", linewidths=0.5, zorder=2)
            ax.set_title(display_name(entry), fontsize=12, pad=20)
            # scipy returns p = 0 for a perfect rank correlation; the true p is just small
            p_txt = "p < 0.001" if p_rho < 1e-3 else f"p = {p_rho:.3g}"
            ax.text(0.5, 1.02, r"Spearman $\rho$" + f" = {rho:.2f} ({p_txt})",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#444444")
            ax.tick_params(labelsize=9)
        for ax in axes.ravel()[len(FREE_PARAMS):]:
            ax.axis("off")
        fig.supxlabel(cov_label, fontsize=13)
        handles = [plt.Line2D([], [], marker="o", ls="", color=col, markeredgecolor="k", markersize=7, label=s)
                   for s, col in zip(states, colors)]
        legend_h = 0.4
        fig.tight_layout(rect=(0, 0, 1, 1 - legend_h / fig.get_figheight()))
        fig.legend(handles=handles, loc="upper center", ncol=min(len(states), 10), fontsize=9, frameon=False)
        out = base.format(f"correlation_{cov}") + ".png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved {out}")

    if corr_rows:
        corr = pd.DataFrame(corr_rows)
        corr.to_csv(base.format("correlations") + ".csv", index=False)
        print(f"\nCorrelations with exercise covariates ({len(ex)} cases):")
        print(corr.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


def make_batch_outputs(settings, cases):
    rows, curves, metric_rows, act_curves = [], [], [], []
    for path, label in cases:
        configure(case_settings(settings, path, label))
        prefix = output_prefix()
        fit_csv, curves_csv = f"{prefix}_bestfit_solver_params.csv", f"{prefix}_fit_curves.csv"
        if not (os.path.exists(fit_csv) and os.path.exists(curves_csv)):
            print(f"  [{label}] no saved fit -- left out of the batch figure/table")
            continue
        fit = pd.read_csv(fit_csv).iloc[0]
        rows.append({"case": label, **{entry: to_table_unit(entry, fit.get(entry, np.nan))
                                       for entry in FREE_PARAMS}})
        df_curves = pd.read_csv(curves_csv)
        curves.append((label, fit.get("rmse_pct", np.nan), df_curves))
        # active-stress metrics, recomputed from the saved solver parameters (works for older runs too)
        t_data = df_curves["time_s"].to_numpy()
        T = float(fit["T"]) if "T" in fit and np.isfinite(fit["T"]) else float(t_data[-1] + (t_data[1] - t_data[0]))
        sp = {k: float(fit[k]) for k in ["t_shift", "tau_1", "tau_2", "m1", "m2", "gamma_sigma_max"]}
        metric_rows.append({"case": label, **active_stress_metrics(sp, T)})
        act_curves.append((label, sp, T))
    if not rows:
        return
    # one period per case, starting before the earliest twitch onset, so the
    # whole upstroke (which begins before the first data sample when
    # t_shift < 0) is visible and the next beat's upstroke is not repeated
    t_start_act = min(0.0, min(sp["t_shift"] for _, sp, _ in act_curves) - 0.05)
    act_curves = [(label,) + active_stress_curve(sp, T, n_pts=2000, t_start=t_start_act)
                  for label, sp, T in act_curves]

    base = f"{OUT_DIR}/batch_{{}}_Guccione_active_{run_tag()}"
    table = pd.DataFrame(rows).rename(columns={e: with_unit(e, table_unit(e)) for e in FREE_PARAMS})
    table.to_csv(base.format("calibrated_parameters") + ".csv", index=False)
    print(f"\nCalibrated parameters ({len(table)} cases):")
    print(table.to_string(index=False, float_format=fmt_value))

    # cases are named by exercise state (when EXERCISE gives one) and colored
    # as in the active-stress / exercise figures (first case blue -> last red)
    names = state_labels()
    case_color = dict(zip([label for label, _, _ in curves],
                          plt.cm.turbo(np.linspace(0.05, 0.95, len(curves)))))

    # ---- all PV loops, one small panel per case ----
    curve_pages = pages(curves)
    for i, page in enumerate(curve_pages):
        ncols = min(4, len(page))
        nrows = -(-len(page) // ncols)
        legend_h = 0.5  # inches reserved on top for the shared legend
        fig_h = 3.0 * nrows + legend_h
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, fig_h), squeeze=False)
        for ax, (label, pct, df) in zip(axes.ravel(), page):
            ax.plot(closed(df["volume_data_mL"]), closed(df["pressure_data_mmHg"]), "-", lw=2,
                    color="C0", label=DATA["data_legend"])
            ax.plot(closed(df["volume_model_mL"]), closed(df["pressure_model_mmHg"]), "-", lw=2,
                    color="C1", label=DATA["model_legend"])
            ax.set_title(names.get(label, label), loc="left", fontsize=11, fontweight="bold",
                         color=case_color[label])
            ax.set_title(f"RMSE {pct:.2f}%", loc="right", fontsize=10, color="black")
            ax.tick_params(labelsize=8)
        for ax in axes.ravel()[len(page):]:
            ax.axis("off")
        fig.supxlabel("Volume (mL)", fontsize=13)
        fig.supylabel("Pressure (mmHg)", fontsize=13)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0, 1, 1 - legend_h / fig_h))
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=11, frameon=False)
        out = page_path(base.format("PV_loops"), i, len(curve_pages))
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved {out}")

    # ---- presentable table image: calibrated parameters ----
    headers = ["Case"] + [display_name(e) for e in FREE_PARAMS]
    cells = [[r["case"]] + [fmt_value(r[e]) for e in FREE_PARAMS] for r in rows]
    save_table_images(headers, cells, base.format("calibrated_parameters"), split_units=False)

    # active stress (curves, metrics table, metrics vs exercise) is shown in
    # the unit of gamma*sigma_max in the parameter table (TABLE_UNITS), else mmHg
    sigma_entry = next((e for e in FREE_PARAMS if "sigma_max" in e.split("*")), None)
    stress_unit = TABLE_UNITS.get(sigma_entry, "mmHg") if sigma_entry else "mmHg"
    mmHg_to_unit = MMHG_TO_BARYE / TABLE_UNIT_FACTORS["Ba"][stress_unit]
    metric_spec = {}  # key: (header, column, format) in stress_unit
    for key, (head, col, fmt) in ACTIVE_METRICS.items():
        if "mmHg" in col:
            head, col = head.replace("mmHg", stress_unit), col.replace("mmHg", stress_unit)
            if key == "act_peak" and mmHg_to_unit < 0.5:  # e.g. kPa: one more decimal
                fmt = "{:.2f}"
        metric_spec[key] = (head, col, fmt)
    metric_rows = [{"case": r["case"], **{metric_spec[k][1]: r[c] * (mmHg_to_unit if "mmHg" in c else 1.0)
                                          for k, (_, c, _) in ACTIVE_METRICS.items()}}
                   for r in metric_rows]

    # ---- active-stress metrics: table (CSV + image) ----
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(base.format("active_stress_metrics") + ".csv", index=False)
    print(f"\nActive-stress metrics ({len(metrics)} cases):")
    print(metrics.to_string(index=False, float_format=fmt_value))
    headers = ["Case"] + [head for head, _, _ in metric_spec.values()]
    cells = [[r["case"]] + [fmt.format(r[col]) for _, col, fmt in metric_spec.values()] for r in metric_rows]
    save_table_images(headers, cells, base.format("active_stress_metrics"), bold_units_only=True)

    # ---- active-stress curves of all cases, overlaid ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, t, tau in act_curves:
        ax.plot(t, tau * mmHg_to_unit, "-", lw=2, color=case_color[label], label=names.get(label, label))
    if t_start_act < 0.0:
        ax.axvline(0.0, color="0.4", ls=":", lw=1.2)  # first data sample
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel(f"Active stress ({stress_unit})", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, ncol=2, frameon=False)
    fig.tight_layout()
    out = base.format("active_stress_curves") + ".png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")

    # ---- calibrated parameters vs exercise state / covariates ----
    if EXERCISE.get("csv"):
        make_exercise_plots(rows, base, metric_rows, metric_spec)


# ============================================================
# plot_only : regenerate the clean figure from saved CSVs, no recompute
# ============================================================

def replot_from_saved():
    if not RUN_IDENTIFIABILITY:
        raise RuntimeError(
            "RUN_IDENTIFIABILITY is False, so there is nothing to re-plot -- a calibration-only "
            "run never saves profile/summary CSVs. Set RUN_IDENTIFIABILITY = True and run first."
        )
    summary = pd.read_csv(f"{output_prefix()}_summary.csv").set_index("quantity")
    rmse_hat = float(summary.loc["rmse_mmHg", "best"])
    p_range_mmHg = float(summary.loc["p_range_mmHg", "best"])

    profiles = {}
    for k, entry in enumerate(FREE_PARAMS):
        df = pd.read_csv(profile_csv_path(entry))
        row = summary.loc[entry]
        profiles[entry] = dict(grid=df[entry].to_numpy(), prof=df["rmse_mmHg"].to_numpy(),
                               best=float(row["best"]), lo=float(row["lo"]), hi=float(row["hi"]),
                               tag=row["tag"], log=entry_is_log(FREE_GROUPS[k]))
    make_clean_profiles_figure(profiles, rmse_hat, p_range_mmHg)


def cli(settings=None):
    """Command line shared by wrappers: [--plot-only]."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="Re-render the clean profiles-only figure from previously saved CSVs; no recompute.")
    args = ap.parse_args()
    run(settings, plot_only=args.plot_only)


configure({})  # module-level globals start at DEFAULT_SETTINGS

if __name__ == "__main__":
    cli()
