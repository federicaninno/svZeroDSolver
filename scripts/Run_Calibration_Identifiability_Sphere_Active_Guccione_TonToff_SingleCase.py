"""
Calibration + identifiability of the active Guccione sphere model -- RUN FILE,
ONE case, two-hill activation reparameterized as (t_on, t_off, w1, w2) with
t_shift fixed.

Single-case version of Run_Calibration_Identifiability_Sphere_Active_Guccione_TonToff.py
(same engine, same parameterization; no batch / exercise outputs):

  * Calibrated: C, sigma_max, t_on, t_off, w1, w2.
  * TWITCH_PARAMETERIZATION = "ton_toff_w": the activation is calibrated through
        t_on   half-activation time          (= t_shift + tau_1)
        t_off  half-relaxation time          (= t_shift + tau_2)
        w1     10-90 % contraction duration  (= 2 tau_1 sinh(ln 9 / m1))
        w2     90-10 % relaxation duration   (= 2 tau_2 sinh(ln 9 / m2))
    all in seconds from the first data sample; the engine converts them to the
    solver's (tau_1, tau_2, m1, m2) before every solver call.
  * t_shift (the twitch clock origin) is FIXED, here to the value calibrated
    for the rest case APR29002 with n = 1 (fw_nfixed batch): -0.0392 s.
  * gamma = 1 and n = 1 are fixed (no geometric information). With gamma
    fixed, C and sigma_max are exactly the solver's gamma*C and gamma*sigma_max.
    If you change n, t_shift above no longer matches.

To use another case, change CASE (or DATA["path"]). Paths are relative to
this file's folder; the engine Calibration_Identifiability_Sphere_Active_Guccione.py
must sit next to it.

Run:            python Run_Calibration_Identifiability_Sphere_Active_Guccione_TonToff_SingleCase.py
Re-plot only:   python Run_Calibration_Identifiability_Sphere_Active_Guccione_TonToff_SingleCase.py --plot-only
(needs the Guccione build of pysvzerod)

Outputs (OUT_DIR/<CASE>/<CASE>_Guccione_active_<free params>...):
  _PV_loop_comparison.png, _pressure_vs_time.png, _convergence.png   fit vs data
  _bestfit_solver_params.csv, _fit_curves.csv
  and, with RUN_IDENTIFIABILITY = True:
  _identifiability.png, _profiles_only.png, _summary.csv, _profile_<entry>.csv, _fim_eigvecs.csv
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Calibration_Identifiability_Sphere_Active_Guccione as engine

CASE = "APR29002"

SETTINGS = {
    # ------------------------------------------------------------------
    # DATA: the file and how it is laid out
    # ------------------------------------------------------------------
    "DATA": {
        "path": os.path.join(HERE, "fw", f"average_over_time_{CASE}.csv"),
        "label": CASE,                  # case name for titles / output folder
        "label_remove": "",
        "sep": ",",
        "skip_rows": 1,                 # header lines before the numbers start
        "time_col": 0,                  # 0-based column indices
        "pressure_col": 1,
        "volume_col": 2,
        "state_col": None,              # phase labels (IVC/ejec/IVR/fill) column, if any
        "time_unit": "s",               # "s" | "ms"
        "pressure_unit": "mmHg",        # "mmHg" | "kPa" | "Pa" | "Barye"
        "volume_unit": "mL",            # "mL" | "cm3" | "L" | "m3" | "uL"
        # which samples form the cycle to fit:
        #   "all"                   the whole file is one cycle (a repeated closing sample is dropped)
        #   (t0, t1)                window [t0, t1) in the file's time unit
        #   "last_full_from_state"  last complete cycle, cycles starting at "cycle_start_state"
        "cycle": "all",
        "cycle_start_state": "IVC",
        # reference (unloaded) volume V0: "min_pressure" | "min_volume" | number in mL
        "V0": "min_pressure",
        "data_legend": "Averaged iCPET data",
        "model_legend": "Calibrated Sphere",
    },
    "OUT_DIR": os.path.join(HERE, "identifiability_active_Guccione", "ton_toff_w_single_case"),

    # ------------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------------
    "TWITCH_PARAMETERIZATION": "ton_toff_w",
    # Starting / fixed values (units: Barye, s; times from the first data sample).
    "BASE_PARAMS": {
        "gamma": 1.0, "n": 1.0,  # FIXED
        "C": 6069,               # Guccione passive scaling [Barye]
        "sigma_max": 4.0e4,      # max active stress [Barye]
        "t_shift": -0.0392,      # FIXED twitch clock origin [s] (rest case APR29002, n = 1)
        "t_on": 0.04,            # half-activation time [s]
        "t_off": 0.30,           # half-relaxation time [s]
        "w1": 0.07,              # 10-90 % contraction duration [s]
        "w2": 0.15,              # 90-10 % relaxation duration [s]
        "b_f": 8.0, "b_t": 3.0,  # Guccione exponents (FIXED)
    },
    # Calibrated (and, if RUN_IDENTIFIABILITY, profiled) parameters.
    "FREE_PARAMS": ["C", "sigma_max", "t_on", "t_off", "w1", "w2"],
    # Parameter bounds to change (defaults: RAW_SPEC in the engine; t_on/t_off lie in (t_shift, T))
    "BOUNDS_OVERRIDES": {},

    # ------------------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------------------
    "RUN_IDENTIFIABILITY": False,  # False: calibration + fit figures only
    "DELTA_MMHG": 0.5,             # confidence region: RMSE <= best + DELTA_MMHG
    "GRID_N": 15,                  # points per profile
    # Profile range of log-scale entries (C, sigma_max, w1, w2): best * 10^(+/-GRID_HALF_WIDTH_LOG10),
    # per-entry values in the OVERRIDES (keyed by FREE_PARAMS name). t_on/t_off: best +/- 0.03 s.
    "GRID_HALF_WIDTH_LOG10": 0.6,
    "GRID_HALF_WIDTH_LOG10_OVERRIDES": {"sigma_max": 0.05, "C": 0.4},
    "N_WORKERS": None,             # parallel profiles; None = one per FREE_PARAMS entry
    "REUSE_SAVED_PROFILES": False, # True: skip entries whose profile CSV already exists
}

if __name__ == "__main__":   # required: the parallel workers re-import this file
    engine.cli(SETTINGS)
