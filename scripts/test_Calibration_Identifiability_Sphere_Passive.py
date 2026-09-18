"""
Regression tests for Calibration_Identifiability_Sphere_Passive.py.

Calibration_Identifiability_Sphere_Passive.py is configured by hand-editing
constants in its "USER SETTINGS" block (deliberately -- that's the point of
the script), not through a programmatic API. So these tests run the *actual
script* as a subprocess, splicing in a complete, self-contained settings
block for each scenario (anchored on the stable "# USER SETTINGS" / "# RAW
PARAMETER REGISTRY" section-header comments, never on the current values
inside that block -- those change every time someone edits the file by
hand). That means these tests exercise the script exactly the way you
actually run it, and stay correct no matter what config you currently have
saved in the file.

Run after any change to Calibration_Identifiability_Sphere_Passive.py:
    /Users/federicaninno/venv311/bin/python -m pytest scripts/test_Calibration_Identifiability_Sphere_Passive.py -v

The reference numbers below were captured from runs verified by hand during
development. If a deliberate change to the science legitimately moves them,
update the reference value in the same commit that explains why -- these
are a guardrail against *accidental* regressions, not a spec set in stone.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "Calibration_Identifiability_Sphere_Passive.py"
REPO_ROOT = SCRIPT.parents[1]
PYTHON = sys.executable

START_MARKER = "# USER SETTINGS\n# ============================================================\n"
END_MARKER = "\n# ============================================================\n# RAW PARAMETER REGISTRY"

SETTINGS_TEMPLATE = """
MATERIAL_MODEL = {material_model!r}

JSON_PATH = {json_path!r}

OUT_DIR = {out_dir!r}

NH_BASE_PARAMS = {nh_base_params!r}
HO_BASE_PARAMS = {ho_base_params!r}

BASE_PARAMS = dict(NH_BASE_PARAMS if MATERIAL_MODEL == "NH" else HO_BASE_PARAMS)

FREE_PARAMS = {free_params!r}

INIT_OVERRIDES = {init_overrides!r}

RUN_IDENTIFIABILITY = {run_identifiability!r}

DELTA_MMHG = {delta_mmhg!r}

GRID_HALF_WIDTH_LOG10 = {grid_half_width_log10!r}
GRID_N = {grid_n!r}

RUN_TAG = {run_tag!r}
"""

DEFAULT_NH_BASE_PARAMS = {"gamma": 0.5, "n": 1.0, "W1": 1.69e5, "W2": 0.0, "eta": 0.0}
DEFAULT_HO_BASE_PARAMS_NO_FIBERS = {
    "gamma": 0.5, "n": 1.0,
    "a": 14720.0, "b": 33.39,
    "a4f": 0.0, "b4f": 0.0,
    "a4s": 0.0, "b4s": 0.0,
    "eta": 0.0,
}
DEFAULT_HO_BASE_PARAMS_REAL_FIBERS = {
    "gamma": 0.5, "n": 1.0,
    "a": 14720.0, "b": 33.39,
    "a4f": 30080, "b4f": 88.39,
    "a4s": 0.942, "b4s": 14.30,
    "eta": 50.0,
}

NH_JSON = str(REPO_ROOT / "build" / "chamber_sphere_NH_calibr_E_100kPa_4calibration_notaunovelonostressnoradius.json")
HO_JSON = str(REPO_ROOT / "build" / "chamber_sphere_HO_passive_calibr_realfibers_correctParams_Nikou.json")


def make_script(tmp_path, *, material_model, json_path, free_params,
                 nh_base_params=None, ho_base_params=None, init_overrides=None,
                 run_identifiability=True, delta_mmhg=0.5, grid_half_width_log10=0.6,
                 grid_n=80, run_tag=None, out_dir=None, filename="under_test.py"):
    """Write a copy of Calibration_Identifiability_Sphere_Passive.py with a complete,
    self-contained USER SETTINGS block, independent of whatever is
    currently saved in the real file."""
    settings = SETTINGS_TEMPLATE.format(
        material_model=material_model,
        json_path=json_path,
        out_dir=str(out_dir if out_dir is not None else tmp_path),
        nh_base_params=nh_base_params if nh_base_params is not None else DEFAULT_NH_BASE_PARAMS,
        ho_base_params=ho_base_params if ho_base_params is not None else DEFAULT_HO_BASE_PARAMS_REAL_FIBERS,
        free_params=free_params,
        init_overrides=init_overrides if init_overrides is not None else {},
        run_identifiability=run_identifiability,
        delta_mmhg=delta_mmhg,
        grid_half_width_log10=grid_half_width_log10,
        grid_n=grid_n,
        run_tag=run_tag,
    )

    src = SCRIPT.read_text()
    start_idx = src.index(START_MARKER) + len(START_MARKER)
    end_idx = src.index(END_MARKER)
    assert start_idx < end_idx, "USER SETTINGS section markers not found -- did the script's structure change?"
    new_src = src[:start_idx] + settings + src[end_idx:]

    out = tmp_path / filename
    out.write_text(new_src)
    return out


def run_script(script_path, extra_args=None, timeout=300):
    return subprocess.run(
        [PYTHON, str(script_path), *(extra_args or [])],
        capture_output=True, text=True, timeout=timeout,
    )


def best_fit_values(stdout, labels):
    """Parse '  <label> = <value>' lines out of the 'Best fit' block."""
    values = {}
    for label in labels:
        m = re.search(rf"^\s*{re.escape(label)}\s*=\s*([0-9.eE+-]+)", stdout, re.MULTILINE)
        assert m, f"could not find a best-fit line for {label!r} in stdout:\n{stdout}"
        values[label] = float(m.group(1))
    return values


# ===========================================================================
# Reference numbers for known-good configurations
# ===========================================================================

def test_ho_gamma_n_matches_reference(tmp_path):
    """Original geometry-only analysis: HO model, real fibers/viscosity fixed."""
    script = make_script(
        tmp_path, material_model="HO", json_path=HO_JSON, free_params=["gamma", "n"],
    )
    result = run_script(script)
    assert result.returncode == 0, result.stderr

    vals = best_fit_values(result.stdout, ["gamma", "n"])
    assert vals["gamma"] == pytest.approx(0.371862, rel=1e-4)
    assert vals["n"] == pytest.approx(1.3251, rel=1e-4)
    assert "RMSE=0.9696 mmHg" in result.stdout
    assert "gamma in [0.2667, 0.5006]" in result.stdout
    assert result.stdout.count("IDENTIFIABLE") == 2  # gamma AND n, no "NOT constrained"


def test_ho_single_compound_matches_reference(tmp_path):
    """One tied product active (gamma*a): gamma still a real, reconstructed number."""
    script = make_script(
        tmp_path, material_model="HO", json_path=HO_JSON, free_params=["gamma*a", "n"],
    )
    result = run_script(script)
    assert result.returncode == 0, result.stderr
    assert "PRODUCT_MODE" not in result.stdout  # only one tied product -- standalone-gamma path

    vals = best_fit_values(result.stdout, ["gamma*a", "n"])
    assert vals["gamma*a"] == pytest.approx(5075.66, rel=1e-3)
    assert vals["n"] == pytest.approx(1.3311, rel=1e-3)
    assert "RMSE=0.9708 mmHg" in result.stdout


def test_nh_product_mode_matches_reference(tmp_path):
    """Two tied products at once (gamma*W1, gamma*eta): PRODUCT_MODE physics
    path, gamma eliminated entirely -- must match optimization_HO_products_
    ...py's own calibration to within a few percent."""
    script = make_script(
        tmp_path, material_model="NH", json_path=NH_JSON,
        free_params=["gamma*W1", "gamma*eta", "n"],
        init_overrides={"eta": 500.0},
    )
    result = run_script(script)
    assert result.returncode == 0, result.stderr
    assert "PRODUCT_MODE: 2 tied gamma products" in result.stdout

    vals = best_fit_values(result.stdout, ["gamma*W1", "gamma*eta", "n"])
    assert vals["gamma*W1"] == pytest.approx(60098.8, rel=1e-3)
    assert vals["gamma*eta"] == pytest.approx(6849.37, rel=1e-3)
    assert vals["n"] == pytest.approx(2.05824, rel=1e-3)
    assert "RMSE=0.6279 mmHg" in result.stdout


# ===========================================================================
# Validation guardrails
# ===========================================================================

def test_rejects_bare_gamma_mixed_with_a_product(tmp_path):
    script = make_script(
        tmp_path, material_model="NH", json_path=NH_JSON,
        free_params=["gamma", "gamma*eta", "n"],
        init_overrides={"eta": 500.0},
    )
    result = run_script(script)
    assert result.returncode != 0
    assert "cannot include a bare 'gamma' entry" in result.stderr


def test_rejects_nonzero_w2_in_product_mode(tmp_path):
    bad_nh_base = dict(DEFAULT_NH_BASE_PARAMS, W2=5.0)
    script = make_script(
        tmp_path, material_model="NH", json_path=NH_JSON,
        free_params=["gamma*W1", "gamma*eta", "n"],
        nh_base_params=bad_nh_base, init_overrides={"eta": 500.0},
    )
    result = run_script(script)
    assert result.returncode != 0
    assert "W2 is still fixed at a nonzero value" in result.stderr


def test_rejects_nonzero_fibers_in_ho_product_mode(tmp_path):
    bad_ho_base = dict(DEFAULT_HO_BASE_PARAMS_REAL_FIBERS)
    script = make_script(
        tmp_path, material_model="HO", json_path=HO_JSON,
        free_params=["gamma*a", "gamma*eta", "n"],
        ho_base_params=bad_ho_base, init_overrides={"eta": 500.0},
    )
    result = run_script(script)
    assert result.returncode != 0
    assert "fiber terms (a4f, a4s) are still fixed at a nonzero value" in result.stderr


def test_rejects_disallowed_nongamma_product(tmp_path):
    script = make_script(
        tmp_path, material_model="HO", json_path=HO_JSON, free_params=["b*n"],
    )
    result = run_script(script)
    assert result.returncode != 0
    assert "only 'gamma*<partner>' products are allowed" in result.stderr


# ===========================================================================
# RUN_IDENTIFIABILITY toggle
# ===========================================================================

def test_run_identifiability_false_skips_profiling(tmp_path):
    script = make_script(
        tmp_path, material_model="HO", json_path=HO_JSON, free_params=["gamma", "n"],
        run_identifiability=False,
    )
    result = run_script(script)
    assert result.returncode == 0, result.stderr
    assert "RUN_IDENTIFIABILITY = False" in result.stdout
    assert "IDENTIFIABLE" not in result.stdout  # no profiling verdicts printed

    assert (tmp_path / "identifiability_HO_gamma_n_fit_only.png").exists()
    assert not (tmp_path / "identifiability_HO_gamma_n_summary.csv").exists()


def test_plot_only_without_prior_full_run_gives_clear_error(tmp_path):
    script = make_script(
        tmp_path, material_model="HO", json_path=HO_JSON, free_params=["gamma", "n"],
        run_identifiability=False,
    )
    result = run_script(script, extra_args=["--plot-only"])
    assert result.returncode != 0
    assert "nothing to re-plot" in result.stderr


def test_plot_only_reuses_saved_csvs(tmp_path):
    script = make_script(
        tmp_path, material_model="HO", json_path=HO_JSON, free_params=["gamma", "n"],
    )
    first = run_script(script)
    assert first.returncode == 0, first.stderr
    profiles_png = tmp_path / "identifiability_HO_gamma_n_profiles_only.png"
    assert profiles_png.exists()
    mtime_after_full_run = profiles_png.stat().st_mtime

    second = run_script(script, extra_args=["--plot-only"])
    assert second.returncode == 0, second.stderr
    assert profiles_png.stat().st_mtime >= mtime_after_full_run


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
