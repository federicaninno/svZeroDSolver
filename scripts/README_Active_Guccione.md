# Active Guccione sphere calibration

- `Calibration_Identifiability_Sphere_Active_Guccione.py`: engine (calibration,
  profile likelihoods, FIM, figures). Not meant to be edited.
- `Run_Calibration_Identifiability_Sphere_Active_Guccione_TonToff_SingleCase.py`:
  settings for one case, calibrating C, sigma_max, t_on, t_off, w1, w2. Edit
  this file (case name, data layout, free parameters) and run it.

## Solver

These scripts need the Guccione ChamberSphere block of **this branch**
(`my-dev-branch`); the `pysvzerod` from PyPI or from other branches will not
work. Build the Python module with CMake (`pip install .` does not work here),
using the Python that will run the scripts:

```bash
git clone -b my-dev-branch https://github.com/federicaninno/svZeroDSolver.git
cd svZeroDSolver
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$(which python3)
make pysvzerod
cp python/pysvzerod*.so "$(python3 -c 'import site; print(site.getsitepackages()[0])')"
pip install numpy scipy pandas matplotlib
```

## Data

An example case is included: `scripts/fw/data_example.csv` (one header line;
columns: time [s], pressure [mmHg], volume [mL]); the wrapper runs it as is.
For another case, put its CSV in `scripts/fw/` and set `CASE` in the wrapper to
the file name without `.csv`; for another file layout, change its `DATA` block.

## Run

```bash
cd scripts
python3 Run_Calibration_Identifiability_Sphere_Active_Guccione_TonToff_SingleCase.py
python3 Run_Calibration_Identifiability_Sphere_Active_Guccione_TonToff_SingleCase.py --plot-only   # re-plot saved profiles
```

Outputs go to `scripts/identifiability_active_Guccione/ton_toff_w_single_case/<case>/`.
Set `RUN_IDENTIFIABILITY = True` in the wrapper to also compute profile
likelihoods and the FIM (many more solver runs).
