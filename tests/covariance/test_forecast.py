"""Check this project's galaxy/shear covariance interface and catalog inputs.

The shared check, check_project_forecast of
cosmolike_core/cocoa_covariance_testing.py, drives the survey adapter
covariance/roman_fourier_covariance.py through the compiled bindings. It
verifies the full layout lengths of both measurement spaces, that a
refinement of the accuracy keeps the measurement bins fixed, that the G,
SSC, cNG and total matrices of a measured subset are finite and symmetric
in real and Fourier space, that every component is bitwise identical with
one and eight OpenMP threads, that the subset's total has positive
variances in every direction, and that the saved archive reads back.

The check uses small numerical settings and a subset of rows, so it does
not certify survey convergence. Run it separately from tests/data_vector
(the two sectors initialize different compiled-library state), with the
project built with covariance support:

    python -m pytest projects/roman_fourier/tests/covariance
"""

from pathlib import Path
import sys

# project = projects/roman_fourier; the inserts put first on the module
# search path the core (cocoa_covariance_testing), interface/ (the
# compiled module) and covariance/ (the survey adapter).
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project.parents[1]/"external_modules/code/cosmolike_core"))
sys.path.insert(0, str(project/"interface"))
sys.path.insert(0, str(project/"covariance"))

from cocoa_covariance_testing import check_project_forecast
import cosmolike_roman_fourier_interface as ci
import roman_fourier_covariance as survey


def test_forecast_adapter(tmp_path):
    """Real and Fourier components repeat at one/eight threads and save intact.

    Arguments:
      tmp_path = a fresh temporary folder; pytest fills an argument with
                 this name automatically (a built-in fixture).

    Raises:
      AssertionError, from check_project_forecast, naming the violated
      condition.

    Side effects:
      Runs CAMB once, replaces this process's cosmolike state, and writes
      temporary archives into tmp_path.
    """
    # Full lengths: (36 + 55 + 8) x 15 = 1485 Fourier entries (shear E modes
    # only) and (2 x 36 + 55 + 8) x 15 = 2025 real-space entries (xi+ and
    # xi- for shear), in the order (real, Fourier).
    check_project_forecast(
        interface=ci, survey=survey, expected_sizes=(2025, 1485), directory=tmp_path,
    )
