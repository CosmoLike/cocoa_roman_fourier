"""Skip the covariance tests when the project lacks covariance support.

pytest imports this conftest.py before it collects the tests of
tests/covariance/. Covariance generation is optional at build time: with
IGNORE_COSMOLIKE_ROMAN_FOURIER_COVARIANCE set, the compiled interface has
no covariance bindings, and every test of this folder is reported as
skipped with the instruction to rebuild.
"""

from pathlib import Path
import sys

# project = projects/roman_fourier (parents[2] climbs out of covariance/
# and tests/); insert(0, ...) puts interface/ first on the module search
# path, so the import below finds the compiled module.
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/"interface"))

import pytest
import cosmolike_roman_fourier_interface as ci


@pytest.fixture(scope="session", autouse=True)
def covariance_build():
    """Skip this sector when covariance generation was left out of the build.

    A fixture is a function pytest runs around the tests; scope="session"
    runs it once per pytest run, and autouse=True applies it to every test
    of this folder without the test naming it. The build sets the module
    attribute has_covariance (True or False); getattr returns False when
    the attribute is absent.

    Returns:
      nothing.

    Raises:
      pytest's skip exception, with the rebuild instruction, when the
      covariance bindings are missing; pytest then reports the tests of
      this folder as skipped, not passed.
    """
    if not getattr(ci, "has_covariance", False):
        pytest.skip(
            "Covariance generation is disabled. Unset "
            "IGNORE_COSMOLIKE_ROMAN_FOURIER_COVARIANCE after start_cocoa.sh, "
            "then recompile this project."
        )
