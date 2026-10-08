"""Unit test 18: the race check with EuclidEmulator2 on.

EuclidEmulator2 (EE2) predicts the nonlinear boost P_nl/P_lin of the
matter power spectrum, trained on N-body simulations. Cocoa pins a
modified EuclidEmulator2 (the EE2_GIT_COMMIT of
set_installation_options.sh): OpenMP threading, a 1,010-redshift
capacity, the get_boost2 API with a pre-built emulator, memory-leak
fixes, and a bilinear interpolation with a border fix (the
repository's own README, external_modules/code/euclidemu2/README.md,
documents them). This test runs the race check with that build
supplying the nonlinear P(k) (non_linear_emul: 1) on the cosmic-shear
configuration (example1, NLA): the fiducial evaluated fresh and again as
the 10th of 10 cosmologies on one model instance. EE2's compute is
OpenMP-threaded, so leaked state or a thread race inside it shifts the
second fiducial value; the two must agree within RACE_TOLERANCE (1e-4).

The gate that compiles the pre-modification EE2 (commit ff59f66)
side by side and scores the installed build against it runs in the
lsst_y1 project alone (its tests/data_vector/test_ee2.py, test 18); the
emulator build is project-independent, so that comparison is not
repeated here.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_ee2.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The shim cocoa_test_utils.py lives in the parent folder tests/; putting
# that folder first on the module search path finds this project's copy
# (every project names its shim the same) under pytest or direct runs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u


class TestEE2Race(unittest.TestCase):
    """Test 18, with the frozen-state verification.

    setUpClass runs once before the test: it moves to ROOTDIR and
    verifies every frozen file against the SHA-256 manifest. No
    frozen reference chi2 is loaded: the test compares one build
    against itself, so the frozen state only supplies the
    configuration and the data files.
    """

    @classmethod
    def setUpClass(cls):
        """Check the environment and the frozen state once, before the tests.

        Moves to ROOTDIR and verifies every frozen file against the SHA-256
        manifest.

        Raises:
          RuntimeError outside an activated Cocoa shell; AssertionError
          when a frozen file differs from the manifest.
        """
        u.require_cocoa_environment()
        u.verify_frozen()

    def test_x18_ee2_race_ten_in_a_row(self):
        """The fiducial with EE2 as 10th of 10 matches a fresh run.

        EE2's OpenMP-threaded compute runs inside every evaluation
        of the row, so a thread race or leaked state in it moves
        the second fiducial value. The x prefix marks a two-digit test
        number (unittest sorts method names alphabetically).

        Raises:
          RuntimeError when OMP_NUM_THREADS is not 4 (assert_omp_threads);
          AssertionError when the two chi2 values differ by
          RACE_TOLERANCE or more.
        """
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2("example1", tatt=False,
                                           ee2=True)
        u.report_race_test(
            18, "example1 (cosmic shear, NLA+EE2) race check: 10 "
            "cosmologies in a row", fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh "
                f"{fresh:.8f}")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
