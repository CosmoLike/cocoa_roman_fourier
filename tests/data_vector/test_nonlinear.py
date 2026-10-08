"""Advisory checks NL1-NL2: Halofit vs EuclidEmulator2.

The likelihoods can source the nonlinear matter power from CAMB's
Takahashi halofit (non_linear_emul: 2, the setting of the frozen
configurations), a fitting formula calibrated on N-body simulations,
or from EuclidEmulator2 (EE2, non_linear_emul: 1), an emulator trained
on N-body simulations. NL1 evaluates the cosmic-shear data vector and
NL2 the 3x2pt data vector with both sources at ten fixed cosmologies
across the omegam/ns/As space (NONLINEAR_COMPARISON_POINTS; every other
parameter stays at the frozen fiducial) and reports, at each cosmology,
the chi2 of the Halofit vector against the EE2 vector (delta^T C^-1
delta; the EE2 vector is that cosmology's fiducial, so the baseline is
zero by construction). Each source runs in its own worker subprocess.

There is no pass/fail: the numbers say how much of the statistical
error budget the Halofit-vs-emulator difference consumes under the
chosen scale cuts, that is, whether Halofit is good enough for a real
data analysis with this mask. The checks read the --mask option of the
comparison sweeps (conftest.py): --mask=frozen (the contract mask, the
default) or --mask=ones (every data point kept). NL2 cannot run under
--mask=ones: with every data point kept the shipped 3x2pt covariance is
not positive definite and cosmolike aborts the model build
(IP::set_inv_cov), which pytest reports as an error of NL2.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_nonlinear.py

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_nonlinear.py --mask=ones
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


class TestHalofitVsEE2(unittest.TestCase):
    """Advisory checks NL1-NL2, sharing the frozen-state verification.

    setUpClass runs once before the tests: it moves to ROOTDIR and
    verifies every frozen file against the SHA-256 manifest. No
    frozen reference chi2 is loaded: the checks compare the two
    nonlinear-P(k) sources against each other, so the frozen state
    only supplies the configuration and the data files.
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

    def test_nl1_halofit_vs_ee2_cosmic_shear(self):
        """Cosmic shear: Halofit scored against EE2 at ten cosmologies.

        Advisory: the printed report is the product. The only
        assertion is structural: every cosmology must have produced
        a number.
        """
        mask = os.environ.get("COCOA_FASTPT_MASK", "frozen")
        dchi2s = u.halofit_vs_ee2_dchi2s("example1", mask=mask)
        u.report_nonlinear_comparison(
            f"NL1: example1 (cosmic shear, NLA, mask {mask}): HALOFIT "
            "vs EE2 at 10 fixed cosmologies", dchi2s)
        self.assertEqual(len(dchi2s), len(u.NONLINEAR_COMPARISON_POINTS))

    def test_nl2_halofit_vs_ee2_3x2pt(self):
        """3x2pt: Halofit scored against EE2 at ten cosmologies.

        Advisory: the printed report is the product. The only
        assertion is structural: every cosmology must have produced
        a number.
        """
        mask = os.environ.get("COCOA_FASTPT_MASK", "frozen")
        dchi2s = u.halofit_vs_ee2_dchi2s("example2", mask=mask)
        u.report_nonlinear_comparison(
            f"NL2: example2 (3x2pt, NLA, mask {mask}): HALOFIT vs "
            "EE2 at 10 fixed cosmologies", dchi2s)
        self.assertEqual(len(dchi2s), len(u.NONLINEAR_COMPARISON_POINTS))


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
