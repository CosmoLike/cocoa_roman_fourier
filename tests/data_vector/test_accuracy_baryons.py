"""Baryonic feedback accuracy checks BF1-BF7: default vs high accuracy.

Baryonic feedback (gas heated and expelled by supernovae and active
galactic nuclei) suppresses the matter power spectrum on small scales.
The bfmt theory block computes that suppression with one of several
methods, and the likelihood multiplies the nonlinear power spectrum by
it (external_baryon_suppression).

Each check evaluates the example1 configuration (cosmic shear, NLA) with
the bfmt theory block switched on for one feedback method, in three
steps: the default-settings model writes its own theory vector during
evaluation; that vector becomes the data of a temporary dataset (so the
default chi2 against it is zero by construction, and nothing is stored
in frozen/); the pushed-settings model evaluates at the same point
against it. Its chi2 is then the reported quantity,

    delta chi2 = chi2(high accuracy) - chi2(default)

a pure numerics (curvature) response at the minimum. The delta is
advisory like test_accuracy.py: the question answered is whether the
numerical error of the default settings stays harmless when the
nonlinear power spectrum carries a baryonic suppression. BF0 also runs
the one-knob-at-a-time scan with the Akino SP(k) method on, so a large
delta names the knob causing it.

The seven checks cover every method the bfmt theory block implements
(fb is the baryon fraction of a halo as a function of its mass, the
input of the SP(k) method):

  BF1. SP(k), power-law fb relation      BF2. SP(k), Akino et al. 2022
  BF3. SP(k), double power-law relation  BF4. BCEmu
  BF5. FlamingoBaryonResponseEmulator    BF6. BACCOemu
  BF7. BCemu2025

Each parameter point is fixed (the SP(k) points are pyspk's
documented examples; the emulator points are the fiducial values
quoted in the example yamls); the exact values live in
cocoa_test_utils.BARYON_METHODS.

Every configuration here is measurable by construction: BACCOemu's
check evaluates at omegab = 0.049, inside its omega_baryon training
box (the floor, 0.04001, sits exactly above the fiducial
omegab = 0.04), and the double-power-law point keeps the baryon
fraction inside SP(k)'s calibrated band over the full redshift grid
(pyspk's documented example exits it at z >~ 1.4). The exact points
live in cocoa_test_utils.BARYON_METHODS and
BARYON_POINT_OVERRIDES.

These checks complement test_accuracy.py. Two evaluations per check, one
of them at high accuracy: expect minutes per check. To run only this
file (from the Cocoa/ folder, cocoa environment active, start_cocoa.sh
sourced):

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_accuracy_baryons.py
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


class TestBaryonAccuracyAdvisory(unittest.TestCase):
    """Advisory checks BF1-BF7: accuracy with baryonic feedback on.

    setUpClass runs once: it moves to ROOTDIR and verifies every
    frozen file against the SHA-256 manifest before any physics runs.
    No frozen reference chi2 is loaded: each check builds its own
    reference vector.
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

    def _baryon_accuracy_check(self, name, baryon, label):
        """Compare default and high accuracy with one feedback method on.

        Arguments:
          name   = the advisory label (BF1-BF7) for the report.
          baryon = a label of cocoa_test_utils.BARYON_METHODS.
          label  = one line naming the feedback method.

        Returns:
          nothing; the delta is printed.

        Raises:
          AssertionError when the delta is not a finite number, the only
          assertion of these advisory checks.
        """
        # the default chi2 is zero by construction (the default
        # model produced the very vector it is compared with), so the
        # pushed evaluation's chi2 is the delta; only that is printed.
        # delta == delta is False only for NaN, which differs from itself.
        delta = u.baryon_accuracy_delta(baryon)
        self.assertTrue(
            delta == delta and abs(delta) != float("inf"),
            f"{name}: non-finite delta")
        print(f"""
{'-' * 66}
ACCURACY: {name}: {label}
  delta chi2 (high-default) = {delta:+.6f}
{'-' * 66}""", flush=True)

    def test_bf0_one_knob_at_a_time(self):
        """BF0: each accuracy knob alone, Akino SP(k) feedback on.

        Advisory: the same knob scan as test_accuracy.py, with the bfmt
        block computing the Akino SP(k) suppression and the chi2
        measured against that method's own generated vector. A knob
        whose delta rivals the all-knobs delta of BF2 is the driver
        of the numerical error under feedback.
        """
        print("", flush=True)
        for label, _, _ in u.ACCURACY_KNOBS:
            # each knob's chi2 against the on-the-fly vector is its
            # delta (the default against that vector is zero)
            delta = u.baryon_accuracy_delta("spk akino", knob=label)
            print(f"  KNOB {label:30s} delta chi2 = {delta:+12.6f}",
                  flush=True)

    def test_bf1_spk_power_law(self):
        """BF1: SP(k) with the power-law fb relation."""
        self._baryon_accuracy_check("BF1", "spk power law",
                                    "SP(k), power-law fb relation")

    def test_bf2_spk_akino(self):
        """BF2: SP(k) with the Akino et al. 2022 fb relation."""
        self._baryon_accuracy_check("BF2", "spk akino",
                                    "SP(k), Akino et al. 2022")

    def test_bf3_spk_double_power_law(self):
        """BF3: SP(k) with the double power-law fb relation."""
        self._baryon_accuracy_check("BF3", "spk double power law",
                                    "SP(k), double power-law fb relation")

    def test_bf4_bcemu(self):
        """BF4: BCEmu."""
        self._baryon_accuracy_check("BF4", "bcemu", "BCEmu")

    def test_bf5_flamingo(self):
        """BF5: FlamingoBaryonResponseEmulator."""
        self._baryon_accuracy_check("BF5", "flamingo",
                                    "FlamingoBaryonResponseEmulator")

    def test_bf6_baccoemu(self):
        """BF6: BACCOemu."""
        self._baryon_accuracy_check("BF6", "baccoemu", "BACCOemu")

    def test_bf7_bcemu2025(self):
        """BF7: BCemu2025."""
        self._baryon_accuracy_check("BF7", "bcemu2025", "BCemu2025")


if __name__ == "__main__":
    unittest.main(verbosity=2)
