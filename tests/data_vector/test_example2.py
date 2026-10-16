"""Unit tests 5-8: the 3x2pt likelihood on the frozen test data.

3x2pt combines three two-point statistics, here angular power spectra in
15 multipole bands: cosmic shear (C_ell^EE, shear E modes), galaxy-galaxy
lensing (C_ell^gs) and galaxy clustering (C_ell^gg). The tests evaluate
the roman_fourier.combo_3x2pt likelihood on the frozen copy of example2's
configuration (see cocoa_test_utils for what "frozen" means and why).
Every chi2 comes from a fresh worker subprocess. The four tests:

  5. chi2 at the frozen fiducial point, within CHI2_TOLERANCE (0.2) of
     the frozen reference value.
  6. race check: on one model, the fiducial evaluated fresh and again
     as the 10th of 10 cosmologies in a row must agree to
     RACE_TOLERANCE (1e-4). A disagreement means state leaked between
     evaluations or OpenMP threads raced (a race: threads that read and
     write shared memory in an unpredictable order).
  7. the same comparison as test 5 with the TATT intrinsic-alignment
     model (IA_model: 1) and roman_A2_1 = 0.05, roman_BTA_1 = 0.05,
     roman_A2_2 = -1.51541 replacing the NLA point's zeros.
  8. the same race check as test 6 with the TATT model.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_fourier/tests/data_vector
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The shared factory (cosmolike_core/cocoa_testing_suites.py) builds
# the chi2 + race test class for one frozen example; assigning its
# return value to the usual module-level name is what makes pytest
# collect it here, with unchanged test ids. The numbers name the
# methods and the printed reports; the label opens every report line.
TestExample2ThreeXTwo = u.example_suite(
    u, name="TestExample2ThreeXTwo", example="example2",
    label="example2 (3x2pt",
    numbers=("5", "6"), tatt_numbers=("7", "8"))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
