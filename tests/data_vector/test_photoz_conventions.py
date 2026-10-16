"""Unit test: the runtime n(z) photo-z conventions.

The likelihood exposes two runtime knobs that control how the n(z)
table files are turned into the smooth distributions the Limber
integrals consume (both settable per likelihood in its yaml, both
defaulting to 0):

  photoz_interpolation_type: cosmolike interpolates n(z) in two stages,
      a spline through the file's nodes resampled once onto a fine
      uniform grid (stage 1), then a fast cubic lookup on that grid at
      every call (stage 2). This knob picks the stage-1 interpolant:
      0 = cubic spline (the default), 1 = linear, 2+ = Steffen monotone
      (no overshoot: n(z) can never ring below zero around a sharp
      feature in the table).
  photoz_zmid_convention: how the z column of the n(z) file is read:
      0 = Z_LOW (the default; the column holds left bin edges, so the
      tabulated value belongs at the cell center z + dz/2), 1 = Z_MID
      (the column holds the sample points themselves). The two
      readings differ by a rigid dz/2 shift of every distribution
      (dz = 0.01 in roman_example.nz), which is percent-level in
      cosmic shear.

This test evaluates the frozen cosmic-shear fiducial under five
settings in this pytest process: the default, each alternative, and the
default again. For each alternative it measures

    delta chi2 = delta^T C^-1 delta,
    delta = dv(alternative) - dv(default),

with C^-1 the masked inverse covariance from the compiled interface:
the same second-order construction the CFASTPT-vs-FASTPT sweep uses,
the chi2 the alternative would score against a dataset whose data vector
is the default prediction. A difference of two chi2 values against the
shipped data would instead change linearly with delta, through the slope
of the chi2 away from its minimum.

Running everything in one process is the point, not a convenience:
the n(z) table caches inside the compiled interface must notice a
runtime flag change and rebuild. A stale cache would make every
delta exactly zero (each alternative asserts a delta chi2 above a small
dead-flag floor), and the final return to the default must reproduce
the first vector (the round-trip assertion), so a cache that over- or
under-invalidates fails loudly here.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_photoz_conventions.py
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
# the class; assigning its return value to the usual module-level name
# is what makes pytest collect it here, with unchanged test ids.
TestPhotozConventions = u.photoz_conventions_suite(
    u, name="TestPhotozConventions")

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
