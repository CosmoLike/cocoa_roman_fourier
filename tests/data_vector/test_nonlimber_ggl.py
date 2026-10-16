"""Unit test: non-Limber galaxy-galaxy lensing (adopt_limber_gs).

The galaxy-galaxy lensing (ggl) spectrum C_ell^gs enters the data vector
through gamma_t(theta) in real space and directly in Fourier space. The
Limber approximation reduces the exact projection, a double integral over
the two radial kernels with spherical Bessel functions, to one integral
along the line of sight with P(k) evaluated at k = (ell + 1/2)/chi; it is
accurate when the kernels vary slowly on the radial scale chi/ell, so it
degrades at low ell and for narrow kernels. The likelihood yaml key
adopt_limber_gs chooses how C_ell^gs is computed:

  adopt_limber_gs: 0 (the default): below ell = 150 the exact
      projection, computed by cosmolike's C_gs_tomo with the split of
      Fang, Krause, Eifler & MacCrann (arXiv:1911.11947): an FFTLog
      integral of the linear power spectrum plus, in Limber, what
      linear theory misses. In Fourier space each band center takes
      the Limber value plus the non-Limber correction interpolated
      linearly between the two integer multipoles around it.
  adopt_limber_gs: 1: the Limber approximation at every multipole.

The Limber approximation fails at low ell for the lens-source pairs
whose kernels overlap in redshift (lens bin = source bin, or the
source bin in front of the lens bin, where the signal is the intrinsic
alignment of the sources times the lens density); this project defaults
to the exact projection because the delta chi2 below is too large to
absorb (galaxy clustering has its own key, adopt_limber_gg; see
test_nonlimber_gg.py). This test measures what Limber would cost.

It evaluates the frozen 3x2pt fiducial (NLA) three times in this pytest
process (one model per setting, all with example2's dimensions, so the
flag must reach the compiled library and its caches must rebuild):
non-Limber, Limber and non-Limber again. It computes

    delta chi2 = delta^T C^-1 delta,
    delta = dv(non-Limber) - dv(Limber),

with C^-1 the masked inverse covariance: the chi2 the Limber model
would score against a data set generated with non-Limber ggl. It prints
the total and the contribution of each lens-source pair,
delta_p^T C^-1 delta_p with delta_p the pair's own block of delta (zero
elsewhere); the cross terms between pairs are left out, so the
contributions need not add up to the total.

Assertions:
  1. delta chi2 is above a dead-flag floor: the flag reaches the C code
     and the ggl cache notices the change (a stale cache gives zero);
  2. only ggl entries change: cosmic shear, clustering, and every other
     block are bitwise equal between the two evaluations;
  3. switching back to the default reproduces the first data vector
     bitwise;
  4. delta chi2 matches the value measured for this project
     (DCHI2_MEASURED below) to 5%: a change in the non-Limber code, the
     kernels, or the covariance shows up here;
  5. the default (non-Limber) evaluation reproduces the frozen
     reference chi2 (checked last, so a stale snapshot cannot hide
     checks 1-4).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_nonlimber_ggl.py
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
TestNonLimberGGL = u.nonlimber_suite(
    u, name="TestNonLimberGGL", kind="ggl",
    measured=1.217, default=0)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
