"""Unit test: sector-wise cache invalidation (the parameter ladder).

Cosmolike caches every expensive stage and rebuilds it only when the key
of its sector changes: cosmology (distances, growth, power-spectrum
tables), intrinsic alignment (FAST-PT tables under TATT), photo-z shifts
(n(z) splines, lens efficiencies) and the shear calibrations (a pure
data-vector rescale). A partial-invalidation bug, where one sector's
update fails to rebuild a cached C value (a static variable, which keeps
its value between calls) that another sector consumes, produces silently
wrong data vectors only in mixed update sequences, which the per-point
tests never exercise.

The test walks a deterministic ladder in one process on the example2
(3x2pt) configuration, evaluating the model after every step. Each
sector's later steps keep the earlier sectors at their last values, so
the ladder ends at one well-defined point:

    3 x cosmology-only steps   (omegam, H0, As_1e9)
    3 x IA-only steps          (A1; under TATT also A2 and BTA)
    3 x source-photo-z steps   (every DZ_S shift)
    3 x lens-photo-z steps     (every DZ_L shift)
    3 x shear-calibration steps (every M)

A sector in which the configuration samples no parameter drops out of
the ladder. In this project the lenses are the source sample, so the lens
shifts DZ_L follow DZ_S and there is no separate lens-photo-z rung.

The test records the final data vector, evaluates one scramble point
(every sector moved at once, galaxy bias included; its chi2 is
discarded) and returns to the ladder's final point: the pipeline must
reproduce the recorded vector bit for bit. A second model instance walks
the mirrored ladder (M -> DZ_L -> DZ_S -> IA -> cosmology) to the same
final point: the answer must depend on the point, never on the order of
the updates.

Assertions, in each intrinsic-alignment model (NLA and TATT; the TATT
ladder exercises the FAST-PT rebuilds that NLA never touches):
  1. every ladder step changes the data vector (a dead sector flag
     would pass the later checks vacuously);
  2. each M-only step rescales the masked vector by the analytic
     (1+m_i)(1+m_j) block factors to 1e-12 relative: cosmic shear
     C_ell^EE by both source bins' factors, galaxy-galaxy lensing
     C_ell^gs by the source bin's factor, clustering C_ell^gg by none;
  3. a no-op update (evaluating the final point again) leaves the
     vector bitwise unchanged;
  4. after the scramble, returning to the ladder's final point
     reproduces the recorded vector and chi2 bit for bit;
  5. the mirrored-order instance lands on the same final vector bit
     for bit.

Every evaluation forces a full recomputation (cobaya's cache is
bypassed), so each assertion tests cosmolike's own invalidation, not
cobaya's memoization (its reuse of results for repeated inputs).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_cache_consistency.py
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
TestCacheConsistency = u.cache_ladder_suite(
    u, name="TestCacheConsistency")

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
