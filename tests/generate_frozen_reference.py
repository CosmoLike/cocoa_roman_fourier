"""Maintainer tool: (re)create the frozen state the unit tests run on.

Running this redefines what the tests protect, so it refuses to run
without the explicit --overwrite flag. Only run it when a change to
the data vectors, n(z), covariance, examples, or likelihood defaults
is deliberate, and review the printed chi2 values before committing:
they become the new references the tests compare against.

The machinery and its documentation live in
cosmolike_core/cocoa_testing_freeze.py (freeze_main lists the modes
and the stage order); this script keeps the project's data:
the generated-vector roster and the per-mask
descriptor variants.

Usage (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python ./projects/roman_fourier/tests/generate_frozen_reference.py --overwrite

Incremental modes on an existing frozen state (each verifies the
state against the manifest, then re-pins it):
  --tatt-masks rewrites the per-mask descriptors;
  --baryons regenerates the feedback vectors.
"""

import os

# The references must be produced under the same OpenMP thread count
# the tests enforce, and before any cobaya/cosmolike import.
os.environ["OMP_NUM_THREADS"] = "4"

import sys

# __file__ is this script's own path; insert(0, ...) puts the tests/
# folder first on the module search path, so the import below finds
# the shim no matter where the script was launched from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

# The data vectors --overwrite regenerates:
# {descriptor dataset: (generating example, TATT flag)},
# built from the shim's generator pairing.
VECTORS = {name: (example, True)
           for name, example in u.TATT_GENERATORS.items()}

# The --mask reruns of the comparison sweeps read one frozen TATT
# dataset descriptor per scale-cut mask: identical to the base TATT
# descriptor except for its mask_file line. The harness finds a variant
# by the name <tatt_dataset stem>_<mask>.dataset, for each non-default
# name of fastpt_masks in cocoa_test_utils. variant -> (base, mask file).
TATT_MASK_VARIANTS = {
    "tatt_roman_fourier_ones.dataset": ("tatt_roman_fourier.dataset", "ones.mask"),
}


if __name__ == "__main__":
    sys.exit(u.freeze_main(
        u, script_file=__file__, usage=__doc__,
        vectors=VECTORS,
        mask_variants=TATT_MASK_VARIANTS,
    ))
