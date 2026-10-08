"""Unit test: non-Limber galaxy clustering (adopt_limber_gg).

The galaxy clustering (gg) spectrum C_ell^gg enters the data vector
through w(theta) in real space and directly in Fourier space. The Limber
approximation reduces the exact projection, a double integral over the
two radial kernels with spherical Bessel functions, to one integral along
the line of sight with P(k) evaluated at k = (ell + 1/2)/chi; it is
accurate when the kernels vary slowly on the radial scale chi/ell, so it
degrades at low ell and for narrow kernels. The likelihood yaml key
adopt_limber_gg chooses how C_ell^gg is computed:

  adopt_limber_gg: 0 (the default): below ell = 150 the exact
      projection, computed by cosmolike's C_cl_tomo with the split of
      Fang, Krause, Eifler & MacCrann (arXiv:1911.11947): an FFTLog
      integral of the linear power spectrum plus, in Limber, what
      linear theory misses. In Fourier space each band center takes
      the Limber value plus the non-Limber correction interpolated
      linearly between the two integer multipoles around it.
  adopt_limber_gg: 1: the Limber approximation at every multipole.

The lens galaxy redshift distributions are narrow, so the Limber
approximation fails at low ell for the clustering auto spectra; this
project defaults to the exact projection because the delta chi2 below
is too large to absorb. This test measures what Limber would cost.

It evaluates the frozen 3x2pt fiducial (NLA) three times in this pytest
process (one model per setting, all with example2's dimensions, so the
flag must reach the compiled library and its caches must rebuild): the
default, the other setting and the default again. It computes

    delta chi2 = delta^T C^-1 delta,
    delta = dv(non-Limber) - dv(Limber),

with C^-1 the masked inverse covariance: the chi2 a Limber model would
score against a data set generated with non-Limber clustering. It prints
the total and the contribution of each lens bin, delta_b^T C^-1 delta_b
with delta_b the bin's own block of delta (zero elsewhere); the cross
terms between bins are left out, so the contributions need not add up to
the total.

Assertions:
  1. delta chi2 is above a dead-flag floor: the flag reaches the C code
     and the clustering cache notices the change (a stale cache gives
     zero);
  2. only clustering entries change: cosmic shear, galaxy-galaxy
     lensing, and every other block are bitwise equal between the two
     evaluations;
  3. switching back to the default reproduces the first data vector
     bitwise;
  4. delta chi2 matches the value measured for this project
     (DCHI2_MEASURED below) to 5%: a change in the non-Limber code, the
     kernels, or the covariance shows up here;
  5. the default evaluation reproduces the frozen reference chi2
     (checked last, so a stale snapshot cannot hide checks 1-4).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_nonlimber_gg.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process (this
# test builds its models in the pytest process itself).
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import time
import unittest

# The shim cocoa_test_utils.py lives in the parent folder tests/; putting
# that folder first on the module search path finds this project's copy
# (every project names its shim the same) under pytest or direct runs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

EXAMPLE = "example2"
REFERENCE_KEY = "example2_nla"

# adopt_limber_gg of the likelihood yamls of this project
DEFAULT = 0

# (report tag, adopt_limber_gg): the default, the other setting, the
# default again; 1 - DEFAULT is the other value of the 0/1 flag
_NAME = {0: "non-Limber", 1: "Limber"}
SETTINGS = (
    (f"{_NAME[DEFAULT]} (default)", DEFAULT),
    (_NAME[1 - DEFAULT], 1 - DEFAULT),
    (f"{_NAME[DEFAULT]} again (round trip)", DEFAULT),
)

# A stale clustering cache gives delta chi2 = 0 exactly; the floor is
# orders of magnitude below the measured value, so it only catches a
# dead flag.
DCHI2_FLOOR = 1.0e-6

# The delta chi2 measured for this project (macOS, arm64), and the
# relative band that assertion 4 allows around it.
DCHI2_MEASURED = 3.445
DCHI2_RTOL = 0.05


class TestNonLimberGG(unittest.TestCase):
    """Limber vs non-Limber galaxy clustering on the frozen fiducial."""

    @classmethod
    def setUpClass(cls):
        """Check the environment and the frozen state once, before the tests.

        Moves to ROOTDIR, verifies every frozen file against the SHA-256
        manifest and stores the frozen reference chi2 values in
        cls.reference (cls is the class itself, shared by its tests).

        Raises:
          RuntimeError outside an activated Cocoa shell; AssertionError
          when a frozen file differs from the manifest.
        """
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def test_nonlimber_gg(self):
        """Measure the Limber cost in clustering and run assertions 1-5.

        Raises:
          AssertionError at the first violated assertion (numbered as in
          the module docstring).

        Side effects:
          Builds three example2 models in this process, replacing
          cosmolike's global state, and prints the report.
        """
        import numpy as np
        import cosmolike_roman_fourier_interface as ci

        vectors = {}
        chi2s = {}
        icov = None
        sizes = None
        nlen = None
        for tag, flag in SETTINGS:
            print(f"  building model ({EXAMPLE}, NLA, {tag}) ...",
                  flush=True)
            info = u.load_frozen_info(EXAMPLE, tatt=False)
            name = u.EXAMPLES[EXAMPLE]["likelihood"]
            info["likelihood"][name]["adopt_limber_gg"] = flag
            model = u.make_model(info)
            point = u.build_point(model, EXAMPLE, tatt=False)
            start = time.perf_counter()
            chi2s[tag] = u.evaluate_chi2(model, point)
            elapsed = time.perf_counter() - start
            print(f"    chi2 = {chi2s[tag]:.6f}  ({elapsed:.1f} s, "
                  "first evaluation of the model)", flush=True)
            # full-precision model vector (the printed one keeps 9 digits)
            vectors[tag] = np.array(ci.compute_data_vector_masked())
            if icov is None:
                # all settings share one data set (mask, covariance), so
                # the first build's masked inverse covariance serves all
                icov = np.array(ci.get_inv_cov_masked())
                like = model.likelihood[name]
                if hasattr(ci, "compute_data_vector_3x2pt_real_sizes"):
                    sizes = ci.compute_data_vector_3x2pt_real_sizes()
                    nlen = int(like.ntheta)
                else:
                    sizes = ci.compute_data_vector_3x2pt_fourier_sizes()
                    nlen = int(like.ncl)

        # tags maps each flag value to its report tag (first two settings
        # only), so delta below is non-Limber (0) minus Limber (1)
        tags = {flag: tag for tag, flag in SETTINGS[:2]}
        dv_default = vectors[SETTINGS[0][0]]
        delta = vectors[tags[0]] - vectors[tags[1]]
        dchi2 = float(delta @ icov @ delta)

        # the clustering block follows cosmic shear and galaxy-galaxy
        # lensing in every probe combination (3x2pt, 2x2pt, 6x2pt):
        # entries [gg0, gg1), one block of nlen entries per lens bin
        gg0 = int(sizes[0]) + int(sizes[1])
        gg1 = gg0 + int(sizes[2])
        nbins = int(sizes[2]) // nlen

        print(f"\n  delta chi2 report ({EXAMPLE}, NLA):")
        print(f"    {SETTINGS[0][0]}: chi2 = {chi2s[SETTINGS[0][0]]:.6f} "
              f"(frozen reference {self.reference[REFERENCE_KEY]:.6f})")
        print(f"    {SETTINGS[1][0]}: chi2 = {chi2s[SETTINGS[1][0]]:.6f}")
        print(f"    delta^T C^-1 delta = {dchi2:.4f} "
              f"(measured {DCHI2_MEASURED:.4f})")
        print("    per lens bin (the bin's block alone):")
        rows = []
        for b in range(nbins):
            block = np.zeros_like(delta)
            sl = slice(gg0 + b*nlen, gg0 + (b + 1)*nlen)
            block[sl] = delta[sl]
            rows.append((float(block @ icov @ block), b))
        # largest contribution first; the printout stops at the first one
        # below 0.1% of the total
        for contribution, b in sorted(rows, reverse=True):
            if contribution < 1.0e-3*max(dchi2, DCHI2_FLOOR):
                break
            print(f"      lens bin {b:<14d} {contribution:.4f}")

        self.assertGreater(
            dchi2, DCHI2_FLOOR,
            "non-Limber clustering did not change the data vector: the "
            "adopt_limber_gg flag did not reach the C code, or the "
            "clustering cache did not rebuild")

        outside = np.concatenate((delta[:gg0], delta[gg1:]))
        self.assertTrue(
            np.all(outside == 0.0),
            "entries outside the clustering block changed with "
            "adopt_limber_gg")

        self.assertTrue(
            np.array_equal(vectors[SETTINGS[-1][0]], dv_default),
            "returning to the default did not reproduce the first data "
            "vector bit for bit; the clustering cache did not rebuild "
            "cleanly")

        self.assertLess(
            abs(dchi2/DCHI2_MEASURED - 1.0), DCHI2_RTOL,
            f"delta chi2 = {dchi2:.4f} differs from the measured "
            f"{DCHI2_MEASURED:.4f} by more than {DCHI2_RTOL:.0%}")

        # last, so that a stale frozen snapshot does not hide the four
        # checks above
        self.assertLess(
            abs(chi2s[SETTINGS[0][0]] - self.reference[REFERENCE_KEY]),
            u.CHI2_TOLERANCE,
            f"{SETTINGS[0][0]}: chi2 = {chi2s[SETTINGS[0][0]]:.6f} vs frozen "
            f"reference {self.reference[REFERENCE_KEY]:.6f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
