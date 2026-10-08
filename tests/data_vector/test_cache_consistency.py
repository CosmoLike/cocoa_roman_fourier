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
# this must run before any cobaya/cosmolike import in the process (this
# test builds its models in the pytest process itself).
os.environ["OMP_NUM_THREADS"] = "4"

import re
import sys
import unittest

# The shim cocoa_test_utils.py lives in the parent folder tests/; putting
# that folder first on the module search path finds this project's copy
# (every project names its shim the same) under pytest or direct runs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

EXAMPLE = "example2"

# Sector membership by sampled-parameter name. A parameter belongs to the
# first sector whose pattern re.search finds in its name (^ and $ anchor
# a pattern to the start and the end), and "other" matches any name, so
# every parameter falls in exactly one sector. bias moves only in the
# scramble step; other never moves (DELTAS has no entry for it).
SECTORS = (
    ("cosmo", re.compile(r"^(As_1e9|H0|ns|omegab|omegam|mnu|w|w0pwa)$")),
    ("ia", re.compile(r"_A1_|_A2_|_BTA_")),
    ("dz_source", re.compile(r"_DZ_S")),
    ("dz_lens", re.compile(r"_DZ_L")),
    ("m", re.compile(r"_M[0-9]+$")),
    ("bias", re.compile(r"_B1_|_B2_|_BMAG_")),
    ("other", re.compile(r".")),
)

# Ladder phases (sector order of the forward walk) and the per-step
# offsets: parameter value = fiducial + step * delta, deterministic.
# A DELTAS key is an exact name (cosmology) or a compiled pattern.
# NSTEP = steps per sector; SCRAMBLE_STEP puts every sector one step
# beyond the ladder; RESCALE_RTOL, the relative tolerance of the analytic
# shear-calibration rescale, sits far above the float64 rounding of the
# few multiplications involved (about 2e-16 each) and far below the
# change a stale cache produces.
PHASES = ("cosmo", "ia", "dz_source", "dz_lens", "m")
DELTAS = {
    "cosmo": {"omegam": 0.002, "H0": 0.2, "As_1e9": 0.02},
    "ia": {re.compile(r"_A1_1$"): 0.05, re.compile(r"_A1_2$"): 0.05,
           re.compile(r"_A2_1$"): 0.05, re.compile(r"_A2_2$"): 0.05,
           re.compile(r"_BTA_1$"): 0.05},
    "dz_source": {re.compile(r"_DZ_S"): 0.001},
    "dz_lens": {re.compile(r"_DZ_L"): 0.001},
    "m": {re.compile(r"_M[0-9]+$"): 0.005},
    "bias": {re.compile(r"_B1_"): 0.05},
}
NSTEP = 3
SCRAMBLE_STEP = 4  # every sector at step 4, bias included
RESCALE_RTOL = 1.0e-12


def _sector_of(name):
    """Return the sector of one sampled parameter.

    Arguments:
      name = a sampled-parameter name, e.g. "roman_DZ_S3".

    Returns:
      the name of the first SECTORS entry whose pattern is found in name.
    """
    for sector, pat in SECTORS:
        if pat.search(name):
            return sector
    return "other"


def _deltas_for(sector, names):
    """Return the per-step offsets of one sector's sampled parameters.

    Arguments:
      sector = a SECTORS name.
      names  = the sampled parameters of that sector.

    Returns:
      dict parameter name -> per-step offset for the names a DELTAS key of
      the sector matches; names without a match are left out and never
      move.
    """
    table = DELTAS.get(sector, {})
    out = {}
    for n in names:
        for key, d in table.items():
            # a conditional expression: a string key must equal the name,
            # a compiled pattern must be found in it
            if (key == n) if isinstance(key, str) else key.search(n):
                out[n] = d
                break
    return out


class TestCacheConsistency(unittest.TestCase):
    """Sector-ladder cache-invalidation check on the frozen fiducial."""

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

    def _point_at(self, fid, steps):
        """Return the ladder point with each sector moved to its step.

        Arguments:
          fid   = the fiducial point, parameter name -> value.
          steps = sector name -> step count (0 = fiducial).

        Returns:
          a new dict: fid with every parameter of a stepped sector at
          fiducial + step * delta (self.sector_deltas).
        """
        point = dict(fid)
        for sector, step in steps.items():
            for n, d in self.sector_deltas[sector].items():
                point[n] = fid[n] + step * d
        return point

    def _mpairs(self, like, np_):
        """Return the shear-calibration bins of every data-vector entry.

        Row k holds (i, j) for entry k of the full-length masked data
        vector: the source bins whose (1 + m) factors multiply it, -1 for
        none. Cosmic shear scales by both of its source bins,
        galaxy-galaxy lensing (and the CMB-lensing x shear spectrum ks,
        where a project has one) by its source bin, and clustering and the
        CMB-lensing spectra gk and kk by nothing. The code serves real- and
        Fourier-space projects: a Fourier data vector (this project) has
        ncl entries per block and one shear block per pair (E modes only);
        a real-space one has ntheta entries per block and two shear blocks
        per pair (xi+ and xi-).

        Arguments:
          like = the likelihood instance (ncl or ntheta, the bin counts
                 and ggl_exclude).
          np_  = the numpy module, passed in because _run_ladder imports
                 it locally.

        Returns:
          int array [n_data, 2]; entries the loops do not reach (the
          clustering block) keep (-1, -1).
        """
        import cosmolike_roman_fourier_interface as ci
        real = hasattr(ci, "compute_data_vector_3x2pt_real_sizes")
        sizes = [int(x) for x in
                 (ci.compute_data_vector_3x2pt_real_sizes() if real else
                  ci.compute_data_vector_3x2pt_fourier_sizes())]
        nlen = int(like.ntheta) if real else int(like.ncl)
        nsrc = int(like.source_ntomo)
        sspairs = [(i, j) for i in range(nsrc) for j in range(i, nsrc)]
        excluded = {(int(a), int(b)) for a, b in
                    (getattr(like, "ggl_exclude", None) or [])}
        gglpairs = [(zl, zs) for zl in range(int(like.lens_ntomo))
                    for zs in range(nsrc) if (zl, zs) not in excluded]
        fac = np_.zeros((sum(sizes), 2), dtype=int) - 1
        k = 0
        ssrep = sspairs + sspairs if real else sspairs  # xi_plus + xi_minus
        for (i, j) in ssrep:
            for t in range(nlen):
                fac[k] = (i, j)
                k += 1
        for (zl, zs) in gglpairs:
            for t in range(nlen):
                fac[k] = (-1, zs)
                k += 1
        k = sizes[0] + sizes[1] + sizes[2]  # skip clustering
        if len(sizes) > 3:  # 6x2pt: gk (no factor), ks (source), kk (none)
            k += sizes[3]
            for zs in range(nsrc):
                for t in range(nlen):
                    fac[k] = (-1, zs)
                    k += 1
        return fac

    def _run_ladder(self, tatt):
        """Walk the forward and the mirrored ladder and assert 1-5.

        Arguments:
          tatt = True runs the TATT variant (IA_model 1), False NLA.

        Returns:
          nothing; the final chi2 of each ladder is printed.

        Raises:
          AssertionError at the first violated assertion (numbered as in
          the module docstring).

        Side effects:
          Builds two example2 models in this process (equal dimensions,
          so the C layer accepts the second) and replaces cosmolike's
          global state.
        """
        import numpy as np
        import cosmolike_roman_fourier_interface as ci

        name = u.EXAMPLES[EXAMPLE]["likelihood"]
        results = {}
        for order in ("forward", "mirrored"):
            info = u.load_frozen_info(EXAMPLE, tatt=tatt)
            model = u.make_model(info)
            fid = dict(u.build_point(model, EXAMPLE, tatt=tatt))
            # sector -> {parameter: per-step delta}: for every sector, the
            # sampled names that fall in it, with their offsets
            self.sector_deltas = {
                s: _deltas_for(s, [n for n in fid if _sector_of(n) == s])
                for s, _ in SECTORS}
            for s in ("cosmo", "dz_source", "m"):
                self.assertTrue(self.sector_deltas[s],
                                f"no sampled parameters in sector {s}")
            # a sector nothing samples drops out of the ladder: lenses
            # that are the source sample carry no separate DZ_L shifts,
            # and a project such as roman_kl fixes every IA amplitude
            active = tuple(s for s in PHASES if self.sector_deltas[s])

            phases = active if order == "forward" else tuple(reversed(active))
            steps = {s: 0 for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, steps))
            prev = np.array(ci.compute_data_vector_masked())
            mfac = self._mpairs(model.likelihood[name], np)

            for sector in phases:
                for r in range(1, NSTEP + 1):
                    # the M values before this step, for the expected
                    # rescale of assertion 2
                    m_prev = {n: fid[n] + steps["m"] * d
                              for n, d in self.sector_deltas["m"].items()}
                    steps[sector] = r
                    point = self._point_at(fid, steps)
                    u.evaluate_chi2(model, point)
                    dv = np.array(ci.compute_data_vector_masked())
                    self.assertFalse(
                        np.array_equal(dv, prev),
                        f"{order}: {sector} step {r} left the data vector "
                        "unchanged (dead sector flag or stale cache)")
                    if sector == "m":
                        m_now = {n: point[n]
                                 for n in self.sector_deltas["m"]}
                        mp = sorted(m_prev)  # M1..M5 in bin order
                        ratio = np.ones(dv.size)
                        for k in range(dv.size):
                            i, j = mfac[k]
                            if j >= 0:
                                ratio[k] *= ((1 + m_now[mp[j]]) /
                                             (1 + m_prev[mp[j]]))
                            if i >= 0:
                                ratio[k] *= ((1 + m_now[mp[i]]) /
                                             (1 + m_prev[mp[i]]))
                        # nz selects the nonzero entries (the masked ones
                        # are zero in both vectors)
                        nz = prev != 0
                        rel = np.abs(dv[nz]/(prev[nz]*ratio[nz]) - 1.0)
                        self.assertLess(
                            rel.max(), RESCALE_RTOL,
                            f"{order}: M step {r} is not the analytic "
                            f"(1+m_i)(1+m_j) rescale (max {rel.max():.2e})")
                    prev = dv

            final_point = self._point_at(fid, steps)
            final_chi2 = u.evaluate_chi2(model, final_point)
            final_dv = np.array(ci.compute_data_vector_masked())

            # no-op probe: identical point again, bitwise
            u.evaluate_chi2(model, dict(final_point))
            self.assertTrue(
                np.array_equal(np.array(ci.compute_data_vector_masked()),
                               final_dv),
                f"{order}: a no-op re-evaluation changed the data vector")

            # scramble: every sector at once, bias included
            scr = {s: SCRAMBLE_STEP for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, scr))

            # return: the ladder's final point must reproduce bitwise
            back_chi2 = u.evaluate_chi2(model, final_point)
            back_dv = np.array(ci.compute_data_vector_masked())
            self.assertTrue(
                np.array_equal(back_dv, final_dv),
                f"{order}: returning after the scramble did not reproduce "
                "the data vector bit for bit (stale sector cache)")
            self.assertEqual(
                back_chi2, final_chi2,
                f"{order}: chi2 after the scramble return differs")
            results[order] = final_dv
            print(f"  {order} ladder ({'TATT' if tatt else 'NLA'}): "
                  f"final chi2 = {final_chi2:.6f}", flush=True)

        self.assertTrue(
            np.array_equal(results["forward"], results["mirrored"]),
            "the mirrored-order ladder landed on a different data vector: "
            "the answer depends on the invalidation history")

    def test_cache_consistency_nla(self):
        """The NLA ladder: every sector invalidates its own caches."""
        self._run_ladder(tatt=False)

    def test_cache_consistency_tatt(self):
        """The TATT ladder, which also exercises the FAST-PT table rebuilds."""
        self._run_ladder(tatt=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
