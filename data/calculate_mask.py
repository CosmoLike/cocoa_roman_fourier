"""Write the scale-cut masks of the roman_fourier data vector.

The data vector holds 99 angular power spectra C_ell, each evaluated at
the centers of 15 multipole bands, 1485 entries in this order:

  cosmic shear             36 source-bin pairs (i <= j), shear E modes
                           only (no xi- rows as in real space),
  galaxy-galaxy lensing    the 55 (lens, source) pairs left after the 9
                           excluded ones,
  galaxy clustering        the 8 lens-bin auto spectra.

A mask file has one row per entry, "index value": value 1 keeps the entry
in the chi2 and 0 removes it (a scale cut). roman_example.mask keeps every
shear entry and the galaxy-galaxy lensing and clustering entries whose band
center lies below LMAX_GC: 1359 entries. ones.mask keeps all 1485.

The script reproduces the shipped data/roman_example.mask and
data/ones.mask. It writes into the current folder, replacing files of the
same name, so run it from data/:

    python calculate_mask.py
"""

import numpy as np

# ---- inputs: these must repeat data/roman_example.dataset (n_cl, l_min,
# l_max, lens_ntomo, source_ntomo) and ggl_exclude of likelihood/*.yaml ----
N_CL   = 15    # Number of C_ell bins (log-spaced)
L_MIN  = 30.   # Minimum ell (lower edge of first bin)
L_MAX  = 4000. # Maximum ell (upper edge of last bin)
N_LENS = 8     # Number of lens tomographic bins
N_SRC  = 8     # Number of source tomographic bins

# Galaxy-galaxy lensing (lens, source) pairs left out of the data vector,
# bins counted from 0: the ggl_exclude list of the likelihood yaml files.
GGL_EXCLUDE = [[4,0],[5,0],[6,0],[6,1],[6,2],[7,0],[7,1],[7,2],[7,3]]

# Scale cuts: shear keeps every band (l_max_shear = 4000); galaxy-galaxy
# lensing and clustering keep the bands whose center lies below LMAX_GC.
# With 15 log bands in [30, 4000] the last three centers are 1770, 2452
# and 3398, so any LMAX_GC between 1770 and 2452 removes exactly the last
# 2 bands of each block, the pattern of the shipped roman_example.mask.
LMAX_GC = 2000.

# ---- block counts ----
N_SHEAR = int(N_SRC * (N_SRC + 1) / 2)              # 36 shear blocks
N_GGL   = N_LENS * N_SRC - len(GGL_EXCLUDE)         # 55 ggl blocks
# data vector = [shear, ggl, clustering] = 99 blocks x 15 ells = 1485

# Band centers: the geometric mean of the two log-spaced edges of each band,
# the multipoles at which cosmolike evaluates the data vector.
ell_edges = np.logspace(np.log10(L_MIN), np.log10(L_MAX), N_CL + 1)
ell = np.sqrt(ell_edges[1:] * ell_edges[:-1])

# Cosmic shear, no cut. The list holds one row of N_CL ones per shear
# block; np.hstack joins the rows end to end into one flat array.
shear_mask = np.hstack([np.ones(N_CL) for i in range(N_SHEAR)])

# Galaxy-galaxy lensing: gc_cut is a boolean array [N_CL], True where the
# band center lies below LMAX_GC, repeated for every kept pair.
gc_cut = (ell < LMAX_GC)
ggl_mask = np.hstack([gc_cut for i in range(N_GGL)])

# Galaxy clustering, auto spectra only (one block per lens bin), with the
# same band cut. The name w_mask comes from the real-space projects, where
# clustering is w(theta).
w_mask = np.hstack([gc_cut for i in range(N_LENS)])

# The three blocks in data-vector order (np.hstack turns the booleans into
# 1.0 and 0.0). Each output row is "index value": fmt '%d %e' prints the
# index as an integer and the value in exponent notation (1.000000e+00).
mask = np.hstack([shear_mask, ggl_mask, w_mask])
np.savetxt("roman_example.mask",
  np.column_stack((np.arange(0, len(mask)), mask)),
  fmt='%d %e')

ones = np.ones(len(mask))
np.savetxt("ones.mask",
  np.column_stack((np.arange(0, len(ones)), ones)),
  fmt='%d %e')
