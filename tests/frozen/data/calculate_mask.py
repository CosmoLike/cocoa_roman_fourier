import numpy as np

#VM Fourier-space mask generator for the roman_fourier project.
#VM Regenerates data/roman_example.mask and data/ones.mask exactly.

#VM INPUT (from data/roman_example.dataset and likelihood/*.yaml) ----------
N_CL   = 15    # Number of C_ell bins (log-spaced)
L_MIN  = 30.   # Minimum ell (lower edge of first bin)
L_MAX  = 4000. # Maximum ell (upper edge of last bin)
N_LENS = 8     # Number of lens tomographic bins
N_SRC  = 8     # Number of source tomographic bins

# ggl (lens, source) pairs excluded from the data vector (likelihood yaml)
GGL_EXCLUDE = [[4,0],[5,0],[6,0],[6,1],[6,2],[7,0],[7,1],[7,2],[7,3]]

# Scale cuts: shear keeps all bins (l_max_shear = 4000); ggl and clustering
# drop the bins whose center ell exceeds LMAX_GC. With 15 log bins in
# [30, 4000] the last three centers are 1770/2452/3398, so any LMAX_GC
# between 1770 and 2452 zeroes exactly the last 2 bins of each block,
# which is the pattern of the shipped roman_example.mask.
LMAX_GC = 2000.

#VM GLOBAL VARIABLES -------------------------------------------------------
N_SHEAR = int(N_SRC * (N_SRC + 1) / 2)              # 36 shear blocks
N_GGL   = N_LENS * N_SRC - len(GGL_EXCLUDE)         # 55 ggl blocks
# data vector = [shear, ggl, clustering] = 99 blocks x 15 ells = 1485

# C_ell bin centers (geometric mean of the log-spaced bin edges)
ell_edges = np.logspace(np.log10(L_MIN), np.log10(L_MAX), N_CL + 1)
ell = np.sqrt(ell_edges[1:] * ell_edges[:-1])

#VM COSMIC SHEAR MASK (no cut) ---------------------------------------------
shear_mask = np.hstack([np.ones(N_CL) for i in range(N_SHEAR)])

#VM GGL MASK ---------------------------------------------------------------
gc_cut = (ell < LMAX_GC)
ggl_mask = np.hstack([gc_cut for i in range(N_GGL)])

#VM w (clustering, auto only) MASK -----------------------------------------
w_mask = np.hstack([gc_cut for i in range(N_LENS)])

#VM output -----------------------------------------------------------------
mask = np.hstack([shear_mask, ggl_mask, w_mask])
np.savetxt("roman_example.mask",
  np.column_stack((np.arange(0, len(mask)), mask)),
  fmt='%d %e')

ones = np.ones(len(mask))
np.savetxt("ones.mask",
  np.column_stack((np.arange(0, len(ones)), ones)),
  fmt='%d %e')
