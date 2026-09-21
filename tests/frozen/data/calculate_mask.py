import numpy as np
import os
from astropy.cosmology import FlatLambdaCDM
import math as mt

#VM code adapted from Supranta's Python script
ggl_efficiency_cut = [0.05]

#VM INPUT BEGINS ---------------------------------------------------------------
for Year in [1]:
  for mask_choice in [1,2,3,4,5,6]:
    if (mask_choice == 1):
      # LSST_YX_M1.mask  (lmax = 3000) on CS -----------------------------------
      # lmax \times \theta_min corresponds to the first zero of the Bessel 𝐽0/4
      # lmax x theta_min corresponds to the first zero of the Bessel 𝐽0/4
      # J0 first zero is 2.4048, J4 first zero is 6.3802
      # For theta = 1arc_min, lmax * theta_min = 0.87
      ξp_CUTOFF = 2.756  # cutoff scale in arcminutes
      ξm_CUTOFF = 8.6955 # cutoff scale in arcminutes
      gc_CUTOFF = 21     # Galaxy clustering cutoff in Mpc/h
    elif (mask_choice == 2):
      # LSST_YX_M2.mask  -----------------------------------
      ξp_CUTOFF = 5.512  # cutoff scale in arcminutes
      ξm_CUTOFF = 17.391 # cutoff scale in arcminutes
      gc_CUTOFF = 21     # Galaxy clustering cutoff in Mpc/h
    elif (mask_choice == 3):
      # LSST_YX_M3.mask  ------------------------------------
      ξp_CUTOFF = 11.024  # cutoff scale in arcminutes
      ξm_CUTOFF = 34.782 # cutoff scale in arcminutes
      gc_CUTOFF = 21     # Galaxy clustering cutoff in Mpc/h
    elif (mask_choice == 4):
      # LSST_YX_M3.mask  ------------------------------------
      ξp_CUTOFF = 22.048 # cutoff scale in arcminutes
      ξm_CUTOFF = 69.564 # cutoff scale in arcminutes
      gc_CUTOFF = 21     # Galaxy clustering cutoff in Mpc/h
    elif (mask_choice == 5):
      # LSST_YX_M3.mask  ------------------------------------
      ξp_CUTOFF = 44.096  # cutoff scale in arcminutes
      ξm_CUTOFF = 139.128 # cutoff scale in arcminutes
      gc_CUTOFF = 21      # Galaxy clustering cutoff in Mpc/h
    elif (mask_choice == 6):
      # LSST_YX_M6.mask  all ones ---------------------------------------------
      ξp_CUTOFF = 0 # cutoff scale in arcminutes
      ξm_CUTOFF = 0 # cutoff scale in arcminutes
      gc_CUTOFF = 0 # Galaxy clustering cutoff in Mpc/h
    #VM INPUT ENDS -------------------------------------------------------------

    #VM GLOBAL VARIABLES -------------------------------------------------------
    THETA_MIN  = 2.5    # Minimum angular scale (in arcminutes)
    THETA_MAX  = 900.  # Maximum angular scale (in arcminutes)
    N_ANG_BINS = 26    # Number of angular bins
    N_LENS = 5  # Number of lens tomographic bins
    N_SRC  = 5  # Number of source tomographic bins
    N_XI_PS = int(N_SRC * (N_SRC + 1) / 2) 
    N_XI    = int(N_XI_PS * N_ANG_BINS)
    

    # COMPUTE SHEAR SCALE CUTS
    vtmin = THETA_MIN * 2.90888208665721580e-4;
    vtmax = THETA_MAX * 2.90888208665721580e-4;
    logdt = (mt.log(vtmax) - mt.log(vtmin))/N_ANG_BINS;
    theta = np.zeros(N_ANG_BINS+1)

    for i in range(N_ANG_BINS):
      tmin = mt.exp(mt.log(vtmin) + (i + 0.0) * logdt);
      tmax = mt.exp(mt.log(vtmin) + (i + 1.0) * logdt);
      x = 2./ 3.
      theta[i] = x * (tmax**3 - tmin**3) / (tmax**2- tmin**2)
      theta[i] = theta[i]/2.90888208665721580e-4

    cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
    def ang_cut(z):
      "Get Angular Cutoff from redshit z"
      theta_rad = gc_CUTOFF / cosmo.angular_diameter_distance(z).value
      return theta_rad * 180. / np.pi * 60.

    if (Year == 1):
      zavg = [0.3269670081723307,
              0.5086885453137051,
              0.6699437575466684,
              0.848472949839094,
              1.0712458524571165]

    #VM COSMIC SHEAR SCALE CUT -------------------------------------------------
    ξp_mask = np.hstack([(theta[:-1] > ξp_CUTOFF) for i in range(N_XI_PS)])
    ξm_mask = np.hstack([(theta[:-1] > ξm_CUTOFF) for i in range(N_XI_PS)])   

    #VM GGL mask ---------------------------------------------------------------
    if (Year == 1):
      ggl_efficiency = [
        [0.4456315654,0.8790767396,0.9389947229,0.8354926862,0.631370374],
        [0.0334525378,0.3739779295,0.8338207849,0.9821921200,0.8639203163],
        [0.0004551936,0.0536064628,0.4178650771,0.8715050829,0.9711497877],
        [0.0000006072,0.0015727852,0.0818817759,0.5363472954,0.9782246343],
        [0.0000000000,0.0000024903,0.0025740396,0.1465300465,0.8300740215]
      ]

    γt_mask = [] 
    if (Year == 1):
      for i in range(N_LENS): 
        for j in range(N_SRC):
          if ggl_efficiency[i][j] > ggl_efficiency_cut[0]:
            γt_mask.append((theta[:-1] > ang_cut(zavg[i])))
          else:
            γt_mask.append(np.zeros(N_ANG_BINS))
    γt_mask = np.hstack(γt_mask) 

    #VM w_theta mask -----------------------------------------------------------
    w_mask = np.hstack([(theta[:-1] > ang_cut(zavg[i])) for i in range(N_LENS)])

    #VM output -----------------------------------------------------------------
    mask = np.hstack([ξp_mask, ξm_mask, γt_mask, w_mask])
    if (Year == 1):
      np.savetxt("LSST_Y" + str(Year) + "_M" + str(mask_choice) +
        "_GGLOLAP" + str(ggl_efficiency_cut[0]) + ".mask", 
        np.column_stack((np.arange(0,len(mask)), mask)),
        fmt='%d %1.1f')

