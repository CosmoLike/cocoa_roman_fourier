"""Shared Cobaya likelihood of the roman_fourier project.

The three likelihoods of this folder, cosmic_shear, combo_2x2pt and
combo_3x2pt, are thin subclasses of _cosmolike_prototype_base below; they
differ only in the probe they select. Cobaya is the sampling framework: it
copies the options of the likelihood yaml into attributes of the instance
(self.accuracyboost, self.IA_model, ...), calls initialize once, asks
get_requirements which theory products to compute, and then calls logp at
every parameter point; self.provider hands those products back.

At each point the class passes the cosmology (distances, growth, linear
and nonlinear matter power spectra) and the nuisance parameters to
cosmolike, the C library compiled into cosmolike_roman_fourier_interface
(imported as ci), asks it for the model data vector m, and returns

    log L = -chi2/2,   chi2 = (d - m)^T C^-1 (d - m),

with d the measured data vector and C^-1 the inverse covariance of the
entries the mask keeps.

The data vector. The dataset file (data/roman_example.dataset) names the
data vector, the covariance, the mask and the redshift distributions n(z)
of 8 lens and 8 source bins (one n(z) file serves both samples). Each
entry is an angular power spectrum C_ell evaluated at the logarithmic
center of one of n_cl = 15 multipole bands between l_min = 30 and
l_max = 4000, in this order:

  cosmic shear            C_ell^EE of the 36 source-bin pairs (E modes
                          only: no B-mode and no xi- rows),
  galaxy-galaxy lensing   C_ell^gs of the 55 lens-source pairs left after
                          ggl_exclude,
  galaxy clustering       C_ell^gg of the 8 lens-bin auto spectra,

1485 entries in all. Every likelihood returns this full layout; the
entries outside its probes or removed by the mask are zero.

One evaluation (internal_get_datavector):
  1. set_cosmo_related: P(k, z) tables, growth and distances from the
     theory code (CAMB through Cobaya, or the hybrid emulators);
  2. set_lens_related: galaxy bias and lens photo-z shifts (skipped for
     cosmic shear);
  3. set_source_related: shear calibration, source photo-z shifts and
     intrinsic-alignment amplitudes;
  4. ci.compute_data_vector_masked, then ci.compute_chi2 in compute_logp.

Units: Cobaya works with k in 1/Mpc, P(k) in Mpc^3 and distances in Mpc;
cosmolike receives k in h/Mpc, P(k) in (Mpc/h)^3 and distances in Mpc/h.
"""

# The __future__ imports give Python 2 the Python 3 behavior (absolute
# imports, true division, print as a function); under Python 3 they change
# nothing. Python accepts them only before every other statement (the
# docstring and comments may precede them).
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import sys
import time
import functools

# Cobaya's base class for a likelihood configured by a dataset file, its
# error type, and getdist's reader of that file (key = value lines).
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

# EuclidEmulator2 (EE2), the emulator of the nonlinear matter power
# spectrum used when non_linear_emul = 1.
import euclidemu2 as ee2
import math

from contextlib import contextmanager
@contextmanager
def timer(label):
  """Print the wall-clock time a block of code takes.

  @contextmanager turns this generator into a context manager: in
  `with timer("label"):` the code before yield runs on entering the block
  and the print after yield runs on leaving it. No code in this file uses
  it.

  Arguments:
    label = text printed before the elapsed time in seconds.
  """
  t0 = time.perf_counter()
  yield
  print(f"{label}: {time.perf_counter() - t0:.4f}s")

# The compiled cosmolike interface (the project's start script puts
# interface/ on PYTHONPATH); every cosmolike call below is ci.<function>.
import cosmolike_roman_fourier_interface as ci

# The OpenMP team size of cosmolike, read once when this module is
# imported: OMP_NUM_THREADS from the environment, 1 when it is unset.
COSMOLIKE_OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))

def with_omp_threads(fn):
    """Wrap a method so that cosmolike runs on its full OpenMP team.

    Cosmolike parallelizes its hot loops with OpenMP and sizes each
    parallel region with omp_get_max_threads(), which starts at
    OMP_NUM_THREADS. Some Python libraries call omp_set_num_threads(1)
    without notice; the setting is global to the process, so every later
    cosmolike loop would run on one core. The wrapper restores
    COSMOLIKE_OMP_THREADS before each call of the wrapped method.

    A decorator: the line @with_omp_threads above a def replaces that
    method by wrapper, which calls ci.set_omp_threads and then the
    original method with the same arguments. functools.wraps copies the
    method's name and docstring onto wrapper.

    Arguments:
      fn = the method to wrap.

    Returns:
      wrapper, a function with the call signature and return value of fn.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        """Restore the OpenMP team size, then call fn with the same arguments."""
        ci.set_omp_threads(COSMOLIKE_OMP_THREADS)
        return fn(*args, **kwargs)
    return wrapper

# Prefix of the sampled nuisance-parameter names of this project:
# roman_M1, roman_DZ_S1, roman_A1_1, roman_B1_1, ... (bins count from 1).
survey = "roman"

class _cosmolike_prototype_base(DataSetLikelihood):
  """Cobaya likelihood shared by the three roman_fourier probe choices.

  DataSetLikelihood is Cobaya's base class for a likelihood configured by
  a dataset file. Before initialize runs, Cobaya sets every option of the
  likelihood yaml, overridden by the user's yaml block, as an attribute of
  the instance. The module docstring describes the data vector and the
  order of one evaluation.
  """

  def initialize(self, probe):
    """Read the dataset, build the redshift and k grids, initialize cosmolike.

    The subclass's initialize passes its probe name here; Cobaya calls it
    once, when it builds the model. The ci.init_* calls below form the
    cosmolike initialization chain: defaults, probes, multipole bands and
    the excluded lens-source pairs first (they fix the data-vector
    layout), then the run-time options, the n(z), the data, mask and
    covariance, and the intrinsic-alignment and galaxy-bias models.

    use_emulator selects the theory path:
      0 = CAMB through Cobaya computes distances and P(k) (the default);
      2 = hybrid: emulators replace CAMB for the background and P(k), and
          cosmolike computes everything else;
      1 = data-vector emulator; its evaluation is not implemented in this
          file (get_datavector returns 0.0).

    Arguments:
      probe = cosmolike's probe name: "xi" (cosmic shear), "2x2pt" or
              "3x2pt".

    Returns:
      nothing.

    Raises:
      LoggedError when pk_z_refinement is not a positive integer.

    Side effects:
      Replaces cosmolike's global state (bands, n(z), data, mask,
      covariance, model choices); a model built later in the same process
      replaces it again. Some options are rewritten instead of refused:
      IA_code 1 becomes 0 under NLA, and the baryon treatments switch each
      other off (comments below).
    """
    # The dataset file holds key = value lines; relativeFileName joins a
    # file name with the dataset's own folder.
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))
    self.probe = probe
    self.data_vector_file = ini.relativeFileName('data_file')
    self.cov_file = ini.relativeFileName('cov_file')
    self.mask_file = ini.relativeFileName('mask_file')
    self.lens_file = ini.relativeFileName('nz_lens_file')
    self.source_file = ini.relativeFileName('nz_source_file')
    # Tomographic bins (8 and 8 here) and n_cl multipole bands, log-spaced
    # between l_min and l_max. Cosmolike computes the shear entries only for
    # band centers below l_max_shear (4000 here: every shear band is kept).
    self.lens_ntomo = ini.int("lens_ntomo")
    self.source_ntomo = ini.int("source_ntomo")
    self.ncl = ini.int("n_cl")
    self.l_min = ini.float("l_min")
    self.l_max = ini.float("l_max")
    self.l_max_shear = ini.float("l_max_shear")

    # ------------------------------------------------------------------------
    # The 1D redshift grid of the background tables (comoving distance and
    # growth factor): three uniform blocks, [0, 3) where the galaxies are,
    # [3, 50.1) for the high-z tail of the lensing kernels, and
    # [1070, 1100] around recombination, the source plane of CMB lensing
    # (unused by the probes of this project). At accuracyboost 1
    # (tmp = 1250) the low block has 1000 nodes, dz = 0.003; the max(...)
    # floors keep every block populated at small boosts.
    tmp=int(1000 + 250*self.accuracyboost)
    self.z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp)),endpoint=False),
                                       np.linspace(3.0,50.1,max(100,int(0.40*tmp)),endpoint=False),
                                       np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    self.len_z_interp_1D = len(self.z_interp_1D)

    # The z nodes of the 2D power-spectrum tables handed to cosmolike.
    # Cosmolike interpolates P(k, z) linearly in z between exactly these
    # nodes: it indexes the two uniform blocks directly (no search) and
    # never regrids them. Linear interpolation leaves a sawtooth-shaped
    # residual of order dz^2 that vanishes at the nodes, so two grids that
    # do not share nodes disagree by the full residual amplitude, and a
    # node count that is not a nested refinement makes the chi2 jitter from
    # one boost value to the next instead of converging.
    #
    # The dyadic factor m = 2^ceil(log2(boost)), capped at 16, refines each
    # uniform block by an integer factor with the same endpoints, so
    #   (a) every block stays uniform and cosmolike keeps its direct
    #       two-block indexing;
    #   (b) the nodes of a grid are a subset of the nodes of every grid whose
    #       factor is a multiple of its m: raising the boost is a true
    #       refinement, and the residual falls like 1/m^2;
    #   (c) m = 1 gives the 140-node grid that CAMB receives as well
    #       (z_interp_2D_camb below).
    # The low block multiplies its node count (endpoint=False, spacing
    # 3/(105 m)); the high block multiplies its interval count
    # (endpoint=True: 35 nodes = 34 intervals -> 34 m + 1 nodes). The grid
    # ends at z = 49.99, inside the z <= 50 range of the hybrid emulators;
    # redshifts that high matter only for CMB lensing.
    #
    # pk_z_refinement multiplies m on top of the boost, which still sets
    # every other grid. A Fourier-space data vector reads P(k, z) at fixed
    # multipoles, where the residual of the linear z interpolation does not
    # average out as it does in a real-space transform: the 3x2pt chi2 of
    # this project moves by 0.25, 0.030 and 0.002 from m = 1 to 2, 4 and 8
    # (roman_real and lsst_y1 move by at most 0.004 from 1 to 2). The
    # likelihood yaml files set accuracyboost 2 and pk_z_refinement 2,
    # so m = 4.
    zref = getattr(self, "pk_z_refinement", 1)
    if not (float(zref) == int(zref) and int(zref) >= 1):
      raise LoggedError(self.log, "pk_z_refinement = %s: must be a positive "
                        "integer", zref)
    m = int(min(2**np.ceil(np.log2(max(1.0, self.accuracyboost))), 16))
    m = m*int(zref)
    self.z_interp_2D = np.concatenate((np.linspace(0,3.0,105*m,endpoint=False), 
                                       np.linspace(3.0,49.99,34*m + 1)),axis=0)
    self.len_z_interp_2D = len(self.z_interp_2D)
    # CAMB's transfer module accepts at most 256 redshifts, so the list
    # handed to CAMB through the Pk_interpolator requirement stays at this
    # boost-independent 140-node grid (the m = 1 grid above). The denser
    # nested nodes re-evaluate the cubic z spline that the Pk_interpolator
    # builds through CAMB's redshifts when the cosmolike tables are filled:
    # raising the boost refines only that table resampling, and CAMB never
    # exceeds its cap.
    self.z_interp_2D_camb = np.concatenate((np.linspace(0,3.0,105,endpoint=False), 
                                            np.linspace(3.0,49.99,35)),axis=0)
    
    # log10 of the table wavenumbers k in 1/Mpc, uniform from k = 1.0e-5 to
    # 100/Mpc, with 1500 nodes at accuracyboost 1; cosmolike receives
    # log10 k in h/Mpc (set_cosmo_related).
    self.log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*self.accuracyboost))
    self.len_log10k_interp_2D = len(self.log10k_interp_2D)
    # ------------------------------------------------------------------------

    # The cosmolike initialization chain. initial_setup resets every
    # cosmolike global to its default. The probes, the multipole bands and
    # the excluded (lens, source) pairs, counted from 0 and flattened to
    # [l0, s0, l1, s1, ...], fix the data-vector layout, so they come before
    # the data, mask and covariance. The bindings refuse a Python float
    # where C expects an integer, hence the int(...) casts.
    ci.initial_setup()
    ci.init_probes(possible_probes=self.probe)
    ci.init_binning(int(self.ncl),int(self.l_min),int(self.l_max),int(self.l_max_shear))
    ci.init_ggl_exclude(np.array(self.ggl_exclude).flatten())

    if self.debug:
      ci.set_log_level_debug()
    else:
      ci.set_log_level_info()

    # getattr(self, name, default) returns the default when the yaml omits
    # the option. n(z) interpolation: 0 = cubic spline, 1 = linear,
    # 2+ = Steffen (monotone, no overshoot below zero); z column of the
    # n(z) files: 0 = left bin edges (Z_LOW), 1 = sample points (Z_MID).
    ci.init_photoz_conventions(
        interpolation_type=int(getattr(self, "photoz_interpolation_type", 0)),
        zmid_convention=int(getattr(self, "photoz_zmid_convention", 0)))

    # C-FAST-PT, the C code of the perturbation-theory integrals of TATT and
    # one-loop galaxy bias: density of its internal convolution grid
    # relative to its output table (1.0 = the two grids are equal).
    ci.init_fpt_internal_boost(
        internal_boost=float(getattr(self, "internal_accuracyboost", 1.0)))

    # The comoving-distance grid of the non-Limber FFTLog integrals, refined
    # on top of the accuracy boost (narrow lens bins need it; see
    # init_nonlimber_accuracy_boost in the core).
    ci.init_nonlimber_accuracy_boost(
        nonlimber_boost=float(getattr(self, "nonlimber_accuracyboost", 1.0)))

    # Galaxy-galaxy lensing (gs) and galaxy clustering (gg): 0 = exact
    # (non-Limber) projection below l = 150 and Limber above, 1 = Limber at
    # every multipole. 0 is this project's default; test_nonlimber_ggl.py
    # and test_nonlimber_gg.py measure what Limber would cost.
    ci.init_adopt_limber_gs(
        adopt_limber_gs=int(getattr(self, "adopt_limber_gs", 0)))

    ci.init_adopt_limber_gg(
        adopt_limber_gg=int(getattr(self, "adopt_limber_gg", 0)))
    # 0 = perturbative galaxy bias, 1 = halo-model (HOD) galaxy power;
    # always set, so a model never inherits the previous model's value
    ci.init_include_HOD_GX(
        include_HOD_GX=int(getattr(self, "include_HOD_GX", 0)))
    # 0 = the init_IA model, 1 = halo-model IA (Fortuna et al. 2021)
    ci.init_include_halo_IA(
        include_halo_IA=int(getattr(self, "include_halo_IA", 0)))
    # Halo statistics use the cold dark matter + baryon spectrum P_cb. The
    # emulators provide no P_cb, so the hybrid path takes the small-scale
    # ratio P_cb = P_lin/(1 - f_nu)^2 (get_neutrino_inputs).
    if self.use_emulator == 2:
      self.log.info("Halo P_cb uses P_lin/(1 - f_nu)^2 because the "
                    "emulators have no cb spectrum (an approximation; "
                    "see get_neutrino_inputs)")

    # use_emulator 1 (data-vector emulator): cosmolike reads the n(z), the
    # data, mask and covariance, with a reduced accuracy boost; the emulator
    # evaluation itself is absent (get_datavector).
    if self.use_emulator == 1:
      ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo), 
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))
      ci.init_data_fourier(self.cov_file, self.mask_file, self.data_vector_file)
      ci.init_accuracy_boost(accuracy_boost=0.35, 
                             integration_accuracy=-1) # seems enough to compute PM
    else:
      ci.init_accuracy_boost(accuracy_boost=self.accuracyboost, 
                             integration_accuracy=int(self.integration_accuracy))
      # is_linear=False keeps the nonlinear P(k) where the model uses it
      # (True would force the linear spectrum everywhere).
      ci.init_cosmo_runmode(is_linear=False)

      # external_nz_modeling: the n(z) arrays are read once here and kept in
      # self.lens_nz and self.source_nz, and every evaluation sends a copy
      # that a user model may modify (set_lens_related,
      # set_source_related). Otherwise cosmolike reads the files itself.
      if self.external_nz_modeling: 
        (self.lens_nz, self.source_nz) = ci.read_redshift_distributions(
            lens_multihisto_file = self.lens_file,
            lens_ntomo = int(self.lens_ntomo), 
            source_multihisto_file = self.source_file,
            source_ntomo = int(self.source_ntomo)
          ) 
        ci.init_lens_sample_size(int(self.lens_ntomo))
        ci.init_source_sample_size(int(self.source_ntomo))
        ci.init_ntomo_powerspectra() # must be called after set_source/lens_size  
      else:
        ci.init_redshift_distributions_from_files(
          lens_multihisto_file = self.lens_file,
          lens_ntomo = int(self.lens_ntomo), 
          source_multihisto_file = self.source_file,
          source_ntomo = int(self.source_ntomo)) 

      # Data vector, mask and covariance; the mask must have one row per
      # entry of the layout fixed above (1485 rows for this dataset).
      ci.init_data_fourier(self.cov_file, self.mask_file, self.data_vector_file)

      if (int(self.IA_model) == 0) and (int(self.IA_code) == 1):
        # Under NLA the C implementation cfastpt (IA_code 0) replaces the
        # python FAST-PT block (IA_code 1), so get_requirements, which
        # Cobaya calls after initialize, requests no python FAST-PT tables.
        self.IA_code = 0
      # ia_model 0 = NLA, 1 = TATT. ia_redshift_evolution 3 (the yaml
      # default) sets each amplitude to A(z) = A_1 ((1+z)/(1+z_0))^A_2, with
      # A_1, A_2 = roman_A1_1, roman_A1_2 (roman_A2_1, roman_A2_2 likewise)
      # and z_0 cosmolike's pivot. ia_code 0 = cfastpt, 1 = python FAST-PT.
      ci.init_IA(ia_model = int(self.IA_model), 
                ia_redshift_evolution = int(self.IA_redshift_evolution),
                ia_code = int(self.IA_code))

      if self.probe != "xi":
        # bias_model: one entry per bias term, in the order (b1, b2, bs2,
        # b3, bmag, bK); each value selects how that term is set: 0 = a
        # value per lens bin, other values = a relation coded in cosmolike's
        # bias.c (1 in the b3 slot, the yaml default, sets b3 = b1 - 1).
        ci.init_bias(bias_model=self.bias_model)

      # non_linear_emul 1: the EuclidEmulator2 object is built once here, and
      # set_cosmo_related asks it for the nonlinear boost at every
      # evaluation (non_linear_emul 2 uses CAMB's halofit instead).
      if self.non_linear_emul == 1:
        self.emulator = ee2.PyEuclidEmulator()

      # The baryon treatments:
      #   external_baryon_suppression: the bfmt theory block multiplies the
      #     nonlinear P(k) by its suppression S(k, z) (set_cosmo_related);
      #   use_baryon_pca: the data vector gains principal components (PCs)
      #     of baryonic effects, with sampled amplitudes roman_BARYON_Q<i>;
      #   add_baryons_on_dv: cosmolike contaminates the matter power
      #     spectrum with the baryonic effect of one hydrodynamical
      #     simulation (which_bsims_add_on_dv in all_sims_hdf5_file);
      #   create_baryon_pca: an evaluation computes PCs from simulations
      #     and writes them (internal_get_datavector).
      # The options are rewritten without an error: external suppression
      # switches off use_baryon_pca and add_baryons_on_dv, and
      # create_baryon_pca switches off external suppression and
      # use_baryon_pca.
      if self.external_baryon_suppression:
          self.use_baryon_pca = False
          self.add_baryons_on_dv = False

      if self.create_baryon_pca:
        self.external_baryon_suppression = False
        self.use_baryon_pca = False
        self.allsims = ini.relativeFileName('all_sims_hdf5_file')
      else:
        if self.add_baryons_on_dv:
          self.external_baryon_suppression = False
          sim = self.which_bsims_add_on_dv
          self.allsims = ini.relativeFileName('all_sims_hdf5_file')
          ci.init_baryons_contamination(sim = sim, allsims=self.allsims)

    if self.use_baryon_pca:
      baryon_pca_file = ini.relativeFileName('baryon_pca_file')
      # The likelihood uses 4 PCs, so baryon_pca_file must hold 4
      # components; their amplitudes are roman_BARYON_Q1 ... Q4.
      self.npcs = 4
      ci.set_baryon_pcs(eigenvectors = np.loadtxt(baryon_pca_file))
      self.log.info('use_baryon_pca = True')
      self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
    else:
      self.log.info('use_baryon_pca = False')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    """Tell Cobaya which theory products each evaluation needs.

    Cobaya calls this after initialize and makes the theory codes compute
    the returned quantities; set_cosmo_related reads them back through
    self.provider. The request depends on use_emulator:
      1 = the data-vector emulator outputs ('ss', 'sg', 'gg' for the
          probes in use), plus H0 and comoving distances where galaxy
          probes need them;
      2 = the hybrid emulators: cosmological parameters, the matter power
          spectrum interpolator and comoving distances, plus the FAST-PT
          and EuclidEmulator2 inputs when those options are on;
      0 = CAMB: the same, plus a minimal Cl request (see the note below),
          Omega_nu h^2, the cold dark matter + baryon spectrum and, with
          external_baryon_suppression, the baryon suppression.

    Returns:
      dict mapping each requirement name to its options (None for none).
      Pk_interpolator asks for the linear and nonlinear P(k) at the
      redshifts z_interp_2D_camb up to k_max = kmax_boltzmann*accuracyboost
      in 1/Mpc; comoving distances are in Mpc.
    """
    if self.use_emulator == 1:
      if self.probe == "xi":
        return {
          'ss': None
        }
      elif self.probe == "3x2pt":
        return {
          "H0": None,
          'ss': None,
          'sg': None,
          'gg': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }
      elif self.probe == "xi_gg":
        return {
          'ss': None,
          'gg': None
        }
      elif self.probe == "xi_ggl":
        return {
          "H0": None,
          'ss': None,
          'sg': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D
          } # in Mpc
        }
      elif self.probe == "2x2pt":
        return {
          "H0": None,
          'sg': None,
          'gg': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }     
    elif self.use_emulator == 2:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "omegab": None,
        "mnu": None,
        "w": None,
        "wa": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D 
        }, # in Mpc
      }
      # IA_code 1: the python FAST-PT theory block supplies the TATT
      # (IA_PS) and one-loop galaxy-bias (bias_PS) tables.
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      # EuclidEmulator2 takes omegab, mnu, w and wa as inputs.
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      return _requirements_
    else:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
        # A minimal Cl request (TT up to ell = 0) keeps CAMB's Cl calculation
        # (WantCls) switched on; the note on the next line records that CAMB
        # misbehaves without it.
        "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
        }
      }
      # The bfmt baryon theory block computes the suppression S(k, z) at the
      # (k, z) nodes this likelihood asks for: the redshifts of the 2D
      # tables and the wavenumbers 10^log10k_interp_2D in 1/Mpc (the block
      # converts them when it works in h/Mpc).
      if self.external_baryon_suppression:
          _requirements_["baryon_suppression"] = {
              "z": self.z_interp_2D,
              "k": np.power(
                  10.0, self.log10k_interp_2D
              ),
          }
      # IA_code 1: the python FAST-PT theory block supplies the TATT
      # (IA_PS) and one-loop galaxy-bias (bias_PS) tables.
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      # EuclidEmulator2 takes omegab, mnu, w and wa as inputs.
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      # Omega_nu h^2 of the massive neutrinos (CAMB's omnuh2) and, for
      # the cold dark matter + baryon halo field, the linear P_cb
      # (get_neutrino_inputs)
      _requirements_["omnuh2"] = None
      # delta_tot = total matter, delta_nonu = cold dark matter + baryons
      # (no neutrinos). CAMB computes both from one transfer-function
      # calculation; the likelihood and the direct halo readers use them.
      _requirements_["Pk_interpolator"]["vars_pairs"] = [
        ("delta_tot", "delta_tot"),
        ("delta_nonu", "delta_nonu")]
      return _requirements_

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_cosmo_related(self):
    """Hand the cosmology of the current parameter point to cosmolike.

    Reads from Cobaya's provider the linear and nonlinear matter power
    spectra, the growth factor and the comoving distances, converts them
    to cosmolike's units and grids, and calls ci.set_cosmology (and, with
    IA_code 1, ci.set_IA_PS and ci.set_bias_PS). Runs at every evaluation,
    on cosmolike's full OpenMP team (with_omp_threads).

    Arrays handed to ci.set_cosmology, with n_z = len(z_interp_2D) and
    n_k = len(log10k_interp_2D):
      lnP_linear, lnP_linear_cb, lnP_nonlinear [n_k*n_z]: ln P in
          (Mpc/h)^3, flattened with the z index running fastest;
      log10k_2D [n_k]: log10 of k in h/Mpc;  z_2D [n_z];
      G [len(z_G)]: the growth factor D(z)(1+z) on the 1D grid up to
          z_2D[-1], divided by its value at z_2D[-1];
      chi [len(z_1D)]: comoving distance in Mpc/h.

    The nonlinear P(k): non_linear_emul 2 takes CAMB's halofit;
    non_linear_emul 1 multiplies CAMB's linear P(k) by the EuclidEmulator2
    boost B(k, z) = P_nl/P_lin at z < 10 and keeps halofit at z >= 10.
    With external_baryon_suppression, ln S(k, z) from the bfmt theory
    block is added to ln P_nl.

    Returns:
      nothing; cosmolike's global cosmology state is replaced.

    Raises:
      NameError, in place of the intended LoggedError, when
      non_linear_emul is neither 1 nor 2 (the error call names
      non_linear_emul instead of self.non_linear_emul).
    """
    h = self.provider.get_param("H0")/100.0
    # h = H0/(100 km/s/Mpc). P in Mpc^3 times h^3 is P in (Mpc/h)^3, hence
    # the + ln(h^3) below; k in h/Mpc is k in 1/Mpc divided by h.
    if not (self.use_emulator == 1):
      PKL  = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), 
                                               nonlinear=False, 
                                               extrap_kmin=1e-6,
                                               extrap_kmax=2.5e2*self.accuracyboost)

      # lnPL: ln P_lin on the (z, k) grid, shape [n_z, n_k], k in 1/Mpc;
      # PKL extrapolates beyond CAMB's k range down to 1e-6/Mpc and up to
      # 250*accuracyboost/Mpc. flatten(order='F') reads the array column by
      # column, so the z index runs fastest: the layout ci.set_cosmology
      # expects.
      lnPL = PKL.logP(self.z_interp_2D,
                      np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)

      if self.non_linear_emul == 1:
        params = {
          'Omm'  : self.provider.get_param("omegam"),
          'As'   : self.provider.get_param("As"),
          'Omb'  : self.provider.get_param("omegab"),
          'ns'   : self.provider.get_param("ns"),
          'h'    : h,
          'mnu'  : self.provider.get_param("mnu"), 
          'w'    : self.provider.get_param("w"),
          'wa'   : self.provider.get_param("wa"),
        }
        # EE2 predicts the boost B(k, z) = P_nl/P_lin for w0waCDM with
        # massive neutrinos inside its training range: z < 10, and k from
        # 10^-2.0589 = 8.73e-3 to 10^0.973 = 9.40 h/Mpc, the grid passed below
        # (with as many k values as the table has).
        kbt, tmp_bt = ee2.get_boost2(params, 
                                     self.z_interp_2D[self.z_interp_2D < 10.0], 
                                     self.emulator, 
                                     10**np.linspace(-2.0589,0.973,self.len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')
        # kbt [n_kbt] in h/Mpc and bt [n_z(<10), n_kbt]. interp1d(...) builds
        # a function that interpolates ln B linearly in log10 k along axis 1,
        # and the trailing (...) calls it at once on the table's log10 k
        # converted to h/Mpc. Above 9.40 h/Mpc ln B is extrapolated linearly
        # in log10 k; below 8.73e-3 h/Mpc it is set to 0 (B = 1, no boost).
        tmp = interp1d(np.log10(kbt), 
                        np.log(bt), 
                        axis=1,
                        kind='linear', 
                        fill_value='extrapolate', 
                        assume_sorted=True)(self.log10k_interp_2D-np.log10(h)) #h/Mpc
        tmp[:,10**(self.log10k_interp_2D-np.log10(h)) < 8.73e-3] = 0.0
        lnbt = np.zeros((self.len_z_interp_2D, self.len_log10k_interp_2D))
        lnbt[self.z_interp_2D < 10.0, :] = tmp
        # Halofit at every redshift first, since EE2 stops at z = 10 ...
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax =2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3) 
        # ... then, at z < 10, ln P_nl = ln P_lin + ln B (EE2). np.where
        # picks, row by row of the [n_z, n_k] arrays, the EE2 value where
        # z < 10 and halofit elsewhere; ravel(order='F') flattens back with
        # the z index fastest.
        lnPNL = np.where((self.z_interp_2D<10)[:,None], 
          lnPL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')+lnbt, 
          lnPNL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')).ravel(order='F')
      elif self.non_linear_emul == 2:
        # CAMB's halofit, with the halofit version chosen in the theory block.
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax=2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)   
      else:
        # non_linear_emul here is an undefined name (self.non_linear_emul is
        # meant), so Python raises NameError before LoggedError is built.
        raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", non_linear_emul)

      # The growth factor G(z) = D(z)(1+z) on the dense 1D z grid, up to the
      # last 2D node. Cosmolike interpolates G linearly in z: on the 2D grid
      # at m = 1 (dz ~ 0.03) that linear read misses D by up to 9e-5 and the
      # growth rate f = dlnD/dlna = 1 - (1+z) dlnG/dz (the slope of the
      # table) by 1%; on the 1D grid at accuracyboost 1 (dz = 0.003) by 1e-6
      # and 0.2%. PKL is a cubic spline in z through CAMB's transfer
      # redshifts, so the dense grid asks CAMB for no extra redshifts (about
      # 0.1 ms per evaluation). The table is divided by G at the last 2D
      # node (z_growth ends below it); cosmolike's growfac divides by G(0),
      # so the normalization cancels and D(z = 0) = 1.
      z_growth = self.z_interp_1D[self.z_interp_1D <= self.z_interp_2D[-1]]
      # G is read at k = growth_k (yaml option, default 0.05/Mpc), a
      # sub-horizon scale: sqrt(P_lin(z, k)/P_lin(0, k)) = D(z) at that k.
      # Near the horizon, at k = 5e-4/Mpc (about 2 H0/c), CAMB's dark-energy
      # perturbations change the growth by 0.5-0.9% for w != -1 (z = 0.5 to
      # 2), while every reader of G (IA amplitudes, the one-loop D^4 terms,
      # sigma(M, z), the growth rate f) describes sub-horizon modes. With
      # 0.06 eV neutrinos the growth varies with k by 0.03% above 0.05/Mpc.
      growth_k = float(getattr(self, "growth_k", 0.05))
      G_growth = np.sqrt(PKL.P(z_growth,growth_k)/PKL.P(0,growth_k))*(1+z_growth)
      z_norm = self.z_interp_2D[-1]
      G_growth /= np.sqrt(PKL.P(z_norm,growth_k)/PKL.P(0,growth_k))*(1+z_norm)
      # With external_baryon_suppression the bfmt theory block returns a dict
      # {z: S(k) array}: the suppression S = P_baryons/P_dark-matter-only on
      # the requested k grid, its calibration masking already applied. ln S
      # is added to the ln P_nl entries of that redshift; in the flattened
      # layout (z fastest) they are lnPNL[i], lnPNL[i + n_z], ..., the slice
      # lnPNL[i :: n_z]. A missing redshift is logged and skipped; any
      # exception is logged and ends the loop, and the evaluation continues
      # with the rows suppressed so far.
      if self.external_baryon_suppression:
        try:
          supp_dict = self.provider.get_result("baryon_suppression")
          self.log.info(
            "Applying baryon suppression: %d redshifts from theory block",
            len(supp_dict),
          )

          for i, z_val in enumerate(self.z_interp_2D):
            if z_val in supp_dict:
              sup_array = supp_dict[z_val]
              lnbt_baryon = np.log(sup_array)
              lnPNL[i :: self.len_z_interp_2D] += lnbt_baryon
              self.log.debug(
                  "Applied baryon suppression at z=%.3f: "
                  "min_sup=%.6f, max_sup=%.6f",
                  z_val,
                  sup_array.min(),
                  sup_array.max(),
              )
            else:
              self.log.warning(
                  "baryon_suppression dict does not contain z=%.3f; skipping",
                  z_val,
              )
        except Exception as e:
            self.log.error(
                "Failed to retrieve baryon suppression from theory block: %s; "
                "skipping baryon suppression",
                str(e),
            )

      # the massive neutrinos: Omega_nu h^2 and, for the cold dark matter
      # + baryon halo field, the linear P_cb (get_neutrino_inputs)
      (omegan2, lnPL_cb) = self.get_neutrino_inputs(lnPL=lnPL, h=h)

      ci.set_cosmology(
        omegam=self.provider.get_param("omegam"),
        omegab=self.provider.get_param("omegab"),
        omegan2=omegan2,
        H0=self.provider.get_param("H0"),
        log10k_2D=self.log10k_interp_2D-np.log10(h), #h/Mpc
        z_2D=self.z_interp_2D,
        lnP_linear=lnPL, 
        lnP_linear_cb=lnPL_cb,
        lnP_nonlinear=lnPNL, 
        G=G_growth,
        z_G=z_growth,
        z_1D=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h # convert to Mpc/h
      )
      
      # IA_code 1: install the python FAST-PT tables. This must follow
      # ci.set_cosmology, which draws a new cosmology.random (the key
      # cosmolike uses to detect a changed cosmology). Row -2 (index 10) of
      # the IA table is its k grid in h/Mpc, so its first and last entries
      # give the k range; N = points per row; flatten(order='C') lays the
      # rows end to end, the layout set_IA_PS and set_bias_PS expect.
      if int(self.IA_code) == 1:
        FPTIA, FPTIA_kcut  = self.provider.get_IA_PS()
        FPTbias, sigma4    = self.provider.get_bias_PS()
        FPT_kmin, FPT_kmax = FPTIA[-2,0], FPTIA[-2,-1]
        
        ci.set_IA_PS(PS=FPTIA.flatten(order='C'), 
                     kmin=FPT_kmin, 
                     kmax=FPT_kmax, 
                     cutoff=FPTIA_kcut, 
                     N=len(FPTIA[0]))
        
        ci.set_bias_PS(PS=FPTbias.flatten(order='C'), 
                       kmin=FPT_kmin, 
                       kmax=FPT_kmax, 
                       cutoff=FPTIA_kcut, 
                       sigma4=sigma4, 
                       N=len(FPTIA[0]))
  
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def get_neutrino_inputs(self, lnPL, h):
    """Return the massive-neutrino inputs of ci.set_cosmology.

    omegan2 is Omega_nu h^2 of massive neutrinos today, part of omegam.
    Halo variances use the cold dark matter + baryon spectrum P_cb at
    each redshift. Their mass-radius relation and mass-function density
    use rho_crit (Omega_m - Omega_nu). Total matter remains available
    for lensing and for the separate total-matter variance.

    lnPL_cb is ln P_cb on the same (k,z) grid and in the same units as
    lnPL. Both spectra are provided so direct halo readers can be used
    even after a likelihood evaluation that did not count halos.

    The two theory paths:
      CAMB (use_emulator = 0): omegan2 is CAMB's omnuh2 and P_cb its
        ("delta_nonu", "delta_nonu") linear spectrum, read like P_lin
        (get_requirements asks for both).
      emulators (use_emulator = 2): the emulators take no neutrino
        parameter (they were trained at mnu = 0.06 eV) and have no cb
        spectrum. omegan2 = mnu (3.046/3)^0.75/94.0708, the neutrino
        density the yaml's omegach2 subtracts, and
        P_cb = P_lin/(1 - f_nu)^2 with f_nu = omegan2/(omegam h^2): the
        ratio of the two spectra far above the neutrino free-streaming
        scale, an approximation on cluster scales. Its measured size is
        in projects/des_cluster/README.md.

    Arguments:
      lnPL = ln P_lin [(Mpc/h)^3], flattened as set_cosmology's
             lnP_linear (Fortran order: k index slow, z index fast)
      h    = H0/100

    Returns:
      (omegan2, lnPL_cb): Omega_nu h^2 (float) and ln P_cb in (Mpc/h)^3,
      a numpy array of lnPL's shape and layout.
    """
    if self.use_emulator == 2:
      mnu = self.provider.get_param("mnu")
      omegan2 = mnu*(3.046/3.0)**0.75/94.0708
    else:
      omegan2 = self.provider.get_param("omnuh2")

    if self.use_emulator == 2:
      # P_cb/P_lin = 1/(1 - f_nu)^2 where the neutrinos no longer
      # cluster (delta_m = (1 - f_nu) delta_cb)
      f_nu = omegan2/(self.provider.get_param("omegam")*h*h)
      lnPL_cb = lnPL - 2.0*np.log(1.0 - f_nu)
    else:
      # the same k extrapolation, (z, k) grid, flattening and units as
      # lnPL in set_cosmo_related
      PKL_cb = self.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"),
                                                 nonlinear=False,
                                                 extrap_kmin=1e-6,
                                                 extrap_kmax=2.5e2*self.accuracyboost)
      k_grid = np.power(10.0, self.log10k_interp_2D)
      lnPL_cb = PKL_cb.logP(self.z_interp_2D, k_grid).flatten(order='F')
      lnPL_cb = lnPL_cb + np.log(h**3)
    return (omegan2, lnPL_cb)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_source_related(self, **params):
    """Hand the source-galaxy nuisance parameters to cosmolike.

    Sets, per source bin i = 1..source_ntomo, the shear calibration m_i
    (roman_M<i>: the measured shear is (1 + m_i) times the true shear, so
    a shear spectrum scales by (1 + m_i)(1 + m_j)), the photo-z shift
    Delta z_i (roman_DZ_S<i>: n_i(z) is evaluated at z - Delta z_i) and the
    intrinsic-alignment parameters (roman_A1_<i>, roman_A2_<i>,
    roman_BTA_<i>). With external_nz_modeling, a copy of the stored source
    n(z), which a user model may modify, is sent first.

    Arguments:
      params = the current parameter point, name -> value; Cobaya passes
               it as keyword arguments. A name absent from params takes 0.

    Returns:
      nothing; cosmolike's source nuisance state is replaced.
    """
    ntomo = self.source_ntomo
    # Each list holds one value per source bin: the inner comprehension
    # builds the names roman_M1 ... roman_M<ntomo>, the outer one reads each
    # value from params (0 when the name is absent).
    ci.set_nuisance_shear_calib(
      M=[params.get(p,0) for p in [survey+"_M"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      if self.external_nz_modeling: 
        # The n(z) is sent at every evaluation, so a user function can
        # modify it per parameter point (for example adding an outlier
        # population): (1) copy() makes an independent copy of the stored
        # array, which keeps the fiducial n(z) intact; (2) a user model
        # modifies the copy; (3) ci.set_source_sample sends it.
        source_nz_local = self.source_nz.copy()

        # Place of step (2): a user model replaces the copy here, as in
        #   source_nz_local = f(source_nz_local, nuisance parameters)

        ci.set_source_sample(source_nz_local)

        # The photo-z shifts DZ_S still apply on top of the n(z) just sent.
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      ci.set_nuisance_ia(
        A1=[params.get(p,0) for p in [survey+"_A1_"+str(i+1) for i in range(ntomo)]],
        A2=[params.get(p,0) for p in [survey+"_A2_"+str(i+1) for i in range(ntomo)]],
        B_TA=[params.get(p,0) for p in [survey+"_BTA_"+str(i+1) for i in range(ntomo)]]
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_lens_related(self, **params):
    """Hand the lens-galaxy nuisance parameters to cosmolike.

    Sets, per lens bin i = 1..lens_ntomo, the galaxy bias parameters
    (roman_B1_<i> linear, roman_B2_<i> quadratic, roman_BMAG_<i>
    magnification, roman_B3NL_<i> third-order, roman_BK_<i> nonlocal; the
    bias_model option decides which of them enter) and the lens photo-z
    shifts (roman_DZ_L<i>; the shipped examples set them equal to the
    source shifts, because both samples use one n(z) file). Called only for
    probes with lens galaxies, not for cosmic shear.

    Arguments:
      params = the current parameter point, name -> value. A name absent
               from params takes 1 for B1 (unit linear bias) and 0 for the
               others.

    Returns:
      nothing; cosmolike's lens nuisance state is replaced.
    """
    ntomo = self.lens_ntomo
    if not (self.use_emulator == 1):
      ci.set_nuisance_bias(
        B1=[params.get(p,1) for p in [survey+"_B1_"+str(i+1) for i in range(ntomo)]],
        B2=[params.get(p,0) for p in [survey+"_B2_"+str(i+1) for i in range(ntomo)]],
        B_MAG=[params.get(p,0) for p in [survey+"_BMAG_"+str(i+1) for i in range(ntomo)]],
        B3nl=[params.get(p,0) for p in [survey+"_B3NL_"+str(i+1) for i in range(ntomo)]],
        BK=[params.get(p,0) for p in [survey+"_BK_"+str(i+1) for i in range(ntomo)]]
      )
      if self.external_nz_modeling: 
        # The n(z) is sent at every evaluation, so a user function can
        # modify it per parameter point (for example adding an outlier
        # population): (1) copy() makes an independent copy of the stored
        # array, which keeps the fiducial n(z) intact; (2) a user model
        # modifies the copy; (3) ci.set_lens_sample sends it.
        lens_nz_local = self.lens_nz.copy()

        # Place of step (2): a user model replaces the copy here, as in
        #   lens_nz_local = f(lens_nz_local, nuisance parameters)

        ci.set_lens_sample(lens_nz_local)

        # The photo-z shifts DZ_L still apply on top of the n(z) just sent.
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def compute_logp(self, datavector):
    """Return the log-likelihood of a model data vector.

    Arguments:
      datavector = the model data vector, float64 [n_data] (1485 entries
                   for the shipped dataset), zero outside the mask, as
                   get_datavector returns it.

    Returns:
      -chi2/2, with chi2 = (d - m)^T C^-1 (d - m) over the kept entries,
      computed by cosmolike from the data and covariance read in initialize.
    """
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def logp(self, **params):
    """Return log L at the parameter point Cobaya passes (Cobaya's entry point).

    Arguments:
      params = every sampled and fixed parameter value, name -> value.

    Returns:
      the log-likelihood, a float (compute_logp of get_datavector).
    """
    return self.compute_logp(self.get_datavector(**params))

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def get_datavector(self, **params):        
    """Return the model data vector at the current parameter point.

    Arguments:
      params = the parameter point, name -> value.

    Returns:
      float64 numpy array [n_data]: the full 3x2pt layout, zero outside the
      mask and outside this likelihood's probes. Under use_emulator 1 the
      emulator call is absent (commented out) and the value is the
      placeholder 0.0, a zero-dimensional array.

    Side effects:
      Replaces cosmolike's cosmology and nuisance state; may write files
      (see internal_get_datavector).
    """
    if self.use_emulator == 1:
      #dv = self.internal_get_datavector_emulator(**params)
      dv = 0.0
    else:
      dv = self.internal_get_datavector(**params)
    return np.array(dv,dtype='float64')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector(self, **params):
    """Update cosmolike for the parameter point and compute the data vector.

    Order: the cosmology first, then the lens parameters (skipped for
    cosmic shear, probe "xi"), then the source parameters. The data vector
    comes last, with one of three baryon treatments:
      create_baryon_pca = compute baryon PCs from the simulations named by
          baryon_pca_select_sims, save them to filename_baryon_pca, and
          return the plain vector;
      use_baryon_pca = add sum_i Q_i PC_i, with Q_i = roman_BARYON_Q<i>
          (i = 1..npcs), to the vector;
      neither = the plain vector.

    Arguments:
      params = the parameter point, name -> value.

    Returns:
      list of floats [n_data] from the compiled interface, zero at the
      entries the mask removes.

    Side effects:
      Writes filename_baryon_pca under create_baryon_pca and, when
      print_datavector is true, writes print_datavector_file at every
      evaluation: two columns, the entry index (%d) and the value (%1.8e).
    """
    self.set_cosmo_related()
    
    if self.probe != "xi":
      self.set_lens_related(**params)
    self.set_source_related(**params)
    
    if self.create_baryon_pca:
      pcs = ci.compute_baryon_pcas(scenarios=self.baryon_pca_select_sims, allsims=self.allsims)
      np.savetxt(self.filename_baryon_pca, pcs)
      datavector = ci.compute_data_vector_masked()
    elif self.use_baryon_pca: 
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]     
      datavector = ci.compute_data_vector_masked_with_baryon_pcs(Q=Q)
    else: 
      datavector = ci.compute_data_vector_masked()

    if self.print_datavector:
      size = len(datavector)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = datavector
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return datavector
