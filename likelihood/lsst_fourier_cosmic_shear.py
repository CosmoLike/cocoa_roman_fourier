from cobaya.likelihoods.lsst_fourier._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_lsst_fourier_interface as ci

class lsst_fourier_cosmic_shear(_cosmolike_prototype_base):
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def initialize(self):
    super(lsst_fourier_cosmic_shear,self).initialize(probe="ss")

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def logp(self, **params_values):
    self.set_cosmo_related()

    self.set_source_related(**params_values)

    datavector = ci.compute_data_vector()

    return self.compute_logp(datavector)