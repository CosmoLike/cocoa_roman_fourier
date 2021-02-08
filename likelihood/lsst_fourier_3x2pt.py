from cobaya.likelihoods.lsst_fourier._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_lsst_fourier_interface as ci

class lsst_fourier_3x2pt(_cosmolike_prototype_base):
	# ------------------------------------------------------------------------
	# ------------------------------------------------------------------------
	# ------------------------------------------------------------------------

	def initialize(self):
		super(lsst_fourier_3x2pt,self).initialize(probe="3x2pt")

	# ------------------------------------------------------------------------
	# ------------------------------------------------------------------------
	# ------------------------------------------------------------------------

	def logp(self, **params_values):
		self.set_cosmo_related()

		self.set_lens_related(**params_values)

		self.set_source_related(**params_values)

		datavector = ci.compute_data_vector()

		return self.compute_logp(datavector)

