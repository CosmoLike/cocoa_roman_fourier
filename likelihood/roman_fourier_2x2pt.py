from cobaya.likelihoods.roman_fourier._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_roman_fourier_interface as ci
import numpy as np

class roman_fourier_2x2pt(_cosmolike_prototype_base):
	def initialize(self):
		super(roman_fourier_2x2pt,self).initialize(probe="2x2pt")

	def logp(self, **params_values):
		datavector = self.internal_get_datavector(**params_values)
    	return self.compute_logp(datavector)

    def get_datavector(self, **params_values):
    	datavector = self.internal_get_datavector(**params_values)
    	return np.array(datavector)

    def internal_get_datavector(self, **params_values):
		self.set_cosmo_related()

		self.set_lens_related(**params_values)

		self.set_source_related(**params_values)

		if self.create_baryon_pca:
	      pcs = ci.compute_baryon_pcas(scenarios = self.baryon_pca_sims)
	      np.savetxt(self.filename_baryon_pca, pcs)
	      # No need to call set_cosmo_related again with self.force_cache_false = True
	      # Why? End of compute_baryon_pcas function forced C cosmo cache renew

	    if self.use_baryon_pca:      
	      datavector = np.array(
	        ci.compute_data_vector_masked_with_baryon_pcs(
	          Q = [
	                params_values.get(p, None) for p in [
	                  survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)
	                ]
	              ]
	        )
	      )
	    else:  
	      datavector = np.array(ci.compute_data_vector_masked())

	    if self.print_datavector:
	      size = len(datavector)
	      out = np.zeros(shape=(size, 2))
	      out[:,0] = np.arange(0, size)
	      out[:,1] = datavector
	      fmt = '%d', '%1.8e'
	      np.savetxt(self.print_datavector_file, out, fmt = fmt)

		return datavector
