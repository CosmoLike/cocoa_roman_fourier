from cobaya.likelihoods.desy3xplanck._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_des_y3_6x2_interface as ci
#import time

class des_2x2pt(_cosmolike_prototype_base):
	# ------------------------------------------------------------------------
	# ------------------------------------------------------------------------
	# ------------------------------------------------------------------------

	def initialize(self):
		super(des_2x2pt,self).initialize(probe="2x2pt")

	# ------------------------------------------------------------------------
	# ------------------------------------------------------------------------
	# ------------------------------------------------------------------------

	def logp(self, **params_values):
		#t0 = time.time()
		self.set_cosmo_related()

		self.set_lens_related(**params_values)

		self.set_source_related(**params_values)

		datavector = ci.compute_data_vector()

		if self.print_intermediate_products == True:
			self.test_all()

		#t1 = time.time()
		#print(t1-t0)
		return self.compute_logp(datavector)