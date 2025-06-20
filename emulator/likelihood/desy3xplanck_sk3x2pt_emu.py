from cobaya.likelihoods.desy1xplanck._cosmolike_emu_prototype_base import _cosmolike_emu_prototype_base

class desy3xplanck_sk3x2pt_emu(_cosmolike_emu_prototype_base):
	''' Attributes needed from the likelihood yaml file:
	- train_config: filename of the training config file
	'''
	def initialize(self):
		super(desy3xplanck_sk3x2pt_emu, self).initialize(probe="sk3x2pt")
