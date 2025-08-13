from cobaya.likelihoods.roman_fourier._cosmolike_emu_prototype_base import _cosmolike_emu_prototype_base

class roman_fourier_2x2pt_emu(_cosmolike_emu_prototype_base):
	''' Attributes needed from the likelihood yaml file:
	- train_config: filename of the training config file
	'''
	def initialize(self):
		super(roman_fourier_2x2pt_emu, self).initialize(probe="2x2pt")
