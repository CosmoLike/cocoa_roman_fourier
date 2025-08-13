import numpy as np
import os
from os.path import join as pjoin
from getdist import IniFile
from cobaya.likelihoods.base_classes import DataSetLikelihood
import torch
### Import the emulator from relative path
import sys
from pathlib import Path
# Get directory containing THIS script
script_dir = Path(__file__).resolve().parent
# Add desired relative path
sys.path.append(str(script_dir.parent))
from cocoa_emu import Config
from cocoa_emu.emulator import NNEmulator

probe_fmts = ["Cl_EE", "Cl_gE", "Cl_gg"]

class _cosmolike_emu_prototype_base(DataSetLikelihood):
	''' Attributes needed from the likelihood yaml file:
	- train_config: filename of the training config file
	'''
	def initialize(self, probe):
		super(_cosmolike_emu_prototype_base, self)
		torch.set_num_threads(1)
		self.device = torch.device("cpu")
		self.probe = probe

		# Note: This config file is only used to calculate data vector
		#       The baryon PCs and priors are overwritten during sampling
		self.log.info(f'Init emulator with config file {self.train_config}')
		config = Config(self.train_config)
		assert config.emu_type.lower()=='nn', f'Only support NN emulator now!'
		self.probe_mask        = config.probe_mask_choices[self.probe]
		self.probe_size        = config.probe_size
		self.survey_name       = config.survey_name

		### Read dataset file: data vector, covariance, mask
		self.log.info("Loading likelihood dataset...")
		ini = IniFile(os.path.normpath(pjoin(self.path, self.data_file)))
		self.data_vector_file = ini.relativeFileName('data_file')
		self.cov_file = ini.relativeFileName('cov_file')
		try:
			self.U_PMmarg_file = ini.relativeFileName('U_PMmarg')
		except:
			self.log.info("Skip point-mass marginalization.")
			self.U_PMmarg_file = ""
		self.mask_file    = ini.relativeFileName('mask_file')
		self.source_ntomo = config.source_ntomo
		self.lens_ntomo   = config.lens_ntomo
		self.Nell         = config.Nell
		self.dv_size      = config.probe_total_size
		
		self.init_data(config)

		### Initialize baryon feedback PCs
		if ini.string('baryon_pca_file', default=''):
			baryon_pca_file = ini.relativeFileName('baryon_pca_file')
			self.baryon_pcs = np.loadtxt(baryon_pca_file)
			self.log.info('use_baryon_pca = True')
			self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
			self.use_baryon_pca = True
		else:
			self.log.info('use_baryon_pca = False')
			self.use_baryon_pca = False
		self.baryon_pcs_qs = np.zeros(4)

		self.shear_calib_mask  = config.shear_calib_mask
		self.n_pars_cosmo      = config.n_pars_cosmo
		self.running_params    = config.running_params
		self.m_shear_fid       = config.m_shear_fid

		### read emulators
		# try include emu_list as object attribute. If not work, global variable
		self.log.info("Reading emulator models...")
		self.emu_list = []
		for i,p in enumerate(probe_fmts):
			_l, _r = sum(config.probe_size[:i]), sum(config.probe_size[:i+1])
			fn = pjoin(config.modeldir, f'{p}_nn{config.nn_model}')
			if os.path.exists(fn+".h5"):
				self.log.info(f'--- Reading {p} NN emulator from {fn}.h5 ...')
				emu = NNEmulator(config.n_dim, config.probe_size[i], 
					config.dv_lkl[_l:_r], config.dv_std[_l:_r],
					config.inv_cov[_l:_r,_l:_r],
					mask=config.mask_lkl[_l:_r],
					param_mask=config.probe_params_mask[i],
					model=config.nn_model, device=self.device,
					deproj_PCA=True, lr=config.learning_rate, 
					reduce_lr=config.reduce_lr, 
					weight_decay=config.weight_decay, dtype="double")
				emu.load(fn)
			else:
				self.log.info(f'{fn} not found! Ignore probe {p} emulator!')
				emu = None
			self.emu_list.append(emu)
		# read sigma8 emulator
		fn = pjoin(config.modeldir, f'sigma8_nn{config.nn_model}')
		if os.path.exists(fn+".h5"):
			self.log.info(f'--- Reading sigma8 NN emulator from {fn}.h5 ...')
			self.emu_s8 = NNEmulator(config.n_pars_cosmo, 1, config.sigma8_fid, 
					config.sigma8_std, np.atleast_2d(1.0/config.sigma8_std**2), 
					model=config.nn_model, device=self.device,
					deproj_PCA=False, lr=config.learning_rate, 
					reduce_lr=config.reduce_lr, 
					weight_decay=config.weight_decay, dtype="double")
			self.emu_s8.load(fn)
		else:
			self.log.info(f'{fn} not found! Ignore sigma8 emulator!')
			self.emu_s8 = None
		self.log.info("Emulator likelihood initialized!")

	def init_data(self, config):
		''' Prepare the likelihood dataset
		Including inverse covariance, data vector, data vector mask
		Equivalent to `ci.init_data`
		'''
		### prepare data vector & mask
		self.dv   = config.dv_lkl.copy()
		self.mask = config.mask_lkl.copy()
		# update the mask if some probes are not included
		for i in range(3):
			_l, _r = sum(config.probe_size[:i]), sum(config.probe_size[:i+1])
			if self.probe_mask[i]==0:
				self.mask[_l:_r] = 0.0
				self.log.info(f'Probe {probe_fmts[i]} is not included.')
			else:
				_Ndp = self.mask[_l:_r].sum()
				self.log.info(f'Probe {probe_fmts[i]} has {_Ndp} elements after scale cut.')
		self.masked_inv_cov = config.masked_inv_cov.copy()

	def emu_predict(self, theta):
		''' Get the emulator prediction for slow parameters
		'''
		theta = torch.Tensor(theta)
		# evaluate data vector using list of emulators
		model_vectors = []
		for i in range(6):
			if self.probe_mask[i]==1:
				_mv = self.emu_list[i].predict(theta)[0]
			else:
				_mv = np.zeros(self.probe_size[i])
			model_vectors.append(_mv)
		modelvector = np.hstack(model_vectors)
		return modelvector

	def get_sigma8_emu(self, **params_values):
		theta = np.array([params_values.get(p, 0.0) for p in self.running_params])
		theta = torch.Tensor(theta[:self.n_pars_cosmo])
		sigma8 = self.emu_s8.predict(theta)[0]
		return sigma8

	def get_model_vector_emu(self, **params_values):
		''' Evaluate model vector given parsed input sampled parameter array
		Note that linear galaxy bias are slow parameters due to magnification
		bias and RSD.
		'''
		# get model vector from emulated parameters
		theta = np.array([params_values.get(p, 0.0) for p in self.running_params])
		mv = self.emu_predict(theta)

		# add shear calibration bias
		m = np.array([params_values.get(self.survey_name+f'_M{i+1}', 0.0) for i in range(self.source_ntomo)])
		for i in range(self.source_ntomo):
			factor = ((1+m[i])/(1+self.m_shear_fid[i]))**self.shear_calib_mask[i]
			mv = factor * mv
		
		# add baryon feedback PCs
		if self.use_baryon_pca:
			# Warning: we assume the PCs were created with the same mask
			# We have no way of testing user enforced that
			self.set_baryon_related(**params_values)
			mv = self.add_baryon_pcs_to_datavector(mv)
		return mv

	def set_baryon_related(self, **params_values):
		self.baryon_pcs_qs[0] = params_values.get(self.survey_name+"_BARYON_Q1", 0.0)
		self.baryon_pcs_qs[1] = params_values.get(self.survey_name+"_BARYON_Q2", 0.0)
		self.baryon_pcs_qs[2] = params_values.get(self.survey_name+"_BARYON_Q3", 0.0)
		self.baryon_pcs_qs[3] = params_values.get(self.survey_name+"_BARYON_Q4", 0.0)

	def add_baryon_pcs_to_datavector(self, datavector):
		return datavector[:] + self.baryon_pcs_qs[0]*self.baryon_pcs[:,0] \
		  + self.baryon_pcs_qs[1]*self.baryon_pcs[:,1] \
		  + self.baryon_pcs_qs[2]*self.baryon_pcs[:,2] \
		  + self.baryon_pcs_qs[3]*self.baryon_pcs[:,3]

	def logp(self, **params_values):
		''' Evaluate the log-posterior of the likelihood
		Input:
		======
		params_values: dict
			dictionary of sampled input parameters
		'''
		mv = self.get_model_vector_emu(**params_values)
		delta_dv = (mv - self.dv)[self.mask]
		log_p = -0.5 * delta_dv @ self.masked_inv_cov @ delta_dv

		# derived parameters: sigma8
		if self.derive_sigma8 and self.emu_s8:
			params_values["_derived"]['sigma8'] = self.get_sigma8_emu(**params_values)

		return log_p
