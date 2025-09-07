import yaml
import numpy as np
import os
from os.path import join as pjoin
import re
import copy
from .utils import readDatasetFile, get_shear_multi_bias_bitmask, get_linear_gal_bias_bitmask

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except:
    rank = 0
    size = 1

def mprint(*args, **kwargs):
    if rank==0:
        print(*args, **kwargs)

class Config:
    ''' Emulator Configuration Class for Roman Fourier Project '''
    # multi-probe mask, in sequence of Cl_EE, Cl_gE, Cl_gg
    # valid probes:
    # xi, 2x2pt, 3x2pt
    probe_mask_choices = {
        "xi":    [1, 0, 0,],
        "2x2pt":    [0, 1, 1,],
        "3x2pt":    [1, 1, 1,],
        # "Cl_gg":    [0, 0, 1,],
        # "Cl_gE":    [0, 1, 0,],
        # "EE_gE":    [1, 1, 0,],
    }

    def __init__(self, configfile):
        ''' Initialize the Config object with an configuration YAML file '''
        with open(configfile, "r") as stream:
            config_args = yaml.safe_load(stream)
        mprint(f'config.py: Initializing the emulator Config object...')
        # save the emulator section
        self.config_args_emu = config_args['emulator']
        self.survey_name = self.config_args_emu["survey_name"]
        # save the sampled parameters section
        # All sampled parameters must be spelt out in the YAML
        self.params = config_args['params'] 
        # save the likelihood section
        config_args_lkl = config_args['likelihood']

        self.load_lkl(config_args_lkl)
        self.load_params(self.params)     
        self.load_emu(self.config_args_emu)

    def load_lkl(self, config_args_lkl):
        ''' Setup likelihood related datasets, including
            - fiducial data vector
            - scale cut mask
            - covariance (including point-mass marginalization template)
            - baryonic feedback PCs
            - number of lens and source bins (and ggl exclude bins)
            - number of angular bins
        Input:
        ======
            - config_args_lkl: the `likelihood` section in the YAML file, dict
        '''
        mprint(f'config.py: Loading likelihood dataset...')
        assert len(config_args_lkl.keys())==1, f'Training config YAML must contain only one likelihood!'
        self.likelihood = list(config_args_lkl.keys())[0]
        # parse the probe in the training likelihood
        project, project_probe = self.likelihood.split('.')
        match = re.match(project+r'_(\S*)', project_probe)
        self.probe = match.group(1)
        if self.probe=='cosmic_shear':
            self.probe = 'xi'
        mprint(f'Initializing with probe = {self.probe}')
        self.probe_mask = self.probe_mask_choices[self.probe]
        self.config_args_lkl = config_args_lkl[self.likelihood]
        
        mprint(f'Loading dataset {self.config_args_lkl["data_file"]}')
        dataset = readDatasetFile(self.config_args_lkl['data_file'], 
            root=self.config_args_lkl['path'])
        #dst = pjoin(self.config_args_lkl['path'], "datasets")
        dst = self.config_args_lkl['path']

        # Read data vector & tomography dimension
        self.source_ntomo = int(dataset.get("source_ntomo", 0))
        self.lens_ntomo = int(dataset.get("lens_ntomo", 0))
        self.Nell = int(dataset.get("n_cl", 0))
        self.ggl_exclude = self.config_args_lkl["ggl_exclude"]
        self.lens_eq_src = self.config_args_lkl["lens_equal_source"]
        self.N_ggl_exclude = len(self.ggl_exclude)
        self.probe_size = [
            int(self.source_ntomo*(self.source_ntomo+1)*self.Nell/2.0),
            (self.source_ntomo*self.lens_ntomo - self.N_ggl_exclude)*self.Nell,
            self.lens_ntomo*self.Nell,
        ]
        self.probe_total_size = np.sum(self.probe_size)
        self.shear_calib_mask = get_shear_multi_bias_bitmask(
            self.source_ntomo, self.lens_ntomo, self.ggl_exclude, self.Nell,
            type_2pcf = "fourier")
        self.galaxy_bias_mask = get_linear_gal_bias_bitmask(
            self.source_ntomo, self.lens_ntomo, self.ggl_exclude, self.Nell,
            type_2pcf = "fourier")

        # Read mask, data vector, and baryon feedback PCs
        mprint(f'Loading scale cut mask from {dataset["mask_file"]}')
        self.mask_lkl = np.loadtxt(pjoin(dst, dataset["mask_file"]))[:,1].astype(bool)
        mprint(f'Loading fiducial data vector from {dataset["data_file"]}')
        self.dv_lkl = np.loadtxt(pjoin(dst, dataset["data_file"]))[:,1]
        assert len(self.dv_lkl)==self.probe_total_size
        mprint(f'Data vector dimension: {self.probe_total_size}')
        try:
            self.baryon_pcas = np.loadtxt(pjoin(dst,dataset["baryon_pca_file"]))
            mprint(f'Loading baryonic feedback PCs from {dataset["baryon_pca_file"]}')
        except:
            self.baryon_pcas = None
            mprint(f'Can not find baryonic feedback PCs, skip PCA...')

        # Read covariance and point-mass correction -> inv cov
        mprint(f'Loading covariance matrix {dataset["cov_file"]}')
        invcov = self.get_full_cov(pjoin(dst, dataset["cov_file"]))
        self.dv_std = np.sqrt(np.diagonal(invcov))
        invcov = np.linalg.inv(invcov[self.mask_lkl][:,self.mask_lkl])
        # Add PM marginalization
        if "U_PMmarg" in dataset:
            U_PMmarg = np.loadtxt(pjoin(dst, dataset["U_PMmarg"]))
            mprint(f'Loading point-mass marginalization template from {dataset["U_PMmarg"]}')
            U = np.zeros([self.probe_total_size, self.lens_ntomo])
            for line in U_PMmarg:
                i, j = int(line[0]), int(line[1])
                U[i,j] = float(line[2])
            U = U[self.mask_lkl,:]
            central_block = np.diag(np.ones(self.lens_ntomo)) + U.T@invcov@U
            w, v = np.linalg.eig(central_block)
            assert np.min(w)>=0, f'Central block not positive-definite!'
            corr = invcov @ (U@np.linalg.inv(central_block)@U.T) @ invcov
            invcov -= corr
        else:
            mprint(f'Can not find point-mass marginalization template, skip PMmarg...')
        self.masked_inv_cov = invcov
        # test positive-definite; compare accu between Python v.s. C++ PMmarg
        w, v = np.linalg.eig(self.masked_inv_cov)
        assert np.min(w)>=0, f'Precision matrix not positive-definite after PMmarg!'
        self.inv_cov = np.zeros([self.probe_total_size,self.probe_total_size])
        for i in range(self.inv_cov.shape[0]):
            for j in range(self.inv_cov.shape[1]):
                if (self.mask_lkl[i]>0) and (self.mask_lkl[j]>0):
                    i_reduce, j_reduce = int(self.mask_lkl[:i].sum()), int(self.mask_lkl[:j].sum())
                    self.inv_cov[i,j] = self.masked_inv_cov[i_reduce,j_reduce]
        mprint(f'config.py: Likelihood dataset loaded.')

    def load_emu(self, config_args_emu):
        ''' Read emulator related data '''
        mprint(f'config.py: Loading emulator training configuration...')
        self.derived = 1
        self.sigma8_fid = np.array([config_args_emu['derived']['sigma8_fid']])
        self.sigma8_std = np.array([config_args_emu['derived']['sigma8_std']])
        try:
            self.chi_sq_cut = config_args_emu['training']['chi_sq_cut']
        except:
            self.chi_sq_cut = 1e+5

        # Set I/O path
        self.savedir = config_args_emu['io']['savedir']
        os.makedirs(self.savedir, exist_ok=True)
        self.traindir = os.path.join(self.savedir, "training_sample")
        self.modeldir = os.path.join(self.savedir, "model_dataset")
        os.makedirs(self.traindir, exist_ok=True)
        os.makedirs(self.modeldir, exist_ok=True)
        try:
            self.save_train_data = config_args_emu['io']['save_train_data']
        except:
            self.save_train_data = False
        try:
            self.save_intermediate_model = config_args_emu['io']['save_intermediate_model']
        except:
            self.save_intermediate_model = False

        # Read emulator architecture
        self.emu_type = config_args_emu['training']['emu_type']
        self.loss_type = config_args_emu['training']['loss_type']
        self.learning_rate = config_args_emu['training']['learning_rate']
        self.weight_decay = config_args_emu['training']['weight_decay']
        self.reduce_lr = config_args_emu['training']['reduce_lr']
        if(self.emu_type.lower()=='nn'):
            self.batch_size = int(config_args_emu['training']['batch_size'])
            self.n_epochs = int(config_args_emu['training']['n_epochs'])
            try:
                self.nn_model  = int(config_args_emu['training']['nn_model'])
            except:
                self.nn_model  = 0
        elif(self.emu_type.lower()=='gp'):
            mprint(f'Gaussian Process is not supported currently!')
            exit(-1)
        else:
            mprint(f'Model {self.emu_type.lower()} is not supported!')
            exit(-1)

        # Read training sample settings
        _init_sample = config_args_emu['init_sample']
        self.init_sample_type = _init_sample["type"]
        if self.init_sample_type == "lhs":
            self.n_lhs = int(_init_sample['lhs_n'])
            self.lhs_minmax = self.get_lhs_minmax()
        elif self.init_sample_type == "gaussian":
            self.gauss_cov = _init_sample['gauss_cov']
            self.gtemp_t = _init_sample.get('gauss_temp_train', 1.)
            self.gshift_t = _init_sample.get('gauss_shift_train', None)
            self.gnsamp_t = _init_sample.get('n_train')
            self.gtemp_v = _init_sample.get('gauss_temp_valid', 1.)
            self.gshift_v = _init_sample.get('gauss_shift_valid', None)
            self.gnsamp_v = _init_sample.get('n_valid')
            self.gauss_minmax = self.get_gaussian_minmax()
        else:
            mprint(f'Can not recognize init sample type {self.init_sample_type}')
            exit(-1)
        self.n_train_iter = int(config_args_emu['training']['n_train_iter'])

        # Read the emcee sampler setting
        self.n_emcee_walkers=int(config_args_emu['sampling']['n_emcee_walkers'])
        self.n_mcmc = int(config_args_emu['sampling']['n_mcmc'])
        self.n_burn_in = int(config_args_emu['sampling']['n_burn_in'])
        self.n_thin = int(config_args_emu['sampling']['n_thin'])
        
        # Read parameter blocking settings
        try:
            _args_block = config_args_emu['sampling']['params_blocking']
            self.block_bias        = _args_block.get('block_bias', False)
            self.block_shear_calib = _args_block.get('block_shear_calib', False)
            self.block_dz          = _args_block.get('block_dz', False)
            self.block_ia          = _args_block.get('block_ia', False)
        except:
            self.block_bias        = False
            self.block_shear_calib = False
            self.block_dz          = False
            self.block_ia          = False
        try:
            block_label = config_args_emu['sampling']['params_blocking']['block_label'].split(',')
            block_value = config_args_emu['sampling']['params_blocking']['block_value'].split(',')
            block_value = [float(val) for val in block_value]
            block_indices = []
            for label in block_label:
                for i, param_label in enumerate(self.running_params):
                    if(label==param_label):
                        block_indices.append(i)
            self.block_indices = block_indices
            self.block_value   = block_value
        except:
            self.block_indices = []
            self.block_value   = None

        # Read debug outputs
        try:
            self.test_sample_file = config_args_emu['test']['test_samples']
            self.test_output_file = config_args_emu['test']['test_output']
        except:
            self.test_sample_file = None
            self.test_output_file = None

        mprint(f'config.py: Emulator training configuration loaded.')

    def load_params(self, param_args):
        ''' Initialize likelihood model parameter settings
        Note that shear calibration bias and baryonic feedback are fast params,
        not included in the running_params
        '''
        mprint(f'config.py: Loading sampled parameter space...')
        params_list = param_args.keys()

        self.running_params       = []
        # 1:cosmo 2:src nui 3:lens nui 4: nui shared by src and lens
        self.running_params_type  = []
        self.running_params_latex = []
        self.running_params_fid   = []
        self.running_params_min   = []
        self.running_params_max   = []
        self.n_pars_cosmo = 0
        self.n_fast_pars = 0
        self.m_shear_fid = np.zeros(self.source_ntomo)
        self.gal_bias_fid= np.ones(self.lens_ntomo)
        self.n_pcas_baryon = 0
        self.fast_linear_gal_bias = False

        for param in params_list:
            keys = param_args[param].keys()
            _args = param_args[param]
            # fast parameter: shear calibration bias
            # must be fixed, only have value and latex
            match = re.match(self.survey_name+r'_M(\d)', param)
            if match:
                i_src = int(match.group(1)) - 1
                self.n_fast_pars += 1
                self.m_shear_fid[i_src] = _args["value"]
                continue
            # fast parameter: baryonic feedback
            # must be fixed, only have value and latex
            match = re.match(self.survey_name+r'_BARYON_Q(\d)', param)
            if match:
                i_PC = int(match.group(1)) - 1
                self.n_fast_pars += 1
                self.n_pcas_baryon += 1
                continue
            # fast parameter: linear galaxy bias
            '''
            Note that linear galaxy bias can be slow or fast params.
            If we do not emulate linear galaxy bias, we fix them to 
            fiducial values and read them here.
            Otherwise, we leave it to slow parameter section below
            '''
            match = re.match(self.survey_name+r'_B1_(\d)', param)
            if match and ("value" in keys):
                i_lens = int(match.group(1)) - 1
                self.gal_bias_fid[i_lens] = _args["value"]
                self.n_fast_pars += 1
                mprint(f'{param}={self.gal_bias_fid[i_lens]} is treated as fast param')
                self.fast_linear_gal_bias = True
                continue

            # slow parameters: skip if not being sampled directly
            if(('value' in keys) or ('derived' in keys) or (len(keys)<=1)):
                continue
            # slow parameters: all the other parameters that need emulation
            self.running_params.append(param)
            self.running_params_latex.append(_args['latex'])
            # set the parameter boundary
            if (_args["prior"].get("dist", "uniform")=="uniform"):
                self.running_params_fid.append(_args["ref"]["loc"])
                self.running_params_min.append(_args["prior"]["min"])
                self.running_params_max.append(_args["prior"]["max"])
            else:
                self.running_params_fid.append(_args["prior"]["loc"])
                self.running_params_min.append(-np.inf)
                self.running_params_max.append(np.inf)
            # determine if the param is cosmological or nuisance
            if param.startswith(self.survey_name)==False:
                # cosmology parameters
                self.running_params_type.append(1)
                self.n_pars_cosmo += 1
            else:
                match = re.match(self.survey_name+r'_(\S*)_(\S*)(\d+)', param)
                # nuisance parameters: intrinsic alignment (source sample)
                if match.group(1) in ['A1', 'A2', 'BTA']:
                    self.running_params_type.append(2)
                # nuisance parameters: source photo-z (source sample)
                elif match.group(1)=='DZ' and match.group(2)=='S':
                    _nui_type_ = 4 if self.lens_eq_src else 2
                    self.running_params_type.append(_nui_type_)
                # nuisance parameters: lens photo-z (lens sample)
                elif match.group(1)=='DZ' and match.group(2)=='L':
                    if self.lens_eq_src:
                        print(f'ERROR: Please use src photo-z for lens=src!')
                        exit(-1)
                    self.running_params_type.append(3)
                # nuisance parameters: lens photo-z stretch (lens sample)
                # NOTE: so far the stretch is only implemented for lens sample
                # For source sample or lens=src sample photo-z stretch, 
                # implement later...
                elif match.group(1)=='STRETCH':
                    self.running_params_type.append(3)
                # nuisance parameters: linear galaxy bias (lens sample)
                elif match.group(1)=='B1':
                    self.running_params_type.append(3)
                    self.fast_linear_gal_bias = False
                else:
                    mprint(f'[config.py:Config.load_params]: Can not support param {param} now!')
                    exit(-1)
        self.n_dim = len(self.running_params) # total param emulated
        self.running_params_type = np.array(self.running_params_type)
        # params mask for each probe in [cosmic shear, ggl, clustering]
        if not self.lens_eq_src:
            self.probe_params_mask = [
                (self.running_params_type==1)|(self.running_params_type==2),
                (self.running_params_type==1)|(self.running_params_type==2)|(self.running_params_type==3),
                (self.running_params_type==1)|(self.running_params_type==3)
            ]
        else:
            self.probe_params_mask = [
                (self.running_params_type==1)|(self.running_params_type==2)|(self.running_params_type==4),
                (self.running_params_type==1)|(self.running_params_type==2)|(self.running_params_type==3)|(self.running_params_type==4),
                (self.running_params_type==1)|(self.running_params_type==3)|(self.running_params_type==4),
            ]

        mprint(f'config.py: Sampled parameter space loaded.')
        return


    def get_lhs_minmax(self):
        lh_minmax = {}
        for x in self.params:
            if('prior' in self.params[x]):
                prior = self.params[x]['prior']
                if('dist' in prior):
                    loc   = prior['loc']
                    scale = prior['scale']
                    lh_min = loc - 4. * scale
                    lh_max = loc + 4. * scale
                else:
                    lh_min = prior['min']
                    lh_max = prior['max']
                lh_minmax[x] = {'min': lh_min, 'max': lh_max}
        return lh_minmax

    def get_gaussian_minmax(self):
        gauss_minmax = {}
        for x in self.params:
            if('prior' in self.params[x]):
                prior = self.params[x]['prior']
                dist = prior.get("dist", "uniform")
                if dist=="norm":
                    gauss_min = -np.inf
                    gauss_max = np.inf
                else:
                    gauss_min = prior['min']
                    gauss_max = prior['max']
                gauss_minmax[x] = {'min': gauss_min, 'max': gauss_max}
        return gauss_minmax
    
    def get_full_cov(self, cov_file):
        full_cov = np.loadtxt(cov_file)
        Ndim = len(self.dv_lkl)
        cov = np.zeros((Ndim, Ndim))
        cov_scenario = full_cov.shape[1]
        
        for line in full_cov:
            i = int(line[0])
            j = int(line[1])

            if(cov_scenario==3):
                cov_ij = line[2]
            elif(cov_scenario==10):
                cov_g_block  = line[8]
                cov_ng_block = line[9]
                cov_ij = cov_g_block + cov_ng_block

            cov[i,j] = cov_ij
            cov[j,i] = cov_ij

        return cov
