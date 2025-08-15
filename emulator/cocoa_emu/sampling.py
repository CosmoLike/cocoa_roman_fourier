import yaml
import numpy as np
import torch

def hard_prior(theta, params_prior):
    is_lower_than_min = bool(np.sum(theta < params_prior[:,0]))
    is_higher_than_max = bool(np.sum(theta > params_prior[:,1]))
    if is_lower_than_min or is_higher_than_max:
        return -np.inf
    else:
        return 0.
    
def gaussian_prior(theta, params_prior):
    mu  = params_prior[:,0]
    std = params_prior[:,1]
    y = (theta - mu) / std
    return -0.5 * np.sum(y * y)

def split_with_comma(configline):
    configline_split = configline.split(',')
    configline_list = []
    for obj in configline_split:
        configline_list.append(float(obj))
    return np.array(configline_list)
    
class EmuSampler:
    def __init__(self, emu_list, config):
        self.emu_list          = emu_list
        self.params            = config.params
        self.probe             = config.probe
        self.probe_mask        = config.probe_mask
        self.probe_size        = config.probe_size
        self.probe_params_mask = config.probe_params_mask
        self.running_params    = config.running_params
        
        self.n_walkers         = config.n_emcee_walkers
        self.n_pcas_baryon     = config.n_pcas_baryon
        self.baryon_pcas       = config.baryon_pcas
        
        self.emu_type          = config.emu_type
        self.mask              = config.mask_lkl
        self.inv_cov           = config.inv_cov
        self.masked_inv_cov    = config.masked_inv_cov
        self.dv_lkl            = config.dv_lkl
        self.shear_calib_mask  = config.shear_calib_mask
        
        self.source_ntomo      = config.source_ntomo
        if config.probe != 'xi':
            self.lens_ntomo = config.lens_ntomo
        
        self.n_fast_pars       = config.n_fast_pars
        
        self.m_shear_fid       = config.m_shear_fid
        try:
            # self.m_shear_prior_std = split_with_comma(config.config_args_emu['shear_calib']['prior_std'])
            self.m_shear_prior_std = np.array(config.config_args_emu['shear_calib']['prior_std'])
        except:
            print(f'Can not find `shear_calib/prior_std` in training YAML!')
            print(f'`shear_calib/prior_std` is an optional setting. It is required when you want to use EmuSampler class to sample a posterior.')
        try:
            self.config_args_baryons = config.config_args_emu['baryons']
        except:
            print(f'Can not find `baryons` in training YAML!')
            print(f'`baryons` is an optional setting. It is required when you want to use EmuSampler class to sample a posterior.')
        
        # Move linear gbias to slow parameters
        # if self.probe!='cosmic_shear':
        #     self.config_args_bias  = config.config_args_emu['galaxy_bias']
        #     try:
        #         self.bias_prior_type = self.config_args_bias['prior_type']
        #     except:
        #         self.bias_prior_type = 'flat'            
        #     self.bias_fid          = split_with_comma(self.config_args_bias['bias_fid'])
        #     self.galaxy_bias_mask  = config.galaxy_bias_mask
        
        self.get_priors()
        
        # fast params dimension
        if self.probe=='wtheta':
            # don't consider shear calibration bias and baryons PCA here
            self.n_sample_dims    = config.n_dim
        else:
            self.n_sample_dims    = config.n_dim + self.source_ntomo + config.n_pcas_baryon

        self.block_indices     = config.block_indices
        self.block_value       = config.block_value
        self.block_bias        = config.block_bias
        self.block_shear_calib = config.block_shear_calib
        # self.block_dz          = config.block_dz
        # self.block_ia          = config.block_ia
        
    def get_priors(self):
        gaussian_prior_indices = []
        gaussian_prior_parameters = []

        flat_prior_indices    = []
        flat_prior_parameters = []

        ### Read priors from params block in training config YAML
        ind = 0
        for x in self.params:
            if 'prior' in self.params[x]:
                prior = self.params[x]['prior']
                # Gaussian prior
                if prior.get("dist", "uniform")=="norm":
                    gaussian_prior_indices.append(ind)
                    gaussian_prior_parameters.append([prior['loc'], prior['scale']])
                # Flat prior
                else:
                    flat_prior_indices.append(ind)
                    flat_prior_parameters.append([prior['min'], prior['max']])
                ind += 1

        self.flat_prior_indices     = flat_prior_indices
        self.gaussian_prior_indices = gaussian_prior_indices
        
        self.gaussian_prior_parameters = np.array(gaussian_prior_parameters)
        self.flat_prior_parameters     = np.array(flat_prior_parameters)
        
        ### Read shear calibration bias prior
        self.m_shear_prior_parameters = np.array([self.m_shear_fid, self.m_shear_prior_std]).T
        ### Read baryonic feedback PCs prior
        if(self.n_pcas_baryon > 0):
            baryon_priors = []
            for i in range(self.n_pcas_baryon):
                baryon_prior_i = self.config_args_baryons['prior_Q%d'%(i+1)].split(',')
                baryon_priors.append([float(baryon_prior_i[0]), float(baryon_prior_i[-1])])
            self.baryon_priors = np.array(baryon_priors)
            print("baryon_priors: "+str(self.baryon_priors))

    def get_starting_pos(self):
        ''' Get starting position of MCMC sampler
        '''
        p0 = []
        # slow parameters
        for x in self.params:
            if('prior' in self.params[x]):
                loc   = float(self.params[x]['ref']['loc'])
                scale = float(self.params[x]['ref']['scale'])
                # ensure the initial sample falls inside prior
                if "prior" in self.params[x]:
                    if self.params[x]["prior"].get("dist", "uniform") == "uniform":
                        x_min, x_max = self.params[x]["prior"]["min"], self.params[x]["prior"]["max"]
                else:
                    x_min, x_max = -np.inf, np.inf
                p0_i = truncnorm.rvs(x_min, x_max, loc=loc, scale=scale, 
                    size=self.n_walkers)
                p0.append(p0_i)
        p0 = np.array(p0).T
        # fast parameters
        fast_pars_std = self.m_shear_prior_std
        fast_pars_mean = self.m_shear_fid
        if(self.n_pcas_baryon > 0):
            baryon_std = np.hstack([0.1 * np.ones(self.n_pcas_baryon)])
            fast_pars_std = np.hstack([fast_pars_std, baryon_std])
            fast_pars_mean = np.hstack([fast_pars_mean, np.zeros(self.n_pcas_baryon)])
        p0_fast = fast_pars_std[np.newaxis] * np.random.normal(size=(self.n_walkers, self.n_fast_pars)) + fast_pars_mean[np.newaxis]
        p0 = np.hstack([p0, p0_fast])
        return p0
            
    def _compute_datavector(self, theta):
        theta = np.array(theta)
        assert self.emu_type=='nn', f'Can not support GP anymore'
        theta = torch.Tensor(theta)
        # evaluate data vector using list of emulators
        model_vectors = []
        for i in range(3):
            if self.probe_mask[i]==1:
                _mv = self.emu_list[i].predict(theta)[0]
            else:
                _mv = np.zeros(self.probe_size[i])
            model_vectors.append(_mv)
        modelvector = np.hstack(model_vectors)
        return modelvector

    def _add_baryon_q(self, Q, datavector):
        for i in range(self.n_pcas_baryon):
            datavector = datavector + Q[i] * self.baryon_pcas[:,i]
        return datavector

    def _add_shear_calib(self, m, datavector):
        for i in range(self.source_ntomo):
            factor = ((1+m[i])/(1+self.m_shear_fid[i]))**self.shear_calib_mask[i]
            datavector = factor * datavector
        return datavector

    def get_data_vector_emu(self, theta, skip_fast=False):
        theta_emu     = theta[:-self.n_fast_pars]
        datavector = self._compute_datavector(theta_emu)
        if skip_fast:
            return datavector
        else:
            # ============== Add shear calibration bias ========================
            if (self.probe!='wtheta'):
                _l = self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo)
                _r = self.n_sample_dims-self.n_pcas_baryon
                m_shear_theta = theta[_l:_r]
                if not self.block_shear_calib:
                    datavector = self._add_shear_calib(m_shear_theta, datavector)       
            # ======================== Add baryons =============================
            if(self.n_pcas_baryon > 0):
                baryon_q   = theta[-self.n_pcas_baryon:]
                datavector = self._add_baryon_q(baryon_q, datavector)
            return datavector

    def ln_prior(self, theta):        
        flat_prior_theta     = theta[self.flat_prior_indices]
        gaussian_prior_theta = theta[self.gaussian_prior_indices]
        m_shear_theta        = theta[self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo):
                                     self.n_sample_dims-self.n_pcas_baryon]
        if len(flat_prior_theta)>0:
            prior_flat    = hard_prior(flat_prior_theta, self.flat_prior_parameters)
        else:
            prior_flat = 0.
        if len(gaussian_prior_theta)>0:
            prior_gauss   = gaussian_prior(gaussian_prior_theta, self.gaussian_prior_parameters)
        else:
            prior_gauss = 0.
        if not self.block_shear_calib:
            prior_m_shear = gaussian_prior(m_shear_theta, self.m_shear_prior_parameters)
        else:
            prior_m_shear = 0.
        if(self.n_pcas_baryon > 0):
            baryon_q   = theta[-self.n_pcas_baryon:]
            prior_baryons = hard_prior(baryon_q, self.baryon_priors)
        else:
            prior_baryons = 0.
        # BBN consistency prior
        _param_dict = {k:v for k,v in zip(self.running_params, theta[:-self.n_fast_pars])}
        if ("omegab" in _param_dict) and ("H0" in _param_dict):
            ombh2 = _param_dict["omegab"]*(_param_dict["H0"]/100)**2
        elif ("omegab" in _param_dict) and ("h" in _param_dict):
            ombh2 = _param_dict["omegab"]*_param_dict["h"]**2
        elif "omegabh2" in _param_dict:
            ombh2 = _param_dict["omegabh2"]
        else:
            ombh2 = 0.02
        if ombh2<0.005 or ombh2>0.04:
            prior_bbn_consistency = -np.inf
        else:
            prior_bbn_consistency = 0.
        # w0-wa parametrization
        prior_DE_EOS = 0.0
        if ("w" in _param_dict) and ("wa" in _param_dict):
            if _param_dict["w"] + _param_dict["wa"] >= - 0.01:
                prior_DE_EOS = -np.inf
        elif ("w0" in _param_dict) and ("wa" in _param_dict):
            if _param_dict["w0"] + _param_dict["wa"] >= - 0.01:
                prior_DE_EOS = -np.inf
        elif ("w0pwa" in _param_dict):
            if _param_dict["w0pwa"] >= - 0.01:
                prior_DE_EOS = -np.inf
                
        return prior_flat + prior_gauss + prior_m_shear + prior_baryons + prior_bbn_consistency + prior_DE_EOS
    
    def ln_lkl(self, theta):
        model_datavector = self.get_data_vector_emu(theta)
        delta_dv = (model_datavector - self.dv_lkl)[self.mask]
        return -0.5 * delta_dv @ self.masked_inv_cov @ delta_dv

    def ln_prob(self, theta, temper=1.):
        if self.block_value is not None:
            theta[self.block_indices] = self.block_value
        return self.ln_prior(theta) + (1/temper) * self.ln_lkl(theta)
