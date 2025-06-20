from pyDOE import lhs
import numpy as np
import emcee
from os.path import join as pjoin

def get_params_from_sample(sample, labels):
    """
    Format arrays into cocoa params
    Input:
        sample: 1D array, input parameters
        labels: 1D array, parameter names
    Output:
        params: dict, key:value = parameter name:value
    """
    assert len(sample)==len(labels), "Length of the labels not equal to the length of samples"
    params = {}
    for i, label in enumerate(labels):
        param_i = sample[i]
        params[label] = param_i
    return params

def get_params_list(samples, labels):
    """
    Input: 
        samples: 2D array of input sample, (Nsample, Nparam)
        labels: 1D array of input parameter names
    Output:
        params_list: 1D array of dicts
    """
    params_list = []
    for i in range(len(samples)):
        params = get_params_from_sample(samples[i], labels)
        params_list.append(params)
    return params_list

def get_params_from_lhs_sample(unit_sample, lhs_prior):
    """
    Format unit LHS arrays into cocoa params
    Input: 
        unit_sample: 1D array, normalized parameter values
        lhs_prior: dict
    Output:
        params: dict, parameter names : values
    """
    assert len(unit_sample)==len(lhs_prior), "Length of the labels not equal to the length of samples"
    params = {}
    for i, label in enumerate(lhs_prior):
        lhs_min = lhs_prior[label]['min']
        lhs_max = lhs_prior[label]['max']
        param_i = lhs_min + (lhs_max - lhs_min) * unit_sample[i]
        params[label] = param_i
    return params

def get_lhs_params_list(samples, lhs_prior):
    params_list = []
    for i in range(len(samples)):
        params = get_params_from_lhs_sample(samples[i], lhs_prior)
        params_list.append(params)
    return params_list

# ============= LHS samples =================

def get_lhs_samples(N_dim, N_lhs, lhs_minmax):
    ''' Generate Latin Hypercube sample at parameter space
    Input:
    ======
        - N_dim: 
            Dimension of parameter space
        - N_lhs:
            Number of LH grid per dimension in the parameter space
        - lhs_minmax:
            The boundary of parameter space along each dimension
    Output:
    =======
        - lhs_params:
            LHS of parameter space
    '''
    unit_lhs_samples = lhs(N_dim, N_lhs)
    lhs_params = get_lhs_params_list(unit_lhs_samples, lhs_minmax)
    return lhs_params

def get_gaussian_samples(param_fid, param_label, param_prior, N_sample,
        param_cov, temp, shift):
    ''' Generate Gaussian sample at parameter space
    Input:
    ======
        - param_fid: list of double
            Center of the Gaussian distribution
        - param_label: list of string
            Labels of the parameters
        - param_prior: dict
            param block of the yaml file
        - N_sample: int
            Number of samples drawn
        - param_cov: string
            Filename of the parameter covariance to draw from
        - temp: float
            Temperature applied to the Gaussian distribution (likelihood)
        - shift: dict
            Shift along each parameter space dimension
    '''
    gauss_cen = np.array(param_fid.copy())
    Ndim, Nwalker = len(gauss_cen), 4*len(gauss_cen)
    cov = retrieveParamCov(param_cov, param_label)
    param_std = np.diag(cov)**0.5
    invcov = np.linalg.inv(cov)

    # apply shift
    _map = {k:v for v,k in enumerate(param_label)}
    if shift is not None:
        for param in shift:
            # TODO: include sigma_8 shift
            if param in _map:
                i = _map[param]
                gauss_cen[i] += shift[param]
            elif param == "sigma8":
                _val, _lab = As2sigma8(gauss_cen, param_label)
                i = np.where(_lab=="sigma8")[0]
                _val[i] += shift[param]
                gauss_cen, _ = sigma82As(_val, _lab)
            else:
                print(f'Parameter {param} in shift can not be recognized!')
                exit(1)

    # setup likelihood
    def lnprior(param):
        ans = 0.0
        for i,par in enumerate(param_label):
            prior = param_prior[par]["prior"]
            dist = prior.get("dist", "uniform")
            if dist == "uniform":
                if param[i] < prior["min"] or param[i]>prior["max"]:
                    ans += -np.inf
            elif dist == "norm":
                # temp here?
                ans += -(0.5)*((param[i]-prior["loc"])/prior["scale"])**2
        # BBN hard prior
        if "omegab" in param_label and "H0" in param_label:
            _par_dict = {k:v for k,v in zip(param_label, param)}
            ombh2 = _par_dict["omegab"]*(_par_dict["H0"]/100)**2
            if ombh2<0.005 or ombh2 > 0.04:
                ans += -np.inf
        return ans
    def lnlkl(param):
        diff = param - gauss_cen
        return (-0.5/temp) * (diff @ invcov @ np.transpose(diff))
    def lnpost(param):
        return lnprior(param)+lnlkl(param)

    # start sampling
    print(f'Retrieving samples...')
    N_mcmc = int(N_sample*40/Nwalker)
    p0 = gauss_cen[np.newaxis] + 0.3*param_std[np.newaxis]*np.random.normal(size=(Nwalker, Ndim))
    sampler = emcee.EnsembleSampler(Nwalker, Ndim, lnpost)
    sampler.run_mcmc(p0, N_mcmc, progress=True)
    sample = sampler.get_chain(flat=True,thin=10,discard=N_mcmc//2)
    subset = np.random.choice(len(sample), size=N_sample, replace=False)
    print(f'Retrieved {N_sample} parameters.')
    return sample[subset,:]


def retrieveParamCov(param_cov, param_label):
    cov = np.genfromtxt(param_cov, names=True)
    N_in = len(cov); N_out = len(param_label)
    _map = {k:v for v,k in enumerate(cov.dtype.names)}
    cov = cov.view(float).reshape([N_in, N_in])
    cov_out = np.zeros([N_out, N_out])
    for i,pi in enumerate(param_label):
        for j,pj in enumerate(param_label):
            ii = _map.get(pi, -1)
            jj = _map.get(pj, -1)
            if ii<0 or jj<0:
                cov_out[i,j] = 0. if i!=j else 1.
                print(f'Parameter {pi}/{pj} not found in Gaussian covariance!')
            else:
                cov_out[i,j] = cov[ii,jj]
    return cov_out

def readDatasetFile(filename, root=None):
        ''' Read the likelihood dataset file
        Input:
        ======
            - filename: filename of the dataset file
        Output:
        =======
            - dataset: dataset file converted to a dict
        '''
        dataset = {}
        if root is not None:
            filename = pjoin(root, filename)
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line=='' or line=='\n' or line[0]=='#':
                    continue
                split_line = (line.replace(' ', '').replace('\n','')).split('=')
                if(len(split_line)==2):
                    dataset[split_line[0]] = split_line[1]
                else:
                    print(f'Can not read line: {line}')
                    exit(1)
        return dataset

def As2sigma8(value, label):
    ''' Change parameters from As_1e9, Omega_m, ... to sigma8, Omega_m, ...
    '''
    _val_dict = {k:v for k,v in zip(label, value)}

    h = _val_dict["H0"]/100
    omnh2 = (3.046/3)**(3/4)/94.1 * _val_dict.get("mnu", 0.06)
    omn = omnh2/(h**2)
    omc = _val_dict["omegam"]-_val_dict["omegab"]-omn
    ombh2 = _val_dict["omegab"]*(h**2)
    omch2 = omc*(h**2)
    ommh2 = _val_dict["omegam"]*(h**2)
    As = _val_dict["As_1e9"]/1.0e9
    sigma8 = (As/3.135e-9)**(1/2) * \
              (ombh2/0.024)**(-0.272) * \
              (ommh2/0.14)**(0.513) * \
              (3.123*h)**((_val_dict["ns"]-1)/2) * \
              (h/0.72)**(0.698) * \
              (_val_dict["omegam"]/0.27)**(0.236) * \
              (1-0.014)
    new_label = label.copy(); new_value = value.copy()
    i = np.where(label=="As_1e9")[0]
    new_label[i] = "sigma8"
    new_value[i] = sigma8
    return new_value, new_label

def sigma82As(value, label):
    ''' Change parameters from sigma8, Omega_m, ... to As_1e9, Omega_m, ...
    '''
    _val_dict = {k:v for k,v in zip(label, value)}

    h = _val_dict["H0"]/100
    omnh2 = (3.046/3)**(3/4)/94.1 * _val_dict.get("mnu", 0.06)
    omn = omnh2/(h**2)
    omc = _val_dict["omegam"]-_val_dict["omegab"]-omn
    ombh2 = _val_dict["omegab"]*(h**2)
    omch2 = omc*(h**2)
    ommh2 = _val_dict["omegam"]*(h**2)
    sigma8 = _val_dict["sigma8"]
    step = (sigma8/(1-0.014)) * \
            (ombh2/0.024)**(0.272) * \
            (ommh2/0.14)**(-0.513) * \
            (3.123*h)**(-(_val_dict["ns"]-1)/2) * \
            (h/0.72)**(-0.698) * \
            (_val_dict["omegam"]/0.27)**(-0.236)
    As_1e9 = (step**2)*3.135
    new_label = label.copy(); new_value = value.copy()
    i = np.where(label=="sigma8")[0]
    new_label[i] = "As_1e9"
    new_value[i] = As_1e9
    return new_value, new_label
