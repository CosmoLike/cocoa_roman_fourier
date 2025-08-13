import sys
import os
print("Current working directory: ", os.getcwd())
from os.path import join as pjoin
import numpy as np
import torch
import cobaya
from cocoa_emu import Config
from cocoa_emu.emulator import NNEmulator
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('config', type=str, help='Configuration file')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Overwrite existing model files')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Turn on debugging mode')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_num_interop_threads(40) # Inter-op parallelism
    torch.set_num_threads(40) # Intra-op parallelism
torch.set_default_dtype(torch.double)
print('Using device: ',device)

#===============================================================================
config = Config(args.config)
print(f'\n>>> Start Emulator Training\n')
if config.init_sample_type == "lhs":
    print("We don't support LHS any more!")
    exit(1)
else:
    iss = f'{config.init_sample_type}'
    label_train = iss+f'_t{config.gtemp_t}_{config.gnsamp_t}'
    label_valid = iss+f'_t{config.gtemp_v}_{config.gnsamp_v}'
    N_sample_train = config.gnsamp_t
    N_sample_valid = config.gnsamp_v
#================== Loading Training & Validating Data =========================
print(f'Loading training data!')
train_samples = np.load(pjoin(config.traindir, f'samples_{label_train}.npy'))
train_data_vectors = np.load(pjoin(config.traindir, f'data_vectors_{label_train}.npy'))
train_sigma8 = np.load(pjoin(config.traindir, f'sigma8_{label_train}.npy'))
print(f'Training dataset dimension: {train_samples.shape}')
print(f'Loading validating data!')
valid_samples = np.load(pjoin(config.traindir, f'samples_{label_valid}.npy'))
valid_data_vectors = np.load(pjoin(config.traindir, f'data_vectors_{label_valid}.npy'))
valid_sigma8 = np.load(pjoin(config.traindir, f'sigma8_{label_valid}.npy'))
print(f'Validation dataset dimension: {valid_samples.shape}')
train_samples = torch.Tensor(train_samples)
train_data_vectors = torch.Tensor(train_data_vectors)
train_sigma8 = torch.Tensor(train_sigma8)
valid_samples = torch.Tensor(valid_samples)
valid_data_vectors = torch.Tensor(valid_data_vectors)
valid_sigma8 = torch.Tensor(valid_sigma8)
#================= Training emulator ===========================================
# switch according to probes
probes = ["Cl_EE", "Cl_gE", "Cl_gg"]
for i in range(len(config.probe_mask)):
    print("============= Training %s Emulator ================="%(probes[i]))
    l, r = sum(config.probe_size[:i]), sum(config.probe_size[:i+1])
    emu = NNEmulator(config.n_dim, config.probe_size[i], 
        config.dv_lkl[l:r], config.dv_std[l:r], 
        config.inv_cov[l:r,l:r],
        mask=config.mask_lkl[l:r], param_mask=config.probe_params_mask[i], 
        model=config.nn_model, device=device,
        deproj_PCA=True, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dtype="double")
    emu_fn = pjoin(config.modeldir, f'{probes[i]}_nn{config.nn_model}')
    if (not os.path.exists(emu_fn)) or args.overwrite:
        emu.train(train_samples, train_data_vectors[:,l:r],
                valid_samples, valid_data_vectors[:,l:r],
                batch_size=config.batch_size, n_epochs=config.n_epochs, 
                loss_type=config.loss_type)
        emu.save(emu_fn)
    else:
        print(f'Emulator files exist, load from {emu_fn}')
# train sigma_8 emulator
if (config.derived==1):
    print("============= Training sigma8 Emulator =================")
    emu_s8 = NNEmulator(config.n_pars_cosmo, 1, 
        config.sigma8_fid, config.sigma8_std, 
        np.atleast_2d(1.0/config.sigma8_std**2), 
        model=config.nn_model, device=device,
        deproj_PCA=False, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dtype="double")
    emu_s8_fn = pjoin(config.modeldir, f'sigma8_nn{config.nn_model}')
    if (not os.path.exists(emu_s8_fn)) or args.overwrite:
        emu_s8.train(train_samples[:,:config.n_pars_cosmo], train_sigma8,
            valid_samples[:,:config.n_pars_cosmo], valid_sigma8,
            batch_size=config.batch_size, n_epochs=config.n_epochs,
            loss_type=config.loss_type)
        emu_s8.save(emu_s8_fn)
    else:
      print(f'Emulator files exist, load from {emu_s8_fn}')
