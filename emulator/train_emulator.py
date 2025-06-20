import sys
import os
from os.path import join as pjoin
import numpy as np
import torch
from cocoa_emu import Config, get_lhs_params_list, get_params_list, CocoaModel
from cocoa_emu.emulator import NNEmulator, GPEmulator
from cocoa_emu.sampling import EmuSampler
import emcee
from argparse import ArgumentParser
from multiprocessing import Pool

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
probes = ["xi_pm", "gammat", "wtheta", "wgk", "wsk", "Ckk"]
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

'''
if (config.probe_mask[0]==1):
    print("=======================================")
    _l, _r = 0, config.probe_size[0]//2
    emu_xi_plus = NNEmulator(config.n_dim, config.probe_size[0]//2, 
        config.dv_lkl[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        mask=config.mask_lkl[_l:_r], param_mask=config.probe_params_mask[0], 
        model=config.nn_model, device=device,
        deproj_PCA=True, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dtype="double")
    emu_xi_plus_fn = pjoin(config.modeldir, f'xi_p_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_xi_plus_fn)):
        print(f'Loading existing xi_plus emulator from {emu_xi_plus_fn}....')
        emu_xi_plus.load(emu_xi_plus_fn, device=device)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'xi_p_{n-1}_nn{config.nn_model}')
            print(f'Retrain xi_plus emulator from {previous_fn}')
            emu_xi_plus.load(previous_fn, device=device)
        else:
            print("Training NEW xi_plus emulator....")
        emu_xi_plus.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            torch.Tensor(valid_samples),
            torch.Tensor(valid_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs, 
            loss_type=config.loss_type)
        if(args.save_emu):
            emu_xi_plus.save(emu_xi_plus_fn)
    print("=======================================")
    print("=======================================")
    _l, _r = config.probe_size[0]//2, config.probe_size[0]
    emu_xi_minus = NNEmulator(config.n_dim, config.probe_size[0]//2, 
        config.dv_lkl[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        mask=config.mask_lkl[_l:_r], param_mask=config.probe_params_mask[0], 
        model=config.nn_model, device=device,
        deproj_PCA=True, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dtype="double")
    emu_xi_minus_fn = pjoin(config.modeldir, f'xi_m_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_xi_minus_fn)):
        print(f'Loading existing xi_minus emulator from {emu_xi_minus_fn}....')
        emu_xi_minus.load(emu_xi_minus_fn, device=device)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'xi_m_{n-1}_nn{config.nn_model}')
            print(f'Retrain xi_minus emulator from {previous_fn}')
            emu_xi_minus.load(previous_fn, device=device)
        else:
            print("Training NEW xi_minus emulator....")
        emu_xi_minus.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            torch.Tensor(valid_samples), 
            torch.Tensor(valid_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs,
            loss_type=config.loss_type)
        if(args.save_emu):
            emu_xi_minus.save(emu_xi_minus_fn)
    print("=======================================")
if (config.probe_mask[1]==1):
    print("=======================================")
    _l, _r = sum(config.probe_size[:1]), sum(config.probe_size[:2])
    emu_gammat = NNEmulator(config.n_dim, config.probe_size[1], 
        config.dv_lkl[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        mask=config.mask_lkl[_l:_r], param_mask=config.probe_params_mask[1], 
        model=config.nn_model, device=device,
        deproj_PCA=True, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dtype="double")
    emu_gammat_fn = pjoin(config.modeldir, f'gammat_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_gammat_fn)):
        print(f'Loading existing gammat emulator from {emu_gammat_fn}....')
        emu_gammat.load(emu_gammat_fn, device=device)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'gammat_{n-1}_nn{config.nn_model}')
            print(f'Retrain gammat emulator from {previous_fn}')
            emu_gammat.load(previous_fn, device=device)
        else:
            print("Training NEW gammat emulator....")
        emu_gammat.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            torch.Tensor(valid_samples), 
            torch.Tensor(valid_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs,
            loss_type=config.loss_type
            )
        if(args.save_emu):
            emu_gammat.save(emu_gammat_fn)
    print("=======================================")
if (config.probe_mask[2]==1):
    print("=======================================")
    _l, _r = sum(config.probe_size[:2]), sum(config.probe_size[:3])
    emu_wtheta = NNEmulator(config.n_dim, config.probe_size[2], 
        config.dv_lkl[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        mask=config.mask_lkl[_l:_r], param_mask=config.probe_params_mask[2], 
        model=config.nn_model, device=device,
        deproj_PCA=True, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dtype="double")
    emu_wtheta_fn = pjoin(config.modeldir, f'wtheta_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_wtheta_fn)):
        print(f'Loading existing wtheta emulator from {emu_wtheta_fn}....')
        emu_wtheta.load(emu_wtheta_fn, device=device)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'wtheta_{n-1}_nn{config.nn_model}')
            print(f'Retrain wtheta emulator from {previous_fn}')
            emu_wtheta.load(previous_fn, device=device)
        else:
            print("Training NEW wtheta emulator....")
        emu_wtheta.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            torch.Tensor(valid_samples), 
            torch.Tensor(valid_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs,
            loss_type=config.loss_type)
        if(args.save_emu):
            emu_wtheta.save(emu_wtheta_fn)
    print("=======================================")
if (config.probe_mask[3]==1):
    print("=======================================")
    _l, _r = sum(config.probe_size[:3]), sum(config.probe_size[:4])
    emu_gk = NNEmulator(config.n_dim, config.probe_size[3], 
        config.dv_lkl[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        mask=config.mask_lkl[_l:_r], param_mask=config.probe_params_mask[3], 
        model=config.nn_model, device=device,
        deproj_PCA=True, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dtype="double")
    emu_gk_fn = pjoin(config.modeldir, f'gk_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_gk_fn)):
        print(f'Loading existing w_gk emulator from {emu_gk_fn}....')
        emu_gk.load(emu_gk_fn, device=device)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'gk_{n-1}_nn{config.nn_model}')
            print(f'Retrain w_gk emulator from {previous_fn}')
            emu_gk.load(previous_fn, device=device)
        else:
            print("Training NEW w_gk emulator....")
        emu_gk.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            torch.Tensor(valid_samples), 
            torch.Tensor(valid_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs,
            loss_type=config.loss_type)
        if(args.save_emu):
            emu_gk.save(emu_gk_fn)
    print("=======================================")
if (config.probe_mask[4]==1):
    print("=======================================")
    _l, _r = sum(config.probe_size[:4]), sum(config.probe_size[:5])
    emu_ks = NNEmulator(config.n_dim, config.probe_size[4], 
        config.dv_lkl[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        mask=config.mask_lkl[_l:_r], param_mask=config.probe_params_mask[4], 
        model=config.nn_model, device=device,
        deproj_PCA=True, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dtype="double")
    emu_ks_fn = pjoin(config.modeldir, f'ks_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_ks_fn)):
        print(f'Loading existing w_sk emulator from {emu_ks_fn}....')
        emu_ks.load(emu_ks_fn, device=device)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'ks_{n-1}_nn{config.nn_model}')
            print(f'Retrain w_sk emulator from {previous_fn}')
            emu_ks.load(previous_fn, device=device)
        else:
            print("Training NEW w_sk emulator....")
        emu_ks.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            torch.Tensor(valid_samples), 
            torch.Tensor(valid_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs,
            loss_type=config.loss_type)
        if(args.save_emu):
            emu_ks.save(emu_ks_fn)
    print("=======================================")
if (config.probe_mask[5]==1):
    print("=======================================")
    _l, _r = sum(config.probe_size[:5]), sum(config.probe_size[:6])
    emu_kk = NNEmulator(config.n_dim, config.probe_size[5], 
        config.dv_lkl[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        mask=config.mask_lkl[_l:_r], param_mask=config.probe_params_mask[5],
        model=config.nn_model, device=device,
        deproj_PCA=True, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dtype="double")
    emu_kk_fn = pjoin(config.modeldir, f'kk_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_kk_fn)):
        print(f'Loading existing CMBL band power emulator from {emu_kk_fn}....')
        emu_kk.load(emu_kk_fn, device=device)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'kk_{n-1}_nn{config.nn_model}')
            print(f'Retrain CMBL band power emulator from {previous_fn}')
            emu_kk.load(previous_fn, device=device)
        else:
            print("Training NEW CMBL band power emulator....")
        emu_kk.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            torch.Tensor(valid_samples), 
            torch.Tensor(valid_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs,
            loss_type=config.loss_type)
        if(args.save_emu):
            emu_kk.save(emu_kk_fn)
    print("=======================================")
if (config.derived==1):
    print("=======================================")
    emu_s8 = NNEmulator(config.n_pars_cosmo, 1, 
        config.sigma8_fid, config.sigma8_std, 
        np.atleast_2d(1.0/config.sigma8_std**2), 
        model=config.nn_model, device=device,
        deproj_PCA=False, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dtype="double")
    emu_s8_fn = pjoin(config.modeldir, f'sigma8_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_s8_fn)):
        print(f'Loading existing derived parameters emulator (sigma8) from {emu_s8_fn}....')
        emu_s8.load(emu_s8_fn, device=device)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'sigma8_{n-1}_nn{config.nn_model}')
            print(f'Retrain derived parameters emulator from {previous_fn}')
            emu_s8.load(previous_fn, device=device)
        else:
            print("Training NEW derived parameters emulator....")
        emu_s8.train(torch.Tensor(train_samples[:,:config.n_pars_cosmo]), 
            torch.Tensor(train_sigma8),
            torch.Tensor(valid_samples[:,:config.n_pars_cosmo]), 
            torch.Tensor(valid_sigma8),
            batch_size=config.batch_size, n_epochs=config.n_epochs,
            loss_type=config.loss_type)
        if(args.save_emu):
            emu_s8.save(emu_s8_fn)
    print("=======================================")
'''