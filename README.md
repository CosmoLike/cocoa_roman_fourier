# Roman Fourier 3x2pt Likelihood Analysis Package

(Placeholder now, under construction...)

## Baseline Analysis Choices

### 1. Cosmology

| Parameter | Fiducial Value | Prior | Notes |
| --- | --- | --- | --- |
| $\Omega_m$ | secret | Flat(0.1, 0.9) | - |
| $\Omega_b$ | secret | Flat(0.0x, 0.0x) | - |
| $H_0$ | secret | Flat(55, 90) |  - |

### 2. Redshift Distribution

Parametrization

Number of bins: 8

Mean, std

Binning strategy

### 3. Correlations

- shear:
- ggl:
- clustering:

### 4. Emulator

We provide an emulator that predicts the 3x2pt model vector in LCDM cosmology assuming that the lens and source samples 
are the same galaxy sample. The underlying assumptions of the emulator are: 
- Galaxy bias: linear galaxy bias
- Baryonic feedback: gravity-only
- Intrinsic alignment: nonlinear alignment
- Photometric redshift: shift of the mean redshift of each tomography bin, with the redshift distribution file mentioned above
- Nonlinear matter power spectrum: CAMB/Halofit-Takahashi
- Neutrino mass modeling: 
- Magnification bias: no
- Redshift-space distortion in galaxy clustering:
- Non-Limber integration in galaxy clustering: 

Emulator dataset download link: [the box link](https://arizona.box.com/s/zzqkre6wip71qvtoipajqesy0ypejwpv)

After download the emulator dataset, please put the directory to the dataset in the emulator configuration yaml file 
`emulator/train_emu_resmlp512_7_HF_NLA_minimal.yaml` under the `emulator/io/savedir` option

To run the emulator, see 
- EXAMPLE_EMUEVAL1.yaml
