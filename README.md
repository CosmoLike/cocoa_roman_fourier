# Roman Fourier 3×2pt Likelihood Analysis Package

(Placeholder now, under construction...)

## Baseline Analysis Choices

### 1. Cosmology

| Parameter | Fiducial Value | Prior | Notes |
| --- | --- | --- | --- |
| $\Omega_m$ | secret | Flat(0.1, 0.9) | - |
| $\Omega_b$ | secret | Flat(0.0x, 0.0x) | - |
| $H_0$ | secret | Flat(55, 90) |  - |
| $A_s$ | secret | Flat
| $n_s$ | secret | Flat
| $m_\nu$ | secret | Flat

### 2. Redshift Distribution

| Parameter         | Value / Description |
|------------------|---------------------|
| parametrization  | Smail type *Note: the Smail parameter comes from the fit (Haley please explain)* |
| expression   | $p (z) = \\left( \\frac{z}{z_0}  \\right) ^\\beta \\exp \\left[ - \\left( \\frac{z}{z_0}  \\right) ^\\alpha\\right] $ |
| redshift range   | - start: 0 <br> - stop: 4 <br> - resolution: 500 |


| **Lens Sample** |
|-----------------|

| Parameter | Symbol | Value | Notes |
|-----------|--------|--------|-------|
| power law index 1 | $\alpha$ |        | the power law index of the prefactor in the Smail distribution |
| power law index 2 | $\beta$  |        | the power law index of the exponenet in the Smail distribution |
| pivot redshift | $z_0$    |        | pivot redshift in the Smail distribution |
| number of bins | $n_{\mathrm{bin}}$ | | number of tomographic bins (TBD) |
| binning strategy| - | equidistant in $z$ | (TBD) |
| mean redshift | $\sigma_{\Delta z}$ | | mean redshift |
| shift parameter | $\sigma_{z}$ |        | the error on the mean of the tomo bin |
| stretch parameter | $\sigma_{\sigma_z}$ |        | the error on the std of the tomo bin |

| **Source Sample** |
|-----------------|

| Parameter | Symbol | Value | Notes |
|-----------|--------|--------|-------|
| power law index 1 | $\alpha$ |        | the power law index of the prefactor in the Smail distribution |
| power law index 2 | $\beta$  |        | the power law index of the exponenet in the Smail distribution |
| pivot redshift | $z_0$    |        | pivot redshift in the Smail distribution |
| number of bins | $n_{\mathrm{bin}}$ | | number of tomographic bins (TBD) |
| binning strategy| - | equidistant in $z$ | (TBD) |
| mean redshift | $\sigma_{\Delta z}$ | | mean redshift |
| shift parameter | $\sigma_{z}$ |        | the error on the mean of the tomo bin |
| stretch parameter | $\sigma_{\sigma_z}$ |        | the error on the std of the tomo bin |


### 3. Correlations

| Probe | Value | Description |
|------------|--------|---------------------|
| Shear | TBD | all correlations, depends on num bins|
| GGL | TBD | some correlations, depends on signal and num bins |
| Clustering | TBD | autocorrelations, depends on signal and num bins |
| 3×2pt | TBD | all together |


### Scales

| Space         | Quantity         | Unit      | Binning Strategy | Min  | Max  | Nbins | Scale Cuts (Min–Max) | Notes |
|---------------|------------------|-----------|------------------|------|------|--------|------------------------|---------------------------------------------|
| Real Space | $\theta$ | arcmin | TBD | TBD  | TBD  | TBD | TBD | Add a note if scales are cut on the cov mat level |
| Fourier Space | $\ell$ | dimensionless | TBD | TBD  | TBD | TBD | TBD | Add a note if scales are cut on the cov mat level |


### Systematics

- IA
- galaxy bias
- baryon
- magnification bias
- calibration: shear calibration, nz, multiplicative bias, additive bias
- shape noise

