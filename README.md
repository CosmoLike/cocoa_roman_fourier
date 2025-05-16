# Roman Fourier 3×2pt Likelihood Analysis Package

(Placeholder now, under construction...)

# Baseline Analysis Choices

## 1. Cosmology

| Parameter | Fiducial Value | Prior | Notes |
| --- | --- | --- | --- |
| $\Omega_m$ | secret | Flat(0.1, 0.9) | - |
| $\Omega_b$ | secret | Flat(0.0x, 0.0x) | - |
| $H_0$ | secret | Flat(55, 90) |  - |
| $A_s$ | secret | Flat
| $n_s$ | secret | Flat
| $m_\nu$ | secret | Flat

Beyond $\Lambda \rm CDM$
| Parameter | Fiducial Value | Prior | Notes |
| --- | --- | --- | --- |
| $w_0$ | secret | Flat
| $w_a$ | secret | Flat

## 2. Redshift Distribution

| Parameter         | Value / Description | Value / Description |
|------------------|---------------------|---------------------|
| parametrization  | Smail type  | *Note: the Smail parameter comes from the fit (Haley please explain)* |
| expression   | $p (z) = \\left( \\frac{z}{z_0}  \\right) ^\\beta \\exp \\left[ - \\left( \\frac{z}{z_0}  \\right) ^\\alpha\\right] $ | - |
| redshift range   | $z_{\mathrm{start}} = 0$ <br> $z_{\mathrm{stop}} = 4$ <br> $z_{\mathrm{res}} = 500$ | the redshift range over which $N (z)$ is defined, including the resolutuon of the $z$ grid|


| **Lens Sample** |
|-----------------|

| Parameter | Symbol | Value | Prior | Notes |
|-----------|--------|--------|-------|-------|
| power law index 1 | $\alpha$ |        |        | the power law index of the prefactor in the Smail distribution |
| power law index 2 | $\beta$  |        |        | the power law index of the exponenet in the Smail distribution |
| pivot redshift | $z_0$    |        |        | pivot redshift in the Smail distribution |
| effective number density | $n_{\mathrm{gal}}$    |        |        | number of galaxies per $[\mathrm{arcmin}^2]$ |
| X band limiting magnitude | $m_{\mathrm{lim}}^X$    |        |        | limiting magnitude in the X band |
| number of bins | $n_{\mathrm{bin}}$ | |        | number of tomographic bins (TBD) |
| binning strategy| - | equidistant in $z$ |        | (TBD) |
| mean redshift | $\sigma_{\Delta z}$ | |        | mean redshift |
| shift parameter | $\sigma_{z}$ |        |        | the error on the mean of the tomo bin |
| stretch parameter | $\sigma_{\sigma_z}$ |        |        | the error on the std of the tomo bin |


| **Source Sample** |
|-----------------|

| Parameter | Symbol | Value | Prior | Notes |
|-----------|--------|--------|-------|-------|
| power law index 1 | $\alpha$ |        |        |  the power law index of the prefactor in the Smail distribution |
| power law index 2 | $\beta$  |        |        |  the power law index of the exponenet in the Smail distribution |
| pivot redshift | $z_0$    |        |        |  pivot redshift in the Smail distribution |
| effective number density | $n_{\mathrm{gal}}$    |        |        | number of galaxies per $[\mathrm{arcmin}^2]$ |
| X band limiting magnitude | $m_{\mathrm{lim}}^X$    |        |        | limiting magnitude in the X band |
| number of bins | $n_{\mathrm{bin}}$ | |        |  number of tomographic bins (TBD) |
| binning strategy| - | equidistant in $z$ |        |  (TBD) |
| mean redshift | $\sigma_{\Delta z}$ | |        |  mean redshift |
| shift parameter | $\sigma_{z}$ |        |        |  the error on the mean of the tomo bin |
| stretch parameter | $\sigma_{\sigma_z}$ |        |        |  the error on the std of the tomo bin |


## 3. Correlations

| Probe | Value | Description |
|------------|--------|---------------------|
| Shear | TBD | all correlations, depends on num bins|
| GGL | TBD | some correlations, depends on signal and num bins |
| Clustering | TBD | autocorrelations, depends on signal and num bins |
| 3×2pt | TBD | all together |


## Scales

| Space         | Quantity         | Unit      | Binning Strategy | Min  | Max  | Nbins | Scale Cuts (Min–Max) | Notes |
|---------------|------------------|-----------|------------------|------|------|--------|------------------------|---------------------------------------------|
| Real Space | $\theta$ | arcmin | TBD | TBD  | TBD  | TBD | TBD | Add a note if scales are cut on the cov mat level |
| Fourier Space | $\ell$ | dimensionless | TBD | TBD  | TBD | TBD | TBD | Add a note if scales are cut on the cov mat level |


## Astrophysical Systematics

### Intrinsic Alignments

| **Nonlinear Linear Alignment (NLA)** |
|-----------------|

| Parameter | Symbol | Value | Prior | Notes |
|-----------|--------|--------|-------|-------|
| amplitude | $A_{\mathrm{IA}}$ |        |        | the amplitude of the alignment |
| power law index | $\eta$ |        |        | the power law index |
| pivot redshift | $z_0$ |        |        | pivot redshift |

| **Luminosity function dependent Nonlinear Linear Alignment (LF-NLA)** |
|-----------------|

| Parameter | Symbol | Value | Prior | Notes |
|-----------|--------|--------|-------|-------|
| amplitude | $A_{\mathrm{IA}}$ |        |        | the amplitude of the alignment |
| power law index 1 | $\beta$ |        |        | tluminosity scaling factor |
| power law index 2 | $\eta_{\mathrm{low}}^z$  |        |        | tcaling of the low redshift term |
| power law index 3 | $z^{\mathrm{low}}_0$ |        |        | tscaling of the high redshift term |
| low pivot redshift | $z^{\mathrm{high}}_0$ |        |        | tpivot redshift |

| **Tidal Alignemnt Tidal Torquing (TATT)** |
|-----------------|

| Parameter | Symbol | Value | Prior | Notes |
|-----------|--------|--------|-------|-------|
| amplitude 1 | $A_1$ |        | - | the amplitude of the linear term (TA) |
| amplitude 2 | $A_2$ |        |  | the amplitude of the quadratic term (TT) |
| scaling 1 | $\eta_1$ |        |  | redshift scaling of the linear term |
| scaling| $\eta_2$ |        |  | redshift scaling of the quadratic term |
| power law index 3 | $b_{\mathrm{TA}}$ |        |  | source weighting bias (density weighting term); $A_\delta = b_{\mathrm{TA}} A_1$ |


### Galaxy Bias
We assume a constant bias per tomographic bin.
This will be filled in once we make a choice on the number of lens bins.


| **Linear Galaxy Bias** |
|-----------------|

| Parameter | Symbol | Value | Prior | Notes |
|-----------|--------|--------|-------|-------|
| parameter 1 | $b_1$ | TBA | - | - |
| parameter 2 | $b_2$ | TBA | - | - |


| **Nonlinear Galaxy Bias** |
|-----------------|

| Parameter | Symbol | Value | Prior | Notes |
|-----------|--------|--------|-------|-------|
| parameter 1 | $b_1$ | TBA | - | - |
| parameter 2 | $b_2$ | TBA | - | - |

### Baryons

| **HMCode** |
|-----------------|

| Parameter | Symbol | Value | Prior | Notes |
|-----------|--------|--------|-------|-------|
| HMCode parameter | $T_{\mathrm{AGN}}$ | TBA | - | - |


### Magnification Bias

## Calibratable Systematics

### Shear Calibration

| **Multiplicative Bias** |
|-----------------|

| Parameter | Symbol | Value | Uncertainty | Notes |
|-----------|--------|--------|-------|-------|
| multiplicative bias 1 | $m_1$ | $\sigma_{m_1} = X$ | - | - |
| multiplicative bias 2 | $m_1$ | TBA | - | - |
| multiplicative bias 3 | $m_1$ | TBA | - | - |
| multiplicative bias 4 | $m_1$ | TBA | - | - |

| **Aditive Bias** |
|-----------------|

| Parameter | Symbol | Value | Uncertainty | Notes |
|-----------|--------|--------|-------|-------|
| aditive bias 1 | $c_1$ | $\sigma_{c_1} = X$ | - | - |
| aditive bias 2 | $c_1$ | TBA | - | - |

| **Shape Noise** |
|-----------------|

| Parameter | Symbol | Value | Notes |
|-----------|--------|--------|-------|
| shape noise 1 | $\sigma_e^1$ | $0.39441348451/\sqrt{2}$ | CosmoLike only need two-component shape noise $\sigma_{e}^2 \equiv (\sigma_{e}^{1})^2+(\sigma_{e}^{2})^2$ |
| shape noise 2 | $\sigma_e^2$ | $0.39441348451/\sqrt{2}$ | - |

## Survey Specs

| Parameter | Symbol | Value| Notes |
| --- | --- | --- | --- |
| Sky fraction | $f_{\mathrm{sky}}$ |  | - |
| Depth |  |  | - |
| Area | $\Omega_s$ | 5000 | Used in the first data challenge |


# Reference
- 1. [Roman Core Community Survey Report (v3, 1/26/2025)](https://asd.gsfc.nasa.gov/roman/comm_forum/forum_17/Core_Community_Survey_Reports-rev03-compressed.pdf) for the latest definition of Roman HLIS. 

