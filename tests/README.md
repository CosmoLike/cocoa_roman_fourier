# Unit tests for the likelihoods

These tests catch two kinds of silent breakage: a $\chi^2$ that drifted
because code or data changed by accident, and a race condition (a bug
where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads).

Every model build runs in its own worker subprocess. In this project
both examples share one data set, so the isolation is preventive: it
keeps the tests immune to the process abort that different data-set
dimensions trigger inside cosmolike (roman_fourier has that layout), and
every project keeps one architecture. The commands below stay the
same.

# Table of contents

1. [Running the tests](#run_tests)
2. [The tests](#the_tests)
    1. [The CFASTPT vs FASTPT comparison](#cfastpt_fastpt)
    2. [The Halofit vs EE2 checks](#halofit_ee2)
    3. [The EE2 race test](#ee2_race)
    4. [Accuracy checks](#accuracy_checks)
    5. [Baryonic feedback accuracy checks](#baryon_accuracy_checks)
    6. [Baryonic feedback drift tests](#baryon_drift_tests)
3. [Appendix](#appendix)
    1. [FAQ: Do the tests keep their own data?](#frozen_copy)
    2. [FAQ: Why do the TATT tests use their own data vector?](#synthetic_vectors)
    3. [FAQ: How can maintainers refresh the snapshot?](#refreeze)

## Running the tests <a name="run_tests"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the tests of this project

    python -m pytest ./projects/roman_fourier/tests

Without pytest:

    python -m unittest discover -s ./projects/roman_fourier/tests -v

The tests change no project files. Each test prints a progress line
per model build and per evaluation, then a report with the
computed $\chi^2$, the stored reference, the difference, and the pass
limit.

A full run performs about 50 likelihood evaluations and takes a
few minutes. The test files force `OMP_NUM_THREADS=4` internally.

## The tests <a name="the_tests"></a>

The standard configurations get four tests each: a $\chi^2$ drift check
and a race-condition check, both in the NLA and in the TATT intrinsic-alignment
model. The TATT variants set

    IA_model: 1
    roman_A2_1: 0.05
    roman_BTA_1: 0.05
    roman_A2_2: -1.51541

The two checks and their pass limits:

| check | pass limit                                        | a failure means                    |
|-------|---------------------------------------------------|------------------------------------|
| $\Delta\chi^2$ | the recomputed $\chi^2$ must stay within 0.2 of the value stored in `frozen/reference_chi2.json` | code or data changed the numbers |
| race condition | the fiducial evaluated on its own vs evaluated again after nine other cosmologies; the two must agree within $10^{-4}$ | leftover state or an OpenMP race |

Everything the tests compare against lives under `frozen/`: one
snapshot of configurations, data, and reference values, captured
together when the references were generated and unchanged since. The
[Appendix](#appendix) explains how the snapshot is protected.

The test files and the configurations they cover:

| test | file | configuration | what it checks |
|---|---|---|---|
| 1 | `test_example1.py` | cosmic shear; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 2 | `test_example1.py` | cosmic shear; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 3 | `test_example1.py` | cosmic shear; IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 4 | `test_example1.py` | cosmic shear; IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 5 | `test_example2.py` | 3x2pt; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 6 | `test_example2.py` | 3x2pt; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 7 | `test_example2.py` | 3x2pt; IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 8 | `test_example2.py` | 3x2pt; IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 11 | `test_example2_2x2pt.py` | 2x2pt (`roman_fourier.combo_2x2pt`: the 3x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing); IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 12 | `test_example2_2x2pt.py` | 2x2pt (`roman_fourier.combo_2x2pt`: the 3x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing); IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 13 | `test_example2_2x2pt.py` | 2x2pt (`roman_fourier.combo_2x2pt`: the 3x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing); IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 14 | `test_example2_2x2pt.py` | 2x2pt (`roman_fourier.combo_2x2pt`: the 3x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing); IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 15 | `test_fastpt.py` | cosmic shear; IA modeling: TATT; the C cfastpt (`IA_code: 0`) vs the python FAST-PT package (`IA_code: 1`) at 30 fixed points (20 across the intrinsic-alignment prior plus a one-parameter-at-a-time family), cosmology at the fiducial | $\Delta\chi^2$ of the FAST-PT data vector against the cfastpt data vector at the same point; the cfastpt vector is that point's fiducial, so agreement means zero |
| 16 | `test_fastpt.py` | 3x2pt; IA modeling: TATT; the same comparison as test 15 on the 3x2pt likelihood (`roman_fourier.combo_3x2pt`) | the same pass rule as test 15, with the data-vector difference weighted by the 3x2pt masked inverse covariance |
| 17 | `test_fastpt.py` | 2x2pt; IA modeling: TATT; the same comparison as test 15 on the 2x2pt likelihood (`roman_fourier.combo_2x2pt`) | the same pass rule as test 15; clustering carries no intrinsic alignment, so the TATT tables enter through galaxy-galaxy lensing alone, weighted by the 2x2pt masked inverse covariance |
| 18 | `test_ee2.py` | cosmic shear; NLA with EE2 (`non_linear_emul: 1`) | race condition (OpenMP threading, including EE2's own threaded compute): fiducial alone vs after nine other cosmologies |

### The CFASTPT vs FASTPT comparison (`test_fastpt.py`, tests 15-17) <a name="cfastpt_fastpt"></a>

Cosmolike computes the TATT perturbation-theory integrals with two
implementations: cfastpt, the C code built into the interface
(`IA_code: 0`), and the python FAST-PT package through the fastpt
theory block (`IA_code: 1`). Both are evaluated at 30 fixed points
across the intrinsic-alignment prior and checked for agreement:
test 15 on cosmic shear, test 16 on the 3x2pt likelihood, test 17
on the 2x2pt likelihood.

At every point the cfastpt data vector is the fiducial: the reported
quantity is the $\Delta\chi^2$ of the FAST-PT vector against it,
zero for identical vectors and quadratic in their difference. A
comparison against the shipped data vector would measure the slope
of the distance to the data instead of the numerics.

The fastpt block computes on two grids: `accuracyboost` multiplies
the density of the output table cosmolike reads with linear
interpolation (the accuracy driver), and `internal_accuracyboost`
the density of the internal grid the FFTLog convolutions run on,
with a cubic spline in log k upsampling the terms from one grid
onto the other.

Both boosts are rebased so 1.0 is the converged configuration; this
project's precision needs `accuracyboost: 2`, the doubled output
density. The tests assert there with the 0.2 band of the other
checks as the pass limit, doubling the configuration again as an
advisory.

In test 16 the TATT terms also enter galaxy-galaxy lensing, and the
difference is weighted by the 3x2pt masked inverse covariance. The
frozen configuration fixes the one-loop bias amplitudes
(`roman_B2_*`) at zero, so the one-loop galaxy-bias tables both
implementations compute multiply by zero: the sweep scores the
intrinsic-alignment tables alone, on the wider data vector.

Test 17 repeats the sweep on the 2x2pt likelihood. Clustering
carries no intrinsic alignment, so there the TATT tables are scored
through galaxy-galaxy lensing alone, under the 2x2pt masked
inverse covariance.

> [!NOTE]
> Before the two-grid upgrade of the fastpt theory block (2026-09)
> there was no upsampling and the difference reached
> $\Delta\chi^2 = 87184$ across the prior.

The point values, the design, and the decision record live with the
lsst_y1 project (its tests/README.md carries the full discussion);
the table below is this project's own measurement:

| output table (points) | internal grid (points) | max $\Delta\chi^2$ | cost per cosmology |
|---|---|---|---|
| 1,100 | 1,100 (shared) | 87184 | 1.5 s |
| 1,024,900 (`accuracyboost: 1`) | 1,100 | 0.209 | 1.9 s |
| 2,048,900 (`accuracyboost: 2`, the pass configuration) | 1,100 | 0.1120 | 2.3 s |
| 4,096,900 (`accuracyboost: 4`) | 1,300 (`internal_accuracyboost: 2`) | 0.0828 | 3.0 s |
| 8,192,900 | 1,100 | 0.073 | 4.2 s |

![The 30 comparison points, colored by the per-point difference](cfastpt_vs_fastpt_points.png)

> [!NOTE]
> This project's precision needs `accuracyboost: 2`, the value its
> example yamls recommend and the tests assert on; the defaults
> land at 0.209, just above the band. cfastpt (`IA_code: 0`)
> remains the reference implementation.

Measured on 2026-09-23:

- Test 16 (3x2pt): max $\Delta\chi^2 = 0.1475$ at the defaults and
  $0.1092$ at the pushed camb/cosmolike settings, above cosmic
  shear's 0.1120 under the 3x2pt masked covariance.
- Test 17 (2x2pt): $0.0021$ and $0.0013$, with the TATT tables
  entering through galaxy-galaxy lensing alone.
- `--mask=ones` (no scale cuts, all 1,485 points weighted): cosmic
  shear repeats its frozen-mask values exactly - the contract mask
  keeps every shear point, so the two masks coincide there - and
  measures 0.0903 at the pushed camb/cosmolike settings; the 2x2pt
  sweep measures max $\Delta\chi^2 = 0.0039$ at the defaults and
  0.0025 pushed.
- Before 2026-09-23 `ones.mask` (in `data/` and pinned under
  `frozen/data/`) was a stray byte-copy of lsst_y1's 1,560-row
  mask, which cosmolike rejects against this project's 1,485-point
  data vector; it was replaced by the correct 1,485-row all-ones
  mask.

> [!Warning]
> `--mask=ones` with the 3x2pt likelihood (test 16, 2026-09-23)
> fails at model build: with every data point kept the shipped
> covariance is not positive definite, and cosmolike aborts
> (`IP::set_inv_cov`). The mask runs with tests 15 and 17.

#### Running the comparison <a name="run_cfastpt_fastpt"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the comparison at the default camb/cosmolike
settings

    python -m pytest ./projects/roman_fourier/tests/test_fastpt.py

**Step :three:**: repeat it at the pushed camb/cosmolike settings

    python -m pytest ./projects/roman_fourier/tests/test_fastpt.py --high=1

> [!NOTE]
> `--high=1`: applies the pushed camb/cosmolike settings of the
> accuracy checks to every block of tests 15-17 (the other tests do
> not read it). The full comparison is both invocations.

> [!NOTE]
> `--mask`: reruns tests 15-17 under a different scale-cut mask.
> `--mask=frozen` (the default) keeps the contract mask
> (`roman_example.mask`, 1,359 of the 1,485 data points);
> `--mask=ones` keeps every point (no scale cuts), the strictest
> comparison. Each choice is a frozen TATT dataset variant
> differing only in its `mask_file` line, and the 0.2 pass rule
> applies unchanged. The 3x2pt sweep does not run under `ones`
> (see the Warning above).

### The Halofit vs EE2 checks (`test_nonlinear.py`, NL1-NL2) <a name="halofit_ee2"></a>

The likelihoods can source the nonlinear matter power from CAMB's
Takahashi halofit (`non_linear_emul: 2`, the frozen contract's
setting) or from EuclidEmulator2 (`non_linear_emul: 1`). NL1
evaluates the cosmic-shear data vector and NL2 the 3x2pt data
vector with both at ten fixed cosmologies across the omegam/ns/As
space (every other parameter at the frozen fiducial) and reports,
per cosmology, the $\Delta\chi^2$ of the Halofit vector against
the EE2 vector.

The EE2 vector is that cosmology's fiducial, so the baseline is
zero by construction and no stored data vector enters the metric.
The checks are advisory - there is no pass limit: the numbers say
how much of the statistical error budget the Halofit-vs-emulator
difference consumes under the chosen scale cuts, the question "can
Halofit be used on real data analysis at this mask". The `--mask`
option of the comparison sweeps applies.

Measured on 2026-09-23 (the figure below, frozen mask):

- NL1 (cosmic shear): per-cosmology $\Delta\chi^2$ between 29.1
  and 864.7 (median 99.9), largest at the high-omegam draws; under
  `--mask=ones` it repeats these values digit for digit - the
  contract mask keeps every shear point, so the two masks coincide
  there.
- NL2 (3x2pt): between 358.4 and 3876.0 (median 879.8), largest at
  the high-omegam draws; it does not run under `ones` - with every
  data point kept the shipped covariance is not positive definite
  and cosmolike aborts at model build (`IP::set_inv_cov`, the
  Warning above).

![The ten cosmologies, colored by the Halofit-vs-EE2 difference](halofit_vs_ee2_points.png)

#### Running the Halofit vs EE2 checks <a name="run_halofit_ee2"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the checks

    python -m pytest ./projects/roman_fourier/tests/test_nonlinear.py

`--mask=ones` reruns them with every data point kept; NL2 aborts
there (the Warning above).

### The EE2 race test (`test_ee2.py`, test 18) <a name="ee2_race"></a>

Cocoa pins a modified EuclidEmulator2: OpenMP threading, a
1,010-redshift capacity, the `get_boost2` API with a pre-built
emulator, memory-leak fixes, and a bilinear interpolation with a
border fix. The modifications and their measured speed-up are
documented in the repository's own README
(`external_modules/code/euclidemu2/README.md`).

The test evaluates the fiducial fresh and again as the 10th of 10
cosmologies on one model instance, with the nonlinear $P(k)$ from
EE2 (`non_linear_emul: 1`) instead of the frozen contract's halofit.
EE2's compute is OpenMP-threaded, so a thread race inside it shifts
the second fiducial value; the two must agree within $10^{-4}$.

The modification gate, which compiles the pre-modification build
(commit `ff59f66`) at test time and scores the installed build
against it, runs as the lsst_y1 project's test 18 (its
`tests/test_ee2.py`); the emulator build is project-independent, so
that comparison is not repeated here.

Measured on 2026-09-23:

- Test 18: the fresh and 10th-in-a-row fiducial agree to all eight
  printed decimals.
- The modified-vs-original comparison of the lsst_y1 gate, run on
  this project's cosmic shear under its frozen mask (ten
  cosmologies), measured max $\Delta\chi^2 = 4.3\times10^{-4}$ (max
  fractional data-vector difference $3.5\times10^{-5}$): the largest
  of the six projects, still far below the 0.2 band.

### Accuracy checks (`test_accuracy.py`, A1-A6) <a name="accuracy_checks"></a>

Checks A1-A6 re-evaluate cosmic shear, 3x2pt, and 2x2pt, with NLA
and TATT, with every setting pushed far beyond the defaults at
once. Before them, we change one accuracy parameter at a time on
the 3x2pt NLA configuration (the `KNOB` lines of the test output),
so a large $\Delta\chi^2$ can be attributed to the parameter
causing it; the measured values are in the project README under
"Minimum accuracy parameters". The settings:

| setting | raised to | what it controls |
|---------|-----------|------------------|
| `accuracyboost` (cosmolike) | 2 | sizes of cosmolike's internal lookup tables, including the dyadic z grid of the power-spectrum tables |
| `integration_accuracy` (cosmolike) | 10 | extra refinement passes of cosmolike's numerical integrals |
| `kmax_boltzmann` (cosmolike) | 40 | the k cutoff of the power spectrum the likelihood requests from CAMB |
| `AccuracyBoost` (CAMB) | 2 | CAMB's overall accuracy multiplier: denser sampling in every internal CAMB grid, the most expensive setting |
| `k_per_logint` (CAMB) | 50 | k samples CAMB computes per logarithmic interval of the transfer functions |
| `kmax` (CAMB) | 50 | highest k of CAMB's matter power spectrum; one physical cutoff with `kmax_boltzmann`, seen from the CAMB side |

There is no `lmax` entry here: the ell range lives in the dataset.

`accuracyboost` refines a nested z grid in the power-spectrum
tables: every coarser grid's nodes are a subset of every finer
grid's, so a higher boost tightens the same interpolation instead of
moving the nodes (the construction is commented in
`likelihood/_cosmolike_prototype_base.py`).

When several settings move the $\chi^2$, settle them in cost order:
raise cosmolike `accuracyboost` first (cheap), then CAMB
`k_per_logint`, and CAMB `AccuracyBoost` last (expensive at run
time, and able to masquerade for the cheap settings).
`kmax_boltzmann` and CAMB `kmax` are one physical cutoff seen from
two sides; move them together.

Each check reports the $\Delta\chi^2$ between the high-accuracy and
the default evaluations: the numerical error of the default
settings. No pass/fail.

> [!NOTE]
> High-accuracy evaluations take minutes.

#### Running Accuracy checks <a name="run_accuracy"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the accuracy checks on their own

    python -m pytest ./projects/roman_fourier/tests/test_accuracy.py

To run every other test while skipping these:

    python -m pytest ./projects/roman_fourier/tests --ignore ./projects/roman_fourier/tests/test_accuracy.py

### Baryonic feedback accuracy checks (`test_accuracy_baryons.py`, BF1-BF7) <a name="baryon_accuracy_checks"></a>

The file `test_accuracy_baryons.py` repeats the default-versus-high
accuracy comparison with the `bfmt` theory block switched on: one
advisory check per feedback method (the three SP(k) fb relations,
BCEmu, Flamingo, BACCOemu, and BCemu2025), at a fixed parameter
point per method.

Each check creates its data vector on the fly, by the same mechanism
as the N-random-models check:

1. The default-settings model writes its own theory vector during
   evaluation.
2. That vector becomes the data of a temporary dataset.
3. The pushed-settings model evaluates at the same point against it.

The fiducial $\chi^2$ is zero by construction and nothing is stored
in the snapshot, so the single reported number, $\Delta\chi^2$, is a
pure numerics response. The check BF0 additionally runs the
one-setting-at-a-time scan with the Akino SP(k) feedback on, so a
large delta names the setting causing it.

Every checked configuration is measurable by construction. The
BACCOemu check evaluates with `omegab: 0.049`, inside that
emulator's baryon-density training box, whose floor sits exactly
above the fiducial `omegab: 0.04`; and the double-power-law point is
chosen to keep the baryon fraction inside SP(k)'s calibrated band
over the full redshift grid.

#### Running the baryonic feedback checks <a name="run_baryon_accuracy"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the baryonic feedback checks of this project

    python -m pytest ./projects/roman_fourier/tests/test_accuracy_baryons.py

### Baryonic feedback drift tests (`test_baryons.py`, BD1-BD7) <a name="baryon_drift_tests"></a>

The file `test_baryons.py` pins the feedback pipeline against change
over time, one test per method.

Each method's default-settings theory prediction was stored at
freeze time (`generate_frozen_reference.py --baryons`), and the test
evaluates today's prediction against that stored vector: zero at
freeze time by construction, so a $\chi^2$ above the tolerance means
cosmolike or the `bfmt` theory block changed its prediction since
the freeze.

These tests complement the accuracy checks above: the accuracy
checks regenerate their vector on the fly per run, so they measure
the numerical settings and can never see drift; the drift tests hold
the frozen vector still, so they measure drift and nothing else.

#### Running the baryonic feedback drift tests <a name="run_baryon_drift"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the drift tests of this project

    python -m pytest ./projects/roman_fourier/tests/test_baryons.py

# Appendix <a name="appendix"></a>

## :interrobang: FAQ: Do the tests keep their own data? <a name="frozen_copy"></a>

The tests read nothing from the live project: not `../data`, not the
`EXAMPLE_EVALUATE` yaml files, and not the likelihood default yaml
files. Instead, `frozen/` holds:

| `frozen/` entry | holds |
|---|---|
| `frozen_config_example{1,2}.py` | the complete cobaya configuration as a yaml string, plus the exact evaluation point |
| `data/` | the tests' own copy of the data vectors, covariance, n(z), and masks |
| `EXAMPLE_EVALUATE{1,2}.yaml` | snapshots kept only so a human can diff how the live examples drifted since the freeze |

In the configuration modules every option and every parameter is
written out, including the ones that normally come from
`params_source.yaml` and the other default files, so editing those
files cannot change what the tests evaluate.


`manifest_sha256.json` stores a SHA-256 hash (a fingerprint that
changes when any byte changes) of every file under `frozen/`. Each test
verifies the manifest first and refuses to run when a file under `frozen/` was
edited, naming the file. The result: users may change the live data
and examples freely, and nobody can quietly edit the snapshot
either.

## :interrobang: FAQ: Why do the TATT tests use their own data vector? <a name="synthetic_vectors"></a>

All TATT variants evaluate against `frozen/data/tatt_roman_fourier.dataset`,
a data vector generated with TATT at the fiducial point when the
snapshot was created: at its own minimum the TATT $\chi^2$ responds quadratically to
numerical changes instead of linearly on the side of a hill.

## :interrobang: FAQ: How can maintainers refresh the snapshot? <a name="refreeze"></a>

A deliberate change to the data vectors, n(z), covariance, examples,
or likelihood defaults requires a re-freeze.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: rebuild the snapshot

    python ./projects/roman_fourier/tests/generate_frozen_reference.py --overwrite

It rebuilds `frozen/` from the current project, prints the four new
reference $\chi^2$ values, and rewrites the manifest. Review the printed
$\chi^2$ values against the old references before committing: they define
what every later test run compares against.
