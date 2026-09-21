## Running Cosmolike projects (Basic instructions) <a name="running_cosmolike_projects"></a> 

From `Cocoa/Readme` instructions:

> [!Note]
> We provide several cosmolike projects that can be loaded and compiled using `setup_cocoa.sh` and `compile_cocoa.sh` scripts. To activate them, comment the following lines on `set_installation_options.sh` 
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     (...)
>
>     # ------------------------------------------------------------------------------
>     # The keys below control which cosmolike projects will be installed and compiled
>     # ------------------------------------------------------------------------------
>     #export IGNORE_COSMOLIKE_LSST_Y1_CODE=1
>     #export IGNORE_COSMOLIKE_DES_Y3_CODE=1
>     export IGNORE_COSMOLIKE_ROMAN_FOURIER_CODE=1
>     #export IGNORE_COSMOLIKE_ROMAN_REAL_CODE=1
>
>     (...)
>     # ------------------------------------------------------------------------------
>     # Cosmolike projects below -------------------------------------------
>     # ------------------------------------------------------------------------------
>     (...)
>     export ROMAN_FOURIER_URL="https://github.com/CosmoLike/cocoa_roman_fourier.git"
>     export ROMAN_FOURIER_NAME="roman_fourier"
>     #Pin the project version with at most one of the keys below (COMMIT, BRANCH, or TAG).
>     #If more than one is set, COMMIT wins over BRANCH, and BRANCH wins over TAG.
>     #If none is set, Cocoa loads the latest commit on the repository default branch.
>     #export ROMAN_FOURIER_GIT_BRANCH="main"
>     #export ROMAN_FOURIER_GIT_COMMIT="abc"
>     export ROMAN_FOURIER_GIT_TAG="v4.10.5"

> [!NOTE]
> If users want to recompile cosmolike, there is no need to rerun the Cocoa general scripts. Instead, run the following three commands:
>
>      source start_cocoa.sh
>
> and
> 
>      source ./installation_scripts/setup_cosmolike_projects.sh
>
> and
> 
>       source ./installation_scripts/compile_all_projects.sh
> 
> or (in case users just want to compile roman_fourier project)
>
>       source ./projects/roman_fourier/scripts/compile_roman_fourier.sh

> [!TIP]
> Assuming Cocoa is installed on a local (not remote!) machine, type the command below after step 2️⃣ to run Jupyter Notebooks.
>
>     jupyter notebook --no-browser --port=8888
>
> The project roman_fourier contains jupyter notebook examples located at `projects/roman_fourier/notebooks`.

To run the example

 **Step :one:**: activate the cocoa Conda environment,  and the private Python environment 

      conda activate cocoa

and

      source start_cocoa.sh
 
 **Step :two:**: Select the number of OpenMP cores (below, we set it to 8).

  - Linux
    
        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)
    
        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

 **Step :three:**: The folder `projects/roman_fourier` contains examples. So, run the `cobaya-run` on the first example following the commands below.

> [!Warning] 
> (Linux only) In some HPC nodes, `numa` can cause you problems. If that is the case,
> replace `numa` with `slot`

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/roman_fourier/EXAMPLE_EVALUATE1.yaml -f

  - macOS (arm)

        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/roman_fourier/EXAMPLE_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/roman_fourier/EXAMPLE_MCMC2.yaml -f

  - macOS (arm)

        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/roman_fourier/EXAMPLE_MCMC2.yaml -f

# Baryonic feedback on EXAMPLE_EVALUATE1 <a name="roman_fourier_baryonic_feedback"></a>

`EXAMPLE_EVALUATE1.yaml` can apply an external baryonic feedback suppression to the
matter power spectrum via the `bfmt` theory block (SP(k), BCEmu, Flamingo, BACCOemu,
or BCemu2025). By default, the example runs without feedback.

**Step :one:**: ensure the lines below are commented out in `set_installation_options.sh`
before running `setup_cocoa.sh` and `compile_cocoa.sh`. *By default, these lines should
be commented out, but it is worth checking*.

      [Adapted from Cocoa/set_installation_options.sh shell script]
      #export IGNORE_PYSPK_CODE=1     # SP(k)
      #export IGNORE_BCEMU_CODE=1     # BCEmu
      #export IGNORE_FBRE_CODE=1      # FlamingoBaryonResponseEmulator
      #export IGNORE_BACCOEMU_CODE=1  # BACCOemu
      #export IGNORE_BFMT_CODE=1      # Baryon Feedback Theory Block

**Step :two:**: in `EXAMPLE_EVALUATE1.yaml`, uncomment the `bfmt` theory block and select
the model:

      theory:
        bfmt:
          baryon_model: 2 # 1 = SP(k), 2 = BCEmu, 3 = FlamingoEmulator, 4 = BACCOemu, 5 = BCemu2025

**Step :three:**: set `external_baryon_suppression: True` on the `roman_fourier.cosmic_shear`
likelihood block.

**Step :four:**: uncomment the selected model's parameters in the `params` block and in
the `sampler: evaluate: override` block (the example carries a commented block for each
model).

> [!TIP]
> For the sampled parameters of each model, their validity ranges, and the `bfmt`
> options, see `Cocoa/external_modules/code/baryon_suppression/README.md`.

# Running Hybrid Cosmolike-ML emulators <a name="roman_fourier_examples_emul2"></a>

> [!Warning]
> The code and examples associated with this section are still in alpha stage

While our data vector emulators are incredibly fast, there is an intermediate 
approach that emulates only the Boltzmann outputs (comoving distance, linear and 
nonlinear matter power spectrum). This hybrid-ML case can offer greater flexibility, 
especially in the initial phases of a research project, as changes to the modeling 
of nuisance parameters or to the assumed galaxy distributions do not require 
retraining of the network. 

Examples in the hybrid case all have the prefix **EXAMPLE_EMUL2** (note the `2`). The required flags on `set_installation_options.sh` are similar to what we showed in the previous emulator section.

Now, users must follow all the steps below.

 **Step :one:**: Activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

 **Step :two:**: Select the number of OpenMP cores. Below, we set it to 4, the ideal setting for hybrid examples.

  - Linux

        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)
    
        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

 **Step :three:** Run `cobaya-run` on the first emulator example, following the commands below.

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/roman_fourier/EXAMPLE_EMUL2_EVALUATE1.yaml -f

  - macOS (arm)
    
        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/roman_fourier/EXAMPLE_EMUL2_EVALUATE1.yaml -f

> [!NOTE]
> **Running on more than one node.** The flag `--mca btl vader,tcp,self` works unchanged across
> nodes: Open MPI picks the transport per pair of ranks, using shared memory (`vader`) within a
> node and TCP between nodes. Three things deserve attention on multi-node runs:
>
> 1. **Network interface.** The TCP layer must not select an interface that is not routable
>    between compute nodes. The flag `--mca btl_tcp_if_exclude lo,docker0,virbr0,ib0` excludes
>    the common offenders. TCP bandwidth is not a limitation for our workloads, which exchange
>    small, infrequent MPI messages.
>
> 2. **Environment.** Ranks on remote nodes must see Cocoa's environment (`ROOTDIR`, `PATH`,
>    `LD_LIBRARY_PATH`, `PYTHONPATH`, `CONDA_PREFIX`, the OpenMP/BLAS thread settings, and
>    `CLIK_PATH`/`CLIK_DATA`/`CLIK_PLUGIN`). Slurm forwards the submitting environment
>    automatically; the explicit `-x` flags in our sbatch templates repeat this so the
>    scripts also work under ssh-based launchers. No other Cocoa installation flags are read at runtime.
>
> 3. **Slurm geometry.** Keep `ntasks-per-node` × `cpus-per-task` no larger than the cores per
>    node, and use `--map-by numa:pe=${OMP_NUM_THREADS}` so each rank reserves the cores its
>    OpenMP threads will use.

> [!NOTE]
> **Note on core oversubscription**: an MPI process that is waiting still burns 100% of its
> core, checking for messages in a loop. With more processes than cores, this stalls the
> processes doing real work. Open MPI usually detects this and makes waiting processes give
> up the CPU, but its detection can be fooled. Adding `--mca mpi_yield_when_idle 1` forces
> that behavior; it is harmless otherwise.

## Unit tests <a name="roman_fourier_unit_tests"></a>

The folder `projects/roman_fourier/tests` contains 12 pass/fail tests covering
the three likelihoods (cosmic shear, 3x2pt, 2x2pt) with both intrinsic-alignment
models (NLA and TATT). For each combination, one test compares the chi2 at a
fixed reference point against the value stored in
`tests/frozen/reference_chi2.json` (pass limit 0.2), and one evaluates that
point fresh and again as the 10th of 10 cosmologies in a row: leftover internal
state or an OpenMP race breaks the agreement (limit 1e-4). Every model builds
in its own worker subprocess, and every frozen file is checked against a
SHA-256 manifest before any physics runs, so an edited frozen state fails
loudly instead of producing a plausible chi2.

Run the suite from the `Cocoa/` folder, with the cocoa conda environment
active and `start_cocoa.sh` sourced:

    python -m pytest ./projects/roman_fourier/tests

`projects/roman_fourier/tests/README.md` describes each test, the frozen
state, and how maintainers regenerate it.

## Minimum accuracy parameters <a name="roman_fourier_accuracy"></a>

`tests/test_accuracy.py` measures the numerical error the default settings
carry. It re-evaluates the frozen configurations with the accuracy knobs
raised, one at a time and all at once, and reports
delta chi2 = chi2(raised) - chi2(default). The target is |delta chi2| below
0.2, the pass limit of the reference tests. On the frozen example2 3x2pt
point (NLA, chi2 = 0.680 at the defaults: cosmolike `accuracyboost` 1.0,
`integration_accuracy` 0, `kmax_boltzmann` 10; CAMB `AccuracyBoost` 1.1,
`k_per_logint` 15, `kmax` 10), the one-knob deltas are:

| knob | raised to | delta chi2 |
|---|---|---|
| cosmolike `accuracyboost` | 1.25 / 1.5 / 2 / 3 / 5 | +0.161 / +0.026 / -0.170 / -0.117 / -0.099 |
| cosmolike `integration_accuracy` | 10 | -0.014 |
| `kmax_boltzmann` + CAMB `kmax` | 40 + 50 | -0.003 |
| CAMB `k_per_logint` | 25 / 50 / 100 | +0.0005 / +0.0005 / +0.0005 |
| CAMB `AccuracyBoost` (at `k_per_logint` 50) | 1.5 / 2 | +0.009 / +0.014 |

With every knob raised at once, the six advisory checks report delta chi2 =
+0.007 (shear NLA), +0.002 (shear TATT), -0.129 (2x2pt NLA), +0.036 (2x2pt
TATT), -0.114 (3x2pt NLA), +0.039 (3x2pt TATT): all within the target.

No default changed. `k_per_logint` sits on its plateau already (25, 50, and
100 agree to 0.0001), and CAMB `AccuracyBoost` at 2 moves the chi2 by +0.014,
so the CAMB side is resolved. The cosmolike `accuracyboost` response is
non-monotone (the sign flips between 1.5 and 2), so a raised value adds
jitter, not convergence; the default stays at 1.0 and the likelihood yaml
files carry this measurement as a comment.

When a large accuracy delta appears, test the knobs in this order: cosmolike
`accuracyboost` first (cheap), then CAMB `k_per_logint`, and only then CAMB
`AccuracyBoost` (expensive at run time). An apparent `AccuracyBoost`
sensitivity can stand in for an unresolved cheap knob: in the roman_kl
project, an apparent +0.80 from `AccuracyBoost` collapsed to +0.002 once
`k_per_logint` reached 50. `kmax_boltzmann` (cosmolike) and `kmax` (CAMB) are
one physical cutoff seen from the two sides; move them together.
