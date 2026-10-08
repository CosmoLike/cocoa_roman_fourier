"""Compute and save the roman_fourier 3x2pt covariance from a YAML file.

A covariance matrix C[i, j] = <(d_i - <d_i>)(d_j - <d_j>)> gives the
statistical scatter of each entry of the measured data vector d and the
correlations between entries; the likelihood weights the data-minus-model
difference by its inverse. This runner forecasts C for the Fourier-space
3x2pt data vector of the project: cosmic shear (shear E modes),
galaxy-galaxy lensing and galaxy clustering, 99 tomographic spectra in 15
multipole bands, 1485 entries before the dataset mask. Each band holds
every integer multipole between its two endpoints, both included. Four
matrices are kept separately: the Gaussian part (G), the super-sample
covariance (SSC, from density modes larger than the survey), the
connected non-Gaussian part (cNG, from the matter trispectrum) and their
sum.

The YAML uses Cobaya's syntax (theory, params, sampler: evaluate) to fix
one cosmology; its covariance block selects the measurement space and the
accuracy, and output names the .npz archive (a numpy file of named
arrays). No sampler runs. From the Cocoa/ folder, with Cocoa activated and
the project compiled with covariance support (project README):

    python projects/roman_fourier/covariance/compute_covariance.py \\
        projects/roman_fourier/EXAMPLE_EVALUATE_COVARIANCE.yaml

--output PATH writes another archive and --overwrite replaces an existing
one; --help lists every option. The survey settings come from
roman_fourier_covariance.py and are shared with the notebook;
covariance/README.md explains the physics, the accuracy settings and the
figures.
"""

import os
from pathlib import Path
import sys

# BLAS libraries (OpenBLAS, MKL, Apple's vecLib: the linear-algebra code
# behind numpy) read their thread counts once, when first loaded, so these
# lines precede every numerical import. One BLAS thread leaves the cores to
# cosmolike's OpenMP team, whose size comes from OMP_NUM_THREADS in the
# environment, never from the YAML.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# This runner evaluates one matrix in one process. Cobaya supplies only the
# YAML reader; COBAYA_NOMPI=1 keeps it from starting MPI (the library that
# runs several cooperating processes).
os.environ["COBAYA_NOMPI"] = "1"

# project = projects/roman_fourier (parents[1] of this file climbs out of
# covariance/); the core is Cocoa/external_modules/code/cosmolike_core.
# insert(0, ...) puts each folder first on Python's module search path:
# the core provides cosmolike_notebook_utils, interface/ the compiled
# module. Python adds this script's own folder by itself, which provides
# roman_fourier_covariance.
project = Path(__file__).resolve().parents[1]
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))
sys.path.insert(0, str(project/"interface"))

import cosmolike_roman_fourier_interface as ci
import roman_fourier_covariance as survey
from cosmolike_notebook_utils.covariance.command_line import run_covariance


# __name__ is "__main__" only when this file runs as a script. The project's
# native measurement is Fourier bandpowers (default_space); joint=False
# selects the galaxy/shear layout, not the cluster layout of other projects.
if __name__ == "__main__":
    run_covariance(
        interface=ci, survey=survey, default_space="fourier", joint=False,
    )
