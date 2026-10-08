"""Profile one sampled parameter for hybrid configuration 2 (3x2pt).

A hybrid example takes the background expansion and the matter power
spectra from trained neural-network emulators instead of a Boltzmann code
such as CAMB; cosmolike still computes the survey projections, the galaxy
bias and the intrinsic alignments.
Configuration 2 is the roman_fourier.combo_3x2pt likelihood (cosmic
shear, galaxy-galaxy lensing and galaxy clustering, NLA intrinsic
alignments) of EXAMPLE_EMUL2_EVALUATE2.yaml.

This file only starts the shared driver, run() in
external_modules/code/cosmolike_core/cocoa_hybrid_sampling.py, whose
docstring lists every option. The driver reads the evaluate YAML (--input
selects another) and evaluates its sampler.evaluate.override point first.
It then fixes one sampled parameter (--profile, a name or a zero-based
index) at each value of a grid centered on a saved minimum (--minfile, the
JSON result of EXAMPLE_EMUL2_MINIMIZE2.py with the same YAML) and repeats
the annealed emcee minimization over the other parameters. The minimized
-2 log posterior against the fixed value is a profile of the posterior;
priors are included, so it is not a pure likelihood profile. --factor sets
the half-width of the grid in proposal standard deviations and --numpts
its number of values.

Results go to projects/roman_fourier/chains/; the driver refuses to
overwrite an existing result, so each run needs a new --outroot.

From the Cocoa/ folder with Cocoa activated, this command evaluates the
fiducial point and prints the sampled-parameter order without sampling:

    python ./projects/roman_fourier/EXAMPLE_EMUL2_PROFILE2.py --check

The project README gives the MPI commands: mpirun -n 2 ... starts two
cooperating copies of this script (MPI ranks); one coordinates and the
others evaluate the model.
"""

from pathlib import Path
import sys

# project is this file's folder (projects/roman_fourier); parents[1] climbs
# two folders up to Cocoa/, which holds the shared core.
project = Path(__file__).resolve().parent
core = project.parents[1]/"external_modules/code/cosmolike_core"
# insert(0, ...) puts the core first on Python's module search path, so the
# import below finds cocoa_hybrid_sampling there. Nothing numerical may be
# imported before it: that module fixes the BLAS thread counts (one thread
# per MPI rank) before numpy loads.
sys.path.insert(0, str(core))

from cocoa_hybrid_sampling import run


# __name__ is "__main__" only when this file runs as a script (on every MPI
# rank), not when another module imports it.
if __name__ == "__main__":
    run(mode="profile", project=project, example=2)
