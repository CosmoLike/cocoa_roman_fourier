"""Project choices for the shared galaxy/shear covariance notebook.

This module initializes 8 lens and 8 source distributions from the
project. Numerical algorithms live in cosmolike_notebook_utils.covariance.
The example is a massless-neutrino, zero-IA forecast with explicitly chosen
number densities; it does not reproduce the project's frozen likelihood.
"""

from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.covariance.forecast import (
    initialize_forecast,
    compute_forecast,
)
from cosmolike_notebook_utils import covariance as cov


def configuration(accuracy_boost=1):
    """Return resolved survey, cosmology and pilot-integration choices.

    The covariance README records the catalog assumptions and their sources.
    Redshift-file normalization sets a shape, not a catalog number density.
    Arguments:
        accuracy_boost = 1, 2, 4 or 8; raises covariance settings together.
    Returns:
        Fully resolved settings. Boost 1 is a pilot, not a certified FoM target.
    """
    numerical = cov.covariance_accuracy(accuracy_boost=accuracy_boost)

    # Inclusive bands count every integer multipole once.
    band_edges = np.rint(np.geomspace(30, 4001, 16)).astype(np.int32)

    # The fiducial is shared by G, SSC and cNG; CAMB runs only once.
    settings = {
        "cosmology": {
            "omegam": 0.3,
            "omegab": 0.04,
            "H0": 67.32,
            "ns": 0.96605,
            "As_1e9": 2.1,
            "w": -1.0,
            "w0pwa": -1.0,
            "mnu": 0.0,
            "AccuracyBoost": 1.0,
            "CLAccuracyBoost": 1.0,
            "CAMBAccuracyBoost": 1.0,
            "kmax": 20.0,
            "k_per_logint": 20,
            "non_linear_emul": 2,
            "lens_potential_accuracy": 1.0,
            "halofit_version": "takahashi",
        },

        # File columns describe radial shapes; these flags fix their z convention.
        "lens_file": "data/roman_example.nz",
        "source_file": "data/roman_example.nz",
        "photoz_interpolation": 0,
        "photoz_zmid": 0,

        # Measured pair and band choices are fixed during accuracy refinement.
        "excluded_gammat": [[4, 0], [5, 0], [6, 0], [6, 1], [6, 2],
                            [7, 0], [7, 1], [7, 2], [7, 3]],
        "band_first": band_edges[:-1],
        "band_last": band_edges[1:]-1,
        "lnm_edges": np.linspace(np.log(1.e6), np.log(1.e17), 9),

        # Densities are per square arcminute. Shape noise is per component.
        "area_deg2": 2415.0,
        "lens_density_arcmin2": [41.3/8]*8,
        "source_density_arcmin2": [41.3/8]*8,
        "sigma_e_component": [0.30]*8,
        "bias": [1.18, 1.40, 1.55, 1.71, 1.90, 2.15, 2.52, 2.44],

        # Shell edges increase in a, from the distant boundary to the observer.
        "theta_edges_arcmin": np.geomspace(start=2.5, stop=250.0, num=16),
        "a_edges": 1.0/(1.0+np.array([4.0, 2., 1.5, 1., .7, .4, .2, 1.e-5])),
    }
    settings.update(numerical)
    return settings


def initialize(interface, settings):
    """Run CAMB once and install the complete forecast state without a covariance.

    Arguments:
        interface = imported cosmolike_roman_fourier_interface module.
        settings = resolved mapping from configuration().
    Returns:
        CAMB input tables as a dict, suitable for saving beside results.
    Side effects:
        Replaces the interface's global cosmology and nuisance state. The
        likelihood covariance, data vector and mask are never loaded.
    """
    return initialize_forecast(
        interface=interface, settings=settings,
        project=Path(__file__).resolve().parents[1],
    )


def compute(interface, settings, space="real", rows=None, progress=None):
    """Return the galaxy/shear forecast with G, SSC, connected and total matrices.

    Arguments: interface = initialized compiled project; settings = configuration();
        space = "real" or "fourier"; rows = optional measured row subset;
        progress = optional (stage, elapsed_seconds) callback.
    Returns: shared forecast dict, including resolved settings and coordinates.
    The full real layout has 2025 entries; Fourier has 1485 entries.
    """
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress,
    )
