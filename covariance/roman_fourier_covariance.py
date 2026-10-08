"""Survey adapter of the shared galaxy/shear covariance forecast.

An adapter is a thin module that hands one project's survey choices to
shared code. Every numerical algorithm lives in
cosmolike_notebook_utils.covariance; this module supplies the roman_fourier
inputs through three functions, which compute_covariance.py, the
covariance notebook and tests/covariance call in this order:

  configuration() -> the resolved settings: cosmology, redshift files,
                     measurement bands, catalog densities, footprint and
                     numerical accuracy;
  initialize()    -> runs CAMB once and installs that forecast, 8 lens and
                     8 source bins from data/roman_example.nz, in the
                     compiled interface;
  compute()       -> the G, SSC, cNG and total covariance matrices.

The example is a forecast with massless neutrinos, linear galaxy bias and
explicit Gaussian non-Limber and intrinsic-alignment choices. Its physical
choices differ from those behind the covariance shipped with the
likelihood (data/roman_example.cov), and it does not reproduce the frozen
likelihood configuration of the tests.
"""

from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.covariance.forecast import (
    initialize_forecast,
    gaussian_model,
    compute_forecast,
)
from cosmolike_notebook_utils import covariance as cov


def configuration(accuracy_boost=None, gaussian=None, **accuracy_overrides):
    """Return the resolved survey, cosmology and accuracy settings.

    covariance/README.md records the catalog assumptions and their sources.
    The n(z) files set the shape of each redshift distribution, not a
    number density; the densities below set the shot and shape noise.

    Arguments:
        accuracy_boost = None keeps the boost of default.yaml; 1, 2, 4 or 8
            refines the interpolation tables and the multipole cutoffs.
        gaussian = optional mapping for the Gaussian spectra only:
            nonlimber (bool), ia (none, NLA or TATT) and the amplitudes A1,
            A2 and B_TA (see forecast.gaussian_model).
        accuracy_overrides = internal controls of default.yaml given by
            name (name=value); the ** in the signature collects them into a
            dictionary.
    Returns:
        dict of fully resolved settings, every default the code applied
        included: cosmology, files, bands, densities, footprint, the
        numerical controls of default.yaml (the unboosted values kept) and
        the resolved Gaussian model.
    Raises:
        TypeError for an unknown accuracy control; ValueError for a
        default.yaml that holds no mapping or an invalid gaussian mapping.
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )

    # The 15 bands: 16 log-spaced edges from 30 to 4001 rounded to integers,
    # within one multipole of the likelihood's log-spaced band edges between
    # l_min = 30 and l_max = 4000. Band k holds every integer multipole from
    # band_first[k] to band_last[k], both included, so adjacent bands share
    # none and the top edge 4001 ends the last band at 4000. The Gaussian
    # covariance of a band sums its multipoles with weight 2 ell + 1, the
    # number of modes at each ell.
    band_edges = np.rint(np.geomspace(30, 4001, 16)).astype(np.int32)

    # The fiducial cosmology, shared by G, SSC and cNG, so CAMB runs once:
    # H0 in km/s/Mpc, As_1e9 = 10^9 A_s, w0pwa = w0 + wa and mnu in eV (zero:
    # the covariance halo model requires massless neutrinos). The other keys
    # are CAMB numerical settings (kmax in 1/Mpc) and the nonlinear model
    # (non_linear_emul 2 = CAMB's halofit, Takahashi version).
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

        # The n(z) files give the shape of each redshift distribution; lenses
        # and sources share one file, as in the likelihood dataset.
        # photoz_interpolation 0 = cubic spline and photoz_zmid 0 = the z
        # column holds left bin edges (Z_LOW), the likelihood defaults.
        "lens_file": "data/roman_example.nz",
        "source_file": "data/roman_example.nz",
        "photoz_interpolation": 0,
        "photoz_zmid": 0,

        # The measurement, held fixed while the accuracy is refined:
        # excluded_gammat lists the (lens, source) pairs, counted from 0,
        # removed from galaxy-galaxy lensing (the ggl_exclude list of the
        # likelihood yaml files; the key keeps its real-space name), and the
        # bands are inclusive integer intervals. lnm_edges are the panel
        # edges of the halo-mass integrals, in ln(M/[Msun/h]).
        "excluded_gammat": [[4, 0], [5, 0], [6, 0], [6, 1], [6, 2],
                            [7, 0], [7, 1], [7, 2], [7, 3]],
        "band_first": band_edges[:-1],
        "band_last": band_edges[1:]-1,
        "lnm_edges": cov.halo_mass_edges(),

        # Footprint and catalogs: survey area in square degrees; galaxy
        # densities per square arcminute in each tomographic bin (41.3 split
        # evenly over 8 bins, lenses and sources alike); the shape dispersion
        # sigma_e per ellipticity component; one linear bias per lens bin.
        "area_deg2": 2415.0,
        "lens_density_arcmin2": [41.3/8]*8,
        "source_density_arcmin2": [41.3/8]*8,
        "sigma_e_component": [0.30]*8,
        "bias": [1.18, 1.40, 1.55, 1.71, 1.90, 2.15, 2.52, 2.44],

        # theta_edges_arcmin: 16 log-spaced angular edges (15 bins from 2.5
        # to 250 arcmin), read only by the optional real-space companion.
        # a_edges: panel edges of the radial integrals in scale factor
        # a = 1/(1+z), increasing from the distant boundary (z = 4) to the
        # observer (z = 1e-5).
        "theta_edges_arcmin": np.geomspace(start=2.5, stop=250.0, num=16),
        "a_edges": 1.0/(1.0+np.array([4.0, 2., 1.5, 1., .7, .4, .2, 1.e-5])),
    }
    # update merges the numerical controls into settings; gaussian_model
    # turns the optional Gaussian choices into explicit per-source-bin
    # amplitude lists.
    settings.update(numerical)
    settings["gaussian"] = gaussian_model(
        gaussian=gaussian, nsource=len(settings["source_density_arcmin2"]),
    )
    return settings


def initialize(interface, settings):
    """Run CAMB once and install the complete forecast state.

    No covariance is computed here; compute() does that.

    Arguments:
        interface = the imported cosmolike_roman_fourier_interface module,
            built with covariance support.
        settings = resolved mapping from configuration().
    Returns:
        dict of the CAMB input tables (the set_cosmology arrays), suitable
        for saving beside the results.
    Raises:
        RuntimeError when the build lacks covariance support,
        FileNotFoundError for a missing n(z) file and ValueError for
        missing bins, a nonzero mnu or invalid densities, all before the
        interface state changes (forecast.initialize_forecast).
    Side effects:
        Replaces the interface's global cosmology and nuisance state. The
        likelihood covariance, data vector and mask are never loaded.
    """
    return initialize_forecast(
        interface=interface, settings=settings,
        project=Path(__file__).resolve().parents[1],
    )


def compute(interface, settings, space="real", rows=None, progress=None,
            backend=None):
    """Return the galaxy/shear forecast with G, SSC, cNG and total matrices.

    Arguments:
        interface = the compiled module, initialized by initialize().
        settings = resolved mapping from configuration().
        space = "real" (angular bins, the default of this function) or
            "fourier" (E-mode bandpowers: this project's measurement and
            the command-line default).
        rows = None for the full layout, or an int32 [n_observable, 3]
            table of (probe, field A, field B) rows to measure a subset.
        progress = optional callable receiving (stage, elapsed_seconds).
        backend = None for the notebook wrappers, interface.covariance for
            the direct production bindings to the same C calculations.
    Returns:
        dict with the "gaussian", "ssc", "cng" and "total" matrices
        [n_data, n_data] (dimensionless), the mean signals, diagnostics,
        the coordinate of each bin (band center sqrt(first*last) in
        Fourier space, geometric bin center in arcmin in real space) and
        the resolved settings. The full Fourier layout has
        (36 + 55 + 8) x 15 = 1485 entries (shear E modes only); the real
        layout has (2 x 36 + 55 + 8) x 15 = 2025 (xi+ and xi- for shear).
    Side effects:
        Reapplies the core accuracy boost and the Gaussian IA model to the
        interface; no files are written.
    """
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )
