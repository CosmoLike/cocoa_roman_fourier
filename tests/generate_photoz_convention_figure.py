"""Regenerate the photo-z convention figures the tests README shows.

Evaluates the frozen cosmic-shear fiducial under the four runtime
photo-z settings (cspline/Z_LOW default, linear, Steffen, Z_MID; see
test_photoz_conventions.py for what they mean) and plots the
fractional data-vector differences against the default,

    delta C_ell / C_ell = C_ell(setting)/C_ell(default) - 1,

per source-bin pair and per multipole band (C_ell^EE at the band
center). Masked bands are left out. Two figures, because the two knobs
act on different scales:

    photoz_zmid_dcl.png    the Z_LOW vs Z_MID reading of the n(z)
                           file z column (percent level),
    photoz_interp_dcl.png  linear and Steffen vs cubic spline
                           (1e-4 level).

The four models are built one after the other in this process (they
share example1's dimensions), and the figures are written into tests/,
replacing the files of the same name.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python ./projects/roman_fourier/tests/generate_photoz_convention_figure.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so it is
# set before any cobaya/cosmolike import; 4 is the count the tests use.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import shutil
import tempfile

# Agg is matplotlib's non-interactive backend: it draws into files without
# opening a window, and it must be selected before pyplot is imported.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# insert(0, ...) puts this folder (tests/) first on the module search path,
# so the import below finds this project's shim.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

# These repeat the frozen dataset (frozen/data/roman_example.dataset):
# source_ntomo, n_cl, l_min and l_max. The shear block of the data vector
# holds NTOMO (NTOMO + 1)/2 = 36 source pairs of NCL bands each.
EXAMPLE = "example1"
NTOMO = 8
NCL = 15
L_MIN, L_MAX = 30.0, 4000.0
MASK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "frozen", "data", "roman_example.mask")

# (report tag, photoz_interpolation_type, photoz_zmid_convention)
SETTINGS = (("cspline/Z_LOW (default)", 0, 0), ("linear", 1, 0),
            ("steffen", 2, 0), ("Z_MID", 0, 1))


def datavectors():
    """Return the theory data vector under each setting, keyed by tag.

    Each model prints its theory vector into a temporary folder, deleted
    on exit (the finally block runs whether or not an error occurred).

    Returns:
      dict tag -> float64 array [1485], the full-length data vector.

    Side effects:
      Builds four example1 models in this process, replacing cosmolike's
      global state.
    """
    vectors_dir = tempfile.mkdtemp(prefix="photoz_conventions_fig_")
    out = {}
    try:
        for tag, interp, zmid in SETTINGS:
            print(f"building model ({EXAMPLE}, NLA, {tag}) ...", flush=True)
            info = u.load_frozen_info(EXAMPLE, tatt=False)
            block = info["likelihood"][u.EXAMPLES[EXAMPLE]["likelihood"]]
            block["photoz_interpolation_type"] = interp
            block["photoz_zmid_convention"] = zmid
            path = os.path.join(vectors_dir, f"dv_{interp}_{zmid}.modelvector")
            block["print_datavector"] = True
            block["print_datavector_file"] = path
            model = u.make_model(info)
            point = u.build_point(model, EXAMPLE, tatt=False)
            u.evaluate_chi2(model, point)
            out[tag] = u._load_datavector(path)
    finally:
        shutil.rmtree(vectors_dir, ignore_errors=True)
    return out


def plot(curves, fname, title, scale=100.0, unit="%", ylim=None):
    """Draw one figure: a 6x6 grid of glued panels, one per source pair.

    The 36 source pairs (i <= j) fill the grid exactly; panels share both
    axes, and the bin labels count from 1.

    Arguments:
      curves = dict label -> fractional-difference array [36, NCL], NaN
               at masked bands.
      fname  = file name of the PNG, written into tests/.
      title  = figure title.
      scale  = factor applied to the fractions before plotting (100 for
               percent).
      unit   = unit text of the y label.
      ylim   = None, or the half-height of the shared symmetric y range,
               in plotted units.

    Returns:
      nothing.

    Side effects:
      Writes tests/<fname> at 120 dpi, replacing an existing file.
    """
    edges = np.geomspace(L_MIN, L_MAX, NCL + 1)
    ell = np.sqrt(edges[1:] * edges[:-1])  # geometric band centers
    pairs = [(i, j) for i in range(NTOMO) for j in range(i, NTOMO)]
    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(20, 16),
                             sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0, "hspace": 0})
    cm = plt.get_cmap("gist_rainbow")
    for p, (i, j) in enumerate(pairs):
        ax = axes.ravel()[p]
        for q, (label, dcl) in enumerate(curves.items()):
            color = cm(q / max(len(curves) - 1, 1) * 0.8)
            ax.semilogx(ell, scale * dcl[p], color=color, lw=1.6,
                        label=label if p == 0 else None)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.text(0.08, 0.85, f"$({i+1},{j+1})$", transform=ax.transAxes,
                fontsize=13)
        if p >= 30:
            ax.set_xlabel(r"$\ell$", fontsize=16)
        if p % 6 == 0:
            ax.set_ylabel(rf"$\Delta C_\ell/C_\ell$ [{unit}]", fontsize=14)
    if ylim is not None:
        axes.ravel()[0].set_ylim(-ylim, ylim)
    axes.ravel()[0].legend(fontsize=9, loc="lower left")
    fig.suptitle(title, fontsize=17)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


def main():
    """Check the frozen state, compute the four vectors and draw both figures.

    Raises:
      RuntimeError outside an activated Cocoa shell; AssertionError from
      verify_frozen when a frozen file changed.

    Side effects:
      Writes tests/photoz_zmid_dcl.png and tests/photoz_interp_dcl.png.
    """
    u.require_cocoa_environment()
    u.verify_frozen()
    dv = datavectors()

    # The mask file has two columns, index and value; the conditional
    # expression keeps the value column (or the array itself if a file
    # held one column).
    mask = np.loadtxt(MASK_FILE)
    mask = mask[:, 1] if mask.ndim == 2 else mask
    npair = NTOMO * (NTOMO + 1) // 2
    ncs = npair * NCL  # the cosmic-shear block leads the data vector

    def frac(tag):
        """Return C_ell(tag)/C_ell(default) - 1, shape [36, NCL], NaN if masked.

        errstate silences the division warnings of masked or zero entries,
        which np.where replaces by NaN anyway.
        """
        ref, cur = dv[SETTINGS[0][0]], dv[tag]
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(mask[:ncs] > 0, cur[:ncs] / ref[:ncs] - 1.0,
                         np.nan)
        return d.reshape(npair, NCL)

    plot({"Z_MID": frac("Z_MID")},
         "photoz_zmid_dcl.png",
         "roman_fourier cosmic shear: n(z) z-column read as Z_MID "
         "instead of Z_LOW (frozen fiducial)", scale=100.0, unit="%",
         ylim=4.0)
    plot({"linear": frac("linear"), "steffen": frac("steffen")},
         "photoz_interp_dcl.png",
         "roman_fourier cosmic shear: linear and Steffen n(z) "
         "interpolation vs cubic spline (frozen fiducial)",
         scale=1.0e4, unit=r"$10^{-4}$", ylim=8.0)


if __name__ == "__main__":
    main()
