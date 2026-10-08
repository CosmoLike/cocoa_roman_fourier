"""Cobaya likelihood roman_fourier.cosmic_shear: the shear power spectra.

Cosmic shear is the weak-lensing distortion of galaxy shapes by the matter
between the galaxies and the observer. In this Fourier-space project the
likelihood compares the shear E-mode angular power spectra C_ell^EE of the
36 source-bin pairs (i <= j), evaluated at the 15 band centers of the
dataset; shear has no B-mode or xi- rows here. The data vector keeps the
full 1485-entry 3x2pt layout of the dataset, and cosmolike masks out its
galaxy-galaxy lensing and clustering entries for this probe.

All the computation is in _cosmolike_prototype_base.py; this class only
selects the probe. "xi" is cosmolike's name for the shear-shear probe in
every project, real or Fourier space. Cobaya reads the options from
cosmic_shear.yaml next to this file.
"""

from cobaya.likelihoods.roman_fourier._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_fourier_interface as ci
import numpy as np

class cosmic_shear(_cosmolike_prototype_base):
  """Cosmic-shear likelihood: the shared base class restricted to shear."""
  def initialize(self):
    """Run the shared initialization with the shear-only probe "xi".

    Cobaya calls initialize once, when it builds the model; the options of
    cosmic_shear.yaml and of the user's yaml block are attributes of self
    by then. super(cosmic_shear, self) reaches the method of the base class.
    """
    super(cosmic_shear,self).initialize(probe="xi")