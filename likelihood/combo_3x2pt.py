"""Cobaya likelihood roman_fourier.combo_3x2pt: shear, lensing and clustering.

3x2pt combines three sets of two-point statistics, here angular power
spectra at the 15 band centers of the dataset: cosmic shear C_ell^EE (36
source pairs, E modes only), galaxy-galaxy lensing C_ell^gs (55
lens-source pairs) and galaxy clustering C_ell^gg (8 lens-bin auto
spectra), 1485 entries before the scale cuts of the mask.

All the computation is in _cosmolike_prototype_base.py; this class only
selects the probe "3x2pt". Cobaya reads the options from combo_3x2pt.yaml
next to this file.
"""

from cobaya.likelihoods.roman_fourier._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_fourier_interface as ci
import numpy as np

class combo_3x2pt(_cosmolike_prototype_base):
  """3x2pt likelihood: the shared base class with all three probes."""
  def initialize(self):
    """Run the shared initialization with the probe "3x2pt".

    Cobaya calls initialize once, when it builds the model; the options of
    combo_3x2pt.yaml and of the user's yaml block are attributes of self
    by then. super(combo_3x2pt, self) reaches the method of the base class.
    """
    super(combo_3x2pt,self).initialize(probe="3x2pt")