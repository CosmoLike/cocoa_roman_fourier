"""Cobaya likelihood roman_fourier.combo_2x2pt: galaxy clustering and lensing.

2x2pt combines two of the three 3x2pt probes: galaxy clustering C_ell^gg
(the 8 lens-bin auto spectra) and galaxy-galaxy lensing C_ell^gs (the 55
lens-source pairs left after ggl_exclude), each at the 15 band centers of
the dataset. Cosmic shear is left out: cosmolike masks its entries of the
full 1485-entry layout.

All the computation is in _cosmolike_prototype_base.py; this class only
selects the probe "2x2pt". Cobaya reads the options from combo_2x2pt.yaml
next to this file.
"""

from cobaya.likelihoods.roman_fourier._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_fourier_interface as ci
import numpy as np

class combo_2x2pt(_cosmolike_prototype_base):
  """2x2pt likelihood: the shared base class without cosmic shear."""
  def initialize(self):
    """Run the shared initialization with the probe "2x2pt".

    Cobaya calls initialize once, when it builds the model; the options of
    combo_2x2pt.yaml and of the user's yaml block are attributes of self
    by then. super(combo_2x2pt, self) reaches the method of the base class.
    """
    super(combo_2x2pt,self).initialize(probe="2x2pt")