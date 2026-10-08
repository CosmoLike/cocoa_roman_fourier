# This file makes likelihood/ a Python package. Cocoa links this folder
# into Cobaya's likelihoods package as roman_fourier, so a yaml block named
# roman_fourier.cosmic_shear (or .combo_2x2pt, .combo_3x2pt) loads the
# class of that name from the module of that name here, with the defaults
# of the matching .yaml file. _cosmolike_prototype_base.py holds the class
# the three share.
