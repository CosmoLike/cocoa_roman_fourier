"""Loader stub of the compiled cosmolike interface of roman_fourier.

cosmolike_roman_fourier_interface.so is a compiled Python extension
module: the C/C++ cosmolike library plus the bindings of interface.cpp,
built by MakefileCosmolike (scripts/compile_roman_fourier.sh runs it). The
likelihood imports it as ci and calls ci.<function>.

This file is the setuptools-style stub for that library. Python's import
system tries extension modules (.so) before .py files in each folder, so
when the .so sits in this folder the import loads it directly and this
file never runs. When the stub does run, it loads the .so next to itself
in place of this module.
"""

def __bootstrap__():
   """Load the .so next to this file as this module, then remove the helper.

   The global statement makes the assignments below rebind the module-level
   names __file__ and __loader__. pkg_resources.resource_filename gives the
   path of the .so in this module's folder, and imp.load_dynamic loads that
   extension under this module's name, so sys.modules then holds the
   compiled module. imp and pkg_resources exist in the declared Python 3.11
   environment; imp is absent from Python 3.12 on.
   """
   global __bootstrap__, __loader__, __file__
   import sys, pkg_resources, imp
   __file__ = pkg_resources.resource_filename(__name__,'cosmolike_roman_fourier_interface.so')
   __loader__ = None; del __bootstrap__, __loader__
   imp.load_dynamic(__name__,__file__)
__bootstrap__()
