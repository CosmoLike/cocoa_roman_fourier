"""Register the --high and --mask command line options of these tests.

pytest imports a file named conftest.py before it collects the tests of
that folder and its subfolders, and calls the functions with the reserved
names pytest_addoption and pytest_configure at fixed moments (they are
"hooks"). The file must sit in each project's tests folder, so it cannot
move; its content is the shared implementation in
cosmolike_core/cocoa_testing.py, bound here the same way
cocoa_test_utils.py binds the test harness.

--high=1 repeats the CFASTPT-vs-FASTPT sweeps (tests 15-17) at the
high-accuracy settings. --mask selects the scale-cut mask of those sweeps
and of the Halofit-vs-EE2 checks: "frozen" (the default, the contract
mask) or "ones" (every data point kept), the choices of this project's
harness (its fastpt_masks tuple).
"""

import os
import sys

# The tests folder is not a package (a folder Python imports as a unit,
# marked by an __init__.py); inserting it first on the module search path
# lets the project shim resolve wherever pytest was launched from (the
# shim itself puts cosmolike_core on the path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u


def pytest_addoption(parser):
    """Register --high and --mask on pytest's parser.

    The shared implementation and its documentation live in
    cocoa_testing.conftest_addoption.

    Arguments:
      parser = pytest's option parser (supplied by pytest).

    Returns:
      nothing; the options become readable through config.getoption.
    """
    u._cct.conftest_addoption(parser, u._H.fastpt_masks)


def pytest_configure(config):
    """Copy the option values where the test classes read them.

    The tests are unittest.TestCase classes, which cannot receive pytest
    fixtures, so the values travel as environment variables. The shared
    implementation and its documentation live in
    cocoa_testing.conftest_configure.

    Arguments:
      config = pytest's configuration object (supplied by pytest).

    Returns:
      nothing; the environment of this process gains COCOA_FASTPT_HIGH
      and COCOA_FASTPT_MASK.
    """
    u._cct.conftest_configure(config)
