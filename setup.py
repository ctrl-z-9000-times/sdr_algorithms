#!/usr/bin/python3
"""
Written by David McDougall, 2018

CYTHON COMPILE COMMAND:
$ ./setup.py build_ext --build-lib `pwd`/../
"""

from distutils.core import setup
from Cython.Build import cythonize

cython_files = (
    "synapses.pyx",
)

setup(
  name = 'sdr_algorithms',
  ext_modules = cythonize(cython_files),
)
