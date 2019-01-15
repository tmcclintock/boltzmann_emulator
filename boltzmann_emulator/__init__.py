"""Emulator for boltzmann codes.

This module provides an emulator for boltzmann codes, or a tool
to predict boltzmann codes in a domain space given a set of
training points in that domain space.

"""
from .boltzmulator import boltzmulator
assert boltzmulator #A hack to get pyflakes to not complain

__version__ = "0.1.0"
__author__ = "Thomas McClintock"
__email__ = "mcclintock@bnl.gov"
