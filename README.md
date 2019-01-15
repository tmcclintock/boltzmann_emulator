# Boltzmann emulator

Boltzmann codes such as CAMB or CLASS can be very slow. This repository contains a tool for emulating the outputs of a given boltzmann code in order to estimate their outputs quickly.

## Installation

One can install the requirements with `pip install -r requirements.txt`. If you have trouble with this, it is almost certainly with the package [george](https://george.readthedocs.io/en/latest/), used for the Gaussian processes. Feel free to raise an issue if this is the case.

With the requirements installed, install this package with `python setup.py install`. Run the unit tests just by typing `pytest` in the main directory of this project. If any tests fail, please raise an issue.

## Usage

A minimal example of this package would consist of the following:

```python

import boltzmann_emulator as BE

#A 2D array of cosmological parameters
parameters = ...
#A list of redshifts
redshifts = ...
#A list of wavenumbers
k = ...
#A 2D array (N_cosmologies x (N_z x N_k)) of the power spectra
#Note that the power spectra have been stacked in the last dimension
Pkz_training = ...

#Create the emulator
Emu = BE.boltzmulator(parameters, redshifts, k, Pkz_training)

#Predict at a new location in parameter space
Pkz_predicted = Emu.predict(test_parameters)
```