import numpy as np
import george
import george.kernels as ker

class boltzmulator(object):
    """
    An emulator for a boltzmann code. The emulator is trained
    on a set of input power spectra at given locations in cosmological
    parameter space. The power spectra are evaluated over a set of
    redshifts and wavenumbers (h/Mpc com.).

    Args:
        parameters (array-like): locations in parameter space
            of the input power spectra.
        redshifts (float array-like): list of redshifts. 
            Can be a single number.
        k (array-like): wavenumbers of the input power spectra.
        power_spectra (array-like): 2D array of power spectra
            evaluated at each location in parameter space.
    """

    def __init__(self, parameters, redshifts, k, power_spectra):
        parameters = np.array(parameters)
        redshifts = np.array(redshifts)
        k = np.array(k)
        power_spectra = np.array(power_spectra)

        if parameters.ndim != 2:
            raise Exception("Parameters must be 2D array.")
        if power_spectra.ndim != 2:
            raise Exception("Power spectra must be a 2D array of dimensions "+
                            "N_parameters x (N_k*N_z).")
        if len(parameters) != len(power_spectra):
            raise Exception("Power spectra must be a 2D array of dimensions "+
                            "N_parameters x (N_k*N_z).")
        if len(redshifts)*len(k) != len(power_spectra[0]):
            raise Exception("Power spectra must be a 2D array of dimensions "+
                            "N_parameters x (N_k*N_z).")
        
        self.parameters = parameters
        self.redshifts = redshifts
        self.k = k
        self.power_spectra = power_spectra

if __name__ == "__main__":
    params = [[0], [1]]
    k = [2, 3]
    z = [4, 5]
    P = [[6, 7, 8, 9], [10, 11, 12, 13]]
    b = boltzmulator(params, k, z, P)
