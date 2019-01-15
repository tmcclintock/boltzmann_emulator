import copy
import numpy as np
import george
from george.kernels import ExpSquaredKernel, Matern52Kernel, \
    ExpKernel, RationalQuadraticKernel, Matern32Kernel
import scipy.optimize as op

#Assert statements to guarantee the linter doesn't complain
assert ExpSquaredKernel
assert Matern52Kernel
assert ExpKernel
assert Matern32Kernel
assert RationalQuadraticKernel


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

    def __init__(self, parameters, redshifts, k, power_spectra,
                 number_of_principle_components=1, kernel=None):
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
        self.Npars = len(self.parameters[0])

        self.NPC = number_of_principle_components
        metric_guess = np.std(self.parameters, 0)
        if kernel is None:
            kernel = 1.*ExpSquaredKernel(metric=metric_guess, ndim=self.Npars)
        self.kernel = kernel
        
    def train(self):
        """
        Train the emulator.
        """
        zs = self.redshifts
        k = self.k
        p = self.power_spectra
        k2p = copy.deepcopy(p)
        Nk = len(k)
        Nz = len(zs)
        #Multiply each P(k) by k^2, but note the shapes
        #of the power spectra array we have to deal with
        for i in range(Nz):
            lo = i*Nk
            hi = (i+1)*Nk
            k2p[:, lo:hi] *= k**2
        #Take the log
        lnk2p = np.log(k2p)
        #Remove the mean
        lnk2p_mean = np.mean(lnk2p)
        lnk2p_std = np.std(lnk2p, 0)
        lnk2p = (lnk2p - lnk2p_mean)/lnk2p_std
        #Save what we have noe
        self.lnk2p = lnk2p
        self.lnk2p_mean = lnk2p_mean
        self.lnk2p_std = lnk2p_std

        #Do SVD to pull out principle components
        u,s,v = np.linalg.svd(lnk2p, 0) #Do the PCA
        s = np.diag(s)
        N = len(s)
        P = np.dot(v.T, s)/np.sqrt(N)
        Npc = self.NPC #number of principle components
        phis = P.T[:Npc]
        ws = np.sqrt(N) * u.T[:Npc]
        #Save the weights and PCs
        self.ws = ws
        self.phis = phis

        #Create the GPs and save them
        gplist = []
        for i in range(Npc):
            ws = self.ws[i, :]
            kern = copy.deepcopy(self.kernel)
            gp = george.GP(kernel=kern, fit_kernel=True, mean=np.mean(ws))
            gp.compute(self.parameters)
            gplist.append(gp)
            continue
        self.gplist = gplist

        #Train the GPs
        for i, gp in enumerate(self.gplist):
            ws = self.ws[i, :]
            def nll(p):
                gp.set_parameter_vector(p)
                ll = gp.log_likelihood(ws, quiet=True)
                return -ll if np.isfinite(ll) else 1e25
            def grad_nll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(ws, quiet=True)
            p0 = gp.get_parameter_vector()
            result = op.minimize(nll, p0, jac=grad_nll)
            gp.set_parameter_vector(result.x)
            continue
        
        self.trained=True
        return

    def predict(self, parameters):
        """
        Predict the power spectrum at a set of cosmological parameters.
        """
        
if __name__ == "__main__":
    params = [[0], [1]]
    k = [2, 3]
    z = [4, 5]
    P = [[6, 7, 8, 9], [10, 11, 12, 13]]
    b = boltzmulator(params, k, z, P)
    b.train()
