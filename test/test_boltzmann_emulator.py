import boltzmann_emulator as be
import numpy as np
import numpy.testing as npt

def test_emulator_builds():
    params = np.atleast_2d(np.arange(10).reshape((2,5)))
    zs = np.ones(3)
    k = np.ones(4)
    ps = np.ones((2, 12))
    Emu = be.boltzmulator(params, zs, k, ps)
    assert Emu #builds
    return

    
