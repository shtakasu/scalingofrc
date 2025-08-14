import numpy as np
from funcs.utils import *
import cupyx.scipy.special as cuspecial

params = ([0.9, 0.5],  [1.3, 0.3], [1.5, 0.0]) #(g, \sigma_n)
sigma_s_tilde = 1.

max_leadout = 100
N=10000
Tobs = 10000
trial = 10
act_func = lambda x: cuspecial.erf(x*0.5*np.sqrt(np.pi))

data = np.zeros([len(params), trial, max_leadout])

for i, [g, sigma_n] in enumerate(params):
    data[i] = mc_calc(g,N,act_func,max_leadout,sigma=sigma_s_tilde/(N**0.25), connectivity="Gaussian", eta=0, trial=trial, Tobs= Tobs, Tinit=1000, max_delay=500,  sigma_noise=sigma_n, dinit=0)

np.save("data/mcscaling_erf.npy", data)
