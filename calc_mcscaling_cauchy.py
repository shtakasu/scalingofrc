import numpy as np
from funcs.utils import *

N = 10000
g_list = [0.1, 0.5, 1, 2]
phi = np.tanh
Tobs = 10000
Tinit = 1000
sigma = 1
nums = 10
max_leadout = 100

data = np.zeros([len(g_list), nums, max_leadout])

for idg, g in enumerate(g_list):
    
    data[idg,:,:] = mc_calc(g,N,phi,max_leadout,sigma, connectivity="Cauchy", eta=0, trial=nums, Tobs= Tobs, Tinit=Tinit, max_delay=500,  sigma_noise=0.0, dinit=0)
    
    print(f"g={g} was done")
    
np.save("data/mcscaling_cauchy.npy", data)
