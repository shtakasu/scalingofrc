import numpy as np
from funcs.utils import *

N = 10000
g_list = [0.3, 0.9]
sigman_list = [1.0, 0.5]
sigma = 1.0/(N**0.25)
phi = lambda x: x
Tobs = 10000
Tinit = 1000
nums = 10
max_leadout = 100

data = np.zeros([len(g_list), nums, max_leadout])

for id, (g, sigman) in enumerate(zip(g_list, sigman_list)):
    
    data[id,:,:] = mc_calc(g,N,phi,max_leadout,sigma, connectivity="Gaussian", eta=0, trial=nums, Tobs= Tobs, Tinit=Tinit, max_delay=500,  sigma_noise=sigman, dinit=0)
    
np.save("data/mcscaling_linear.npy", data)
