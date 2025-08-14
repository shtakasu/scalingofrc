import numpy as np
from funcs.utils import *

N = 10000
g_list = [0.9, 0.9, 1.3, 1.3]
sigma_list = [0.1, 1.0, 0.1, 1.0]
sigma_noise_list = [0.1, 0.1, 0.1, 0.1]
phi = np.tanh
Tobs = 10000
Tinit = 1000
nums = 10
max_leadout = 50

data = np.zeros([len(g_list), nums, max_leadout])

for i in range(len(g_list)):
    g = g_list[i]
    sigma_s = sigma_list[i]
    sigma_n = sigma_noise_list[i]
    data[i,:,:] = mc_calc(g,N,phi,max_leadout,sigma=sigma_s, sigma_noise=sigma_n, connectivity="Gaussian", eta=0, trial=nums, Tobs= Tobs, Tinit=Tinit, max_delay=500, dinit=0)

np.save("data/mcscaling_gaussian.npy", data)
