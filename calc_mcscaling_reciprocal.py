import numpy as np
from funcs.utils import *

N = 10000
g = 0.6
eta_list = [-0.65, 0.65, -0.5, 0.5, 0]
phi = np.tanh
Tobs = 10000
Tinit = 1000
sigma = 0.1
nums = 10
max_leadout = 100

data = np.zeros([len(eta_list), nums, max_leadout])

for ideta, eta in enumerate(eta_list):
    
    data[ideta,:,:] = mc_calc(g,N,phi,max_leadout,sigma, connectivity="Correlated", eta=eta, trial=nums, Tobs= Tobs, Tinit=Tinit, max_delay=500,  sigma_noise=0.0, dinit=0)
    
    print(f"eta={eta} was done")
    
np.save("data/mcscaling_reciprocal.npy", data)
