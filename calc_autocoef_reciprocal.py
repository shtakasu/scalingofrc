import numpy as np
from funcs.utils import *

""" hyperparameter setting"""
N = 10000
L = 50
Tinit = 1000 # washout time step
Tobs = 10000 # simulation time step
max_delay = 100 # max delay time for calculating MC
g = 0.6
sigma = 0.1
eta_list = [-0.65, 0.65, -0.5, 0.5, 0]
phi = np.tanh
nums = 10

data = np.zeros([len(eta_list), nums, max_delay+1])

for ideta, eta in enumerate(eta_list):
    data[ideta] = mean_autocoef(g, N, act_func=phi, sigma=sigma, max_delay=max_delay, connectivity="Correlated", eta=eta, trial=nums, Tobs= Tobs, Tinit=Tinit, sigma_noise=0.0)
    
np.save("data/autocoef_reciprocal.npy", data)
