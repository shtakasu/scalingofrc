import numpy as np
from funcs.utils import *

N_list = [1000,5000,10000]
eta_list = np.arange(-0.65, 0.7, 0.05)
g = 0.6
phi = np.tanh
T = 10000
Tinit = 1000
sigma = 0.1
nums = 10

coefs = np.zeros([len(N_list), len(eta_list), nums])

for idN, N in enumerate(N_list):
    for ideta, eta in enumerate(eta_list):
        coefs[idN, ideta, :] = mean_coef(g, N, act_func=phi, sigma=sigma, connectivity="Correlated", eta=eta, trial=10, Tobs= 10000, Tinit=1000, sigma_noise=0.0)
        print(f"N={N}, eta={np.round(eta, decimals=3)} was done")
        
np.save("data/coef_reciprocal.npy", coefs)
