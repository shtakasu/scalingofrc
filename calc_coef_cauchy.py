import numpy as np
from funcs.utils import *

N_list = [1000, 5000, 10000]
gamma_list = np.arange(0.1, 5.1, 0.1)
phi = np.tanh
T = 10000
Tinit = 1000
sigma = 1.
nums = 10

coefs = np.zeros([len(N_list), len(gamma_list), nums])

for idN, N in enumerate(N_list):
    for idgamma, gamma in enumerate(gamma_list):
        coefs[idN, idgamma, :] = mean_coef(gamma, N, act_func=phi, sigma=sigma, connectivity="Cauchy", eta=0, trial=10, Tobs= 10000, Tinit=1000, sigma_noise=0.0)
        print(f"N={N}, gamma={np.round(gamma, decimals=3)} was done")
        
np.save("data/coef_cauchy.npy", coefs)
