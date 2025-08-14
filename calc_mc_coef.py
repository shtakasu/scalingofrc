import numpy as np
from funcs.utils import *

N = 10000
actfunc_type = "relu" #"relu" or "tanh" or "linear"
points = 100
max_leadout=100
trial=10
Tobs= 10000 
Tinit=1000 
max_delay=500 
min_g = 0.2; max_g = 1.4 
min_sigmas = 0.1; max_sigmas = 3.
min_sigman = 0. ; max_sigman = 3.

memory_capacity, coef, params = mcscaling_and_coef(N, actfunc_type, min_g, max_g, min_sigmas, max_sigmas, min_sigman, max_sigman, max_leadout=max_leadout, points=points, trial=trial, Tobs= Tobs, Tinit=Tinit, max_delay=max_delay, dinit=0)

np.save("data/mccoef_mc_"+actfunc_type+".npy", memory_capacity)
np.save("data/mccoef_coef_"+actfunc_type+".npy", coef)
np.save("data/mccoef_params_"+actfunc_type+".npy", params)
