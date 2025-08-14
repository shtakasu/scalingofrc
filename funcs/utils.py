import numpy as np
try:
    import cupy as cp
except ImportError:
    import numpy as cp
from tqdm import tqdm
import scipy.stats as sps


def generate_correlated_J(N, g, eta):
    k_squared = (1-eta)/(1+eta)

    A = cp.random.normal(0, 1/cp.sqrt(N*(1+k_squared)), [N,N])
    Js = cp.triu(A, k=1) + cp.triu(A).T
    B = cp.random.normal(0, 1/cp.sqrt(N*(1+k_squared)), [N,N])
    Ja = cp.triu(B, k=1) - cp.triu(B, k=1).T
    J = Js + cp.sqrt(k_squared) * Ja

    return g*J

def mc_calc(g,N,act_func,max_leadout,sigma, connectivity, eta=0, trial=10, Tobs= 10000, Tinit=1000, max_delay=500,  sigma_noise=0.0, dinit=0):
    mcs = np.zeros([trial, max_leadout]) 
    
    for i in tqdm(range(trial)):

        if connectivity == "Gaussian":
            J = cp.random.normal(0, g/cp.sqrt(N), [N,N])
        elif connectivity == "Correlated":
            J = generate_correlated_J(N, g, eta)
        elif connectivity == "Cauchy":
            J = (g / N) * cp.random.standard_cauchy(size=(N, N))

        v = cp.random.normal(0,1,N)
        
        x = cp.zeros([N, Tinit+Tobs])
        inputs = sigma * cp.random.normal(0, 1, Tinit+Tobs)
        noise = sigma_noise * cp.random.normal(0, 1, [N,Tinit+Tobs])

        for t in range(Tinit+Tobs-1):
            x[:,t+1] = J@act_func(x[:,t]) + v*inputs[t+1] + noise[:, t+1]

    
        for l in range(1, max_leadout+1):
            kernel = np.linalg.inv((x[:l,Tinit:]@x[:l,Tinit:].T)/Tobs)
            
            eps = sps.chi2.ppf(q = 1-1e-4, df = l)*2/Tobs #threshold for Md
            
            memory = np.zeros(max_delay-dinit)
            for d in range(dinit,max_delay):
                a = np.dot(x[:l,Tinit:], inputs[Tinit-d : Tinit+Tobs-d])/Tobs
                md = np.dot(a, np.dot(kernel, a))/(sigma**2)
                memory[d] = md*(md>eps)
            mcs[i,l-1] = np.sum(memory) 
                
    return mcs

def memory_curve(g,N,act_func,leadout,sigma, connectivity, eta=0, trial=10, Tobs= 10000, Tinit=1000, max_delay=500, sigma_noise=0.0):
    
    memorys = np.zeros([trial, max_delay]) 
    
    for i in tqdm(range(trial)):
        if connectivity == "Gaussian":
            J = cp.random.normal(0, g/cp.sqrt(N), [N,N])
        elif connectivity == "Correlated":
            J =  generate_correlated_J(N, g, eta)
        elif connectivity == "Cauchy":
            J = (g / N) * cp.random.standard_cauchy(size=(N, N))
            
        v = cp.random.normal(0,1,N)

        x = cp.zeros([N,Tinit+Tobs])
        inputs = sigma * cp.random.normal(0, 1, Tinit+Tobs)
        noise = sigma_noise * cp.random.normal(0, 1, [N,Tinit+Tobs])

        for t in range(Tinit+Tobs-1):
            x[:,t+1] = J@act_func(x[:,t]) + v*inputs[t+1] + noise[:, t+1]
        
        kernel = np.linalg.inv((x[:leadout,Tinit:]@x[:leadout,Tinit:].T)/Tobs)
        
        eps = sps.chi2.ppf(q = 1-1e-4, df = leadout)*2/Tobs #threshold for Md
            
        for d in range(max_delay):
            a = np.dot(x[:leadout,Tinit:], inputs[Tinit-d : Tinit+Tobs-d])/Tobs
            md = np.dot(a, np.dot(kernel, a))/(sigma**2)
            memorys[i, d] = md*(md>eps)
        
    return memorys

def mean_coef(g, N, act_func, sigma, connectivity, eta=0, trial=10, Tobs= 10000, Tinit=1000, sigma_noise=0.0):
    
    coef_list = np.zeros([trial])

    for i in tqdm(range(trial)):
        if connectivity == "Gaussian":
            J = cp.random.normal(0, g/cp.sqrt(N), [N,N])
        elif connectivity == "Correlated":
            J = generate_correlated_J(N, g, eta)
        elif connectivity == "Cauchy":
            J = (g / N) * cp.random.standard_cauchy(size=(N, N))

        v = cp.random.normal(0,1,N)

        x = cp.zeros([N,Tinit+Tobs])
        inputs = sigma * cp.random.normal(0, 1, Tinit+Tobs)
        noise = sigma_noise * cp.random.normal(0, 1, [N,Tinit+Tobs])

        for t in range(Tinit+Tobs-1):
            x[:,t+1] = J@act_func(x[:,t]) + v*inputs[t+1] + noise[:, t+1]
        
        R = cp.corrcoef(x[:,Tinit:])
        R[range(N), range(N)] = cp.nan
        coef_list[i] = np.sqrt(cp.nanmean((R**2).reshape(-1))) 
    
    return coef_list

def mean_autocoef(g, N, act_func, sigma, max_delay, connectivity, eta=0, trial=10, Tobs= 10000, Tinit=1000, sigma_noise=0.0):
    
    autocoef= np.zeros([trial, max_delay+1])

    for i in tqdm(range(trial)):
        if connectivity == "Gaussian":
            J = cp.random.normal(0, g/cp.sqrt(N), [N,N])
        elif connectivity == "Correlated":
            J = generate_correlated_J(N, g, eta)
        elif connectivity == "Cauchy":
            J = (g / N) * cp.random.standard_cauchy(size=(N, N))

        v = cp.random.normal(0,1,N)

        x = cp.zeros([N,Tinit+Tobs])
        inputs = sigma * cp.random.normal(0, 1, Tinit+Tobs)
        noise = sigma_noise * cp.random.normal(0, 1, [N,Tinit+Tobs])

        for t in range(Tinit+Tobs-1):
            x[:,t+1] = J@act_func(x[:,t]) + v*inputs[t+1] + noise[:, t+1]
        
        for d in range(0, max_delay+1):
            squaredmean = cp.mean(x[:, Tinit:]**2, axis=1)
            autocoef[i, d] = cp.mean(cp.mean(x[:, Tinit:]*x[:,Tinit-d:Tinit+Tobs-d], axis=1) / squaredmean).get()
    
    return autocoef
        

def mcscaling_and_coef(N, actfunc_type, min_g, max_g, min_sigmas, max_sigmas, min_sigman, max_sigman, max_leadout=100, points=100, trial=10, Tobs= 10000, Tinit=1000, max_delay=500, dinit=0):
    #activation func
    if actfunc_type == "tanh":
        phi = cp.tanh
    elif actfunc_type == "relu":
        phi = lambda x: cp.maximum(0, x)
    elif actfunc_type == "linear":
        phi = lambda x: x
    
    memory_capacity = np.zeros([points, trial, max_leadout])
    coef = np.zeros([points, trial])
    params = np.zeros([points,3]) #save (g, sigma_s, sigma_n) for each point
 
    for point in range(points):
        # sample hyperparameters
        g = min_g + (max_g-min_g)*np.random.rand()
        sigma_s = min_sigmas + (max_sigmas-min_sigmas)*np.random.rand() 
        sigma_n = min_sigman + (max_sigman-min_sigman)*np.random.rand()    
        params[point, 0] = g; params[point, 1] = sigma_s; params[point, 2] = sigma_n
        
        for num in tqdm(range(trial)):
            # sample weights and inputs
            J = cp.random.normal(0, g/cp.sqrt(N), [N,N])
            v = cp.random.normal(0,1,N)
            x = cp.zeros([N, Tinit+Tobs])
            inputs = sigma_s * cp.random.normal(0, 1, Tinit+Tobs)
            noise = sigma_n * cp.random.normal(0, 1, [N,Tinit+Tobs])
            
            # run
            for t in range(Tinit+Tobs-1):
                x[:,t+1] = J@phi(x[:,t]) + v*inputs[t+1] + noise[:, t+1]
                
            # calc MC
            mcs = np.zeros(max_leadout)
            for l in range(1, max_leadout+1):
                kernel = np.linalg.inv((x[:l,Tinit:]@x[:l,Tinit:].T)/Tobs)
                
                eps = sps.chi2.ppf(q = 1-1e-4, df = l)*2/Tobs #threshold for Md
                
                memory = np.zeros(max_delay-dinit)
                for d in range(dinit,max_delay):
                    a = np.dot(x[:l,Tinit:], inputs[Tinit-d : Tinit+Tobs-d])/Tobs
                    md = np.dot(a, np.dot(kernel, a))/(sigma_s**2)
                    memory[d] = md*(md>eps)
                mcs[l-1] = np.sum(memory) 
            memory_capacity[point, num, :] = mcs
            
            # calc coef
            R = cp.corrcoef(x[:,Tinit:])
            R[range(N), range(N)] = cp.nan
            coef[point, num] = np.sqrt(cp.nanmean((R**2).reshape(-1)))
        
        print(f"Point {point+1}/{points} was done")
    
    return memory_capacity, coef, params
                

def calc_MLE(g, N, actfunc_type, sigma, sigma_noise, connectivity, eta=0, trial=10, Tobs= 10000, Tinterval=10, pertb_strength=1e-3):
    
    #activation func
    if actfunc_type == "tanh":
        phi = cp.tanh
    elif actfunc_type == "relu":
        phi = lambda x: cp.maximum(0, x)
    elif actfunc_type == "linear":
        phi = lambda x: x
        
    LE_trial = np.zeros(10)
    
    for i in tqdm(range(10)):
        
        if connectivity == "Gaussian":
            J = cp.random.normal(0, g/cp.sqrt(N), [N,N])
        elif connectivity == "Correlated":
            J = generate_correlated_J(N, g, eta)
        elif connectivity == "Cauchy":
            J = (g / N) * cp.random.standard_cauchy(size=(N, N))
            
        win = cp.random.normal(0,1,N)
        
        x1 = cp.zeros([N,Tobs])
        x2 = cp.zeros([N,Tobs])
        x1[:,0] = cp.random.normal(0,1,N)

        # Washout
        for t in range(10**3):
            x1[:,0] = J@phi(x1[:,0]) + cp.random.normal(0,sigma)*win + np.random.normal(0, sigma_noise,N)

        x2[:,0] = x1[:,0].copy() + cp.random.normal(0, 1, N)

        ratio = []
        # 時間発展
        for t in range(Tobs-1):
            
            if t%Tinterval == 0:
                delta = x2[:,t] - x1[:,t]
                perturbation = delta * pertb_strength/np.sqrt(np.mean(delta**2))
                x2[:,t] = x1[:,t] + perturbation
                
            indata = cp.random.normal(0,sigma)
            noise = cp.random.normal(0,sigma_noise,N)
            x1[:,t+1] = J@phi(x1[:,t]) + indata*win + noise
            x2[:,t+1] = J@phi(x2[:,t]) + indata*win + noise
            
            if t%Tinterval == Tinterval-1:
                delta = x1[:,t+1]-x2[:,t+1]
                growth_rate = np.mean(delta**2)/(pertb_strength**2)
                ratio.append(growth_rate)

        ratio = np.array(ratio)
        LE_trial[i] =  0.5*np.mean(np.log(ratio+1e-9)/Tinterval)

    LE_mean = np.mean(LE_trial)
    LE_std = np.std(LE_trial)
    
    return LE_mean, LE_std
