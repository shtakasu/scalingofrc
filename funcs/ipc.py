import numpy as np
import math
from scipy.special import  eval_hermitenorm, eval_legendre
from tqdm import tqdm
from scipy.linalg import pinv
import scipy.stats as sps

def gen_degs(degsum,num):
    if num==1:
        return [[degsum]]
    gen_list = []
    for i in range(1,degsum-num+2):
        list_ = gen_degs(degsum-i,num-1)
        for j in range(len(list_)):
            list_[j].append(i)
        gen_list += list_
    return gen_list


# 最初が0、最後がwindowとなる合計num個の狭義単調増加列を全て列挙
# ただし num>=2、window >= num-1
def gen_delays(window,num):
    if num==2:
        return [[0,window]]
    gen_list = []
    for i in range(num-2,window):
        list_ = gen_delays(i,num-1)
        for j in range(len(list_)):
           list_[j].append(window)
        gen_list += list_
    return gen_list

# レザバーの信号列x（素子数 x 時間）、逆行列xx_inv、教師時系列z、カットオフ値epsとしたとき、capacity C_εを求める
def calc_Ceps(z,x,xx_inv,eps):
    zx = x@z/len(z)
    z2 = np.dot(z,z)/len(z)
    C_T = ((zx.T)@xx_inv@zx)/z2
    return C_T * (C_T > eps) 

# 正規化されたエルミート多項式の値を返す
def norm_hermite(deg, input):
    return  eval_hermitenorm(deg,input)/np.sqrt(math.factorial(deg))

def norm_Legendre(deg, input):
    """
    if deg==1:
        return np.sqrt(3/2)*input
    elif deg==2:
        return np.sqrt(5/8)*(3*(input**2)-1)
    elif deg==3:
        return np.sqrt(7/8)*(5*(input**3)-3*input)
    else:
        raise ValueError("higher degree is required")
    """
    return np.sqrt((2/(2*deg+1))) * eval_legendre(deg,input)

def generate_correlated_J(N, g, eta):
    k_squared = (1-eta)/(1+eta)

    A = np.random.normal(0, 1/np.sqrt(N*(1+k_squared)), [N,N])
    Js = np.triu(A, k=1) + np.triu(A).T
    B = np.random.normal(0, 1/np.sqrt(N*(1+k_squared)), [N,N])
    Ja = np.triu(B, k=1) - np.triu(B, k=1).T
    J = Js + np.sqrt(k_squared) * Ja

    return g*J

"""
IPCの計算
"""
def calc_ipc(N, g, input_intensity, max_leadout, degree_list, Tobs, Tinit, act_func, noise_std=0.0, connectivity="Gaussian", eta=0):

    # sample recurrent weight
    if connectivity == "Gaussian":
        J = np.random.normal(0, g/np.sqrt(N), [N,N])
    elif connectivity == "Correlated":
        J = generate_correlated_J(N, g, eta)
    elif connectivity == "Cauchy":
            J = (g / N) * sps.cauchy.rvs(size=(N, N))
        
    v = np.random.normal(0, 1, N) #入力結合重み

    L_list = np.arange(1,max_leadout+1) #leadout数のリスト
    
    ipc = np.zeros([len(L_list),len(degree_list)])

    x = np.zeros([N,Tinit+Tobs])
    x[:,0] = np.random.normal(0,1,N)
    input = np.random.normal(0, 1, Tinit+Tobs) #input = 2*np.random.rand(Tinit+Tobs) -1 #np.random.normal(0, 1, Tinit+Tobs)
    noise = np.random.normal(0,noise_std, [N,Tinit+Tobs])

    for t in tqdm(range(1, Tinit+Tobs)):
        x[:,t] = J@act_func(x[:,t-1]) + input[t-1]*v*input_intensity + noise[:,t]

    for idL in tqdm(range(len(L_list))):
        L = L_list[idL]
        eps = sps.chi2.ppf(q = 1-1e-4, df = L)*2/Tobs #閾値の決定

        cov = x[:L, Tinit:]@x[:L, Tinit:].T/Tobs
        x_inv  = np.linalg.inv(cov)
        
        ##SVD
        #x_inv  = pinv(x[:L, Tinit:])

        for deg_id in range(len(degree_list)):
            degree = degree_list[deg_id]
            C_deg = 0

            if degree>=7:
                MAX_WINDOW = 10
            else:
                MAX_WINDOW = 100

            for num_deg in range(1,degree+1):
                deg_list = gen_degs(degree, num_deg)
                jdg = 1
                C_numdeg = 0

                for degs in deg_list:
                    C_maxwindow = 0
                    for window in range(num_deg-1,MAX_WINDOW):
                        # num_degが1のとき、rel_delay_listが[[0]]で重複することを防ぐ
                        if jdg ==0:
                            break 
                        if num_deg >= 2:
                            rel_delay_list = gen_delays(window,num_deg)
                        else:
                            rel_delay_list = [[0]]
                            jdg -= 1

                        C_mindelay = 0
                        for rel_delay in rel_delay_list:
                            rel_delay = np.array(rel_delay)

                            for mindelay in range(1,10000):
                                delay = rel_delay + mindelay
                                
                                # 教師時系列z(t)を生成
                                z = np.ones(Tobs)
                                for n in range(num_deg):
                                    #z *= norm_Legendre(degs[n],input[Tinit-delay[n]:Tobs+Tinit-delay[n]])
                                    z *= norm_hermite(degs[n],input[Tinit-delay[n]:Tobs+Tinit-delay[n]])
                                z = z.reshape([Tobs,1])
                                zx = (x[:L, Tinit:]@z)/Tobs
                                C_T = (zx.T@(x_inv@zx))[0,0]/np.mean(z**2)
                                #w = x_inv.T@z
                                #zhat = x[:L, Tinit:].T@w                           
                                #C_T = 1-np.mean((z-zhat)**2)/np.mean(z**2)
                                Ceps = C_T*(C_T>eps)
                                """
                                print("num_deg={0},window={1},mindelay={2}".format(num_deg,window,mindelay))
                                print("=====================")
                                print("degs:")
                                print(degs)
                                print("delay")
                                print(delay)
                                print("C_eps:")
                                print(Ceps)
                                print("=====================")
                                """
                                C_mindelay += Ceps
                                if Ceps==0 and mindelay>5:
                                    break

                        C_maxwindow += C_mindelay
                        if C_mindelay == 0:
                            break

                    C_numdeg += C_maxwindow
                
                C_deg += C_numdeg

            ipc[idL,deg_id] = C_deg
    
    return ipc
