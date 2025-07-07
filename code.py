'''
This code is for the paper titled "A Model-Based Monitoring Framework for Tensor Count Data in Passenger Flow Surveillance".
'''

'''
Software: Python 3.10.9
Computer Hardware:  MacBook Pro with Apple M2 chip, featuring 8 cores (4 performance and 4 efficiency), 16 GB of RAM.
'''

import math
import time
import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.optimize import root
import matplotlib.pyplot as plt
import tensorly as tl
from scipy.linalg import sqrtm
from sklearn.utils.extmath import fast_logdet

'''
The code below implements the parameter estimation by VGA in Phase I.
This part implements the specific steps of Algorithm 1 as proposed in the paper. The IC sample size in Phase I is 2000. To estimate the variational parameters and model parameters in the Tensor Poisson Log-Normal (TPLN) model. We denote mu as the model mean and vec_m as the variational mean. The finalELBO corresponds to the ELBO defined in equation (7) of the paper.
'''

E1 = []
for w in range(d1):
    E1_w = np.zeros((d1, d1))
    E1_w[w, w] = 1
    E1.append(E1_w)
E2 = []
for w in range(d2):
    E2_w = np.zeros((d2, d2))
    E2_w[w, w] = 1
    E2.append(E2_w)
E3 = []
for w in range(d3):
    E3_w = np.zeros((d3, d3))
    E3_w[w, w] = 1
    E3.append(E3_w)

def tplnVGA(poiTTens,nT):

    m = [None]*nT
    vec_m = [None]*nT 
    for i in range(nT):
        m[i] = np.log(poiTTens[:, :, :, i]+1)
        vec_m[i] = m[i].flatten(order='F').reshape(-1, 1)
    mu = np.mean([vec_m[i] for i in range(nT)], axis=0)
    
    delta1 = [np.eye(d1)]*nT
    delta2 = [np.eye(d2)]*nT
    delta3 = [np.eye(d3)]*nT

    Sigma1 = np.eye(d1)
    Sigma2 = np.eye(d2)
    Sigma3 = np.eye(d3)
   
    S2 = [None]*nT
    two=[None]*nT
    it = 1
    check = 0
    elbo = []
    loglik = []
    aloglik = []
    aloglik[:3] = [0]*3
    itMax = 200
    eps = 0.0005
    
    w = [None]*nT
    for i in range(nT):
        w[i] = lmbda * (1 - lmbda)**(nT - i)
    w_sum = np.sum(w)

    while check == 0:
        
        iSigma1 = np.linalg.inv(Sigma1)
        iSigma2 = np.linalg.inv(Sigma2)
        iSigma3 = np.linalg.inv(Sigma3)
        S1 = np.kron(np.kron(Sigma3, Sigma2), Sigma1) 

        for i in range(nT):

            Omega = np.diag(np.array([np.dot(np.exp(vec_m[i].T + 0.5 * np.diag(np.kron(np.kron(delta3[i], delta2[i]), delta1[i])).reshape(1, -1)), np.diag(np.kron(np.kron(delta3[i], delta2[i]), E1[w])).reshape(-1, 1)) for w in range(d1)]).flatten())
            delta1[i] = d2 * d3 * np.linalg.inv(Omega + iSigma1 * np.trace(iSigma2 @ delta2[i]) * np.trace(iSigma3 @ delta3[i]))
            
            Omega = np.diag(np.array([np.dot(np.exp(vec_m[i].T + 0.5 * np.diag(np.kron(np.kron(delta3[i], delta2[i]), delta1[i])).reshape(1, -1)), np.diag(np.kron(np.kron(delta3[i], E2[w]), delta1[i])).reshape(-1, 1)) for w in range(d2)]).flatten())
            delta2[i] = d1 * d3 * np.linalg.inv(Omega + iSigma2 * np.trace(iSigma1 @ delta1[i]) * np.trace(iSigma3 @ delta3[i]))
            
            Omega = np.diag(np.array([np.dot(np.exp(vec_m[i].T + 0.5 * np.diag(np.kron(np.kron(delta3[i], delta2[i]), delta1[i])).reshape(1, -1)), np.diag(np.kron(np.kron(E3[w], delta2[i]), delta1[i])).reshape(-1, 1)) for w in range(d3)]).flatten())
            delta3[i] = d1 * d2 * np.linalg.inv(Omega + iSigma3 * np.trace(iSigma1 @ delta1[i]) * np.trace(iSigma2 @ delta2[i]))
            
            S1 = np.kron(np.kron(Sigma3, Sigma2), Sigma1)   
            S2[i] = np.kron(np.kron(delta3[i], delta2[i]), delta1[i])
            
            vec_m[i] = vec_m[i] - np.linalg.inv(np.diag(np.exp(vec_m[i] + 0.5 * np.diag(S2[i]).reshape(-1,1)).flatten())+np.linalg.inv(S1)) @ (np.exp(vec_m[i] + 0.5 * np.diag(S2[i]).reshape(-1, 1)) + np.linalg.inv(S1)@(vec_m[i]-mu)-vec_dataset[i]) 
        mu = np.sum([w[i]*vec_m[i] for i in range(nT)],axis=0)/w_sum 
        
        tensor_vec_m = [vec_m[i].reshape((d1,d2,d3),order='F') for i in range(nT)] 
        tensor_mu =  mu.reshape((d1,d2,d3),order='F') 
        
        Sigma1 = np.sum([w[i]*delta1[i]*np.trace(iSigma2@delta2[i])*np.trace(iSigma3@delta3[i]) + w[i]*Matrixing(tensor_vec_m[i]-tensor_mu,0)@np.linalg.inv(np.kron(Sigma3,Sigma2))@(Matrixing(tensor_vec_m[i]-tensor_mu,0)).T for i in range(nT)], axis=0)/(d2*d3*w_sum)
        iSigma1 = np.linalg.inv(Sigma1)
        Sigma2 = np.sum([w[i]*delta2[i]*np.trace(iSigma1@delta1[i])*np.trace(iSigma3@delta3[i]) + w[i]*Matrixing(tensor_vec_m[i]-tensor_mu,1)@np.linalg.inv(np.kron(Sigma3,Sigma1))@(Matrixing(tensor_vec_m[i]-tensor_mu,1)).T for i in range(nT)], axis=0)/(d1*d3*w_sum)
        iSigma2 = np.linalg.inv(Sigma2)
        Sigma3 = np.sum([w[i]*delta3[i]*np.trace(iSigma1@delta1[i])*np.trace(iSigma2@delta2[i]) + w[i]*Matrixing(tensor_vec_m[i]-tensor_mu,2)@np.linalg.inv(np.kron(Sigma2,Sigma1))@(Matrixing(tensor_vec_m[i]-tensor_mu,2)).T for i in range(nT)], axis=0)/(d1*d2*w_sum)
        
        Sigma1=Sigma1/Sigma1[0,0]
        Sigma2=Sigma2/Sigma2[0,0]
        Sigma3=Sigma3
        one = d2*d3*fast_logdet(Sigma1) + d1*d3*fast_logdet(Sigma2) + d1*d2 *fast_logdet(Sigma3)
        for i in range(nT):
            two[i] = d2*d3 *fast_logdet(delta1[i]) + d1*d3 * fast_logdet(delta2[i]) + d1*d2 * fast_logdet(delta3[i])   
        
        iSigma1 = np.linalg.inv(Sigma1)
        iSigma2 = np.linalg.inv(Sigma2)
        iSigma3 = np.linalg.inv(Sigma3)
        iS1=np.linalg.inv(S1)
        for i in range(nT):
            elbo_i = w[i]*(vec_m[i].reshape(1, -1) @ vec_dataset[i].reshape(-1, 1) - np.sum(gammaln((vec_dataset[i]) + 1))-np.sum(np.exp(vec_m[i].reshape(-1, 1) + 0.5 * np.diag(S2[i]).reshape(-1, 1))) - 0.5*one + 0.5*two[i] -0.5*((vec_m[i]-mu).T)@iS1@(vec_m[i]-mu) - 0.5*np.trace(iSigma1@delta1[i])*np.trace(iSigma2@delta2[i])*np.trace(iSigma3@delta3[i]) + 0.5*q)  
            elbo.append(elbo_i)  

        elbo_D = np.sum(elbo[-nT:], axis=0)  
        loglik.append(elbo_D)  

        if it > 3:
            if (loglik[-2] - loglik[-3]) == 0:
                check = 1
            else:
                a = (loglik[-1] - loglik[-2]) / (loglik[-2] - loglik[-3])
                aa = (loglik[-1] - loglik[-2])/ (1 - a)
                aloglik.append(loglik[-2] + aa)
                if abs(aloglik[-1] - aloglik[-2]) <= eps:
                    check = 1
                else:
                        check = check

        it = it+1

        if it == itMax:
            check = 1
            print("No convergence")

    Results = {
        'vec_Mu':mu,
        'Mu': tensor_mu,
        'Sigma1': Sigma1,
        'Sigma2': Sigma2,
        'Sigma3': Sigma3,
        'S1':S1,
        'finalELBO':loglik[-1],
        'variationalMu':vec_m,
        'delta1':delta1,
        'delta2':delta2,
        'delta3':delta3,
        'S2':S2
    }

    return Results

'''
 The code below implements the Laplace approximation approach in Phase II.
 This part implements the detailed steps of Algorithm 2 in the paper. The output of the function H0_laplace corresponds to the monitoring statistic defined in equation (14) in the paper. The in-control average run length is uniformly set to 200 across all settings.

'''

def H0_laplace(poiTTens,nT):
    
    loglik0=[]
    loglik=[]
    vec_dataset1 = [None]*nT
    for i in range(nT):
        vec_dataset1[i] = poiTTens[:, :, :, i].flatten(order='F')
        
    w = [None]*nT
    for i in range(nT):
        w[i] = lmbda * (1 - lmbda)**(nT - i)
    w_sum = np.sum(w)
        
    vec_m = [None]*nT 
    vec_m0 = [None]*nT
    m=[None]*nT
    mu = np.zeros([q]) 
    for i in range(nT):
       m[i] = np.zeros([q]) 
    
    for i in range(nT):     
        def g(x):
            g = np.exp(x)-vec_dataset1[i] + np.dot(iS10,x-mu)
            return(g)
        vec_m[i] = root(g, m[i]).x
    mu = (np.sum([w[i]*vec_m[i] for i in range(nT)],axis=0)) / w_sum 

    y = [None]*nT
    for i in range(nT):
        y[i]=vec_m[i].reshape((d1,d2,d3),order='F')
    
    for i in range(nT):     
        def g(x):
            g = np.exp(x)-vec_dataset1[i] + np.dot(iS10,x-Mu0.flatten(order='F'))
            return(g)
        vec_m0[i] = root(g, m[i]).x
 
    y0 = [None]*nT
    for i in range(nT):
        y0[i]=vec_m0[i].reshape((d1,d2,d3),order='F')
    
    one = 0.5*d2*d3*fast_logdet(Sigma10)+ 0.5*d1*d3*fast_logdet(Sigma20)+ 0.5*d1*d2*fast_logdet(Sigma30) 
    
    for i in range(nT): 
        loglik0_i = w[i]*(-0.5*np.dot(np.dot((vec_m[i]-Mu0).T,iS10),vec_m[i]-Mu0) - np.sum(np.exp(y0[i]))+np.sum(np.multiply(poiTTens[:,:,:,i],y0[i]))-one - np.sum(gammaln(poiTTens[:,:,:,i] + 1))-0.5*np.log(np.det(-(np.diag(np.exp(vec_m0[i]))-iS10))))
        loglik0.append(loglik0_i) 
    
    for i in range(nT): 
        loglik_i = w[i]*(-0.5*np.dot(np.dot((vec_m[i]-mu).T,iS10),vec_m[i]-mu) - np.sum(np.exp(y[i]))+np.sum(np.multiply(poiTTens[:,:,:,i],y[i]))-one - np.sum(gammaln(poiTTens[:,:,:,i] + 1))-0.5*np.log(np.det(-(np.diag(np.exp(vec_m[i]))-iS10))))
        loglik.append(loglik_i)
        
    Results = {'H1_laplace' : np.sum(loglik),
                'H0_laplace' : np.sum(loglik0)}

    return Results

''' 
find ARL0 with moving windows
'''

def ARL0_la(k,Sigma1, Sigma2, Sigma3, Mu, Mu1, h, M, m):
    
    rl0 = np.zeros([M])
    for i in range(M):
        j = 11
        W = 0
        all_poiTTens = generate_poiTTens(Sigma1,Sigma2,Sigma3, Mu1, 1010)
        while W < h and j <= 1000:
            
            poiTTens = all_poiTTens[:, :, :, j-m:j]
            res_laplace = H0_laplace(poiTTens,m)
            
            #WLRT
            W = res_laplace['H1_laplace'] - res_laplace['H0_laplace']
            j +=1
            
        rl0[i] = j-11
        print('当前的arl=',np.sum(rl0)/np.sum(rl0>0))
    arl0 = np.round(np.mean(rl0),2)
    sdrl0 = np.round(np.std(rl0),2)
    a0 = [arl0, sdrl0]
    q_10 = int(np.percentile(rl0,25))
    q_50 = int(np.percentile(rl0,50))
    q_90 = int(np.percentile(rl0,75))
    b = [q_10,q_50,q_90]

    return a0,b,rl0

'''
The code below generates the count tensor data and the simulated data.
Sigma1, Sigma2, Sigma3: Covariance matrices for each mode.
tensors size is (d1, d2, d3). nT is Number of tensor samples to generate.
'''

def mode_n_product(x, m, mode):
    x = np.asarray(x)
    m = np.asarray(m)
    if mode <= 0 or mode % 1 != 0:
        raise ValueError('`mode` must be a positive interger')
    if x.ndim < mode:
        raise ValueError('Invalid shape of X for mode = {}: {}'.format(mode, x.shape))
    if m.ndim != 2:
        raise ValueError('Invalid shape of M: {}'.format(m.shape))
    return np.swapaxes(np.swapaxes(x, mode - 1, -1).dot(m.T), mode - 1, -1)

def Matrixing(tensor,mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1),order="F")

def generate_poiTTens(Sigma1,Sigma2,Sigma3,Mu,nT):
   
    Tens = np.random.normal(0, 1, size=(d1, d2, d3, nT))
    TTens = mode_n_product(Tens, sqrtm(Sigma1), mode=1)
    TTens = mode_n_product(TTens, sqrtm(Sigma2), mode=2)
    TTens = mode_n_product(TTens, sqrtm(Sigma3), mode=3)
    TTens = (TTens.transpose(3, 0, 1, 2) + Mu).transpose(1, 2, 3, 0)
    eTTens = np.exp(TTens) 
    poiTTens = np.random.poisson(lam=eTTens)
    
    return poiTTens

d1 = 5
d2 = 5
d3 = 5
q = d1 * d2 * d3
lmbda = 0.05

Mu = np.zeros([d1,d2,d3])
Mu0 = np.zeros([d1,d2, d3]).flatten()

Sigma1 = np.eye(d1) 
Sigma2 = np.eye(d2)
Sigma3 = np.eye(d3)
S1 = np.kron(np.kron(Sigma3, Sigma2), Sigma1) 

Sigma10=Sigma1
Sigma20=Sigma2
Sigma30=Sigma3
one0=d2*d3*fast_logdet(Sigma10) + d1*d3*fast_logdet(Sigma20) + d1*d2 *fast_logdet(Sigma30)
iSigma10 = np.linalg.inv(Sigma10)
iSigma20 = np.linalg.inv(Sigma20)
iSigma30 = np.linalg.inv(Sigma30)
S10 = np.kron(np.kron(Sigma30, Sigma20), Sigma10) 
iS10=np.linalg.inv(S10)

poiTTens = generate_poiTTens(Sigma1,Sigma2,Sigma3,Mu,2000)


# OC performence for Mu
shift = [-5,-4,-3,-2,-1,0.2,0.4,0.6,0.8,1] 
Mu1 = [np.copy(Mu) for i in range(len(shift))]
for i in range(len(shift)):
    Mu1[i][0,0,0] = Mu1[i][0,0,0] + shift[i]
    Mu1[i][1,1,1] = Mu1[i][1,1,1] + shift[i]
    Mu1[i][2,2,2] = Mu1[i][2,2,2] + shift[i]
    Mu1[i][3,3,3] = Mu1[i][3,3,3] + shift[i]
    Mu1[i][4,4,4] = Mu1[i][4,4,4] + shift[i]

s = np.zeros((2 * len(shift), 1))
for k in range(len(shift)):
    A = ARL0_la(k,Sigma1, Sigma2, Sigma3, Mu, Mu1[k],3.265, 1000,10)
    s[(2*k)] = A[0][0]
    s[(2*k)+1] = A[0][1]

'''
The code below is the parameter setting for the real case.

'''
txn_loc_map =  {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '16': 6, '17': 7, '18': 8, '19': 9, '20': 10} 
travel_time_map =  {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4, 6.0: 5, 7.0: 6, 8.0: 7, 9.0: 8, 10.0: 9, 11.0: 10, 12.0: 11, 13.0: 12, 14.0: 13, 15.0: 14, 16.0: 15} 
txn_type_co_map = txn_type_co_map = {'ENT': 0, 'USE': 1} 
hour_map = {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14, 21: 15, 22: 16, 23: 17} 
d1 = len(txn_loc_map) 
d2 = len(travel_time_map) 
d3 = len(txn_type_co_map)
q = d1 * d2 * d3
lmbda = 0.2
