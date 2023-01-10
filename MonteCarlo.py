import numpy as np
from processes import g_process
from scipy.stats import norm

def csi(ip):
    """
    Input:
        ip : 4-dimensional gaussian processes, dim = (4, Np)

    Output:      
        cnt : csi evaluated for each process, dim = (Np,)

    """
    cnt = np.count_nonzero(ip>8, axis=0)
    cnt[cnt>0]=1
    
    return cnt


def MC(mu, sigma, N):
    """
    Input:
        mu : mean of gaussian process, dim = (4,1)
        sigma : variance of gaussian process, dim = (4,4)
        N : number of gaussian processes

    Output:
        mean_est : float, estimated value of E[csi(ip)] with crude Monte Carlo
        var_est : float, estimated value of Var[csi(ip)] with Crude Monte Carlo
    """
    
    mu = mu.reshape(4, 1)
    ip = g_process(mu, sigma, n_process = N)
    
    est = csi(ip)
    mean_est = np.mean(est)
    std_est = np.std(est, ddof = 1)
    
    return mean_est, std_est

def MC_2stages(mu, sigma, tol, N0 = 100000):
    # per ker_exp
    alpha = 0.05
    c = norm.ppf(1-alpha/2) #1.96
    N_old = N0
    mean_est_, std_est_ = MC(mu, sigma, N_old)
    N = N_old
    if(std_est_!=0):        
        N = int((c*std_est_/tol)**2)
    mean_est, std_est = MC(mu, sigma, N)
    while std_est >= std_est_:
        N_old = N
        mean_est_, std_est_ = MC(mu, sigma, N_old)
        if std_est_ != 0:            
            N = int((c*std_est_/tol)**2)
        mean_est, std_est = MC(mu, sigma, N)
    return N, mean_est, std_est

def MC_seq(mu, sigma, tol):
    # per ker_SE
    alpha = 0.05
    c = norm.ppf(1-alpha/2) #1.96
    N = 1000000
    mean_est, std_est = MC(mu, sigma, N)
    while c*std_est/np.sqrt(N) > tol:
        mean_old = mean_est
        std_old = std_est
        Z = csi(g_process(mu.reshape(4,1), sigma))
        mean_est = (mean_old*N+Z)/(N+1)
        std_est = np.sqrt((N-1)/N*std_old**2+(Z-mean_old)**2/(N+1))
        N = N+1
    return N, mean_est, std_est

def MC_AV(mu, sigma, N):
    """
    Input:
        mu : mean of gaussian process, dim = (4,1)
        sigma : variance of gaussian process, dim = (4,4)
        N : number of gaussian processes

    Output:
        mean_est : float, estimated value of E[csi(ip)] with antithetic variables
        var_est : float, estimated value of Var[csi(ip)] with antithetic variables

    """

    mu = mu.reshape(4, 1)
    N2 = int(N/2)
    X = g_process(mu, sigma, n_process = N2)
    Xa = 2*mu - X
    Z = csi(X)
    Za = csi(Xa)
    
    est = (Z + Za)/2
    mean_est = np.mean(est)
    std_est = np.std(est, ddof=1)
    
    return mean_est, std_est


def MC_IS(mu, sigma, N):
    """
    Input:
        mu : mean of gaussian process, dim = (4,1)
        sigma : variance of gaussian process, dim = (4,4)
        N : number of gaussian processes

    Output:
        mean_est : float, estimated value of E[csi(ip)] with importance sampling
        var_est : float, estimated value of Var[csi(ip)] with importance sampling

    """

    gauss_func = lambda x, m, s: np.exp(-0.5*np.sum((x-m)*np.linalg.solve(s,x-m), axis=0))
    # perform sampling   
    mu = mu.reshape(4,1)
    mu_s = np.array([[8,8,8,8]]).reshape(4,1)
    X = g_process(mu_s, sigma, n_process = N)
    
    g = gauss_func(X, mu_s, sigma)
    f = gauss_func(X, mu, sigma)
    likelihood_ratio = f/g
    
    Z = csi(X)
    
    est = Z*likelihood_ratio
    
    mean_est = np.mean(est)
    var_est = np.var(est)
    
    return mean_est, var_est

