import numpy as np
from processes import g_process
from scipy.stats import norm

# define the function whose expected value we want to calculate
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

# define the Crude Monte Carlo method
def MC(mu, sigma, N):
    """
    Input:
        mu : mean of gaussian process, dim = (4,1)
        sigma : variance of gaussian process, dim = (4,4)
        N : number of gaussian processes

    Output:
        mean_est : float, estimated value of E[csi(ip)] with crude Monte Carlo
        std_est : float, estimated value of Std[csi(ip)] with Crude Monte Carlo
    """
    
    mu = mu.reshape(4, 1)
    ip = g_process(mu, sigma, n_process = N)
    
    est = csi(ip)
    mean_est = np.mean(est)
    std_est = np.std(est, ddof = 1)
    
    return mean_est, std_est


# define the 2stages Monte Carlo method
def MC_2stages(mu, sigma, tol, N0 = 100000):
    """
    Input:
        mu : mean of gaussian process, dim = (4,1)
        sigma : variance of gaussian process, dim = (4,4)
        tol: tolerance for confidence interval
        N0 : initial number of gaussian processes

    Output:
        N: final number of gaussian processes
        mean_est : float, estimated value of E[csi(ip)] with crude Monte Carlo
        std_est : float, estimated value of Std[csi(ip)] with Crude Monte Carlo
    """

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


# define the antithetic variables method
def MC_AV(mu, sigma, N):
    """
    Input:
        mu : mean of gaussian process, dim = (4,1)
        sigma : variance of gaussian process, dim = (4,4)
        N : number of gaussian processes

    Output:
        mean_est : float, estimated value of E[csi(ip)] with antithetic variables
        std_est : float, estimated value of Std[csi(ip)] with antithetic variables
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



