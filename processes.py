import numpy as np
from scipy.fftpack import fft2, ifft2

# predict mean and covariance of the conditioned gaussian process
def prediction(f, K_Z, K_Y, K_ZY, s_2):
    """
                 ___Nz_____Ny___
               |       |       |
               | K_Z   | K_ZY  | Nz (train)
    Cov(Z,Y) = |_______|_______|  
               |       |       |
               | K_YZ  | K_Y   | Ny (test)
               |_______|_______|
               
    Input:
        f: observed values, dim = (Nz,1)
        K_Z: kernel of train points Z, dim = (Nz,Nz)
        K_Y: kernel of test points Y, dim = (Ny,Ny)
        K_ZY: mixed kernel of Z and Y, dim = (Nz,Ny)
               
    Output:
        mu_pred: fitted conditioned mean, dim = (Ny,1)
        cov_pred: fitted conditioned covariance, dim = (Ny,Ny)
    """

    I = s_2*np.eye(K_Z.shape[0])
    L = np.linalg.cholesky(K_Z+I)
    mu_pred = K_ZY.T@(np.linalg.solve(L.T, np.linalg.solve(L, f)))
    cov_pred = K_Y - K_ZY.T@(np.linalg.solve(L.T, np.linalg.solve(L, K_ZY)))
    
    return mu_pred, cov_pred

# generate different gaussian processes
def g_process(mean, sigma, n_process = 1):
    """
    Input:
        mean: mean of multivariate gaussian, dim = (Ny,1)
        sigma: covariance of multivariate gaussian, dim = (Ny,Ny)
        n_process: number of process to fit
               
    Output:
        proc: fitted processes, dim = (Ny, n_process)
    """
    
    U,D,_ = np.linalg.svd(sigma)
    A = U@np.diag(np.sqrt(D))
    Ny = sigma.shape[0]
    proc = mean + A@np.random.standard_normal(size = (Ny,n_process))
    
    return proc


# Circulant Embedding in 2D
def build_P(sigma):
    """
    Input:
        sigma: covariance matrix dim=(6600,6600)

    Output:
        P: 2D-Circular matrix, dim=(119,219)
    """
    
    upper_left_P = np.empty((60,110))
    for j in range(6600):
        delta_y = int(j/60)
        delta_x = int(j%60)
        
        upper_left_P[delta_x, delta_y] = sigma[0,j]
    
    upper_right_P = np.flip(upper_left_P[:,1:], axis=1)
    upper_P = np.c_[upper_left_P, upper_right_P]
    lower_P = np.flip(upper_P[1:,:], axis=0)
    
    P = np.r_[upper_P, lower_P]
    
    return P


def CE(sigma, n_process):
    """
    Input:        
        sigma : covariance matrix, dim = (6600,6600)
        n_process : number of gaussian processes to fit

    Output:        
        gauss_process : zero-mean gaussian processes, dim=(6600,n_process)
    """
    
    P = build_P(sigma)
    W = fft2(P)
    gauss_process = np.zeros((6600, n_process))
    
    for i in range(n_process):
        Xi = np.random.standard_normal(size=W.shape)
        T = 2*np.sqrt(59*109*W)*Xi
        Z = ifft2(T)
        Z_prime = Z.real + Z.imag
        
        sample = (Z_prime[:60, :110]).T.flatten()            
        gauss_process[:,i] = sample
        
    return gauss_process
    
# condition the zero-mean gaussian processes to the observations  
def conditional(K_ZY, K_Z, K_Y, Points, Values, n_process):
    """
    Input:
        K_ZY: mixed kernel of Z and Y, dim = (Nz,Ny)
        K_Z: kernel of train points Z, dim = (Nz,Nz)
        K_Y: kernel of test points Y, dim = (Ny,Ny)
        Values : perm evaluated in Mesh, dim = (Nz,)
        Points : array of points in Mesh, dim = (Nz,2)
        n_process : number of processes to fit
    
    Output:
        post : conditioned gaussian processes, dim = (6600, n_process)
    """
    
    index = 60*Points[:,0]+Points[:,1] #indexes of Z
    precond = CE(K_Y, n_process)
    z_iniz = precond[index]
    post = precond + K_ZY.T@np.linalg.solve(K_Z, Values.reshape(-1,1)-z_iniz)
    
    return post

            
