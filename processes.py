import numpy as np
from scipy.fftpack import fft2, ifft2


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

# generate thanks to Circular Embedding
def build_P(sigma):
    upper_left_P = np.empty((110,60))
    for j in range(6600):
        # switch 110 with 60
        delta_y = int(j/60)
        delta_x = int(j%60)
        
        upper_left_P[delta_y, delta_x] = sigma[0,j]
    
    upper_right_P = np.flip(upper_left_P[:,1:], axis=1)
    upper_P = np.c_[upper_left_P, upper_right_P]
    lower_P = np.flip(upper_P[1:,:], axis=0)
    
    P = np.r_[upper_P, lower_P]
    
    return P


# Circulant Embedding in 2D
def CE(sigma, n_process, naive):
    """
    Input:        
        sigma : covariance matrix, dim = (6600,6600)
        n_process : number of gaussian processes to fit

    Output:        
        gauss_process : fitted gaussian processes, dim = (6600, n_process)
    """
    
    if (naive):
        U,D,_ = np.linalg.svd(sigma)
        A = U@np.diag(np.sqrt(D))
        Ny = sigma.shape[0]
        gauss_process = A@np.random.standard_normal(size = (Ny,n_process))
    
    else:       
        # Following the algorithm
        
        # Build the matrix P
        # We should also add the option of the padding to make W positive definite
        P = build_P(sigma)
        
        # Calculate the 2D FFT of P
        W = fft2(P)
        # print(np.all(linalg.eigvals(np.conjugate(W).T.dot(W)) > 0))
        
        gauss_process = np.zeros((6600, n_process))
        
        for i in range(n_process):
            
            # Generate iid standard normal random variables and put them in Xi
            Xi = np.random.standard_normal(size=W.shape)
            
            # Build T
            T = np.sqrt(W)*Xi
            
            # Calculate the 2D iFFT of T
            Z = ifft2(T)
            
            # Extract the real and imaginary part of Z and sum them
            Z_prime = Z.real + Z.imag
            
            sample = (Z_prime[:60, :110]).T.flatten()            
            gauss_process[:,i] = sample
        
    return gauss_process
    
    
def conditional(mean_perm, cov_perm, K_ZY, K_Z, K_Y, Points, Values, n_process):
    index = 60*Points[:,0]+Points[:,1] #indexes of Z
    precond = CE(K_Y, n_process, naive = False)
    z_iniz = precond[index]
    post = precond + K_ZY.T@np.linalg.solve(K_Z, Values.reshape(-1,1)-z_iniz)
    
    #print(np.mean(np.abs(np.diag(np.cov(precond)-cov_perm))))
    #print(np.mean(np.abs(np.cov(post)-cov_perm))/np.mean(np.abs(cov_perm)))
    
    return post

            
