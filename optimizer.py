import numpy as np
from kernels import kernel_exp, kernel_SE
from scipy.linalg import solve_triangular

# define the minimizer for GPR-hyperparameters
def minimizer(X, Y, ker, prob): 
    """
    Input:
        X: array of Points dim=(Nz,2).
        Y: observations at Points dim=(Nz,1).
        ker: type of kernel (0 or 1)
        prob: number of problem (1 or 2)
               
    Output:
        log_likelihood: function to minimize wrt theta
    """
    
    if ker == 0:
        kernel = lambda x1, x2, l, var: kernel_exp(x1,x2,l,var)
    elif ker == 1:
        if prob == 1:
            kernel = lambda x1, x2, l, var: kernel_SE(x1,x2,l,var)
        elif prob == 2:
            kernel = lambda x1, x2, l1, l2, var: kernel_SE(x1,x2,np.array([l1,l2]),var)
    
    Y = Y.flatten()
        
    def log_likelihood(theta):  
        """
        Input:
            theta: list of hyperparameters (l, var, s_2)
                   
        Output:
            logL: log_likelihood wrt theta
        """
        
        if (prob == 2 and ker == 1):
            K = kernel(X, X, l1=theta[0], l2=theta[1], var=theta[2]) + theta[3] * np.eye(len(X))
        else:            
            K = kernel(X, X, l=theta[0], var=theta[1]) + theta[2] * np.eye(len(X))
                
        L = np.linalg.cholesky(K)       
        U = solve_triangular(L, Y, lower=True)
        B = solve_triangular(L.T, U, lower=False)       
        logL = np.sum(np.log(np.diagonal(L))) + 0.5 * Y.dot(B) 
        
        return logL 
    
    return log_likelihood

