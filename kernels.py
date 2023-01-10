import numpy as np

def dist_2(X1, X2):
    """
    Input:
        X1: array of m points (m x d)
        X2: array of n points (n x d)
    Output:
        distances: (m x n matrix), where distances[i, j] = dist2(X1[i], X2[j])
    """
    
    assert X1.ndim <= 2
    assert X2.ndim <= 2
    
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1) 
        
    assert X1.shape[1] == X2.shape[1]
    
    rows, cols = np.indices((X1.shape[0], X2.shape[0]))
    distances = np.sqrt(np.sum((X1[rows, :] - X2[cols, :])**2, axis=2))
    return distances

def dist_w(X1, X2, l):
    """
    Input:
        X1: array of m points (m x d)
        X2: array of n points (n x d)
        l: array of d length scales (d,)
    Output:
        distances: (m x n matrix), where 
                    distances[i, j] = dist_w(X1[i], X2[j]) 
    """
    assert X1.ndim <= 2
    assert X2.ndim <= 2
    
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1) 
        
    assert X1.shape[1] == X2.shape[1]
    
    #print(type(l))
    #if (type(l)!=int and type(l)!=float and type(l)!= float64):    
    #    assert l.shape[0] == X1.shape[1]
    
    rows, cols = np.indices((X1.shape[0], X2.shape[0]))
    distances = np.sum(((X1[rows, :] - X2[cols, :])/l)**2, axis=2)
    return distances


def kernel_exp(X1, X2, l=1.0, var=1.0):
    """
    Exponential (Exp) kernel.
    
    Input:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).
        l: length scale
        var: Variance of the diagonal
    Output:
        kernel: (m x n) matrix, where kernel[i, j] = K[X1(i), X2(j)]
    """
    
    distances = dist_2(X1, X2)
    return var * np.exp(-distances/l)

def kernel_SE(X1, X2, l=1.0, var=1.0):
    """
    Squared Exponential (SE) kernel.
    
    Input:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).
        l: Array of d length scales (d,)
        var: Variance of the diagonal
    Output:
        kernel: (m x n) matrix, where kernel[i, j] = K[X1(i), X2(j)]
    """
    
    distances = dist_w(X1, X2, l)
    return var * np.exp(-0.5*distances)

def evaluate_kernels(Z, Y, kernel, l, var):
    """
                 ___Nz_____Ny___
               |       |       |
               | K_Z   | K_ZY  | Nz (train)
    Cov(Z,Y) = |_______|_______|  
               |       |       |
               | K_YZ  | K_Y   | Ny (test)
               |_______|_______|
    
    Input:
        Z: Array of m points (Nz x d).
        Y: Array of n points (Ny x d).
        kernel: lambda function of the kernel
        l: Array of 1 or d length scales (d,)
        var: Variance of the diagonal

    Output:
        K_Z: (Nz x Nz) matrix, where K_Z[i, j] = K[Z(i), Z(j)]
        K_Y: (Ny x Ny) matrix, where K_Y[i, j] = K[Y(i), Y(j)]
        K_ZY: (Nz x Ny) matrix, where K_ZY[i, j] = K[Z(i), Y(j)]
    """
    
    K_Z = kernel(Z,Z,l,var)   
    K_Y = kernel(Y,Y,l,var)       
    K_ZY = kernel(Z,Y,l,var)   
    return K_Z, K_Y, K_ZY
