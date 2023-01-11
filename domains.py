import numpy as np
from kernels import kernel_exp, kernel_SE

# create Dataset for task 1
def create_Domain1(Nz, f_tilde, Ny):
    """
    Input:
        Nz : number of observations Z
        f_tilde : lambda function Z-->f_tilde_Z
        Ny : number of posterior points Y

    Output:
        Z : random uniform domain in [0, 2*pi], dim = (Nz, 1)
        f_tilde_Z : observations corresponding to the input Z, dim = (Nz, 1)
        Y : uniform domain in [0, 2*pi], dim = (Ny, 1)
    """
    
    Z = 2*np.pi*np.random.uniform(size = Nz)
    Z = np.sort(Z).reshape(-1, 1)
    f_tilde_Z = f_tilde(Z) 
    Y = np.linspace(0, 2*np.pi, Ny).reshape(-1, 1)
    
    return Z, f_tilde_Z, Y

# create datasets for task2
def create_Domain2(perm):
    """
    Input:
        perm : dataset, dim = (110,60)

    Output:
        Mesh_total : meshgrid of all domain, dim = (2x(60,110))
        Points_total : array of all 2D-points, dim = (6600,2)
    """
    
    Y = perm.shape[0]
    X = perm.shape[1]
    
    Mesh_total = np.meshgrid(np.arange(Y), np.arange(X))         
    Points_total = np.array([Mesh_total[0].T.flatten(), Mesh_total[1].T.flatten()]).T
    
    return Mesh_total, Points_total

def create_Dataset(perm, Coord):   
    """    
    Input:
        perm : dataset, dim = (110,60)
        Coord : locations for train , dim = (11, 2)

    Output:
        Mesh : meshgrid of Coord, dim = (2x(11,11)) 
        Values : perm evaluated in Mesh, dim = (121,)
        Points : array of points in Mesh, dim = (121,2)
    """
    
    Mesh = np.meshgrid(Coord[:,0], Coord[:,1])        
    Points = np.array([Mesh[0].T.flatten(), Mesh[1].T.flatten()]).T
    Values = perm[Points[:, 0], Points[:,1]]
    
    return Mesh, Values, Points

# define the case study for task1
def case_study1(KER, PTS, Z0, f_tilde_Z0, Z1, f_tilde_Z1, res_exp0, res_SE0, res_exp1, res_SE1):
    """
    Input:
        KER : index: 0 for ker_exp, 1 for ker_SE
        PTS : index: 0 for Nz = 10, 1 for Nz = 100
        Z0 : X_train, dim = (Nz,)
        f_tilde_Z0 : y_train, dim = (Nz,) 
        Z1 : X_train, dim = (Nz,)
        f_tilde_Z1 : y_train, dim = (Nz,) 
        res_exp0 : struct with optimal paramaters for ker_exp and Z0
        res_SE0 : struct with optimal paramaters for ker_SE and Z0
        res_exp1 : struct with optimal paramaters for ker_exp and Z1
        res_SE1 : struct with optimal paramaters for ker_SE and Z1

    Output:
        kernel : lambda functions, kernel of the case_study
        Z : X_train of the case study, dim = (Nz,)
        f_tilde_Z : y_train of the case_study, dim = (Nz,)
        l_opt : float, optimal length scale
        var_opt : float, optimal variance
        s_2_opt : float, optimal noise
    """
    
    if KER == 0:
        kernel = lambda x1, x2, l, var: kernel_exp(x1,x2,l,var)
        if PTS == 0:
            Z = Z0
            f_tilde_Z = f_tilde_Z0
            l_opt, var_opt, s_2_opt = res_exp0.x
        elif PTS == 1:
            Z = Z1
            f_tilde_Z = f_tilde_Z1
            l_opt, var_opt, s_2_opt = res_exp1.x
    elif KER == 1:
        kernel = lambda x1, x2, l, var: kernel_SE(x1,x2,l,var)
        if PTS == 0:
            Z = Z0
            f_tilde_Z = f_tilde_Z0
            l_opt, var_opt, s_2_opt = res_SE0.x
        elif PTS == 1:
            Z = Z1
            f_tilde_Z = f_tilde_Z1
            l_opt, var_opt, s_2_opt = res_SE1.x
    return kernel, Z, f_tilde_Z, l_opt, var_opt, s_2_opt


# define the case study for task2
def case_study2(KER, PTS, Points1, Values1, Points2, Values2, res1, res1_, res2, res2_):
    """
    Input:
        KER : index: 0 for ker_exp, 1 for ker_SE
        PTS : index: 1 for Nz = 10, 2 for Nz = 100
        Points1 : X_train, dim = (25,2)
        Values1 : y_train, dim = (25,) 
        Points2 : X_train, dim = (121,2)
        Values2 : y_train, dim = (121,) 
        res1 : struct with optimal paramaters for ker_exp and Points1
        res1_ : struct with optimal paramaters for ker_SE and Points1
        res2 : struct with optimal paramaters for ker_exp and Points2
        res2_ : struct with optimal paramaters for ker_SE and Points2

    Output:
        kernel : lambda functions, kernel of the case_study
        Points : X_train of the case study 
        Values : y_train of the case_study
        l_opt : list of float, optimal length scale (dim=1 for ker_exp, dim=2 for ker_SE)
        var_opt : float, optimal variance
        s_2_opt : float, optimal noise
    """

    if KER == 0:
        kernel = lambda x1, x2, l, var: kernel_exp(x1,x2,l,var)
        if PTS == 1:
            Points = Points1
            Values = Values1
            l_opt, var_opt, s_2_opt = res1.x
        elif PTS == 2:
            Points = Points2
            Values = Values2
            l_opt, var_opt, s_2_opt = res2.x
    elif KER == 1:
        kernel = lambda x1, x2, l, var: kernel_SE(x1,x2,l,var)
        if PTS == 1:
            Points = Points1
            Values = Values1
            l1_opt, l2_opt, var_opt, s_2_opt = res1_.x
        elif PTS == 2:
            Points = Points2
            Values = Values2
            l1_opt, l2_opt, var_opt, s_2_opt = res2_.x
        l_opt = np.array([l1_opt, l2_opt])
    
    return kernel, Points, Values, l_opt, var_opt, s_2_opt
