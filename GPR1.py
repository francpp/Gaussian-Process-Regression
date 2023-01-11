import numpy as np
from scipy.optimize import minimize
from kernels import kernel_exp, kernel_SE, evaluate_kernels
from optimizer import minimizer
from processes import prediction, g_process
from plots import plot_GPR_1D, plot_compare_kernels
from domains import create_Domain1, case_study1
np.random.seed(12345)

# define initial parameters 

NNz = [10, 100]
Ny = 1000
var = 0.1
s_2 = 0.01 
ll = [0.1, 0.3, 1.0]

f_tilde = lambda z: np.sin(z)

# do simulations for all parameters

for ker in range(2): 
    if ker == 0:
        kernel = lambda x1, x2, l, var: kernel_exp(x1,x2,l,var)
        type_ker = 'ker_exp'
    elif ker == 1:
        kernel = lambda x1, x2, l, var: kernel_SE(x1,x2,l,var)
        type_ker = 'ker_SE'
        
    for l in ll:
        for Nz in NNz:  
            
            #create domain
            Z, f_tilde_Z, Y = create_Domain1(Nz, f_tilde, Ny)
            
            #evaluate kernels
            K_Z, K_Y, K_ZY = evaluate_kernels(Z, Y, kernel, l, var)
            
            #computes conditional mean and variance of the process
            mu_YgivenZ, K_YgivenZ = prediction(f_tilde_Z, K_Z, K_Y, K_ZY, s_2)
            
            #samples again
            y_cond=g_process(mu_YgivenZ, K_YgivenZ, n_process = 3)
            
            #plot all
            plot_GPR_1D(Z, f_tilde_Z, Y, mu_YgivenZ, K_YgivenZ, y_cond, l, ker)
            
            if(Nz==100 and l==1):
                if(ker==0):
                    K_Y_exp = K_Y
                else:
                    K_Y_se = K_Y

# compare the two kernels
plot_compare_kernels(K_Y_exp, K_Y_se)
    
# Minimize THETA of the log-likelihood for Nz = 10
np.random.seed(3)

Z0, f_tilde_Z0, Y = create_Domain1(NNz[0], f_tilde, Ny)

res_exp0 = minimize(minimizer(Z0, f_tilde_Z0, 0, prob = 1), [1, 0.1, 0.01], 
               bounds=((0.01, None), (0, None), (0, None)),
               method='L-BFGS-B')

res_SE0 = minimize(minimizer(Z0, f_tilde_Z0, 1, prob = 1), [1, 0.1, 0.001],
                  bounds=((0, None), (0, None), (0, None)), tol = 1e-1,
                  method='L-BFGS-B') # occhio ai parametri iniziali


# Minimize THETA of the log-likelihood for Nz = 100

Z1, f_tilde_Z1, Y = create_Domain1(NNz[1], f_tilde, Ny)

res_exp1 = minimize(minimizer(Z1, f_tilde_Z1, 0, prob = 1), [1, 0.1, 0.01], 
               bounds=((0.01, None), (0, None), (0, None)),
               method='L-BFGS-B')

res_SE1 = minimize(minimizer(Z1, f_tilde_Z1, 1, prob = 1), [1, 0.1, 0.001],
                  bounds=((0, None), (0, None), (0, None)), tol = 1e-1,
                  method='L-BFGS-B') 

# Compute a run with the optimal parameters

for KER in range(0,2):
    for PTS in range(0,2):
        
        #define the case study 
        
        kernel, Z, f_tilde_Z, l_opt, var_opt, s_2_opt = case_study1(KER, PTS, 
                                                                     Z0, f_tilde_Z0, 
                                                                     Z1, f_tilde_Z1, 
                                                                     res_exp0, res_SE0, 
                                                                     res_exp1, res_SE1)
        
        print('KER=', KER, '; PTS=', PTS,'; l=', l_opt, '; var=', var_opt, '; s_2=', s_2_opt )
                
        #evaluate kernels
        K_Z, K_Y, K_ZY = evaluate_kernels(Z, Y, kernel, l_opt, var_opt)
        
        #computes conditional mean and variance of the process
        mu_YgivenZ, K_YgivenZ = prediction(f_tilde_Z, K_Z, K_Y, K_ZY, s_2_opt)
        
        #samples again
        y_cond=g_process(mu_YgivenZ, K_YgivenZ, n_process = 3)
        
        #plot all
        plot_GPR_1D(Z, f_tilde_Z, Y, mu_YgivenZ, K_YgivenZ, y_cond, l_opt, KER)


