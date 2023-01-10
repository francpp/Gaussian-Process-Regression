import numpy as np
from scipy.optimize import minimize
from kernels import evaluate_kernels
from optimizer import minimizer 
from processes import conditional, prediction
from plots import plot_compare_three, plot_MC_AV, plot_CE
from domains import create_Domain2, create_Dataset, case_study2
from MonteCarlo import MC_2stages
np.random.seed(12345)


# load files and extract values
# Note that all the dimensions in the docstring are relative to the second case study

perm = np.load('true_perm.npy')

Coord1 = (np.array([np.linspace(15, 95, 5), np.linspace(10, 50, 5)]).T).astype(int)
Coord2 = (np.array([np.linspace(5, 105, 11), np.linspace(5, 55, 11)]).T).astype(int)

Mesh_total, Points_total = create_Domain2(perm)
Mesh1, Values1, Points1 = create_Dataset(perm, Coord1)
Mesh2, Values2, Points2 = create_Dataset(perm, Coord2)

# Minimize the log-likelihood for Points1

res1_exp = minimize(minimizer(Points1, Values1, ker = 0, prob=2), 
                x0 = [1, 0.1, 0.1], 
                bounds=((1e-10, None), (1e-10, None), (1e-10, None)),
                method='L-BFGS-B')

res1_se = minimize(minimizer(Points1, Values1, ker = 1, prob=2), 
                 x0 = [5., 8, 1., 1.], 
                 bounds=((1e-10, None), (1e-10, None), (1e-10, None),(1e-10, None)),
                 method='L-BFGS-B')

# Minimize the log-likelihood for Points2

res2_exp = minimize(minimizer(Points2, Values2, ker = 0, prob=2), 
                x0 = [1, 0.1, 0.1], 
                bounds=((1e-10, None), (1e-10, None), (1e-10, None)),
                method='L-BFGS-B')

res2_se = minimize(minimizer(Points2, Values2, ker = 1, prob=2),
                 x0 = [5., 10., 1., 1.], 
                 bounds=((1e-10, None), (1e-10, None), (1e-10, None),(1e-10, None)),
                 method='L-BFGS-B')


# Compute a run with the optimal parameters

# select kernel and points
#(0 for ker_exp ,1 for ker_SE)
#(1 for Points1, 2 for Points2)

for KER in range(2):
    for PTS in range(1,3):
        print('\nKER=', KER, '; PTS=', PTS)
        kernel, Points, Values, l_opt, var_opt, s_2_opt = case_study2(KER, PTS, 
                                                                     Points1, Values1, 
                                                                     Points2, Values2, 
                                                                     res1_exp, res1_se, 
                                                                     res2_exp, res2_se)
        
        # Fit Gaussian Process Regression
        
        K_Z, K_Y, K_ZY = evaluate_kernels(Points, Points_total, kernel, l=l_opt, var=var_opt)
        mean_perm, cov_perm = prediction(Values, K_Z, K_Y, K_ZY, s_2 = s_2_opt)
        var_perm = np.diag(cov_perm) 
        err = np.mean(np.abs(mean_perm-perm.flatten()))
        print('err w.r.t true_perm =', err)
        
        
        # Plot covariance matrix (HUGE!)
        #plot_cov(K_Y)
        #plot_cov(cov_perm)
        
        # Plot true perm, mean_pred, var_perm
        plot_compare_three(perm, mean_perm, var_perm, Points)
        
        # Generate gaussian processes with Circular Embedding
        perm_cond = conditional(mean_perm, cov_perm, K_ZY, K_Z, K_Y, Points, Values, n_process = 1000)
        
        # Plotta due processi a caso
        if KER==0 and PTS==1:
            perm_cond1 = perm_cond[:,0]
        elif KER==0 and PTS==2:
            perm_cond2 = perm_cond[:,0]
        elif KER==1 and PTS==1:
            perm_cond3 = perm_cond[:,0]
        else:
            perm_cond4 = perm_cond[:,0]
                
        
        # MONTE CARLO and VARIANCE REDUCTION
        
        Critical = (np.array([np.ones(4)*85, np.array([35, 40, 45, 50])]).T).astype(int)
        indexes = np.array([60*Critical[0,0] + i for i in Critical[:,1]])
        mean_interest = mean_perm[indexes]
        var_interest = cov_perm[indexes,:][:,indexes]
        
        if (KER == 0):
            N0 = 50000
            tol = 1e-4
        
        if(KER == 1):
            N0 = 100000
            tol = 1e-5
            
        N, mean_est, std_est = MC_2stages(mean_interest, var_interest, tol = tol, N0 = N0)
        print('N=', N, '; mu_MC=', mean_est, '; std_est=', std_est)
        
        # Generate additional data for the convergence graph
        np.random.seed(12345)
        Ns_all = np.array([100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000])
        
        means_MC, stds_MC, means_AV, stds_AV = plot_MC_AV(Ns_all, mean_interest, var_interest, KER, PTS)
        print('mu_MC=', means_MC[-1], '+-', stds_MC[-1]*1.96/np.sqrt(Ns_all[-1]))
        print('mu_AV=', means_AV[-1], '+-', stds_AV[-1]*1.96/np.sqrt(Ns_all[-1]/2))
        
plot_CE(perm_cond1, perm_cond2, perm_cond3, perm_cond4, Points1, Points2)
