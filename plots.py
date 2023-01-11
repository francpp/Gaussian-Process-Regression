import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from MonteCarlo import MC, MC_AV
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

###############################################################################

def plot_GPR_1D(Z, f_tilde_Z, Y, mu_YgivenZ, K_YgivenZ, y_cond, l, ker):
    """
    Input:
        Z : array of Nz points
        f_tilde_Z : array of Nz observations
        Y : array of Ny points 
        mu_YgivenZ : conditional mean, dim = Ny
        K_YgivenZ : conditional covariance, dim = (Ny, Ny)
        y_cond : 3 conditional processes, dim = (3, Ny)
        l : float, length scale parameter
        ker : 0 for ker_exp, 1 for ker_SE
    """
    
    var_YgivenZ = np.diag(K_YgivenZ).reshape(-1,1)
    if (ker == 0):
        type_ker = 'exp'
    elif (ker == 1):
        type_ker = 'SE'
        
    plt.figure()
    
    plt.plot(Y,
             y_cond[:,0],
    #         marker="X",
    #         markersize=20,
    #         linestyle="None",
    #          color="blue",
             label="$y_{1,cond}$",
             linewidth=1.5,
            )
    
    plt.plot(Y,
             y_cond[:,1],
    #         marker="X",
    #         markersize=20,
    #         linestyle="None",
    #          color="blue",
             label="$y_{2,cond}$",
             linewidth=1.5,
            )
    
    plt.plot(Y,
             y_cond[:,2],
    #         marker="X",
    #         markersize=20,
    #         linestyle="None",
    #          color="blue",
             label="$y_{3,cond}$",
             linewidth=1.5,
            )
    
    plt.plot(Y,
             mu_YgivenZ,
    #         marker="X",
    #         markersize=20,
    #         linestyle="None",
    #         color="blue",
             label="$\mu_{cond}$",
             linewidth=2.5,
            )

    plt.plot(Z,
             f_tilde_Z,
             marker="o",
             markersize=6,
             linestyle="None",
             color="b",
             label= r'$\tilde{f}(z)$',
             linewidth=2,
            )

    plt.fill_between(Y.reshape(-1),
                     (mu_YgivenZ - 2*np.sqrt(var_YgivenZ)).reshape(-1), 
                     (mu_YgivenZ + 2*np.sqrt(var_YgivenZ)).reshape(-1), 
                     color = 'yellow', 
                     label="CI: $2\sigma$", 
                     alpha = 0.3)
    
    plt.grid(True)
    plt.legend(fontsize=10)
        
    plt.title('ker = ' + str(type_ker) + ', $\ell=$' + format(l, '.4f') + ', $N_Z=$' + str(Z.shape[0]), fontsize = 17)
    plt.show()
 
###########################################################################################################################

def plot_cov(K):
    '''Contourf of covariance matrix
    Input:
        K: covariance matrix, dim = (NxN)
    '''
    
    plt.figure(figsize=(10,10))
    plt.contourf(np.flip(K, axis=0), 36, cmap='plasma')
    # plt.axis('equal')
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    plt.show()
 
###########################################################################################################################   
    
def plot_compare_three(perm, mean_perm, var_perm, Points):
    """
    Input:
        perm : true observations, dim = (110,60)
        mean_perm : conditional mean, dim = (6600,)
        var_perm : conditional variance, dim = (6600,)
        Points : array of 2D training Points, dim = (Nz, 2)    
    """
    
    absolute_min = np.min((np.min(perm), np.min(mean_perm)))
    absolute_max = np.max((np.max(perm), np.max(mean_perm)))
    levels = np.linspace(absolute_min, absolute_max, 36)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,5))
    col = absolute_min*np.ones(perm.shape)
    col[:3, :] = absolute_max
    cf0 = ax1.contourf(col, levels)

    # cf1 = ax1.contourf(X, Y, Z1, levels=10, cmap='viridis')
    cf1 = ax1.contourf(perm, levels)
    ax1.plot(Points[:,1],
             Points[:,0],
             marker="o",
             markersize=0.5,
             linestyle="None",
             color="k",
    #         label="f_tilde_z",
    #         linewidth=2,
            )
    ax1.set_title('True')

    cf2 = ax2.contourf(mean_perm.reshape(110,60) , levels)
    ax2.plot(Points[:,1],
             Points[:,0],
             marker="o",
             markersize=0.5,
             linestyle="None",
             color="k",
    #         label="f_tilde_z",
    #         linewidth=2,
            )
    ax2.set_title('Predicted')

    cax = fig.add_axes([0.125, -0.25, 0.5, 0.03])       
    fig.colorbar(cf0, cax=cax, orientation='horizontal')
    cax.set_position([cax.get_position().x0, cax.get_position().y0 + 0.25, cax.get_position().width, cax.get_position().height])

    cf3 = ax3.contourf(var_perm.reshape(110,60), levels=36, cmap='plasma')
    ax3.plot(Points[:,1],
             Points[:,0],
             marker="o",
             markersize=0.5,
             linestyle="None",
             color="k",
    #         label="f_tilde_z",
    #         linewidth=2,
            )
    ax3.set_title('Variance')

    cax3 = fig.add_axes([1, -0.25, 0.23, 0.03])
    fig.colorbar(cf3, cax=cax3, orientation='horizontal')
    cax3.set_position([cax3.get_position().x0 - 0.33, cax.get_position().y0, cax3.get_position().width, cax3.get_position().height])

    plt.show()

###################################################################################################################  
    
def plot_MC_AV(Ns_all, mean_interest, var_interest, KER, PTS):
    """
    Input:      
    Ns_all : array of different N to compute MC and AV
    mean_interest : mean of the 4 interested points, dim = (4,1) 
    var_interest : covariance matrix of the 4 interested points, dim = (4,4)
    KER : 0 for ker_exp, 1 for ker_SE
    PTS : 1 for Nz = 25, 2 for Nz = 121

    Output:        
    -------
    means_MC : array of means compute by MC, dim = (len(Ns_all))
    stds_MC : array of std deviations compute by MC, dim = (len(Ns_all))
    means_AV : array of means compute by MC, dim = (len(Ns_all))
    stds_AV : array of std deviations compute by MC, dim = (len(Ns_all))
    """
    
    means_MC = np.zeros(Ns_all.shape) 
    means_AV = np.zeros(Ns_all.shape)
    stds_MC = np.zeros(Ns_all.shape)
    stds_AV = np.zeros(Ns_all.shape) 

    for i in range(len(Ns_all)):
        
        means_MC[i], stds_MC[i] = MC(mean_interest, var_interest, Ns_all[i]) 
        means_AV[i], stds_AV[i] = MC_AV(mean_interest, var_interest, Ns_all[i])

    plt.figure()
    plt.semilogx(Ns_all, means_MC, label=r'Mean MC')
    upperlim = means_MC + 1.96*stds_MC/np.sqrt(Ns_all)
    lowerlim = means_MC - 1.96*stds_MC/np.sqrt(Ns_all)
    plt.semilogx(Ns_all, upperlim,'--',color='red',label=r'CI $95\%$ MC')
    plt.semilogx(Ns_all, lowerlim,'--',color='red')

    plt.semilogx(Ns_all, means_AV, color='green', label=r'Mean AV')
    upperlim = means_AV + 1.96*stds_AV/np.sqrt(Ns_all/2)
    lowerlim = means_AV - 1.96*stds_AV/np.sqrt(Ns_all/2)
    plt.semilogx(Ns_all, upperlim,'--',color='orange',label=r'CI $95\%$ AV')
    plt.semilogx(Ns_all, lowerlim,'--',color='orange')

    plt.xlabel(r'N')
    plt.legend()
    plt.grid(which='both')

    if (KER == 0 and PTS==1):
        plt.title('Exponential Kernel, Dataset Z1', fontsize=15)
    elif (KER==0 and PTS==2):
        plt.title('Exponential Kernel, Dataset Z2', fontsize=15)
    elif (KER==1 and PTS==1):
        plt.title('Squared Exponential Kernel, Dataset Z1', fontsize=15)
    else:
        plt.title('Squared Exponential Kernel, Dataset Z2', fontsize=15)
            
    # plt.title()
    plt.show()
    return means_MC, stds_MC, means_AV, stds_AV    


#####################################################################################################

def plot_compare_kernels(Kexp, Kse):
    """
    Input:
        Kexp: covariance matrix of kernel_exp, dim = (6600,6600)
        Kse: covariance matrix of kernel_SE, dim = (6600,6600)
    """
    
    absolute_min = np.min((np.min(Kexp), np.min(Kse)))
    absolute_max = np.max((np.max(Kexp), np.max(Kse)))
    levels = np.linspace(absolute_min, absolute_max, 36)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (12,5))
    col = absolute_min*np.ones(Kexp.shape)
    col[:3, :] = absolute_max
    cf0 = ax1.contourf(col, 36, cmap='plasma')

    # cf1 = ax1.contourf(X, Y, Z1, levels=10, cmap='viridis')
    cf1 = ax1.contourf(np.flip(Kexp,axis=0), levels, cmap='plasma')
    ax1.set_title('Exponential Kernel', fontsize=15)
    ax1.get_yaxis().set_visible(False)

    cf2 = ax2.contourf(np.flip(Kse,axis=0), levels, cmap='plasma')
    ax2.set_title('Squared Exponential Kernel', fontsize=15)
    ax2.get_yaxis().set_visible(False)

    cax = fig.add_axes([0.125, -0.25, 0.775, 0.03])
    fig.colorbar(cf0, cax=cax, orientation='horizontal')

    cax.set_position([cax.get_position().x0, cax.get_position().y0 + 0.25, cax.get_position().width, cax.get_position().height])

    plt.show()
 
###############################################################################################################

def plot_CE(perm_cond1, perm_cond2, perm_cond3, perm_cond4, Points1, Points2):
    """
    Parameters
    ----------
    perm_cond1 : gaussian process for (Nz = 25, ker_exp), dim = (6600,)
    perm_cond2 : gaussian process for (Nz = 121, ker_exp), dim = (6600,)
    perm_cond3 : gaussian process for (Nz = 25, ker_SE), dim = (6600,)
    perm_cond4 : gaussian process for (Nz = 121, ker_SE), dim = (6600,)
    Points1 : training points for Nz = 25, dim = (25,2)
    Points2 : training points for Nz = 121, dim = (121,2)
    """
    
    absolute_min = np.min((np.min(perm_cond1), np.min(perm_cond2), np.min(perm_cond3)))
    absolute_max = np.max((np.max(perm_cond1), np.max(perm_cond2), np.max(perm_cond3)))
    levels = np.linspace(absolute_min, absolute_max, 36)
    
    perm_cond1 = perm_cond1.reshape(110,60)
    perm_cond2 = perm_cond2.reshape(110,60)
    perm_cond3 = perm_cond3.reshape(110,60)
    perm_cond4 = perm_cond4.reshape(110,60)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
    col = absolute_min*np.ones(perm_cond1.shape)
    col[:3, :] = absolute_max
    cf0 = ax1.contourf(col, levels)
    #------------------------------------------------
    cf1 = ax1.contourf(perm_cond1, levels)
    ax1.plot(Points1[:,1],
             Points1[:,0],
             marker="o",
             markersize=0.5,
             linestyle="None",
             color="k",
    #         label="f_tilde_z",
    #         linewidth=2,
            )
    ax1.set_title('EXP Ker, Dataset Z1')
    #------------------------------------------------
    cf2 = ax2.contourf(perm_cond2, levels)
    ax2.plot(Points2[:,1],
             Points2[:,0],
             marker="o",
             markersize=0.5,
             linestyle="None",
             color="k",
    #         label="f_tilde_z",
    #         linewidth=2,
            )
    ax2.set_title('EXP Ker, Dataset Z2')
    #------------------------------------------------
    cf3 = ax3.contourf(perm_cond3, levels)
    ax3.plot(Points1[:,1],
             Points1[:,0],
             marker="o",
             markersize=0.5,
             linestyle="None",
             color="k",
    #         label="f_tilde_z",
    #         linewidth=2,
            )
    ax3.set_title('SE Ker, Dataset Z1')
    #------------------------------------------------
    
    cax = fig.add_axes([0.125, -0.25, 0.575, 0.03])    
    fig.colorbar(cf0, cax=cax, orientation='horizontal')
    cax.set_position([cax.get_position().x0, cax.get_position().y0 + 0.25, cax.get_position().width, cax.get_position().height])

    cf4 = ax4.contourf(perm_cond4, 36)
    ax4.plot(Points2[:,1],
             Points2[:,0],
             marker="o",
             markersize=0.5,
             linestyle="None",
             color="k",
    #         label="f_tilde_z",
    #         linewidth=2,
            )
    ax4.set_title('SE Ker, Dataset Z2')
    
    cax4 = fig.add_axes([1.06, 0, 0.17, 0.03])
    fig.colorbar(cf4, cax=cax4, orientation='horizontal')
    cax4.set_position([cax4.get_position().x0 - 0.33, cax4.get_position().y0, cax4.get_position().width, cax4.get_position().height])

    plt.show()
