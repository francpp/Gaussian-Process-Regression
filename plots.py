import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from MonteCarlo import *
# from matplotlib import rc
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def plot_GPR_1D(Z, f_tilde_Z, Y, mu_YgivenZ, K_YgivenZ, y_cond, l, ker):
    """
    Plot 3 Gaussian processes in y_cond. These processes are obtained by means
    of GPR in 1D, so they are conditioned upon the observations f_tilde_Z.
    All the processes follow the Gaussian (conditioned) distribution with mean
    mu_YgivenZ and covariance matrix K_YgivenZ.
    
    The inputs
    

    Parameters
    ----------
    Z : TYPE
        DESCRIPTION.
    f_tilde_Z : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    mu_YgivenZ : TYPE
        DESCRIPTION.
    K_YgivenZ : TYPE
        DESCRIPTION.
    y_cond : TYPE
        DESCRIPTION.
    l : TYPE
        DESCRIPTION.
    ker : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

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
    
    #plt.legend(['y_{test_1}', 'y_{test_2}', 'y_{test_3}',
    #            'mu(y_{test})','y_{train}','conf. int. = '],
    #           fancybox=True, 
    #           framealpha=0.0,
    #           loc='upper center', 
    #           bbox_to_anchor=(0.5, -0.05), 
    #           shadow=False, 
    #           ncol=3)
    
    plt.title('ker = ' + str(type_ker) + ', $\ell=$' + format(l, '.4f') + ', $N_Z=$' + str(Z.shape[0]), fontsize = 17)
    plt.show()
 
###########################################################################################################################

def plot_cov(K):
    plt.figure(figsize=(10,10))
    plt.contourf(np.flip(K, axis=0), 36, cmap='plasma')
    # plt.axis('equal')
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    plt.show()
 
###########################################################################################################################   
    
def plot_compare_three(perm, mean_perm, var_perm, Points):
    
    absolute_min = np.min((np.min(perm), np.min(mean_perm)))
    absolute_max = np.max((np.max(perm), np.max(mean_perm)))
    levels = np.linspace(absolute_min, absolute_max, 36)
    
    # Crea una figura con due sottofigure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,5))
    col = absolute_min*np.ones(perm.shape)
    col[:3, :] = absolute_max
    cf0 = ax1.contourf(col, levels)

    # Aggiungi il primo subplot
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

    # Aggiungi il secondo subplot
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

    # Crea la colorbar come una sottofigura separata
    cax = fig.add_axes([0.125, -0.25, 0.5, 0.03])
    
    # Crea la colorbar e associala ai subplot
    
    fig.colorbar(cf0, cax=cax, orientation='horizontal')

    # Sposta la colorbar di 1/4 dell'altezza della figura verso il basso
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
    # Crea la colorbar come una sottofigura separata
    cax3 = fig.add_axes([1, -0.25, 0.23, 0.03])
    fig.colorbar(cf3, cax=cax3, orientation='horizontal')
    cax3.set_position([cax3.get_position().x0 - 0.33, cax.get_position().y0, cax3.get_position().width, cax3.get_position().height])

    plt.show()

###################################################################################################################  
    
def plot_MC_AV(Ns_all, mean_interest, var_interest, KER, PTS):
    
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
    # plt.ylabel(r'I')
    # plt.ylim(0,1e-4)
    plt.legend()
    plt.grid(which='both')
    # plt.gca().set_aspect(3.0)
    # plt.savefig('../figures/EX_1_1_CONF.eps',bbox_inches='tight')
    if (KER == 0 and PTS==1):
        plt.title('Exponential Kernel, Dataset Z1', fontsize=15)
    elif (KER==0 and PTS==2):
        plt.title('Exponential Kernel, Dataset Z2', fontsize=15)
    elif (KER==1 and PTS==1):
        plt.title('Squared Exponential Kernel, Dataset Z1', fontsize=15)
    else:
        plt.title('Squared Exponential Kernel, Dataset Z2', fontsize=15)
    
    if(KER == 1):
        N0 = 100000
        tol = 1e-5
        
    # plt.title()
    plt.show()
    return means_MC, stds_MC, means_AV, stds_AV    

"""    
def plot_MC_IS(Ns_all, mean_interest, var_interest):
    
    means_MC = np.zeros(Ns_all.shape) 
    means_IS = np.zeros(Ns_all.shape)
    stds_MC = np.zeros(Ns_all.shape)
    stds_IS = np.zeros(Ns_all.shape)
    

    for i in range(len(Ns_all)):
        
        means_MC[i], stds_MC[i] = MC(mean_interest, var_interest, Ns_all[i]) 
        means_IS[i], stds_IS[i] = MC_IS(mean_interest, var_interest, Ns_all[i])

    plt.figure()
    plt.semilogx(Ns_all, means_MC, label=r'Mean MC')
    upperlim = means_MC + 1.96*stds_MC/np.sqrt(Ns_all)
    lowerlim = means_MC - 1.96*stds_MC/np.sqrt(Ns_all)
    plt.semilogx(Ns_all, upperlim,'--',color='red',label=r'CI $95\%$ MC')
    plt.semilogx(Ns_all, lowerlim,'--',color='red')

    plt.semilogx(Ns_all, means_IS, color='green', label=r'Mean IS')
    upperlim = means_IS + 1.96*stds_IS/np.sqrt(Ns_all)
    lowerlim = means_IS - 1.96*stds_IS/np.sqrt(Ns_all)
    plt.semilogx(Ns_all, upperlim,'--',color='orange',label=r'CI $95\%$ IS')
    plt.semilogx(Ns_all, lowerlim,'--',color='orange')

    plt.xlabel(r'N')
    # plt.ylabel(r'I')
    # plt.ylim(0,1e-4)
    plt.legend()
    plt.grid(which='both')
    # plt.gca().set_aspect(3.0)
    # plt.savefig('../figures/EX_1_1_CONF.eps',bbox_inches='tight')
    plt.show()
"""

#####################################################################################################

def plot_compare_kernels(Kexp, Kse):
    
    absolute_min = np.min((np.min(Kexp), np.min(Kse)))
    absolute_max = np.max((np.max(Kexp), np.max(Kse)))
    levels = np.linspace(absolute_min, absolute_max, 36)
    
    # Crea una figura con due sottofigure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (12,5))
    col = absolute_min*np.ones(Kexp.shape)
    col[:3, :] = absolute_max
    cf0 = ax1.contourf(col, 36, cmap='plasma')

    # Aggiungi il primo subplot
    # cf1 = ax1.contourf(X, Y, Z1, levels=10, cmap='viridis')
    cf1 = ax1.contourf(np.flip(Kexp,axis=0), levels, cmap='plasma')
    ax1.set_title('Exponential Kernel', fontsize=15)
    ax1.get_yaxis().set_visible(False)

    # Aggiungi il secondo subplot
    cf2 = ax2.contourf(np.flip(Kse,axis=0), levels, cmap='plasma')
    ax2.set_title('Squared Exponential Kernel', fontsize=15)
    ax2.get_yaxis().set_visible(False)

    # Crea la colorbar come una sottofigura separata
    cax = fig.add_axes([0.125, -0.25, 0.775, 0.03])
    fig.colorbar(cf0, cax=cax, orientation='horizontal')

    # Sposta la colorbar di 1/4 dell'altezza della figura verso il basso
    cax.set_position([cax.get_position().x0, cax.get_position().y0 + 0.25, cax.get_position().width, cax.get_position().height])

    plt.show()
 
###############################################################################################################

def plot_CE(perm_cond1, perm_cond2, perm_cond3, perm_cond4, Points1, Points2):
    
    absolute_min = np.min((np.min(perm_cond1), np.min(perm_cond2), np.min(perm_cond3)))
    absolute_max = np.max((np.max(perm_cond1), np.max(perm_cond2), np.max(perm_cond3)))
    levels = np.linspace(absolute_min, absolute_max, 36)
    
    perm_cond1 = perm_cond1.reshape(110,60)
    perm_cond2 = perm_cond2.reshape(110,60)
    perm_cond3 = perm_cond3.reshape(110,60)
    perm_cond4 = perm_cond4.reshape(110,60)
    
    # Crea una figura con due sottofigure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
    col = absolute_min*np.ones(perm_cond1.shape)
    col[:3, :] = absolute_max
    cf0 = ax1.contourf(col, levels)
    #------------------------------------------------
    # Aggiungi il primo subplot
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
    # Aggiungi il secondo subplot
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
    # Aggiungi il primo subplot
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
    # Aggiungi il secondo subplot
    
    # Crea la colorbar come una sottofigura separata
    cax = fig.add_axes([0.125, -0.25, 0.575, 0.03])
    
    # Crea la colorbar e associala ai subplot
    
    fig.colorbar(cf0, cax=cax, orientation='horizontal')

    # Sposta la colorbar di 1/4 dell'altezza della figura verso il basso
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
    
    # Crea la colorbar come una sottofigura separata
    cax4 = fig.add_axes([1.06, 0, 0.17, 0.03])
    fig.colorbar(cf4, cax=cax4, orientation='horizontal')
    cax4.set_position([cax4.get_position().x0 - 0.33, cax4.get_position().y0, cax4.get_position().width, cax4.get_position().height])

    plt.show()
