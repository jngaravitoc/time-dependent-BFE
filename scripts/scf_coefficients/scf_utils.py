"""
Various routines to work with SCF coefficients

TODO:

 - Add docstrings 

"""

import numpy as np
import h5py
from scipy import special
from numpy import linalg
import matplotlib.pyplot as plt
import gala.potential as gp
from astropy import units as u
from gala.units import galactic

def read_coefficients(filename, verbose=False):
    """
    Read coefficients in  hdf5 format

    """
    hf=h5py.File(filename + ".hdf5", 'r')
    if verbose==True:
        print(hf.keys())
        print("* Loading coefficients")

    Snlm = np.array(hf.get('Snlm'))
    Tnlm = np.array(hf.get('Tnlm'))
    nmax = np.array(hf.get('nmax'))
    lmax = np.array(hf.get('lmax'))
    mmax = np.array(hf.get('mmax'))
    rs = np.array(hf.get('rs'))
    pmass = np.array(hf.get('pmass'))
    G = np.array(hf.get('G'))
    rcom = np.array(hf.get('rcom'))
    coefficients = [Snlm, Tnlm]

    if 'var_Snlm' in hf.keys():
        var_Snlm = np.array(hf.get('var_Snlm'))
        coefficients.append(var_Snlm)
    if 'var_Tnlm' in hf.keys():
        var_Tnlm = np.array(hf.get('var_Tnlm'))
        coefficients.append(var_Tnlm)
    if 'var_STnlm' in hf.keys():
        var_STnlm = np.array(hf.get('var_STnlm'))
        coefficients.append(var_STnlm)
        
    hf.close()

    return coefficients, [nmax, lmax, mmax], [rs, pmass, G], rcom

def array_coefficients2(filename, init_snap, final_snap):
    """
    Read coefficients from subsequent snapshots

    Parameters:
    -----------

    filename:
        base name of the coefficients
    init_snap:
        initial snap number
    final_snap:
        final snap number

    Return:
    -------
        Sjnlm : 
        Tjnlm : 
        rcom : 
        constants: rs, pmass, G 

    """

    first_scf = read_coefficients(filename+"{:03d}".format(init_snap))
    nmax = first_scf[1][0]
    lmax = first_scf[1][1]
    mmax = first_scf[1][2]
    rs = first_scf[2][0]
    pmass = first_scf[2][1]
    G = first_scf[2][2]
    rj_array = np.zeros((final_snap - init_snap, 3))
    Sjnlm_array = np.zeros((final_snap - init_snap, nmax+1, lmax+1, mmax+1))
    Tjnlm_array = np.zeros((final_snap - init_snap, nmax+1, lmax+1, mmax+1))
    Sjnlm_var_array = np.zeros((final_snap - init_snap, nmax+1, lmax+1, mmax+1))
    Tjnlm_var_array = np.zeros((final_snap - init_snap, nmax+1, lmax+1, mmax+1))
    STjnlm_var_array = np.zeros((final_snap - init_snap, nmax+1, lmax+1, mmax+1))

    for k in range(init_snap, final_snap):
        coeff_all = read_coefficients(filename+"{:03d}".format(k))
        Sjnlm_array[k-init_snap] = coeff_all[0][0]
        Tjnlm_array[k-init_snap] = coeff_all[0][1]
        Sjnlm_var_array[k-init_snap] = coeff_all[0][2]
        Tjnlm_var_array[k-init_snap] = coeff_all[0][3]
        STjnlm_var_array[k-init_snap] = coeff_all[0][4]
        rj_array[k-init_snap] = np.array(coeff_all[3][0])
    coefficients = [Sjnlm_array, Tjnlm_array, Sjnlm_var_array, Tjnlm_var_array, STjnlm_var_array]
    return coefficients, [nmax, lmax, mmax], [rs, pmass, G], rj_array

   
def array_coefficients(filename, init_snap, final_snap):
    """
    Read coefficients from subsequent snapshots

    Parameters:
    -----------

    filename:
        base name of the coefficients
    init_snap:
        initial snap number
    final_snap:
        final snap number

    Return:
    -------
        Sjnlm : 
        Tjnlm : 
        rcom : 
        constants: rs, pmass, G 

    """

    first_scf = read_coefficients(filename+"{:03d}".format(init_snap))
    nmax = first_scf[1][0]
    lmax = first_scf[1][1]
    mmax = first_scf[1][2]
    rs = first_scf[2][0]
    pmass = first_scf[2][1]
    G = first_scf[2][2]
    rj_array = np.zeros((final_snap - init_snap, 3))
    Sjnlm_array = np.zeros((final_snap - init_snap, nmax+1, lmax+1, mmax+1))
    Tjnlm_array = np.zeros((final_snap - init_snap, nmax+1, lmax+1, mmax+1))
    #Sjnlm_var_array = np.zeros((final_snap - init_snap, nmax+1, lmax+1, mmax+1))
    #Tjnlm_var_array = np.zeros((final_snap - init_snap, nmax+1, lmax+1, mmax+1))
    #STjnlm_var_array = np.zeros((final_snap - init_snap, nmax+1, lmax+1, mmax+1))

    for k in range(init_snap, final_snap):
        coeff_all = read_coefficients(filename+"{:03d}".format(k))
        Sjnlm_array[k-init_snap] = coeff_all[0][0]
        Tjnlm_array[k-init_snap] = coeff_all[0][1]
        rj_array[k-init_snap] = np.array(coeff_all[3][0])
    coefficients = [Sjnlm_array, Tjnlm_array, Sjnlm_var_array, Tjnlm_var_array, STjnlm_var_array]
    return Sjnlm_array, Tjnlm_array, rj_array, [rs, pmass, G]
   
## Reading coefficients
def _reshape_matrix(matrix):
    """ 
    Build a matrix with shape (nmax, lmax, mmax) for coefficients that are
    not in matrix form e.g., flatten(). 

    """
    coef_matrix = np.zeros((self.nmax+1, self.lmax+1, self.mmax+1))
    
    counter = 0
    for n in range(self.nmax+1):
       for l in range(self.lmax+1):
            for m in range(0, slef.lmax+1):
                coef_matrix[n][l][m] = matrix[counter]
                counter +=1
    return coef_matrix


def _smoothing(Snlm, Tnlm, varSnlm, varTnlm):
    """
    Computes optimal smoothing following Eq.8 in Weinberg+96.
    
    Returns:
    --------
    
    bSnlm : 
    bTnlm : 
    """
    bSnlm = 1 / (1 + (varSnlm/Snlm**2))
    bTnlm = 1 / (1 + (varTnlm/Tnlm**2))
    if Snlm == 0:
        bSnlm=  0
    if Tnlm == 0:
        bTnlm = 0
    return bSnlm, bTnlm

def _covariance_matrix(Snlm, Tnlm, Snlm_var, Tnlm_var, STnlm_var, pmass):
    """
    Build covariance matrix 
    """

    cov_matrix = np.zeros((2,2))
    cov_matrix[0][0] = Snlm_var - pmass*Snlm**2
    cov_matrix[0][1] = STnlm_var - pmass*Snlm*Tnlm
    cov_matrix[1][1] = Tnlm_var - pmass*Tnlm**2
    cov_matrix[1][0] = cov_matrix[0][1]

    return cov_matrix

def _smooth_coefficients(S, T, Svar, Tvar, STvar, pmass, sn_threshold=0, verb=False, SN=False):

    #build covariance matrix
    cov_matrix = _covariance_matrix(S, T, Svar, Tvar, STvar, pmass)
    
    # SVD decomposition of the covariance matrix
    T_rot, v, TL = linalg.svd(cov_matrix)
    
    # Computes invert transformation matrix
    T_rot_inv = linalg.inv(T_rot)

    # Variances of the coefficients in the principal base.
    varS = v[0]
    varT = v[1]
    
    ## uncorrelated coefficients
    coeff_base = np.array([S, T])
    S_unc, T_unc = np.dot(T_rot, coeff_base)
    b_S_unc, b_T_unc = _smoothing(S_unc, T_unc, varS, varT)
    S_unc_smooth = S_unc*b_S_unc
    T_unc_smooth = T_unc*b_T_unc
    SN_coeff_unc = (S_unc**2/varS)**0.5
    SN_coeff = SN_coeff_unc
    S_smooth, T_smooth = np.dot(T_rot_inv, np.array([S_unc_smooth, T_unc_smooth]))
    
    if verb==True:
        print("* S,T  correlated = ", S, T)
        print("* S,T uncorrelated = ", S_unc, T_unc)
        print("* Uncorrelated smoothing = ", b_S_unc, b_T_unc)
        print("* S, T uncorrelated smoothed =", S_unc_smooth, T_unc_smooth)
    
    if SN_coeff_unc < sn_threshold:
        S_smooth = 0 
        T_smooth = 0 
        SN_coeff = 0
        
    if SN == False: 
        return S_smooth, T_smooth
    elif SN == True:
        return S_smooth, T_smooth, SN_coeff

    

def smooth_coefficients_matrix(Snlm, Tnlm, Snlm_var, Tnlm_var, STnlm_var, pmass, verb, sn_threshold, SN=False):
    S_matrix_smooth = np.zeros_like(Snlm)
    T_matrix_smooth = np.zeros_like(Tnlm)
    SN_coeff = np.zeros_like(S_matrix_smooth)
    nmax = np.shape(S_matrix_smooth)[0]
    lmax = np.shape(S_matrix_smooth)[1]
    for n in range(nmax):
        for l in range(lmax):
            for m in range(lmax):
                S_matrix_smooth[n][l][m], T_matrix_smooth[n][l][m], SN_coeff[n][l][m] = _smooth_coefficients(Snlm[n][l][m],
                                                                                  Tnlm[n][l][m], 
                                                                                  Snlm_var[n][l][m], 
                                                                                  Tnlm_var[n][l][m],
                                                                                  STnlm_var[n][l][m],
                                                                                  pmass, verb=False, 
                                                                                  sn_threshold=sn_threshold,
                                                                                  SN=True)
    ncoeff = len(np.nonzero(S_matrix_smooth)[0])
                   
    if SN == False :
        return S_matrix_smooth, T_matrix_smooth, ncoeff
    
    elif SN == True :
        return S_matrix_smooth, T_matrix_smooth, ncoeff, SN_coeff



# Energy calculations 


def _Anl(n, l):
    knl = 0.5*n*(n+4*l+3) + (l+1)*(2*l+1)
    A_nl = - 2**(8*l+6)/(4*np.pi*knl) * (special.factorial(n)*(n+2*l+3/2.)*(special.gamma(2*l+3/2.))**2)/(special.gamma(n+4*l+3))
    return A_nl

def _Anl_array(nmax, lmax):
    A_nl_array = np.zeros((nmax, lmax))
    for j in range(nmax):
        for i in range(lmax):
            A_nl_array[j][i] = _Anl(j, i)
    return A_nl_array

def coefficients_energy(S, T, nmax, lmax, mmax):
    A_nl = _Anl_array(nmax, lmax)
    U = np.zeros_like(S)
    for m in range(mmax):
        if m==0:
            U[:,:,m] = (S[:,:,m]**2 + T[:,:,m]**2)/(A_nl)
        else:
            U[:,:,m] = (S[:,:,m]**2 + T[:,:,m]**2)/(2*A_nl)
    return U

def coefficients_energy_n(S, T, n, nmax, lmax):
    A_nl = _Anl_array(n, lmax)
    A = (S[n,:,:]**2 + T[n,:,:]**2)
    A_nl_m = np.zeros((lmax, lmax))
    for i in range(1, lmax):
        A_nl_m[:,i] = A_nl[0]/2
    A_nl_m[:,0] = A_nl[0]
    return A/A_nl_m

def coefficients_energy_array(Sjnlm, Tjnlm):
    Ujnlm = np.zeros_like(Sjnlm)
    nsnaps = np.shape(Sjnlm)[0]
    nmax = np.shape(Sjnlm)[1]
    lmax = np.shape(Sjnlm)[2]
    mmax = np.shape(Sjnlm)[3]

    for k in range(nsnaps):
        Ujnlm[k] = coefficients_energy(Sjnlm[k], Tjnlm[k], nmax, lmax, mmax)
    return Ujnlm

## Visualization

class SCFvis:
    def __init__(self, Snlm, Tnlm, nmax, lmax, mmax):
        self.Snlm = Snlm
        self.Tnlm = Tnlm
        self.nmax = nmax
        self.lmax = lmax
        self.mmax = mmax

    def hist_energy(self, m):
        A_nl = _Anl_array(self.nmax, self.lmax)
        A = (self.Snlm[:,:,m]**2 + self.Tnlm[:,:,m]**2)
        fig, ax = plt.subplots(1, 1, figsize=(5,4))
        im = ax.imshow(np.log10(np.abs(A/A_nl)).T, origin='lower', cmap='viridis')
        ax.set_xlabel(r'$n$')
        ax.set_ylabel(r'$l$')
        ax.set_title(r'$Log_{10} Unlm$')
        fig.colorbar(im, ax=ax)
        return 0

    def hist_scf(self, m, axis):
        """
        TODO: Add functionality to pick different projections in n, l, m.
        """

        fig, ax = plt.subplots(1, 2, figsize=(10,4), sharey=True)
        im1 = ax[0].imshow(np.log10(np.abs(self.Snlm[:,:,m])).T, origin='lower', cmap='viridis')
        im2 = ax[1].imshow(np.log10(np.abs(self.Tnlm[:,:,m])).T, origin='lower', cmap='viridis')
        ax[0].set_xlabel(r'$n$')
        ax[0].set_ylabel(r'$l$')
        ax[0].set_title(r'$Log_{10} Snlm$')
        ax[1].set_title(r'$Log_{10} Tnlm$')
        cbar1 = fig.colorbar(im1, ax=ax[0])
        cbar2 = fig.colorbar(im2, ax=ax[1])
        return 0

    def density_contour(self, S, T, grid_size, m, rs, snap, ngrid=128, delta_rho=False):
        """
        TODO: use self variables! 
        """
        S0 = np.zeros_like(S)
        T0 = np.zeros_like(T)
        S0[0,0,0] = S[0,0,0]
        T0[0,0,0] = T[0,0,0]
        
        circle1 = plt.Circle((0, 0), 100, color='w', fill=False, ls='--', alpha=0.7)

        pot = gp.SCFPotential(m=m*u.Msun, r_s=rs*u.kpc, Snlm=S, Tnlm=T, units=galactic)
        
        x0 = np.linspace(grid_size[0], grid_size[1], ngrid)
        y0 = np.linspace(grid_size[0], grid_size[1], ngrid)

        #x = np.linspace(grid_size[0]-orbit[snap,1], grid_size[1]-orbit[snap,1], ngrid)
        #y = np.linspace(grid_size[0]-orbit[snap,2], grid_size[1]-orbit[snap,2], ngrid)

        x = np.linspace(grid_size[0], grid_size[1], ngrid)
        y = np.linspace(grid_size[0], grid_size[1], ngrid)
        
        grid = np.meshgrid(x, y)    
        xyz = np.zeros((3, ngrid**2))
        xyz[1] = grid[0].flatten()
        xyz[2] = grid[1].flatten()
        
        rho = pot.density(xyz)
        #rho_0 = pot_0.density(xyz)

        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        if delta_rho == False :
            levels = np.linspace(np.min(np.log10(np.abs(rho.value))),  np.max(np.log10(np.abs(rho.value))), 20)
            
            ax.contourf(x0, y0, np.log10(np.abs(rho.value.reshape(ngrid, ngrid))), 20, cmap='inferno', 
                        origin='lower', extent=[-250, 250, -250, 250])
        elif delta_rho == True :
            pot_0 = gp.SCFPotential(m=m*u.Msun, r_s=rs*u.kpc, Snlm=S0, Tnlm=T0, units=galactic)
            rho_0 = pot_0.density(xyz)
            ax.contourf(x0, y0, (rho.value/rho_0.value).reshape(128, 128)-1 , 20, cmap='inferno', 
                        origin='lower', vmin=-0.4, vmax=0.4, extent=[-250, 250, -250, 250])

        #ax.scatter(orbit[snap,7], orbit[snap,8], c='w')
        #ax.plot(orbit[:snap+1,7], orbit[:snap+1,8], lw='1.5', c='w', alpha=0.7)
        ax.add_patch(circle1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot([-150, -50], [-220, -220], c='w')
        ax.text( -150, -210, r'$\rm{100\ kpc}$', c='w', fontsize=22)
        ax.text( -200, 220, r'$t={:0.1f}\ $Gyr'.format(snap*0.02), c='w', fontsize=22)

        #plt.savefig('density_contour_{:03d}.png'.format(snap), bbox_inches='tight', dpi=300)
        #plt.close()


class SCF_coeff:
    """
    A class with to work with SCF coefficients.

    """
    def __init__(self, filename, ninit, nfinal):
        # read coefficients.
        coefficients, exp_length, exp_params, rjcom = array_coefficients(filename,  ninit, nfinal)

        self.Snlm = coefficients[0]
        self.Tnlm = coefficients[1]
        self.Snlm_var = coefficients[2]
        self.Tnlm_var = coefficients[3]
        self.STnlm_var = coefficients[4]
       
        self.Sshape = np.shape(self.Snlm)
        self.Tshape = np.shape(self.Tnlm)
        self.Svarshape = np.shape(self.Snlm_var)
        self.Tvarshape = np.shape(self.Tnlm_var)
        self.STvarshape = np.shape(self.STnlm_var)
        
        assert self.Sshape == self.Tshape == self.Svarshape == self.Tvarshape == self.STvarshape
        #else :
        #           raise ValueError("coefficients elements should be 2 (Snlm, Tnlm) or 5 (Snlm, Tnlm, STnlm_var, TSnlm_var, STnlm_var)")
        
        self.nmax = exp_length[0]
        self.lmax = exp_length[1]
        self.mmax = exp_length[2]
        self.rjcom = rjcom
        self.rs = exp_params[0]
        self.pmass = exp_params[1]
        self.nsnaps = len(rjcom)

    def mean_coeff(self, variance=False):
        """
        Compute the mean of the coefficients from multiple files and return the mean values.
        # TODO: implement mean of covariance matrix
        """
              
        assert len(self.Sshape) > 3, "Check you are passing an array of coefficients"
        snaps = self.Sshape[0]
        Sj_mean = np.zeros_like(self.Snlm[0])
        Tj_mean = np.zeros_like(self.Tnlm[0])

        Svarj_mean = np.zeros_like(self.Snlm_var[0])
        Tvarj_mean = np.zeros_like(self.Tnlm_var[0])
        STvarj_mean = np.zeros_like(self.STnlm_var[0])

        for k in range(self.Sshape[0]):
            Sj_mean[i] = np.mean(self.Snlm[k], axis=0)
            Tj_mean[i] = np.mean(self.Tnlm[k], axis=0)
            if variance == True:
                Svarj_mean[i] = np.mean(self.Snlm_var[k], axis=0)
                Tvarj_mean[i] = np.mean(self.Tnlm_var[k], axis=0)
                STvarj_mean[i] = np.mean(self.STnlm_var[k], axis=0)
        if variance == False:
            return Sj_mean, Tj_mean
        elif variance == True:
            return Sj_mean, Tj_mean, Svarj_mean, Tvarj_mean, STvarj_mean

    def smooth_coefficients_array(self, sn_threshold=4, SN=False):
        """
        Returns smoothed coefficients for an array of snapshots 

        Parameters:
        ----------
        sn_threshold : int 
            minimum signal-to-noise threshold value that will be use to select
            coefficients.

        Returns:
        --------

        Sjnlm_smooth
        Tjnlm_smooth
        N_coefficients

        """
        Sjnlm_smooth = np.zeros_like(self.Snlm)
        Tjnlm_smooth = np.zeros_like(self.Tnlm)
        SN_coeff = np.zeros_like(Sjnlm_smooth)
        Ncoeff = np.zeros(self.nsnaps)
        for k in range(self.nsnaps):
            Sjnlm_smooth[k], Tjnlm_smooth[k], Ncoeff[k], SN_coeff[k] = smooth_coefficients_matrix(self.Snlm[k],
                                                                                       self.Tnlm[k], 
                                                                                       self.Snlm_var[k], 
                                                                                       self.Tnlm_var[k],
                                                                                       self.STnlm_var[k],
                                                                                       self.pmass, verb=True, 
                                                                                       sn_threshold=sn_threshold,
                                                                                       SN=True)
            
        if SN == False :
            return Sjnlm_smooth, Tjnlm_smooth, Ncoeff
        
        elif SN == True :
            return Sjnlm_smooth, Tjnlm_smooth, Ncoeff, SN_coeff
          

