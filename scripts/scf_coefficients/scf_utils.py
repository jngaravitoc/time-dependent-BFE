"""
Various routines to work with SCF coefficients

"""

import numpy as np
import h5py

from numpy import linalg

def read_coefficients(filename):
    """
    Read cefficients into an hdf5 file

    """
    hf=h5py.File(filename + ".hdf5", 'r')
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

## Reading coefficients
def reshape_matrix(matrix):
    """ 
    Build a matrix with the shape (nmax, lmax, mmax) for coefficients that are
    not in matrix form e.g., flatten(). 

    """
    col_matrix = np.zeros((self.nmax+1, self.lmax+1, self.mmax+1))
    
    counter = 0
    for n in range(self.nmax+1):
        for l in range(self.lmax+1):
            for m in range(0, slef.lmax+1):
                col_matrix[n][l][m] = matrix[counter]
                counter +=1
    return col_matrix

class SCF_coeff:
    """
    A class with to work with SCF coefficients.

    """
    def __init__(self, coefficients, nmax, lmax, mmax):

        if len(coefficients) == 2:
            self.Snlm = coefficients[0]
            self.Tnlm = coefficients[1]
            self.Sshape = np.shape(self.Snlm)
            self.Tshape = np.shape(self.Tnlm)
            assert self.Sshape == self.Tshape

        elif len(coefficients) == 5:
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

        else :
            raise ValueError("coefficients elements should be 2 (Snlm, Tnlm) or 5 (Snlm, Tnlm, STnlm_var, TSnlm_var, STnlm_var)")
        
        self.nmax = nmax
        self.lmax = lmax
        self.mmax = mmax
        
        


    def mean_coeff():
        """
        Compute the mean of the coefficients from multiple files and return the mean values.
        # TODO: implement mean of covariance matrix
        """
        
        if len(self.Sshape) > 3:
            nsnaps = self.Sshape[0]
            S_mean = np.zeros_like(self.Snlm)
            T_mean = np.zeros_like(self.Tnlm)

        for k in range(nsnaps):
            S_mean[i] = np.mean(S_matrix[i], axis=0)
            T_mean[i] = np.mean(T_matrix[i], axis=0)

        return S_mean, T_mean 


    def smoothing(self, S, T, varS, varT):
        """
        Computes optimal smoothing following Eq.8 in Weinberg+96.
        
        returns:
        --------
        
        bs
        bt : 
        """
        bs = 1 / (1 + (varS/S**2))
        bt = 1 / (1 + (varT/T**2))
        if S == 0:
            bs=0
        if T == 0:
            bt=0
        return bs, bt

    def covariance_matrix(self, S, T, Svar, Tvar, STvar, pmass):
        """

        """
        cov_matrix = np.zeros((2,2))
        print("here", Svar, pmass, S)
        cov_matrix[0][0] = Svar - pmass*S**2
        cov_matrix[0][1] = STvar - pmass*S*T
        cov_matrix[1][1] = Tvar - pmass*T**2
        cov_matrix[1][0] = cov_matrix[0][1]

        return cov_matrix
        
        
    def smoothing_coeff_uncorrelated(self, S, T, Svar, Tvar, STvar, pmass, sn=0, verb=False, sn_out=0):

        #build matrix
        cov_matrix = self.covariance_matrix(S, T, Svar, Tvar, STvar, pmass)
        # SVD decomposition of the covariance matrix
        T_rot, v, TL = linalg.svd(cov_matrix)
        
        # Computes inverted transformation matrix
        T_rot_inv = linalg.inv(T_rot)

        # Variances of the coefficients in the uncorrelated base.
        varS = v[0]
        varT = v[1]
        
        ## uncorrelated coefficients
        coeff_base = np.array([S, T])
        S_unc, T_unc = np.dot(T_rot, coeff_base)
        b_S_unc, b_T_unc = self.smoothing(S_unc, T_unc, varS, varT)
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
        
        n=1
        if SN_coeff_unc < sn:
            S_smooth = 0 
            T_smooth = 0 
            SN_coeff = 0
            
        if sn_out == 0: 
            return S_smooth, T_smooth, n
        elif sn_out == 1:
            return S_smooth, T_smooth, n, SN_coeff

        
    def smooth_coeff(self, S, T, Svar, Tvar, STvar, pmass, verb=False, sn=0, sn_out=0):
        if sn_out==0:
            S_smooth, T_smooth, n_coeff = self.smoothing_coeff_uncorrelated(S, T, Svar, Tvar, STvar, pmass, sn, verb, sn_out)
            return S_smooth, T_smooth, n_coeff
        elif sn_out == 1:
            S_smooth, T_smooth, n_coeff, sn_coeff =  self.moothing_coeff_uncorrelated(S, T, Svar, Tvar, STvar, pmass, sn, verb, sn_out)
            return S_smooth, T_smooth, n_coeff, sn_coeff

    def smooth_coeff_matrix(self, pmass, sn, sn_out=False):
        S_matrix_smooth = np.zeros_like(self.Snlm)
        T_matrix_smooth = np.zeros_like(self.Tnlm)
        SN_coeff = np.zeros_like(S_matrix_smooth)
        n_coefficients = 0
        for n in range(self.nmax+1):
            for l in range(self.lmax+1):
                for m in range(self.lmax+1):
                    if sn_out :
                        S_matrix_smooth[n][l][m], T_matrix_smooth[n][l][m], n_coeff, SN_coeff[n][l][m] = self.smooth_coeff(self.Snlm[n][l][m],
                                                                                      self.Tnlm[n][l][m], 
                                                                                      self.Snlm_var[n][l][m], 
                                                                                      self.Tnlm_var[n][l][m],
                                                                                      self.STnlm_var[n][l][m],
                                                                                      pmass, verb=False, 
                                                                                      sn=sn,
                                                                                      sn_out=sn_out)
                
                    S_matrix_smooth[n][l][m], T_matrix_smooth[n][l][m], n_coeff  = self.smooth_coeff(self.Snlm[n][l][m],
                                                                                  self.Tnlm[n][l][m], 
                                                                                  self.Snlm_var[n][l][m], 
                                                                                  self.Tnlm_var[n][l][m],
                                                                                  self.STnlm_var[n][l][m],
                                                                                  pmass, verb=False, 
                                                                                  sn=sn,
                                                                                  sn_out=sn_out)
                    n_coefficients += n_coeff
        if sn_out == 0:
            return S_matrix_smooth, T_matrix_smooth, n_coefficients
        
        elif sn_out == 1:
            return S_matrix_smooth, T_matrix_smooth, n_coefficients, SN_coeff
     