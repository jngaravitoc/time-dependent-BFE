import numpy as np
import scf_utils


if __name__ == "__main__":
    # READ coefficients test
    coeff_filename = "../../data/expansion/MWLMC5_snap_1e6/BFE_MWLMC5_grav_MO5_ss_COM_n20_20_nsamp_1e6_host_snap_"
    #coefficients = scf_utils.read_coefficients(coeff_filename)
    S_coefficients, T_coefficients, rcom = scf_utils.array_coefficients(coeff_filename, 207, 209)
    print(np.shape(S_coefficients))
    print(np.shape(T_coefficients))
    pint(np.shape(rcom))
    #print(len(coefficients[0]))
    #print(coefficients[1])
    #print(coefficients[2])
    #print(coefficients[3])
    #nmax = coefficients[1][0]
    #lmax = coefficients[1][1]
    #mmax = coefficients[1][2]
    #pmass = coefficients[2][1]
    ##  Smoothing coefficients test
    #SCFcoeff = scf_utils.SCF_coeff(coefficients[0], nmax, lmax, mmax)
    #SCFcoeff.smooth_coeff_matrix(pmass, 4)
