import numpy as np
import scf_utils


if __name__ == "__main__":
    # READ coefficients test
    coeff_filename = "./BFE_MWLMC5_grav_MO5_ss_COM_n20_20_nsamp_1e6_host_snap_207"
    coefficients = scf_utils.read_coefficients(coeff_filename)
    print(len(coefficients[0]))
    print(coefficients[1])
    print(coefficients[2])
    print(coefficients[3])
    nmax = coefficients[1][0]
    lmax = coefficients[1][1]
    mmax = coefficients[1][2]
    pmass = coefficients[2][1]
    ##  Smoothing coefficients test
    SCFcoeff = scf_utils.SCF_coeff(coefficients[0], nmax, lmax, mmax)
    SCFcoeff.smooth_coeff_matrix(pmass, 4)
