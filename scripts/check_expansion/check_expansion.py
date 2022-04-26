import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.style.use('/mnt/home/nico/matplotlib.mplstyle')

from astropy import units as u

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
import gala.integrate as gi
from gala.units import galactic

import pynbody

import bfe
import nba 

# Read data

# Plot coefficients 

def hist_coefficients(coeff_matrix, figname=0):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(np.log10(np.abs(coeff_matrix.T)), origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax)
    plt.show()
    if figname!=0:
        plt.savefig(figname, bbox_inches='tight')
        plt.close()
   
def particle_density_profile(snapshot):
    s = pynbody.load(snapshot)
    #h = s.halos()
    pynbody.analysis.angmom.faceon(s)
    s.physical_units()
    p = pynbody.analysis.profile.Profile(s, min=0.0, max=400, nbins=256, ndim=3)
    return p['rbins'], p['density']
    

# Plot density profiles
def density_profile(S, T, rmin, rmax, m, rs, snapshot):
    pot = gp.SCFPotential(m=m*u.Msun, r_s=rs*u.kpc, Snlm=S, Tnlm=T, units=galactic)
    x = np.logspace(np.log10(rmin), np.log10(rmax), 128)
    xyz = np.zeros((3, len(x)))
    xyz[0] = x
    sims_profile = particle_density_profile(snapshot)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, pot.density(xyz), label='SCF')
    ax.plot(sims_profile[0], sims_profile[1], label='Gadget')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.show()

# Plot enclosed mass?
# Plot delta rho plot


if __name__ == "__main__":
    coefficients = "/mnt/home/nico/projects/time-dependent-BFE/data/expansion/MW2_100M_beta1_vir_OM3_G4/BFE_MW_grav_MO3_ssc_COM_host_snap_000"
    snapshot = "/mnt/home/nico/ceph/MWLMC_sims/ICs/MW/MO3/MW2_100M_beta1_vir_OM3_G4_000.hdf5"
    expansion = bfe.ios.read_coefficients(coefficients)
    S, T = expansion[0]
    hist_coefficients(S[:,:,0])
    density_profile(S, T, rmin=0.10, rmax=400, m=1e10, rs=expansion[2][0], snapshot=snapshot)
    #particle_density_profile(snapshot)
    
