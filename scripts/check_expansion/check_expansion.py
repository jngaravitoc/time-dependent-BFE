import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
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


def density_contour(S, T, grid_size, m, rs, snap, ngrid=128, delta_rho=False):
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

    #x.scatter(orbit[snap,7], orbit[snap,8], c='w')
    #ax.plot(orbit[:snap+1,7], orbit[:snap+1,8], lw='1.5', c='w', alpha=0.7)
    ax.add_patch(circle1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([-150, -50], [-220, -220], c='w')
    ax.text( -150, -210, r'$\rm{100\ kpc}$', c='w', fontsize=22)
    ax.text( -200, 220, r'$t={:0.1f}\ $Gyr'.format(snap*0.02), c='w', fontsize=22)

    plt.savefig('density_contour_{:03d}.png'.format(snap), bbox_inches='tight', dpi=300)
    plt.close()

# Plot enclosed mass?
# Plot delta rho plot


if __name__ == "__main__":
    for i in range(0, 100, 10):
        coefficients = "/mnt/home/nico/projects/time-dependent-BFE/data/expansion/MW2_100M_beta1_vir_OM3_G4/BFE_MW_grav_MO3_nsamp_1e6_host_snap_{:03d}".format(i)
        #hist_coefficients(S[:,:,0])
        #density_profile(S, T, rmin=0.10, rmax=400, m=1e10, rs=expansion[2][0], snapshot=snapshot)
        grid_size =  [-250, 250]
        expansion = bfe.ios.read_coefficients(coefficients)
        S, T = expansion[0]
        
        print(expansion[2][0])
        density_contour(S, T, grid_size, m=1e10, rs=expansion[2][0], snap=i)
        #particle_density_profile(snapshot)
    
