#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import gala
import gala.potential as gp
import gala.dynamics as gd
import bfe
from bfe.coefficients import fields
from scipy.spatial.transform import Rotation as Rot
import nba
import sys


def sample_hern(size=1):
    mu = np.random.random(size=size)
    return mu**0.5 / (1-mu**0.5)

def spherical_hern(a, b, c, theta_rot, size=1):
    phi = np.random.uniform(0, 2*np.pi, size=size)
    theta = np.arccos(2*np.random.random(size=size) - 1)
    r = sample_hern(size)
    
    xyz = np.zeros((size, 3))
    xyz[:,0] = r * np.cos(phi) * np.sin(theta) / a
    xyz[:,1] = r * np.sin(phi) * np.sin(theta) / b
    xyz[:,2] = r * np.cos(theta) / c
    
    rot = Rot.from_euler('z', theta_rot, degrees=True)
    rot_xyz = rot.apply(xyz)
    #xa = rot_xyz[0]
    #yb = rot_xyz[1]
    #zc = rot_xyz[2]
    
    return rot_xyz

def flattened_hernquist_sample(x, y, z, M, a, b, c, rs, theta):
    ## implement rotation
    xyz = np.array([x/a, y/b, z/c])
    rot = Rot.from_euler('x', theta, degrees=True)
    rot_xyz = rot.apply(xyz)
    xa = rot_xyz[0]
    yb = rot_xyz[1]
    zc = rot_xyz[2]
    #s = np.sqrt((x/a)**2 + (y/b)**2 + (z/c)**2)
    s = np.sqrt((xa)**2 + (yb)**2 + (zc)**2)
    return hernquist_density(s, M, rs)


def hernquist_density(r, M, a):
    return M*a / (2*np.pi) / (r*(r+a)**3)

def flattened_hernquist_density(x, y, z, M, a, b, c, rs, theta):
    ## implement rotation
    xyz = np.array([x/a, y/b, z/c])
    rot = Rot.from_euler('x', theta, degrees=True)
    rot_xyz = rot.apply(xyz)
    xa = rot_xyz[0]
    yb = rot_xyz[1]
    zc = rot_xyz[2]
    #s = np.sqrt((x/a)**2 + (y/b)**2 + (z/c)**2)
    s = np.sqrt((xa)**2 + (yb)**2 + (zc)**2)
    return hernquist_density(s, M, rs)





def compute_triax_halo_coeff(size, a, b, c, theta, nmax=10, lmax=10):
    xyz_hern = spherical_hern(a, b, c, theta, size)
    print('Done generating halo')
    Snlm_hern, Tnlm_hern = gp.scf.compute_coeffs_discrete(xyz_hern, 
                                                          mass=np.ones(len(xyz_hern))/len(xyz_hern), 
                                                          nmax=nmax, lmax=lmax, r_s=1)
    print('Done computing coefficients')
    halo_coefs = bfe.coefficients.Coeff_properties(Snlm_hern, Tnlm_hern, nmax=nmax+1, lmax=lmax+1)
    U = halo_coefs.U_all()
    return -U, [Snlm_hern, Tnlm_hern]




def make_shape_coeff_plot(pot, U , title):
    grid = np.linspace(-6, 6, 128)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _ = pot.plot_contours((grid, grid, 0), ax=axes[0])
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')

    _ = pot.plot_contours((grid, 0, grid), ax=axes[1])
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$z$')

    im = axes[2].imshow(np.log10(np.sum(U[:,:,:], axis=0)).T, origin='lower', vmin=-0.5, vmax=-4, cmap='Blues')
    fig = plt.colorbar(im, ax=axes[2])
    axes[2].set_xlabel('$l$')
    axes[2].set_ylabel('$m$')
    plt.suptitle(title, fontsize=17)
    
    return fig


def spherical_shell(R, size=1):
    phi = np.random.uniform(0, 2*np.pi, size=size)
    theta = np.arccos(2*np.random.random(size=size) - 1)
    xyz = np.zeros((size, 3))
    xyz[:,0] = R * np.cos(phi) * np.sin(theta) 
    xyz[:,1] = R * np.sin(phi) * np.sin(theta) 
    xyz[:,2] = R * np.cos(theta) 
    return xyz



def get_fields_radial_contribution(coeff, rbins, nmax, lmax, ngrid):
    """
    Function that compute the contribution of the l-order term to the density and potential
    
    
    
    """
    rho_lr = np.zeros((len(rbins), lmax+1))
    pot_lr = np.zeros((len(rbins), lmax+1))

    rho_shell = np.zeros(len(rbins))
    pot_shell = np.zeros(len(rbins))

    for r in range(len(rbins)):
        print(rbins[r])
        pos_rshell = spherical_shell(rbins[r], size=ngrid)
        rho_shell[r] = np.sum(np.abs(halo_pot.density(pos_rshell.T)))
        pot_shell[r] = np.sum(np.abs(halo_pot.energy(pos_rshell.T)))

        test_f = fields.BFEpot(pos_rshell, coeff[0], coeff[1], nmax=nmax+1, lmax=lmax+1, rs=1, G=1, M=1)
        pot_nlm = np.zeros((nmax+1, lmax+1, lmax+1))
        dens_nlm = np.zeros((nmax+1, lmax+1, lmax+1))

        for n in range(nmax+1):
            for l in range(lmax+1):
                for m in range(l+1):
                    pot_nlm[n,l,m] = np.sum(np.abs(test_f.potential_nlm(n,l,m)))
                    dens_nlm[n,l,m] = np.sum(np.abs(test_f.density_nlm(n,l,m)))
        rho_lr[r] = np.sum(np.abs(np.sum((dens_nlm[:,:,:]), axis=2)), axis=0)
        pot_lr[r] = np.sum(np.abs(np.sum((pot_nlm[:,:,:]), axis=2)), axis=0)
    return rho_lr, pot_lr, rho_shell, pot_shell

    np.savetxt(filename, np.array([U.flatten(), S.flatten(), T.flatten()]).T


"""
a = major
b = intermediate
c = minor
"""

npart = 100000
bval = np.linspace(0.5, 1, 10)
cval = np.linspace(0.5, 1, 10)
theta = 30

for b in bval:
    for c in cval:
        U, coeff_halo = compute_triax_halo_coeff(npart, a, b, c, theta, nmax=10, lmax=5)
        S = coeff_halo[0]
        T = coeff_halo[0]
        filename = "triaxial_halo_{}_{}.txt".format(b, c)
        np.savetxt(filename, np.array([U.flatten(), S.flatten(), T.flatten()]).T


