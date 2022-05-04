import numpy as np
import matplotlib.pyplot as plt
import linecache

from astropy.constants import Constant
import astropy.constants
import astropy.units as u
import numpy as np

G_unit = u.m**3 / u.kg / u.s**2
astropy.constants.G = Constant(
    'G', 
    'Gravitational constant',
    (43007.1 * u.kpc * u.km**2 / u.s**2 / u.Msun / 1E10 ).to_value(G_unit),
    'm3 / (kg s2)', 
    np.nan, 
    system='si',
    reference='GADGET'
)

# gala
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
#from gala.units import galactic
#import gala.integrate as gi
from gala.units import galactic
import sys

sys.path.append("/mnt/home/nico/projects/time-dependent-BFE/scripts/scf_coefficients/")
import scf_utils


def analytic_orbits(pos_ics, vel_ics, dt, nsnaps):
    halo_teo = gp.HernquistPotential(m=m2*u.Msun, c=rs_halo*u.kpc,
                                     units=galactic)
    w0 = gd.PhaseSpacePosition(pos=pos_ics*u.kpc, vel=vel_ics*u.km/u.s)
    orbit_halo_teo = gp.Hamiltonian(halo_teo).integrate_orbit(w0, dt=dt*u.Gyr, n_steps=nsnaps-1)
    pos = orbit_halo_teo.xyz.value.T
    vel = orbit_halo_teo.v_xyz.value.T
    return pos, vel

def scf_orbits(S, T, pos_ics, vel_ics, dt, nsnaps, rcom=0, com_xj=1, com_vj=1):
    t = nsnaps*dt
    dt2 = dt/HR

    if rcom==1:
        print('using moving COM')
        halo_t = gp.scf.InterpolatedSCFPotential(m=1e10*u.Msun, r_s=rs_halo*u.kpc, Sjnlm=S, Tjnlm=T,
                                                 tj=tj*u.Gyr,
                                                 com_xj=com_xj*u.kpc, com_vj=np.zeros_like(com_xj),
                                                 units=galactic)
    else:
        print('not using moving COM')
        halo_t = gp.scf.InterpolatedSCFPotential(m=1e10*u.Msun, r_s=rs_halo*u.kpc, Sjnlm=S, Tjnlm=T,
                                                 tj=np.arange(0, t, dt)*u.Gyr,
                                                 com_xj=np.zeros_like(com_xj), com_vj=np.zeros_like(com_xj),
                                                 units=galactic)

    if len(np.shape(S))==3:
        print("here")
        halo_t = gp.scf.SCFPotential(m=1e10*u.Msun, r_s=rs_halo, Snlm=S, Tnlm=T,
                                      units=galactic)

    w0 = gd.PhaseSpacePosition(pos=pos_ics*u.kpc,  vel=vel_ics*u.km/u.s)

    orbit_halo = gp.Hamiltonian(halo_t).integrate_orbit(w0, dt=dt2*u.Gyr, n_steps=HR*nsnaps-1)
    pos_all_t = orbit_halo.xyz.value.T 
    vel_all_t = orbit_halo.v_xyz.value.T
    pot_all_t = halo_t.energy(pos_all_t.T * u.kpc).value
    return pos_all_t, vel_all_t ,pot_all_t

def orbits_plots(time, qs, labels, ref_orbit=0):

    cl = ["#00429d","#00ab7d","#2c2c2c","#ff005e","#810000"]
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axs = ax.flatten()
    norbits = len(qs)
    print(norbits)
    for i in range(norbits):
        print(i, np.shape(qs[i]))
        print(time[i], qs[i])
        axs[0].plot(time[i], np.sqrt(np.sum(qs[i]**2, axis=1)), label=labels[i], c=cl[i], lw=2)
        #axs[1].plot(time[i], np.sqrt(np.sum((qs[i][::10]-qs[ref_orbit])**2, axis=1)))
        axs[0].legend(ncol=5, fontsize=13)
        #axs[0].set_xlim(-0.2, 5)
        #axs[0].set_ylim(0, 40)

    axs[1].set_title('Residuals with respect to Gadget', fontsize=18)
    axs[1].set_xlabel('Time [Gyrs]')

    axs[0].set_ylabel(r'$r_{gal} \rm{[kpc]}$')
    axs[1].set_ylabel(r'$r_{gal} \rm{[kpc]}$')
    plt.savefig("bfe_test_orbit.png", bbox_inches='tight')
    plt.close()
    return fig


def center_plots(pos_com, vel_com):

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    rcom = np.sqrt(np.sum(pos_com**2, axis=1))
    vcom = np.sqrt(np.sum(vel_com**2, axis=1))
    nsnaps = np.linspace(0, len(rcom), len(rcom))
    ax[0].plot(nsnaps, rcom)
    ax[1].plot(nsnaps, vcom)

    ax[0].set_xlabel('snapshot')
    ax[1].set_xlabel('snapshot')
    ax[0].set_ylabel(r'$r_{com} \rm{[kpc]}$')
    ax[1].set_ylabel(r'$c_{com} \rm{[km/s]}$')
    plt.savefig('com_vcom.png', bbox_inches='tight')
    plt.close()
	
    return fig



if __name__ == "__main__":
    ########### Parameters defintion #########################
    coeff_filename = "../../data/expansion/MWLMC5_snap_1e6/BFE_MWLMC5_grav_MO5_ss_COM_n20_20_nsamp_1e6_host_snap_"

    Sjnlm, Tjnlm, rcom, exp_params = scf_utils.array_coefficients(coeff_filename, 0, 100)
    
 
    m2 = 1.34E12 # Msun
    rs_halo = 40.85 # kpc
    ## ICS:
    r_init = [10, 0, 0] # kpc
    v_init = [0, 100, 0] # km/s
    dt = 0.001 #* u.Gyr
    nsnaps = 100 # nsteps
    HR = 10
    t_orbit = np.arange(0, nsnaps*dt, dt)
    t_orbit_HR = np.arange(0, nsnaps*dt, dt/HR)

	# Integrating orbits
    pos_teo, vel_teo = analytic_orbits(r_init, v_init, dt, nsnaps)
    pos_scf, vel_scf = scf_orbits(Sjnlm[0], Tjnlm[0], r_init, v_init, dt, nsnaps)
    #pos_scf_rcom, vel_scf_rcom = scf_orbits(Sjnlm, Tjnlm, r_init, v_init], dt,  nsnaps_nbody, rcom=1, com_xj=rcom)

    f = 0.9997033035183764
    #print(pos_teo[:,0])
    #labels = ['SCF rcom', '-SCF rcom', 'Analytic',  'Gadget-4']
    orbits_plots([t_orbit], np.array([pos_teo]), ['Analytic'])
    #orbits_plots([t_orbit2, t_orbit2, t_orbit, t_orbit], np.array([pos_scf_rcom[0], pos_scf_rcom2[0], pos_teo[0],  xyz_all[pp]]), labels, pp, 1)
    #	np.savetxt(path_to_orbit + orbit_name, np.array([pos_scf_rcom[0][:,0], pos_scf_rcom[0][:,1], pos_scf_rcom[0][:,2], vel_scf_rcom[0][:,0], vel_scf_rcom[0][:,1], vel_scf_rcom[0][:,2], pot_scf_rcom[0]]).T)
