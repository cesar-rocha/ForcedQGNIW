"""
    Plots diagnostics of reference solutions
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from Utils import *
import scipy.signal as signal

plt.close('all')

pathi_nodrag = "output/new/512_nodrag/"
pathi_nowaves = "output/new/512_nowaves/"
pathi_reference = "output/new/512_reference/"
patho = "../writeup/figs/"

#
# parameters
#

# domain and grid
L = 2*np.pi*200e3
nx = 512
dx = L/nx

# environmental
f0 = 1.e-4
N = 0.005

# vertical plane wave
λz = 1000
λz = 850
m = 2*np.pi/λz

# vorticity dissipation
Tmu = 200*86400
mu = 1./Tmu
gamma = 4*mu
muw = gamma
Tgamma = 1./gamma

# no drag
#mu = 0.

#forcing
dk = 2*np.pi/L
kf = 8*dk
Lf = 1./kf
dkf = 1*dk
U0 = 0.5                   # guessed equilibrated RMS velocity
epsilon = (U0**2)*mu       # estimated energy input
sigma_q = np.sqrt(epsilon) # the standard deviation of the random forcing
sigma_w = 2*sigma_q

# hard
sigma_q = np.sqrt(epsilon)/2 # the standard deviation of the random forcing
sigma_w = 4*sigma_q

epsilon_q = (sigma_q**2)/2
epsilon_w = (sigma_w**2)/2

# time
dt = 0.000125*Tmu/4
tmax = 40.*Tgamma

#
# scaling non-dimensional numbers
#

# energy predictions (excluding factors of 1/2 and 2pi)
K  = epsilon_q/mu
Kw = epsilon_w/gamma

# wave equation parameters
lamb = N/f0/m
eta = f0*(lamb**2)

# scaling
U  = K**0.5
Uw = Kw**0.5 

T = 1./gamma
Teddy = 1./(U*kf)
Tf = 1./f0

PSI = U*Lf            # streamfunction
PHI = Uw              # wave velocity
Q   = (kf**2) * PSI   # vorticity
B   = m*eta*kf*PHI    # wave buoyancy

#
# get grid
#
setup = h5py.File(pathi_reference+"setup.h5","r")
x, y = setup['grid/x'][:]*kf, setup['grid/y'][:]*kf

#
# get diagnostics
#
diags_reference = h5py.File(pathi_reference+"diagnostics.h5","r")
diags_nodrag = h5py.File(pathi_nodrag+"diagnostics.h5","r")
diags_nowaves = h5py.File(pathi_nowaves+"diagnostics.h5","r")
time = diags_reference['time'][:]*gamma

#
# plotting
#

# scaling
#E = epsilon_q/gamma
Ew = PHI**2 / 2
E = Ew
K = (sigma_q**2)/mu/2
POWER = (sigma_w**2 / 2)

# energies
fig = plt.figure(figsize=(8.5,8.))
ax = fig.add_subplot(311)
plt.plot([-5,65],[K,K]/E,'k--')
pk = plt.plot(time,diags_reference['ke_qg'][:]/E)
pp = plt.plot(time,diags_nodrag['ke_qg'][:]/E)
plt.plot(time,diags_nowaves['ke_qg'][:]/E)
plt.xlim(-2,60)
plt.ylabel(r"Balanced kinetic energy $[\mathcal{K}/E]$")
plt.yticks([0,0.5,1.0,1.5,2.0])
plt.legend(loc=(0.35,-0.2),ncol=3)
remove_axes(ax,bottom=True)
plot_fig_label(ax,xc=0.025,yc=0.95,label=r'$\mathcal{K}$')

plt.text(47,1.28,'No drag')
plt.text(33,0.1,'Reference')
plt.text(15,0.575,'No waves')

ax = fig.add_subplot(312)
plt.plot(time,(diags_reference['pe_niw'][:])/E,label=r'Reference',color=pk[0].get_color())
plt.plot(time,(diags_nodrag['pe_niw'][:])/E,label=r'No drag',color=pp[0].get_color())
plt.ylabel(r"Wave potential energy $[\mathcal{P}/E]$")
plt.ylim(0,.35)
plt.yticks([0,0.1,.2,.3])
plt.xlim(-2,60)
remove_axes(ax)
plot_fig_label(ax,xc=0.025,yc=0.95,label=r'$\mathcal{P}$')
remove_axes(ax,bottom=True)
plt.text(47,0.3,'No drag')
plt.text(28,0.04,'Reference')

ax = fig.add_subplot(313)
plt.plot([-5,65],[Ew,Ew]/E,'k--')
plt.plot(time,(diags_reference['ke_niw'][:])/E,label=r'Reference',color=pk[0].get_color())
plt.plot(time,(diags_nodrag['ke_niw'][:])/E,label=r'No drag',color=pp[0].get_color())
plt.ylabel(r"Wave kinetic energy $[f_0\mathcal{A}/E]$")
plt.xlabel(r"Time $[t\,\,\gamma]$")
plt.xlim(-2,60)
remove_axes(ax)
plt.ylim(0,3.5)
plt.yticks([0,1.0,2.0,3.0],["0.0","1.0","2.0","3.0"])
plot_fig_label(ax,xc=0.025,yc=0.95,label=r'$\mathcal{A}$')
plt.savefig(patho+'energies_comparison' , pad_inces=0, bbox_inches='tight')

## averages
#Am =  diags['ke_niw'][eq].mean()
#Km =  diags['ke_qg'][eq].mean()
#Pm =  diags['pe_niw'][eq].mean()
#
## smooth out
#gamma_r = diags['gamma_r'][:]
#gamma_a = diags['gamma_a'][:]
#
## First, design the Buterworth filter
N  = 2    # Filter order
Wn = 0.05 # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')
#gamma_a_filt = signal.filtfilt(B,A, gamma_a)
#gamma_r_filt = signal.filtfilt(B,A, gamma_r)

# energy budgets

# reference
fig = plt.figure(figsize=(8.5,8.))
ax = fig.add_subplot(311)
plt.plot([-5,65],[0,0],'k--')
plt.plot(time,(diags_reference['Work_q'][:])/(time/gamma)/POWER,label=r'$-\leftangle \psi \xi_q \rightangle$')
plt.plot(time,diags_reference['ep_psi'][:]/POWER,label=r'$-2\mu\,\mathcal{K}$')
plt.plot(time,-(signal.filtfilt(B,A,diags_reference['gamma_r'][:]))/POWER,label=r'$-\Gamma_r$')
plt.plot(time,-(signal.filtfilt(B,A,diags_reference['gamma_a'][:]))/POWER,label=r'$-\Gamma_a$')
plt.plot(time,(diags_reference['xi_a'][:]+diags_reference['xi_r'][:])/POWER,label=r'$\Xi$')
plt.legend(loc=(0.25,1.035),ncol=5)
plt.ylabel(r"Power [$\dot \mathcal{K} \,/\, W$]")
plt.ylim(-.17,.17)
plt.xlim(-2,60)
remove_axes(ax,bottom=True)
plt.text(2,.145,'Balanced kinetic energy budget')
plot_fig_label(ax,xc=0.025,yc=0.95,label='a')

ax = fig.add_subplot(312)
plt.plot([-5,65],[0,0],'k--')
plt.plot(time,(signal.filtfilt(B,A,diags_reference['gamma_r'][:]))/POWER,label=r'$-\Gamma_r$')
plt.plot(time,(signal.filtfilt(B,A,diags_reference['gamma_a'][:]))/POWER,label=r'$-\Gamma_a$')
plt.plot(time,diags_reference['chi_phi'][:]/POWER,label=r'$-2\gamma\mathcal{P}$')
plt.ylim(-.17,.17)
plt.legend(loc=(.55,.975),ncol=3)
plt.ylabel(r"Power [$\dot \mathcal{P} \,/\, W$]")
remove_axes(ax,bottom=True)
plot_fig_label(ax,xc=0.025,yc=0.95,label='b')
plt.xlim(-2,60)
plt.text(2,.145,'Wave potential energy budget')

ax = fig.add_subplot(313)
plt.plot([-5,65],[0,0],'k--')
plt.plot(time,diags_reference['Work_w']/(time/gamma)/POWER,label=r'Re$\leftangle \phi^*\!\xi_\phi\rightangle$')
plt.plot(time,diags_reference['ep_phi']/POWER,label=r'$-2\gamma\, f_0 \mathcal{A}$')
plt.xlabel(r"$t\,\, \gamma$")
plt.ylabel(r"Power [$\dot f_0\mathcal{A} \,/\, W$]")
plt.legend(loc=1,ncol=2)
plt.ylim(-2,2)
plt.xlim(-2,60)
remove_axes(ax)
plot_fig_label(ax,xc=0.025,yc=0.95,label='c')
plt.text(2,1.75,'Wave action budget')
plt.savefig(patho+'K_and_P_and_A_budget_reference', pad_inces=0, bbox_inches='tight')

# no-drag
fig = plt.figure(figsize=(8.5,8.))
ax = fig.add_subplot(311)
plt.plot([-5,65],[0,0],'k--')
plt.plot(time,(diags_nodrag['Work_q'][:])/(time/gamma)/POWER,label=r'$-\leftangle \psi \xi_q \rightangle$')
plt.plot(time,diags_nodrag['ep_psi'][:]/POWER,label=r'$-2\mu\,\mathcal{K}$')
plt.plot(time,-(signal.filtfilt(B,A,diags_nodrag['gamma_r'][:]))/POWER,label=r'$-\Gamma_r$')
plt.plot(time,-(signal.filtfilt(B,A,diags_nodrag['gamma_a'][:]))/POWER,label=r'$-\Gamma_a$')
plt.plot(time,(diags_nodrag['xi_a'][:]+diags_nodrag['xi_r'][:])/POWER,label=r'$\Xi$')
plt.legend(loc=(0.25,1.035),ncol=5)
plt.ylabel(r"Power [$\dot \mathcal{K} \,/\, W$]")
plt.ylim(-.3,.3)
plt.xlim(-2,60)
remove_axes(ax,bottom=True)
plt.text(2,.27,'Balanced kinetic energy budget')
plot_fig_label(ax,xc=0.025,yc=0.95,label='a')

ax = fig.add_subplot(312)
plt.plot([-5,65],[0,0],'k--')
plt.plot(time,(signal.filtfilt(B,A,diags_nodrag['gamma_r'][:]))/POWER,label=r'$-\Gamma_r$')
plt.plot(time,(signal.filtfilt(B,A,diags_nodrag['gamma_a'][:]))/POWER,label=r'$-\Gamma_a$')
plt.plot(time,diags_nodrag['chi_phi'][:]/POWER,label=r'$-2\gamma\mathcal{P}$')
plt.ylim(-.3,.3)
plt.legend(loc=(.55,.975),ncol=3)
plt.ylabel(r"Power [$\dot \mathcal{P} \,/\, W$]")
remove_axes(ax,bottom=True)
plot_fig_label(ax,xc=0.025,yc=0.95,label='b')
plt.xlim(-2,60)
plt.text(2,.27,'Wave potential energy budget')

ax = fig.add_subplot(313)
plt.plot([-5,65],[0,0],'k--')
plt.plot(time,diags_nodrag['Work_w']/(time/gamma)/POWER,label=r'Re$\leftangle \phi^*\!\xi_\phi\rightangle$')
plt.plot(time,diags_nodrag['ep_phi']/POWER,label=r'$-2\gamma\, f_0 \mathcal{A}$')
plt.xlabel(r"$t\,\, \gamma$")
plt.ylabel(r"Power [$\dot f_0\mathcal{A} \,/\, W$]")
plt.legend(loc=1,ncol=2)
plt.ylim(-2,2)
plt.xlim(-2,60)
remove_axes(ax)
plot_fig_label(ax,xc=0.025,yc=0.95,label='c')
plt.text(2,1.75,'Wave action budget')
plt.savefig(patho+'K_and_P_and_A_budget_nodrag', pad_inces=0, bbox_inches='tight')



#
# calculate average budget after equilibration 
#

eq = time>20

## reference

# A budget
residual = diags_reference['Work_w'][eq]/(time[eq]/gamma)+diags_reference['ep_phi'][eq]
work_w_tot =  (diags_reference['Work_w'][eq]/(time[eq]/gamma) ).mean()
norm = work_w_tot.sum()
work_w_tot =  (diags_reference['Work_w'][eq]/(time[eq]/gamma) ).mean()/norm
ep_phi_tot = (diags_reference['ep_phi'][eq]).mean()/norm
residual_A = residual.mean()/norm

# P buget
residual = diags_reference['gamma_r'][eq]+diags_reference['gamma_a'][eq]+diags_reference['chi_phi'][eq]
gamma_tot = (diags_reference['gamma_r'][eq]+diags_reference['gamma_a'][eq]).mean()
norm = gamma_tot.sum()
gamma_r = (diags_reference['gamma_r'][eq]).mean()/norm
gamma_a = (diags_reference['gamma_a'][eq]).mean()/norm
chi_phi_tot = diags_reference['chi_phi'][eq].mean()/norm
residual_P = residual.mean()/norm

# K budget
residual =  -(diags_reference['gamma_r'][eq]+diags_reference['gamma_a'][eq]) + (diags_reference['xi_r'][eq]+diags_reference['xi_a'][eq]) + diags_reference['ep_psi'][eq].mean() + (diags_reference['Work_q'][eq]/(time[eq]/gamma))
work_q_tot = (diags_reference['Work_q'][eq]/(time[eq]/gamma)).mean()
norm = work_q_tot.sum()
work_q_tot = (diags_reference['Work_q'][eq]/(time[eq]/gamma)).mean()/norm
gamma_q_tot =  -(diags_reference['gamma_r'][eq]+diags_reference['gamma_a'][eq]).mean()/norm
xi_tot =  (diags_reference['xi_r'][eq]+diags_reference['xi_a'][eq]).mean()/norm
ep_psi_tot = diags_reference['ep_psi'][eq].mean()/norm
residual_K = residual.mean()/norm


##  no drag

# A budget
residual = diags_nodrag['Work_w'][eq]/(time[eq]/gamma)+diags_nodrag['ep_phi'][eq]
work_w_tot =  (diags_nodrag['Work_w'][eq]/(time[eq]/gamma) ).mean()
norm = work_w_tot.sum()
work_w_tot =  (diags_nodrag['Work_w'][eq]/(time[eq]/gamma) ).mean()/norm
ep_phi_tot = (diags_nodrag['ep_phi'][eq]).mean()/norm
residual_A = residual.mean()/norm

# P buget
residual = diags_nodrag['gamma_r'][eq]+diags_nodrag['gamma_a'][eq]+diags_nodrag['chi_phi'][eq]
gamma_tot = (diags_nodrag['gamma_r'][eq]+diags_nodrag['gamma_a'][eq]).mean()
norm = gamma_tot.sum()
gamma_r = (diags_nodrag['gamma_r'][eq]).mean()/norm
gamma_a = (diags_nodrag['gamma_a'][eq]).mean()/norm
chi_phi_tot = diags_nodrag['chi_phi'][eq].mean()/norm
residual_P = residual.mean()/norm

# K budget
residual =  -(diags_nodrag['gamma_r'][eq]+diags_nodrag['gamma_a'][eq]) + (diags_nodrag['xi_r'][eq]+diags_nodrag['xi_a'][eq]) + diags_nodrag['ep_psi'][eq].mean() + (diags_nodrag['Work_q'][eq]/(time[eq]/gamma))
work_q_tot = (diags_nodrag['Work_q'][eq]/(time[eq]/gamma)).mean()
norm = work_q_tot.sum()
work_q_tot = (diags_nodrag['Work_q'][eq]/(time[eq]/gamma)).mean()/norm
gamma_q_tot =  -(diags_nodrag['gamma_r'][eq]+diags_nodrag['gamma_a'][eq]).mean()/norm
xi_tot =  (diags_nodrag['xi_r'][eq]+diags_nodrag['xi_a'][eq]).mean()/norm
ep_psi_tot = diags_nodrag['ep_psi'][eq].mean()/norm
residual_K = residual.mean()/norm

## no waves

residual =  -(diags_nowaves['gamma_r'][eq]+diags_nowaves['gamma_a'][eq]) + (diags_nowaves['xi_r'][eq]+diags_nowaves['xi_a'][eq]) + diags_nowaves['ep_psi'][eq].mean() + (diags_nowaves['Work_q'][eq]/(time[eq]/gamma))
work_q_tot = (diags_nowaves['Work_q'][eq]/(time[eq]/gamma)).mean()
norm = work_q_tot.sum()
work_q_tot = (diags_nowaves['Work_q'][eq]/(time[eq]/gamma)).mean()/norm
gamma_q_tot =  -(diags_nowaves['gamma_r'][eq]+diags_nowaves['gamma_a'][eq]).mean()/norm
xi_tot =  (diags_nowaves['xi_r'][eq]+diags_nowaves['xi_a'][eq]).mean()/norm
ep_psi_tot = diags_nowaves['ep_psi'][eq].mean()/norm
residual_K = residual.mean()/norm










