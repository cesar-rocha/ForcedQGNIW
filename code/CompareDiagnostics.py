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
time = diags_reference['time'][:]*gamma

#
# plotting
#

# scaling
#E = epsilon_q/gamma
Ew = PHI**2 / 2
E = Ew
POWER = (sigma_w**2 / 2)

# energies
fig = plt.figure(figsize=(8.5,8.))
ax = fig.add_subplot(311)
pk = plt.plot(time,diags_reference['ke_qg'][:]/E)
pp = plt.plot(time,diags_nodrag['ke_qg'][:]/E)
plt.ylabel(r"Balanced kinetic energy $[\mathcal{K}/E]$")
plt.yticks([0,0.5,1.0,1.5,2.0])
plt.legend(loc=(0.35,-0.2),ncol=3)
remove_axes(ax,bottom=True)
plot_fig_label(ax,xc=0.025,yc=0.95,label=r'$\mathcal{K}$')

plt.text(47,1.28,'No drag')
plt.text(28,0.1,'Reference')

ax = fig.add_subplot(312)
plt.plot(time,(diags_reference['pe_niw'][:])/E,label=r'Reference',color=pk[0].get_color())
plt.plot(time,(diags_nodrag['pe_niw'][:])/E,label=r'No drag',color=pp[0].get_color())
plt.ylabel(r"Wave potential energy $[\mathcal{P}/E]$")
plt.ylim(0,.35)
plt.yticks([0,0.1,.2,.3])
remove_axes(ax)
plot_fig_label(ax,xc=0.025,yc=0.95,label=r'$\mathcal{P}$')
remove_axes(ax,bottom=True)
plt.text(47,0.3,'No drag')
plt.text(28,0.04,'Reference')

ax = fig.add_subplot(313)
plt.plot(time,(diags_reference['ke_niw'][:])/E,label=r'Reference',color=pk[0].get_color())
plt.plot(time,(diags_nodrag['ke_niw'][:])/E,label=r'No drag',color=pp[0].get_color())
plt.ylabel(r"Wave kinetic energy $[f_0\mathcal{A}/E]$")

plt.xlabel(r"Time $[t\,\,\gamma]$")
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
#N  = 2    # Filter order
#Wn = 0.05 # Cutoff frequency
#B, A = signal.butter(N, Wn, output='ba')
#gamma_a_filt = signal.filtfilt(B,A, gamma_a)
#gamma_r_filt = signal.filtfilt(B,A, gamma_r)

# energies
fig = plt.figure(figsize=(8.5,5.))
ax = fig.add_subplot(211)
pa = plt.plot(time,diags['ke_niw']/E,label=r'$f_0\mathcal{A}$')
pk = plt.plot(time,diags['ke_qg'][:]/E,label=r'$\mathcal{K}$')
pp = plt.plot(time,diags['pe_niw']/E,label=r'$\mathcal{P}$')
#plt.xlabel(r"Time $[t\,\,\gamma]$")
plt.ylabel(r"Energy $[\mathcal{E}/E]$")
plt.yticks(np.arange(0,4))
plt.legend(loc=(0.35,-0.2),ncol=3)
remove_axes(ax,bottom=True)
plot_fig_label(ax,xc=0.025,yc=0.95,label='a')

ax = fig.add_subplot(212)
plt.plot(time,np.zeros_like(time),'--',color='0.5')
plt.plot(time,(diags['ke_qg'][:]-Km)/E,label=r'$\mathcal{K}$',color=pk[0].get_color())
plt.plot(time,(diags['pe_niw']-Pm)/E,label=r'$\mathcal{P}$',color=pp[0].get_color())
plt.xlabel(r"Time $[t\,\,\gamma]$")
plt.ylabel(r"Energy diff $[\Delta \mathcal{E}/E]$")
plt.ylim(1.,1.)
plt.yticks([-1.,-0.5,0,0.5,1.])
remove_axes(ax)
plot_fig_label(ax,xc=0.025,yc=0.95,label='b')
plt.savefig(patho+'energies_nodrag' , pad_inces=0, bbox_inches='tight')

# energy budgets
fig = plt.figure(figsize=(8.5,8.))

ax = fig.add_subplot(311)
plt.plot(time,np.zeros_like(time),'--',color='0.5')
plt.plot(time,(diags['Work_q'][:])/(time/gamma)/POWER,label=r'$-\leftangle \psi \xi_q \rightangle$')
plt.plot(time,diags['ep_psi'][:]/POWER,label=r'$-2\mu\,\mathcal{K}$')
plt.plot(time,-(gamma_r_filt+gamma_a_filt)/POWER,label=r'$-\Gamma$')
plt.plot(time,(diags['xi_a'][:]+diags['xi_r'][:])/POWER,label=r'$\Xi$')
plt.legend(loc=(0.45,1.025),ncol=4)
plt.ylabel(r"Power [$\dot \mathcal{K} \,/\, W$]")
plt.ylim(-.4,.4)
remove_axes(ax,bottom=True)

plot_fig_label(ax,xc=0.025,yc=0.95,label='a')

ax = fig.add_subplot(312)
plt.plot(time,np.zeros_like(time),'--',color='0.5')
plt.plot(time,gamma_r_filt/POWER,label=r'$\Gamma_r$')
plt.plot(time,gamma_a_filt/POWER,label=r'$\Gamma_a$')
plt.plot(time,diags['chi_phi'][:]/POWER,label=r'$-2\gamma\mathcal{P}$')
plt.ylim(-.4,.4)
plt.legend(loc=3,ncol=3)
plt.ylabel(r"Power [$\dot \mathcal{P} \,/\, W$]")
remove_axes(ax,bottom=True)
plot_fig_label(ax,xc=0.025,yc=0.95,label='b')

ax = fig.add_subplot(313)
plt.plot(time,np.zeros_like(time),'--',color='0.5')
plt.plot(time,diags['Work_w']/(time/gamma)/POWER,label=r'Re$\leftangle \phi^*\!\xi_\phi\rightangle$')
plt.plot(time,diags['ep_phi']/POWER,label=r'$-2\gamma\, f_0 \mathcal{A}$')
plt.xlabel(r"$t\,\, \gamma$")
plt.ylabel(r"Power [$\dot \mathcal{P} \,/\, W$]")
plt.legend(loc=1,ncol=2)
remove_axes(ax)
plot_fig_label(ax,xc=0.025,yc=0.95,label='c')

plt.savefig(patho+'K_and_P_and_A_budget_nodrag', pad_inces=0, bbox_inches='tight')


#
# calculate average budget after equilibration 
#

# A budget
residual = diags['Work_w'][eq]/(time[eq]/gamma)+diags['ep_phi'][eq]
work_w_tot =  (diags['Work_w'][eq]/(time[eq]/gamma) ).mean()
ep_phi_tot = (diags['ep_phi'][eq]).mean()
residual_A = residual.mean()/work_w_tot

# P buget
residual = diags['gamma_r'][eq]+diags['gamma_a'][eq]+diags['chi_phi'][eq]
gamma_tot = (diags['gamma_r'][eq]+diags['gamma_a'][eq]).mean()
chi_phi_tot = diags['chi_phi'][eq].mean()/gamma_tot
residual_P = residual.mean()/gamma_tot

# K budget
residual =  -(diags['gamma_r'][eq]+diags['gamma_a'][eq]) + (diags['xi_r'][eq]+diags['xi_a'][eq]) + diags['ep_psi'][eq].mean() + (diags['Work_q'][eq]/(time[eq]/gamma))
work_q_tot = (diags['Work_q'][eq]/(time[eq]/gamma)).mean()
gamma_q_tot =  -(diags['gamma_r'][eq]+diags['gamma_a'][eq]).mean()/work_q_tot
xi_tot =  (diags['xi_r'][eq]+diags['xi_a'][eq]).mean()/work_q_tot
ep_psi_tot = diags['ep_psi'][eq].mean()/work_q_tot
residual_K = residual.mean()/work_q_tot

