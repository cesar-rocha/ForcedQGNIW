"""
    Forced-disspative QG: still testing.
"""
import timeit

import matplotlib.pyplot as plt
plt.rcParams['contour.negative_linestyle'] = 'dashed'
import numpy as np

from niwqg import CoupledModel
from niwqg import UnCoupledModel
from niwqg import InitialConditions as ic
import cmocean

from pyspec import spectrum

from Utils import *

plt.close('all')

# parameters
nx = 512

f0 = 1.e-4
N = 0.005
L = 2*np.pi*200e3

λz = 750
m = 2*np.pi/λz

# dissipation
Tmu = 200*86400
mu = 1./Tmu

Tgamma = 200*86400/4
gamma = 1./Tgamma

dt = 0.000125*Tmu/4
tmax = 4*Tmu

#dt = 0.000125*Tgamma
#tmax = 12*Tgamma

#forcing
dk = 2*np.pi/L

kf = 8*dk
dkf = 1*dk
Lf = 2*np.pi/kf
# energy input
U0 = 0.5
epsilon = (U0**2)*mu
sigma = np.sqrt(epsilon)

path = "output/reference512_filter"

nu4  = 0e8
nu4w = 0.e8

# Forced only dynamics
model = CoupledModel.Model(L=L,nx=nx, tmax = tmax,dt = dt, twrite=200,
                    nu4=nu4,mu=mu,nu4w=0*nu4w,nu=0,nuw=0,muw=0*mu,
                    use_filter=True,save_to_disk=True,
                    tsave_snapshots=100,path=path,
                    U = 0., tdiags=100,
                    f=f0,N=N,m=m,
                    wavenumber_forcing=kf,width_forcing=dkf,
                    sigma_q=sigma, sigma_w=0.,
                    use_mkl=True,nthreads=8)

# non-dimensional numbers
lamb = N/f0/m
eta = f0*(lamb**2)
epsilon_q = sigma**2
PSI = sigma/np.sqrt(model.mu)/kf
#PHI = np.sqrt(model.epsilon_w/mu)
PHI = 1.
hslash = eta/PSI

# initial conditions
model.set_q(np.zeros([model.nx]*2))
model.set_phi(np.zeros([model.nx]*2)+0j)
model._invert()

# run the model
model.run()

# get snapshot of wave fields
uw, vw, ww, pw ,bw = wave_fields(model.phi, model.f0, lamb**2, model.t, model.k, model.l, model.m, z=0)

# # plot spectrum and a realization of the forcing
Q = (2*np.pi)**-2 * epsilon/(mu**2 / kf**2)

# plot potential vorticity
fig = plt.figure(figsize=(6.5,5))
cv = np.linspace(-.2,.2,100)
cphi = np.linspace(0,3.,100)

ax = fig.add_subplot(111,aspect=1)
im1 = plt.contourf(model.x/Lf,model.y/Lf,model.q/Q,cv,\
                cmin=-0.2,cmax=0.2,extend='both',cmap=cmocean.cm.curl)
plt.xlabel(r'$x\, k_f/2\pi$')
plt.ylabel(r'$y\, k_f/2\pi$')

plt.title(r"$t\,\,\mu = %3.2f$" % (model.t*model.mu))
plt.colorbar(im1, ticks=[-.2,-.1,0,.1,.2],label=r'Potential vorticity $[q/Q]$')

plt.savefig('figs/snapshots_pv_qg', pad_inces=0, bbox_inches='tight')

time = model.diagnostics['time']['value']
KE_qg = model.diagnostics['ke_qg']['value']
KE_niw = model.diagnostics['ke_niw']['value']
PE_niw = model.diagnostics['pe_niw']['value']

ENS_qg = model.diagnostics['ens']['value']
ep_psi = model.diagnostics['ep_psi']['value']
smalldiss_psi = model.diagnostics['smalldiss_psi']['value']

chi_q =  model.diagnostics['chi_q']['value']

energy_input = model.diagnostics['energy_input']['value']
wave_energy_input = model.diagnostics['wave_energy_input']['value']
ep_phi = model.diagnostics['ep_phi']['value']
chi_phi = model.diagnostics['chi_phi']['value']
smalldiss_phi = model.diagnostics['smalldiss_phi']['value']
smallchi_phi = model.diagnostics['smallchi_phi']['value']
gamma_r = model.diagnostics['gamma_r']['value']
gamma_a = model.diagnostics['gamma_a']['value']
xi_r = model.diagnostics['xi_r']['value']
xi_a = model.diagnostics['xi_a']['value']
work = model.diagnostics['Work_q']['value']/time
work[0] = 0

def remove_axes(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

dt = time[1]-time[0]
dKE_qg = np.gradient(KE_qg,dt)

# plot energy
fig = plt.figure(figsize=(8.5,4.))
E = epsilon_q/model.mu/2

ax = fig.add_subplot(111)
plt.plot(time*model.mu*4,np.ones_like(time),'r--')
plt.plot(time*model.mu*4,KE_qg/E,label=r'$\mathcal{K}$')
plt.xlabel(r"Time $[t\,\,\gamma]$")
plt.ylabel(r"Energy $[\mathcal{K} \,\, 2 \mu/\sigma_q^2]$")
#plt.legend(loc=5)
remove_axes(ax)
plt.savefig('figs/energy_and_action_qg' , pad_inces=0, bbox_inches='tight')

# a KE_niw energy budget
residual = -gamma_r-gamma_a+ep_psi+smalldiss_psi-dKE_qg+energy_input
POWER = (E)**2/Tmu

fig = plt.figure(figsize=(9.5,4.))
ax = fig.add_subplot(111)
plt.plot(time*model.mu,np.ones_like(time)*epsilon_q/POWER,label=r'$\epsilon_q$')
plt.plot(time*model.mu,np.ones_like(time)*work/POWER,label=r'$W/t$')
plt.plot(time*model.mu,ep_psi/POWER,label=r'$-2\mu\mathcal{K}$')
plt.legend(loc=3,ncol=2)
plt.ylabel(r"Power [$\dot \mathcal{K} \,/\, W$]")
plt.xlabel(r"$t\,\, \mu$")
remove_axes(ax)
plt.savefig('figs/KE_qg_budget' , pad_inces=0, bbox_inches='tight')

# calculate spectrum
E = 0.5 * np.abs(model.wv*model.ph)**2
ki, Ei = spectrum.calc_ispec(model.kk, model.ll, E)
np.savez("spec_qg-only",k=ki,K=Ei,kf=kf)

# plot spectra
spec_qg = np.load("spec_qg-only.npz")
S = epsilon_q/model.mu/2/kf

fig = plt.figure(figsize=(6.5,4.5))
ax = fig.add_subplot(111)
plt.loglog(ki/kf,spec_qg['K']/S,'k',linewidth=2,label=r'$\mathcal{K}$, waveless QG')
plt.loglog(ki/kf,Ei/S,linewidth=2,label=r'$\mathcal{K}$, QG-NIW')
#plt.loglog(ki/kf,Eiphi,linewidth=2,label=r'$\mathcal{A}$')
#plt.loglog(ki/kf,Piphi,linewidth=2,label=r'$\mathcal{P}$')
plt.ylim(1e-7,1e1)
plt.legend(loc=3)
remove_axes(ax)
plt.xlabel(r"Wavenumber [$k/k_f$]")
plt.ylabel(r"Balanced kinetic energy density [$\mathcal{K} 2 \mu k_f \epsilon_q^{-1}$]")
plt.savefig('figs/spectral' , pad_inces=0, bbox_inches='tight')
