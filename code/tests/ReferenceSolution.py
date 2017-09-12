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

plt.close('all')

# parameters
nx = 128
f0 = 1.e-4
N = 0.005
L = 2*np.pi*200e3

λz = 400
m = 2*np.pi/λz

# dissipation
Tmu = 200*86400
mu = 1./Tmu

dt = 0.000125*Tmu
tmax = 5*Tmu

#forcing
dk = 2*np.pi/L

kf = 8*dk
dkf = 1*dk

# energy input
U0 = 0.5
epsilon = (U0**2)*mu

path = "output/reference_2"

nu4  = 1e10
nu4w = 1e10

# Force only dynamics
model = CoupledModel.Model(L=L,nx=nx, tmax = tmax,dt = dt, twrite=20,
                    nu4=nu4,mu=mu,nu4w=nu4w,nu=0,nuw=0,muw=mu/4, use_filter=False,save_to_disk=True,
                    tsave_snapshots=25,path=path,
                    U = 0., tdiags=1,
                    f=f0,N=N,m=m,
                    wavenumber_forcing=kf,width_forcing=dkf,
                    epsilon_q=epsilon, epsilon_w=4*epsilon )

model.set_q(np.zeros([model.nx]*2))
model.set_phi(np.zeros([model.nx]*2)+0j)
model._invert()

# run the model
model.run()

# # plot spectrum and a realization of the forcing
# fig = plt.figure(figsize=(8.5,4.5))
Q = (2*np.pi)**-2 * epsilon/(mu**2 / kf**2)
#
# ax = fig.add_subplot(121,aspect=1)
# plt.contourf(np.fft.fftshift(model.k)/kf,np.fft.fftshift(model.l)/kf,\
#                 1e3*np.fft.fftshift(model.spectrum_qg_forcing),40)
# plt.xlim(-2.5,2.5)
# plt.ylim(-2.5,2.5)
# plt.xlabel(r"$k/k_f$")
# plt.ylabel(r"$l/k_f$")
# plt.colorbar(orientation="horizontal",ticks=[0,2,4,6,8],shrink=0.8,\
#                 label=r"Power spectrum $[10^{-3}\,\, \mathcal{F}_{kl}]$")
#
# ax = fig.add_subplot(122,aspect=1)
Lf = 2*np.pi/kf
cf = np.linspace(-1,1,40)
# plt.contourf(model.x/Lf,model.y/Lf,1e6*np.fft.ifft2(model.forceh)/np.sqrt(model.dt)/Q,\
#                     cf,cmap=cmocean.cm.curl,extend='both')
# plt.colorbar(orientation="horizontal",shrink=0.8,ticks=[-1,-.5,0,.5,1.],\
#                 label=r"Realization of white-noise forcing [$10^{-6}\,\,\xi_q/Q$]")
# plt.xlabel(r'$x\, k_f/2\pi$')
# plt.ylabel(r'$y\, k_f/2\pi$')
#
# plt.savefig('figs/forcing_qg-only')

# plot potential vorticity
fig = plt.figure(figsize=(10.5,4))
cv = np.linspace(-.2,.2,40)
cphi = np.linspace(0,1.,40)
Ew = model.epsilon_w/model.muw/2

ax = fig.add_subplot(121,aspect=1)
plt.contourf(model.x/Lf,model.y/Lf,model.q/Q,cv,\
                cmin=-0.2,cmax=0.2,extend='both',cmap=cmocean.cm.curl)
plt.colorbar(ticks=[-.2,-.1,0,.1,.2],label=r'Potential vorticity $[q/Q]$',shrink=0.8)
plt.xlabel(r'$x\, k_f/2\pi$')
plt.ylabel(r'$y\, k_f/2\pi$')
ax = fig.add_subplot(122,aspect=1)
plt.contourf(model.x/Lf,model.y/Lf,np.abs(model.phi)**2/Ew,cphi,\
                cmin=0,cmax=1,extend='max',cmap=cmocean.cm.ice_r)
plt.colorbar(ticks=[0,0.5,1,],label=r'Wave action density $[\mathcal{A}/A]$',shrink=0.8)
plt.xlabel(r'$x\, k_f/2\pi$')
plt.ylabel(r'$y\, k_f/2\pi$')
plt.savefig('figs/snapshots_pv_qg-niw')

# diagnostics
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

gamma_r = model.diagnostics['gamma_r']['value']
gamma_a = model.diagnostics['gamma_a']['value']

# plot energy
fig = plt.figure(figsize=(9.5,4.))
E = model.epsilon_q/model.mu/2
Ew = model.epsilon_w/model.muw/2

ax = fig.add_subplot(121)
plt.plot(time*model.mu,KE_qg/E,label=r'$\mathcal{K}$')
plt.plot(time*model.mu,PE_niw/E,label=r'$\mathcal{P}$')
#plt.plot(time*model.mu,np.ones_like(time),'r--')
plt.xlabel(r"Time $[t\,\,\mu]$")
plt.ylabel(r"Energy $[\mathcal{K},\mathcal{P} \,\, 2 \mu/\sigma_q^2]$")
plt.legend()
ax = fig.add_subplot(122)
plt.plot(time*model.mu,KE_niw/Ew,label=r'$\mathcal{P}$')
#plt.plot(time*model.muw,np.ones_like(time),'r--')
plt.xlabel(r"Time $[t\,\,\mu]$")
plt.ylabel(r"Wave action $[\mathcal{A} \,\, 2 \mu_w/\sigma_w^2]$")

plt.savefig('figs/kinetic_energy_qg-niw')

# calculate spectrum
# calculate spectrum
E = 0.5 * np.abs(model.wv*model.ph)**2
ki, Er = spectrum.calc_ispec(model.kk, model.ll, E, ndim=2)


# dt = time[1]-time[0]
# dKE = np.gradient(KE_qg,dt)
# dKEw = np.gradient(KE_niw,dt)
