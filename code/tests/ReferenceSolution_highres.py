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
nx = 128*4

f0 = 1.e-4
N = 0.005
L = 2*np.pi*200e3

λz = 750
m = 2*np.pi/λz

# dissipation
Tmu = 200*86400
mu = 1./Tmu

dt = 0.000125*Tmu/4
tmax = 4*Tmu

#forcing
dk = 2*np.pi/L

kf = 8*dk
dkf = 1*dk
Lf = 2*np.pi/kf
# energy input
U0 = 0.5
epsilon = (U0**2)*mu

path = "output/reference512_2"

nu4  = 1e8
nu4w = 5e8

# Forced only dynamics
model = CoupledModel.Model(L=L,nx=nx, tmax = tmax,dt = dt, twrite=100,
                    nu4=nu4,mu=mu,nu4w=nu4w,nu=0,nuw=0,muw=4*mu, use_filter=False,save_to_disk=True,
                    tsave_snapshots=100,path=path,
                    U = 0., tdiags=100,
                    f=f0,N=N,m=m,
                    wavenumber_forcing=kf,width_forcing=dkf,
                    epsilon_q=epsilon, epsilon_w=2*epsilon,
                    use_mkl=True,nthreads=16)

# non-dimensional numbers
lamb = N/f0/m
eta = f0*(lamb**2)
PSI = Lf*np.sqrt(model.epsilon_q)/np.sqrt(model.mu)
PHI = np.sqrt(model.epsilon_w/model.muw)
hslash = eta/PSI

# initial conditions
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
fig = plt.figure(figsize=(10.5,5))
cv = np.linspace(-.2,.2,100)
cphi = np.linspace(0,3.,100)
Ew = model.epsilon_w/model.muw/2
PHI = np.sqrt(model.epsilon_w/model.muw)

ax = fig.add_subplot(121,aspect=1)
im1 = plt.contourf(model.x/Lf,model.y/Lf,model.q/Q,cv,\
                cmin=-0.2,cmax=0.2,extend='both',cmap=cmocean.cm.curl)
plt.xlabel(r'$x\, k_f/2\pi$')
plt.ylabel(r'$y\, k_f/2\pi$')

plt.text(8.,8.5,r"$t\,\,\mu = %3.2f$" % (model.t*model.mu))
#plt.title(r'$t \times U_e k_e$= %3.2f' %(t))
#figname = pathi+"figs2movie/"+fni[-18:-3]+".png"


cbaxes = fig.add_axes([0.15, 1., 0.3, 0.025]) 
plt.colorbar(im1, cax = cbaxes, ticks=[-.2,-.1,0,.1,.2],orientation='horizontal',label=r'Potential vorticity $[q/Q]$')

ax = fig.add_subplot(122,aspect=1)
im2 = plt.contourf(model.x/Lf,model.y/Lf,np.abs(model.phi)**2/PHI,cphi,\
                cmin=0,cmax=1,extend='max',cmap=cmocean.cm.ice_r)

plt.xlabel(r'$x\, k_f/2\pi$')
#plt.ylabel(r'$y\, k_f/2\pi$')
plt.yticks([])

cbaxes = fig.add_axes([0.575, 1., 0.3, 0.025]) 
plt.colorbar(im2,cax=cbaxes,ticks=[0,1,2,3,],orientation='horizontal',label=r'Wave action density $[\mathcal{A}/A]$')

plt.savefig('figs/snapshots_pv_qg-niw', pad_inces=0, bbox_inches='tight')

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

gamma_r = model.diagnostics['gamma_r']['value']
gamma_a = model.diagnostics['gamma_a']['value']
xi_r = model.diagnostics['xi_r']['value']
xi_a = model.diagnostics['xi_a']['value']



def remove_axes(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    

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
plt.legend(loc=5)
remove_axes(ax)
plt.subplots_adjust(wspace=.35)
ax = fig.add_subplot(122)
plt.plot(time*model.mu,KE_niw/Ew,label=r'$\mathcal{P}$')
#plt.plot(time*model.muw,np.ones_like(time),'r--')
plt.xlabel(r"Time $[t\,\,\mu]$")
plt.ylabel(r"Wave action $[\mathcal{A} \,\, 2 f_0 \mu_w/\sigma_w^2]$")
remove_axes(ax)
plt.savefig('figs/energy_and_action_qg-niw' , pad_inces=0, bbox_inches='tight')

# plot energy
fig = plt.figure(figsize=(9.5,4.))
ax = fig.add_subplot(121)
plt.plot(time*model.mu,KE_qg/E,label=r'$\mathcal{K}$')
plt.plot(time*model.mu,PE_niw/E,label=r'$\mathcal{P}$')
#plt.plot(time*model.mu,np.ones_like(time),'r--')
plt.xlabel(r"Time $[t\,\,\mu]$")
plt.ylabel(r"Energy $[\mathcal{K},\mathcal{P} \,\, 2 \mu/\sigma_q^2]$")
plt.legend(loc=5)
remove_axes(ax)
plt.subplots_adjust(wspace=.35)
ax = fig.add_subplot(122)
plt.plot(time*model.mu,KE_niw/E,label=r'$\mathcal{P}$')
#plt.plot(time*model.muw,np.ones_like(time),'r--')
plt.xlabel(r"Time $[t\,\,\mu]$")
plt.ylabel(r"Wave action $[\mathcal{A} \,\, 2 \mu_w f_0 /\sigma_q^2]$")
remove_axes(ax)
plt.savefig('figs/energy_and_action_qg-niw_2' , pad_inces=0, bbox_inches='tight')

# a quasi energy budget
fig = plt.figure(figsize=(9.5,4.))

ax = fig.add_subplot(121)
plt.plot(time*model.mu,-gamma_r,label=r'$-\Gamma_r$')
plt.plot(time*model.mu,-gamma_a,label=r'$-\Gamma_a$')
plt.plot(time*model.mu,ep_psi,label=r'$\epsilon_\psi$')
plt.legend()
plt.ylabel("Power")

ax = fig.add_subplot(122)
plt.plot(time*model.mu,gamma_r,label=r'$\Gamma_r$')
plt.plot(time*model.mu,gamma_a,label=r'$\Gamma_a$')
plt.plot(time*model.mu,chi_phi,label=r'$-\chi_\phi$')
plt.legend()
plt.ylabel("Power")
plt.savefig('figs/rough_budgets' , pad_inces=0, bbox_inches='tight')


# calculate spectrum
# calculate spectrum
E = 0.5 * np.abs(model.wv*model.ph)**2
ki, Er = spectrum.calc_ispec(model.kk, model.ll, E)

# dt = time[1]-time[0]
# dKE = np.gradient(KE_qg,dt)
# dKEw = np.gradient(KE_niw,dt)
