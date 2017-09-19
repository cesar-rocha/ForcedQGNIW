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

path = "output/reference512"

nu4  = 1e8
nu4w = 5.e8

# Forced only dynamics
model = CoupledModel.Model(L=L,nx=nx, tmax = tmax,dt = dt, twrite=200,
                    nu4=nu4,mu=mu,nu4w=0*nu4w,nu=0,nuw=0,muw=0*mu, use_filter=False,save_to_disk=True,
                    tsave_snapshots=100,path=path,
                    U = 0., tdiags=100,
                    f=f0,N=N,m=m,
                    wavenumber_forcing=kf,width_forcing=dkf,
                    epsilon_q=epsilon, epsilon_w=0*epsilon,
                    use_mkl=True,nthreads=8)

# non-dimensional numbers
lamb = N/f0/m
eta = f0*(lamb**2)
PSI = Lf*np.sqrt(model.epsilon_q)/np.sqrt(model.mu)
#PHI = np.sqrt(model.epsilon_w/mu)
PHI = 1.
hslash = eta/PSI

# initial conditions
model.set_q(np.zeros([model.nx]*2))
model.set_phi(np.zeros([model.nx]*2)+0j)
model._invert()

# run the model
model.run()

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
#plt.title(r'$t \times U_e k_e$= %3.2f' %(t))
#figname = pathi+"figs2movie/"+fni[-18:-3]+".png"
#cbaxes = fig.add_axes([0.15, 1., 0.3, 0.025])
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


def remove_axes(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

dt = time[1]-time[0]
dKE_qg = np.gradient(KE_qg,dt)

# plot energy
fig = plt.figure(figsize=(8.5,4.))
E = model.epsilon_q/model.mu/2

ax = fig.add_subplot(111)
plt.plot(time*model.mu,np.ones_like(time),'r--')
plt.plot(time*model.mu,KE_qg/E,label=r'$\mathcal{K}$')
plt.xlabel(r"Time $[t\,\,\mu]$")
plt.ylabel(r"Energy $[\mathcal{K} \,\, 2 \mu/\sigma_q^2]$")
#plt.legend(loc=5)
remove_axes(ax)
plt.savefig('figs/energy_and_action_qg' , pad_inces=0, bbox_inches='tight')

# a KE_niw energy budget
residual = -gamma_r-gamma_a+ep_psi+smalldiss_psi-dKE_qg+energy_input
POWER = (E)**2/Tmu

fig = plt.figure(figsize=(9.5,4.))
ax = fig.add_subplot(111)
#plt.plot(time*model.mu,-gamma_r/POWER,label=r'$\Gamma_r$')
#plt.plot(time*model.mu,-gamma_a/POWER,label=r'$\Gamma_a$')
#plt.plot(time*model.mu,xi_a,label=r'$\Xi_a$')
#plt.plot(time*model.mu,xi_r,label=r'$\Xi_r$')
plt.plot(time*model.mu,np.ones_like(time)*model.epsilon_q/POWER,label=r'$\sigma_q^2$')
#plt.plot(time*model.mu,energy_input/POWER,label=r'$-\langle \psi \xi_q \rangle$')
plt.plot(time*model.mu,ep_psi/POWER,label=r'$-2\mu\mathcal{K}$')
plt.plot(time*model.mu,smalldiss_psi/POWER,label=r'$\epsilon_\psi$')

#plt.plot(time*model.mu,dKE_qg/POWER,label=r'$d\mathcal{K}/dt$')
#plt.plot(time*model.mu,residual/POWER,label=r'residual')
#plt.ylim(-10,10)
plt.legend(loc=3,ncol=2)
plt.ylabel(r"Power [$\dot \mathcal{K} \,/\, W$]")
plt.xlabel(r"$t\,\, \mu$")
remove_axes(ax)
plt.savefig('figs/KE_qg_budget_qg-only' , pad_inces=0, bbox_inches='tight')

# calculate spectrum
E = 0.5 * np.abs(model.wv*model.ph)**2
ki, Ei = spectrum.calc_ispec(model.kk, model.ll, E)


