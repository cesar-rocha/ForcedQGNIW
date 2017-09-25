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

λz = 1000
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
sigma = np.sqrt(epsilon)
path = "output/reference512"

nu4  = 0e8
nu4w = 0e8

# Forced only dynamics
model = CoupledModel.Model(L=L,nx=nx, tmax = tmax,dt = dt, twrite=200,
                    nu4=nu4,mu=mu,nu4w=nu4w,nu=0,nuw=0,muw=4*mu,
                    use_filter=True,save_to_disk=True,
                    tsave_snapshots=100,path=path,
                    U = 0., tdiags=100,
                    f=f0,N=N,m=m,
                    wavenumber_forcing=kf,width_forcing=dkf,
                    sigma_q=sigma, sigma_w=2*sigma,
                    use_mkl=True,nthreads=8)

# The rate of work into the system
epsilon_q = model.sigma_q**2
epsilon_w = (model.sigma_w**2)/2

# non-dimensional numbers
lamb = N/f0/m
eta = f0*(lamb**2)
PSI = model.sigma_q/np.sqrt(model.mu)/kf
PHI = np.sqrt(epsilon_w/model.muw)
hslash = eta/PSI
Ew = epsilon_w/model.muw/2
PHI = np.sqrt(epsilon_w/model.muw)
POWER = (Ew)**2/Tmu

# initial conditions
model.set_q(np.zeros([model.nx]*2))
model.set_phi(np.zeros([model.nx]*2)+0j)
model._invert()

# run the model
model.run()

# plot snapshot of potential vorticity
Q = (2*np.pi)**-2 * epsilon/(mu**2 / kf**2)
fig = plt.figure(figsize=(10.5,5))
cv = np.linspace(-.2,.2,100)
cphi = np.linspace(0,3.,100)
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

dt = time[1]-time[0]
dPE_niw = np.gradient(PE_niw,dt)

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


work_q = model.diagnostics['Work_q']['value']/time
work_phi = model.diagnostics['Work_w']['value']/time

def remove_axes(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


# plot energy
fig = plt.figure(figsize=(9.5,4.))
E = epsilon_q/model.mu/2
Ew = epsilon_w/model.muw/2

ax = fig.add_subplot(121)
plt.plot(time*model.muw,KE_qg/E,label=r'$\mathcal{K}$')
plt.plot(time*model.muw,PE_niw/E,label=r'$\mathcal{P}$')
#plt.plot(time*model.mu,np.ones_like(time),'r--')
plt.xlabel(r"Time $[t\,\,\gamma]$")
plt.ylabel(r"Energy $[\mathcal{K},\mathcal{P} \,\, 2 \mu/\epsilon_q]$")
plt.legend(loc=5)
remove_axes(ax)
plt.subplots_adjust(wspace=.35)
ax = fig.add_subplot(122)
plt.plot(time*model.muw,KE_niw/Ew,label=r'$\mathcal{P}$')
#plt.plot(time*model.muw,np.ones_like(time),'r--')
plt.xlabel(r"Time $[t\,\,\gamma]$")
plt.ylabel(r"Wave action $[\mathcal{A} \,\, 2 f_0 \mu_w/\epsilon]$")
remove_axes(ax)
plt.savefig('figs/energy_and_action_qg-niw' , pad_inces=0, bbox_inches='tight')

fig = plt.figure(figsize=(9.5,4.))
ax = fig.add_subplot(111)
plt.plot(time*model.muw,KE_qg/E,label=r'$\mathcal{K}$')
plt.plot(time*model.muw,PE_niw/E,label=r'$\mathcal{P}$')
plt.plot(time*model.muw,KE_niw/E,label=r'$f_0\mathcal{A}$')
plt.xlabel(r"Time $[t\,\,\gamma]$")
plt.ylabel(r"Energy $[\mathcal{E} \,\, 2 f_0 \mu/\epsilon_q]$")
remove_axes(ax)
plt.savefig('figs/energy_qgniw' , pad_inces=0, bbox_inches='tight')

# a PE_niw energy budget
residual = gamma_r+gamma_a+chi_phi+smallchi_phi-dPE_niw

fig = plt.figure(figsize=(9.5,4.))
ax = fig.add_subplot(111)
plt.plot(time*model.muw,gamma_r/POWER,label=r'$\Gamma_r$')
plt.plot(time*model.muw,gamma_a/POWER,label=r'$\Gamma_a$')
#plt.plot(time*model.mu,xi_a,label=r'$\Xi_a$')
#plt.plot(time*model.mu,xi_r,label=r'$\Xi_r$')
plt.plot(time*model.muw,chi_phi/POWER,label=r'$-2\gamma\mathcal{P}$')
#plt.plot(time*model.muw,smallchi_phi/POWER,label=r'$\chi_\phi$')
#plt.plot(time*model.muw,dPE_niw/POWER,label=r'$d\mathcal{P}/dt$')
#plt.plot(time*model.mu,residual/POWER,label=r'residual')
plt.ylim(-65,65)
plt.legend(loc=3,ncol=2)
plt.ylabel(r"Power [$\dot \mathcal{P} \,/\, W$]")
plt.xlabel(r"$t\,\, \gamma$")
remove_axes(ax)
plt.savefig('figs/PE_niw_budget' , pad_inces=0, bbox_inches='tight')

fig = plt.figure(figsize=(9.5,4.))
ax = fig.add_subplot(111)
plt.plot(time*model.muw,(gamma_r+gamma_a)/POWER,label=r'$\Gamma_r$')
#plt.plot(time*model.muw,gamma_a/POWER,label=r'$\Gamma_a$')
#plt.plot(time*model.mu,xi_a,label=r'$\Xi_a$')
#plt.plot(time*model.mu,xi_r,label=r'$\Xi_r$')
plt.plot(time*model.muw,-chi_phi/POWER,label=r'$2\gamma\mathcal{P}$')
#plt.plot(time*model.muw,smallchi_phi/POWER,label=r'$\chi_\phi$')
#plt.plot(time*model.muw,dPE_niw/POWER,label=r'$d\mathcal{P}/dt$')
#plt.plot(time*model.mu,residual/POWER,label=r'residual')
plt.ylim(-5,65)
plt.legend(loc=2,ncol=2)
plt.ylabel(r"Power [$\dot \mathcal{P} \,/\, W$]")
plt.xlabel(r"$t\,\, \gamma$")
remove_axes(ax)
plt.savefig('figs/PE_niw_budget2' , pad_inces=0, bbox_inches='tight')

#ax = fig.add_subplot(122)
#plt.plot(time*model.mu,gamma_r,label=r'$\Gamma_r$')
#plt.plot(time*model.mu,gamma_a,label=r'$\Gamma_a$')
#plt.plot(time*model.mu,chi_phi,label=r'$\chi_\phi$')
#plt.plot(time*model.mu,smallchi_phi,label=r'$\delta_\phi$')
#plt.legend()
#plt.ylabel("Power")
#plt.savefig('figs/rough_budgets' , pad_inces=0, bbox_inches='tight')

# calculate spectrum
E = 0.5 * np.abs(model.wv*model.ph)**2
ki, Ei = spectrum.calc_ispec(model.kk, model.ll, E)

E = 0.5 * np.abs(model.phih)**2
_, Eiphi = spectrum.calc_ispec(model.kk, model.ll, E)

E = (lamb**2) * 0.4 * np.abs(model.wv*model.phih)**2
_, Piphi = spectrum.calc_ispec(model.kk, model.ll, E)

np.savez("spec_qgniq",k=ki,K=Ei,P=Piphi,A=Eiphi,kf=kf)

# dt = time[1]-time[0]
# dKE = np.gradient(KE_qg,dt)
# dKEw = np.gradient(KE_niw,dt)
