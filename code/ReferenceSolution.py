"""
    Forced-disspative vertical plane-wave model: a reference solution.

    Details:

        Vorticity forcing: a stochastic forcing with renovation time-scale 
                           equals the time-step (approximate white-noise forcing).
                           The horizontal structure of the forcing is random with
                           an annulus-like gaussian spectrum peaking at kf and 
                           with a decay scale dkf.

        Wave forcing:     similar to vorticity forcing but without horizontal structure.

        Dissipation:      both vorticity and waves are damped through linear dissipation.
                          A spectral filter is added to mop the enstrophy variance at 
                          small scales, for numerical stability, and dealiasing purpouses.
                          Small-scale dissipation removes very little energy.

    Cesar B. Rocha
    SIO, Fall 2017
"""

import timeit

import matplotlib.pyplot as plt
plt.rcParams['contour.negative_linestyle'] = 'dashed'
import numpy as np

from niwqg import CoupledModel
from niwqg import InitialConditions as ic
import cmocean

from Utils import *

from pyspec import spectrum

plt.close('all')

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
m = 2*np.pi/λz

# vorticity dissipation
Tmu = 200*86400
mu = 1./Tmu
gamma = 4*mu
Tgamma = 1./gamma

#forcing
dk = 2*np.pi/L
kf = 8*dk
Lf = 1./kf
dkf = 1*dk
U0 = 0.25                   # guessed equilibrated RMS velocity
epsilon = (U0**2)*mu       # estimated energy input
sigma_q = np.sqrt(epsilon) # the standard deviation of the random forcing
sigma_w = 4*sigma_q

# no drag
mu = 0.

# time
dt = 0.000125*Tmu/4
tmax = 40.*Tgamma
tmax = 80.*Tgamma

# outputs
path = "output/512_newest_nodrag"

#
# theoretical predictions
#

# The rate of work into the system
epsilon_q = sigma_q**2
epsilon_w = (sigma_w**2)/2

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

# non-dimensional parameters
hslash = eta/PSI      # dispersivity

#
# the set up the model
#

# initialize model class
model = CoupledModel.Model(L=L,nx=nx, tmax = tmax,dt = dt, twrite=200,
                    nu4=0.,mu=mu,nu4w=0.,nu=0,nuw=0,muw=gamma,
                    use_filter=True,save_to_disk=True,
                    tsave_snapshots=50,path=path,
                    U = 0., tdiags=50,
                    f=f0,N=N,m=m,
                    wavenumber_forcing=kf,width_forcing=dkf,
                    sigma_q=sigma_q, sigma_w=sigma_w,
                    use_mkl=True,nthreads=8)

# rest initial conditions
model.set_q(np.zeros([model.nx]*2))
model.set_phi(np.zeros([model.nx]*2)+0j)
model._invert()


#
# run the model
#
model.run()

#
# some quick plots
#

# snapshots

# wave fields
u,v, w, p, b = wave_fields(model.phi,model.f0,lamb**2,model.t,model.k,model.l,m,z=0)

# potential vorticity and near-inertial kinetic energy
fig = plt.figure(figsize=(10.5,5))
cv = np.linspace(-2,2,100)
cphi = np.linspace(0,3.,100)
ax = fig.add_subplot(121,aspect=1)
im1 = plt.contourf(model.x/Lf,model.y/Lf,model.q/Q,cv,\
                cmin=cv.min(),cmax=cv.max(),extend='both',cmap=cmocean.cm.curl)
plt.xlabel(r'$x\, k_f$')
plt.ylabel(r'$y\, k_f$')

plt.text(50.,51.5,r"$t\,\,\gamma = %3.2f$" % (model.t*model.muw))
#plt.title(r'$t \times U_e k_e$= %3.2f' %(t))
#figname = pathi+"figs2movie/"+fni[-18:-3]+".png"

cbaxes = fig.add_axes([0.15, 1., 0.3, 0.025])
plt.colorbar(im1, cax = cbaxes, ticks=[-2.,-1.,0,1,2],orientation='horizontal',
                    label=r'Potential vorticity $[q/Q]$')

ax = fig.add_subplot(122,aspect=1)
im2 = plt.contourf(model.x/Lf,model.y/Lf,np.abs(model.phi)**2/(PHI**2),cphi,\
                cmin=0,cmax=cphi.max(),extend='max',cmap=cmocean.cm.ice_r)

plt.xlabel(r'$x\, k_f$')
plt.yticks([])

cbaxes = fig.add_axes([0.575, 1., 0.3, 0.025])
plt.colorbar(im2,cax=cbaxes,ticks=[0,1.,2,3],orientation='horizontal',
                 label=r'Wave action density $[\mathcal{A}/A]$')
plt.savefig('figs/snapshots_pv_qg-niw', pad_inces=0, bbox_inches='tight')

# potential vorticity and near-inertial vertical velocity
cw = np.linspace(-0.05,0.05)

fig = plt.figure(figsize=(10.5,5))
cv = np.linspace(-2,2,100)
cphi = np.linspace(0,1.5,100)
ax = fig.add_subplot(121,aspect=1)
im1 = plt.contourf(model.x/Lf,model.y/Lf,model.q/Q,cv,\
                cmin=cv.min(),cmax=cv.max(),extend='both',cmap=cmocean.cm.curl)
plt.xlabel(r'$x\, k_f$')
plt.ylabel(r'$y\, k_f$')

plt.text(50.,51.5,r"$t\,\,\gamma = %3.2f$" % (model.t*model.muw))
#plt.title(r'$t \times U_e k_e$= %3.2f' %(t))
#figname = pathi+"figs2movie/"+fni[-18:-3]+".png"

cbaxes = fig.add_axes([0.15, 1., 0.3, 0.025])
plt.colorbar(im1, cax = cbaxes, ticks=[-2.,-1.,0,1,2],orientation='horizontal',
                    label=r'Potential vorticity $[q/Q]$')

ax = fig.add_subplot(122,aspect=1)
im2 = plt.contourf(model.x/Lf,model.y/Lf,w/PHI,cw,\
                cmin=0,cmax=cphi.max(),extend='both',cmap=cmocean.cm.balance)

plt.xlabel(r'$x\, k_f$')
plt.yticks([])

cbaxes = fig.add_axes([0.575, 1., 0.3, 0.025])
plt.colorbar(im2,cax=cbaxes,ticks=[-0.05,-0.025,0.,0.025,0.05],orientation='horizontal',
                 label=r'Wave vertical velocity $[w/\Phi]$')
plt.savefig('figs/snapshots_pv_w_qg-niw', pad_inces=0, bbox_inches='tight')

# potential vorticities 
fig = plt.figure(figsize=(10.5,5))
cv = np.linspace(-2,2,100)
cphi = np.linspace(0,3.,100)
ax = fig.add_subplot(121,aspect=1)
im1 = plt.contourf(model.x/Lf,model.y/Lf,model.q_psi/Q,cv,\
                cmin=cv.min(),cmax=cv.max(),extend='both',cmap=cmocean.cm.curl)
plt.xlabel(r'$x\, k_f$')
plt.ylabel(r'$y\, k_f$')

plt.text(50.,51.5,r"$t\,\,\gamma = %3.2f$" % (model.t*model.muw))
#plt.title(r'$t \times U_e k_e$= %3.2f' %(t))
#figname = pathi+"figs2movie/"+fni[-18:-3]+".png"

cbaxes = fig.add_axes([0.15, 1., 0.3, 0.025])
plt.colorbar(im1, cax = cbaxes, ticks=[-2.,-1.,0,1,2],orientation='horizontal',
                    label=r'Potential vorticity $[q/Q]$')

ax = fig.add_subplot(122,aspect=1)
im1 = plt.contourf(model.x/Lf,model.y/Lf,(model.q-model.q_psi)/Q,cv,\
                cmin=cv.min(),cmax=cv.max(),extend='both',cmap=cmocean.cm.curl)

plt.xlabel(r'$x\, k_f$')
plt.yticks([])
plt.savefig('figs/snapshots_pvs_qg-niw', pad_inces=0, bbox_inches='tight')

#
# quick look at the diagnostics
#
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

skew = model.diagnostics['skew']['value']

work_q = model.diagnostics['Work_q']['value']/time
work_phi = model.diagnostics['Work_w']['value']/time


# plot energy
fig = plt.figure(figsize=(9.5,4.))
E = epsilon_q/model.mu/2
Ew = PHI**2 / 2
POWER = Ew/Tmu

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


# K and P with mean energy removed (after equilibration)
eq = time*model.muw > 4.
teq = time[eq]
dK_eq = KE_qg[eq]-KE_qg[eq].mean()
dP_eq = PE_niw[eq]-PE_niw[eq].mean()

fig = plt.figure(figsize=(9.5,4.))
ax = fig.add_subplot(111)
plt.plot(teq*model.muw,dK_eq/E,label=r'$\Delta\mathcal{K}$')
plt.plot(teq*model.muw,dP_eq/E,label=r'$\Delta\mathcal{K}$')

plt.xlabel(r"Time $[t\,\,\gamma]$")
plt.ylabel(r"Energy anomaly $[\Delta\mathcal{E} \,\, 2 f_0 \mu/\epsilon_q]$")
remove_axes(ax)
plt.savefig('figs/energy_qgniw_anomaly' , pad_inces=0, bbox_inches='tight')


# accumulated work
fig = plt.figure(figsize=(9.5,4.))
ax = fig.add_subplot(111)
plt.plot(time*model.muw,time*(model.sigma_w**2)/2/Ew,'--',label='Theory')
plt.plot(time*model.muw,work_phi*time/Ew,)
plt.xlabel(r"Time $[t\,\,\gamma]$")
plt.ylabel(r"Accumulated work ")
plt.ylabel(r"Accumulated work $[W \,\, 2 \mu_w/\epsilon_q]$")
remove_axes(ax)
plt.savefig('figs/accumulated_work' , pad_inces=0, bbox_inches='tight')

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
plt.ylim(-5,5)
plt.legend(loc=3,ncol=2)
plt.ylabel(r"Power [$\dot \mathcal{P} \,/\, W$]")
plt.xlabel(r"$t\,\, \gamma$")
remove_axes(ax)
plt.savefig('figs/PE_niw_budget' , pad_inces=0, bbox_inches='tight')

fig = plt.figure(figsize=(9.5,4.))
ax = fig.add_subplot(111)
plt.plot(time*model.muw,(gamma_r+gamma_a)/POWER,label=r'$\Gamma_r+\Gamma_a$')
#plt.plot(time*model.muw,gamma_a/POWER,label=r'$\Gamma_a$')
#plt.plot(time*model.mu,xi_a,label=r'$\Xi_a$')
#plt.plot(time*model.mu,xi_r,label=r'$\Xi_r$')
plt.plot(time*model.muw,-chi_phi/POWER,label=r'$2\gamma\mathcal{P}$')
#plt.plot(time*model.muw,smallchi_phi/POWER,label=r'$\chi_\phi$')
#plt.plot(time*model.muw,dPE_niw/POWER,label=r'$d\mathcal{P}/dt$')
#plt.plot(time*model.mu,residual/POWER,label=r'residual')
plt.ylim(-.1,5)
plt.legend(loc=2,ncol=2)
plt.ylabel(r"Power [$\dot \mathcal{P} \,/\, W$]")
plt.xlabel(r"$t\,\, \gamma$")
remove_axes(ax)
plt.savefig('figs/PE_niw_budget2' , pad_inces=0, bbox_inches='tight')

fig = plt.figure(figsize=(9.5,4.))
ax = fig.add_subplot(111)
plt.plot(time*model.muw,np.zeros_like(time),'--',color='0.5')
plt.plot(time*model.muw,(work_q)/POWER,label=r'Forcing')
plt.plot(time*model.muw,ep_psi/POWER,label=r'Bottom-drag diss.')
plt.plot(time*model.muw,-(gamma_r+gamma_a)/POWER,label=r'Stimulated generation')
plt.plot(time*model.muw,(xi_a+xi_r)/POWER,label=r'Streaming')
remove_axes(ax)
plt.legend(loc=(0.7,0.6))
plt.ylabel(r"Power [$\dot \mathcal{P} \,/\, W$]")
plt.xlabel(r"$t\,\, \gamma$")
plt.savefig('figs/ke_budget_generation' , pad_inces=0, bbox_inches='tight')



# calculate spectrum
E = 0.5 * np.abs(model.wv*model.ph)**2
ki, Ei = spectrum.calc_ispec(model.kk, model.ll, E)

E = 0.5 * np.abs(model.phih)**2
_, Eiphi = spectrum.calc_ispec(model.kk, model.ll, E)

E = (lamb**2) * 0.4 * np.abs(model.wv*model.phih)**2
_, Piphi = spectrum.calc_ispec(model.kk, model.ll, E)

np.savez("spec_qgniq_niw",k=ki,K=Ei,P=Piphi,A=Eiphi,kf=kf)

# plot spectra
#spec_qg = np.load("spec_qg-only.npz")
S = epsilon_q/model.mu/2/kf

fig = plt.figure(figsize=(6.5,4.5))
ax = fig.add_subplot(111)
#plt.loglog(ki/kf,spec_qg['K']/S,'k',linewidth=2,label=r'$\mathcal{K}$, waveless QG')
plt.loglog(ki/kf,Ei/S,linewidth=2,label=r'$\mathcal{K}$, QG-NIW')
#plt.loglog(ki/kf,Eiphi,linewidth=2,label=r'$\mathcal{A}$')
#plt.loglog(ki/kf,Piphi,linewidth=2,label=r'$\mathcal{P}$')
plt.ylim(1e-7,1e1)
#plt.legend(loc=3)
remove_axes(ax)
plt.xlabel(r"Wavenumber [$k/k_f$]")
plt.ylabel(r"Balanced kinetic energy density [$\mathcal{K} 2 \mu k_f \epsilon_q^{-1}$]")
plt.savefig('figs/spectra' , pad_inces=0, bbox_inches='tight')






