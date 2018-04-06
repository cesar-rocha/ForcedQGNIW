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
λz = 800
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

# time
dt = 0.000125*Tmu/4
tmax = 60.*Tgamma
tmax = 40*Tgamma
#tmax = 80.*Tgamma

# outputs
path = "output/newest/512_reference"
path_nodrag = "output/newest/512_nodrag"
path_nowaves = "output/newest/512_nowaves"

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
#model = CoupledModel.Model(L=L,nx=nx, tmax = tmax,dt = dt, twrite=200,
#                    nu4=0.,mu=mu,nu4w=0.,nu=0,nuw=0,muw=gamma,
#                    use_filter=True,save_to_disk=True,
#                    tsave_snapshots=50,path=path,
#                    U = 0., tdiags=50,
#                    f=f0,N=N,m=m,
#                    wavenumber_forcing=kf,width_forcing=dkf,
#                    sigma_q=sigma_q, sigma_w=sigma_w,
#                    use_mkl=True,nthreads=12)

model_nodrag = CoupledModel.Model(L=L,nx=nx, tmax = tmax,dt = dt, twrite=200,
                    nu4=0.,mu=0.,nu4w=0.,nu=0,nuw=0,muw=gamma,
                    use_filter=True,save_to_disk=True,
                    tsave_snapshots=50,path=path_nodrag,
                    U = 0., tdiags=50,
                    f=f0,N=N,m=m,
                    wavenumber_forcing=kf,width_forcing=dkf,
                    sigma_q=sigma_q, sigma_w=sigma_w,
                    use_mkl=True,nthreads=12)

#model_nowaves = CoupledModel.Model(L=L,nx=nx, tmax = tmax,dt = dt, twrite=200,
#                    nu4=0.,mu=mu,nu4w=0.,nu=0,nuw=0,muw=gamma*0,
#                    use_filter=True,save_to_disk=True,
#                    tsave_snapshots=50,path=path_nowaves,
#                    U = 0., tdiags=50,
#                    f=f0,N=N,m=m,
#                    wavenumber_forcing=kf,width_forcing=dkf,
#                    sigma_q=sigma_q, sigma_w=sigma_w*0,
#                    use_mkl=True,nthreads=12)


# rest initial conditions
#model.set_q(np.zeros([model.nx]*2))
#model.set_phi(np.zeros([model.nx]*2)+0j)
#model._invert()

model_nodrag.set_q(np.zeros([model_nodrag.nx]*2))
model_nodrag.set_phi(np.zeros([model_nodrag.nx]*2)+0j)
model_nodrag._invert()
#
#model_nowaves.set_q(np.zeros([model.nx]*2))
#model_nowaves.set_phi(np.zeros([model.nx]*2)+0j)
#model_nowaves._invert()


#
# run the model
#
model_nodrag.run()
#model.run()
#model_nowaves.run()


