import numpy as np
import matplotlib.pyplot as plt
import h5py
import os, glob
from pyspec import spectrum
import cmocean 
plt.close('all')

pathi = "output/512/"

setup = h5py.File(pathi+"setup.h5")
k,l = setup['grid/k'][:], setup['grid/l'][:]
x,y = setup['grid/x'][:], setup['grid/y'][:]
ki,li = np.meshgrid(k,l)
kf = 8*k[1]
wv2 = ki**2 + li**2
wv2i = 1./wv2
wv2i[0,0] = 0
wv = np.sqrt(wv2)

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
muw = gamma
Tgamma = 1./gamma

#forcing
dk = 2*np.pi/L
kf = 8*dk
Lf = 1./kf
dkf = 1*dk
U0 = 0.5                   # guessed equilibrated RMS velocity
epsilon = (U0**2)*mu       # estimated energy input
sigma_q = np.sqrt(epsilon) # the standard deviation of the random forcing
sigma_w = 2*sigma_q

# time
dt = 0.000125*Tmu/4
tmax = 20.*Tgamma

# outputs
path = "output/512"


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

files = glob.glob(pathi+"snapshots/"+"*.h5")[::1]

fni = files[0]

nmax = 512//2

def plot_snapshot(snap):

    """ Plot snapshots of potential vorticity and wave action """

    q = snap['q'][:]
    phi = snap['phi'][:]
    t = snap['t'][()]

    fig = plt.figure(figsize=(10.5,5))
    cv = np.linspace(-.2,.2,30)
    cphi = np.linspace(0,4.,30)
    Ew = epsilon_w/muw/2
    Q = (2*np.pi)**-2 * epsilon/(mu**2 / kf**2)
    PHI = np.sqrt(epsilon_w/muw)
    PHI2 = PHI**2

    ax = fig.add_subplot(121,aspect=1)
    im1 = plt.contourf(x[:nmax,:nmax]/Lf,y[:nmax,:nmax]/Lf,q[:nmax,:nmax]/Q,cv,\
                cmin=-0.2,cmax=0.2,extend='both',cmap=cmocean.cm.curl)
    plt.xlabel(r'$x\, k_f$')
    plt.ylabel(r'$y\, k_f$')

    plt.text(25.25,25.5,r"$t\,\,\gamma = %3.2f$" % (t*muw))

    cbaxes = fig.add_axes([0.15, 1., 0.3, 0.025]) 
    plt.colorbar(im1, cax = cbaxes, ticks=[-.2,-.1,0,.1,.2],orientation='horizontal',label=r'Potential vorticity $[q/Q]$')

    ax = fig.add_subplot(122,aspect=1)
    im2 = plt.contourf(x[:nmax,:nmax]/Lf,y[:nmax,:nmax]/Lf,np.abs(phi[:nmax,:nmax])**2/PHI2,cphi,\
                   cmin=0,cmax=4,extend='max',cmap=cmocean.cm.ice_r)

    plt.xlabel(r'$x\, k_f$')
    plt.yticks([])

    cbaxes = fig.add_axes([0.575, 1., 0.3, 0.025]) 
    plt.colorbar(im2,cax=cbaxes,ticks=[0,1,2,3,4],orientation='horizontal',label=r'Wave action density $[\mathcal{A}/A]$')

    figname = "figs2movie/qgniw/"+ fni[-18:-3]+".png"

    plt.savefig(figname, dpi=80, pad_inces=0, bbox_inches='tight')

    plt.close("all")

for fni in files[:]:
    snap = h5py.File(fni)
    plot_snapshot(snap)


