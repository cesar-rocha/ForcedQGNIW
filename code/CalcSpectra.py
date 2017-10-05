import numpy as np
import matplotlib.pyplot as plt
import h5py
import os, glob
from pyspec import spectrum

from Utils import *

plt.close('all')

pathi = "output/512/"
pathi_nowaves = "output/512_nowaves/"
patho = "../writeup/figs/"

# parameters
L  = 200e3*2*np.pi
dk = 2*np.pi/L
kf = 8*dk
Lf = 1./kf

f0 = 1.e-4
N = 0.005
λz = 1000
m = 2*np.pi/λz
lamb = N/f0/m
lamb2 = lamb**2
eta = f0*(lamb**2)
Tmu = 200*86400
mu = 1./Tmu
gamma = 4*mu
U0 = 0.5                   # guessed equilibrated RMS velocity
epsilon = (U0**2)*mu       # estimated energy input
sigma_q = np.sqrt(epsilon) # the standard deviation of the random forcing
sigma_w = 2*sigma_q
epsilon_q = sigma_q**2
epsilon_w = (sigma_w**2)/2
K  = epsilon_q/mu
Kw = epsilon_w/gamma
lamb = N/f0/m
eta = f0*(lamb**2)
PHI  = Kw**0.5  


# scaling
U  = K**0.5
Uw = Kw**0.5
setup = h5py.File(pathi+"setup.h5")
k,l = setup['grid/k'][:], setup['grid/l'][:]
ki,li = np.meshgrid(k,l)
kf = 8*k[1]
wv2 = ki**2 + li**2
wv2i = 1./wv2
wv2i[0,0] = 0

wv = np.sqrt(wv2)
files = glob.glob(pathi+"snapshots/"+"*.h5")[800::10]

fni = files[0]



def invert_qgniw(qh,phi,phih,k,l,f0):

    """ Calculate the streamfunction given the potential vorticity.
            The algorithm is:
                1) Calculate wave potential vorticity
                2) Invert for wave, pw, and vortex stremfunctions, pv.
                3) Calculate the geostrophic stremfunction, p = pv+pw.
    """

    wv2 = k**2 + l**2
    wv2i = 1./wv2
    wv2i[0,0] = 0

    phih = np.fft.fft2(phi)

    phix, phiy = np.fft.ifft2(1j*k*phih), np.fft.ifft2(1j*l*phih)
    jach = np.fft.fft2((1j*(np.conj(phix)*phiy - np.conj(phiy)*phix)).real)
    jach[0,0] = 0

    # the wavy PV
    phi2 = np.abs(phi)**2
    gphi2h = -wv2*np.fft.fft2(phi2)
    qwh = 0.5*(0.5*gphi2h  + jach)/f0

    # invert for psi
    pw = np.fft.ifft2((wv2i*qwh)).real
    pv = np.fft.ifft2(-(wv2i*qh)).real
    p = pv+pw
    ph = np.fft.fft2(p)

    return ph

# calculate spectrum from reference solution

for fni in files:
    snap = h5py.File(fni)

    qh = np.fft.fft2(snap['q'][:])
    phi = snap['phi'][:]
    phih = np.fft.fft2(snap['phi'][:])

    #ph = -wv2i*qh

    ph = invert_qgniw(qh,phi,phih,ki,li,f0)

    spec = 0.5 * np.abs(wv*ph)**2
    kiso, Eiso = spectrum.calc_ispec(k,l,spec)

    spec = lamb2 * 0.25 * np.abs(wv*phih)**2
    _, Piso = spectrum.calc_ispec(k,l,spec)

    spec = 0.5 * np.abs(phih)**2
    _, Aiso = spectrum.calc_ispec(k,l,spec)

    try:
        E = np.vstack([E,Eiso[np.newaxis]])
        P = np.vstack([P,Piso[np.newaxis]])
        A = np.vstack([A,Aiso[np.newaxis]])
    except:
        E = Eiso[np.newaxis]
        P = Piso[np.newaxis]
        A = Aiso[np.newaxis]

E = E.mean(axis=0)
A = A.mean(axis=0)
P = P.mean(axis=0)


# calculate referece spectrum from nowave experiment
for fni in files:
    snap = h5py.File(fni)

    qh = np.fft.fft2(snap['q'][:])

    ph = -wv2i*qh

    spec = 0.5 * np.abs(wv*ph)**2
    kiso, Eiso = spectrum.calc_ispec(k,l,spec)

    try:
        Enw = np.vstack([Enw,Eiso[np.newaxis]])
    except:
        Enw = Eiso[np.newaxis]

Enw = Enw.mean(axis=0)

#
# Plot spectra 
#

kr = np.array([0.1,80])

SPEC = (PHI**2 / kf)

fig = plt.figure(figsize=(8.5,4))

ax = fig.add_subplot(121)
plt.loglog(kr,(kr**-3)/40,color='0.5')
plt.loglog(kr,(kr**-(5/3))/100,color='0.5')
p = plt.loglog(kiso/kf,E/SPEC,label=r'$\mathcal{K}$, coupled model')
plt.loglog(kiso/kf,Enw/SPEC,'--',color=p[0].get_color(),label=r'$\mathcal{K},$ waveless QG')
plt.ylim(1e-7,5e0)
plt.xlim(0.1,25)
plt.xlabel(r'Wavenumber [$|\mathbf{k}|/k_f$]')
plt.ylabel(r'Energy density [$\mathcal{E}/S$]')

plt.legend(loc=(0.35,0.05))

plt.text(0.125,2.5,"-3")
plt.text(0.125,0.07,"-5/3")

remove_axes(ax)
plot_fig_label(ax, xc=.05, yc=0.05 ,label="a",)
ax.spines['left'].set_position(('axes', -0.1))

plt.subplots_adjust(wspace=0.05)

ax2 = fig.add_subplot(122)

plt.loglog(kiso/kf,A/SPEC,label=r'$f_0\mathcal{A}$')
plt.loglog(kiso/kf,P/SPEC,label=r'$\mathcal{P}$')

plt.legend(loc=(0.65,0.05))
plt.loglog(kr,(kr**-(5/3))/100,color='0.5')
plt.text(0.125,0.325,"-5/3")
remove_axes(ax2)
plot_fig_label(ax2, xc=.05, yc=0.05 ,label="b",)

plt.ylim(1e-7,5e0)
plt.xlim(0.1,25)
plt.xlabel(r'Wavenumber [$|\mathbf{k}|/k_f$]')

ax2.set_yticks([])
ax2.spines['left'].set_visible(False)
plt.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left='off',        # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off

#plt.savefig("figs/spectrum_qg-niw")
plt.savefig(patho+'spectrum_qg-niw', pad_inces=0, bbox_inches='tight')

