import numpy as np
import matplotlib.pyplot as plt
import h5py
import os, glob
from pyspec import spectrum

plt.close('all')

pathi = "output/reference256/"

setup = h5py.File(pathi+"setup.h5")
k,l = setup['grid/k'][:], setup['grid/l'][:]
ki,li = np.meshgrid(k,l)
kf = 8*k[1]
wv2 = ki**2 + li**2
wv2i = 1./wv2
wv2i[0,0] = 0
wv = np.sqrt(wv2)
files = glob.glob(pathi+"snapshots/"+"*.h5")[750:]

fni = files[0]

for fni in files:
    snap = h5py.File(fni)

    qh = np.fft.fft2(snap['q'][:])
    ph = -wv2i*qh
    spec = 0.5 * np.abs(wv*ph)**2
    ki, Ei = spectrum.calc_ispec(k,l,spec)

    try:
        E = np.vstack([E,Ei[np.newaxis]])
    except:
        E = Ei[np.newaxis]

E = E.mean(axis=0)

plt.figure()
kr = np.array([0.1,22])
plt.loglog(kr,(kr**-3)/0.5,color='0.5')
#plt.loglog(kr,(kr**-(5/3))/1.5,color='0.5')
plt.loglog(ki/kf,E,linewidth=3)
plt.ylim(1e-4,1e2)
plt.xlim(0.1,22)
plt.xlabel(r'Wavenumber [$|\mathbf{k}|/k_f$]')
plt.ylabel(r'Kinetic energy density [m$^3$ s$^{-2}$]')
plt.text(8,1.25e-3,"-3")
#plt.text(7.5,4e-2,"-5/3")
plt.savefig("figs/spectrum_qg-niw")
