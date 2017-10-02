#
# Utility functions used in simulation and plotting
#   scripts
#

import matplotlib.pyplot as plt
import numpy as np

def remove_axes(ax,bottom=False):

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if bottom:
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        #ax.yaxis.set_ticks_position('left')
        #ax.xaxis.set_ticks_position('bottom')


def save_parameters(model):
    """ Save simulation parameters """
    fno = model.path+"/parameters.h5"
    h5file = h5py.File(fno, 'w')

    h5file.create_dataset("dimensional/Ue", data=(Ue))
    h5file.create_dataset("dimensional/Uw", data=(Uw))
    h5file.create_dataset("dimensional/ke", data=(ke))
    h5file.create_dataset("dimensional/m", data=(m))
    h5file.create_dataset("dimensional/N", data=(N))
    h5file.create_dataset("dimensional/f0", data=(f0))
    h5file.create_dataset("dimensional/Te", data=(Te))
    h5file.create_dataset("dimensional/L", data=(L))
    h5file.create_dataset("dimensional/nu4", data=(nu4))
    h5file.create_dataset("dimensional/nu4w", data=(nu4w))
    h5file.create_dataset("nondimensional/nx", data=(nx))
    h5file.create_dataset("nondimensional/alpha", data=(alpha))
    h5file.create_dataset("nondimensional/hslash", data=(hslash))
    h5file.create_dataset("nondimensional/Ro", data=(Ro))

    h5file.close()

def plot_fig_label(ax, xc=.95, yc=0.075 ,label="a",boxstyle='circle',
                        facecolor='white',edgecolor=None,color=None,alpha=1.):
    """ Plot label numbering for multi-panel figures """
    plt.text(xc, yc,label,
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes,bbox=dict(boxstyle=boxstyle,
                                                    facecolor=facecolor,
                                                    edgecolor=edgecolor,
                                                    alpha=alpha))

def wave_fields(phi,f0,lam2,t,k,l,m,z=0):
    """ Calculate near-inertial fields """
    
    phase = m*z - f0*t 
    phih = np.fft.fft2(phi)
    phi_phase = phi*np.exp(1j*phase)

    phix, phiy = np.fft.ifft2(1j*k*phih), np.fft.ifft2(1j*l*phih)

    dphi = phix+1j*phiy

    u, v = phi_phase.real, phi_phase.imag
    b = np.real(m*f0*lam2*dphi*np.exp(1j*phase))
    p = np.real(1j*f0*lam2*dphi*np.exp(1j*phase))
    w = np.real(1j*dphi*np.exp(1j*phase)/m)

    return u, v, w, p ,b


def balanced_fields(q,k,l,f0):
    """ Calculates balanced fields """
    wv2 = k**2 + l**2
    wv2i = wv2**-1
    wv2i[0,0] = 0.

    qh = np.fft.fft2(q)
    ph  = -wv2i*qh
    u,v = np.fft.ifft2(-1j*l*ph).real, np.fft.ifft2(1j*k*ph).real
    p = f0*np.fft.ifft2(ph).real

    return u, v, p


