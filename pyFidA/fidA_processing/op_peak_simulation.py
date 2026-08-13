#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:17:40 2026

@author: nearlabmacbook1
"""

import numpy as np
from pyFidA.fidA_common import FID, GAMMA_DICT
from .alter_return_args import alter_return_args

@alter_return_args
def op_addNoise(indat,sdnoise,return_noisevec=None):
    """
    Add noise to a spectrum. Useful for simulated data

    Parameters
    ----------
    indat : FID object
        input data.
    sdnoise : float or numpy array
        If float, the standard deviation of Gaussian noise to be added. If numpy
        array, the (complex) values to be added onto indat.fids.

    Returns
    -------
    outdat : FID class.
        output data with noise added
    noisevec : numpy array
        The array of Gaussian noise values that were added to the data
    """
    outdat=indat.copy()
    if type(sdnoise) is np.ndarray:
        noisevec=sdnoise
    else:
        noisevec=sdnoise*np.random.randn(*indat.sz)+1j*sdnoise*np.random.randn(*indat.sz)
    outdat.fids=indat.fids+noisevec
    return outdat,noisevec

def op_gaussianPeak(npts,sw,Bo,lw,ppm0,amp,nucleus='1H',center_freq_ppm=4.65):
    """
    outdat=op_gaussianPeak(npts,sw,Bo,lw,ppm0,amp,nucleus='1H',center_freq_ppm=4.65)
    Generate a noiseless spectrum containing a single Gaussian peak with desired
    parameters (frequency, amplitude, linewidth, etc.)

    Parameters
    ----------
    npts : int
        Number of points in the spectrum.
    sw : float
        Spectral width of the spectrum in Hz.
    Bo : float
        Magnetic field strength in Tesla.
    lw : float
        Linewidth (FWHM) of the Gaussian peak in Hz.
    ppm0 : float
        Frequency of the Gaussian peak in ppm.
    amp : float
        Amplitude of the Gaussian peak. The amplitude corresponds to the area
        under the curve, not the maximum amplitude (differs from op_gauss 
        where the amplitude is the max value of that peak)
    nucleus : str, optional
        Nucleus. Used as key to obtain the gyromagnetic ratio from GAMMA_DICT. 
        The default is '1H'.
    center_freq_ppm : float, optional
        Center frequency of the spectrum in ppm. The default is 4.65 ppm.

    Returns
    -------
    outdat : FID object
        The Lorentzian lineshape peak as a pyFidA FID object.

    """
    dt=1/sw
    txfreq=Bo*GAMMA_DICT[nucleus]
    # Note that there seems to be an error in the Matlab code or its comments.  
    # It says that lw is the linewidth of the Guassian peak in Hz, which 
    # presumably means FWHM, but the Gaussian equation uses sigma, the standard 
    # deviation. This code converts lw assuming that is the FWHM in Hz into a 
    # sigma value for the lineshape calculation. Note that there is a factor of 
    # 1/2 to get the HWHM and then 2*pi to convert to radians/s, so that's 
    # where the extra factor of np.pi comes from
    sigma=lw/np.sqrt(2*np.log(2))*np.pi
    w0=(center_freq_ppm-ppm0)*txfreq*2*np.pi
    t=np.linspace(0,(npts-1)*dt,npts)
    fids=amp*np.exp(-(t)**2*(2/4*sigma**2))*np.exp(-1j*w0*t)#*1/np.sqrt(2*np.pi)
    outdat=FID(fids,spectralwidth=sw,txfreq=txfreq*1e6,te=0,tr=0,sequence='simulated',dims=['t'],nucleus=nucleus,center_freq_ppm=center_freq_ppm)
    return outdat

def op_lorentzianPeak(npts,sw,Bo,lw,ppm0,amp,nucleus='1H',center_freq_ppm=4.65):
    """
    outdat=op_lorentzianPeak(npts,sw,Bo,lw,ppm0,amp,nucleus='1H',center_freq_ppm=4.65)
    Generate a noiseless spectrum containing a single Lorentzian peak with desired
    parameters (frequency, amplitude, linewidth, etc.)

    Parameters
    ----------
    npts : int
        Number of points in the spectrum.
    sw : float
        Spectral width of the spectrum in Hz.
    Bo : float
        Magnetic field strength in Tesla.
    lw : float
        Linewidth (FWHM) of the Lorentzian peak in Hz.
    ppm0 : float
        Frequency of the Lorentzian peak in ppm.
    amp : float
        Amplitude of the Lorentzian peak. The amplitude corresponds to the area
        under the curve, not the maximum amplitude (differs from op_lorentz 
        where the amplitude is the max value of that peak)
    nucleus : str, optional
        Nucleus. Used as key to obtain the gyromagnetic ratio from GAMMA_DICT. 
        The default is '1H'.
    center_freq_ppm : float, optional
        Center frequency of the spectrum in ppm. The default is 4.65 ppm.

    Returns
    -------
    outdat : FID object
        The Lorentzian lineshape peak as a pyFidA FID object.

    """
    dt=1/sw
    txfreq=Bo*GAMMA_DICT[nucleus]
    # The lw should be entered in Hz and is the FWHM. This makes the constant 
    # for HWHM lw/2, then need to multiply by 2*pi to get in radians/s. 
    # So 2*pi*lw/2 = pi*lw
    decay=1/(lw*np.pi)
    w0=(center_freq_ppm-ppm0)*txfreq*2*np.pi
    t=np.linspace(0,npts*dt,npts)
    fids=amp*np.exp(-t/decay)*np.exp(-1j*w0*t)
    outdat=FID(fids,spectralwidth=sw,txfreq=txfreq*1e6,te=0,tr=0,sequence='simulated',dims=['t'],nucleus=nucleus,center_freq_ppm=center_freq_ppm)
    return outdat

def op_makeECArtifact(indat,A,tc):
    fFunc=A*np.exp(indat.t/tc)
    # Both indat.t and fFunc should be 1-D vectors with length matching the 
    # first dimension of indat.fids, so I can broadcast to make the multiplication
    # work
    outdat=indat.copy()
    newfid=indat.fids.T*np.exp(-1j*indat.t*fFunc*2*np.pi)
    outdat.fids=newfid.T
    return outdat

@alter_return_args
def op_makePhaseDrift(indat,totalDrift,noise,return_extra_args=None):
    # Matlab adjusts frequency for averages and subspecs. Presumably you would 
    # want to do coil combination before and, if you hadn't, don't adjust those
    # with different frequency drifts because those are data acquired from different
    # physical coils at the same time, so should have the same frequency drift.
    outdat=indat.copy()
    newfid=indat.fids
    if 'coils' in indat:
        print('WARNING: Coils have not been combined. Coil dimension will not be adjusted for frequency drift. Only averages and subspecs.')
    if np.isscalar(totalDrift):
        ph=np.reshape(np.linspace(0,totalDrift,indat.averages*indat.subspecs),[indat.subspecs,indat.averages])
    else:
        if len(totalDrift.shape)==1:
            totalDrift=np.expand_dims(totalDrift,0)
        if totalDrift.shape==tuple([indat.subspecs,indat.averages]):
            ph=totalDrift
        elif totalDrift.shape==tuple([indat.averages,indat.subspecs]):
            ph=totalDrift.T
        else:
            raise TypeError('ERROR: totalDrift must be either scalar or vector matching length of averages*subspecs')
    phDrift=ph+noise*np.random.randn(indat.subspecs,indat.averages)
    whichslice=[slice(None)]*indat.ndim
    # There is probably a faster/better way to do this in this case without for loops because
    # we're adding a constant phase, so I should be able to adjust phDrift to match the
    # size of fids, but leaving it like this for now.
    for specct in range(indat.subspecs): # This will run 1 loop if indat.subspecs=1, but there likely isn't any subspecs dim in this case, so don't need to set it in the slice
        if 'subspecs' in indat:
            whichslice[indat.dims['subspecs']]=specct
        for avct in range(indat.averages):
            if 'averages' in indat:
                whichslice[indat.dims['averages']]=avct
            # No need for transposition because not broadcasting in this case since exponential is a scalar, not a vector including time dimension
            newfid[tuple(whichslice)]=indat.fids[tuple(whichslice)]*np.exp(1j*phDrift[specct,avct]*np.pi/180)
    outdat.fids=newfid
    return outdat,phDrift

@alter_return_args
def op_makeFreqDrift(indat,totalDrift,noise,return_extra_args=None):
    # Matlab adjusts frequency for averages and subspecs. Presumably you would 
    # want to do coil combination before and, if you hadn't, don't adjust those
    # with different frequency drifts because those are data acquired from different
    # physical coils at the same time, so should have the same frequency drift.
    outdat=indat.copy()
    newfid=indat.fids
    if 'coils' in indat:
        print('WARNING: Coils have not been combined. Coil dimension will not be adjusted for frequency drift. Only averages and subspecs.')
    if np.isscalar(totalDrift):
        f=np.reshape(np.linspace(0,totalDrift,indat.averages*indat.subspecs),[indat.subspecs,indat.averages])
    else:
        if len(totalDrift.shape)==1:
            totalDrift=np.expand_dims(totalDrift,0)
        if totalDrift.shape==tuple([indat.subspecs,indat.averages]):
            f=totalDrift
        elif totalDrift.shape==tuple([indat.averages,indat.subspecs]):
            f=totalDrift.T
        else:
            raise TypeError('ERROR: totalDrift must be either scalar or vector matching length of averages*subspecs')
    fDrift=f+noise*np.random.randn(indat.subspecs,indat.averages)
    whichslice=[slice(None)]*indat.ndim
    for specct in range(indat.subspecs): # This will run 1 loop if indat.subspecs=1, but there likely isn't any subspecs dim in this case, so don't need to set it in the slice
        if 'subspecs' in indat:
            whichslice[indat.dims['subspecs']]=specct
        for avct in range(indat.averages):
            if 'averages' in indat:
                whichslice[indat.dims['averages']]=avct
            newfid[tuple(whichslice)]=(indat.fids[tuple(whichslice)].T*np.exp(1j*indat.t*fDrift[specct,avct]*2*np.pi)).T
    outdat.fids=newfid
    return outdat,fDrift
