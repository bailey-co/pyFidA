#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:59:55 2026
pyFidA.fidA_processing.op_common_processing.py

@author: Colleen Bailey (cbailey@sri.utoronto.ca), based on Matlab code by 
    Jamie Near

pyFidA processing functions that are needed for many basic operations in other
processing steps. Note that this module needs to be imported first in 
pyFidA.fidA_processing.__init__.py so that these functions are available
before other modules in the fidA_processing sub-package are loaded.

Functions:
    * add_phase
    * add_phase1
    * op_addphase
    * op_ampScale
    * op_addScans
    * op_subtractScans
    * op_freqrange
    * freqrange
    * op_zeropad
"""

import numpy as np
from pyFidA.fidA_common import GAMMA_DICT, FidAException

def add_phase(invec,added_phase):
    """
    Add equal amounts of phase to each point of a complex vector. This function
    operates on a numpy array. To operate on a pyFidA.FID object, use 
    op_addphase instead.
    
    Parameters
    ----------
    invec : numpy array
        Vector to add phase to. This function will also work for 
        complex scalars and multi-dimensional arrays.
    added_phase : float
        Amount of phase (in degrees) to add.

    Returns
    -------
    output vector
        0th order phased version of the input.

    """
    return invec*np.exp(1j*added_phase*np.pi/180)

def add_phase1(invec,ppm,timeShift,ppm0=4.65,B0=7,nucleus='1H'):
    """
    Add first order phase to a spectrum (added phase is linearly dependent on 
    frequency). This function operates on a numpy array. To phase shift a
    pyFidA.FID object, use op_addphase instead.

    Parameters
    ----------
    invec : numpy array
        input spectrum. Usually 1D but this function can operate on 
        multi-dimensional data as long as the first dimension corresponds to
        the frequency dimension.
    ppm : numpy array
        Frequency scale (ppm) corresponding to invec. In the case of a multi-
        dimensional invec, this vector can be a 1D vector that applies to every
        non-frequency dimension
    timeShift : float
        Amount of 1st order phase shift (specified as horizontal shift in 
        seconds in the time domain).
    ppm0 : float, optional
        The frequency "origin" (in ppm) of the 1st order phase shift (this
        point will undergo 0 phase shift). The default is 4.65.
    B0 : float, optional
        Magnetic field strength in Tesla (needed to convert ppm to Hz). The 
        default is 7.
    nucleus : string, optional
        The nucleus that will be used to determine the gyromagnetic ratio for
        the ppm to Hz conversion. The 'nucleus' string is used as a key for
        GAMMA_DICT. The default is '1H'.

    Returns
    -------
    phased_spec : numpy array
        1st order phased version of the input.

    """
    f=(ppm-ppm0)*GAMMA_DICT[nucleus.upper()]*B0
    # f is in Hz and timeshift in s, so multiplyling them gives result in 
    # cycles. Multiply by 2*pi to get phase in radians
    phased_spec=(invec.T*np.exp(-1j*f*timeShift*2*np.pi)).T
    return phased_spec

def op_addphase(indat,ph0,ph1=0,ppm0=4.65,suppress_plot=True):
    """
    Add 0th and/or 1st order phase to the spectrum of a FID object.

    Parameters
    ----------
    indat : pyFidA.FID object
        input spectrum to which phase will be added.
    ph0 : float
        Zeroth-order phase (in degrees) to be added to the fid.
    ph1 : float, optional
        First-order phase (in seconds) to be added to the spectrum.
    ppm0 : float, optional
        The frequency "origin" (in ppm) of the 1st order phase shift (this
        point will undergo 0 phase shift). The default is 4.65.
    suppress_plot : boolean, optional
        Whether to suppress the plot of the final spectrum. Only spectra with
        fewer than 3 dimensions will be plotted. The default is True.

    Returns
    -------
    outdat : pyFidA.FID object
        Phase-adjusted output spectrum.

    """
    outdat=indat.copy()
    outdat.fids=indat.fids*np.exp(1j*ph0*np.pi/180)
    outdat.added_ph0=indat.added_ph0+ph0
    if ph1!=0:
        outdat.specs=add_phase1(outdat.specs,indat.ppm,ph1,ppm0,indat.Bo,indat.nucleus[0])
        outdat.added_ph1=indat.added_ph1+ph1
    if outdat.ndim<3 and not suppress_plot:
        outdat.plot_spec(xlims=[outdat.ppm[0],outdat.ppm[-1]])
    return outdat

def op_ampScale(indat1,A):
    """
    Scale the amplitude of a spectrum by factor A. This function exists for 
    those replicating Matlab work. However, the __mult__ method is defined for 
    the FID object so that users can just call A*indat1 directly.

    Parameters
    ----------
    indat1 : FID object
        Spectrum to scale.
    A : float
        Amplitude scaling factor

    Returns
    -------
    outdat : FID object
        The output resulting from amplitude scaling.
    """
    return A*indat1

def op_addScans(indat1,indat2,subtract=False):
    """
    Add or subtract two scans. This function exists for those replicating
    Matlab work. However, the __add__ method is defined for the FID object 
    so that users can just call indat1+indat2 or indat1-indat2 directly.

    Parameters
    ----------
    indat1 : FID object
        First spectrum to add.
    indat2 : FID object
        Second spectrum to add.
    subtract : BINARY, optional
        Indicates whether to or subtract (True or non-zero int) or add (False 
        or 0) the two spectra. The default is False.

    Returns
    -------
    outdat : FID object
        The output resulting from adding (or subtracting) indat1 and indat2.

    """
    # I don't fully understand this case but Jamie includes it in Matlab for the
    # case of looping through multiple spectra. However, seems to assume that
    # input will be an empty structure for index 0, whereas Python is itself
    # 0-based for arrays. So this may not be what I want. Any such loops in 
    # processing functions should likely just use Python's sum function with an
    # appropriate start value. eg. sum(list_of_fids,start=0*list_of_fids[0])
    if indat1 is None:
        indat1=0*indat2
    if indat1.sz != indat2.sz:
        raise FidAException('ERROR:  Spectra must be the same number of points')
    if indat1.spectralwidth != indat2.spectralwidth:
        raise FidAException('ERROR:  Spectra must have the same spectral width')
    # No need to check dwelltime since it is automatically defined by spectral 
    # width, but I've added a check to central frequency, which is the final 
    # check needed to ensure that both spectra have the same ppm.
    if indat1.center_freq_ppm != indat2.center_freq_ppm:
        raise FidAException('ERROR:  Spectra must have the same central frequency')
    if subtract:
        outdat=indat1-indat2
    else:
        outdat=indat1+indat2
    return outdat

def op_subtractScans(indat1,indat2):
    """
    Subtract input 2 from input 1. This function exists for those replicating
    Matlab work. However, the __sub__ method is defined for the FID object 
    so that users can just call indat1-indat2 directly.

    Parameters
    ----------
    indat1 : FID object
        First spectrum
    indat2 : FID object
        Second spectrum to subtract from first.

    Returns
    -------
    outdat : FID object
        The output resulting from subtracting indat2 from indat1.

    """
    outdat=indat1-indat2
    return outdat

def op_freqrange(indat,ppmmin,ppmmax):
    fullspec=indat.specs.copy()
    outdat=indat.copy()
    indvals=np.logical_and(np.greater(indat.ppm,ppmmin),np.less(indat.ppm,ppmmax))
    outdat.specs=fullspec[indvals,...]
    # Need to redefine the ppm range, which is done by setting the center_freq_ppm
    # and the spectralwidth
    outdat.center_freq_ppm=ppmmin+(ppmmax-ppmmin)/2
    outdat.spectralwidthppm=np.abs(ppmmax-ppmmin)
    outdat.flags['freqranged']=True
    return outdat

def freqrange(inspec,ppm,ppmmin,ppmmax):
    # differs from op_freqrange in that that operates on a fid object, whereas
    # this only requires the frequency spectrum and ppm. It seems to have been
    # moved or removed in later fid-A versions so maybe can get rid of it?
    indvals=np.logical_and(np.greater(ppm,ppmmin),np.less(ppm,ppmmax))
    specpart=inspec[indvals,...]
    ppmpart=ppm[indvals]
    return ppmpart,specpart

def op_zeropad(indat,zpfact):
    outdat=indat.copy()
    continue_flag='y'
    if indat.flags['zeropadded']:
        continue_flag=input('WARNING: zero padding has already been performed. Continue? (y or n): ')
    if continue_flag.lower()=='y':
        newsz=list(indat.sz)
        newsz[0]=int(np.ceil(zpfact*indat.sz[0])) # adjusted to allow for non-integer factors
        outdat.fids=np.zeros(newsz,dtype=indat.fids.dtype)
        outdat.fids[:indat.sz[0],...]=indat.fids
    # Note that dwelltime, spectralwidth and center frequency are all unchanged
    # and other parameters (t, ppm, etc) are calculated from these plus matrix
    # sizes, so no need to recalculate here.
    outdat.flags['zeropadded']=True
    return outdat