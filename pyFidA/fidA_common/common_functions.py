#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:58:48 2026
pyFidA.fidA_common.common_functions.py

@author: Colleen Bailey (cbailey@sri.utoronto.ca), based on Matlab code by 
    Jamie Near

Functions that are used across multiple sub-modules in the pyFidA module.
Currently, these contain the functions to convert between free induction decays
in the time domain and spectra in the frequency domain:
    * fid_from_specs(oldspec)
    * spec_from_fids(oldfid)
    * phase(vec_to_phase)

The warning and exception types for pyFidA are also defined here:
    * FidAWarning
    * FidAException
These warning and exception types are thrown in cases of certain unexpected 
behaviours of pyFidA objects that are not easily defined by ValueError or
TypeError. eg. attempting to perform averaging on a FID object that does not
have an "averages" dimension. FidAWarning and its sub-types (eg. FidAWarningRF
can be used to suppress warning messages or raise them to errors using the
standard Python methods related to warning filters)

"""
import numpy as np
from scipy.fft import fftshift, fft, ifft

class FidAWarning(UserWarning):
    pass

class FidAException(Exception):
    pass

def fid_from_specs(oldspec):
    """
    Converts an array of frequency domain spectra into an array of time-domain
    free induction decays. Includes an np.roll correction for odd numbers of
    frequency points to avoid introducing a small phase shift in those cases.

    Parameters
    ----------
    oldspec : numpy array
        Array of spectra to be Fourier-transformed to the time domain. The 
        first dimension must contain the frequency domain information. Any 
        higher dimensions are optional.

    Returns
    -------
    newfids : numpy array
        Array of free induction decays with the same dimensions as oldspec, 
        where the first dimension is the time domain information.

    """
    if np.mod(oldspec.shape[0],2)==0:
        newfids=fft(fftshift(oldspec,axes=0),axis=0)
    else:
        newfids=fft(np.roll(fftshift(oldspec,axes=0),1,axis=0),axis=0)
    return newfids

def spec_from_fids(oldfid):
    """
    Converts an array of time domain free induction decays into an array of 
    freuqency domain spectra.

    Parameters
    ----------
    oldfid : numpy array
        Array of free induction decays to be inverse Fourier-transformed to the 
        frequency domain. The first array dimension must contain the time 
        domain information. Any higher dimensions are optional.

    Returns
    -------
    newspec : numpy array
        Array of spectra with the same dimensions as oldfid, where the first
        dimension is the frequency domain information.

    """
    newspec=fftshift(ifft(oldfid,axis=0),axes=0)
    return newspec

def phase(G):
    """
    Computes the phase (in radians) of a complex vector, including adjusting 
    for phase discontinuties

    Parameters
    ----------
    G : 1D complex numpy array OR complex number
        Vector to compute the phase on. This function also accepts a single
        complex number

    Returns
    -------
    phi : 1D complex numpy array OR complex number
        Phase of the input vector, in radians.

    """
    phi=np.arctan2(np.imag(G),np.real(G))
    if hasattr(phi, '__iter__') and len(phi)>1:
        df=phi[:-1]-phi[1:]
        idx=np.flatnonzero(np.abs(df)>3.5)
        for ict in idx:
            phi=phi+2*np.pi*np.sign(df[ict])*np.r_[np.zeros([ict+1,]),np.ones([len(phi)-ict-1,])]
    return phi