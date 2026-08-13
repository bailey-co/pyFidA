#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:25:55 2026

@author: nearlabmacbook1
"""

from pyFidA.fidA_common import FidAException

def op_takeaverages(indat,idx):
    """
    outdat=op_takeaverages(indat,idx)
    Extract the averages with the indices corresponding to the 'idx' input array,
    int or slice object.

    Parameters
    ----------
    indat : FID object
        Input data.
    idx : int, array or slice object
        The average or set of averages to be extracted. Examples of possible
        idx formats include idx=5, idx=np.r_[5:10], idx=np.r_[5:10, 30:40], 
        idx=slice(5,50,5).

    Returns
    -------
    outdat : FID object
        Output data consisting of averages extracted from indat.

    """
    if 'averages' not in indat:
        raise FidAException('ERROR: There are no averages in this dataset! Aborting!')
    avg_slice=[slice(None)]*indat.ndim
    avg_slice[indat.dims['averages']]=idx
    outdat=indat[tuple(avg_slice)]
    # Added this in to try to make op_takeaverages the inverse of op_concatAverages
    # I think that it's debatable whether this is correct/useful but it's 
    # consistent with the Matlab code. Note that slicing using indat.__getitem__ 
    # above does not alter rawAverages
    outdat._rawAverages=outdat.averages
    # averages dimension should be automatically removed from outdat.dims 
    # by __getitem__ if idx is of type int
    if 'averages' not in outdat:
        # I want to change these to be properties in the FID object to be 
        # automatically returned based on dimensions or lack thereof (Update:
        # done, I think. Still need to test). Will still need an if statement here to set the flag.
        outdat.flags['averaged']=True
    return outdat

def op_takecoils(indat,idx):
    """
    outdat=op_takecoils(indat,idx)
    Extract the rf coil channels with the indices corresponding to the 'idx' 
    input array, int or slice object.

    Parameters
    ----------
    indat : FID object
        Input data.
    idx : int, array or slice object
        The coil or set of coils to be extracted. Examples of possible
        idx formats include idx=5, idx=np.r_[:3], idx=np.r_[:4, 8:], 
        idx=slice(0,16,2).

    Returns
    -------
    outdat : FID object
        Output data consisting of coils extracted from indat.

    """
    if 'coils' not in indat:
        raise FidAException('ERROR: There are not multiple coils in this dataset! Aborting!')
    coil_slice=[slice(None)]*indat.ndim
    coil_slice[indat.dims['coils']]=idx
    outdat=indat[tuple(coil_slice)]
    # coils dimension should be automatically removed from outdat.dims 
    # by __getitem__ if idx is of type int. I could add the flags in to that
    # part of the slicing function as well?
    if 'coils' not in outdat:
        outdat.flags['addedrcvrs']=True
    return outdat

def op_takeextras(indat,idx):
    """
    outdat=op_takeextras(indat,idx)
    Extract the extras with the indices corresponding to the 'idx' input array,
    int or slice object.

    Parameters
    ----------
    indat : FID object
        Input data.
    idx : int, array or slice object
        The extras or set of extras to be extracted. Examples of possible
        idx formats include idx=5, idx=np.r_[5:10], idx=np.r_[5:10, 30:40], 
        idx=slice(5,50,5).

    Returns
    -------
    outdat : FID object
        Output data consisting of extras extracted from indat.

    """
    if 'extras' not in indat:
        raise FidAException('ERROR: There are no extras in this dataset! Aborting!')
    extras_slice=[slice(None)]*indat.ndim
    extras_slice[indat.dims['extras']]=idx
    outdat=indat[tuple(extras_slice)]
    # extras dimension should be automatically removed from outdat.dims 
    # by __getitem__ if idx is of type int
    return outdat
        
def op_takesubspec(indat,idx):
    """
    outdat=op_takesubspec(indat,idx)
    Extract the subspectra with the indices corresponding to the 'idx' array,
    int or slice object.

    Parameters
    ----------
    indat : FID object
        Input data.
    idx : int, array or slice object
        The subspectrum or set of subspectra to be extracted. Examples of 
        possible idx formats include idx=0, idx=np.r_[:2], idx=np.r_[:1, 3:4], 
        idx=slice(0,None,2).

    Returns
    -------
    outdat : FID object
        Output data consisting of the subspectra extracted from indat.

    """
    if 'subspecs' not in indat:
        raise FidAException('ERROR: There are not subspectra in this dataset. ABORTING!')
    subspec_slice=[slice(None)]*indat.ndim
    subspec_slice[indat.dims['subspecs']]=idx
    outdat=indat[tuple(subspec_slice)]
    # Added this in to try to make op_takesusbpec the inverse of op_concatSubspecs
    # I think that it's debatable whether this is correct/useful but it's 
    # consistent with the Matlab code. Note that slicing using indat.__getitem__ 
    # above does not alter rawSubspecs
    outdat._rawSubspecs=outdat.subspecs
    # subspec dimension should be automatically removed from outdat.dims 
    # by __getitem__ if idx is of type int
    if 'subspecs' not in outdat:
        outdat.flags['subtracted']=True
        outdat.flags['isFourSteps']=False
    else:
        if outdat.sz[outdat.dims['subspecs']]!=4:
            outdat.flags['isFourSteps']=False
    return outdat
