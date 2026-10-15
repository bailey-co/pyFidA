#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:41:38 2026
pyFidA.fidA_processing.op_concatenation.py

@author: Colleen Bailey (cbailey@sri.utoronto.ca), based on Matlab code by 
    Jamie Near

pyFidA processing functions that are needed for many basic operations in other
processing steps. Note that this module needs to be imported first in 
pyFidA.fidA_processing.__init__.py so that these functions are available
before other modules in the fidA_processing sub-package are loaded.

Functions:
    * op_concatAverages
    * op_concatFreq
    * op_concatSubspecs
"""

import numpy as np
from pyFidA.fidA_common import FidAException

def op_concatAverages(indat1, indat2):
    """
    Concatenate two scans along the averages dimension. Two scans with 50 
    averages, the output will be a single pyFidA.FID object with 100 averages. 
    Scans without an explicit averages dimension can also be entered (and will
    be expanded to have an averages dimension of size 1). For looping, it is 
    possible to enter None as the first input. In this case, the function will 
    simply return the second input.

    Parameters
    ----------
    indat1 : pyFidA.FID object
        First input spectra to be concatenated.
    indat2 : pyFidA.FID object
        Second input spectra to be concatenated.

    Returns
    -------
    outdat : FID object
        Output following concatenation of inputs along the averages dimension.
    """
    if indat1 is None:
        outdat=indat2.copy()
    else:
        dims1_noavg=[dimnm for dimnm in indat1._dimlist if dimnm!='averages']
        dims2_noavg=[dimnm for dimnm in indat2._dimlist if dimnm!='averages']
        if len(dims1_noavg)!=len(dims2_noavg) or any([d1!=d2 for d1,d2 in zip(dims1_noavg,dims2_noavg)]):
            raise FidAException('ERROR: Data dimensions other than averages must match. indat1 has dims {:s}. indat2 has dims {:s}'.format(str(indat1._dimlist),str(indat2._dimlist)))
        # Deal with absent averages dimensions, if squeezed out
        if 'averages' not in indat1 and 'averages' not in indat2:
            avgdim=indat1.ndim
            fid1=np.expand_dims(indat1.fids,axis=avgdim)
            fid2=np.expand_dims(indat2.fids,axis=avgdim)
            dimlist=indat1._dimlist.copy()+['averages']
        elif 'averages' not in indat1 and 'averages' in indat2:
            fid1=np.expand_dims(indat1.fids,axis=indat2.dims['averages'])
            fid2=indat2.fids
            avgdim=indat2.dims['averages']
            dimlist=indat2._dimlist.copy()
        elif 'averages' in indat1 and 'avereages' not in indat2:
            fid2=np.expand_dims(indat2.fids,axis=indat1.dims['averages'])
            fid1=indat1.fids
            avgdim=indat1.dims['averages']
            dimlist=indat1._dimlist.copy()
        else: # both dimensions have averages.
            fid1=indat1.fids
            fid2=indat2.fids
            avgdim=indat1.dims['averages']
            dimlist=indat1._dimlist.copy()
        outdat=indat1.copy()
        outdat.fids=np.concatenate((fid1,fid2),axis=avgdim)
        outdat._dimlist=dimlist
        outdat._rawAverages=indat1._rawAverages+indat2._rawAverages
        outdat.flags['averaged']=False
    return outdat
    
def op_concatFreq(indat1,indat2,shift=None):
    """
    Note that this function "stitches" two spectra together so that they can
    can appear side-by-side in a plot. There is a check that the spectral
    resolutions match but the center frequency of indat2 is adjusted based on
    shift to determine how much space will separate it from indat1 in the ppm
    dimension. The time vector of the output spectrum will therefore no longer
    have a sensible temporal meaning because of the "apparent" new spectral 
    width.

    Parameters
    ----------
    indat1 : pyFidA.FID
        First spectrum in the ppm dimension.
    indat2 : pyFidA.FID
        Second spectrum to be concatenated after indat1 in the ppm dimension.
    shift : float, optional
        The point, in ppm, where the second spectrum should be place in the
        concatenation. The default is None, which will place the second spectrum
        where the first ends.

    Raises
    ------
    FidAException
        The time vectors for the two spectra do not match (and therefore the
        spectral resolutions do not match, nor do the spectral widths.
    ValueError
        The shift provided is less than the spectral width of indat1.

    Returns
    -------
    outdat : pyFidA.FID
        Output "spectrum" where indat1 and indat2 will appear alongside each
        other in the ppm dimension. Note that the temporal information will be
        meaningless in most cases unless the spectra truly have the ppm values
        that they have been assigned. The outdat spectrum should therefore not
        be used for further processing and seems like it would mainly be useful
        for plotting.

    """
    if all([indat1.t==indat2.t]):
        raise FidAException('time/frequency dimension must be the same for both inputs.')
    dppm=np.abs(indat1.ppm[1]-indat1.ppm[0])
    if shift is None:
        newspec=np.concatenate((indat1.specs,indat2.specs),axis=indat1.dims['t'])
        # The ppm calculation has to work differently because of how I've set
        # up the setters in the FID object
        newspectralwidthppm=2*indat1.spectralwidthppm+dppm
        # using ppm[1]-ppm[0] instead of dppm because the signs will be consistent this way
        newcenterfreq=indat1.ppm[-1]+(indat1.ppm[1]-indat1.ppm[0])/2
    elif shift<indat1.spectralwidthppm:
        raise ValueError('ERROR: shift must be at least as large as the spectral width in ppm! ABORTING!!')
    else:
        # Make a dummy spectrum that is the same width as the desired spectral gap
        dummyspec=0*indat1.specs
        # as I interpret the Matlab code, shift is the ppm value where you want
        # the second spectrum to start. gap is then the distance between these two
        gap=shift-indat1.spectralwidthppm
        npts=np.int(np.ceil(gap/dppm))
        dummyspec=dummyspec[:npts,...]
        dummy_freqrange=npts*dppm # Note that you may end up with a shift slightly different than specified
        newspec=np.concatenate((indat1.specs,dummyspec),axis=indat1.dims['t'])
        newspec=np.concatenate((newspec,indat2.specs),axis=indat1.dims['t'])
        newspectralwidthppm=indat1.spectralwidthppm+dummy_freqrange+indat2.spectralwidthppm
        if indat1.ppm[-1]<indat1.ppm[0]:
            newcenterfreq=indat1.ppm[-1]+newspectralwidthppm/2
        else:
            newcenterfreq=indat1.ppm[0]+newspectralwidthppm/2
    outdat=indat1.copy()
    outdat.specs=newspec
    outdat.spectralwidthppm=newspectralwidthppm
    outdat.center_freq_ppm=newcenterfreq
    return outdat
    
def op_concatSubspecs(indat1,indat2):
    """
    Concatenate two scans along the subspecs dimension. Two scans with 2 
    subspecs each will not look like a single scan with 4 subspecs. Scans 
    without an explicit subspecs dimension can also be entered (and will
    be expanded to have a subspecs dimension of size 1).

    Parameters
    ----------
    indat1 : pyFidA.FID
        First input spectrum with subspecs to be concatenated.
    indat2 : pyFidA.FID
        Second input spectrum with subspecs to be concatenated.

    Raises
    ------
    FidAException
        Thrown when dimensions (other than subspecs) do not match for the two 
        input subspecs.

    Returns
    -------
    outdat : pyFidA.FID
        Output spectrum with all subspecs concatenated.

    """
    dims1_noss=[dimnm for dimnm in indat1._dimlist if dimnm!='subspecs']
    dims2_noss=[dimnm for dimnm in indat2._dimlist if dimnm!='subspecs']
    if len(dims1_noss)!=len(dims2_noss) or any([d1!=d2 for d1,d2 in zip(dims1_noss,dims2_noss)]):
        raise FidAException('ERROR: Data dimensions other than subspecs must match. indat1 has dims {:s}. indat2 has dims {:s}'.format(str(indat1._dimlist),str(indat2._dimlist)))
    if 'subspecs' not in indat1 and 'subspecs' not in indat2:
        ssdim=indat1.ndim
        fid1=np.expand_dims(indat1.fids,axis=ssdim)
        fid2=np.expand_dims(indat2.fids,axis=ssdim)
        dimlist=indat1._dimlist.copy()+['subspecs']
    elif 'subspecs' not in indat1 and 'subspecs' in indat2:
        fid1=np.expand_dims(indat1.fids,axis=indat2.dims['subspecs'])
        fid2=indat2.fids
        ssdim=indat2.dims['subspecs']
        dimlist=indat2._dimlist.copy()
    elif 'subspecs' in indat1 and 'subspecs' not in indat2:
        fid2=np.expand_dims(indat2.fids,axis=indat1.dims['subspecs'])
        fid1=indat1.fids
        ssdim=indat1.dims['subspecs']
        dimlist=indat1._dimlist.copy()
    else: # both dimensions have averages. Simplest case
        fid1=indat1.fids
        fid2=indat2.fids
        ssdim=indat1.dims['subspecs']
        dimlist=indat1._dimlist.copy()
    outdat=indat1.copy()
    outdat.fids=np.concatenate((fid1,fid2),axis=ssdim)
    outdat._dimlist=dimlist
    outdat._rawSubspecs=indat1._rawSubspecs+indat2._rawSubspecs
    outdat.flags['subtracted']=False
    return outdat