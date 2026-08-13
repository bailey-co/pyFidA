#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:41:38 2026

@author: nearlabmacbook1
"""

import numpy as np
from pyFidA.fidA_common import FidAException

def op_concatAverages(indat1, indat2):
    """
    Concatenate two scans along the averages dimension. Two scans with 50 
    averages each will now look like a single scan with 100 averages. For
    looping, it is possible to enter None as the first input. In this case, the
    function will simply return the second input.

    Parameters
    ----------
    indat1 : FID object
        First input spectra to be concatenated.
    indat2 : TFID object
        Second input spectra to be concatenated.

    Returns
    -------
    outdat : FID object
        Output following concatenation of inputs along the averages dimension.
    """
    # I don't fully understand this case but Jamie includes it in Matlab for the
    # case of looping.
    if indat1 is None:
        outdat=indat2.copy()
    else:
        # Jamie has a check that both inputs have the same number of averages
        # (unless one or both input have no averages) but I don't understand 
        # why that restriction exists. I just need to expand_dims for singleton 
        # cases. The other dimensions have to match but np.concatenate will 
        # already fail if that is untrue
        if 'averages' not in indat1 and 'averages' not in indat2:
            avgdim=max(indat1.dims.values())+1
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
        else: # both dimensions have averages. Simplest case
            fid1=indat1.fids
            fid2=indat2.fids
            avgdim=indat1.dims['averages']
            dimlist=indat1._dimlist.copy()
        outfid=np.concatenate((fid1,fid2),axis=avgdim)
        outdat=indat1.copy()
        outdat.fids=outfid
        outdat._dimlist=dimlist
        outdat._rawAverages=indat1._rawAverages+indat2._rawAverages
        outdat.flags['averaged']=False
    return outdat
    
def op_concatFreq(indat1,indat2,shift=None):
    # I don't understand the use case for this. Maybe for plotting but then
    # you should just deal with that in the plot. Function seems incomplete in
    # Matlab. Documentation hasn't been updated. Lots of concatAverages has 
    # been copied and commented out but not deleted.
    # Note that things are implemented quite differently here compared with
    # Matlab because of how ppm is defined by the spectral width and center 
    # frequency, rather than being directly settable.
    if indat1.dims['t'] != indat2.dims['t']:
        # Really I want the actual 't' values to be the same, no? Otherwise, their
        # ppm values can't be easily concatenated alongside one another
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
        # The ppm calculation has to work differently because of how I've set
        # up the setters in the FID object
        newspectralwidthppm=indat1.spectralwidthppm+dummy_freqrange+indat2.spectralwidthppm
        if indat1.ppm[-1]<indat1.ppm[0]:
            newcenterfreq=indat1.ppm[-1]+newspectralwidthppm/2
        else:
            newcenterfreq=indat1.ppm[0]+newspectralwidthppm/2
    outdat=indat1.copy()
    outdat.specs=newspec
    outdat.spectralwidthppm=newspectralwidthppm
    outdat.center_freq_ppm=newcenterfreq
    # Matlab sets the averaged flag to false. I think it's a holdover from
    # copying from op_concatAverages and doesn't belong so I've removed it. But
    # I'm also not sure it matters since you don't really want to be doing new 
    # operations on this FID because its time domain representation is 
    # completely messed up.
    return outdat
    
def op_concatSubspecs(indat1,indat2):
    if indat1.dims['subspecs'] != indat2.dims['subspecs'] or indat1.dims['t'] != indat2.dims['t'] or indat1.dims['averages'] != indat2.dims['averages'] or indat1.dims['coils'] != indat2.dims['coils']:
        raise FidAException('ERROR: subspecs dimensions must be the same for both inputs')
    # if there is no subspecs dimension, make one. Since we already checked that
    # the dimensions are the same, we can do fewer cases than op_concatAverages
    if 'subspecs' not in indat1 and 'subspecs' not in indat2:
        ssdim=max(indat1.dims.values())+1
        fid1=np.expand_dims(indat1.fids,axis=ssdim)
        fid2=np.expand_dims(indat2.fids,axis=ssdim)
        dimlist=indat1._dimlist.copy()+['subspecs']
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