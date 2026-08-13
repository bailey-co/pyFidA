#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 12:29:21 2022
fidA_processing.op_main_processing.py

@author: cbailey, based on Matlab code by Jamie Near
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d,CubicSpline
from scipy.linalg import hankel,svd
from pyFidA.fidA_common import spec_from_fids, FidAException, phase
from .op_common_processing import op_addphase, op_freqrange, op_zeropad, add_phase, add_phase1
from .alter_return_args import alter_return_args
from .op_splitspecs import op_takesubspec

def op_addphaseSubspec(indat,ph0):
    """
    For spectra with two subspectra, add zero order phase to the second 
    subspectra in a dataset. With edited spectroscopy sequences (eg. mega-
    press), there can be small frequency drifts between edit-on and edit-off 
    spectra that can result in residual signals from uncoupled spins 
    (Cr, Ch, etc).

    Parameters
    ----------
    indat : FID object
        Input data with two subspectra.
    ph0 : float
        Phase (in degrees) to add to the second subspectrum.

    Returns
    -------
    outdat : FID object
        Output dataset with phase adjusted subspectrum..

    """
    # Updated to use the __container__ method
    if 'coils' in indat and indat.sz[indat.dims['coils']]>1:
        raise FidAException('ERROR: Cannot operate on data with multilple coils! ABORTING!!')
    if 'averages' in indat:
        raise FidAException('ERROR: Cannot operate on data with multiple averages! ABORTING!!')
    if indat.subspecs!=2:
        raise FidAException('ERROR: Input spectrum must have two subspecs! ABORTING!!')
    outdat=indat.copy()
    # This slice1 construction should allow the 2nd spectrum of the subspec 
    # dimension to be selected for calculation. Keeps function generalizable.
    slice1=[slice(None)]*outdat.ndim
    slice1[outdat.dims['subspecs']]=1
    outdat.fids[tuple(slice1)]=outdat.fids[tuple(slice1)]*np.exp(1j*ph0*np.pi/180)
    return outdat

@alter_return_args
def op_addrcvrs(indat,phasept=0,mode='w',coilcombos=None,return_extra_args=None):
    """
    Perform weighted coil recombination for MRS data acquired with receiver
    coil array.

    Parameters
    ----------
    indat : FID object
        Input data with multiple receiver info.
    phasept : float, optional
        Point of fid to use for phase estimation and amplitude when mode='w'. 
        The default is 0.
    mode : char, optional
        Method for estiamting the coil weights and phases if not provided in 
        coilcombos. Can be:
            'w' - performs amplitude weighting of channels based on the max 
                signal of each coil channel
            'h' - performs amplitude weighting of channles based on the max signal 
                of each coil channel divided by the square of the noise in each 
                coil channel (as described by Hall et al. Neuroimage 2014).
            The default is 'w'.
    coilcombos : dict with keys 'phs' and 'sigs', optional
        The predetermined coil phases (in degrees) and amplitudes as generated
        by op_getcoilcombos. If this argument is provided, the 'point' and 'mode'
        arguments will be ignored. The default is None.

    Returns
    -------
    outdat : FID object
        The output dataset with coil channels combined.
    fids_presum : numpy array
        Input time domain data (fid) with coils phase-adjusted, before combination.
    specs_presum : numpy array
        Input frequency domain data (spectrum) with coils phase-adjusted, before 
        combination.
    coilcombos : dict with keys 'phs' and 'sigs'
        The vectors of the coil phases (in degrees) used for alignment and the 
        coil weights.

    """
    if 'coils' not in indat or indat.sz[indat.dims['coils']]==1:
        print('WARNING: Only one receiver channel found! Returning input without modification!')
        outdat=indat.copy()
        outdat.flags['addedrcvrs']=True
        fids_presum=indat.fids
        specs_presum=indat.specs
        coilcombos={'phs':0, 'sigs':1}
    else:
        #To get best possible SNR, add the averages together (if it hasn't already been done):
        if 'averages' in indat:
            av=op_averaging(indat)
        else:
            av=indat.copy()
        # also, for best results, we will combine all subspectra:
        if coilcombos is None:
            if indat.flags['isFourSteps']:
                av=op_fourStepCombine(av)
            if 'subspecs' in indat:
                av=op_combinesubspecs(av,'summ')
        if coilcombos is None:
            # Code was repeated in Matlab rather than calling function
            coilcombos=op_getcoilcombos(av,phasept=phasept,mode=mode)
        phs=coilcombos['phs']
        sigs=coilcombos['sigs']/np.linalg.norm(coilcombos['sigs'].flatten())
        # now expand these matrices to match the size of indat.fids for 
        # multiplication. Each coil has a different phase and amplitude
        ph=np.ones(indat.sz)
        sig=np.ones(indat.sz)
        slice1=[slice(None)]*indat.fids.ndim
        for nct in range(indat.sz[indat.dims['coils']]):
            slice1[indat.dims['coils']]=nct
            ph[tuple(slice1)]=phs[nct]
            sig[tuple(slice1)]=sigs[nct]
        
        # now apply the phases by multiplying the data by exp(-i*ph);
        fids=indat.fids*np.exp(-1j*ph*np.pi/180)
        fidobj_presum=indat.copy()
        fidobj_presum.fids=fids
        fids_presum=fidobj_presum.fids
        specs_presum=fidobj_presum.specs
        fids=fids*sig
        #Make the coilcombos structure:
        coilcombos={'phs':phs,'sigs':sigs}
        #now sum along coils dimension
        fids=np.sum(fids,axis=indat.dims['coils'])
        outdat=indat.copy()
        outdat.fids=np.squeeze(fids)
        # change the dims variables
        outdat._dimlist.remove('coils')
        outdat.flags['addedrcvrs']=True    
    return outdat,fids_presum,specs_presum,coilcombos

# Matlab fidA has a huge number of align functions. I have attempted to simplify
# and avoid duplicating code. However, I did create functions with the Matlab
# names for those familiar with the Matlab calls. The "fundamental" alignment
# function here in Python with the fitting functions is op_alignScans
@alter_return_args
def op_alignAllScans(inlist, tmax=None, ref='f', mode='fp',freq_range=None,initPars=None,return_extra_args=None):
    # Make sure input is a list of length 2 or greater
    if type(inlist) is not list or len(inlist)<2:
        TypeError('ERROR: The input must be a list of two or more MRS datasets in FID-A FID object form. ABORTING!!')
    # Figure out what reference spectrum will be
    if ref=='f':
        inref=inlist[0]
    elif ref=='a':
        inref=sum(inlist,start=inlist[0]*0)/len(inlist) # you have to provide a start value for sum of type FID.
    else:
        ValueError('ERROR: Reference spectrum type not recognized. Must be either "f" or "a".')
    ampref=np.max(np.abs(inref))
    outlist=list()
    phlist=list(); frqlist=list()
    for specct,eachspec in enumerate(inlist):
        if ref=='f' and specct==0:
            outlist.append(inlist[0])
            phlist=[0]
            frqlist=[0]
        else:
            amp1=np.max(np.abs(eachspec))
            [dummyOut,dummy_ph,dummy_frq]=op_alignScans(inref, eachspec*ampref/amp1, tmax=tmax, mode=mode, freq_range=freq_range, initPars=initPars, return_extra_args=True)
            phlist.append(dummy_ph)
            frqlist.append(dummy_ph)
            outlist.append(op_addphase(op_freqshift(eachspec,dummy_frq),dummy_ph))
    return outlist, phlist, frqlist

@alter_return_args
def op_alignAllScans_fd(inlist, fmin, fmax, tmax=None, ref='f', mode='fp', initPars=None, return_extra_args=None):
    outlist,phlist,frqlist=op_alignAllScans(inlist, tmax, ref=ref, mode=mode,freq_range=[fmin,fmax], initPars=initPars, return_extra_args=True)
    return outlist, phlist, frqlist
    
@alter_return_args
def op_alignAverages(indat,tmax=None,med='n',ref=None,mode='fp',freq_range=None,initPars=None,return_extra_args=None):
    outdat=indat.copy()
    fs=0; phs=0;
    if 'coils' in indat and indat.sz[indat.dims['coils']]>1:
        raise FidAException('ERROR: I think it only makes sense to do this after you have combined the channels using op_addrcvrs. ABORTING!!')
    elif med.lower()=='r' and ref is None:
        raise FidAException("ERROR:  If using the 'r' option for input variable 'med', then an argument for 'ref' must be provided")
    elif indat.averages==1:
        print('WARNING: No averages found. Returning input without modification')
    # Note that this function is not as generalizable as some others. It 
    # assumes that you only have averages and/or subspecs. This makes some
    # sense because the fitting function needs a 1D vector to run least squares
    # minimization so you can't just keep other directions.
    elif not all([dimnm in ['t','averages','subspecs'] for dimnm in indat._dimlist]):
        raise FidAException('ERROR: Only t, averages and subspecs dimensions allowed for op_alignAverages. This FID has dimensions ({:s})'.format(', '.join(indat._dimlist)))
    else:
        # Note that, even though tmax can be calculated in op_alignScans, it makes
        # sense to do it here so that the same tmax is used for each average
        if tmax is None:
            # Just need to find an SNR point so I can use the first spectrum??
            print('tmax not supplied. Calculating when SNR drops below 5...')
            slicetmp=[0]*indat.ndim
            slicetmp[0]=slice(None)
            sig=np.abs(indat.fids[tuple(slicetmp)])
            slicetmp[0]=slice(int(np.ceil(0.75*indat.fids.shape[0])),None)
            noise=np.std(np.real(indat.fids[tuple(slicetmp)]),axis=0)
            tmaxpt=np.flatnonzero(sig/noise>5)[-1]
            tmax=indat.t[tmaxpt]
            print('tmax = {:.2e} ms'.format(tmax*1000))
        # Note that indat.subspecs will return 1 in the case where there are no 
        # subspecs, which allows the loop to run. Matrices will be squeezed and
        # re-ordered before return in order to match input form
        B=indat.subspecs
        fs=np.zeros([indat.averages,B])
        phs=np.zeros_like(fs)
        newfid=np.zeros([indat.sz[indat.dims['t']],indat.averages,B],dtype=np.complex64)
        for mct in range(B):
            # Create a slice to select the mct'th subspec
            subspec_pts=[slice(None)]*indat.ndim
            if 'subspecs' in indat:
                subspec_pts[indat.dims['subspecs']]=mct
            # after this, tmpfid should no longer have a subspecs dimension, even if indat did (removed during slicing)
            tmpfid=indat[tuple(subspec_pts)]
            # Get the reference spectrum
            if med.lower()=='y':
                ref2=op_median(tmpfid)
                indmin=-1
            elif med.lower()=='a':
                ref2=op_averaging(tmpfid)
                indmin=-1
            elif med.lower()=='n':
                # Since tmpfid only has averages (subspecs removed), "metric"
                # is a 1D array of len averages. You want the min value of that 
                # metric
                ref2,metric,badavgs=op_rmbadaverages(tmpfid,return_extra_args=True)
                indmin=np.argmin(metric)
            elif med.lower()=='f':
                # Not sure why this option isn't available in Matlab but makes
                # sense to add it
                first_pts=[slice(None)]*tmpfid.ndim
                first_pts[tmpfid.dims['averages']]=0
                ref2=tmpfid[tuple(first_pts)]
                indmin=0
            elif med.lower()=='r':
                if 'subspecs' in ref:
                    # Presumably ref does not have averages, so need to make a new slice object
                    subspec_pts2=[slice(None)]*ref.ndim
                    subspec_pts2[indat.dims['subspecs']]=mct
                    ref2=ref[tuple(subspec_pts2)]
                else:
                    ref2=ref.copy()
                indmin=-1
            else:
                raise ValueError("ERROR: Invalid value for 'med'. Allowed values are 'y', 'a', 'n', 'f' and 'r'.")
            # The selection of points between tmin and tmax is done in 
            # op_alignScans, as is any freq_range restriction, so just pass 
            # those arguments in rather than repeating code
            for avct in range(indat.averages):
                newfidslice=[slice(None)]*tmpfid.ndim
                newfidslice[tmpfid.dims['averages']]=avct
                if avct==indmin:
                    newfid[:,avct,mct]=ref2.fids
                    fs[avct,mct]=0
                    phs[avct,mct]=0
                else:
                    tmpobj,phs[avct,mct],fs[avct,mct]=op_alignScans(ref2, tmpfid[tuple(newfidslice)], tmax=tmax, mode=mode, freq_range=freq_range,initPars=initPars,return_extra_args=True)
                    newfid[:,avct,mct]=tmpobj.fids
        # If necessary, re-order dimensions and squeeze subspecs dimension if singleton
        if 'subspecs' in indat and indat.dims['averages']>indat.dims['subspecs']:
            outdat.fids=np.transpose(newfid,[0,2,1])
        else:
            outdat.fids=np.squeeze(newfid)
        if 'f' in mode:
            outdat.flags['freqcorrected']=True
        if 'p' in mode:
            outdat.flags['phasecorrected']=True
    return outdat,np.squeeze(phs),np.squeeze(fs)

@alter_return_args
def op_alignAverages_fd(indat, minppm, maxppm, tmax=None, med='n', ref=None, mode='fp', initPars=None, return_extra_args=None):
    out1,ph,frq=op_alignAverages(indat, tmax=tmax, med=med, ref=ref, mode=mode, freq_range=[minppm, maxppm],initPars=initPars,return_extra_args=True)
    return out1, ph, frq

@alter_return_args
def op_alignISIS(indat,tmax=None, mode='diff', freq_range=None, initPars=None, return_extra_args=None):
    if not all([dval in ['t','averages','subspecs'] for dval in indat._dimlist]):
        raise FidAException('ERROR: op_alignISIS can only operate on data with subspecs and (optionally) averages dimensions. Combine coils with op_addrcvrs or limit other dimensions before running. ABORTING!!')
    if 'subspecs' not in indat:
        raise FidAException('ERROR: Must have multiple subspectra. ABORTING!!')
    elif indat.subspecs>2:
        print('WARNING: op_alignISIS only aligns first two subspectra.')
        
    # Here, the second subspec is aligned with the first. In Matlab, the aim 
    # seems to be to align both subspectra with their sum/difference but the
    # fitting process is strange, particularly for multiple averages.
    subspec_slice=[slice(None)]*indat.ndim
    subspec_slice[indat.dims['subspecs']]=0
    outdat=indat.copy()
    phs=np.zeros([indat.averages,indat.subspecs])
    fs=np.zeros_like(phs)
    for avct in range(indat.averages):
        if 'averages' in indat:
            subspec_slice[indat.dims['averages']]=avct
        subspec_slice[indat.dims['subspecs']]=0
        base_ref=indat[tuple(subspec_slice)]
        for sct in range(1,indat.subspecs):
            subspec_slice[indat.dims['subspecs']]=sct
            infloat=indat[tuple(subspec_slice)]
            # Following the convention in op_combinesubspecs, mode 'diff' is for 
            # summing (all postive). Mode 'summ' is for subtraction.
            if mode=='summ' and np.mod(sct,2)==1:
                infloat=-1*infloat
            outtmp,phs[avct,sct],fs[avct,sct]=op_alignScans(base_ref, infloat,tmax=tmax,mode='fp',freq_range=freq_range,initPars=initPars,return_extra_args=True)
            if mode=='summ' and np.mod(sct,2)==1:
                outdat.fids[tuple(subspec_slice)]=-1*outtmp.fids
            else:
                outdat.fids[tuple(subspec_slice)]=outtmp.fids
    if 'averages' in indat and indat.dims['subspecs']<indat.dims['averages']:
        phs=phs.T
        fs=fs.T
    outdat.flags['freqcorrected']=True
    outdat.flags['phasecorrected']=True
    return outdat, np.squeeze(phs), np.squeeze(fs)

@alter_return_args
def op_alignMPSubspecs(indat,mode='o',initPars=[0,0],ppmWeights=None,return_extra_args=None):
    # No tmax since work is done on the frequency spectrum. And, I guess, if we
    # can't use op_alignScans anyway, we can get rid of freq_range and just use
    # ppmWeights (set to 0 in any part of the frequency range that you don't want)
    # This one also needs its own fitting function because it is comparing the
    # spectra, not the fid, and the weights are for the points on the spectrum
    # Averages not allowed in this case (compare to op_alignISIS). Nor other
    # dimensions beyond subspecs
    if not all([dval in ['t','subspecs'] for dval in indat._dimlist]) or indat.ndim>2:
        raise FidAException('ERROR: op_alignISIS can only operate on data with subspecs. Combine coils with op_addrcvrs, average with op_averaging or limit other dimensions before running. ABORTING!!')
    if 'subspecs' not in indat or indat.sz[indat.dims['subspecs']]!=2:
        raise FidAException('ERROR: Input data must have two subspectra. ABORTING!!')
    def freqPhaseShiftComplexNest(in2,f,p):
        fid1=in2[:len(in2)//2]+1j*in2[len(in2)//2:]
        t=np.linspace(0,(len(fid1)-1)*infloat.dwelltime,len(fid1))
        shiftedFids=add_phase(fid1*np.exp(-1j*t*f*2*np.pi),p)
        shiftedSpecs=spec_from_fids(shiftedFids)
        y=np.r_[np.real(shiftedSpecs),np.imag(shiftedSpecs)]
        return y
    
    if ppmWeights is None:
        ppmWeights=np.ones_like(indat.ppm)
    # Do basic error check on ppmWeights
    if len(ppmWeights)!=len(indat.ppm) or np.amin(ppmWeights)<0:
        raise FidAException('ERROR: ppmWeights must be a vector of real positive weights the same size as indat.ppm')
    # Normalize weights
    print('Aligning the MEGA-PRESS edit-ON sub-spectrum to the edit-OFF sub-spectrum')
    base0=indat[:,0]
    base=np.r_[np.real(base0.specs),np.imag(base0.specs)]
    infloat=indat[:,1]
    # Matlab's nlinfit uses weights, while scipy's curve_fit accepts sigma=1/sqrt(w)
    # to weight residuals. Note that we send in infloat.fids to do the frequency
    # and phase shift and then take the specs in the function to compare with 
    # base (which is specs).
    # Ignore divide by 0. Weights of 0 have sigma of np.inf, which will give that
    # residual no weight
    with np.errstate(divide='ignore'):
        sigma_vals=1/np.r_[ppmWeights,ppmWeights]
    parsFit,pcov=curve_fit(freqPhaseShiftComplexNest,np.r_[np.real(infloat.fids),np.imag(infloat.fids)],base,initPars, sigma=sigma_vals, maxfev=5000)
    fs=parsFit[0]; phs=parsFit[1]
    if mode.lower()=='o':
        phs=phs+180
    newfid2=add_phase(infloat.fids*np.exp(-1j*infloat.t*fs*2*np.pi),phs)
    outdat=indat.copy()
    outdat.fids[:,1]=newfid2
    outdat.flags['freqcorrected']=True
    outdat.flags['phasecorrected']=True
    return outdat,phs,fs

@alter_return_args
def op_alignMPSubspecs_fd(indat,minppm,maxppm,mode='o',initPars=None,return_extra_args=None):
    # Oh, actually, I need to do the freq_range stuff here I guess. Following 
    # along with Matlab and allowing initPars and mode to be entered here, but
    # not ppmWeights. So either you pass ppmWeights to the main op_alignMPSubspecs
    # in a way that can limit the frequency range or you use this to chop indat 
    # first and send it in.
    # Checks on dimensions and number of subspecs are run in op_alignMPSubspecs
    indat_range=op_freqrange(indat, minppm, maxppm)
    outtmp,phs,fs=op_alignMPSubspecs(indat_range,mode=mode,initPars=initPars,return_extra_args=True)
    # Note that, you get the frequency and phase back from the fit but outtmp is
    # the spectrum just in the range specified by minppm and max ppm, so you need
    # to apply to the whole second subspectrum in order to get the answer. The
    # first subspectrum can stay as is
    outdat=indat.copy()
    outdat.fids[:,1]=add_phase(indat.fids*np.exp(-1j*indat.t*fs*2*np.pi),phs)
    return outdat,phs,fs

@alter_return_args
def op_alignrcvrs(indat,phasept=0,mode='w',coilcombos=None,return_extra_args=None):
    # No frequency range or initPars needed for this align function
    if 'coils' not in indat or indat.shape[indat.dims['coils']]==1:
        raise FidAException('ERROR: Receivers have already been combined! Aborting!')
        
    #To get best possible SNR, add the averages together (if it hasn't already been done):
    if 'averages' in indat:
        av=op_averaging(indat)
    else:
        av=indat.copy()
    # also, for best results, we will combine all subspectra:
    if coilcombos is None:
        if indat.flags['isFourSteps']:
            av=op_fourStepCombine(av)
        if indat.subspecs>1:
            av=op_combinesubspecs(av,'summ')
            
    if coilcombos is None:
        # Code was repeated in Matlab rather than calling function
        coilcombos=op_getcoilcombos(av,phasept=phasept,mode=mode)
    phs=coilcombos['phs']
    # sigs does not seem to be normalized in alignrcvrs the way it is in addrcvrs
    # (there is some code to normalize by max sig but this is commented out). In Matlab,
    # op_getcoilcombos already normalized by max sig and this info can't be easily
    # recovered to match the lack of normalization here in op_alignrcvrs. So I've
    # removed the normalization in op_getcoilcombos and will have to add an
    # appropriate normalization back in after any function that calls it. (I'm
    # not sure why these different functions had different normalization approaches
    # in the first place)
    sigs=coilcombos['sigs']
    
    # now expand these matrices to match the size of indat.fids for 
    # multiplication. Each coil has a different phase and amplitude
    ph=np.ones(indat.sz)
    sig=np.ones(indat.sz)
    slice1=[slice(None)]*indat.fids.ndim
    for nct in range(indat.sz[indat.dims['coils']]):
        slice1[indat.dims['coils']]=nct
        ph[tuple(slice1)]=phs[nct]
        sig[tuple(slice1)]=sigs[nct]
    
    # now apply the phases by multiplying the data by exp(-i*ph);
    fids=indat.fids*np.exp(-1j*ph*np.pi/180)
    outdat=indat.copy()
    outdat.fids=fids
    #Make the coilcombos structure:
    coilcombos={'phs':phs,'sigs':sigs}
    return outdat,coilcombos

@alter_return_args
def op_alignScans(inref, infloat, tmax=None, mode='fp', freq_range=None, initPars=None, return_extra_args=None):
    # Based on the Matlab code. The idea here is that we have parameters operating
    # on complex data, but least squares calculation for minimizing will do 
    # strange things with complex data. So one alternative is to concatenate the
    # real and imaginary parts of the data into a single vector, twice as long
    # and compare all components.
    def freqShiftComplexNest(in2,f):
        #np.r_[0:(len(in2))*infloat.dwelltime:infloat.dwelltime]
        fid1=in2[:len(in2)//2]+1j*in2[len(in2)//2:]#in2.copy()#flatten()
        t=np.linspace(0,(len(fid1)-1)*infloat_range.dwelltime,len(fid1))
        y=fid1*np.exp(-1j*t.T*f*2*np.pi)
        y=np.r_[np.real(y),np.imag(y)]
        return y
    def phaseShiftComplexNest(in2,p):
        fid1=in2[:len(in2)//2]+1j*in2[len(in2)//2:]#in2.flatten()
        y=add_phase(fid1,p)
        y=np.r_[np.real(y),np.imag(y)]
        return y
    def freqPhaseShiftComplexNest(in2,f,p):
        fid1=in2[:len(in2)//2]+1j*in2[len(in2)//2:]
        t=np.linspace(0,(len(fid1)-1)*infloat_range.dwelltime,len(fid1))
        y=add_phase(fid1*np.exp(-1j*t.T*f*2*np.pi),p)
        y=np.r_[np.real(y),np.imag(y)]
        return y
    def freqPhasePhase1ShiftComplexNest(in2,f,p0,p1):
        fid1=in2[:len(in2)//2]+1j*in2[len(in2)//2:]
        t=np.linspace(0,(len(fid1)-1)*infloat_range.dwelltime,len(fid1))
        y=add_phase1(add_phase(fid1*np.exp(-1j*t.T*f*2*np.pi),p0),xdata.ppm,p1,ppm0=xdata.center_freq_ppm,B0=xdata.Bo)
        y=np.r_[np.real(y),np.imag(y)]
        return y
    
    # Matlab runs a bunch of checks but the main thing is that the spectra have 
    # to be 1D in order for the fitting to work correctly. So that is the main 
    # check here that raises an error and other checks only run if that fails, 
    # as a way to provide extra info about what needs to be done (averaging, 
    # coil combination, etc.)
    if not (inref.ndim==1 and infloat.ndim==1):
        if 'coils' in inref or 'coils' in infloat:
            raise FidAException('ERROR: Only makes sense to do this after channels have been combined using op_addrcvrs. ABORTING!!')
        if 'averages' in inref or 'averages' in infloat:
            raise FidAException('ERROR: Only makes sense to do this after you have combined averages using op_averaging. ABORTING!!')
        if not inref.flags['isFourSteps'] or not infloat.flags['isFourSteps']:
            raise FidAException('ERROR: Only makes sense to do this after you have performed op_fourStepCombine. ABORTING!!')
        raise FidAException('ERROR: Data appear to have more than 1 dimension. If you have singleton dimensions, use op_squeeze(myfid). ABORTING!! \ninref.ndim={:d} and infloat.ndim={:d}'.format(inref.ndim, infloat.ndim))
    if initPars is None:
        initPars=[0]*len(mode)
    # Changed from Matlab to find a point based on SNR if no value input. Previous default was 0.5 s.
    if tmax is None:
        slicetmp=[0]*infloat.ndim
        slicetmp[0]=slice(None)
        sig=np.abs(infloat.fids[tuple(slicetmp)])
        slicetmp[0]=slice(int(np.ceil(0.75*infloat.sz[0])),None)
        noise=np.std(np.real(infloat.fids[tuple(slicetmp)]),axis=0)
        tmaxpt=np.flatnonzero(sig/noise>5)[-1]
        tmax=infloat.t[tmaxpt]
        print('tmax = {:.2e} ms'.format(tmax*1000))
    # curve_fit needs an array of floats to fit to, so putting the real and imaginary components
    # into one longer real-valued vector for fitting
    if freq_range is not None:
        inref_range=op_freqrange(inref, freq_range[0], freq_range[1])
        infloat_range=op_freqrange(infloat, freq_range[0], freq_range[1])
    else:
        inref_range=inref
        infloat_range=infloat
    whichpts=np.logical_and(inref_range.t>=0,inref_range.t<tmax)
    baseref=np.r_[np.real(inref_range.fids[whichpts]),np.imag(inref_range.fids[whichpts])]
    xdata=np.r_[np.real(infloat_range.fids[whichpts]),np.imag(infloat_range.fids[whichpts])]
    if mode.lower()=='f':
        parsFit,pcov=curve_fit(freqShiftComplexNest,xdata.squeeze(),baseref.squeeze(),initPars, maxfev=5000)
        frq=parsFit[0]; ph=0;
    elif mode.lower()=='p':
        parsFit,pcov=curve_fit(phaseShiftComplexNest,xdata.squeeze(),baseref.squeeze(),initPars, maxfev=5000)
        ph=parsFit[0]; frq=0;
    elif mode.lower()=='fp' or mode.lower()=='pf':
        parsFit,pcov=curve_fit(freqPhaseShiftComplexNest,xdata.squeeze(),baseref.squeeze(),initPars, maxfev=5000)
        frq=parsFit[0]; ph=parsFit[1];
    elif mode.lower()=='fpp':
        parsFit,pcov=curve_fit(freqPhasePhase1ShiftComplexNest,xdata.squeeze(),baseref.squeeze(),initPars, maxfev=5000)
        frq=parsFit[0]; ph=[parsFit[1],parsFit[2]]
    else:
        raise ValueError('ERROR: unrecognized mode. Please enter either "f", "p" or "fp"')
    # Apply frequency and phase to the full spectrum, not just selected frequency range
    out1=op_addphase(op_freqshift(infloat,frq),ph)
    return out1, ph, frq

@alter_return_args
def op_alignScans_fd(inref, infloat, fmin, fmax, tmax=None, mode='fp', initPars=None,return_extra_args=None):
    out1,ph,frq=op_alignScans(inref, infloat, tmax=tmax, mode=mode, freq_range=[fmin,fmax],initPars=initPars,return_extra_args=True)
    return out1, ph, frq

def op_arsos(indat,domain='t'):
    # The Matlab version seems to assume that there is only an averages dimension
    # and it is the second dimension. Here, the function is generalized, but
    # presumably subspecs should be combined, etc.
    if indat.averages==1:
        raise FidAException('ERROR: Averaging has already been performed. Aborting!')
        
    outdat=indat.copy()
    if domain=='t':
        # Matlab sorts the real and imaginary parts separately. What?!
        fids=np.sort(np.real(indat.fids),axis=indat.dims['averages'])+1j*np.sort(np.imag(indat.fids),axis=indat.dims['averages'])
        # Do a DC shift correction using the last 25% of the time domain data
        tminIndex=np.ceil(np.median(0.75*len(indat.t)))
        medianReal=np.median(np.real(fids),indat.dims['averages'])
        medianImag=np.median(np.imag(fids),indat.dims['averages'])
        partslice=[slice(None)]*indat.ndim
        avslice=[slice(None)]*indat.ndim
        medianslice=[slice(None)]*indat.ndim
        for nct in range(indat.averages):
            avslice[0]=slice(tminIndex,-1)
            medianslice[0]=slice(tminIndex,-1)
            avslice[indat.dims['averages']]=nct
            partslice[indat.dims['averages']]=nct
            DC0real=np.mean(np.real(fids[tuple(avslice)]),axis=0)-np.mean(medianReal[tuple(medianslice)],axis=0)
            DC0imag=np.mean(np.imag(fids[tuple(avslice)]),axis=0)-np.mean(medianImag[tuple(medianslice)],axis=0)
            fids[tuple(partslice)]=fids[tuple(partslice)]-(DC0real+1j*DC0imag)
        outdat.fids=fids
    elif domain=='f': # Again, seems odd to sort the real and imaginary parts separately
        specs=np.sort(np.real(indat.specs),axis=indat.dims['averages'])+1j*np.sort(np.imag(indat.specs),axis=indat.dims['averages'])
        # Do a DC shift correction using everything below 0 ppm and everything
        # above 10 ppm.
        ppmrange=np.nonzero((indat.ppm<0) | (indat.ppm>10))[0]
        medianReal=np.median(np.real(specs),indat.dims['averages'])
        medianImag=np.median(np.imag(specs),indat.dims['averages'])
        partslice=[slice(None)]*indat.ndim
        avslice=[slice(None)]*indat.ndim
        medianslice=[slice(None)]*indat.ndim
        for nct in range(indat.averages):
            avslice[0]=slice(ppmrange[0],ppmrange[-1])
            medianslice[0]=slice(ppmrange[0],ppmrange[-1])
            avslice[indat.dims['averages']]=nct
            partslice[indat.dims['averages']]=nct
            DC0real=np.mean(np.real(specs[tuple(avslice)]),axis=0)-np.mean(medianReal[tuple(medianslice)],axis=0)
            DC0imag=np.mean(np.imag(specs[tuple(avslice)]),axis=0)-np.mean(medianImag[tuple(medianslice)],axis=0)
            specs[tuple(partslice)]=specs[tuple(partslice)]-(DC0real+1j*DC0imag)
        outdat.specs=specs
    else:
        print("Valid domains are 't' and 'f'. Returning input unaltered")
    return outdat

@alter_return_args
def op_autophase(indat,ppmmin=4.4,ppmmax=4.8,ph_init=0,new_method=False,tmax=0.5,show_plots=False,dimNum=None,return_extra_args=None):
    """
    outdat,phShft=op_autophase(indat,ppmmin,ppmmax,ph=0,dimNum=None)
    Search for the peak located between ppmmin and ppmmax and then phase the 
    spectrum so that that peak reaches the desired phase.

    Parameters
    ----------
    indat : FID object
        input data.
    ppmmin : float
        Minimum of the ppm search range.
    ppmmax : float
        Maximum of the ppm search range.
    ph : float, optional
        Desired phase value in degrees (to add on after autophase calculation).
        The default is 0.
    dimNum : int, optional
        Which subspec dimension to use for phasing (only for use in data with
        multiple subspectra). The default is None.

    Returns
    -------
    outdat : FID object
        output following automatic phasing.
    phShft : float
        The phase shift (in degrees) that was applied.

    """
    # It is possible to generalize this function to allow multiple dimensions, 
    # but it is less clear what the most sensible autophase decision is. Subsequent
    # averages should generally have the same phase correction, for example.
    # If there are differences in phase between averages, probably you would use
    # op_alignAverages() instead. So I have allowed indat to have multiple 
    # averages but the phase is only calculated for the average spectrum (the
    # phase function is only designed for 1D data) and then applied that phase
    # to all averages.
    
    if 'averages' in indat:
        print('Multiple averages detected. Estimating phase for averaged data and applying to all averages.')
        avdat=op_averaging(indat)
    else:
        avdat=indat.copy()
    if avdat.ndim>1:
        if 'coils' in avdat:
            raise FidAException('ERROR: Cannot operate on data with multiple coils! ABORTING!!')
        if 'extras' in avdat:
            raise FidAException('ERROR: Cannot operate on data with extras dimension! ABORTING!!')
        if avdat.ndim==2 and 'subspecs' in avdat:
            if dimNum is None:
                print('WARNING: No dimNum entered for subspectra selection. Autophasing the 0th subspec')
                dimNum=0
            avdat=avdat[:,dimNum]
        else:
            raise FidAException('ERROR: Cannot do autophase on data with these dimensions. {:s}'.format(str(indat)))
    # Note that avdat should only ever be 1D. Either it was 1D to start with,
    # became 1D after averaging, or became 1D after subspec selection (possibly 
    # which may itself have come after averaging)
    if not indat.flags['zeropadded']:
        in_zp=op_zeropad(avdat,10)
    else:
        in_zp=avdat.copy()
    if new_method:
        in_zp=op_freqrange(in_zp,ppmmin,ppmmax)
        maxpt=np.flatnonzero(in_zp.t>tmax)[0]
        xdata=in_zp.t[:maxpt]
        phase_part=np.unwrap(np.angle(in_zp.fids[:maxpt]))
        phase_p=np.polyfit(xdata,phase_part,deg=1)
        phase_yfunc=np.poly1d(phase_p)
        phase_yFit=phase_yfunc(xdata)
        fshift=phase_p[0]/2/np.pi/(indat.txfreq/1e6)
        ph0=-1*phase_p[1]*180/np.pi
        if show_plots:
            f2,ax2=plt.subplots(1,1)
            ax2.plot(xdata,phase_part,'o')
            ax2.plot(xdata,phase_yFit,'-')
            ax2.set_title('ph0={:3.1f}, fshift={:3.2f} ppm'.format(ph0,fshift))
    else:
        in_zp=op_freqrange(in_zp,ppmmin,ppmmax)
        ppmindex=np.argmax(np.abs(in_zp.specs))
        ph0=-1*phase(in_zp.specs[ppmindex])*180/np.pi
    # Now apply to all averages of initial input data (or selected subspec of it)
    phShft=ph_init+ph0
    if 'subspecs' in indat:
        whichslice=[slice(None)]*indat.ndim
        whichslice[indat.dims['subspecs']]=dimNum
        outdat=indat.copy()
        outdat[tuple(whichslice)]=op_addphase(indat[tuple(whichslice)],phShft)
    else:
        outdat=op_addphase(indat,phShft)
    return outdat,phShft

def op_averaging(indat):
    """
    A function to average the free induction decays of a FID object

    Parameters
    ----------
    indat : pyFidA.FID
        Input fid/spectra to be averaged.

    Returns
    -------
    outdat : pyFidA.FID
        Averaged fid/spectra. The 'averages' dimension is now removed from the
        pyFidA.FID object.

    """
    if indat.averages<2:
        print('WARNING: No averages found. Returning input without modification!')
        outdat=indat.copy()
    else:
        outdat=indat.copy()
        # average spectrum along averages dimension (previously it was a sum and divide by shape, but why?)
        outdat.fids=np.mean(indat.fids,axis=indat.dims['averages']).squeeze()
        outdat._dimlist.remove('averages')
        outdat.flags['averaged']=True
    return outdat

def op_blockAvg(indat,N):
    # Check if there are averages
    if indat.averages<2:
        print('WARNING: No averages found. Returning input without modification!')
        outdat=indat.copy()
    else:
        # Check if the number of averages is an even multiple of N. In the original
        # Matlab, this throws an error. Here, it only gives a warning because slicing
        # will just mean the last block will have fewer averages
        if np.remainder(indat.averages,N)!=0:
            print('WARNING: The number of averages {:d} does not divide evenly into N={:d}. The last grouping will be made up of fewer averages.'.format(indat.sz[indat.dims['averages']],N))
        avslice=[slice(None)]*indat.ndim
        tmpslice=[slice(None)]*indat.ndim
        Ntrans=int(np.ceil(indat.averages/N))
        newsz=list(indat.sz)
        newsz[indat.dims['averages']]=Ntrans
        tmpfid=np.zeros(newsz,dtype=indat.fids.dtype)
        for nct in range(Ntrans):
            avslice[indat.dims['averages']]=slice(nct*N,(nct+1)*N)
            tmpslice[indat.dims['averages']]=nct
            tmpfid[tuple(tmpslice)]=np.mean(indat.fids[tuple(avslice)],axis=indat.dims['averages'])
        outdat=indat.copy()
        outdat.fids=tmpfid
    return outdat

@alter_return_args
def op_combineRcvrs(indat,inw,return_extra_args=None):
    # First find the weights using the water unsuppressed data
    weights=op_getcoilcombos(inw,2,'h')
    weights['sigs']=weights['sigs']/np.max(weights['sigs'])
    # Now apply the weights to both the water unsuppressed and water suppressed
    #data, but don't combine the averages:
    out_presum=op_alignrcvrs(indat,2,'h',weights,return_extra_args=False)
    outw_presum=op_alignrcvrs(inw,2,'h',weights,return_extra_args=False)
    # Now apply the weights and combine the averages:
    outdat=op_addrcvrs(indat,2,'h',weights,return_extra_args=False)
    outw=op_addrcvrs(inw,2,'h',weights,return_extra_args=False)
    return outdat,outw,out_presum,outw_presum

def op_combinesubspecs(indat,mode):
    """
    Combine the subspectra in an acquisition either by addition or subtraction

    Parameters
    ----------
    indat : FID object
        Input spectrum.
    mode : str, 'diff' or 'summ'
        How to combine the data:
            -'diff' adds the subspectra together. This is counter-intuitive but
            the reason is that many "difference editing" sequences use phase
            cycling of the readout ADC to achieve "subtraction by addition".
            -'summ' performs a subtraction of the subspectra

    Returns
    -------
    outdat : FID object
        Output spectrum following combination of subspectra.

    """
    outdat=indat.copy()
    if indat.subspecs==1:
        raise FidAException('ERROR: Subspectra have already been combined. Aborting!')
    if indat.flags['isFourSteps']:
        raise FidAException('ERROR: Data with four steps must first be converted using op_fourStepCombine. Aborting!')
    if mode=='diff':
        # add the spectrum along the subspecs dimension
        newfid=np.sum(indat.fids,axis=indat.dims['subspecs'])/indat.subspecs
    elif mode=='summ':
        # Note that the assumption from Matlab is that there are only two subspecs
        # but I did generalize this code to assume that even transients should 
        # be subtracted from odd
        sumslice1=[slice(None)]*indat.ndim
        sumslice1[indat.dims['subspecs']]=slice(1,None,2)
        sumslice0=[slice(None)]*indat.ndim
        sumslice0[indat.dims['subspecs']]=slice(0,None,2)
        newfid=np.sum(indat.fids[tuple(sumslice1)]-indat.fids[tuple(sumslice0)],axis=indat.dims['subspecs'])/indat.subspecs
    outdat.fids=newfid
    outdat._dimlist.remove('subspecs')
    # setting number of subspecs now done via property
    outdat.flags['subtracted']=True
    return outdat

def op_complexConj(indat):
    """
    Take the complex conjugate of the data

    Parameters
    ----------
    indat : FID object
        Input data.

    Returns
    -------
    outdat : FID object
        Output following conjugation.

    """
    conjfid=np.conj(indat.fids)
    outdat=indat.copy()
    outdat.fids=conjfid
    return outdat

def op_dccorr(indat,mode,var1=None):
    if mode=='p':
        if var1 is None: # No limits provided
            # find the ppm of the maximum peak magnitude within a given range
            dcOffset1=np.mean(indat.specs[(indat.ppm>max(indat.ppm)-0.5)*(indat.ppm<max(indat.ppm))])
            dcOffset2=np.mean(indat.specs[(indat.ppm>min(indat.ppm))*(indat.ppm<min(indat.ppm)+0.5)])
            dcOffset=np.mean([dcOffset1,dcOffset2])
        else:
            dcOffset=np.mean(indat.specs[(indat.ppm>var1[0])*(indat.ppm<var1[1])])
    elif mode=='v':
        dcOffset=var1
    outdat=indat.copy()
    outdat.specs=indat.specs-dcOffset
    return outdat

def op_downsamp(indat,dsFactor):
    # Note that the effect of this function is basically to get just a portion of
    # the spectrum back. But also some of the lower frequencies wrap back around
    # to the higher frequencies (like aliasing)? (I checked and this happens in 
    # Matlab, so it's not the implementation here. It's the expected output from 
    # Matlab. So I'm not sure what the point of this function is. Seems like it 
    # would be easier to just use op_freqrange or whatever. 
    if indat.ndim>=2:
        # In Matlab this gives an error, but you could see cases to allow it to
        # work at least for averages
        print('WARNING: Expected FID object to have combined averages, subspecs and coils. Downsampling anyway.')
    
    # Matlab uses a nearest neighbour interpolation. The description says 
    # "by default" but there is no input argument that changes this so
    # I have only implemented nearest neighbour here. (Python's resample options
    # differ from the resample function in Matlab so any other interpolation would
    # have to be implemented carefully if the intention is to match Matlab's output)
    # Note also that the resample function called in Matlab uses integers so
    # dsFactor must be an integer. In that case, it seems silly to use a nearest
    # neighbour interpolator at all because you could just use the slicing notation
    # on both indat.t and indat.fids, but here we are.
    interpfunc=interp1d(indat.t,indat.fids,kind='nearest',axis=0)
    tnew=indat.t[::dsFactor]
    newfid=interpfunc(tnew)
    outdat=indat.copy()
    outdat.fids=newfid
    outdat.spectralwidth=indat.spectralwidth/dsFactor
    # Everything else (dwell time, etc) should be calculated automatically
    outdat.flags['downsampled']=True
    return outdat
    
@alter_return_args
def op_ecc_klose(indat,inw,return_extra_args=None):
    # Not sure if this was complete? It seems so much more straightforward than the op_ecc
    if 'coils' not in inw or inw.averages!=1 or inw.subspecs!=1:
        raise FidAException('ERROR: Must combine receivers, averages and subspecs prior to running ecc!! Aborting!!')
    # save the phase 
    inph=phase(inw.fids)
    # Now subtract the line from the spline to get the eddy current related phase offset
    ecphase_rep=np.transpose(np.tile(inph,list(indat.sz[1:][::-1])+[1]))
    # Now apply the eddy current correction to both the water suppressed and the
    # water unsuppressed data
    outdat=indat.copy()
    outdat.fids=outdat.fids*np.exp(-1j*ecphase_rep)
    outdat=op_addphase(outdat,180*ecphase_rep[0]/np.pi)
    outw=inw.copy()
    outw.fids=outw.fids*np.exp(-1j*inph)
    outw=op_addphase(outw,180*inph[0]/np.pi)
    return outdat,outw

@alter_return_args
def op_ecc(indat,inw,return_extra_args=None):
    # I think that this isn't a million miles from what is done in Matlab, but
    # I would need some data that needs eddy current correction to check the
    # results against one another. Part of the issue in Matlab is that I don't
    # think that the Bruker data is read in correctly (the cutoff for the "junk"
    # at the start of the Bruker file is one point earlier in Matlab and also the
    # real and imaginary parts of the data are swapped around and have a negative
    # sign), so you really should do leftshift and phase correction beforehand 
    # in that case. There is a DC offset in the plot of inph. The basic shapes of
    # the graph are similar other than that, but not exactly the same.
    if inw.dims['coils']!=-1 or inw.dims['averages']!=-1 or inw.dims['subspecs']!=-1:
        raise FidAException('ERROR: Must combine receivers, averages and subspecs prior to running ecc!! Aborting!!')
    # save the phase 
    inph=phase(inw.fids)
    f1,ax1=plt.subplots(1,1)
    ax1.plot(inw.t,inph)
    tmin=float(input('input min t value: '))
    tmax=float(input('input max t value: '))
    # Now fit a straight line to the linear part of the phase function
    p=np.polyfit(inw.t[(inw.t>tmin)*(inw.t<tmax)],inph[(inw.t>tmin)*(inw.t<tmax)],1)
    # Now fit a spline to approximate a smooth version of the phase function. I
    # am using scipy.interpolate's CubicSpline, which may work differently than
    # the method used in Matlab
    npieces=150
    piece_size=len(inw.t)//npieces
    cs=CubicSpline(inw.t[::piece_size],inph[::piece_size])
    plt.figure()
    plt.plot(inw.t,p[1]+p[0]*inw.t)
    plt.plot(inw.t,cs(inw.t))
    # Now subtract the line from the spline to get the eddy current related phase offset
    ecphase=cs(inw.t)-(p[1]+p[0]*inw.t)
    # Now subtract the line from the spline to get the eddy current related phase offset
    ecphase_rep=np.transpose(np.tile(inph,list(indat.sz[1:][::-1])+[1]))
    plt.figure()
    plt.plot(inw.t,ecphase)
    # Now apply the eddy current correction to both the water suppressed and the
    # water unsuppressed data
    outdat=indat.copy()
    outdat.fids=outdat.fids*np.exp(-1j*ecphase_rep)
    outdat=op_addphase(outdat,180*ecphase_rep[0]/np.pi)
    outw=inw.copy()
    outw.fids=outw.fids*np.exp(-1j*ecphase)
    outw=op_addphase(outw,180*ecphase[0]/np.pi)
    return outdat,outw

def op_fddccorr(indat,npts):
    # Find the frequency domain vertical offset to correct
    tails=np.concatenate((indat.specs[:npts,...],indat.specs[-npts:,...]),axis=0)
    dcOffset=np.mean(tails,axis=0)
    # Subtract the offset (real and imaginary) from specs
    outdat=indat.copy()
    outdat.specs=indat.specs-dcOffset
    return outdat

@alter_return_args
def op_filter(indat,lb,return_extra_args=None):
    """
    Perform line broadening by multiplying the time domain signal by an 
    exponential decay function.  

    Parameters
    ----------
    indat : FID class
        input data.
    lb : float
        Line broadening factor in Hz.

    Returns
    -------
    outdat : FID class
        Output following alignment of averages.
    lor : Numpy array
        Exponential time domain filter that was applied
    """
    if lb==0:
        outdat=indat.copy()
        lor=None
    else:
        outdat=indat.copy()
        if indat.flags['filtered']:
            print('WARNING:  Line Broadening has already been performed! Performing again.')
        t2=1/(np.pi*lb)
        lor=np.exp(-1*indat.t/t2)
        # Matlab makes a bunch of vectors of ones to get the array sizes to match
        # with ngrid. I'm not sure why repmat wasn't used. However, in Python,
        # since 't' is always the first dimension, we can just use broadcasting
        # and the arrays will be expanded automatically (but the dimension being
        # broadcast needs to be last, so we have to transpose, multiply, then
        # transpose back).
        newfid=(indat.fids.T*lor).T
        outdat.fids=newfid
        outdat.flags['filtered']=True
    return outdat,lor

def op_fourStepCombine(indat,mode=0):
    if not indat.flags['isFourSteps']:
        raise FidAException('ERROR: requires a dataset with 4 subspecs as input!  Aborting!')
    if indat.subspecs!=4:
        raise FidAException('ERROR: subspecs dimension must have length 4!!  Aborting!')
        
    # I've set this up slightly differently than Matlab. In Matlab, subspecs must
    # be the last dimension. Here it can be any dimension
    inidx=[slice(None)]*indat.ndim
    outidx=[slice(None)]*indat.ndim
    newsz=indat.sz
    newsz[indat.dims['subspecs']]=2
    newfid=np.zeros(newsz)
    for eachsub in range(2):
        if mode==0 or mode==1:
            inidx[indat.dims['subspecs']]=slice(eachsub*2,eachsub*2+2)
            outidx[indat.dims['subspecs']]=eachsub
            if mode==0:
                newfid[tuple(outidx)]=np.sum(indat.fids[tuple(inidx)],axis=indat.dims['subspecs'])
            else:
                newfid[tuple(outidx)]=np.diff(indat.fids[tuple(inidx)],axis=indat.dims['subspecs'])
        elif mode==2 or mode==3:
            inidx[indat.dims['subspecs']]=slice(eachsub,4,2)
            outidx[indat.dims['subspecs']]=eachsub
            if mode==2:
                newfid[tuple(outidx)]=np.sum(indat.fids[tuple(inidx)],axis=indat.dims['subspecs'])
            else:
                newfid[tuple(outidx)]=np.diff(indat.fids[tuple(inidx)],axis=indat.dims['subspecs'])
        else:
            raise ValueError('ERROR: mode not recognized. Value must be 0, 1, 2 or 3.')
    outdat=indat.copy()
    outdat.fids=newfid/2  #Divide by 2 so that this is an averaging operation
    outdat.flags['isFourSteps']=False
    return outdat

@alter_return_args
def op_freqAlignAverages(indat,tmax=None,med='a',ref=None,freq_range=None,initPars=None,return_extra_args=None):
    # Note that the Matlab version of the function rewrites basically all of the
    # code from op_alignAverages. I have instead turned this into a wrapper 
    # function that calls op_alignAverages. However, there are differences in
    # the arguments of the two functions. Matlab's op_freqAlignAverages has an 
    # argument avg in place of med, which has limited options for the reference
    # spectrum. Thus, I have replaced avg with med here to give more options and
    # used the argument name med to try to make this clear, so this also changes 
    # what 'y' and 'n' mean in Matlab's op_freqAlignAverages vs Python's. You 
    # can read the details in that function's docstring but the main thing to 
    # note is that 'y' in this Python function uses the median as the reference
    # spectrum (not the average as in Matlab's version of this function) and 
    # 'n' will fit to the 'best' individual average (not to the first average 
    # as is done in Matlab). You can fit to the average using by med='a' or to 
    # the first average by med='f'
    # Also, in Matlab op_freqAlignAverages and op_alignAverages calculate the 
    # frequency shift as exp(1i*t*f*2*pi) in the fit whereas op_alignScans
    # uses exp(-1i*t'*f*2*pi). Presumably the aligned fids should come out the
    # same, but the frequency shift will have a different sign. I call op_alignScans
    # from op_alignAverages and that uses np.exp(-1j*t.T*f*2*np.pi) to match
    # Matlab. So the frequency vector from op_alignAverages from Python will be
    # the negative of what it is in Matlab
    if initPars is not None and np.isscalar(initPars):
        initPars=[initPars]
    outdat,phs,fs=op_alignAverages(indat,tmax=tmax,med=med,ref=ref,freq_range=freq_range,mode='f',initPars=initPars,return_extra_args=True)
    return outdat,fs

@alter_return_args
def op_freqAlignAverages_fd(indat, minppm, maxppm, tmax=None, med='a', ref=None, initPars=None,return_extra_args=None):
    out1,frq=op_freqAlignAverages(indat, tmax=tmax, med=med, ref=ref, freq_range=[minppm, maxppm], initPars=initPars,return_extra_args=True)
    return out1,frq

def op_freqshift(indat,fshift):
    outdat=indat.copy()
    # broadcasting in the case of multiple dimensions
    newfid=(indat.fids.T*np.exp(-1j*indat.t*fshift*2*np.pi)).T
    outdat.fids=newfid
    return outdat

def op_freqshiftSubspec(indat,fshift):
    # Matlab requires coils combination and averaging to be done but I don't think
    # that's necessary. However, there should be 2 subspecs
    if 'subspecs' not in indat:
        raise FidAException('ERROR:  Can not operate on data with no subspecs!  ABORTING!!')
    if indat.subspecs!=2:
        raise FidAException('ERROR:  Input spectrum must have two subspecs!  ABORTING!!')
    outdat=indat.copy()
    subspec_slice=[slice(None)]*indat.ndim
    subspec_slice[indat.dims['subspecs']]=1
    second_subspec=indat[tuple(subspec_slice)]
    # broadcasting in the case of multiple dimensions
    newfid=(second_subspec.fids.T*np.exp(-1j*indat.t*fshift*2*np.pi)).T
    outdat.fids[tuple(subspec_slice)]=newfid
    return outdat

def op_getcoilcombos(indat,phasept=0,mode='w'):
    """
    Finds the relative coil phases and amplitudes for coil data in indat. The
    result can be fed to op_addrcvrs for coil combination (although data generally
    need to be rephased after)

    Parameters
    ----------
    indat : FID object
        Input data with multiple receiver info. Note that Matlab fidA allows 
        you to enter a filename but assumes twix format. This differs from other
        processing functions, so I'm removing that option in Python.
    phasept : float, optional
        Point of fid to use for phase estimation and amplitude when mode='w'. 
        The default is 0.
    mode : char, optional
        Method for estiamting the coil weights and phases if not provided in 
        coilcombos. Can be:
            'w' - performs amplitude weighting of channels based on the max 
                signal of each coil channel
            'h' - performs amplitude weighting of channles based on the max signal 
                of each coil channel divided by the square of the noise in each 
                coil channel (as described by Hall et al. Neuroimage 2014).
            The default is 'w'.

    Returns
    -------
    coilcombos : dict with keys 'phs' and 'sigs'
        The vectors of the coil phases (in degrees) used for alignment and the 
        coil weights.

    """
    # I have not included the option for indat to be a filename, which exists
    # in Matlab because it only loads twix files. Seems to make more sense to
    # keep most things in the processing module have FID objects as input and
    # use the io toolbox to load files into that format. Separately
    if 'coils' not in indat or indat.sz[indat.dims['coils']]==1:
        print('WARNING: Only one receiver channel found! Coil phase will be 0.0 and coil amplitude will be 1.0.')
        coilcombos={'phs':0, 'sigs':1}
    else:
        # Find the relative phases between the channels and populate the ph matrix
        # The use of the slice object here allows this to be done for the 'coils'
        # dimension regardless of the other dimensions in indat.dims
        whichpts=[0]*indat.ndim
        whichpts[indat.dims['coils']]=slice(None)
        whichpts[indat.dims['t']]=phasept
        # Updated from np.angle to phase function that unwraps the phase on 2025-11-07
        phs=phase(indat.fids[tuple(whichpts)])*180/np.pi
        if mode=='w':
            sigs=np.abs(indat.fids[tuple(whichpts)])
        elif mode=='h':
            whichpts[indat.dims['t']]=slice(None)
            S=np.max(np.abs(indat.fids[tuple(whichpts)]),axis=0)
            # -100 is copied from the Matlab code, assuming this is in the noise for the fid.
            whichpts[indat.dims['t']]=slice(-100,None)
            N=np.std(indat.fids[tuple(whichpts)],axis=0)
            sigs=S/(N**2)
        else:
            raise ValueError("ERROR: mode must have value 'w' or 'h'.")
        # In Matlab, op_getcoilcombos normalizes so that the max signal amplitude
        # is 1, whereas op_addrcvrs normalizes so the sum of the amplitudes is
        # 1 and op_alignrcvrs doesn't normalize at all (commented out). I am
        # therefore removing the normalization within the function (ie. the 
        # Python result for this function will differ from Matlab) and any
        # normalization will need to be added after the function call. It doesn't
        # really make sense to me that different normalization methods were being
        # used for very similar function calls anyway.
        #sigs=sigs/np.max(sigs)
        coilcombos={'phs':phs,'sigs':sigs}
    return coilcombos

def op_getcoilcombos_specReg(indat,tmin=0,tmax=0.2,pt=0):
    def op_phaseShiftRealNest(in2,p):
        # Reshape vector for processing, then convert back to 1D for comparison with base
        fid1=in2[:len(in2)//2]+1j*in2[len(in2)//2:]
        y=add_phase(fid1,p)
        y=np.r_[np.real(y),np.imag(y)]
        return y
    # I have not included the option for indat to be a filename, which exists
    # in Matlab because it only loads twix files. Seems to make more sense to
    # keep most things in the processing module have FID objects as input and
    # use the io toolbox to load files into that format. Separately
    if 'coils' not in indat or indat.sz[indat.dims['coils']]==1:
        print('WARNING: Only one receiver channel found! Coil phase will be 0.0 and coil amplitude will be 1.0.')
        coilcombos={'phs':0, 'sigs':1}
    else:
        ncoils=indat.sz[indat.dims['coils']]
        ph=np.zeros([ncoils,])
        whichpts=[0]*indat.ndim
        whichpts[indat.dims['coils']]=slice(None)
        whichpts[indat.dims['t']]=pt
        sig=np.abs(indat.fids[tuple(whichpts)])
        bestSNRidx=np.argmax(sig)
        phGuess=0
        print('aligning all coils to the first coil')
        whichpts[indat.dims['coils']]=bestSNRidx
        whichpts[indat.dims['t']]=slice(np.nonzero(indat.t>tmin)[0][0],np.nonzero(indat.t>tmax)[0][0]-1)
        # I think you can't follow Matlab here and take the phase because we need
        # to input x as something real-valued
        base=np.r_[np.real(indat.fids[tuple(whichpts)]),np.imag(indat.fids[tuple(whichpts)])]#phase(indat.fids(tuple(whichpts)))
        for nct in range(ncoils):
            if nct!=bestSNRidx:
                whichpts[indat.dims['coils']]=nct
                xdata=indat.fids[tuple(whichpts)]
                parsFit,pcov=curve_fit(op_phaseShiftRealNest,np.r_[np.real(xdata),np.imag(xdata)]/sig[nct],base,phGuess, maxfev=5000)
                # Note that the phase value that you get back is already in degrees. No need to convert.
                ph[nct]=-1*parsFit[0]
            else:
                ph[nct]=0
        whichpts['t']=pt
        whichpts[indat.dims['coils']]=bestSNRidx
        coilcombos={'phs':ph+phase(indat.fids[tuple(whichpts)])*180/np.pi,'sigs':sig/np.amax(sig)}
    return coilcombos

def op_HSVDfit(indat,ppmlim=[0.2,4.2],Kinit=20,M=None,plot_bool=True):
    if indat.ndim==1:
        fid=indat.fids
    elif indat.ndim==2:
        if 1 in indat.sz:
            fid=indat.fids.squeeze()
        else:
            raise FidAException('ERROR: HSVDfit requires 1D fid')
    else:
        raise FidAException('ERROR: HSVDfit requires 1D fid')
    # Note that this assumes 1D data for fitting. Not a generalizable function.
    if M is None:
        M=int(np.floor(indat.sz[0]*0.75))
    N=indat.sz[0]
    dt=indat.dwelltime
    t=indat.t[:N]
    H=hankel(fid[:M],fid[M-1:])
    [U,s,V]=svd(H,full_matrices=True)
    amp=[0]; count=0
    #while np.sum([av==0 for av in amp])>=1 or np.sum(np.isnan(amp))>=1:
    while np.any(np.isnan(amp)) or np.any([av==0 for av in amp]):
        K=Kinit-count
        # truncate the data
        Uk=U[:,:K]
        # Get the eigenvalues of the transform matrix
        Utk=Uk[1:,:]
        Ubk=Uk[:-1,:]
        Eh, *_ = np.linalg.lstsq(Utk,Ubk,rcond=None)
        Eeigs,Eevecs=np.linalg.eig(Eh.T)
        # convert eigenvalues to poles to get freq and damping factor
        #w=-1*np.angle(Eeigs)/dt
        w=1*np.angle(Eeigs)/dt
        freqs=w/2/np.pi
        alpha=(np.abs(Eeigs)-1)/dt
        # Make a model guess using only damping factor and freq
        fid_temp=np.exp((-1*alpha-(1j*w))*t[:,np.newaxis]) # mostly 0 except for 1 very small row
        # do a least square fit of your model guess to the data
        phamp,*_=np.linalg.lstsq(fid_temp,fid,rcond=None)
        # Convert the eigenvalues to phase and amplitude
        #ph=-1*np.angle(phamp)
        ph=1*np.angle(phamp)
        amp=np.abs(phamp)
        amp=np.where(amp/np.amax(amp)>1e-12,amp,0)
        ph=np.where(amp/np.amax(amp)>1e-12,ph,0)
        count=count+1
        if K<2:
            print('##### Could not find a suitable number of components #######')
            break
    # Use amplitude, phase, damping factor and frequency to model data
    # Model the water signal
    ppms=-1*freqs/(indat.txfreq/1e6)+indat.center_freq_ppm
    ppminrange=(ppms>ppmlim[0])*(ppms<ppmlim[1])
    fid_model=np.matmul((amp[ppminrange]*np.exp(1j*ph[ppminrange])),(np.exp((-1*alpha[ppminrange]-(1j*w[ppminrange]))*t[:,np.newaxis])).T)
    # Remove water signal from the data
    fid_resid=fid-fid_model
    model=indat.copy()
    model.fids=fid_model
    resids=indat.copy()
    resids.fids=fid_resid
    if plot_bool:
        f1,ax1=plt.subplots(1,2)
        ax1[0].plot(indat.ppm,indat.specs,label='Data')
        ax1[0].plot(indat.ppm,(indat.specs-resids.specs),label='Water')
        ax1[0].set_xlabel('Freq')
        ax1[0].set_xlim([ax1[0].get_xlim()[1],ax1[0].get_xlim()[0]])
        ax1[0].set_title('Original Data Spectrum + Water Estimate')
        ax1[1].plot(indat.ppm,resids.specs)
        ax1[1].set_xlabel('Freq')
        ax1[1].set_xlim([ax1[1].get_xlim()[1],ax1[1].get_xlim()[0]])
        ax1[1].set_title('Water Suppressed Spectrum')
    if not np.sum(amp[ppminrange]):
        print('######## The fit did not work. Try reducing the number of components K. ######')
    return [model, resids, K, ppms, amp, alpha, ph]

def op_leftshift(indat,ls):
    outdat=indat.copy()
    cont='y'
    if indat.flags['leftshifted']:
        cont=str(input('WARNING: Left shifting has already been performed! Continue anyway? (y or n): '))
    if cont.lower()=='y':
        fids=indat.fids
        fids_trunc=fids[ls:,...]
        # It is automatically updated to the new length. And spectralwidth
        # is the same, with the ppm endpoints being adjusted to account for
        # the new length with the same central frequency
        outdat.fids=fids_trunc
        outdat.flags['leftshifted']=True
    else:
        print('Aborting without additional left shifting.')
    return outdat

@alter_return_args
def op_matchLW(indat, inref, ppmmin, ppmmax, tmax=0.5, mode='l', initPars=None, return_extra_args=None):
    # Based on the Matlab code. The idea here is that we have parameters operating
    # on complex data, but least squares calculation for minimizing will do 
    # strange things with complex data. So one alternative is to concatenate the
    # real and imaginary parts of the data into a single vector, twice as long
    # and compare all components.
    def op_lorFilter(in2,lb):
        fid1=in2[:len(in2)//2]+1j*in2[len(in2)//2:]
        t=np.linspace(0,(len(fid1)-1)*infloat_range.dwelltime,len(fid1))
        y=fid1*np.exp(-t/(np.pi*lb))
        y=np.r_[np.real(y),np.imag(y)]
        return y
    def op_gaussFilter(in2,lb):
        fid1=in2[:len(in2)//2]+1j*in2[len(in2)//2:]
        t=np.linspace(0,(len(fid1)-1)*infloat_range.dwelltime,len(fid1))
        thalf=np.log(0.5)/(np.pi*0.5*lb)
        sigma=np.sqrt((thalf**2)/(-2*np.log(0.5)))
        y=fid1*np.exp(-1*(t**2)/(2*sigma**2))
        y=np.r_[np.real(y),np.imag(y)]
        return y
    def op_lorGaussFilter(in2,lb,k):
        fid1=in2[:len(in2)//2]+1j*in2[len(in2)//2:]
        t=np.linspace(0,(len(fid1)-1)*infloat_range.dwelltime,len(fid1))
        lorFilt=np.exp(-t/(np.pi*lb))
        thalf=np.log(0.5)/(np.pi*0.5*lb)
        sigma=np.sqrt((thalf**2)/(-2*np.log(0.5)))
        gaussFilt=np.exp(-1*(t**2)/(2*sigma**2))
        y=fid1*((k*lorFilt)+((1-k)*gaussFilt))
        y=np.r_[np.real(y),np.imag(y)]
        return y
    
    if not (inref.ndim==1 and indat.ndim==1):
        raise FidAException('ERROR: Data appear to have more than 1 dimension. If you have singleton dimensions, use op_squeeze(myfid). ABORTING!! \ninref.ndim={:d} and indat.ndim={:d}'.format(inref.ndim, indat.ndim))
    if initPars is None:
        initPars=[0.5]*len(mode)
    if tmax>inref.t[-1]:
        tmax=inref.t[-1]
    # curve_fit needs an array of floats to fit to, so putting the real and imaginary components
    # into one longer real-valued vector for fitting
    inref_range=op_freqrange(inref, ppmmin, ppmmax)
    infloat_range=op_freqrange(indat, ppmmin, ppmmax)
    baseref=np.r_[np.real(inref_range.fids[np.logical_and(inref_range.t>=0,inref_range.t<tmax)]),np.imag(inref_range.fids[np.logical_and(inref_range.t>=0,inref_range.t<tmax)])]
    xdata=np.r_[np.real(infloat_range.fids[np.logical_and(infloat_range.t>=0,infloat_range.t<tmax)]),np.imag(infloat_range.fids[np.logical_and(infloat_range.t>=0,infloat_range.t<tmax)])]
    if mode=='l':
        # But here send in complex-valued indat.fids for the same t range because
        # the real and imaginary parts will be separated after frq and ph adjustment
        parsFit,pcov=curve_fit(op_lorFilter,xdata.squeeze(),baseref.squeeze(),initPars, maxfev=5000)
        lb=parsFit[0]; k=1
    elif mode=='g':
        parsFit,pcov=curve_fit(op_gaussFilter,xdata.squeeze(),baseref.squeeze(),initPars, maxfev=5000)
        lb=parsFit[0]; k=0
    elif mode=='lg' or mode=='gl':
        parsFit,pcov=curve_fit(op_lorGaussFilter,xdata.squeeze(),baseref.squeeze(),initPars, maxfev=5000)
        lb=parsFit[0]; k=parsFit[1];
    else:
        raise ValueError('ERROR: unrecognized mode. Please enter either "l", "g" or "lg"')
    lorFilt=np.exp(-indat.t/(np.pi*lb))
    thalf=np.log(0.5)/(np.pi*0.5*lb)
    sigma=np.sqrt((thalf**2)/(-2*np.log(0.5)))
    gaussFilt=np.exp(-1*(indat.t**2)/(2*sigma**2))
    newfid=indat.fids*((k*lorFilt)+((1-k)*gaussFilt))
    out1=indat.copy()
    out1.fids=newfid
    # Should you set out.flags['filtered']=True? Not done in Matlab
    return out1, lb, k

def op_median(indat):
    if indat.averages<2:
        print('ERROR:  Averaging has already been performed!  Aborting!')
        outdat=indat.copy()
    else:
        outdat=indat.copy()
        # add the spectrum along the averages dimension
        outdat.fids=np.median(indat.fids,axis=indat.dims['averages']).squeeze()
        # change dims variable and update flags
        outdat._dimlist.remove('averages')
        outdat.flags['averaged']=True
    return outdat

def op_movef0(indat,newf0):
    newfid=(indat.fids.T*np.exp(-1j*indat.t*newf0*2*np.pi)).T
    outdat=indat.copy()
    outdat.fids=newfid
    # Just need to adjust center frequency and ppm should adjust automatically
    outdat.center_freq_ppm=newf0/(indat.txfreq/1e6)
    return outdat

@alter_return_args
def op_phaseAlignAverages(indat,npts=None,med='a',ref=None,weighting='n',freq_range=None,return_extra_args=None):
    # Note that phase alignment across averages can be accomplished by calling
    # op_alignAverages with mode='p'. However, op_phaseAlignAverages works 
    # differently in Matlab and I have replicated that behaviour here for this
    # alignment function. op_phaseAlignAverages calculates the difference in 
    # phase between indat and ref at each time point, weighting those differences, 
    # and then takes the mean across all time points for a particular average and
    # subspec. This is therefore not a fitting function. It is a calculated
    # (weighted) average phase difference. 
    # For some consistency with the other align functions. This also changes 
    # what 'y' and 'n' mean in Matlab's op_phaseAlignAverages vs Python's. You 
    # can read the details in the op_alignAverages docstring but the main thing to 
    # note is that 'y' in this Python function uses the median as the reference
    # spectrum (not the average as in Matlab's version of this function) and 
    # 'n' will fit to the 'best' individual average (not to the first average 
    # as is done in Matlab). You can fit to the average using by med='a' or to 
    # the first average by med='f'. I have named the argument med instead of avg
    # to try to make this difference from Matlab's arguments clearer.

    outdat=indat.copy()
    phs=0
    if 'coils' in indat and indat.sz[indat.dims['coils']]>1:
        raise FidAException('ERROR: I think it only makes sense to do this after you have combined the channels using op_addrcvrs. ABORTING!!')
    elif med.lower()=='r' and ref is None:
        raise FidAException("ERROR:  If using the 'r' option for input variable 'med', then an argument for 'ref' must be provided")
    elif indat.averages<2:
        print('WARNING: No averages found. Returning input without modification')
    else:
        # It seems that the weighting function is based on the average even if 
        # the reference spectrum isn't the average. It shouldn't matter too much
        # as long as there isn't some outlier skewing the average and thus the 
        # weightings
        avg_fid=op_averaging(indat)
        if weighting.lower()=='y':
            wgt_func=np.abs(avg_fid.fids)**2
        else:
            wgt_func=np.ones_like(np.real(avg_fid))
        wgt_func=np.moveaxis(np.tile(wgt_func,[indat.averages]+[1]*wgt_func.ndim),0,indat.dims['averages'])
        B=indat.subspecs
        # Note that this function is not as generalizable as some others. It 
        # assumes that you only have averages and/or subspecs, with the averages
        # dimension before subspecs. Since this is not a fitting function like
        # some other align functions, it might be possible to generalize further.
        # For SVS, multi-receiver data are ruled out by the addedcrvrs check but
        # spatial data from MRS are not.
        phs=np.zeros(indat.sz[1:])
        ph_slices=[slice(None)]*phs.ndim
        for mct in range(B):
            # Create a slice to select the mct'th subspec
            subspec_pts=[slice(None)]*indat.ndim
            subspec_pts[0]=slice(0,npts) # Note that this means that we will already have limited to the first npts in time domain for tmpfid
            if 'subspecs' in indat:
                subspec_pts[indat.dims['subspecs']]=mct
            # after this, tmpfid should no longer have a subspecs dimension, even if indat did (removed during slicing)
            tmpfid=indat[tuple(subspec_pts)]
            wgt_func2=wgt_func[tuple(subspec_pts)]
            if med.lower()=='y':
                ref2=op_median(tmpfid)
                indmin=-1
            elif med.lower()=='a':
                ref2=op_averaging(tmpfid)
                indmin=-1
            elif med.lower()=='n':
                # Note that op_rmbadaverages is only designed to work for FID
                # objects with averages and (optionally) subspecs. Will need to
                # alter if there are spatial dimensions of interest.
                ref2,metric,badavgs=op_rmbadaverages(tmpfid,return_extra_args=True)
                try:
                    indmin=np.argmin(metric[:,mct])
                except IndexError:
                    indmin=np.argmin(metric)
            elif med.lower()=='f':
                # Not sure why this option isn't available in Matlab but makes
                # sense to add it
                first_pts=[slice(None)]*tmpfid.ndim
                first_pts[tmpfid.dims['averages']]=0
                ref2=tmpfid[tuple(first_pts)]
                indmin=0
            elif med.lower()=='r':
                # The assumption here is that, if indat has subspecs, then ref 
                # also has subspecs and each subspec is meant to be aligned to its
                # own subspec reference. This makes sense because subspectra will
                # be quite different from one another
                try:
                    ref2=ref[tuple(subspec_pts)]
                except IndexError:
                    raise IndexError('ERROR: ref must have size that relates to indat, except for averages or subspecs')
                indmin=-1 # Note that I never use any of the indmins in this case. They're used in op_alignAverages to avoid fitting 2 vectors that are the same, but here the simple subtraction should give you a phase difference of zero regardless
            else:
                raise ValueError("ERROR: Invalid value for 'med'. Allowed values are 'y', 'a', 'n', 'f' and 'r'.")
            # npts was already selected in tmpfid above, but we haven't done op_freqrange, Could do at beginning?
            # Now we can expand ref2 to have size tmpfid and do our subtraction (still only for a particular subspec)
            ref_expanded=np.moveaxis(np.tile(ref2,[indat.averages]+[1]*ref2.ndim),0,tmpfid.dims['averages'])
            if freq_range is not None:
                inref_range=op_freqrange(ref_expanded, freq_range[0], freq_range[1])
                infloat_range=op_freqrange(tmpfid, freq_range[0], freq_range[1])
                indvals=np.logical_and(np.greater(indat.ppm,freq_range[0]),np.less(indat.ppm,freq_range[1]))
                wgt_func3=wgt_func2[indvals,...]
            else:
                inref_range=ref_expanded
                infloat_range=tmpfid
                wgt_func3=wgt_func2
            # Initially, I had ref_expanded and tmpfid here, so I need to run some
            # test code to see whether I implemented this correctly (it has more
            # options than Matlab so that it more closely mimics the other
            # op_alignAverages type functions).
            phase_diffs=(phase(inref_range)-phase(infloat_range))*180/np.pi
            phase_mct=np.mean(phase_diffs*wgt_func3,axis=0)/np.mean(wgt_func3,axis=0)
            # There are adjustments here for positive or negative phase, I think to get things into
            # the 0 to 360 range. Not sure how needed those are. Also, they subtract off the phase
            # of the first average but not sure how important this is. Also not sure that it is
            # done correctly because firstPhase has index m but then subtracts during mct, meaning
            # the subtraction happens multiple times for the first subspec?? Leaving it out for now.
            if 'subspecs' in indat:
                ph_slices[indat.dims['subspecs']-1]=mct
            phs[tuple(ph_slices)]=phase_mct
        # phs will broadcast to add a constant phase across the time dimension
        # but not sure about doing it outside of the loop like this where phs
        # is a matrix.
        newfid=add_phase(indat.fids,phs)
        outdat.fids=newfid
        outdat.flags['phasecorrected']=True
    return outdat,phs
    
@alter_return_args
def op_phaseAlignAverages_fd(indat, minppm, maxppm, npts=None,med='a',ref=None,weighting='n',freq_range=None,return_extra_args=None):
    out1,phs=op_phaseAlignAverages(indat, npts=npts, med=med, ref=ref, weighting=weighting, freq_range=[minppm, maxppm], return_extra_args=True)
    return out1,phs

@alter_return_args
def op_ppmref(indat,ppmmin,ppmmax,ppmrefval,dimNum=0,zpfact=10,return_extra_args=None):
    # Going to generalize to allow multiple dimensions, but you still need to pick
    # a dimension for subspecs
    whichslice=[slice(None)]*indat.ndim
    if 'subspecs' in indat:
        print('Multiple subspecs. Operating on dimension {:d}. Enter argument dimNum to change'.format(dimNum))
        whichslice[indat.dims['subspecs']]=dimNum
    # zeropad if it's not already done
    if not indat.flags['zeropadded']:
        in_zp=op_zeropad(indat,zpfact)
    else:
        print('Data already zeropadded. Using existing zero padding.')
        in_zp=indat.copy()
    # find the ppm of the maximum peak magnitude within a given range
    ppmidx=np.nonzero(np.logical_and(in_zp.ppm>=ppmmin,in_zp.ppm<=ppmmax))[0]
    whichslice[0]=slice(ppmidx[0],ppmidx[-1])
    specmask=np.zeros_like(np.real(in_zp.specs))
    specmask[tuple(whichslice)]=1
    ppmindex=np.argmax(specmask*np.abs(in_zp.specs),axis=0)
    ppmvals=in_zp.ppm[ppmindex]
    frqshift=(ppmvals-ppmrefval)*indat.txfreq/1e6
    # Not calling op_freqshift because that assumes that frqshift is a scalar,
    # whereas here it can be a matrix. Need to expand dimensions before broadcasting
    outdat=indat.copy()
    frqfull=np.tile(frqshift,[indat.sz[0]]+[1]*frqshift.ndim)
    # Need to deal with subspecs. In the original function, all subspecs are 
    # shifted even though only one subspec is used to determine the frequency shift
    if 'subspecs' not in indat:
        newfid=(indat.fids.T*np.exp(-1j*indat.t*frqfull.T*2*np.pi)).T
    else:
        newfid=indat.fids.copy()
        whichslice[0]=slice(None)
        for specct in range(indat.subspecs):
            whichslice[indat.dims['subspecs']]=specct
        newfid[tuple(whichslice)]=(indat.fids[tuple(whichslice)].T*np.exp(-1j*indat.t*frqfull.T*2*np.pi)).T
    outdat.fids=newfid
    return outdat,frqshift

@alter_return_args
def op_removeWater(indat,wlim=[4.4,5],Kinit=20,M=None,plot_bool=True,return_extra_args=None):
    # This seems to be the same as op_HSVDfit but with different default ppm 
    # limits to define the water peak and then a bunch of the fit parameters
    # added as a new property to the object
    [model, out1, K, wppm, amp, alpha, ph]=op_HSVDfit(indat,ppmlim=wlim,Kinit=Kinit,M=M,plot_bool=plot_bool)
    w=-2*np.pi*(wppm-indat.center_freq_ppm)*(indat.txfreq/1e6)
    water=(wppm>wlim[0])*(wppm<wlim[1])
    fid_model=np.matmul((amp*np.exp(1j*ph)),(np.exp((-1*alpha-(1j*w))*indat.t[:,np.newaxis])).T)
    tmpfid=indat.copy()
    tmpfid.fids=fid_model
    er = sum(np.abs(indat.specs-tmpfid.specs)**2)/indat.sz[0]
    # You can just add the dict on as a new part of the object. However, it won't
    # copy using the copy method.
    watersupp=dict()
    watersupp['damp']=alpha[water]
    watersupp['freq']=w[water]
    watersupp['phase']=ph[water]
    watersupp['amp']=amp[water]
    watersupp['wppm']=wppm[water]
    watersupp['wppm_all']=wppm
    watersupp['damp_all']=alpha
    watersupp['k']=K
    watersupp['residual_error']=er
    out1.watersupp=watersupp
    model.watersupp=watersupp.copy()
    return out1, K, wppm, amp, alpha, ph, model
    
def get_zmetric(indat,which_domain):
    # Common code used in all of the remove averages functions
    datslice=[slice(None)]*indat.ndim
    if which_domain=='t':
        datmat=indat.fids.copy()
        tmax=0.4
        datslice[1]=slice(0,np.nonzero(indat.t>tmax)[0][0])
    elif which_domain=='f':
        filt=10
        datmat=op_filter(indat,filt,return_extra_args=False).specs.copy()
    # not sure why this is a median, but it's like that in the Matlab code 
    # and a similar call to op_averaging(infilt) is commented out
    datavg=np.median(datmat,axis=indat.dims['averages']).squeeze()
    # Everything here gets very complicated dimensionally. Simplest seems to
    # be to move averages to the first dimension to allow broadcasting and 
    # subspecs (if they exist) are now accommodated in the same way as any
    # other "extra" dimensions (except for coils, which are caught by exceptions
    # in the calling functions)
    datmat_rs=np.moveaxis(datmat,indat.dims['averages'],0)
    #first, make a metric by subtracting all averages from the first average, 
    #and then taking the sum of all the spectral points. 
    metric=np.sum((np.real(datmat_rs[tuple(datslice)])-np.real(datavg[tuple(datslice[1:])]))**2,axis=1)
    avg1=np.mean(metric,axis=0)
    sd1=np.std(metric,axis=0)
    #Now z-transform the metric  
    zmetric=(metric-avg1)/sd1
    # Now it gets difficult because there is a polynomial fit that needs
    # 1D data. To preserve dimensions, I will reshape and then loop through
    # Note that reshape automatically gives you a second dimension here.
    zmet_rs=np.reshape(zmetric,[zmetric.shape[0],-1])
    return metric,zmetric,zmet_rs
    
@alter_return_args
def op_rmbadaverages(indat,nsd=3,which_domain='t',return_extra_args=None):
    which_domain=which_domain.lower()
    if indat.averages==1:
        raise FidAException('ERROR:  Averaging has already been performed!  Aborting!')
    elif 'coils' in indat:
        raise FidAException('ERROR:  Receivers should be combined first!  Aborting!')
    else:
        metric,zmetric,zmet_rs=get_zmetric(indat,which_domain)
        P=np.zeros([3,zmet_rs.shape[1]])
        f1,ax1=plt.subplots(1,1)
        xvals=np.r_[:indat.averages]
        yfit=np.zeros_like(zmet_rs)
        for dimct in range(zmet_rs.shape[1]):
            # Note that polyfit operates on 2D yvalues but polyval requires a 
            # 1D vec of parameters
            P[:,dimct]=np.polyfit(xvals,zmet_rs[:,dimct],deg=2)
            yfit[:,dimct]=np.polyval(P[:,dimct],xvals)
            linevec=ax1.plot(xvals,zmet_rs[:,dimct],'.')
            ax1.plot(xvals,yfit[:,dimct],'-',color=linevec[0].get_color())
            ax1.plot(xvals,yfit[:,dimct]+nsd,':',color=linevec[0].get_color())
        ax1.set_xlabel('Average Number')
        ax1.set_ylabel('Unlikeness Metric z-score')
        ax1.set_title('Metric for rejection of motion corrupted scans')
        mask=(zmet_rs>(yfit+nsd))
        # Now, the Matlab code says that, if one average is corrupted then 
        # all of the subspecs corresponding to that average have to be thrown
        # away. I am not sure that the same is true for spatial or extra dimensions
        # But you won't be able to reshape the fids if you remove averages in
        # only certain cases. So I will mark True (bad average) where anything
        # in the second dimension is True. I will likely need to revisit this
        # if running for each voxel because there you presumably want to find
        # the bad averages for each voxel separately (and then eventually average
        # them so you could go back too all data having the same dimension rather
        # than different numbers of averages)
        # Note that mask's 2nd dimension should be subspecs if there are subspecs
        # and 1 if no subspecs. Any extra dimensions (spatial, etc.) will not be 
        # dealt with here.
        mask_avg=np.any(mask,axis=1)
        # mask_avg should now be a 1D vector with length equal to dims['averages']
        badAverages=np.nonzero(mask_avg)[0]
        goodAverages=np.nonzero(1-mask_avg)[0]
        whichslice=[slice(None)]*indat.ndim
        whichslice[indat.dims['averages']]=goodAverages
        outdat=indat.copy()
        outdat.fids=indat.fids[tuple(whichslice)]
    return outdat,metric,badAverages
    
@alter_return_args
def op_rmNworstaverages(indat,n,which_domain='f',return_extra_args=None):
    # For some reason the default domain here is 'f' and it's 't' for rmbadaverages
    which_domain=which_domain.lower()
    if indat.averages==1:
        print('ERROR:  Averaging has already been performed!  Aborting!')
        outdat=indat
    elif 'coils' in indat:
        print('ERROR:  Receivers should be combined first!  Aborting!')
        outdat=indat
    else:
        metric,zmetric,zmet_rs=get_zmetric(indat,which_domain)
        P=np.zeros([3,zmet_rs.shape[1]])
        f1,ax1=plt.subplots(1,1)
        xvals=np.r_[:indat.averages]
        yfit=np.zeros_like(zmet_rs)
        for dimct in range(zmet_rs.shape[1]):
            # Note that polyfit operates on 2D yvalues but polyval requires a 
            # 1D vec of parameters
            P[:,dimct]=np.polyfit(xvals,zmet_rs[:,dimct],deg=2)
            yfit[:,dimct]=np.polyval(P[:,dimct],xvals)
            linevec=ax1.plot(xvals,zmet_rs[:,dimct],'.')
            ax1.plot(xvals,yfit[:,dimct],'-',color=linevec[0].get_color())
        ax1.set_xlabel('Average Number')
        ax1.set_ylabel('Unlikeness Metric z-score')
        # Sort in the averages dimension. For some reason, Matlab sorts the
        # difference between zmetric and the polyval fit and not zmetric itself
        # Note that the n highest values will be at the end and we want their 
        # indices, not their values
        zmet_idx=np.argsort(zmet_rs-yfit,axis=0)
        zmet_keep=zmet_idx[-n:,:]
        mask=np.zeros_like(zmet_rs)
        mask[zmet_keep]=1
        # In the Matlab code, the assumption is that the only extra dimension
        # might be subspecs. And the idea is to find the n worst scores for each
        # subspec. Then, if the worst scores are different for each subspecs
        # case, you discard any where an average is bad. This means that you might
        # actually discard more than n averages in multi-dimensional cases. Again
        # it's worth noting that you will probably want to reconsider how this
        # is done in the case that any "extra" dimensions are voxels (probably
        # send each voxel to this function separately and you can keep different
        # averages for different voxels).
        mask_avg=np.any(mask,axis=1)
        # mask_avg should now be a 1D vector with length equal to dims['averages']
        badAverages=np.nonzero(mask_avg)[0]
        goodAverages=np.nonzero(1-mask_avg)[0]
        whichslice=[slice(None)]*indat.ndim
        whichslice[indat.dims['averages']]=goodAverages
        outdat=indat.copy()
        outdat.fids=indat.fids[tuple(whichslice)]
        # Some questions here about whether averages should be total averages 
        # (including subspecs) or not. Thinking of making this a calculated
        # parameter on the object so that it wouldn't need to be set at all.
        #outdat.averages=len(goodAverages)*indat.rawSubspecs
    return outdat,metric,badAverages
    
@alter_return_args
def op_rmworstaverage(indat,which_domain='f',return_extra_args=None):
    # This could be done just by finding argmax of metric, but I might as well
    # re-use the code from rmNworstaverages with n=1. You could return 
    # badAverages[0] just to make it a scalar, but everything else remains 
    # as-is (eg. outdat.fids is the good averages so still has size > 0)
    [outdat,metric,badAverages]=op_rmNworstaverages(indat,1,which_domain=which_domain,return_extra_args=True)
    return outdat,metric,badAverages[0]

def op_squeeze(indat):
    """
    Removes singleton dimensions from the FID object. ie. if myfid.fids has size
    (2048, 1, 250) and dimensions t, coils, averages, squeeze will return a fid with
    size (2048,2 50) and dimensions t, averages. This is useful because some
    functions (particularly peak fitting and alignment functions) require 1D
    data. Data with dims['t']=0 and dims['averages']=1 are not recognized as 1D
    even if there is only 1 average, but they can be run through this function
    for fitting. Note: this function does not exist in Matlab fid-A.

    Parameters
    ----------
    indat : FID object
        Input FID object.

    Returns
    -------
    outdat : FID object
        FID object with singleton dimensions removed. outdat._dimlist is 
        adjusted to contain the remaining dimensions.

    """
    new_dimlist=[dimnm for dimct,dimnm in enumerate(indat._dimlist) if indat.sz[dimct]!=1]
    outdat=indat.copy()
    outdat.fids=np.squeeze(outdat.fids)
    outdat._dimlist=new_dimlist
    return outdat

def op_timerange(indat,tmin,tmax):
    outdat=indat.copy()
    if tmin!=0:
        print('WARNING: The first point in the time vector is always 0. This function will select the points between tmin and tmax for the fid, but adjust outdat.t to start at 0 (and therefore end at tmax-tmin).')
    indvals=np.logical_and(indat.t>=tmin,indat.t<tmax)
    outdat.fids=indat.fids[indvals,...]
    # Note that I have defined t value as starting at zero. No other tmin possibility
    # is allowed or even really makes sense (and in fact the Matlab code for this 
    # function defines the t value as starting at 0). Since the dwelltime remains
    # the same, the t vector will automatically run from 0 to tmax-tmin. In addition,
    # center frequency and spectralwidth are unchanged by altering tmax (only the
    # spectral resolution changes). Therefore no other changes are needed here.
    return outdat

@alter_return_args
def op_unfilter(indat,lb,return_extra_args=None):
    """
    Multiply the fid by an inverted exponential decay function to undo the effects
    of filtering.

    Parameters
    ----------
    indat : FID class
        input data.
    lb : float
        Line narrowing factor in Hz.

    Returns
    -------
    outdat : FID class
        Output following alignment of averages.
    lor : Numpy array
        Exponential time domain filter that was applied
    """
    outdat=indat.copy()
    cont_flag='y'
    if not indat.flags['filtered']:
        cont_flag=input('WARNING: Line Broadening has not been performed! Proceed with unbroadening (y/n)? ')
    if cont_flag.lower():
        lbt=1/lb
        lor=(1/np.pi)*((lbt/2)/(indat.t**2+(lbt/2)**2))
        lor=np.amax(lor)/lor
        # Matlab makes a bunch of vectors of ones to get the array sizes to match
        # with ngrid. I'm not sure why repmat wasn't used. However, in Python,
        # since 't' is always the first dimension, we can just use broadcasting
        # and the arrays will be expanded automatically (but the dimension being
        # broadcast needs to be last, so we have to transpose, multiply, then
        # transpose back).
        newfid=(indat.fids.T*lor).T
        outdat.fids=newfid
        # Matlab sets the filtered flag to True even though the action is the
        # opposite of op_filter, but this probably makes sense because line
        # narrowing is still a sort of filter and the data have been manipulated.
        outdat.flags['filtered']=True
    return outdat,lor

def op_zerotrim(indat,npts):
    outdat=indat.copy()
    continue_flag='y'
    if not indat.flags['zeropadded']:
        continue_flag=input('WARNING: You are trimming points from the end of the FID even though zero padding has not been performed. Continue? (y or n): ')
    if continue_flag.lower()=='y':
        outdat.fids=indat.fids[:npts,...]
    # Note that dwelltime, spectralwidth and center frequency are all unchanged
    # and other parameters (t, ppm, etc) are calculated from these plus matrix
    # sizes, so no need to recalculate here.
    # Following along with Matlab and setting the zeropadded Flag to False.
    outdat.flags['zeropadded']=False
    return outdat

if __name__ == '__main__':
    """
    for debugging
    """
    import fidA_io as fio
    import os
    from brukerapi.jcampdx import JCAMPDX
    # pname='/Users/nearlabmacbook1/Documents/BrukerData/SchuurmansMice'
    # fct=2
    # with open(os.path.join(pname,'flist_smallvox_right')) as f:
    #     ftmp=f.readlines()
    #     fright=[fn.strip() for fn in ftmp if fn.startswith('2024')]
    # with open(os.path.join(pname,'flist_smallvox_left')) as f:
    #     ftmp=f.readlines()
    #     fleft=[fn.strip() for fn in ftmp if fn.startswith('2024')]
    # out_left,ref_left,info_left=fio.io_loadspec_bruk(os.path.join(pname,fleft[fct]),try_raw=False)
    # out_right,ref_right,info_right=fio.io_loadspec_bruk(os.path.join(pname,fright[fct]),try_raw=False)
    # f1,ax1=plt.subplots(1,2)
    # out_left.plot_spec(plotax=ax1[0])
    # out_right.plot_spec(plotax=ax1[0])
    # new_right, ph1, frq1=op_align_scans(out_left, out_right)
    # out_left.plot_spec(plotax=ax1[1])
    # new_right.plot_spec(plotax=ax1[1])
    
    # #from curvefit_tools import alter_func_args
    # pname='/Users/nearlabmacbook1/Documents/BrukerData/FUS_pentobarbital/20240916_133721_SKWU1A_Sept16_2024_RK50_HL_SKWU1A_Sept16_20_1_4'
    # fid1=fio.io_loadspec_bruk(os.path.join(pname,'5','rawdata.job0'))
    # outdat,fids_presum,specs_presum,coilcombos=op_addrcvrs(fid1)
    # fid_avg,ph0=op_autophase(op_averaging(outdat),2.9,3.1)
    # test=op_creFit(fid_avg)
    
    pname='/Users/nearlabmacbook1/Documents/BrukerS4_Data/OrganoidAlaConstructs/2026_04_10_HRMAS_PBS_ExpSetup/10'
    outdict=fio.io_loadspec_brukNMR(os.path.join(pname,'fid'),spectrometer=True,ADC_OFFSET=68)
    phased_spec,ph0=op_autophase(outdict,new_method=True,show_plots=True)
    # # aligned_scan=aligned_scan[:,1:]
    # # tmpfreq=np.zeros([30,])
    # # for specct in range(1,30):
    # #     print(specct)
    # #     tmpspec,tmpphase,tmpfreq[specct]=op_alignScans(aligned_scan[:,0], aligned_scan[:,specct], tmax=0.5, mode='f',freq_range=[-0.3,5])
    
    # ppmvec=fid1.ppm
    # parvec=[1.5,0.5,2,0]
    # #lbs=[0,0,np.amin(ppmvec),0]
    # #ubs=[5,2,np.amax(ppmvec),1]
    # lorentz_peak=np.real(op_lorentz_linbas(parvec, ppmvec)+0.05*np.random.randn(len(ppmvec))+1j*0.05*np.random.randn(len(ppmvec)))
    # multi_parvec=[[1.5,2,1],[0.5,0.2,0.5],[2,6,8],0]
    # #multi_parvec=[[1.5,2],[0.5,0.2],[2,6],0,0.1]
    # ## Might typically restrict ppm0 more based on expected peak position and fwhm
    # lbs=[[0]*len(multi_parvec[0]),[0]*len(multi_parvec[1]),[np.amin(ppmvec)]*len(multi_parvec[2]),-1]
    # ubs=[[5]*len(multi_parvec[0]),[2]*len(multi_parvec[1]),[np.amax(ppmvec)]*len(multi_parvec[0]),1]
    # multi_lorentz_peak=np.real(op_lorentz_linbas(multi_parvec, ppmvec)+0.02*np.random.randn(len(ppmvec))+1j*0.02*np.random.randn(len(ppmvec)))
    # plt.figure()
    # plt.plot(ppmvec,multi_lorentz_peak)
    # #parsFit=nlinfit(ppmvec, lorentz_peak, op_lorentz_linbas, parvec,bounds=(lbs,ubs))
    # parsFit=nlinfit(ppmvec, multi_lorentz_peak, op_lorentz_linbas, multi_parvec,bounds=(lbs,ubs),full_output=True)
    # print(parsFit)
    # fitPeak=op_lorentz_linbas(parsFit, ppmvec)
    # plt.plot(ppmvec,fitPeak,ls=':')