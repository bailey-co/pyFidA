#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:05:04 2026

@author: nearlabmacbook1
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew,kurtosis
from .curvefit_tools import nlinfit
from .op_common_processing import op_zeropad
from .op_peak_fitting import op_lorentz_linbas
from pyFidA.fidA_common import FidAException

def op_getLW(indat,Refppmmin=4.4,Refppmmax=5.0,zpfactor=8,suppressPlots=True,method=0):
    # Matlab default is suppressPlots=False but this gets messy if you run on 250
    # individual averages so I've set the default as True instead
    indat=op_zeropad(indat,zpfactor)
    whichpts=np.flatnonzero(np.logical_and(indat.ppm>Refppmmin,indat.ppm<Refppmmax))
    ppmwindow=indat.ppm[whichpts]
    Refwindow=indat.specs[whichpts,...]
    # In order to generalize to FIDs with multiple dimensions, I think the easiest
    # is just to put all averages/coils/etc into the second dimension
    oldsz=Refwindow.shape
    newref=np.reshape(Refwindow,[oldsz[0],-1])
    maxRef=np.amax(np.abs(np.real(newref)),axis=0)
    if not suppressPlots:
        f1,ax1=plt.subplots(1,1)
        ax1.plot(ppmwindow,np.abs(np.real(Refwindow)),'.',label='data')
    # METHOD 1:  MEAUSURE FWHM OF PEAK FROM DATA
    if method==1 or method==0:
        # argmax will find the first instance of True.
        gtHalfMax_a=np.argmax(np.abs(np.real(newref))>=0.5*maxRef,axis=0)
        gtHalfMax_b=newref.shape[0]-np.argmax(np.abs(np.real(newref[::-1,:]))>=0.5*maxRef,axis=0)-1
        FWHM1_unshaped=np.abs(ppmwindow[gtHalfMax_a]-ppmwindow[gtHalfMax_b])*indat.txfreq/1e6
        if len(oldsz)>1:
            FWHM1=np.reshape(FWHM1_unshaped,oldsz[1:])
        else:
            if len(FWHM1_unshaped)==1:
                FWHM1=FWHM1_unshaped[0]
            else:
                FWHM1=FWHM1_unshaped.squeeze()
    if method==2 or method==0:
        # METHOD 2:  FIT WATER PEAK TO DETERMINE FWHM PARAM
        sat='n'
        maxind=np.argmax(np.abs(np.real(newref)),axis=0)
        waterFreq=ppmwindow[maxind[0]]
        fwhm_guess=10/(indat.txfreq/1e6)
        while sat=='n':
            FWHM2=list()
            yFit=np.zeros_like(newref)
            # Need to fit each line
            parsGuess=np.zeros(5)
            parsGuess[1]=fwhm_guess #FWHM in ppm.
            for linect,eachind in enumerate(maxind):
                parsGuess[0]=maxRef[linect] #AMPLITUDE
                parsGuess[2]= waterFreq #FREQUENCY position of peak in ppm     
                yGuess=op_lorentz_linbas(parsGuess,ppmwindow);
                parsFit=nlinfit(ppmwindow, newref[:,linect], op_lorentz_linbas, parsGuess,real_nest=True)
                FWHM2.append(parsFit[1])
                # Note that we are calling nlinfit for op_lorentz_linbas directly, 
                #so parsGuess[1] is in ppm and parsFit[1] that is returned will 
                # also be in ppm and can be sent to op_lorentz_linbas as is.
                yFit[:,linect]=op_lorentz_linbas(parsFit,ppmwindow)        
            if not suppressPlots:
                ax1.plot(ppmwindow,yFit,'-',label='fit')
                ax1.plot(ppmwindow,np.real(yGuess),':',label='guess')
                ax1.legend()
                sat=input('are you satisfied with fit? y/n ')
                if sat.lower()=='n':
                    waterFreq=float(input('input new water frequency guess: '))
                    fwhm_guess=float(input('input new water frequency guess: '))
            else:
                sat='y'
        if len(oldsz)>1:
            FWHM2=np.reshape(np.array(FWHM2),oldsz[1:])
        else:
            if len(FWHM2)==1:
                FWHM2=FWHM2[0]
            else:
                FWHM2=np.array(FWHM2)
        FWHM2=FWHM2*indat.txfreq/1e6
    if method==1:
        FWHM=FWHM1
    elif method==2:
        FWHM=FWHM2
    elif method==0:
        FWHM=np.mean([FWHM1,FWHM2],axis=0)
    else:
        raise ValueError('ERROR: method must be 0, 1 or 2 (default 0).')
    if not suppressPlots:
        print('The calculated linewidth is: '+ str(FWHM) + ' Hz.')#.format(FWHM1))
    return FWHM

def op_getPeakHeight(indat,ppmmin=1.8,ppmmax=2.2):
    # Differs from Matlab somewhat in order to allow multiple dimensions
    whichpts=np.flatnonzero(np.logical_and(indat.ppm>ppmmin,indat.ppm<ppmmax))
    peak_window=indat.specs[whichpts,...]
    ht=np.amax(np.abs(peak_window),axis=0)
    return ht

def op_getSNR(indat,ppmmin=1.8,ppmmax=2.2,noiseppmmin=-2,noiseppmmax=0,suppressPlots=True):
    # Find the max peak height in the given range
    maxht=op_getPeakHeight(indat,ppmmin,ppmmax)
    if not suppressPlots and (noiseppmmin is None and noiseppmmax is None):
        f1,ax1=plt.subplots(1,1)
        ax1.plot(indat.ppm,np.real(indat.specs))
        noiseppmmin=float(input('input lower ppm limit for noise: '))
        noiseppmmax=float(input('input upper ppm limit for noise: '))
    # Now find the standard deviation of the noise
    whichpts=np.flatnonzero(np.logical_and(indat.ppm>noiseppmmin,indat.ppm<=noiseppmmax))
    noisewindow=indat.specs[whichpts,...]
    ppmwindow2=indat.ppm[whichpts]
    oldsz=noisewindow.shape
    newnoisewindow=np.reshape(noisewindow,[noisewindow.shape[0],-1])
    noisevals=np.zeros_like(newnoisewindow)
    noisefit=np.zeros_like(newnoisewindow)
    for eachline in range(newnoisewindow.shape[1]):
        p=np.polyfit(ppmwindow2,newnoisewindow[:,eachline],2)
        noisefit[:,eachline]=np.polyval(p,ppmwindow2)
        noisevals[:,eachline]=newnoisewindow[:,eachline]-noisefit[:,eachline]
    if not suppressPlots:
        f2,ax2=plt.subplots(1,1)
        ax2.plot(ppmwindow2,np.real(newnoisewindow))
        ax2.plot(ppmwindow2,np.real(noisefit))
        ax2.plot(ppmwindow2,np.real(noisevals))
    signal=maxht-np.mean(np.real(noisewindow),axis=0)
    noisesd=np.reshape(np.std(np.real(noisevals),axis=0),oldsz[1:])
    SNR=signal/noisesd
    if not suppressPlots:
        print('The calculated signal-to-noise ratio is: '+ str(SNR)+'.')
    return SNR,signal,noisesd

def op_relyTest(indat,show_plots=False):
    if 'averages' not in indat or indat.averages<2:
        raise FidAException('ERROR: Averaging has already been performed! Aborting!')
    # Find the index where the SNR drops below 2. Since this is just to get a
    # time point for later cutoff, I'll use the first point in all dimensions
    # except averages
    outdict=dict()
    whichslice=[0]*indat.ndim
    whichslice[0]=slice(np.int(indat.sz[0]*0.75),indat.sz[0]-1)
    noise=np.mean(np.std(np.real(indat.fids[tuple(whichslice)]),axis=0)) # should be a single number
    signal=np.mean(np.real(indat.fids),indat.dims['averages'])
    SNR=signal/noise
    endpt=np.nonzero(SNR>2)[0][-1]
    if show_plots:
        f1,ax1=plt.subplots(1,1)
        ax1.plot(indat.t,np.abs(SNR))
        ax1.plot([0,indat.t[-1]],[2,2],':k')
    outdict['skewVect']=skew(np.real(indat.fids),axis=indat.dims['averages'])
    # Matlab code subtracts 3, but scipy has a fisher=True flag to do this automatically
    outdict['kurtVect']=kurtosis(np.real(indat.fids),fisher=True,axis=indat.dims['averages'])
    outdict['k_skew']=np.mean(np.abs(outdict['skewVect'][:endpt,...]),axis=0)
    outdict['k_kurt']=np.mean(np.abs(outdict['kurtVect'][:endpt,...]),axis=0)
    outdict['k_skew_noise']=np.mean(np.abs(outdict['skewVect'][endpt:,...]),axis=0)
    outdict['k_kurt_noise']=np.mean(np.abs(outdict['kurtVect'][endpt:,...]),axis=0)
    # This should broadcast okay
    outdict['var_k_skew']=np.mean((np.abs(outdict['skewVect'][endpt:,...])-outdict['k_skew_noise'])**2,axis=0)
    outdict['var_k_kurt']=np.mean((np.abs(outdict['kurtVect'][endpt:,...])-outdict['k_kurt_noise'])**2,axis=0)
    outdict['std_k_skew']=np.sqrt(outdict['var_k_skew'])
    outdict['std_k_kurt']=np.sqrt(outdict['var_k_kurt'])
    outdict['skewInterval']=[outdict['k_skew']-outdict['std_k_skew'],outdict['k_skew']+outdict['std_k_skew']]
    outdict['kurtInterval']=[outdict['k_kurt']-outdict['std_k_kurt'],outdict['k_kurt']+outdict['std_k_kurt']]
    # Now determine if the data is reliable. According to Slotboom et al., the 
    # spectrum is reliable if the skewInterval contains the values 0.3272 and
    # if the kudrtosisInterval contains the value 0.6101
    if np.all(np.logical_and(outdict['skewInterval'][0]<=0.3272,outdict['skewInterval'][1]>=0.3272)) and np.all(np.logical_and(outdict['kurtInterval'][0]<=0.6101,outdict['kurtInterval'][1]>=0.6101)):
        outdict['isReliable']=True
    else:
        outdict['isReliable']=False
    return outdict