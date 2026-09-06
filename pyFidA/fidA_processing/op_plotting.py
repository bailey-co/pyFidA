#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:54:24 2026

@author: nearlabmacbook1
"""
import numpy as np
import matplotlib.pyplot as plt

def op_plotfid(indat,tmax=None,xlab='Time (s)',ylab='FID Amplitude (arb units)',title='',stagger=0,plotax=None,**kwargs):
    # Matlab formatting is set explicitly, fontsize, linewidth, etc. Here, I 
    # leave it up to either plt.rc to set the rcParams or aspects of linewidth
    # colors, etc. can be passed to plt.plt using kwargs
    if plotax is None:
        [f1,plotax]=plt.subplots(1,1)
    if isinstance(indat, list):
        if tmax is None:
            tmax=np.amax(indat[0].t)
        # Check that every entry in the list is of effective size 1
        if any([eachit.fids.ndim>2 for eachit in indat]) or any([eachit.fids.ndim==2 and (not (1 in eachit.sz)) for eachit in indat]):
            print("List entries cannot have more than 1 dimension. Did you forget to average?")
        else:
            for itct,eachit in enumerate(indat):
                plotax.plot(eachit.t,np.real(eachit.fids)+itct*stagger,**kwargs)
    else:
        if tmax is None:
            tmax=np.amax(indat.t)
        if indat.ndim>2:
            print("Cannot plot more than 2 dimensions. Input dims: ")
            for kval,vval in indat.dims.items():
                if vval!=-1:
                    print('{:s}: {:d}'.format(kval,vval))
            whichdim=int(input('Which dimension would you like to display? '))
            whichslice=[0]*indat.ndim
            whichslice[0]=slice(None)
            whichslice[whichdim]=slice(None)
            stagvec=np.r_[0:indat.sz[indat.dims[whichdim]]*stagger:stagger]
            plotax.plot(indat.t,np.real(indat.fids)+stagvec,**kwargs)
        else:
            plotax.plot(indat.t,np.real(indat.fids),**kwargs)
    plotax.set_xlim([0,tmax])
    plotax.set_xlabel(xlab)
    plotax.set_ylabel(ylab)
    plotax.set_title(title)
    return plotax
    
def op_plotspec(indat,xlims=None,xlab='Chemical Shift (ppm)',ylab='Signal',title='',stagger=0,plotax=None,**kwargs):
    if xlims is None:
        if indat.nucleus[0]=='1H':
            xlims=[5.2,0.2]
        else:
            xlims=[indat.ppm[0],indat.ppm[-1]]
    # Need to update to deal with multiple averages and other possible dimensions, as in Matlab op_plotspec
    if plotax is None:
        [f1,plotax]=plt.subplots(1,1)
    # Two cases: a list of FID objects, or a single FID object that may have 2+dimensions
    if isinstance(indat, list):
        # Check that every entry in the list is of effective size 1
        if any([eachit.fids.ndim>2 for eachit in indat]) or any([eachit.fids.ndim==2 and (not (1 in eachit.sz)) for eachit in indat]):
            print("List entries cannot have more than 1 dimension. Did you forget to average?")
        else:
            for itct,eachit in enumerate(indat):
                plotax.plot(eachit.ppm,np.real(eachit.specs)+itct*stagger,**kwargs)
    else:
        if indat.fids.ndim>2:
            print("Cannot plot more than 2 dimensions. Input dims: ")
            for kval,vval in indat.dims.items():
                if vval!=-1:
                    print('{:s}: {:d}'.format(kval,vval))
            whichdim=int(input('Which dimension would you like to display? '))
            whichslice=[0]*indat.ndim
            whichslice[0]=slice(None)
            whichslice[whichdim]=slice(None)
            stagvec=np.r_[0:indat.sz[indat.dims[whichdim]]*stagger:stagger]
            plotax.plot(indat.t,np.real(indat.specs[whichslice])+stagvec,**kwargs)
        else:
            plotax.plot(indat.ppm,np.real(indat.specs),**kwargs)
    plotax.set_xlim(xlims)
    plotax.set_xlabel(xlab)
    plotax.set_ylabel(ylab)
    plotax.set_title(title)
    return plotax