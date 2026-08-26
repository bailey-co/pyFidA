#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:43:55 2026
pyFidA.fidA_sim.sim_makebasis.py

@author: Colleen Bailey (@cbailey@sri.utoronto.ca), based on Matlab code by Jamie Near

High-level functions for simulating basis sets, including reading in metabolite
information and simulating a basis set for writing to LCModel.

Functions:
    * read_all_spinsys_matlab
    * read_spinsys_matlab
    * sim_lcmrawbasis
"""

import importlib.resources as importlib_resources
import os
from scipy.io import loadmat
from pyFidA.fidA_io import io_writelcmraw
from .sim_sequences import sim_spinecho, sim_press, sim_steam, sim_laser

def read_all_spinsys_matlab(fname=None):
    """
    Imports the .mat file for stored spin systems from pyFidA's assets folder
    into a dict format

    Parameters
    ----------
    fname : string or path object, optional
        DESCRIPTION.

    Returns
    -------
    fullDict : TYPE
        DESCRIPTION.

    """
    if fname is None:
        pname=importlib_resources.files('pyFidA')
        fname=os.path.join(pname,'assets','metabolites','spinSystems.mat')
    # A dict with keys sysAla, sysAsc, etc. where the values are either dicts
    # of 'J', 'shifts', etc OR a list of such dicts when the spin system is broken up.
    fullDict={kn:vn for kn,vn in loadmat(fname,simplify_cells=True).items() if kn.startswith('sys')}
    return fullDict

def read_spinsys_matlab(fname):
    """
    Import the .mat file for a spin system into a dict

    Parameters
    ----------
    fname : string or path object
        Filename of the .mat file to import. If the filename, as entered, 
        doesn't exist, function will try to prepend the location of pyFidA's
        metabolites folder so that users can enter just 'Ala.mat', 'GPC.mat', 
        etc.

    Returns
    -------
    sysmet : dict or list of dicts
        For an inseparable spin system, spin system parameters in the format:
            {'name': str,
             'shifts': 1D np.ndarray,
             'J': 2D np.ndarray,
             'scaleFactor': int}
        For a separable spin system, each part of the system is its own dict 
        entry in a list, with the dicts having the above format.

    """
    # If user doesn't enter a full file name, try to add filename to anticipated path
    if not os.path.exists(fname):
        pname=importlib_resources.files('pyFidA')
        fname=os.path.join(pname,'assets','metabolites',fname)
    # loadmat will contain a number of variable, including __header__ and __version___.
    # Want to select just the spin system, which always starts with 'sys'
    #is is either a dict with the 'J', 'shifts', etc. keys OR a list of such dicts if the spin system is broken up
    sysmet=[vn for kn,vn in loadmat(fname,simplify_cells=True).items() if kn.startswith('sys')][0]
    return sysmet

def sim_lcmrawbasis(npts,sw,Bfield,linewidth,metab,tau1,tau2,addref,makeraw,seq,fname=None):
    allSys=read_all_spinsys_matlab()
    spinSys=allSys['sys'+metab]
    if spinSys is dict:
        spinSys=[spinSys]
    tot_spins=sum([len(eachpart.shifts) for eachpart in spinSys])
    print('simulating metabolite {:s} with {:d} spins... Please wait...'.format(metab,tot_spins))
    spinlist=[spinSys]
    out1=list()
    if addref.lower()=='y':
        # Note that spin systems in spinSystems.mat assume water at 4.65 ppm
        sysRef=[{'name':'Ref_0ppm','shifts':[0],'J':[0],'scaleFactor':1}]
        spinlist=[spinSys,sysRef]
    for eachsys in spinlist:
        # default shift reference (water=4.65 ppm) and center frequency (4.65 ppm) used.
        if seq.lower()=='se':
            out1.append(sim_spinecho(npts,sw,Bfield,linewidth,eachsys,tau1))
        elif seq.lower()=='p':
            out1.append(sim_press(npts,sw,Bfield,linewidth,eachsys,tau1,tau2))
        elif seq.lower()=='st':
            out1.append(sim_steam(npts,sw,Bfield,linewidth,eachsys,tau1,tau2))
        elif seq.lower()=='l':
            out1.append(sim_laser(npts,sw,Bfield,linewidth,eachsys,tau1))
        else:
            raise ValueError('ERROR:  Sequence {:s} not recognized!!!'.format(seq))
    out1=sum(out1,start=0*out1[0])
    if makeraw.lower()=='y':
        if fname is None:
            fname=metab+'.RAW'
        # Should I have a pathname where this gets written as an input argument?
        RF=io_writelcmraw(out1,fname,metab)
        return RF,out1
    else:
        return out1
    
def sim_make2DSimPlot():
    # May write a wrapper function at some point although I don't think that these
    # are quite the same. lcm_ridgeplot may specifically take a list of FIDs
    # whereas lcm_ridgeplot takes a dict of ... something??
    print('Note: Please use fidA_disp.disp_lcm_ridgeplot instead')
    
if __name__ == '__main__':
    """
    for debugging
    """
    import os
    from pyFidA.fidA_processing import op_plotspec,op_autophase
    import matplotlib.pyplot as plt
    import time
    from .sim_sequences import sim_onepulse
    #pname='/Users/nearlabmacbook1/Documents/Matlab/FID-A-master_20250521/simulationTools/metabolites'
    #pname='../exampleData/rfPulses'
    #RF1=io_loadRFwaveform(os.path.join(pname,'sampleExcPulse.pta'),type_p='exc')
    AllSys=read_all_spinsys_matlab(os.path.join()
    m1=read_spinsys_matlab('Gln.mat')
    out1=sim_onepulse(4096,4000,7,linewidth=2,spinSys=m1)
    op_plotspec(out1)
    #out1=sim_cosy(4096,4000,Bfield=7,linewidth=2,spinSys=m1,npts2=256)
    #f1,ax1=plt.subplots(1,1)
    #for specct in range(16):
    #    ax1.plot(out1[specct].ppm+0.1*specct,np.real(out1[specct].specs)+specct*0.01)
    #ax1.set_xlim([7,1.0])
    
    
    # tic=time.perf_counter()
    # for tct in range(20):
    #     out1=sim_onepulse(2048,4000,Bfield=7,linewidth=3,spinSys=m1,anglein=90,ph1=None,centerFreq=4.65)
    # toc=time.perf_counter()
    # print('Time for no phase {:.4f}'.format(toc-tic))
    # tic=time.perf_counter()
    # for tct in range(20):
    #     out1=sim_onepulse(2048,4000,Bfield=7,linewidth=3,spinSys=m1,anglein=90,ph1=90,centerFreq=4.65)
    # toc=time.perf_counter()
    # print('Time for 0 phase {:.4f}'.format(toc-tic))
    # # So actually sim_excite_arbPh seems faster than sim_excite?? They're 
    # # quite similar. So calling sim_excite_arbPh seems fine.
    # plt.figure()
    # plt.plot(out1.ppm,np.real(out1.specs))
    
    
    # H1,d1=sim_Hamiltonian(m1,7,center_freq_ppm=4.65)
    # t=90
    # d2=sim_excite(d1,H1,whichax='x',anglein=90)
    # #d2=sim_evolve(d2,H1,0.1)
    # out1,dfinal=sim_readout(d2,H1,2048,sw=4000,linewidth=3,rcvPhase=90,center_freq_ppm=4.65)
    # plt.figure()
    # plt.plot(out1.ppm,np.abs(out1.specs))
    
    #H1,d1=sim_Hamiltonian(m1,7,center_freq_ppm=0)
    #d2b=sim_shapedRF(d1,H1,RF1,5,flipAngle=90,ph1=0)
    #checkb=d2b[0]
    #out1b,dfinalb=sim_readout(d2b,H1,2048,sw=4000,linewidth=3,rcvPhase=90,center_freq_ppm=4.65)
    #plt.figure()
    #plt.plot(out1b.ppm,np.abs(out1b.specs))
    #f1,ax1=plt.subplots(2,3,sharey=True)
    #ax1=ax1.flatten()
    #for axct,eachang in enumerate([0,45,90,135,180,270]):
    #    d2b=sim_shapedRF(d1,H1,RF1,5,flipAngle=eachang,ph1=90)
    #    out1b,dfinalb=sim_readout(d2b,H1,2048,sw=4000,linewidth=3,rcvPhase=90,center_freq_ppm=4.65)
    #    ax1[axct].plot(out1b.ppm,np.real(out1b.specs))