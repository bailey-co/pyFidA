#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:29:32 2026

@author: Colleen Bailey
Example run script for Siemens megapress data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyFidA
from pathlib import Path

def run_megapressproc(filestring,coilcombos=None,avgAlignDomain='f',alignSS=False):
    filefull=Path(filestring)
    fname=filefull.stem
    pname=filefull.parent
    pname_w=pname.parent.joinpath(pname.stem+'_w')
    outdir=pname.parent
    #outdir.joinpath('report').mkdir(exist_ok=True)
    #outdir.joinpath('report','figs').mkdir(exist_ok=True)
    if pname_w.joinpath(fname+'_w.dat').exists():
        file_water=pname_w.joinpath(fname+'_w.dat')
        water_exists=True
    else:
        water_exists=False
    # read in data sets
    raw1=pyFidA.io_loadspec_twix(filefull)
    if water_exists:
        raww=pyFidA.io_loadspec_twix(file_water)
    # Combine coils
    if water_exists:
        if coilcombos is None:
            coilcombos=pyFidA.op_getcoilcombos(raww)
        outw_cc,fidw_pre,specw_pre,_=pyFidA.op_addrcvrs(raww,coilcombos=coilcombos)
    else:
        if coilcombos is None:
            coilcombos=pyFidA.op_getcoilcombos(pyFidA.op_averaging(pyFidA.op_combinesubspecs(raw1,'summ')))
    out1_cc,fid1_pre,spec1_pre,_=pyFidA.op_addrcvrs(raw1,coilcombos=coilcombos)
    out1_av_cc,fid1_av_pre,spec1_av_pre,_=pyFidA.op_addrcvrs(pyFidA.op_averaging(raw1),coilcombos=coilcombos)
    raw1_av=pyFidA.op_averaging(raw1)
    # Make plots 
    
    # Skipping removal of bad averages (will add later)
    out1_rm=out1_cc.copy()
    nsd1='N/A'
    
    # Align averages
    if water_exists:
        out_w_aa=pyFidA.op_alignAverages(outw_cc,tmax=0.2,med='f',return_extra_args=False)
    if avgAlignDomain=='t':
        [out_aa,phs,fs]=pyFidA.op_alignAverages(out1_rm,tmax=0.2,med='y')
    else:
        [out_aa,phs,fs]=pyFidA.op_alignAverages(out1_rm,freq_range=[1.6,4],tmax=0.2,med='y')
    xdata=np.r_[:out_aa.averages]
    f2,ax2=plt.subplots(1,2)
    ax2[0].plot(out1_rm.ppm,np.real(pyFidA.op_averaging(out1_rm).specs))
    ax2[0].set_xlim([6,-6])
    ax2[0].set_xlabel('Frequency (ppm)')
    ax2[0].set_ylabel('Amplitude (a.u.)')
    ax2[0].set_title('Before average alignment')
    ax2[1].plot(out_aa.ppm,np.real(pyFidA.op_averaging(out_aa).specs))
    ax2[1].set_xlim([6,-6])
    ax2[1].set_xlabel('Frequency (ppm)')
    ax2[1].set_ylabel('Amplitude (a.u.)')
    ax2[1].set_title('After average alignment')
    f3,ax3=plt.subplots(1,1)
    ax3.plot(xdata,fs,'o')
    f4,ax4=plt.subplots(1,1)
    ax4.plot(xdata,phs,'o')
    for dimct in range(fs.shape[1]):
        fsPoly=np.polyfit(xdata,fs[:,dimct],deg=1)
        fs_yfunc=np.poly1d(fsPoly)
        fs_yFit=fs_yfunc(xdata)
        ax3.plot(xdata,fs_yFit,'-')
        phsPoly=np.polyfit(xdata,phs[:,dimct],deg=1)
        phs_yfunc=np.poly1d(phsPoly)
        phs_yFit=phs_yfunc(xdata)
        ax4.plot(xdata,phs_yFit,'-')
    #f2.savefig(outdir.joinpath('report','figs','alignAvgs_prePostFig.jpg'))
    # Note that I am plotting the linear fit to the frequency shift but what
    # I have in out_aa is the individual alignments, not the linear fit correction
    ax3.set_ylabel('Frequency Drift [Hz]')
    ax3.set_xlabel('Scan Number')
    #f3.savefig(outdir.joinpath('report','figs','freqDriftFig.jpg'))
    ax4.set_ylabel('Phase Drift [Deg]')
    ax4.set_xlabel('Scan Number')
    #f4.savefig(outdir.joinpath('report','figs','phaseDriftFig.jpg'))
    totalFreqDrift=np.amax(fs)-np.amin(fs)
    totalPhaseDrift=np.amax(phs)-np.amin(phs)
    
    # Now combine the aligned averages
    out_av=pyFidA.op_averaging(out_aa)
    if water_exists:
        out_w_av=pyFidA.op_averaging(out_w_aa)
        
    # Align the subspectra
    if alignSS:
        out1=pyFidA.op_alignMPSubspecs_fd(out_av,0.2,4,return_extra_args=False)
    else:
        out1=out_av.copy()
        
    out1_diff=pyFidA.op_combinesubspecs(out1,'diff')
    out1_sum=pyFidA.op_combinesubspecs(out1,'summ')
    
    if water_exists and 'subspecs' in out_w_av:
        # I think that I need to distinguish between two sequence types that give 'diff' vs 'summ'
        outw=pyFidA.op_combinesubspecs(out_w_av,'summ')
        outw=pyFidA.op_addphase(outw,-1*np.angle(outw.fids[0])*180/np.pi,suppress_plot=True)
    
    f5,ax5=plt.subplots(1,1)
    ax5.plot(out1_diff.ppm,np.real(out1_diff.specs))
    ax5.plot(out1_sum.ppm,np.real(out1_sum.specs))
    ax5.set_xlim([5.2,0.2])
    ax5.set_xlabel('Frequency (ppm)')
    ax5.set_ylabel('Amplitude (a.u.)')
    ax5.set_title('Result: Final Difference and summed spectra')
    #f5.savefig(outdir.joinpath('report','figs','finalSpecFig.jpg'))
    if water_exists:
        return out1_diff,out1_sum,out1,outw,coilcombos
    else:
        return out1_diff,out1_sum,out1,coilcombos

if __name__ == '__main__':
    fname='/Users/nearlabmacbook1/Documents/Matlab/FID-A-master/exampleData/Siemens/sample01_megapress/megapress/megapressDLPFC.dat'
    [out1_diff,out1_sum,out1,outw,coilcombos]=run_megapressproc(fname)