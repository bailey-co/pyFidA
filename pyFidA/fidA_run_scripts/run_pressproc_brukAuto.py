#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:51:24 2026

@author: Colleen Bailey
Example run script for Bruker raw data, where metabolite data have multiple
coils and multiple averages. (Reference data read in from fid.refscan are
already coil-combined and for 1 average)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pyFidA
from pathlib import Path


def run_pressproc_brukAuto(fname, fname_w=None,aaDomain='t',tmaxin=0.2,iterin=20,volume=8.0):
    fname=Path(fname)
    outdir=fname.parent
    outdir.joinpath('report').mkdir(exist_ok=True)
    outdir.joinpath('report','figs').mkdir(exist_ok=True)
    raw_mets=pyFidA.io_loadspec_bruk(fname)
    if fname_w is not None:
        raw_water=pyFidA.io_loadspec_bruk(fname_w)
    # Combine coil channels if available
    if 'coils' in raw_mets:
        # Note that the Bruker fid.refscan file does not have the coil info in it (the version in method does, I believe)
        if fname_w is not None and 'coils' in raw_water:
            coil_combos=pyFidA.op_getcoilcombos(raw_water)
            out_w_cc=pyFidA.op_addrcvrs(raw_water,coilcombos=coil_combos,return_extra_args=False)
        else:
            if fname_w is not None:
                out_w_cc=raw_water.copy()
            coil_combos=pyFidA.op_getcoilcombos(pyFidA.op_averaging(raw_mets))
        out_mets_cc=pyFidA.op_addrcvrs(raw_mets,coilcombos=coil_combos,return_extra_args=False)
        out_av_cc,fid_av_pre,spec_av_pre,*_=pyFidA.op_addrcvrs(pyFidA.op_averaging(raw_mets),coilcombos=coil_combos,return_extra_args=True)
        raw_av=pyFidA.op_averaging(raw_mets)
        f1,ax1=plt.subplots(1,2)
        ax1[0].plot(raw_av.ppm,np.real(raw_av.specs))
        ax1[0].set_xlim([5,1])
        ax1[0].set_xlabel('Frequency (ppm)')
        ax1[0].set_ylabel('Amplitude (a.u.)')
        ax1[0].set_title('Before correction')
        ax1[1].plot(raw_av.ppm,np.real(spec_av_pre))
        ax1[1].set_xlim([5,1])
        ax1[1].set_xlabel('Frequency (ppm)')
        ax1[1].set_ylabel('Amplitude (a.u.)')
        ax1[1].set_title('After correction')
        f1.savefig(outdir.joinpath('report','figs','coilReconFig.jpg'))
    else:
        print('No coils to average')
        out_mets_cc=raw_mets.copy()
        out_av_cc=pyFidA.op_averaging(raw_mets)
    out_noproc=pyFidA.op_averaging(out_mets_cc)
    if fname_w is not None:
        out_w_noproc=pyFidA.op_averaging(out_w_cc)
    # Skipping removal of bad averages
    out_rm=out_av_cc.copy()
    # Now align averages
    if fname_w is not None:
        out_w_aa=pyFidA.op_alignAverages(out_w_cc,tmax=tmaxin,med='f',return_extra_args=False)
    if out_mets_cc.averages>1:
        if aaDomain=='t':
            [out_aa,phs,fs]=pyFidA.op_alignAverages(out_mets_cc,tmax=tmaxin,med='f')
        else:
            [out_aa,phs,fs]=pyFidA.op_alignAverages(out_mets_cc,freq_range=[1.6,4],tmax=tmaxin,med='f')
        xdata=np.r_[:out_aa.averages]
        fsPoly=np.polyfit(xdata,fs,deg=1)
        fs_yfunc=np.poly1d(fsPoly)
        fs_yFit=fs_yfunc(xdata)
        phsPoly=np.polyfit(xdata,phs,deg=1)
        phs_yfunc=np.poly1d(phsPoly)
        phs_yFit=phs_yfunc(xdata)
        f2,ax2=plt.subplots(1,2)
        ax2[0].plot(out_rm.ppm,np.real(out_rm.specs))
        ax2[0].set_xlim([5,1])
        ax2[0].set_xlabel('Frequency (ppm)')
        ax2[0].set_ylabel('Amplitude (a.u.)')
        ax2[0].set_title('Before average alignment')
        ax2[1].plot(out_aa.ppm,np.real(out_aa.specs))
        ax2[1].set_xlim([5,1])
        ax2[1].set_xlabel('Frequency (ppm)')
        ax2[1].set_ylabel('Amplitude (a.u.)')
        ax2[1].set_title('After average alignment')
        f2.savefig(outdir.joinpath('report','figs','alignAvgs_prePostFig.jpg'))
        # Note that I am plotting the linear fit to the frequency shift but what
        # I have in out_aa is the individual alignments, not the linear fit correction
        f3,ax3=plt.subplots(1,1)
        ax3.plot(xdata,fs,'o')
        ax3.plot(xdata,fs_yFit,'-')
        ax3.set_ylabel('Frequency Drift [Hz]')
        ax3.set_xlabel('Scan Number')
        f3.savefig(outdir.joinpath('report','figs','freqDriftFig.jpg'))
        f4,ax4=plt.subplots(1,1)
        ax4.plot(xdata,phs,'o')
        ax4.plot(xdata,phs_yFit,'-')
        ax4.set_ylabel('Phase Drift [Deg]')
        ax4.set_xlabel('Scan Number')
        f4.savefig(outdir.joinpath('report','figs','phaseDriftFig.jpg'))
        totalFreqDrift=np.amax(fs)-np.amin(fs)
        totalPhaseDrift=np.amax(phs)-np.amin(phs)
    else:
        out_aa=out_mets_cc.copy()
    # Now combine the aligned averages
    out_av=pyFidA.op_averaging(out_aa)
    if fname_w is not None:
        out_w_av=pyFidA.op_alignAverages(out_w_aa,return_extra_args=False)
    # Now do automatic zero-order phase correction (use creatine peak)
    out_ph,ph0=pyFidA.op_autophase(out_av,ppmmin=2.9,ppmmax=3.1,return_extra_args=True)
    if fname_w is not None:
        out_w_ph,ph0w=pyFidA.op_autophase(out_w_av,ppmmin=4,ppmmax=5.5,return_extra_args=True)
    # Do the same phase correction on non-processed for comparison
    out_noproc=pyFidA.op_addphase(out_noproc,ph0)
    if fname_w is not None:
        out_w_noproc=pyFidA.op_addphase(out_w_noproc,ph0w)
    # Frequency shift metabolite spectra so Creatine appears at 3.027 ppm
    _,frqShift=pyFidA.op_ppmref(out_ph,ppmmin=2.9,ppmmax=3.1,ppmrefval=3.027)
    out_final=pyFidA.op_freqshift(out_ph,frqShift)
    out_noproc=pyFidA.op_freqshift(out_noproc,frqShift)
    # And water appears at 4.65 ppm
    if fname_w is not None:
        _,frqShiftw=pyFidA.op_ppmref(out_w_ph,ppmmin=4,ppmmax=5.5,ppmrefval=4.65)
        out_w_final=pyFidA.op_freqshift(out_w_ph,frqShiftw)
        out_w_noproc=pyFidA.op_freqshift(out_w_noproc,frqShiftw)
    f5,ax5=plt.subplots(1,1)
    ax5.plot(out_final.ppm,np.real(out_final.specs))
    ax5.set_xlim([5.2,0.2])
    ax5.set_xlabel('Frequency (ppm)')
    ax5.set_ylabel('Amplitude (a.u.)')
    ax5.set_title('Result: Final Spectrum')
    f5.savefig(outdir.joinpath('report','figs','finalSpecFig.jpg'))
    pressdir=outdir.joinpath('press')
    pressdir.mkdir(exist_ok=True)
    pyFidA.io_writelcm(out_final,pressdir.joinpath('press_lcm'),te=out_final.te,vol=volume)
    if fname_w is not None:
        pressdir2=outdir.joinpath('press_w')
        pressdir2.mkdir(exist_ok=True)
        RF=pyFidA.io_writelcm(out_w_final,pressdir2.joinpath('press_w_lcm'),te=out_final.te,vol=volume)
    # Then you can write a report
    if fname_w is not None:
        return [out_final,out_w_final]
    else:
        return out_final

if __name__ == '__main__':
    
    pname='../../exampleData/bruker'
    out1=run_pressproc_brukAuto(os.path.join(pname,'rawdata.job0'),fname_w=os.path.join(pname,'fid.refscan'),volume=15.0)
    