#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:32:56 2026

@author: nearlabmacbook1
"""
import pyFidA
import os
import numpy as np

def run_getLWandSNR(indat,lw_ppmmin=4,lw_ppmmax=6.0,zpfactor=4,snr_ppmmin=1.8,snr_ppmmax=2.25):
    FWHM=pyFidA.op_getLW(indat,Refppmmin=lw_ppmmin,Refppmmax=lw_ppmmax,zpfactor=zpfactor,suppressPlots=False,method=0)
    SNR1,*_=pyFidA.op_getSNR(indat,ppmmin=snr_ppmmin,ppmmax=snr_ppmmax,noiseppmmin=-2,noiseppmmax=0,suppressPlots=False)
    SNR4,*_=pyFidA.op_getSNR(indat,ppmmin=snr_ppmmin,ppmmax=snr_ppmmax,noiseppmmin=9,noiseppmmax=11,suppressPlots=False)
    SNR=np.mean([SNR1,SNR4])
    return FWHM, SNR

if __name__ == '__main__':
    pname='/Users/nearlabmacbook1/Documents/BrukerData/StressMice/baseline/20230517_142659_768_wang_stress_c639_mL_baseline_1_1/5'
    out1=pyFidA.io_loadspec_bruk(os.path.join(pname,'fid'))
    run_getLWandSNR(out1)