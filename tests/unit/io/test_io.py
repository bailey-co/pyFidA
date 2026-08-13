#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 18:14:24 2026

@author: Colleen Bailey
Placeholder for old __main__ stuff from io_fidspec.py
"""
import pyFidA.fidA_io as fio
import os

if __name__=='__main__':
    #from pyFidA.fidA_processing import op_addrcvrs,op_plotspec
    #pname='/Users/nearlabmacbook1/Documents/Matlab/FID-A-master/exampleData/Bruker/sample01_press/press'
    #pname='/Users/nearlabmacbook1/Documents/BrukerS4_Data/2022-08-24_ChaoWang_Extract_CW2A/2'
    #out1,ref1=io_loadspec_bruk(pname,spectrometer=True)
    #pname='/Users/nearlabmacbook1/Documents/BrukerData/StressMice/baseline/20230517_142659_768_wang_stress_c639_mL_baseline_1_1/5/'
    #out1,ref1,info1=io_loadspec_bruk(pname,spectrometer=False,try_raw=False,ADC_OFFSET=68)
    ##2dseq looks worse than fid or averaged rawdata for some reason
    #fname='/Users/nearlabmacbook1/Documents/BrukerData/FUS_pentobarbital/20240916_115232_SKWU1A_Sept16_2024_RK50_HL_SKWU1A_Sept16_20_1_3/5/rawdata.job0'
    #fname='/Users/nearlabmacbook1/Documents/BrukerData/StressMice/baseline/20230517_162843_768_wang_stress_c642_mR_baseline_1_1/5/'#'rawdata.job0'#'pdata/1/2dseq'
    #fname='/Users/nearlabmacbook1/Documents/BrukerS4_Data/ChaoWang_extract/2022-08-25_ChaoWang_Extract_CW1B/2/fid'
    #fname='/Users/nearlabmacbook1/Documents/BrukerS4_Data/DiffusionExpts/2025_10_24_HRMAS_ALSMedia5/212'
    #fname='/Users/nearlabmacbook1/Documents/papersNpresentations/2dnmr/2025_11_12_BBO_Vincent_Sucrose_QC/5'
    #out1=io_loadspec_bruk(fname)
    #out2=io_loadspec_irBruk(fname)
    #out2,reffid=io_loadspec_brukNMR(fname,spectrometer=False)
    #print(out2)
    #fname='/Users/nearlabmacbook1/Documents/BrukerS4_Data/ChaoWang_extract/2022-08-26_ChaoWang_Extract_CW2B/4/fid'
    #d1=test_bruker(fname)
    #out2,ref2,junk2=io_loadspec_bruk_old('/Users/nearlabmacbook1/Documents/BrukerData/FUS_pentobarbital/20240916_115232_SKWU1A_Sept16_2024_RK50_HL_SKWU1A_Sept16_20_1_3/5',try_raw=True)
    #fname='/Users/nearlabmacbook1/Documents/Matlab/FID-A-master/exampleData/GE/sample01_press/press/P17920.7'
    #out2=io_loadspec_GE(fname)
    #pname2='/Users/nearlabmacbook1/Documents/BrukerData/Wilfred/20240425_085624_910_C5_rt_lt_ear_Baseline_1_1'
    #pname2='/Users/nearlabmacbook1/Documents/BrukerData/Wilfred/20240425_085624_910_C5_rt_lt_ear_Baseline_1_1'
    #outraw=io_loadspec_bruk(os.path.join(pname2,'6','rawdata.job0'))
    #fname='/Users/nearlabmacbook1/Documents/Matlab/FID-A-master/exampleData/GE/sample01_press/press/P17920.7'
    #fname='/Users/nearlabmacbook1/Documents/Matlab/FID-A-master/exampleData/GE/sample02_megapress/megapress/P21504.7'
    #out_m,out_w=io_loadspec_GE(fname)
    #fname='/Users/nearlabmacbook1/Documents/Matlab/FID-A-master/exampleData/Siemens/sample01_megapress/megapress/megapressDLPFC.dat'
    #fname='/Users/nearlabmacbook1/Documents/Matlab/FID-A-master/exampleData/Siemens/sample01_megapress/megapress_w/megapressDLPFC_w.dat'
    #fname='/Users/nearlabmacbook1/Documents/Matlab/FID-A-master/exampleData/Siemens/sample02_special/special_w/specialDLPFC_w.dat'
    #out1=io_loadspec_twix(fname)
    #data,f_out=io_loadjmrui(fname)
    #fname='/Users/nearlabmacbook1/Documents/BrukerData/FUS_pentobarbital/lcm-out/20240916_115232_SKWU1A_Sept16_2024_RK50_HL_SKWU1A_Sept16_20_1_3/20240916_post.print'
    #metabs,corrMatrix=io_loadlcmdetail(fname)
    #fname='/Users/nearlabmacbook1/Documents/test_data/Siemens/20250905_PTRU1U_svs_test/special/meas_MID00512_FID06216_jn_svs_special_xa60a_jhead.dat'
    #out1=io_loadspec_twix(fname,subspecs=2)
    ## This is a converted Bruker fid file that is just one average. I 
    ## converted with spec2nii. I could try converting something with more dimensions
    ## Can't do Bruker because that spec2nii doesn't handle raw data.
    #fname='/Users/nearlabmacbook1/Documents/test_data/nifti_mrs_from_bruker/FID.nii.gz'
    #out1=nifti_mrs.NIFTI_MRS(fname)
    # fname='/Users/nearlabmacbook1/Documents/BrukerData/FUS_pentobarbital/lcm-out/20240916_115232_SKWU1A_Sept16_2024_RK50_HL_SKWU1A_Sept16_20_1_3/20240916_pre.coord'
    # lcm_dict,info_dict=io_readlcmcoord(fname)
    # Naa_spec=io_readlcmcoord(fname,'NAA')
    # outdat=io_readlcmcoord_getBackground(fname,'NAA')
    # bgrd=io_readlcmcoord_getBackground(fname,'bg')
    # op_plotspec(outdat-bgrd)
    #fname='/Users/nearlabmacbook1/Documents/BrukerData/Wilfred/macromolecular_baseline/simulations/RAW_files_gauss/NAA.RAW'
    #outdat=io_readlcmraw_dotraw(fname)
    #op_plotspec(outdat)
    #fname='/Users/nearlabmacbook1/Documents/lcmodel/basis-sets/PRESS_TE9ms.basis'
    #outdict,infodict=io_readlcmraw_basis(fname,return_info_dict=True)
    #out1=outdict['Lac']
    #io_writejmrui(out1, '/Users/nearlabmacbook1/Documents/lcmodel/fake_jmrui.txt')
    #fname='/Users/nearlabmacbook1/Documents/BrukerData/FUS_pentobarbital/lcm-out/20240916_115232_SKWU1A_Sept16_2024_RK50_HL_SKWU1A_Sept16_20_1_3/20240916_post.table'
    #info_dict=io_readlcmtab(fname)
    
    #fname='/Users/nearlabmacbook1/Documents/Matlab/FID-A-master/exampleData/GE/sample01_press/press/P17920.7'
    #myfid,reffid=io_loadspec_GE(fname)
    
    pname='/Users/nearlabmacbook1/Documents/BrukerS4_Data/AlejandraScans/2026_07_22_HRMAS_chickenheart_20ul/200'
    out2,reffid=fio.io_loadspec_brukNMR(os.path.join(pname,'ser'),try_raw=True,return_info_dict=False)