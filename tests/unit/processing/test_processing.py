#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 13:47:27 2025

@author: nearlabmacbook1
"""
import unittest
import pyFidA
#from pyFidA import fidA_io
#from pyFidA import FID
#from pyFidA import fidA_processing as fop
import numpy as np
from pathlib import Path
from scipy.fft import fftshift, ifft, fft


# Might be able to get rid of these if I've incorporated all of them (or a version
# of them) into the common FID testing file.
def make_lorentzian(ppm,amp,fwhm,ppm0,base_off=0,ph0=0):
    if isinstance(amp,list):
        amp=[amp]
        fwhm=[fwhm]
        ppm0=[ppm0]
    # don't need to make baseline and ph0 into lists because assume same for all peaks
    hwhm=[fv/2 for fv in fwhm]
    y=np.zeros([len(amp),len(ppm)],dtype=np.complex128)
    for act,aval in enumerate(amp):
        ytmp=np.sqrt(2/np.pi)*(hwhm[act]-1j*(ppm-ppm0[act]))/(hwhm[act]**2+(ppm-ppm0[act])**2)
        # Scale it, add baseline, phase by ph0, and take the real part
        ytmp=ytmp/np.amax(np.abs(ytmp))*aval
        y[act,:]=ytmp*np.exp(1j*ph0*np.pi/180)
    y=np.sum(y,axis=0)+base_off
    return y

def simulated_FID(yspec,dimlist):
    yfid=fft(fftshift(yspec,axes=0),axis=0)
    newfid=pyFidA.FID(yfid,spectralwidth=4000,txfreq=300.32e6,te=9,tr=2500,dims=dimlist,nucleus=['1H'],center_freq_ppm=4.65)
    print(newfid.GAMMA)
    return newfid

class TestProcessing(unittest.TestCase):
    
    def setUp(self):
        # An example of how to use make_lorentzian and simulated_FID. In this case
        # this simulates raw data for a very basic spectrum (three peaks) with
        # noise, then sends that spectrum to simualted_FID to be fft-ed and made
        # into an object. Note that ppm and other parameters are hard-coded in that 
        # function. can follow similar concepts to test adding, etc. I guess then,
        # that I want different TestX classes for different setups (averages, coils, etc)
        sdnoise=0.1
        n_av=250
        ppm_vec=np.linspace(11.3095,-2.0095,2048)
        ybasic=make_lorentzian(ppm_vec,[3,1,2],[0.03,0.06,0.03],[4.65,3.2,2.0],base_off=0,ph0=0)
        self.fidbasic=simulated_FID(ybasic,dimlist=['t'])
        yvals=np.tile(ybasic,[n_av,1]).T
        ysz=yvals.shape
        noisevec=sdnoise*np.random.randn(*ysz)+1j*sdnoise*np.random.randn(*ysz)
        yvals=yvals+noisevec
        self.myfid=simulated_FID(yvals,['t','averages'])
        
    def test_averaging(self):
        f1=pyFidA.op_averaging(self.myfid).fids
        f2=np.mean(self.myfid.fids,axis=self.myfid.dims['averages'])
        np.testing.assert_array_equal(f1,f2,err_msg='The averages are not equal')
        #self.assertTrue((fop.op_averaging(self.myfid).fids==np.mean(self.myfid.fids,axis=self.myfid.dims['averages'])).all(),'The averages are not equal')
        
    def test_addphase(self):
        f1=self.myfid.fids
        f2=pyFidA.add_phase(self.myfid.fids,180)
        np.testing.assert_allclose(f1,-1*f2,err_msg='add_phase failed for adding 180 degrees')
        #f2=fop.add_phase(self.fidbasic.fids,360)
        #np.testing.assert_allclose(f1,f2,err_msg='add_phase failed for adding 360 degrees')
        #self.assertTrue((self.fidbasic.fids==fop.add_phase(self.fidbasic.fids,360)).all(),'add_phase failed for adding 360 degrees')
        
class TestRepeats(unittest.TestCase):
    def setUp(self):
        self.n_rep=10
        self.n_coil=4
        ppm_vec=np.linspace(11.3095,-2.0095,2048)
        ybasic=make_lorentzian(ppm_vec,[3,1,2],[0.03,0.06,0.03],[4.65,3.2,2.0],base_off=0,ph0=0)
        yvals=np.tile(ybasic,[self.n_rep,self.n_coil,1]).T
        self.fidrep=simulated_FID(yvals,['t','coils','averages'])
        #yvals=np.tile(ybasic,[self.n_rep,1]).T
        #self.fidrep=simulated_FID(yvals,dim_dict={'t': 0,'coils': -1,'averages': 1,'subspecs': -1})
        
    def test_addphase_broadcasting(self):
        # Don't know that this needs to be a separate module but maybe it does make sense
        # to run processing tests on 1D things and then check broadcasting
        # So maybe they should be separate modules?
        phasevec=np.linspace(0,180,self.n_rep)
        newfid=pyFidA.op_addphase(self.fidrep,phasevec)
        self.assertEqual(newfid.sz,tuple([self.fidrep.fids.shape[0],self.n_coil,self.n_rep]))
        newfid[:,0,::3].plot_spec(xlims=[5.5,0])
        
if __name__=='__main__':
    # can add verbosity=2 as an argument to unittest.main() for more info
    unittest.main()
    #newfid=simulated_FID()
    #fop.op_plotspec(fop.op_averaging(newfid),xlims=[5.0,0])