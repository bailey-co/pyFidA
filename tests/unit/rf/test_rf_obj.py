#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:44:09 2026

@author: nearlabmacbook1
"""

import unittest
import numpy as np
from pathlib import Path
from pyFidA import RF_pulse, io_loadRFwaveform
from pyFidA.fidA_rf import rf_freqshift, rf_blochSim, rf_addGrad
from pyFidA.fidA_common import FidAWarningRF, FidAWarning, FidAException
import matplotlib.pyplot as plt
import warnings

# For a lot of things. I might be able to start with the stuff that I've got
# in main??
# Note that this will just be for testing (a) the loading of rf pulses?? Or
# should that be in the io toolbox? I guess I will need to load but I mostly
# want to test that the various properties are accessible. And (b) the manipulations
# in the rfPulse module. It won't test the simulation toolbox.
# I'll want to test things for phase-modulate and non-phase-modulated pulses
# And I'll want to test gradient-modulated for various things (both that they
# function in some cases and throw exceptions in others.) Should also test 
# pulses with even time steps and without.

# Test normal inputs (standard values that the function expects to receive)
# Test edge cases (zero, emtpy lists, very large numbers)
# Error Handling (test how code reacts to bad data or missing values)
# State changes (test if the object or system updates correctly after running code)
# You don't have to write for every case but try to focus on things that you think
# realistically might break (eg. I would think that most processing functions
# should check that the size of myfid.fids and length of myfid._dimlist match)

# Not sure if there's an inconsistency here. I put the testing for the FID object
# in common. So then the testing for the RF_pulse object itself would also go
# there and this would just be for the RF toolbox.
@unittest.skip("Already tested. Skipping while I implement new tests in next class")
class TestRF_pulse(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        print('\nGenerating Hanning-windowed sinc pulse for testing as rfpulse')
        npts=100
        t=np.linspace(-2.5e-3,2.5e-3,npts)
        self.tvec=t-t[0]
        rfwf=np.zeros([npts,2])
        sinc_tmp=np.hanning(npts)*(np.sin(2*np.pi*600*t)/(t))
        rfwf[:,0]=(-1*np.sign(sinc_tmp)/2+0.5)*180
        rfwf[np.sign(sinc_tmp)==0,0]=0
        rfwf[:,1]=np.abs(sinc_tmp)
        self.tp=5e-3
        self.rfpulse=RF_pulse(rfwf,'exc',Tp=self.tp,f0=0,suppress_plots=True)
        
        # An off-resonance pulse
        self.offres_f0=1230
        print('\nGenerating off-resonance sinc pulse for testing as offres_pulse. f0={:.0f}'.format(self.offres_f0))
        rfwf[:,0]=rfwf[:,0]+(t-t[0])*self.offres_f0*360
        self.offres_pulse=RF_pulse(rfwf,self.rfpulse.pulse_type,Tp=self.tp,f0=self.offres_f0,suppress_plots=True)
        
        # A gradient-modulated pulse (constant gradient, so just narrows bandwidth)
        self.gradval=0.9
        print('\nGenerating gradient-modulated pulse as gm_pulse. Gradient {:3.2f} G/cm'.format(self.gradval))
        rfwf=np.ones([npts,4])
        sinc_tmp=np.hanning(npts)*(np.sin(2*np.pi*600*t)/(t))
        rfwf[:,0]=(-1*np.sign(sinc_tmp)/2+0.5)*180
        rfwf[np.sign(sinc_tmp)==0,0]=0
        rfwf[:,1]=np.abs(sinc_tmp)
        rfwf[:,3]=self.gradval*np.ones([npts,])
        self.gm_pulse=RF_pulse(rfwf,'exc',Tp=self.tp,f0=0,suppress_plots=True)
        
        # A pulse with non-uniform time steps (square pulse)
        print('\nGenerating square pulse with non-uniform time steps as condensed_square')
        rfwf=np.zeros([5,3])
        rfwf[2,1]=1
        rfwf[:,2]=np.r_[1,1,10,1,1]
        self.expanded_tpts=np.ones([int(np.sum(rfwf[:,2])),])
        self.expanded_tvec=np.cumsum(self.expanded_tpts-self.expanded_tpts[0])
        self.condensed_square=RF_pulse(rfwf,'exc',Tp=self.tp,f0=0,suppress_plots=True)
        rfwf=np.concatenate([np.zeros([14,1]),np.expand_dims(np.array([0,0]+[1]*10+[0,0]),1)],axis=1)
        self.expanded_square=RF_pulse(rfwf,'exc',Tp=self.tp,f0=0,suppress_plots=True)
        
        
        # An adiabatic pulse? Maybe can combine this with the testing for rf_verse, etc.
        # Also would be good to have a phase-modulated pulse (I guess that the off-resonance pulse is this)
    
    def test_rf_creation_sinc(self):
        self.assertIsInstance(self.rfpulse, RF_pulse,'Created object is not an instance of RF_pulse')
        self.assertEqual(self.rfpulse.waveform.shape[1],3)
        self.assertFalse(self.rfpulse.isGM, 'Standard sinc pulse should not return isGM=True')
        self.assertFalse(self.rfpulse.isPhsMod, 'Standard sinc pulse should be phase-modulated')
        self.assertFalse(self.rfpulse.isAdiabatic, 'Standard sinc pulse should not be adiabatic')
        
    def test_rf_creation_offres(self):
        self.assertIsInstance(self.offres_pulse, RF_pulse,'Created object is not an instance of RF_pulse')
        self.assertEqual(self.offres_pulse.waveform.shape[1],3)
        self.assertTrue(self.offres_pulse.isPhsMod, 'Off-resonance pulse should not be phase-modulated')
    
    def test_rf_creation_gm(self):
        self.assertIsInstance(self.gm_pulse, RF_pulse,'Created object is not an instance of RF_pulse')
        self.assertEqual(self.gm_pulse.waveform.shape[1],4)
        self.assertTrue(self.gm_pulse, 'Gradient-modulated pulse should not return isGM=False')
        self.assertFalse(self.gm_pulse.isPhsMod, 'This gradient-modulated pulse is not phase-modulated but incorrectly returns isPhsMod=True')
        self.assertFalse(self.gm_pulse.isAdiabatic, 'This gradient-modulated pulse should not be adiabatic')
        with self.assertWarns(FidAWarningRF,msg='Gradient-modulated pulse should generate FidAWarningRF when self.bw is called'):
            self.gm_pulse.bw
            
    def test_creation_fail_badsize(self):
        with self.assertRaises(FidAException,msg="rf waveform of size n x 1 should raise FidAException"):
            RF_pulse(np.ones([10,1]),'exc')
        with self.assertRaises(FidAException,msg="rf waveform with 1 dimension should raise FidAException"):
            RF_pulse(np.ones([10]),'exc')
        
    def test_gm_pulse_bw(self):
        self.assertAlmostEqual(self.rfpulse.bw/(self.gradval*self.rfpulse.gamma)*10,self.gm_pulse._tbw/self.tp/1000,3, 'Bandwidth calculation for gradient-modulated case does not match expected bandwidth for {:3.2f} gradient.'.format(self.gradval))
        
    def test_rf_creation_freqfail(self):
        with self.assertWarns(FidAWarningRF,msg='On-resonance pulse with incorrect f0=900 input argument should generate FidAWarning'):
            RF_pulse(self.rfpulse.waveform,self.rfpulse.pulse_type,self.tp,f0=900,suppress_plots=True)
            
    def test_rf_creation_highflip(self):
        with self.assertWarns(FidAWarningRF,msg='High flip angles should give warning that w1 estimation may fail'):
            RF_pulse(self.rfpulse.waveform,360,self.tp,suppress_plots=True)
        
    def test_estimate_f0(self):
        newrf=RF_pulse(self.offres_pulse.waveform,self.offres_pulse.pulse_type,self.tp,f0=None,suppress_plots=True)
        self.assertLess(np.abs((newrf.f0-self.offres_f0)/self.offres_f0), 0.05, 'Incorrect f0 estimate when RF_pulse created with input argument f0=None ({:3.1f} vs {:3.1f})'.format(newrf.f0,self.offres_f0))
        
    def test_copy(self):
        newrf=self.rfpulse.copy()
        np.testing.assert_array_equal(newrf.waveform,self.rfpulse.waveform,err_msg='Copy failed. Waveforms not equal.')
        np.testing.assert_array_equal(newrf._pulse_freq_profile,self.rfpulse._pulse_freq_profile,err_msg='Copy failed. Frequency profiles not equal.')
        self.assertEqual(newrf.w1max,self.rfpulse.w1max,'Copy failed. w1max not equal.')
        self.assertEqual(newrf.bw,self.rfpulse.bw, 'Copy failed. Bandwidths not equal.')
        
    def test_get_tvec(self):
        dt=(self.tvec[1]-self.tvec[0])
        returned_tvec=self.rfpulse.get_tvec(self.tvec[-1]+dt)
        np.testing.assert_array_almost_equal(returned_tvec,self.tvec,err_msg='get_tvec function failed for uniform time steps')
        # For expanded waveform, check dt and last timepoint against condensed 
        # (these specific tests only work in this case because first/last timepoint 
        # have step size 1, but proves the principle for more general case. You 
        # can't test if the arrays are equal because they have different lengths.
        # You could test if the expanded_square tvec matched what you expect
        # based on the waveform, but that tells you nothing about the tvec for
        # the condensed waveform, although I suspect that using get_tvec for the 
        # condensed waveform would be rare)
        self.assertEqual(self.expanded_square.get_tvec()[1], self.condensed_square.get_tvec()[1],'Incorrect time spacing for get_tvec when called for pulse with non-uniform time steps')
        self.assertEqual(self.expanded_square.get_tvec()[-1], self.condensed_square.get_tvec()[-1],'Incorrect total time for get_tvec when called for pulse with non-uniform time steps')
        
    def test_timestep_creation(self):
        newrf=RF_pulse(np.ones([50,2]),self.tp,f0=0,iscopy=True)
        self.assertEqual(newrf.waveform.shape,tuple([50,3]),'Third array column of time steps was not automatically generated for RF_pulse with input waveform of two columns.')
        
    def test_rfCentre(self):
        self.assertAlmostEqual(self.rfpulse.rfCentre,0.5,'Failed rfCentre calculation for sinc pulse')
        rfwf=np.array([[0,0,0,0,0],[1,0,0,0,0]]).T
        # This is not a real rf waveform but can use iscopy=True to avoid Bloch
        # simulations and just test rfCentre expectations
        newrf=RF_pulse(rfwf,self.tp,f0=0,iscopy=True)
        self.assertAlmostEqual(newrf.rfCentre,0.2,'Failed rfCentre calculation for pulse with rfCentre=0.2')
        rfwf=np.array([[0,0,0,0,0],[0,0,0,0,1]]).T
        newrf=RF_pulse(rfwf,self.tp,f0=0,iscopy=True)
        self.assertAlmostEqual(newrf.rfCentre,1,'Failed rfCentre calculation for pulse with rfCentre=1')
        
    def test_adiabatic_override(self):
        newrf=self.rfpulse.copy()
        self.assertFalse(newrf.isAdiabatic,'Standard sinc pulse should start with isAdiabatic=False')
        newrf.isAdiabatic=True
        self.assertTrue(newrf.isAdiabatic,'Adiabatic override to force isAdiabatic=True failed.')
        
    def test_movef0_sinc(self):
        offres_moved=self.rfpulse.copy()
        offres_moved.f0=self.offres_f0
        # Note that there are two bits of code to shift/change f0. Testing both.
        self.assertEqual(offres_moved.f0,self.offres_pulse.f0,'Moving f0 failed when assigning self.f0=new value')
        offres_moved=rf_freqshift(self.rfpulse,F=self.offres_f0)
        self.assertEqual(offres_moved.f0,self.offres_pulse.f0,'Moving f0 failed when using rf_freqshift function')
        # Resolution of fvec is 0.001 kHz
        np.testing.assert_array_almost_equal(self.offres_pulse._fvec , offres_moved._fvec, decimal=3, err_msg='Frequency vector _fvec not correct following f0 shift')
        # Note that testing the w1 profile doesn't work, even if you do precision of just 2 decimal places.
        # I assume because Mz changes rapidly in some regions so, if the x-values are off by
        # 1 Hz then y could have large apparent error for the same index point.
        # But, if f0 is equal and bw is equal, then at least that's a good indication?
        np.testing.assert_array_almost_equal(self.offres_pulse._pulse_freq_profile[2,:], offres_moved._pulse_freq_profile[2,:], decimal=1, err_msg='Frequency profiles _pulse_freq_profile[2,:] do not match following f0 shift')
        self.assertAlmostEqual(self.offres_pulse.bw, offres_moved.bw, places=3, msg='Bandwidth not equal after frequency shift')
        self.assertAlmostEqual(self.offres_pulse.w1max, offres_moved.w1max,4, msg='Power does not match after rf pulse frequency shift')
        
    def test_movef0_gm(self):
        # Check the basics for gradient-modulated case
        gm_moved=self.gm_pulse.copy()
        gm_moved.f0=self.offres_f0
        self.assertAlmostEqual(self.gm_pulse._tbw/self.tp, gm_moved._tbw/self.tp, places=3, msg='Bandwidth not equal after frequency shift')
        self.assertAlmostEqual(self.gm_pulse.w1max, gm_moved.w1max,4, msg='Power does not match after rf pulse frequency shift')
        self.assertAlmostEqual(self.gm_pulse._fvec[0]+self.offres_f0/1000, gm_moved._fvec[0],3, msg='Frequency vectors do not match after frequency shift')
        
    def test_expanded_wf(self):
        expanded_wf=self.condensed_square.get_expanded_wf()
        self.assertEqual(expanded_wf.shape[0],sum(self.condensed_square.waveform[:,2]),'Total waveform length for expanded waveform does not match sum of timesteps from equivalent condensed waveform.')
        self.assertAlmostEqual(self.condensed_square.w1max, self.expanded_square.w1max,'Error for get_expanded_wf(). w1max incorrect')
        self.assertAlmostEqual(self.condensed_square.bw, self.expanded_square.bw,'Error for get_expanded_wf(). bw incorrect')
        np.testing.assert_array_almost_equal(self.condensed_square._w1_profile, self.expanded_square._w1_profile,err_msg='Error for get_expanded_wf(). _w1_profile incorrect')
        np.testing.assert_array_almost_equal(self.condensed_square._pulse_freq_profile, self.expanded_square._pulse_freq_profile,err_msg='Error for get_expanded_wf(). _pulse_freq_profile incorrect')
        
    def test_flipangle(self):
        # Test first for sinc
        pulse_30deg=RF_pulse(self.rfpulse.waveform,30,Tp=self.tp,f0=0,suppress_plots=True)
        self.assertAlmostEqual(self.rfpulse.w1max, 3*pulse_30deg.w1max,3,'Power not proportional to flip angle for sinc pulse')
        # For phase-modulate (off-res)
        pulse_30deg=RF_pulse(self.offres_pulse.waveform,30,Tp=self.tp,f0=self.offres_f0,suppress_plots=True)
        self.assertAlmostEqual(self.offres_pulse.w1max, 3*pulse_30deg.w1max,3,'Power not proportional to flip angle for sinc pulse')
        # And for gradient-modulated
        newrf=RF_pulse(self.gm_pulse.waveform,35,self.tp,self.gm_pulse.f0,suppress_plots=True)
        self.assertAlmostEqual(self.gm_pulse.w1max, (90/35)*newrf.w1max,3,'Power not proportional to flip angle for gradient-modulated rf pulse')
        
    def test_tw1(self):
        # Test first for sinc
        w1_point=np.flatnonzero(self.rfpulse._w1_profile[2,:]<0)[0]
        self.assertAlmostEqual(self.rfpulse.w1max,self.rfpulse._f1vec[w1_point],3,'Power does not match _w1_profile value for 90 degree flip angle of sinc pulse')
        newrf=RF_pulse(self.rfpulse.waveform,self.rfpulse.pulse_type,3.5*self.tp,self.rfpulse.f0,suppress_plots=True)
        self.assertAlmostEqual(self.rfpulse._tw1,newrf._tw1,3,'Time-w1max product not constant when pulse duration changes')
        # And then for gradient-modulated
        w1_point=np.flatnonzero(self.gm_pulse._w1_profile[2,:]<0)[0]
        self.assertAlmostEqual(self.gm_pulse.w1max,self.gm_pulse._f1vec[w1_point],3,'Power does not match _w1_profile value for 90 degree flip angle of gradient-modulated pulse')
        newrf=RF_pulse(self.gm_pulse.waveform,self.gm_pulse.pulse_type,3.5*self.tp,self.gm_pulse.f0,suppress_plots=True)
        self.assertAlmostEqual(self.gm_pulse._tw1,newrf._tw1,3,'Time-w1max product not constant when pulse duration changes')
        
    def test_tbw(self):
        # Test for sinc
        newrf=RF_pulse(self.rfpulse.waveform,self.rfpulse.pulse_type,3.5*self.tp,self.rfpulse.f0,suppress_plots=True)
        self.assertAlmostEqual(self.rfpulse._tbw,newrf._tbw,3,'Time-bandwidth product not constant when pulse duration changes')
        # Test for gradient-modulated. First check that .tbw returns string. Then check that ._tbw matches expected
        self.assertIsInstance(self.gm_pulse.tbw,str,'Gradient-modulated pulse should return string for .tbw')
        newrf=RF_pulse(self.gm_pulse.waveform,self.gm_pulse.pulse_type,3.5*self.tp,self.gm_pulse.f0,suppress_plots=True)
        self.assertAlmostEqual(self.gm_pulse._tbw,newrf._tbw,3,'Time-bandwidth product not constant when pulse duration changes')
        
    def test_tthk(self):
        self.assertEqual(self.gm_pulse.tthk,self.gm_pulse._tbw/1000,'Incorrect .tthk value for gm_pulse')
        self.assertIsInstance(self.rfpulse.tthk,str,'Non-gradient modulated pulse should return string for .tthk')
        
    def test_get_ampint(self):
        # Test that square wave returns 1 if it takes up full time
        basic_square=RF_pulse(np.concatenate([np.zeros([10,1]),np.ones([10,1])],axis=1),'exc',iscopy=True)
        basic_square.isAdiabatic=False
        self.assertEqual(basic_square.get_ampint(),1)
        # Test that sinc returns??? A number, I guess. Can just run the calculation with amplitude since it's not phase-modulated
        self.assertEqual(self.rfpulse.get_ampint(),np.sum(np.abs(self.rfpulse.waveform[:,1]))/self.rfpulse.npts,'Incorrect value for get_ampint() for sinc pulse')
        # Test that off-resonance ampint is the same as on-resonance (I think that should be true)
        self.assertAlmostEqual(self.rfpulse.get_ampint(), self.offres_pulse.get_ampint(),msg="Off-resonance and on-resonance results for get_ampint() don't match")
        # Test that adiabatic returns error if ignore_adiabatic=False
        with self.assertRaises(FidAException,msg="Adiabatic pulse should raise error for get_ampint()"):
            newrf=self.rfpulse.copy()
            newrf.isAdiabatic=True
            newrf.get_ampint()
            
    def test_get_complex_wf(self):
        def adjusted_angle(complex_x1):
            angle_wf=np.angle(complex_x1)
            angle_wf=np.where(angle_wf<0,angle_wf+2*np.pi,angle_wf)
            return angle_wf
        complex_wf=self.rfpulse.get_complex_wf()
        # Anything with magnitude zero is ambiguous for phase
        test_idx=np.flatnonzero(self.rfpulse.waveform[:,1])
        np.testing.assert_array_almost_equal(adjusted_angle(complex_wf[test_idx]),self.rfpulse.waveform[test_idx,0]*np.pi/180,err_msg='Phase of complex waveform incorrect')
        np.testing.assert_array_almost_equal(np.abs(complex_wf),self.rfpulse.waveform[:,1],err_msg='Magnitude of complex waveform incorrect')
        # The phases can be off by a factor of 2*pi and still match (not an issue above because no phase modulation)
        complex_wf=self.offres_pulse.get_complex_wf()
        test_idx=np.flatnonzero(self.offres_pulse.waveform[:,1])
        np.testing.assert_array_almost_equal(adjusted_angle(complex_wf[test_idx]),np.remainder(self.offres_pulse.waveform[test_idx,0],360)*np.pi/180,decimal=4,err_msg='Phase of complex waveform incorrect for off-resonance pulse')
        np.testing.assert_array_almost_equal(np.abs(complex_wf),self.offres_pulse.waveform[:,1],err_msg='Magnitude of complex waveform incorrect for off-resonance pulse')
        complex_wf=self.condensed_square.get_complex_wf(expanded=True)
        test_idx=np.flatnonzero(self.expanded_square.waveform[:,1])
        np.testing.assert_array_almost_equal(adjusted_angle(complex_wf[test_idx]),np.remainder(self.expanded_square.waveform[test_idx,0],360)*np.pi/180,err_msg='Phase of complex waveform incorrect for expanded waveform')
        np.testing.assert_array_almost_equal(np.abs(complex_wf),self.expanded_square.waveform[:,1],err_msg='Magnitude of complex waveform incorrect for expanded waveform')
        
    def test_add_phase(self):
        offres_copy=self.offres_pulse.copy()
        offres_copy.add_phase(45)
        offres_copy.add_phase(360-45)
        test_idx=np.flatnonzero(self.offres_pulse.waveform[:,1])
        np.testing.assert_array_almost_equal(offres_copy.waveform[:,1],self.offres_pulse.waveform[:,1],err_msg='RF_pulse.add_phase not producing expected result')
        np.testing.assert_array_almost_equal(np.remainder(offres_copy.waveform[test_idx,0],360),np.remainder(self.offres_pulse.waveform[test_idx,0],360),err_msg='RF_pulse.add_phase not producing expected result')
        
class TestRFPulseTools(unittest.TestCase):
    # I separated out the above tests of the RF pulse object (except for I did
    # test the shift f0 function because it aligned with .f0 tests but I can move
    # that here). In case I want to move them to the folder for fidA_common
    # where RF_pulse is located. This part will test the fidA_rfPulseTools 
    # functions, although setup is largely the same
    
    @classmethod
    def setUpClass(self):
        print('\nGenerating Hanning-windowed sinc pulse for testing as rfpulse')
        npts=100
        t=np.linspace(-2.5e-3,2.5e-3,npts)
        self.tvec=t-t[0]
        rfwf=np.zeros([npts,2])
        sinc_tmp=np.hanning(npts)*(np.sin(2*np.pi*600*t)/(t))
        rfwf[:,0]=(-1*np.sign(sinc_tmp)/2+0.5)*180
        rfwf[np.sign(sinc_tmp)==0,0]=0
        rfwf[:,1]=np.abs(sinc_tmp)
        self.tp=5e-3
        self.rfpulse=RF_pulse(rfwf,'exc',Tp=self.tp,f0=0,suppress_plots=True)
        
        # An off-resonance pulse
        self.offres_f0=1230
        print('\nGenerating off-resonance sinc pulse for testing as offres_pulse. f0={:.0f}'.format(self.offres_f0))
        rfwf[:,0]=rfwf[:,0]+(t-t[0])*self.offres_f0*360
        self.offres_pulse=RF_pulse(rfwf,self.rfpulse.pulse_type,Tp=self.tp,f0=self.offres_f0,suppress_plots=True)
        
        # A gradient-modulated pulse (constant gradient, so just narrows bandwidth)
        self.gradval=0.9
        print('\nGenerating gradient-modulated pulse as gm_pulse. Gradient {:3.2f} G/cm'.format(self.gradval))
        rfwf=np.ones([npts,4])
        sinc_tmp=np.hanning(npts)*(np.sin(2*np.pi*600*t)/(t))
        rfwf[:,0]=(-1*np.sign(sinc_tmp)/2+0.5)*180
        rfwf[np.sign(sinc_tmp)==0,0]=0
        rfwf[:,1]=np.abs(sinc_tmp)
        rfwf[:,3]=self.gradval*np.ones([npts,])
        self.gm_pulse=RF_pulse(rfwf,'exc',Tp=self.tp,f0=0,suppress_plots=True)
        
        #self.null_pulse=RF_pulse(np.concatenate([np.zeros([10,1]),np.zeros([10,1])],axis=1),'exc',iscopy=True)
        #self.null_pulse.isAdiabatic=False
        
    def test_rf_addGrad(self):
        # For scalar gradval
        newrf=rf_addGrad(self.rfpulse,self.gradval)
        self.assertAlmostEqual(newrf.bw,self.gm_pulse.bw,'rf_addGrad failed: bandwidth not as expected with scalar gradient')
        self.assertAlmostEqual(newrf.w1max,self.gm_pulse.w1max,'rf_addGrad failed: w1 not as expected with scalar gradient')
        # For vector gradval
        newrf=rf_addGrad(self.rfpulse,self.gradval*np.ones([self.rfpulse.npts,]))
        self.assertAlmostEqual(newrf._tbw,self.gm_pulse._tbw, 'rf_addGrad failed: bandwidth not as expected from vector gradient')
        self.assertAlmostEqual(newrf.w1max,self.gm_pulse.w1max,'rf_addGrad failed: w1 not as expected from vector gradient')
        grad_fac=2.5
        # Okay, so I don't really want to run tests that require user input. In
        # this case, I can raise the warning to an error to test that this case
        # gets to the right part of the program and I know that works, but then 
        # raising the error means that it exits before asking for input, so the 
        # tests can continue. This has the handy side effect of demonstrating 
        # to users how they can turn off or raise errors themselves.
        warnings.filterwarnings("error",message="WARNING: Input waveform ",category=FidAWarningRF)
        with self.assertRaises(FidAWarningRF):
            newrf=rf_addGrad(self.gm_pulse,self.gradval/grad_fac)
        warnings.resetwarnings()
        newrf=rf_addGrad(self.gm_pulse,self.gradval/grad_fac,overwrite_wf=True)
        self.assertAlmostEqual(self.rfpulse._tbw/(self.gradval/grad_fac*self.rfpulse.gamma)*10,newrf._tbw,3, 'rf_addGrad failed: Bandwidth not as expected.')
        # off-resonance should throw a warning because f0 will change
        with self.assertWarns(FidAWarningRF,msg='rf_addGrad: Adding gradient to off-resonance pulse should throw warning that f0 will change'):
            rf_addGrad(self.offres_pulse,self.gradval*np.ones([self.offres_pulse.npts,]))
    
    def test_rf_blochSim(self):
        # Very limited testing here. The BlochSimulator can have its own testing
        # in that testing folder. Here, I am just going to check a few zero/null
        # cases and a simple waveform.
        # Note that self.rfpulse was generated with RF_pulse, which calls the
        # BlochSimulator so, if there's something wrong with that, we may just
        # be replicating it here and getting two wrongs to agree. That's why a
        # separate test for the BlochSimulator itself would be good.
        npts=10000
        mv, fvec=rf_blochSim(self.rfpulse,self.tp*1000,fspan=10,f0=0,peakB1=self.rfpulse.w1max,ph=0,npts=npts,M0=np.r_[0,0,1],display_output=False)
        # This is a 90 degree pulse. On-resonance, around x, it should produce a vector along y
        np.testing.assert_array_almost_equal(mv[:,npts//2], np.r_[0,1,0],decimal=2,err_msg='rf_blochSim not working as expected properly for 90 degree sinc pulse')
        half_max_pt=np.flatnonzero(fvec>-1*self.rfpulse.bw/2)[0]
        # Very low precision for these, but I guess that's because w1max isn't exact and maybe fvec has issues as well.
        self.assertAlmostEqual(mv[2,half_max_pt],0.5,2,msg='Magnetization from rf_blochSim is not half maximum at expected bandwidth point for sinc pulse')
        mv, fvec=rf_blochSim(self.rfpulse,self.tp*1000,fspan=10,f0=0,peakB1=0,ph=0,npts=npts,M0=np.r_[0,0,1],display_output=False)
        np.testing.assert_array_almost_equal(mv[:,npts//2], np.r_[0,0,1],decimal=5,err_msg='rf_blochSim not working as expected properly for 0 power sinc pulse')
        
if __name__=='__main__':
    # can add verbosity=2 as an argument to unittest.main() for more info
    unittest.main()