#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:08:23 2026

@author: nearlabmacbook1
"""
import unittest
import numpy as np
from pyFidA.fidA_common import FID, FidAException

# Test normal inputs (standard values that the function expects to receive)
# Test edge cases (zero, emtpy lists, very large numbers)
# Error Handling (test how code reacts to bad data or missing values)
# State changes (test if the object or system updates correctly after running code)
# You don't have to write for every case but try to focus on things that you think
# realistically might break (eg. I would think that most processing functions
# should check that the size of myfid.fids and length of myfid._dimlist match)

# The major unittest assertions are assertEqual, assertNotEqual, assertTrue,
# assertFalse, assertIs/IsNot, assertIsNone/IsNotNone, assertIn/NotIn, 
# assertIsInstance/NotIsInstance and assertRaises to see that a particular exception gets raised
# Note that numpy has its own testing support to test arrays, including
# np.testing.assert_allclose, assert_array_equal, etc. (https://numpy.org/doc/stable/reference/routines.testing.html)


def make_simulated_fidobj(dt=250e-6,txfreq=300e6,npts=2048,peak_pos_ppm=2,center_freq_ppm=4.65,T2=0.1):
    t=np.r_[0:npts*dt:dt]
    rel_freq=(center_freq_ppm-peak_pos_ppm)*txfreq/1e6
    fid1=make_peak(t,rel_freq,T2)
    # Just making an FID with a single dimension here
    fidobj=FID(fid1,1/dt,txfreq,dims=['t'],center_freq_ppm=center_freq_ppm)
    return fidobj

def make_peak(t,rel_freq,T2):
    fid1=1*np.exp(-1j*2*np.pi*rel_freq*t)*np.exp(-t/T2)
    return fid1

def make_fidobj_multidim(dt=250e-6,txfreq=300e6,npts=2048,peak_pos_ppm=2,center_freq_ppm=4.65,T2=0.1,n_av=250,n_coil=2):
    # Debatable exactly what to do here, but going to try to make some fake averages and coils
    t=np.r_[0:npts*dt:dt]
    rel_freq=(center_freq_ppm-peak_pos_ppm)*txfreq/1e6
    fidtmp=np.zeros([npts,n_av,n_coil],dtype=complex)
    # Need to do something better for coil combos I think
    coil_amps=[0.7,0.9]
    coil_phases=[0,30*np.pi/180]
    noise_est=0.01
    for coilct in range(n_coil):
        sig1=coil_amps[coilct]*(make_peak(t,rel_freq,T2)*np.exp(-1j*coil_phases[coilct]))
        for avct in range(n_av):
            fidtmp[:,avct,coilct]=sig1+noise_est*np.random.randn(npts)+1j*noise_est*np.random.randn(npts)
    fidobj=FID(fidtmp,1/dt,txfreq,dims=['t','averages','coils'],center_freq_ppm=center_freq_ppm)
    return fidobj

class TestFID(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        # making test objects for both 1D FID case (time dimension only) and 
        # multi-dimensional case with 250 averages and 2 coils
        self.myfid=make_simulated_fidobj()
        self.second_fid=make_simulated_fidobj(peak_pos_ppm=3.5,T2=0.08)
        self.n_av=250
        self.n_coil=2
        self.multidim_fid=make_fidobj_multidim(n_av=self.n_av,n_coil=self.n_coil)
    
    def test_fid(self):
        self.assertIsInstance(self.myfid, FID, 'Object returned is not instance of FID')
        self.assertIn('t',self.myfid,'Could not find dimension "t" in FID')
        self.assertNotIn('averages',self.myfid,'Averages dimension incorrectly added to FID')
        self.assertEqual(self.myfid.dims,{'t':0},'FID.dims not correctly generated')
        self.assertEqual(len(self.myfid.t),self.myfid.sz[0])
        self.assertIsInstance(self.multidim_fid,FID,'Object returned is not instance of FID')
        self.assertIn('t',self.multidim_fid,'Could not find dimension "t" in FID')
        self.assertIn('averages',self.multidim_fid,'Could not find dimension "averages" in FID')
        self.assertEqual(self.multidim_fid.dims,{'t':0,'averages':1,'coils':2},'FID.dims not correctly generated')
        self.assertEqual(len(self.multidim_fid.t),self.multidim_fid.sz[0])
        self.assertEqual(self.multidim_fid.ndim,3)
        # Could be worth trying to generate a new FID with some "extra" parameters
        # to ensure that all aspects of the FID input arguments are working. This
        # example that I have for shifted center frequency might be better in
        # test_properties but I don't mind it here either.
        tmpfid=self.myfid.fids.copy()
        newfid=FID(tmpfid,self.myfid.spectralwidth,self.myfid.txfreq,dims=['t'],center_freq_ppm=5.65)
        self.assertAlmostEqual(self.myfid.ppm[np.argmax(self.myfid.specs)],newfid.ppm[np.argmax(newfid.specs)]-1)
        # Test for assigning too many dimensions. Note that, if it raises an Exception, this test will PASS
        # We would only expect a fail if no exception were raised here
        with self.assertRaises(FidAException):
            newfid=FID(tmpfid,self.myfid.spectralwidth,self.myfid.txfreq,dims=['t','averages'])
        with self.assertRaises(FidAException):
            newfid=FID(self.multidim_fid.fids,self.multidim_fid.spectralwidth,self.multidim_fid.txfreq,dims=['t'])
        # Lots that you could test, like that putting in twice the number of points
        # gets you a time vector that's twice as long. And halving the dt gets
        # you double the resolution (I think??)
        
    def test_properties(self):
        # Already did a bit above with dims but anyway
        self.assertEqual(self.myfid.dwelltime,1/self.myfid.spectralwidth)
        # Want to test a bunch of others. Doesn't just have to be assertEqual either. Should try to fail some tests? Or I guess pass them but with notIn, IsNot, etc.
        # Not sure best way to check specs as fft of fids, but I think it's 
        # important to check both in the basic object and that some of the basic
        # functions work (eg. if you double fids then specs should also double).
        # although I guess that a lot of that can be left until the processing toolbox
        # but can do some here.
        
    def test_size(self):
        self.assertEqual(self.myfid.sz, self.myfid.fids.shape, 'Object size is not correct')
        self.assertEqual(self.multidim_fid.sz, self.multidim_fid.fids.shape, 'Object size is not correct')
        # Just as an example, trying to fail a test. Also testing whether you can do multiple assertions in one test
        #self.assertEqual(self.myfid.sz, 12, 'Object size is not correct')
        
    def test_dims(self):
        self.assertEqual(len(self.myfid._dimlist),len(self.myfid.fids.shape), 'Number of dimensions in dimlist does not match fid size')
        self.assertEqual(len(self.multidim_fid._dimlist),len(self.multidim_fid.fids.shape), 'Number of dimensions in dimlist does not match fid size')
        self.assertTrue(all([self.multidim_fid.dims[kval]==self.multidim_fid._dimlist.index(kval) for kval in self.multidim_fid._dimlist]))
        # I keep forgetting to add error messages for these things
        self.assertEqual(self.multidim_fid.sz[1],self.n_av)
        self.assertEqual(self.multidim_fid.sz[2],self.n_coil)
        # Not sure how assertRaises works. Maybe only works on functions?
        #self.assertRaises(IndexError,self.multidim_fid._dimlist[4])
        #self.assertRaises(KeyError,self.myfid.dims['averages'])
        self.assertEqual(self.multidim_fid.dims['averages'],1)
        self.assertEqual(self.multidim_fid.dims.averages,1)
        
    def test_math(self):
        # Could just do all of the math testing here, if that makes sense
        self.assertIsInstance(2*self.myfid,FID)
        np.testing.assert_array_equal((self.myfid+self.second_fid).fids, self.myfid.fids+self.second_fid.fids)
        np.testing.assert_array_equal((3.6*self.myfid).fids, (self.myfid*3.6).fids)
        
    def test_copy(self):
        # Basically just want to test that a bunch of the properties are equal
        # to one another but the instances are distinct. And then the dates are
        # different but I guess the rawAverages should match?? Probably not
        # vital to check the rawAverages stuff
        pass
        
class TestRF(unittest.TestCase):
    # Basically need a whole new setup and set of tests here
    pass
        
if __name__=='__main__':
    # can add verbosity=2 as an argument to unittest.main() for more info
    unittest.main()