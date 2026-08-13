#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:47:02 2026
pyFidA.fidA_common.FID.py

@author: Colleen Bailey (cbailey@sri.utoronto.ca), based on Matlab code by 
    Jamie Near

Contains the FID class, which holds data about MR spectra in both free 
inducation decay and spectral forms. This class is the pyFidA version of the
fid struct in Matlab's fid-A but with several convenient functions and 
properties that make it easier to use. See Matlab_differences_basic.md in 
the docs/explanation folder for more.

These objects can be processed using the functions in fidA_processing. This 
class has only been tested on proton data but is designed to work with data for 
other nuclei, provided that accurate nucleus and center frequency information 
are provided.

The class can also hold MRSI data but most processing functions in 
pyFidA.fidA_processing are not set up for or tested on MRSI data.
"""

from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from pyFidA.fidA_common import GAMMA_DICT, FidAException
from pyFidA.fidA_common import fid_from_specs, spec_from_fids

class FID(object):
    """
    A class to hold the free induction decay data from magnetic resonance
    spectroscopy, as well as to calculate and plot key aspects. The first 
    dimension of the data must be time. Other optional dimensions include 
    coils, averages and subspecs, as identified in FID.dims
    
    Parameters
    ----------
    fids : numpy array
        free induction decay data from MRS, with time as the first dimension.
    spectralwidth : float
        Width of the spectral data in Hz, equivalent to 1/dwelltime.
    txfreq : float
        Center frequency in MHz.
    te : float, optional
        Echo time of the acquisition sequence in ms (only used for simulated
        data and writing to some file formats; not used in processing). The 
        default is None.
    tr : float, optional
        Repetition time of the acquisition sequency in ms (for information and
        writing to some file formats; not used in processing). The default is 
        None.
    sequence : string, optional
        Type of sequence (only used for simulated data). The default is None.
    flags : dict, optional
        dict with boolean values specifying which processing operations have 
        already been done on the data. The default is None, which returns the 
        default dict, where all flag values are False.
    dims : list, optional
        list of the dimensions of the data in fids (in order). Note that the
        dimensions are input as a list, even though FID.dims is a read-only
        dictionary in the object itself. The default is ['t'].
    hdr : nifti_mrs.NIFTI_MRS header, optional
        header portion of NIFT_MRS object containing sequence info (only needed
        to hold this info for cases where object will be written to file in 
        nifti_mrs format). The default is None.
    hdr_ext : nifti_mrs.hdr_ext.Hdr_Ext, optional
        Header extension of NIFT_MRS object containing sequence info (only 
        needed to hold this info for cases where object will be written to file 
        in nifti_mrs format). The default is None.
    nucleus : list of strings, optional
        Nucleus for which to get the gyromatic ratio from GAMMA_DICT. Formatted
        as a list for future 2D spectroscopy use but currently only the first 
        element is used. The default is ['1H'].
    center_freq_ppm : float, optional
        Center frequency of the spectrum, used to generate the ppm vector 
        corresponding to the data. The default is 4.65.
    copy_rawsub : int, optional
        If copy_rawsub is 0, FID._rawSubspecs is set to the size of the 
        subspecs dimension in the input array fids (or 1 if this dimension
        doesn't exist). For any other int, FID._rawSubspecs is set to the value
        of copy_rawsub. This allows FID._rawSubspecs to be set when a new 
        spectrum is loaded (by using the default value of 0) and then this
        value can be preserved during processing even though the result of each
        processing step is a new instance of the FID object (because FID.copy()
        will set copy_rawsub to old_fid._rawSubspecs)
    copy_rawavg : ing, optional
        If copy_rawavg is 0, FID._rawAverages is set to the size of the 
        averages dimension in the input array fids (or 1 if this dimension
        doesn't exist). For any other int, FID._rawAverages is set to the value
        of copy_rawavg. This allows FID._rawAverages to be set when a new 
        spectrum is loaded (by using the default value of 0) and then this
        value can be preserved during processing even though the result of each
        processing step is a new instance of the FID object (because FID.copy()
        will set copy_rawavg to old_fid._rawAverages)

    Other Properties and Functions of interest
    ----------------------------------------------
    specs : numpy array
        Property that returns the inverse Fourier transform of self.fids as a
        getter. When the corresponding setter is called, the provided spectrum
        is Fourier transformed and the result saved in self.fids.
    dims : read-only Dotdict
        Turns the list in self._dimlist into a set of key,value pairs where the
        key is the dimension name and the value is its list index, which 
        corresponds to that dimension of self.fids. The elements can be called
        in dict format, eg. self.dims['t'], or dot format, eg. self.dims.t, but
        are read-only. To alter a value, the underlying self._dimlist needs to 
        be changed
    averages : read-only int
        Returns the size of self.fids in the averages dimension. If there is no
        averages dimension, then 1 is returned.
    subspecs : read-only int
        Returns the size of self.fids in the subspecs dimension. If there is no
        subspecs dimension, then 1 is returned.
    rawAverages : read-only int
        Number of averages in the initial FID, before processing. If there is
        no averages dimension then 1 is returned.
    rawSubspecs : read-only int
        Number of subspectra in the initial FID, before processing. If there is
        no subspecs dimension then 1 is returned.
    GAMMA : float
        Returns the value in self._GAMMA, which is typically the gyromagnetic 
        ratio corresponding to self.nucleus in GAMMA_DICT, in Hz/T. There is a 
        setter so that the user can set their own GAMMA value if there is no 
        value in GAMMA_DICT.
    Bo : float
        Magnetic field strength in T, calculated from self.txfreq and 
        self.GAMMA. (The setter sets self.txfreq based on self.GAMMA and the
        user-provided Bo value)
    spectralwidthppm : float
        Spectral width in ppm, calculated from self.spectralwidth and 
        self.GAMMA. (The setter sets self.spectralwidth based on self.GAMMA and 
        the spectral width in ppm provided by the user)
    dwelltime : float
        Dwell time in seconds. Calculated from self.spectralwidth, calculated 
        from self.spectralwidth. (The setter sets self.spectralwidth a dwell 
        time value provided by the user)
    ppm : read-only numpy array
        ppm vector of the frequencies corresponding to the points of 
        self.specs.
    t : read-only numpy array
        time vector of the time points corresponding to the first dimension of
        self.fids.
    sz : tuple
        Size of all dimensions of the data, self.fids.shape.
    isMRSI : read-only boolean
        Is the data MRSI data? Determine from whether any of 'x', 'y' or 'z' 
        are in self._dimlist.
    ndim : read-only int
        Number of dimensions the data has, len(self.sz)
    sim : None OR str
        For simulated spectra, either 'ideal' (for spectra simulated with ideal
        RF pulses) or 'shaped' (for spectra simulated with shaped RF pulses).
    date : datetime date
        Date at which the FID instance was generated
    plot_spec : function
        Plot the spectra for this data. If there are more than 2 dimensions, 
        separate spectra corresponding to the second dimension will be plotted,
        using just the first index for any remaining dimensions.
    
    """
    def __init__(self,fids,spectralwidth,txfreq,te=None,tr=None,sequence=None,flags=None,dims=['t'],hdr=None,hdr_ext=None,nucleus=['1H'],center_freq_ppm=4.65,copy_rawsub=0,copy_rawavg=0):
        self.fids=fids
        if 't' not in dims or dims.index('t')!=0:
            raise FidAException('ERROR: dims for FID must always contain t as first dimension.')
        if fids.ndim!=len(dims):
            raise FidAException('ERROR: length of dims ({:d}) does not match array size of fids ({:d})'.format(len(dims),fids.ndim))
        self._dimlist=dims
        self.spectralwidth=spectralwidth
        self.nii_mrs=dict()
        self.nii_mrs['hdr']=hdr
        self.nii_mrs['hdr_ext']=hdr_ext
        self.nucleus=nucleus
        self.txfreq=txfreq
        self._GAMMA=GAMMA_DICT[self.nucleus[0].upper()]*1e6
        self.te=te
        self.tr=tr
        self.sequence=sequence
        self.sim=None
        self.date=date.today()
        if copy_rawavg:
            self._rawAverages=copy_rawavg
        else:
            try:
                self._rawAverages=fids.shape[dims.index('averages')]
            except ValueError:
                self._rawAverages=1
        if copy_rawsub:
            self._rawSubspecs=copy_rawsub
        else:
            try:
                self._rawSubspecs=fids.shape[dims.index('subspecs')]
            except ValueError:
                self._rawSubspecs=1
        self.added_ph0=0
        self.added_ph1=0
        self.center_freq_ppm=center_freq_ppm
        if flags:
            self.flags=flags.copy()
        else:
            self.flags=_get_default_flag_dict()
            if 'coils' not in dims or fids.shape[dims.index('coils')]==1:
                self.flags['addedrcvrs']=True
            if 'averages' not in dims or fids.shape[dims.index('averages')]==1:
                self.flags['averaged']=True
            if 'subspecs' not in dims or fids.shape[dims.index('subspecs')]==1:
                self.flags['subtracted']=True
            if 'subspecs' in dims and fids.shape[dims.index('subspecs')]==4:
                self.flags['isFourSteps']=True
    @property
    def dims(self):
        dimdict=dict()
        for idx,dimval in enumerate(self._dimlist):
            dimdict[dimval]=idx
        return Dotdict(dimdict)
    @property
    def specs(self):
        return spec_from_fids(self.fids)
    @specs.setter
    def specs(self,newspec):
        self.fids=fid_from_specs(newspec)
    @property
    def rawSubspecs(self):
        return self._rawSubspecs
    @property
    def subspecs(self):
        if 'subspecs' not in self._dimlist:
            return 1
        else:
            return self.sz[self._dimlist.index('subspecs')]
    @property
    def rawAverages(self):
        return self._rawAverages
    @property
    def averages(self):
        if 'averages' not in self._dimlist:
            return 1
        else:
            return self.sz[self._dimlist.index('averages')]
    @property
    def GAMMA(self):
        return self._GAMMA
    @GAMMA.setter
    def GAMMA(self,newGAMMA):
        """
        GAMMA will initially be set based on the nucleus provided at 
        initialization (1H by default). However, the value can alteraed 
        afterward, if needed.

        """
        self._GAMMA=newGAMMA
    @property
    def Bo(self):
        return self.txfreq/self.GAMMA
    @Bo.setter
    def Bo(self,newBo):
        self.txfreq=newBo*self.GAMMA
    @property
    def spectralwidthppm(self):
        return self.spectralwidth/(self.txfreq/1e6)
    @spectralwidthppm.setter
    def spectralwidthppm(self,new_sw_ppm):
        self.spectralwidth=new_sw_ppm*(self.txfreq/1e6)
    @property
    def _ppmmin(self):
        return self.center_freq_ppm+self.spectralwidthppm/2
    @property
    def _ppmmax(self):
        return self.center_freq_ppm-self.spectralwidthppm/2
    @property
    def ppm(self):
        return np.linspace(self._ppmmin,self._ppmmax,self.fids.shape[0])
    @property
    def dwelltime(self):
        return 1/self.spectralwidth
    @dwelltime.setter
    def dwelltime(self,newdwell):
        self.spectralwidth=1/newdwell
    @property
    def t(self):
        t2=np.linspace(0,(self.fids.shape[0]-1)*self.dwelltime,self.fids.shape[0])
        return t2
    @property
    def sz(self):
        return self.fids.shape
    @property
    def isMRSI(self):
        if 'x' in self or 'y' in self or 'z' in self:
            return True
        else:
            return False
    @property
    def ndim(self):
        return len(self.sz)
    def __add__(self,add1):
        """
        eg. new_fid = old_fid + add1
        When used with a FID object, adds the two fids properties together. 
        When used with a scalar, it is assumed that the intention is to add
        a DC offset to the frequency-domain spectrum.
        Outputs a new FID object with the altered fids/specs properties.

        """
        out1=self.copy()
        if isinstance(add1, FID):
            out1.fids=self.fids+add1.fids
        elif np.isscalar(add1):
            out1.specs=self.specs+add1
        else:
            raise TypeError(f"sorry, don't know how to add {type(add1).__name__}")
        return out1
    def __sub__(self,sub1):
        """
        eg. new_fid = old_fid - sub1
        When used with a FID object, subtracts sub1.fids from self.fids. When 
        used with a scalar, it is assumed that the intention is to subtract
        a DC offset from the frequency-domain spectrum.
        Outputs a new FID object with the altered fids/specs properties.

        """
        out1=self.copy()
        if isinstance(sub1, FID):
            out1.fids=self.fids-sub1.fids
        elif np.isscalar(sub1):
            out1.specs=self.specs-sub1
        else:
            raise TypeError(f"sorry, don't know how to add {type(sub1).__name__}")
        return out1
    def __mul__(self,mult1):
        out1=self.copy()
        if isinstance(mult1, FID):
            out1.fids=self.fids*mult1.fids
        elif np.isscalar(mult1):
            out1.fids=self.fids*mult1
        else:
            raise TypeError(f"sorry, don't know how to multiply by {type(mult1).__name__}")
        return out1
    def __rmul__(self,mult1):
        out1=self.copy()
        if np.isscalar(mult1):
            out1.fids=self.fids*mult1
        else:
            raise TypeError(f"sorry, don't know how to multiply by {type(mult1).__name__}")
        return out1
    def __div__(self,mult1):
        out1=self.copy()
        if isinstance(mult1, FID):
            out1.fids=self.fids/mult1.fids
        elif isinstance(mult1, int) or isinstance(mult1, float):
            out1.fids=self.fids/mult1
        else:
            raise TypeError(f"sorry, don't know how to multiply by {type(mult1).__name__}")
        return out1
    def __truediv__(self,mult1):
        return self.__div__(mult1)
    def __repr__(self):
        return '{:s} has fids size {:s} and dimensions: {:s}'.format(self.__class__.__name__,str(self.sz),', '.join(self._dimlist))
    def __contains__(self,item):
        return item in self._dimlist
    def __getitem__(self, key):
        """
        Returns a new FID object with a portion of self.fids as the new fid. 
        The corresponding dimensions in self._dimlist are also adjusted.
        
        eg. part_fid = old_fid[:1000,:] OR part_fid=old_fid[:,::10] OR 
        part_fid=old_fid[:,0]. Can also be called using slice objects instead
        of slicing syntax: 
        part_fid=old_fid[tuple([slice(None),slice(0,None,10)])]

        """
        if len(key)!=self.fids.ndim:
            raise FidAException("ERROR: __getitem__ only implemented for case where every dimension is sliced explicitly. \n Use ':' or 'slice(None)' to indicate dimensions not being sliced. \n key length={:d},  number fid dimensions={:d}".format(len(key),self.fids.ndim))
        outdat=self.copy()
        outdat.fids=outdat.fids[key]
        # Remove any singleton dimensions
        new_dimlist=[dimnm for idx,dimnm in enumerate(self._dimlist) if type(key[idx]) is not int]
        outdat._dimlist=new_dimlist
        return outdat
    def __setitem__(self, key, newfid):
        """
        Sets part of self.fids (part defined by key) to a new numpy array or
        the fids property of another FID object, newfid.
        
        eg. myfid[:,0]=new_first_average
        Note that new_first_average is not checked against key so an error will
        be thrown if these do not match and it shouldn't be possible to change
        the dimensions of myfid.fids, so no changes are made to myfid._dimlist.

        """
        # newfid can be either and ndarray to set fids or a FID object.
        if isinstance(newfid,FID):
            newfid=newfid.fids
        self.fids[key]=newfid
    def copy(self):
        if self.nii_mrs['hdr'] is None:
            new_nii_mrs_hdr=None
        else:
            new_nii_mrs_hdr=self.nii_mrs['hdr'].copy()
        if self.nii_mrs['hdr_ext'] is None:
            new_nii_mrs_hdr_ext=None
        else:
            new_nii_mrs_hdr_ext=self.nii_mrs['hdr_ext'].copy()
        newfid=FID(self.fids.copy(),self.spectralwidth,self.txfreq,
                   self.te,self.tr,self.sequence,self.flags.copy(),
                   self._dimlist.copy(),new_nii_mrs_hdr,new_nii_mrs_hdr_ext,
                   self.nucleus,self.center_freq_ppm,
                   copy_rawsub=self._rawSubspecs,copy_rawavg=self._rawAverages)
        newfid.date=date.today()
        return newfid
    def plot_spec(self,xlims=[4.5,0],xlab='Chemical Shift (ppm)',ylab='Signal',title='',plotax=None, **kwargs):
        """
        Plot the spectrum in self.specs. If there are more than 2 dimensions,
        only the first 2 dimensions will be plotted (using the first index for
        any remaining dimensions).

        Parameters
        ----------
        xlims : 2-element list, optional
            Bounds of the x-axis, from left to right. The default is [4.5,0].
        xlab : str, optional
            Label for the x-axis. The default is 'Chemical Shift (ppm)'.
        ylab : str, optional
            Label for the y-axis. The default is 'Signal'.
        title : str, optional
            String to include as plot title. The default is ''.
        plotax : matplotlib.axes._subplots.AxesSubplot, optional
            Axis on which to create the graph. The default is None, which 
            generates a new figure and axis.
        **kwargs : dict, optional
            Dictionary of additional keyword arguments to pass to 
            matplotlib.pyplot.plot.

        Returns
        -------
        plotax : matplotlib.axes._subplots.AxesSubplot
            Axis object on which the plot was created.

        """
        if plotax is None:
            [f1,plotax]=plt.subplots(1,1)
        if self.fids.ndim>2:
            print("More than 2 dimensions. Plotting first 2 dimensions ({:s}, {:s}) only.".format(self._dimlist[0],self._dimlist[1]))
            whichslice=[0]*self.ndim
            whichslice[0]=slice(None)
            whichslice[1]=slice(None)
            spec_for_plot=np.real(self.specs[tuple(whichslice)])
        else:
            spec_for_plot=np.real(self.specs)
        plotax.plot(self.ppm,spec_for_plot,**kwargs)
        plotax.set_xlim(xlims)
        plotax.set_xlabel(xlab)
        plotax.set_ylabel(ylab)
        plotax.set_title(title)
        return plotax
        
def _get_default_flag_dict():
    flag_dict={'filtered': False,
     'zeropadded': False,
     'freqcorrected': False,
     'phasecorrected': False,
     'freqranged': False,
     'subtracted': False,
     'downsampled': False,
     'isFourSteps': False,
     'leftshifted': False,
     'averaged': False,
     'addedrcvrs': False}
    return flag_dict

def _readonly_dims(self,*args,**kwargs):
    raise FidAException('FID.dims is read-only and cannot be modified. If dimensions of the FID have changed, alter the underlying list FID._dimlist.')
    
class Dotdict(dict):
    """
    Alter the dict class so that it will return values with dot notation. This
    allows calls to check the dims similar to Matlab.
    eg. myfid.dims.t is equivalent to myfid.dims['t']
    Other methods that alter values within the dict (by dot notation or 
    otherwise) are removed to make it read-only.
    """
    __getattr__=dict.__getitem__
    __setattr__=_readonly_dims
    __setitem__=_readonly_dims
    __delattr__=_readonly_dims
    __delitem__=_readonly_dims
    pop=_readonly_dims
    popitem=_readonly_dims
    clear=_readonly_dims
    update=_readonly_dims
    setdefault=_readonly_dims