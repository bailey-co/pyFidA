#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:12:32 2026
pyFidA.fidA_common.RF_pulse.py

@author: Colleen Bailey (cbailey@sri.utoronto.ca), based on Matlab code by 
    Jamie Near

Contains the RF_pulse class, which combines information about the rf waveform,
pulse duration, offset frequency, etc. and can calculate information about the
power required to achieve a flip angle, the frequency profile, etc.

Also in this module are several functions that are evaluated when a new RF_pulse
instance is created but that are useful at some other points and so are available
separately. See the individual docstrings for these functions for more info:
    * _calc_tw1max
    * _calc_tbw
    * estimate_f0

Currently, the RF_pulse object is used in the pyFidA.fidA_rf_pulseTools module 
and in pyFidA_io.io_rf.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyFidA.fidA_common import GAMMA_DICT, BlochSimulator, FidAWarning, FidAException
import warnings

class FidAWarningRF(FidAWarning):
    """
    Sub-classing FidAWarning for the RF toolbox in case people just want to turn
    off the RF warnings (or just want to turn off other fid-A warnings and leave
    the warnings from the RF toolbox on). The RF warnings will be thrown in a
    number of "normal" cases, like when dual-band pulses are generated or when
    users want to look at off-resonance behaviour
    """
    pass
    
class RF_pulse(object):
    """
    A class to hold rf pulse information, as well as to calculate and plot
    key variables.
    Note that the default f0 value is assumed to be 0 (on-resonance) but it is
    possible to enter an off-resonance waveform whose resonance frequency does 
    not match f0. If the calculation of the pulse power fails or the frequency
    profile is not as expected, consider running with f0=None to find the
    resonance frequency.

    Parameters
    ----------
    waveform : (n x 2) or (n x 3) or (n x 4) array
        waveform where n is the number of time points in the rf pulse, the first 
        column [:,0] contains the rf pulse phase in degrees, the second 
        column [:,1] contains the rf amplitude (will be scaled), and the third 
        column [:,2] contains the relative duration of each time step. An n x 2 
        array can be entered and the third column will default to ones. The 
        fourth column [:,4] is only used for gradient-modulated pulses and 
        contains the gradient amplitude in G/cm at each time point.
    pulse_type : str or float
        The type of RF pulse. Options are 'exc' for an excitation pulse, 
        'inv' for an inversion pulse, and 'ref' for a refocusing pulse. It is
        also possible to enter a numeric value corresponding to the desired
        flip angle in degrees.
    Tp : float , optional
        The duration of the rf pulse in seconds. Used for initial calculation 
        of the time-w1max product (RF_pulse._tw1) and the time-bandwidth 
        product (RF_pulse._tbw). These calculations are done on initialization
        using Bloch simulations. The w1 power needed to achieve the desired 
        flip angle for other pulse lengths and the bandwidth for those pulse 
        lengths can then be found by simple division of the tw1 and tbw values. 
        If no value is entered, the calculation is done for Tp=0.005 s.
    f0 : float or list of floats, optional
        The frequency offset of the pulse in Hz, used in the Bloch simulations 
        to find the w1max for that frequency. Incorrect f0 values will result 
        in incorrect (or failed) w1max estimates and strange bandwidth plots. A 
        warning will be given if the frequency at the center of the bandwidth 
        estimated from the Bloch simulations differs significantly from the 
        entered f0, which may indicate that f0 is incorrect. Users who do not 
        know the frequency of their waveform can set f0 = None to estiamte f0 
        from Bloch simulations but this is a time-intensive process
        that iterates through w1 values. The default f0 is 0 (assumed 
        on-resonance pulse). Note that a dual-band pulse is not well-described
        by a single f0 value. A list of floats can be entered in this case but
        many rfPulseTools are not yet designed for dual-band pulses and will 
        run calculations for just the first entry in the list.
    iscopy : boolean, optional
        Generate an RF_pulse instance without running the tw1 and tbw 
        calculations. This is useful when making a copy of an existing RF_pulse 
        instance where these values can just be copied over (as seen in
        self.copy()). The default is False.
    gamma : float, optional
        The gyromagnetic ratio of the nucleus the pulse is intended to be used
        for. Some rfPulseTools functions require this to convert between 
        frequencies and gradient strengths. The default is the gyromagnetic 
        ratio for hydrogen, 42.577 MHz/T.
    suppress_plots : boolean, optional
        Determines whether the Mz-vs-w1 and Mz-vs-frequency (bandwidth) plots 
        are output, as well as some text. In general, it is a good idea to 
        check these plots to ensure that w1max is correctly estimated and that
        the f0 value is correct. However, when generating large numbers
        of pulses or pulses that have been used before, it may be convenient
        to suppress these outputs. The default is False.
        
    Other Properties and Functions of interest
    ----------------------------------------------
    f0 : The centre frequency of the pulse. When called as a getter, my_fid.f0
        returns the value stored in my_fid._f0, which is set to the f0 value 
        entered as an input argument on __init__. When called as a setter 
        (my_rf.f0=newval), creates a phase ramp that shifts the existing 
        waveform to the offset frequency newval. Saves this as the new offset 
        frequency (self._f0) and shifts the frequency vector used in the 
        bandwidth plot to reflect the new frequency offset.
    isGM : boolean property that checks whether a gradient component of the 
        waveform exists (and is non-zero if it does exist). ie. is the pulse
        gradient-modulated?
    isPhsMod : boolean property that checks whether the phase values in the
        waveform are all 0 or 180. ie. is the pulse phase-modulated?
    isAdiabatic : boolean property that (by default) checks whether the Mz 
        values between w1max and 1.5*w1max are all within 10% of the Mz value 
        at w1max. This is a somewhat crude estimate of whether a pulse is 
        adiabatic and can be overwritten by the user. e.g. set 
        my_rf.isAdiabatic=True and this value will then be saved for future 
        checks instead of using the w1 profile.
    rfCentre : a number between 0 and 1 that indicates where the peak of the 
        RF amplitude waveform occurs. If rfCentre = 0.5, then the pulse is 
        symmetric. If rfCentre<0.5 then the peak occurs near the beginning of 
        the pulse (i.e. a max phase pulse). If rfCentre>0.5, then the peak 
        occurs near the end of the pulse (i.e. min phase pulse). Used in 
        sim_steam_shaped to determine how to time-reverse the rf pulse.
    get_tvec() : returns a vector of the time values for the waveform. Useful in 
        cases where the time steps are not uniform.
    get_ampint() : calculation of the amplitude integral used by Siemens to 
        scale the amplifier power.
    get_expanded_wf() : a version of the rf waveform with even timesteps. Some
        rfPulseTools in Matlab fid-A assume even timesteps without checking that
        this is the case. Replacing the waveform with the expanded waveform
        allows these functions to return the correct values. (The expanded
        waveform will take longer for any function that re-calculates matrix 
        exponentials or matrix multiplications for each timestep and those 
        functions should instead be re-written to account for non-uniform 
        timesteps)
    get_complex_wf() : return the complex waveform from the phase and amplitude
        in my_rf.waveform. The complex waveform simplifies several calculations.
        The optional expanded=True input argument (default is False) can be 
        used to get the expanded complex waveform with even time steps.
    w1max : The w1max value in kHz that is needed to achieve the flip angle 
        described by pulse_type for the pulse length in _Tp. This value is 
        based on the number stored in the private variable my_rf._tw1, which is 
        first set when the RF_pulse is initialized, using a plot of Mz vs w1 at 
        the frequency my_rf._f0. If this estimate is incorrectly calculated on
        __init__, this property has a setter so that you can run my_rf.w1max=3 
        (in kHz) and this will save a new time-w1max product based on the 
        existing pulse length my_rf._Tp. The time-bandwidth product will then 
        be re-estimated at the new w1max.
    bw : For non-gradient-modulated pulses, the bandwidth of a pulse in kHz at 
        the pulse length stored in _Tp. If the pulse is gradient-modulated, the 
        value returned is the thickness in cm. The time-bandwidth product is 
        stored in the private variable _tbw. There is no setter for the 
        bandwidth property.
    tbw : the time-bandwidth product in kHz*ms, only available for non-
        gradient-modulated pulses.
    tthk : the time-slice thickness product in cm*s, only available for 
        gradient-modulated pulses.
    add_phase : adds a constant phase to the RF_pulse's phase. Used in 
        generating some dual-band pulses.
    plot_w1_profile : plots the magnetization vs w1 (based on values saved 
        during the Bloch simulation at initialization unless a re-calculation
        is forced). By default, the z-magnetization is plotted but other
        directions can be shown. Useful if w1max estimate failed, in order to 
        gain more information about potential issues.
    plot_freq_profile : plots the magnetization vs frequency (based on values 
        saved during the Bloch simulation at initialization unless a 
        re-calculation is forced). By default, the z-magnetization is plotted 
        but other directions can be shown. Useful if w1max estimate failed, in
        order to check if the f0 estimate is incorrect.
    plot_onres : plots the Mz magnetization over time (as the RF pulse plays
        out) at the frequency my_rf._f0 and power my_rf.w1max
    """
    def __init__(self,waveform,pulse_type,Tp=0.005,f0=0,iscopy=False,gamma=GAMMA_DICT['1H'],suppress_plots=False):
        # If there are any phase discontinuities equal to a 360 degree jump, 
        # remove them (preserves phase ramps that create off-resonance pulses 
        # and also needed for rf_resample to work properly)
        if waveform.ndim<2 or waveform.shape[1]<2:
            raise FidAException('ERROR: input waveform must have shape n x 2, n x 3 or n x 4.')
        # Scale the amplitude so that the max value is 1.
        waveform[:,1]=waveform[:,1]/np.amax(np.abs(waveform[:,1]))
        jumps=np.diff(waveform[:,0])
        jumpIdx=np.flatnonzero((np.abs(jumps)>355)*(np.abs(jumps)<365))
        for nct,jumpct in enumerate(jumpIdx):
            waveform[jumpct+1:,0]=waveform[jumpct+1:,0]-360*np.sign(jumps[jumpct])
        if waveform.shape[1]==4: # gradient-modulated case
            self._waveform=waveform
        elif waveform.shape[1]==3:
            self._waveform=waveform
        elif waveform.shape[1]==2:
            self._waveform=np.concatenate((waveform,np.ones([waveform.shape[0],1])),axis=1)
        else:
            raise FidAException('ERROR: input waveform must have shape n x 2, n x 3 or n x 4.')
        self._ptype=pulse_type
        self._Tp=Tp
        if f0 is None:
            # Note that the default value is 0 rather than None because this 
            # can take a long time to run. Better to let the user enter the
            # value if known.
            f0=estimate_f0(waveform,Tp,type_p=pulse_type,gamma=gamma)
        self._f0=f0
        self._gamma=gamma
        self._adiabatic_override=False
        self._adiabatic_override_val=False
        # If you are copying then you can just get these values from the previous 
        # waveform instead of re-running the calculation
        if not iscopy:
            self._tw1,self._f1vec,self._w1_profile=_calc_tw1max(self._ptype,self._waveform,self._Tp,self._f0,self._gamma,suppress_plots=suppress_plots)
            self._tbw,self._fvec,self._pulse_freq_profile,self._on_res_Mz=_calc_tbw(self._ptype,self._waveform,self._Tp,self._f0,self._tw1/self._Tp,self._gamma,suppress_plots=suppress_plots)
    @property
    def npts(self):
        return self._waveform.shape[0]
    @property
    def gamma(self):
        """
        gyromagnetic ratio in MHz/T

        """
        return self._gamma
    @gamma.setter
    def gamma(self,new_gamma):
        # new_gamma in MHz/T
        self._gamma=new_gamma
    @property
    def waveform(self):
        return self._waveform.copy()
    @property
    def pulse_type(self):
        return self._ptype
    @property
    def f0(self):
        if hasattr(self._f0,'__iter__'):
            # intended to deal with multi-band case that is not yet fully implemented
            return self._f0[0]
        else:
            return self._f0
    @f0.setter
    def f0(self,newf0):
        """
        newf0 in Hz

        """
        fshift=newf0-self.f0
        if self.isGM:
            # This single dt works because the phaseRamp calculation uses
            # self.waveform[:,2] so the dt is multiplied by the timestep
            dt=self._Tp/np.sum(self.waveform[:,2])
            phaseRamp=(np.cumsum(self.waveform[:,3]*self.waveform[:,2])-self.waveform[0,3]*self.waveform[0,2])*(self.gamma/10)*dt*fshift*360
        else:
            phaseRamp=self.get_tvec()*fshift*360
        self._waveform[:,0]=self._waveform[:,0]+phaseRamp
        self._fvec=self._fvec+fshift/1000
        self._f0=self._f0+fshift
    @property
    def isGM(self):
        if (self.waveform.shape[1]==4) and np.any(self.waveform[:,3]!=0):
            return True
        else:
            return False
    @property
    def isPhsMod(self):
        a=(np.round(self.waveform[:,0])==180)+(np.round(self.waveform[:,0])==0)
        if np.sum(a)<self.npts:
            return True
        else: # every point has phase 0 or 180
            return False
    @property
    def isAdiabatic(self):
        if not self._adiabatic_override:
            # A crude estimate. Tests whether the Mz values between w1max and
            # 1.5*w1max are <10%. If yes, the pulse is assumed to be quite 
            # insensitive to B1 and therefore to be adiabatic.
            f1vec=self._f1vec
            w1_profile=self._w1_profile
            w1max=self.w1max
            w1_range=np.logical_and(f1vec>w1max,f1vec<1.5*w1max)
            Mz_range=w1_profile[2,w1_range]
            diff_Mz=Mz_range-Mz_range[0]
            return np.all(np.abs(diff_Mz)<0.1)
        else:
            return self._adiabatic_override_val
    @isAdiabatic.setter
    def isAdiabatic(self,adiabatic_override):
        self._adiabatic_override=True
        self._adiabatic_override_val=adiabatic_override
    @property
    def rfCentre(self):
        """
        Where the peak of the RF amplitude waveform occurs (range 0-1). If 
        rfCentre = 0.5, then the pulse is symmetric. If rfCentre<0.5 then the 
        peak occurs near the beginning of the pulse. If rfCentre>0.5, then the 
        peak occurs near the end of the pulse.
        """
        maxIdx=np.argmax(self.waveform[:,1])+1
        return maxIdx/len(self.waveform[:,1])
    def get_tvec(self,tp=None,expanded=False):
        if expanded:
            rfwf=self.get_expanded_wf()
        else:
            rfwf=self.waveform
        if tp is None:
            tp=self._Tp
        dt=tp/np.sum(rfwf[:,2])
        timesteps=rfwf[:,2]
        return dt*(np.cumsum(timesteps)-timesteps[0])
    @property
    def w1max(self):
        """
        w1max in kHz because self._Tp*1000 in ms
        """
        return self._tw1/(self._Tp*1000)
    @w1max.setter
    def w1max(self,neww1):
        # Note that this function is intended to allow the user to correct
        # w1max calculations that fail during initialization. But of course it 
        # is possible to enter an incorrect w1max value that will not produce
        # the pulse_type/flip angle that you've assigned. This will likely be 
        # obvious from the frequency-Mz plot if it is not suppressed, but 
        # please code responsibly.
        print('Setting new w1={:.3f} kHz and re-running time-bandwidth calculation.'.format(neww1))
        self._tw1=self._Tp*1000*neww1
        self._tbw,self._fvec,self._pulse_freq_profile,self._on_res_Mz=_calc_tbw(self._ptype,self._waveform,self._Tp,self.f0,self._tw1/self._Tp,self._gamma,suppress_plots=False)
    @property
    def bw(self):
        """
        bw in kHz because self._Tp*1000 in ms
        """
        tp=self._Tp*1000
        if self.isGM:
            warnings.warn('WARNING: Gradient-modulated pulse. Estimating thickness in cm.',FidAWarningRF)
            return self.tthk/(tp/1000)
        else:
            return self._tbw/tp
    @property
    def tbw(self):
        """
        tbw in kHz*ms or Hz*s
        """
        if self.isGM:
            return 'N/A - gradient modulated pulse'
        else:
            return self._tbw
    @property
    def tthk(self):
        """
        tthk in cm*s
        """
        if self.isGM:
            return self._tbw/1000 
        else:
            return 'N/A - frequency selective pulse'
    def get_ampint(self,ignore_adiabatic=False):
        """
        Compute the amplitude integral, which is used by Magnetom to calculate 
        the transmitter power that is required in order to achieve the desired 
        flip angle (the ratio of the flip angle of the desired pulse to the 
        flip angle of a square pulse with the same length and B1max).

        Parameters
        ----------
        ignore_adiabatic : boolean, optional
            In io_writepta, it says that the calculation is different for 
            adiabatic pulses but doesn't explain how to calculate in this case 
            so, currently, an Exception is thrown for adiabatic cases unless 
            ignore_adiabatic=True. The default is False.

        Returns
        -------
        AMPINT : float
            The amplitude integral for the pulse, if non-adiabatic.

        """
        
        if self.isAdiabatic and not ignore_adiabatic:
            raise FidAException('ERROR: This appears to be an adiabatic pulse. AMPINT calculation is not valid. Run with ignore_adiabatic=True to get result anyway.')
        rf_complex=self.get_complex_wf(expanded=True)
        AMPINT=np.sum(np.abs(rf_complex))/np.amax(np.abs(rf_complex))/rf_complex.shape[0]
        return AMPINT
    def get_expanded_wf(self):
        rfwf=self.waveform
        if np.all(rfwf[:,2]==1):
            expanded_wf=rfwf
        else:
            expanded_wf=list()
            for tpt,each_dt in enumerate(rfwf[:,2]):
                for tct in range(int(each_dt)):
                    if self.isGM:
                        # Shouldn't this just be 1 instead of rfwf[0,2]
                        expanded_wf.append([rfwf[tpt,0],rfwf[tpt,1],1,rfwf[tpt,3]])
                    else:
                        expanded_wf.append([rfwf[tpt,0],rfwf[tpt,1],1])
            expanded_wf=np.array(expanded_wf)
        return expanded_wf
    def get_complex_wf(self,expanded=False):
        if expanded:
            rfwf=self.get_expanded_wf()
        else:
            rfwf=self.waveform
        return rfwf[:,1]*np.exp(1j*rfwf[:,0]*np.pi/180)
    def add_phase(self,ph0):
        """
        Add a constant phase to the waveform. Only implemented for pulses that
        are not gradient-modulated.

        Parameters
        ----------
        ph0 : float
            Phase to add in degrees.

        Returns
        -------
        None. Sets self._waveform to the new waveform with phase added.

        """
        #
        if hasattr(ph0,'__iter__'):
            print('WARNING: This function is intended to add constant phase. You entered an iterable. To adjust frequency with a phase ramp, set the f0 attribute of your pulse. Proceeding anyway.')
        if self.isGM:
            raise FidAException('ERROR: add_phase only implemented for non-GM pulses')
        fullwf=self.get_complex_wf()
        newwf=fullwf*np.exp(1j*ph0*np.pi/180)
        self._waveform[:,0]=np.arctan2(np.imag(newwf),np.real(newwf))*180/np.pi
        self._waveform[:,1]=np.abs(newwf)
        
    def plot_w1_profile(self,dir_to_plot=2,ax1=None,force_recalc=False):
        """
        Plot the magnetization as a function of RF power w1 for the rf pulse.

        Parameters
        ----------
        dir_to_plot : int between 0 and 2, optional
            Defines whether to plot the x, y or z-magnetization. The default 
            is 2 (z-direction).
        ax1 : matplotlib.axes._subplots.AxesSubplot, optional
            Axis on which to display the plot. The default is None, which 
            generates a new figure and axis.
        force_recalc : boolean, optional
            If False, uses the _f1vec and _w1_profile vectors that were 
            previously generated. If True, these vectors are re-calculated for 
            the current pulse type, waveform, pulse duration, and f0 values. 
            The default is False.
        """
        if ax1 is None:
            f1,ax1=plt.subplots(1,1)
        if force_recalc:
            self._tw1,self._f1vec,self._w1_profile=_calc_tw1max(self._ptype,self.waveform,self._Tp,self.f0,gamma=self._gamma,suppress_plots=True)
        ax1.plot(self._f1vec,self._w1_profile[dir_to_plot,:])
        ax1.set_xlabel('Frequency (kHz)')
        ax1.set_ylabel('$M_{:s}$'.format(['x','y','z'][dir_to_plot]))
    def plot_freq_profile(self,dir_to_plot=2,ax1=None,force_recalc=False):
        """
        Plot the magnetization as a function of frequency (displaying the pulse
        bandwidth) for the rf pulse

        Parameters
        ----------
        dir_to_plot : int between 0 and 2, optional
            Defines whether to plot the x, y or z-magnetization. The default 
            is 2 (z-direction).
        ax1 : matplotlib.axes._subplots.AxesSubplot, optional
            Axis on which to display the plot. The default is None, which 
            generates a new figure and axis.
        force_recalc : boolean, optional
            If False, uses the _fvec and _pulse_freq_profile vectors that were 
            previously generated. If True, these vectors are re-calculated for 
            the current pulse type, waveform, pulse duration, f0 and _tw1 
            (includes recalculation of _tw1 for the waveform). The default is 
            False.
        """
        if ax1 is None:
            f1,ax1=plt.subplots(1,1)
        if force_recalc:
            self._tw1,self._f1vec,self._w1_profile=_calc_tw1max(self._ptype,self.waveform,self._Tp,self.f0,gamma=self._gamma,suppress_plots=True)
            self._tbw,self._fvec,self._pulse_freq_profile,self._on_res_Mz=_calc_tbw(self._ptype,self._waveform,self._Tp,self.f0,self._tw1/self._Tp,self._gamma,suppress_plots=True)
        ax1.plot(self._fvec,self._pulse_freq_profile[dir_to_plot,:])
        ax1.set_xlabel('Frequency (kHz)')
        ax1.set_ylabel('$M_{:s}$'.format(['x','y','z'][dir_to_plot]))
    def plot_onres(self,ax1=None,force_recalc=False):
        """
        Plot the evolution of the on-resonance Mz magnetization over time as 
        the rf pulse plays out.

        Parameters
        ----------
        ax1 : matplotlib.axes._subplots.AxesSubplot, optional
            Axis on which to display the plot. The default is None, which 
            generates a new figure and axis.
        force_recalc : boolean, optional
            If False, uses the _on_res_Mz vector that was previously generated. 
            If True, this vector is re-calculated using a new Bloch Simulation
            with the current parameters in the object, in case any have changed
            since initialization. The default is False.
        """
        if ax1 is None:
            f1,ax1=plt.subplots(1,1)
        if force_recalc:
            self._tw1,self._f1vec,self._w1_profile=_calc_tw1max(self._ptype,self.waveform,self._Tp,self.f0,gamma=self._gamma,suppress_plots=True)
            self._tbw,self._fvec,self._pulse_freq_profile,self._on_res_Mz=_calc_tbw(self._ptype,self._waveform,self._Tp,self.f0,self._tw1/self._Tp,self._gamma,suppress_plots=True)
        ax1.plot(self.get_tvec(),self._on_res_Mz)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('$M_z$ at {:.0f} Hz'.format(self.f0))
    def copy(self):
        newRF=RF_pulse(self.waveform,self.pulse_type,self._Tp,self._f0,iscopy=True,gamma=self._gamma)
        newRF._tw1=self._tw1
        newRF._f1vec=self._f1vec.copy()
        newRF._w1_profile=self._w1_profile.copy()
        newRF._tbw=self._tbw
        newRF._fvec=self._fvec.copy()
        newRF._pulse_freq_profile=self._pulse_freq_profile.copy()
        newRF._adiabatic_override=self._adiabatic_override
        newRF.adiabatic_override_val=self._adiabatic_override_val
        return newRF
    def __repr__(self):
        tstr='{:s} has size {:s}: \n'.format(self.__class__.__name__,str(self.waveform.shape))
        plist=['Pulse type','f0','Tp','w1','phase-modulated','gradient-modulated','adiabatic','bandwidth']
        pdict={'Pulse type':self.pulse_type, 'f0':self._f0, 'Tp':self._Tp,'w1':self.w1max,
               'phase-modulated':self.isPhsMod,'gradient-modulated':self.isGM,
               'adiabatic':self.isAdiabatic,'bandwidth':self.tbw/self._Tp/1000}
        unitdict={'Pulse type':'', 'f0':'Hz', 'Tp':'s','w1':'kHz','phase-modulated':'','gradient-modulated':'','adiabatic':'','bandwidth':'kHz'}
        for eachit in plist:
            tstr=tstr+eachit+': '+str(pdict[eachit])+' '+unitdict[eachit]+'\n'
        return tstr
    
def _calc_tw1max(ptype,rf,Tp,f0=0,gamma=GAMMA_DICT['1H'],suppress_plots=False):
    """
    Calculate the time-w1 product in ms*kHz for an RF pulse with certain 
    parameters. This is done using a Bloch Simulation for phase-modulated pulse 
    and with a more straightforward calculation for non-phase-modulated pulses.

    Parameters
    ----------
    ptype : str or float
        Type of rf pulse. Can be 'exc' for 90, 'ref' or 'inv' for 180, or a 
        numerical value that give the flip angle in degrees.
    rf : n x 3 or n x 4 numpy array
        rf waveform containing the n timesteps, with columns defining the 
        phase (in degrees), amplitude, relative time step length, and 
        (optionally) the gradient amplitude in G/cm at each time point.
    Tp : float
        RF pulse duration in seconds.
    f0 : float, optional
        Offset frequency of the rf pulse in Hz. If this number is incorrect and
        does not correspond to the actual offset frequency of the pulse in "rf",
        things will end badly. Use the estimate_f0 function to find the correct
        value if you are uncertain of the resonance frequency of the rf 
        waveform. The default is 0.
    gamma : float, optional
        The gyromagnetic ratio of the nucleus the pulse is intended to be used
        for. The default is the gyromagnetic ratio for hydrogen, 42.577 MHz/T.
    suppress_plots : boolean, optional
        Whether to display the plot of w1 vs Mz following the calculation. The 
        default is False.

    Returns
    -------
    Tp*w1max
        The time-w1 product in kHz*ms for the rf pulse needed to achieve the 
        desired flip angle.
    B1vec : npts numpy array
        Array of rf power values in kHz at which the calculation was done.
    w1_profile : 3 x npts numpy array
        Magnetization in the x, y, and z directions at the end of the rf pulse
        for each of the rf powers in B1vec.

    """
    print('Calculating time-w1max product')
    # If the pulse is not phase-modulated, the time-w1 product is straightforward.
    if ptype=='exc':
        flipCyc=0.25
    elif ptype=='ref' or ptype=='inv':
        flipCyc=0.5
    else: # numerical flip angle in degrees
        flipCyc=ptype/360
    a=(np.round(rf[:,0])==180)+(np.round(rf[:,0])==0)
    if np.sum(a)==rf.shape[0]: #not phase-modulated. Every point has phase 0 or 180
        intRF=np.sum(rf[:,1]*-1*np.sign(rf[:,0]-179)*rf[:,2])/np.sum(rf[:,2])
        if intRF!=0:
            w1max=flipCyc/(intRF*Tp)
        else:
            w1max=0
        # No need to run the Bloch Simulation but double-checking. Can use fewer points.
        B1vec=np.linspace(0,w1max/1000/flipCyc*1.5,4000)
        bloch1=BlochSimulator(rf,Tp*1000,f0/1000,B1=B1vec,gamma=gamma)
        w1_profile=bloch1.finalM0
    # If phase-modulated, need to estimate numerically from Bloch simulations.
    # Get Mz as a function of w1 and find the value of w1 that results in the 
    # desired flip angle
    else:
        B1vec=np.linspace(0,5,40000)
        bloch1=BlochSimulator(rf,Tp*1000,f0/1000,B1=B1vec,gamma=gamma)
        # In Matlab, the simulation result is plotted and the user is asked for 
        # input. However, depending on the backend that you are using for 
        # Python, Figures may not display when the program runs. Therefore, I 
        # have opted to estimate the point where Mz is at its target and then 
        # plot afterward (if suppress_plots=False) for users to double-check.
        if flipCyc<=0.5:
            try:
                target_val=np.amax([-0.98,np.cos(flipCyc*2*np.pi)])
                idx_target=np.flatnonzero(bloch1.finalM0[2,:]<target_val)[0]
                idx_target=int(np.round(idx_target))
            except IndexError:
                # If that fails and the flip angle cannot be reached, try to
                # find the minimum Mz (largest flip angle available). Because
                # adiabatic pulses have a long, flat profile with the min near
                # the end, take w1 as the point where Mz crosses 95% of the
                # min value
                warnings.warn('WARNING: TARGET VALUE NOT FOUND. It is likely pulse frequency does not match provided f0. Check your_RF_pulse_name.plot_w1_profile() and your_RF_pulse_name.plot_freq_profile(). Returning w1 for largest flip angle.',FidAWarningRF)
                #print('ERROR: TARGET VALUE NOT FOUND. It is likely pulse frequency does not match provided f0. Check your_RF_pulse_name.plot_w1_profile() and your_RF_pulse_name.plot_freq_profile(). Returning w1 for largest flip angle.')
                target_val=np.amin(bloch1.finalM0[2,:])
                idx_target=np.flatnonzero(bloch1.finalM0[2,:]<0.95*target_val)[0]
                idx_target=int(np.round(idx_target))
        else:
            warnings.warn('WARNING: w1max estimates for phase-modulated pulses with flip angles > 180 may not be correct. Check graphically with your_RF_pulse_name.plot_w1_profile().',FidAWarningRF)
            #print('WARNING: w1max estimates for phase-modulated pulses with flip angles > 180 may not be correct. Check graphically')
            # Harder because we're going past 180 and will be getting a repeat value.
            # Use the zero-crossing or minimum in this case.
            try:
                idx_360=4*np.flatnonzero(bloch1.finalM0[2,:]<0)[0]
                idx_target=int(np.round(flipCyc*idx_360))
            except IndexError:
                warnings.warn('WARNING: NO ZERO CROSSING FOUND. It is likely pulse frequency does not match provided f0. Check your_RF_pulse_name.plot_w1_profile() and your_RF_pulse_name.plot_freq_profile(). Returning w1 for largest flip angle.',FidAWarningRF)
                target_val=np.amin(bloch1.finalM0[2,:])
                idx_target=np.flatnonzero(bloch1.finalM0[2,:]<0.95*target_val)[0]
                idx_target=int(np.round(idx_target))
        w1max=B1vec[idx_target]*1000
        w1_profile=bloch1.finalM0
        print('Pulse is phase-modulated. Numerical w1max estimation of {:3.0f} Hz.'.format(w1max))
        print('It is recommended that you double check this value on a graph. If interactive plotting is off, you can display the graph with your_RF_pulse_name.plot_w1_profile(). To set a new w1max, use your_RF_pulse_name.w1max=<w1_in_kHz>.')
    if not suppress_plots:
        f1,ax1=plt.subplots(1,1)
        ax1.plot(B1vec,w1_profile[2,:])
        ax1.set_xlabel('w1 (kHz)')
        ax1.set_ylabel('$M_z$')
    return Tp*w1max,B1vec,w1_profile

def _calc_tbw(ptype,rf,Tp,f0,w1max,gamma=GAMMA_DICT['1H'],suppress_plots=False):
    """
    Calculate the time-bandwidth product in ms*kHz for an RF pulse with certain 
    parameters. This is done using a Bloch Simulation.

    Parameters
    ----------
    ptype : str or float
        Type of rf pulse. Can be 'exc' for 90, 'ref' or 'inv' for 180, or a 
        numerical value that give the flip angle in degrees.
    rf : n x 3 or n x 4 numpy array
        rf waveform containing the n timesteps, with columns defining the 
        phase (in degrees), amplitude, relative time step length, and 
        (optionally) the gradient amplitude in G/cm at each time step.
    Tp : float
        RF pulse duration in seconds.
    f0 : float, optional
        Offset frequency of the rf pulse in Hz. If this number is incorrect and
        does not correspond to the actual offset frequency of the pulse in "rf",
        the bandwidth likely cannot be calculated correctly. Use the 
        estimate_f0 function to find the correct value if you are uncertain of 
        the resonance frequency of the rf waveform. The default is 0.
    w1max : float
        Maximum rf power in Hz need to reach the flip angle in ptype. Typically 
        calculated from a previous Bloch simulation and will be incorrect if f0 
        was incorrect.
    gamma : float, optional
        The gyromagnetic ratio of the nucleus the pulse is intended to be used
        for. The default is the gyromagnetic ratio for hydrogen, 42.577 MHz/T.
    suppress_plots : boolean, optional
        Whether to display the plot of frequency vs Mz following the 
        calculation. The default is False.

    Returns
    -------
    Tp*1000*bw
        The time-bandwidth product for the rf pulse in ms*kHz.
    fvec : npts numpy array
        Array of frequencies in kHz at which the calculation of the final 
        magnetization at the end of the rf pulse was done.
    pulse_freq_profile : 3 x npts numpy array
        Magnetization in the x, y, and z directions at the end of the rf pulse
        for each of the rf powers in fvec.
    on_res_Mz_vs_t : numpy array of length n (number of time steps in rf pulse)
        Shows the Mz evolution of the rf pulse over time, at frequency f0.
    """
    # First make a plot of the pulse profile over a wide bandwidth
    print('Calculating bandwidth')
    # f0 sent to BlochSimulator needs to be in kHz and w1max also in kHz
    fvec=np.linspace(-5+f0/1000,5+f0/1000,10000)
    bloch1=BlochSimulator(rf,Tp*1000,fvec,w1max/1000,gamma=gamma)
    # To get bandwidth, need to find FWHM. Half-max depends on flip angle (ptype)
    if ptype=='exc':
        target_mz=0.5
    elif ptype=='ref' or ptype=='inv':
        target_mz=0
    else: #numeric case
        target_mz=(1+np.cos(ptype*np.pi/180))/2
    multiband=False
    try:
        # Will give you bandwidth across the first band for dual-band pulses but then the repeat will be off
        halfmax_idx1=np.flatnonzero(bloch1.finalM0[2,:]<target_mz)[0]
        halfmax_idx2=halfmax_idx1+np.flatnonzero(bloch1.finalM0[2,halfmax_idx1:]>target_mz)[0]+1
        bw=fvec[halfmax_idx2]-fvec[halfmax_idx1]
        # Is there a second band?
        try:
            halfmax_idx3=np.flatnonzero(bloch1.finalM0[2,halfmax_idx2+1:]<target_mz)
            if len(halfmax_idx3)!=0:
                multiband=True
        except IndexError:
            pass 
    except IndexError:
        bw=0
        warnings.warn('ERROR: Bandwidth calculation failed. You may have entered an off-resonance pulse with an incorrect f0 estimate. Run your_RF_pulse.plot_freq_profile() to view Mz-frequency plot',FidAWarningRF)
    else:
        # Now repeat over a narrower bandwidth to get more exact
        if not multiband:
            fvec=np.linspace(-bw+f0/1000,bw+f0/1000,100000)
            bloch1=BlochSimulator(rf,Tp*1000,fvec,w1max/1000,gamma=gamma)
            halfmax_idx1=np.flatnonzero(bloch1.finalM0[2,:]<target_mz)[0]
            halfmax_idx2=halfmax_idx1+np.flatnonzero(bloch1.finalM0[2,halfmax_idx1:]>target_mz)[0]+1
            bw=fvec[halfmax_idx2]-fvec[halfmax_idx1]
            estf0=fvec[halfmax_idx1]+bw/2
            if np.abs(estf0-f0/1000)>0.25:
                warnings.warn('WARNING: Entered f0={:g} is more that 250 Hz different than estimated f0={:g}. You may have incorrectly entered the f0 for an off-resonance pulse.'.format(f0,estf0*1000),FidAWarningRF)
        else:
            warnings.warn('WARNING: Suspected multi-band pulse. Bandwidth estimate will be for first band and coarse')
            estf0=fvec[halfmax_idx1]+bw/2
    finally:
        pulse_freq_profile=bloch1.finalM0
        fidx=np.flatnonzero(fvec>f0/1000)[0]
        on_res_Mz_vs_t=bloch1.mvec[:,2,fidx]
        if not suppress_plots:
            f1,ax1=plt.subplots(1,1)
            ax1.plot(fvec,bloch1.finalM0[2,:])
            ax1.set_xlabel('Frequency (kHz)')
            ax1.set_ylabel('$M_z$')
        # Returning fvec in kHz. bw is in kHz so Tp*1000*bw is in ms*kHz (or s*Hz)
        return Tp*1000*bw,fvec,pulse_freq_profile,on_res_Mz_vs_t
    
def estimate_f0(rfwf,tp,type_p='inv',fvec=None,gamma=GAMMA_DICT['1H'],w1_start=0.03,w1_step=0.02,return_w1max=False):
    """
    Estimated offset frequency of an rf pulse, found using iterative Bloch 
    simulations.

    Parameters
    ----------
    rfwf : n x 3 or n x 4 numpy array
        rf waveform containing the n timesteps, with columns defining the 
        phase (in degrees), amplitude, relative time step length, and 
        (optionally) the gradient amplitude in G/cm at each time step.
    tp : float
        RF pulse duration in seconds.
    type_p : str or float, optional
        Type of rf pulse. Can be 'exc' for 90, 'ref' or 'inv' for 180, or a 
        numerical value that gives the flip angle in degrees. The default is 
        'inv'.
    fvec : 1D numpy array, optional
        Vector of frequency values in kHz over which the Bloch simulations 
        will be run. The f0 value for the pulse must be contained within this
        vector and its resolution determines the accuracy of the f0_est. Larger
        vectors take longer to simulation. The default is None, which generates
        a vector of np.linspace(-5,5,1000)
    gamma : float, optional
        The gyromagnetic ratio of the nucleus the pulse is intended to be used
        for. The default is the gyromagnetic ratio for hydrogen, 42.577 MHz/T.
    w1_start : float, optional
        The rf power value in kHz at which to start iterating through the Bloch
        simulations. The default is 0.03.
    w1_step : float, optional
        The amount to increase the rf power estimate, in kHz, for each 
        iteration of the Bloch simulations. Smaller values will more accurately
        find the rf power needed to generate the pulse's desired flip angle 
        but take longer to run. The default is 0.02.
    return_w1max : boolean, optional
        Whether or not to return the rf power that generated the desired flip
        angle. The default is False, meaning that only the estimated f0 is
        returned.

    Returns
    -------
    f0_est : float
        Estimated offset frequency for the rf pulse in Hz.
    tmp_w1 : float
        RF power, in kHz, needed to reach the desired flip angle for the rf
        pulse at the offset frequency f0_est. Only returned if 
        return_w1max=True in the input arguments.
    """
    tmp_w1=w1_start; tmp_min=1
    # Target Mz is set slightly above the theoretical value in case w1_step 
    # and/or fvec are too coarse to catch it exactly (mostly a problem for inv 
    # case where Mz can't go below -1).
    if type_p=='exc':
        min_floor=0+0.02
    elif type_p=='inv' or type_p=='ref':
        min_floor=-1+0.02
    else: # numeric case
        min_floor=(1+np.cos(type_p))+0.02
    if fvec is None:
        fvec=np.linspace(-5,5,1000)
    while tmp_min > min_floor:
        bloch1=BlochSimulator(rfwf,tp*1000,f0=fvec,B1=tmp_w1,gamma=gamma)
        tmp_min=np.amin(bloch1.finalM0[2,:])
        tmp_w1=tmp_w1+w1_step
    #tmp_min_pos=np.argmin(bloch1.finalM0[2,:])
    minval=np.amin(bloch1.finalM0[2,:])
    target_mz=1-(1-minval)/2
    halfmax_idx1=np.flatnonzero(bloch1.finalM0[2,:]<target_mz)[0]
    halfmax_idx2=halfmax_idx1+np.flatnonzero(bloch1.finalM0[2,halfmax_idx1:]>target_mz)[0]+1
    tmp_min_pos=halfmax_idx1+(halfmax_idx2-halfmax_idx1)//2
    f0_est=fvec[tmp_min_pos]*1000
    if return_w1max:
        return f0_est, tmp_w1
    else:
        return f0_est