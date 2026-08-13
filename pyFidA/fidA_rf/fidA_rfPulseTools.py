#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 12:58:22 2025
pyFidA.fidA_rf.fidA_rfPulseTools.py

@author: Colleen Bailey (@cbailey@sri.utoronto.ca), based on Matlab code by Jamie Near

RF pulse tools for manipulating and analyzing RF_pulse objects in pyFidA.

Functions:
    * rf_addGrad,
    * rf_blochSim,
    * rf_combineRF,
    * rf_dualBand,
    * rf_freqshift,
    * rf_gauss,
    * rf_getRFPeakPower,
    * rf_getRFPulseEnergy,
    * rf_goia,
    * rf_hs,
    * rf_plotWaveform,
    * rf_refocusedComponent,
    * rf_resample,
    * rf_scaleGrad,
    * rf_sinc,
    * rf_single2dualBand,
    * rf_timeReverse,
    * rf_verse,

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from scipy.interpolate import CubicSpline,interp1d
import warnings

from pyFidA.fidA_common import BlochSimulator, RF_pulse, GAMMA_DICT, FidAException, FidAWarningRF, phase

def rf_addGrad(RF_in,grad,overwrite_wf=False,suppress_plots=True):
    """
    Add a gradient to an existing RF_pulse object.

    Parameters
    ----------
    RF_in : pyFidA.RF_pulse
        If the rf pulse does not already have an associated gradient, the 
        supplied gradient will be appended to the existing rf pulse waveform.
        If the rf pulse already has an associated gradient, the user will be
        asked whether to overwrite it with the new gradient info, unless
        overwrite_wf=True, in which case the gradient will be overwritten 
        without prompting.
    grad : scalar or numpy array
        Gradient value(s) in G/cm that will be added to the waveform. If scalar,
        all time points in the waveform will have the same gradient value. If
        numpy array, the array must be the same length as the existing waveform
        and will be appended to the waveform.
    overwrite_wf : boolean, optional
        If RF_in is already gradient-modulated, this parameter controls whether
        the gradient will automatically be overwritten (True) or whether the
        user will be prompted (False) for an answer about overwriting. The 
        default is False.
    suppress_plots : boolean, optional
        Whether to suppress the plots of the w1 and frequency profiles (Mz 
        values) for the output gradient-modulated waveform. The default is True.

    Raises
    ------
    TypeError
        Raised when RF_in is not of type pyFidA.RF_pulse.
    FidAException
        Raised when the existing waveform of RF_in does not have 3 or 4 columns.

    Returns
    -------
    RF_out : pyFidA.RF_pulse
        RF pulse with npts x 4 waveform, where the last column is the gradient
        applied at each pulse time step, in G/cm. This is a new RF_pulse object 
        that re-calculates the time-w1 product

    """
    if not isinstance(RF_in,RF_pulse):
        raise TypeError('ERROR: the input RF pulse must be of type RF_pulse. Try using io_loadRFwaveform to convert. Aborting!')
    newWaveform=RF_in.waveform
    if RF_in.isGM:
        if not overwrite_wf:
            warnings.warn('WARNING: Input waveform already has a gradient waveform!',FidAWarningRF)
            keepGoing=input('Do you wish to overwrite the existing gradient waveform (y or n): ')
            if keepGoing.lower()=='y':
                print('Okay. Overwriting existing gradient waveform.')
            elif keepGoing.lower()=='n':
                return RF_in.copy()
            else:
                raise ValueError('ERROR: Response not recognized.')
    elif newWaveform.shape[1]==3:
        # Initialize
        newWaveform=np.concatenate((newWaveform,np.zeros([RF_in.npts,1])),axis=1)
    else:
        raise FidAException('ERROR: Waveform should have at least 3 columns.')
    newWaveform[:,3]=grad
    # For an off-resonance pulse, a gradient will alter the pulse frequency f0. 
    # It's unlikely that most users would intend this but it is possible, so 
    # the action is allowed and the f0 value is re-calcuated, but a warning is
    # thrown so that users know that f0 has changed. Users who know that they
    # will not want to add gradients to off-resonance pulses can raise this
    # warning to an error by adding the appropriate filter with the "error" 
    # action via Python's "warnings" module (https://docs.python.org/3/library/warnings.html)
    if RF_in.f0!=0:
        warnings.warn('WARNING: Attempting to use rf_addGrad on off-resonance pulse. Will attempt to calculate new f0. This is time-consuming.',FidAWarningRF)
        RF_out=RF_pulse(newWaveform,RF_in.pulse_type,RF_in._Tp,f0=None,gamma=RF_in._gamma,suppress_plots=suppress_plots)
    else:
        RF_out=RF_pulse(newWaveform,RF_in.pulse_type,RF_in._Tp,0,gamma=RF_in._gamma, suppress_plots=suppress_plots)
    return RF_out
    
def rf_blochSim(RF,tp,fspan=10,f0=0,peakB1=None,ph=0,npts=10000,M0=np.r_[0,0,1],display_output=True):
    """
    Perform a Bloch simulation of an RF pulse. Different units are possible since
    the code uses the products tp*peakB1 and fspan*(tp/tsteps) but, if tp is in
    ms then peakB1, f0 and fspan should all be in kHz. This code simply runs the 
    BlochSimulator from fidA_common, which is adapted from Martyn Klassen's 
    Matlab simulator. For more information see fidA_common.BlochSimulator.

    Parameters
    ----------
    RF : RF_pulse object
        The RF pulse with the waveform (and, if needed, time-w1max product) to 
        be used.
    tp : float
        Pulse duration in ms (if f0, fspan and B1 in kHz).
    fspan : float, optional
        Frequency span in kHz or, if the RF pulse includes a gradient waveform
        then fspan is the span of the spatial positions in cm. The default is 10.
    f0 : float, optional
        Centre of the frequency span in kHz. The default is 0.
    peakB1 : float, optional
        Peak B1 amplitude in kHz. The default is None, which calculates 
        RF.w1max(tp)
    ph : float, optional
        Starting phase of the rf pulse in degrees. The default is 0.
    npts : int, optional
        Number of points to create in the frequency vector (spanning fspan). 
        The default is 10,000.
    M0 : 3-element numpy vector, optional
        Starting magnetization vector. The default is np.r_[0,0,1].
    display_output : boolean, optional
        Flag that indicates whether the magnetization vs frequency is to be
        plotted. The default is True.
    
    Returns
    -------
    mv : 3 x npts numpy array
        Simulated magnetization vector as a function of frequency.
    fvec : 1D numpy array of length npts
        The frequency vector created from f0-fspan/2 to f0+fspan/2.

    """
    if peakB1 is None:
        peakB1=RF.w1max(tp)
    fvec=np.linspace(f0-fspan/2,f0+fspan/2,npts)
    bloch1=BlochSimulator(RF.waveform,tp,fvec,peakB1,ph,M0,gamma=RF.gamma)
    mv=bloch1.finalM0
    if display_output:
        f1,ax1=plt.subplots(2,2,sharex=True)
        ax1[0,0].plot(fvec,mv[0,:],lw=1.2)
        ax1[0,0].set_ylabel('$M_x$')
        ax1[0,1].plot(fvec,mv[1,:],lw=1.2)
        ax1[0,1].set_ylabel('$M_y$')
        ax1[1,0].plot(fvec,np.sqrt(mv[0,:]**2+mv[1,:]**2),lw=1.2)
        ax1[1,0].set_ylabel('$M_{xy}$')
        ax1[1,1].plot(fvec,mv[2,:],lw=1.2)
        ax1[1,1].set_ylabel('$M_z$')
        if RF.isGM<4:
            ax1[1,0].set_xlabel('Position (cm)')
        else:
            ax1[1,0].set_xlabel('Frequency (kHz)')
        f1.tight_layout()
    return mv, fvec

def rf_combineRF(rf1,rf2,suppress_plots=False):
    """
    Combines two rf waveforms (intended for pulses with the same pulse shape
    but one frequency-shifted). Note that cases where the frequency difference
    between the pulses is small relative to the bandwidth will throw a warning
    but the w1max and bandwidth are re-calculated for the new pulse (differs
    from Matlab where tw1 and tbw are summed, which is not accurate in the 
    case of small frequency differences/large bandwidths)

    Parameters
    ----------
    rf1 : pyFidA.RF_pulse
        First rf pulse.
    rf2 : pyFidA.RF_pulse
        Second rf pulse.
    suppress_plots : boolean, optional
        Whether to suppress the plots of w1 vs Mz and frequency vs Mz that are
        generated for the pulse from Bloch simulations. The default is False.

    Raises
    ------
    FidAException
        Raise in the case that either rf pulse is gradient-modulated OR in the
        case that the time steps of the functions don't match.

    Returns
    -------
    rf_out : pyFidA.RF_pulse
        Output rf waveform of the combined rf pulse.
    AMPINT : float
        Calculated amplitude integral (for use in Siemens .pta files).

    """
    if np.abs(rf1.f0-rf2.f0)<1.3*1000*(rf1.bw/2+rf2.bw/2):
        warnings.warn('WARNING: Entered frequency difference is small relative to bandwidth. Peaks will overlap and frequency bands may not look as expected.',FidAWarningRF)
    if rf1.isGM or rf2.isGM:
        raise FidAException('ERROR: rf_combineRF not implemented for gradient-modulated waveforms')
    if any(rf1.waveform[:,2]!=rf2.waveform[:,2]):
        raise FidAException('ERROR: time steps of waveforms must match')
    rf1_wf=rf1.get_complex_wf()
    rf2_wf=rf2.get_complex_wf()
    combined_waveform=rf1_wf+rf2_wf
    combined_waveform_scaled=combined_waveform/np.amax(np.abs(combined_waveform))
    
    newrfwf=rf1.waveform
    newrfwf[:,0]=phase(combined_waveform_scaled)*180/np.pi
    newrfwf[:,1]=np.abs(combined_waveform_scaled)
    # For cases with overlapping bands, need to estimate f0 because the maximum
    # flip angle may no longer be at the expected points.
    if np.abs(rf1.f0-rf2.f0)<1.3*1000*(rf1.bw/2+rf2.bw/2):
        rf_out=RF_pulse(newrfwf, rf1.pulse_type,rf1._Tp,f0=None,gamma=rf1.gamma,suppress_plots=suppress_plots)
    else:
        rf_out=RF_pulse(newrfwf, rf1.pulse_type,rf1._Tp,rf1.f0,gamma=rf1.gamma,suppress_plots=suppress_plots)
    AMPINT=rf_out.get_ampint()
    
    # Tough to know how to define _f0 because there are two minima. Taking the
    # average doesn't put you at an Mz minimum and will fail to find the correct
    # w1max if that calculation is run. Currently storing as a vector but
    # rf_out.f0 will return just the first value in this vector.
    rf_out._f0=np.r_[rf_out._f0,rf2._f0]
    return rf_out,AMPINT

def rf_dualBand(tp,df,npts,bw,ph0,shft,ptype='inv',asym_factor=0,filter_flag=False,gamma=GAMMA_DICT['1H'],suppress_plots=False):
    """
    Creates a dual-banded Gaussian inversion RF pulse. The plots generated for
    dual band pulses are a work-in-progress.

    Parameters
    ----------
    tp : float
        Duration of the pulse in ms.
    df : float
        Frequency difference between the first and second pulses in Hz. Note 
        that this shifts in the opposite direction compared to Matlab's 
        rf_dualBand.
    npts : int
        Number of points in the rf waveform.
    bw : float
        FWHM of the Gaussian profiles in Hz. Note that this differs from the
        number input in the equivalent Matlab function, which is off by a
        factor of sqrt(pi) from the FWHM.
    ph0 : float
        Phase of the second Gaussian.
    shft : float
        Frequency shift applied to both bands, in Hz
    ptype : float or str, optional
        Flip angle in degrees if float. str values can be 'exc' (for 90 degree 
        pulse) or 'inv' or 'ref' for 180 degree pulse. The default is 'inv'
    asym_factor : float, optional
        Asymmetry factor for the amount to shift the pulse in the time domain,
        in ms. Note that this interacts with df and shft. The default is 0.
    filter_flag : boolean, optional
        Whether to apply a cosine filter to the pulse to avoid ringing 
        artefacts. If True, the user will be asked for an attenuation factor.
        The default is False.
    gamma : float optional
        Gyromagnetic ratio for the nucleus that the pulse will be applied to in
        MHz/T. Only saved in case a gradient is later applied to the pulse and 
        G/cm needs to be converted to dephasing. The default is 
        GAMMA_DICT['1H'] = 42.577 MHz/T.
    suppress_plots : boolean, optional
        Whether to suppress the plots of w1 vs Mz and frequency vs Mz that are
        generated for the pulse from Bloch simulations. The default is False.

    Returns
    -------
    rf_out : pyFidA.RF_pulse
        Output rf waveform for the dual banded rf pulse.
    AMPINT : float
        Calculated amplitude integral, for use in Siemens .pta files.

    """
    # rf_combineRF already runs the check that the bandwidth is small relative
    # to the frequency so I've removed that from here and let it be handled there.
    rf_gauss1,_=rf_gauss(tp,bw,npts,ptype,df=shft,asym_factor=asym_factor,filter_flag=filter_flag,gamma=gamma,suppress_plots=True)
    rf_gauss2=rf_freqshift(rf_gauss1,F=shft+df)
    rf_gauss2.add_phase(ph0)
    rf_out,_=rf_combineRF(rf_gauss1,rf_gauss2,suppress_plots=suppress_plots)
    AMPINT=rf_out.get_ampint()
    return rf_out, AMPINT

def rf_freqshift(rf_in,F=0,Tp=None):
    """
    Apply a frequency shift to an RF pulse

    Parameters
    ----------
    rf_in : pyFidA.RF_pulse
        Initial rf pulse.
    F : float, optional
        Amount to shift the frequency of the rf pulse, in Hz. The default is 0.
    Tp : float, optional
        Duration of the rf pulse, in ms. The default is None, which uses the
        value of rf_in._Tp.

    Returns
    -------
    rf_out : pyFidA.RF_pulse
        Output rf pulse following the frequency shift.

    """
    rf_out=rf_in.copy()
    rfwf=rf_in.waveform
    if Tp is None:
        Tp=rf_in._Tp
    else:
        Tp=Tp/1000
    dt=Tp/np.sum(rfwf[:,2])
    timesteps=rfwf[:,2]
    # Calculation accounts for non-uniform timesteps in rfwf[:,2], as well as
    # the effects of any gradient associated with the pulse
    if rf_in.isGM:
        phaseRamp=(np.cumsum(rfwf[:,3]*timesteps)-rfwf[0,3]*timesteps[0])*(rf_in.gamma/10)*dt*F*360
    else:
        phaseRamp=dt*(np.cumsum(timesteps)-timesteps[0])*F*360
    rf_out._waveform[:,0]=rf_out._waveform[:,0]+phaseRamp
    rf_out._f0=rf_in._f0+F # Note that you need the underscore on rf_out._f0 because if you use rf_out.f0 because this calls the setter, which does the frequency shift again
    rf_out._fvec=rf_in._fvec+F/1000
    return rf_out
    
def rf_gauss(tp,bw,npts,ptype,df=0,asym_factor=0,filter_flag=False,gamma=GAMMA_DICT['1H'],suppress_plots=False):
    """
    Create an RF_pulse object with a Gaussian waveform of length npts. The

    Parameters
    ----------
    tp : float
        Duration of the pulse in ms.
    bw : float
        FWHM of the Gaussian profile in Hz. Note that this differs from the
        number input in the equivalent Matlab function, which is off by a
        factor of sqrt(pi) from the FWHM.
    npts : int
        Number of points in the rf waveform.
    ptype : float or str
        Flip angle in degrees if float. str values can be 'exc' (for 90 degree 
        pulse) or 'inv' or 'ref' for 180 degree pulse.
    df : float, optional
        Frequency of the Gaussian pulse in Hz. Note that this shifts in the 
        opposite direction compared to Matlab's rf_gauss. The default is 0.
    asym_factor : float, optional
        Asymmetry factor for the amount to shift the pulse in the time domain,
        in ms. Note that this interacts with df. The default is 0.
    filter_flag : boolean, optional
        Whether to apply a cosine filter to the pulse to avoid ringing 
        artefacts. If True, the user will be asked for an attenuation factor.
        The default is False.
    gamma : float optional
        Gyromagnetic ratio for the nucleus that the pulse will be applied to in
        MHz/T. Only saved in case a gradient is later applied to the pulse and 
        G/cm needs to be converted to dephasing. The default is 
        GAMMA_DICT['1H'] = 42.577 MHz/T.
    suppress_plots : boolean, optional
        Whether to suppress the plots of w1 vs Mz and frequency vs Mz that are
        generated for the pulse from Bloch simulations. The default is False.

    Returns
    -------
    rf_out : pyFidA.RF_pulse
        RF_pulse object for the Gaussian waveform.
    AMPINT : float
        Calculated amplitude integral (for use in Siemens .pta files).

    """
    tps=tp/1000
    if asym_factor is None:
        asym=input('Would you like to make the pulse asymmetric (y/n)? ')
        if asym.lower()=='y':
            asym_factor=float(input('Enter asymmetry factor: '))
        else:
            asym_factor=0
    t=np.linspace(0,tps,npts)-tps/2-asym_factor*tps
    fwhmt=1/bw
    c=fwhmt/2/np.sqrt(2*np.log(2))/np.sqrt(np.pi)
    gauss1=np.exp(-t**2/(2*c**2))
    if filter_flag:
        print('Cosine filtering is on. Check plot to ensure filter is not negative at the tails. (Change asymmetry factor or attenuation if so.)')
        attn=float(input('By what factor would you like to attenuate the edges of the pulse? '))
        fcos=np.cos(t*np.pi/(np.pi/np.arccos(attn)*np.amax(t)))
        plt.figure()
        plt.plot(t,fcos)
        gauss1=gauss1*fcos
    rf1=np.ones([npts,3])
    gauss1=gauss1/np.max(gauss1)
    rf1[:,0]=phase(gauss1)*180/np.pi
    rf1[:,1]=np.abs(gauss1)
    if df==0:
        rf_out=RF_pulse(rf1,ptype,tps,f0=0,gamma=gamma,suppress_plots=suppress_plots)
    else:
        # Calculate tw1 and tbw for f=0 and then shift, to save time on tw1 estimation
        rf_gauss1=RF_pulse(rf1,ptype,tps,f0=0,gamma=gamma,suppress_plots=True)
        rf_out=rf_freqshift(rf_gauss1,df)
        if not suppress_plots:
            rf_out.plot_w1_profile()
            rf_out.plot_freq_profile()
    AMPINT=rf_out.get_ampint()  
    return rf_out, AMPINT

def rf_getRFPeakPower(RF1,tp,TxAmpPwrRating,sysB1max,flipAngle=None,gamma=GAMMA_DICT['1H']):
    # Required peak B1 for the input RF pulse. For tp is entered at the input in
    # ms, this will convert to s and the w1max will be in Hz and rfPeakB1 in microtesla
    rfPeakB1=RF1._tw1/(tp/1000)/gamma
    # This is correct for pulses of type exc, ref or inv but, if a different flip
    # angle was specified, need to adjust the B1_required
    if flipAngle is not None:
        if RF1.pulse_type=='exc':
            rfPeakB1=rfPeakB1*flipAngle/90
        elif RF1.pulse_type=='ref' or RF1.pulse_type=='inv':
            rfPeakB1=rfPeakB1*flipAngle/180
        else: # numerical flip angle in degrees
            rfPeakB1=rfPeakB1*flipAngle/RF1.pulse_type
    f_B1=rfPeakB1/sysB1max
    if f_B1>1:
        print('WARNING: Pulse exceeds system B1 amplitude limit! Calculating anyway.')
    # Calculate the instantaneous peak power of the RF pulse
    rfPeakPower=(f_B1**2)*TxAmpPwrRating
    return rfPeakPower,rfPeakB1

def rf_getRFPulseEnergy(RF1,tp,peakPower):
    # Calculate the power waveform as the square of the normalized B1 waveform
    # multiplied by the peak power of the pulse.
    # I am a little confused about this because Matlab asks for the waveform 
    # but that has size nx3 or nx4 but I assume that you just want the amplitude
    powerWaveform=peakPower*(RF1.waveform[:,1]**2)/np.amax(RF1.waveform[:,1]**2)
    # You probably want to account for varying step size here> Matlab doesn't.
    deltat_vec=tp/1000/np.sum(RF1.waveform[:,2])*RF1.waveform[:,2]
    return sum(powerWaveform*deltat_vec)

def rf_goia(npts,Tp,dx,tbw,gmax=4,ord_hs=4,ptype='inv',xshift=0,gamma=GAMMA_DICT['1H']):
    Tp=Tp/1000
    # Make npts even if it isn't
    if np.mod(npts,2)!=0:
        npts=npts+1
    halfn=int(npts/2)
    tvec=np.linspace(0,Tp,npts)
    # tau has N steps from -1 to 1 (useful for defining AM and GM functions)
    tau=tvec*2/Tp-1
    # Create a time vector from 0 to 1 that is N/2 in length (useful for 
    # creation of FM)
    tau2=tau[halfn:]
    # Create truncation factor. Note that numpy does not have an arcsech(x) function
    # so use arccosh(1/x)
    B=np.arccosh(1/0.01)
    # Find bandwidth factor A
    bw=tbw/Tp
    A=bw/2
    # Find time step size
    dt=tvec[1]-tvec[0]
    # Get gyromagnetic ratio in Hz/G (entered in MHz/T)
    gyro=gamma*1e6/1e4
    # First define the AM function
    F1=1/np.cosh(B*tau**ord_hs)
    # Now calculate the FM function based on teh assumption of a constant 
    # gradient by integrating the AM function
    F2a=np.zeros_like(F1)
    F2a[halfn:]=np.cumsum(F1[halfn:])
    F2a[:halfn]=-1*F2a[-1:halfn-1:-1]
    F2a=F2a/np.amax(F2a)
    # Calculate a static gradient value based on desired slice thickness and 
    # bandwidth. In Gauss/cm
    G=bw/(gyro*dx)
    # Calculate and x of t function based on a constant gradient using the
    # desired slice thickness. Never used (FM2 is never used)
    xoft=A*F2a/(gyro*G)
    # Now offset the x of t function by the amount given by xoffset
    xoft=xoft+xshift
    # Now define the GM function
    F3=(1-0.9/np.cosh(B*tau**2))
    # Now define the FM function that was derived from the solution to the 
    # differential equation in Tannus et al, NMR Biomed, 10:43-434 (1997). The
    # differential equation was solved by Jamie Near using Maple.
    # Note that we're overwriting G here.
    G=np.cosh(B*tau2**2)/((5*np.cosh(B*tau2**4+B*tau2**2))+(5*np.cosh(B*tau2**4-B*tau2**2))-(9*np.cosh(B*tau2**4)))
    h=(10*np.cosh(B*tau2**2)-9)/np.cosh(B*tau2**2)
    # The above function G must be integrated from 0 to t and divided by the 
    # the expression h to obtain the value of the FM function at time t for t>0.
    # For t<0, simply mirror the function about the origin.
    # Integrate the first half of G and multiply by h
    halfg=np.cumsum(G)
    halfg=halfg*(tau2[1]-tau2[0])*h
    fullg=np.r_[-1*halfg[-1::-1],halfg[:]]
    F2=fullg/np.amax(fullg)
    # Now calculate the max gradient strength based on slice thickness desired
    # (assume slice located at x=0)
    Ag=A*(1/gyro)*2/dx
    # GM is the max gradient value, so check if Ag is greater than 4 and, if
    # so, ask permission to reduce the A value of the pulse.
    # I don't really understand this because we change A, which affects FM and
    # the phase input, but we don't change Ag so gradient portion of the waveform
    # isn't changed. Does this matter?
    if Ag >= gmax:
        chg=input('Gradient too high for such a narrow slice. Allow reduction of tbw (y/n)? ')
        if chg.lower=='y':
            A=gmax*gyro*dx/2
            print('Reducing max gradient to {:.1f} G/cm.'.format(gmax))
            # Matlab has this as Tp/1000 but Tp is already divided by 1000.
            tbw=2*A*Tp
    GM=Ag*F3
    AM=F1
    FM=A*F2
    # Create new FM function based on x of t function and gradient function
    # This is never used???
    FM2=xoft*gyro*GM
    # If xshift is not zero then we need to adjust the FM function.
    FM=FM+GM*gyro*xshift
    rfwf=np.ones([npts,4])
    rfwf[:,0]=np.cumsum(FM)*dt*360
    rfwf[:,1]=AM
    rfwf[:,3]=GM
    # Now we can create the RF_pulse object. Note that Matlab simulates with 5 ms
    # regardless of Tp input. It also creates an inversion pulse.
    RF_out=RF_pulse(rfwf,ptype,Tp=0.005,f0=xshift*1000,gamma=gamma)
    # Note that, because RF_out has the frequency vector and profile with it, 
    # I am not returning these values as output arguments the way that Matlab
    # does. The user can call plot_freq_profile to see them.
    return RF_out,FM

def rf_hs(npts,ord_hs,tbw,Tp,trunc=0.01,thk=None,ptype='inv',gamma=GAMMA_DICT['1H']):
    # There is a lot of overlap between rf_goia and this so it's probably possible
    # to write this as a wrapper (I've added an ord_hs argument to rf_goia). 
    # Currently, no xshift is implemented here. This function assumes constant
    # gradient so is presumably more limited. Leaving both for now.
    
    # This function only allows constant gradient and, currently, xshift is not
    # implemented.
    # This is missing in Matlab even though it says Tp is in ms
    Tp=Tp/1000
    # Make npts even if it isn't
    if np.mod(npts,2)!=0:
        npts=npts+1
    halfn=int(npts/2)
    tvec=np.linspace(0,Tp,npts)
    # tau has N steps from -1 to 1 (useful for defining AM and GM functions)
    tau=tvec*2/Tp-1
    # Create a time vector from 0 to 1 that is N/2 in length (useful for 
    # creation of FM)
    tau2=tau[halfn:]
    # Create truncation factor. Note that numpy does not have an arcsech(x) function
    # so use arccosh(1/x)
    B=np.arccosh(1/trunc)
    # Find bandwidth factor A
    bw=tbw/Tp
    A=bw/2
    # Find time step size
    dt=tvec[1]-tvec[0]
    # Get gyromagnetic ratio in Hz/G (entered in MHz/T)
    gyro=gamma*1e6/1e4
    # First define the AM function
    F1=1/np.cosh(B*tau**ord_hs)
    # Now calculate the FM function based on teh assumption of a constant 
    # gradient by integrating the AM function
    F2a=np.zeros_like(F1)
    F2a[halfn:]=np.cumsum(F1[halfn:])
    F2a[:halfn]=-1*F2a[-1:halfn-1:-1]
    F2a=F2a/np.amax(F2a)
    # Calculate a static gradient value based on desired slice thickness and 
    # bandwidth. In Gauss/cm
    if thk is not None:
        G=bw/(gyro*thk)
    else:
        G=0
    AM=F1
    # So here G is just a constant, which I don't really understand. Seems easier
    # To produce a generalized version of rf_GOIA where the hypersecant order can be
    # entered
    GM=G*np.ones_like(AM)
    FM=A*F2a
    rfwf=np.ones([npts,3+(G!=0)])
    rfwf[:,0]=np.cumsum(FM)*dt*360
    rfwf[:,1]=AM
    if G!=0:
        rfwf[:,3]=GM
    # Now we can create the RF_pulse object. Note that Matlab simulates with 5 ms
    # regardless of Tp input. It also creates an inversion pulse, although it
    # also allows the user to input
    RF_out=RF_pulse(rfwf,ptype,Tp=0.005,f0=0,gamma=gamma)
    # Note that, because RF_out has the frequency vector and profile with it, 
    # I am not returning these values as output arguments the way that Matlab
    # does. The user can call plot_freq_profile to see them.
    return RF_out,FM

def rf_plotWaveform(RF1,mode='all',Tp=None,**kwargs):
    if mode=='gm' and not RF1.isGM:
        raise TypeError('ERROR: Cannot plot GM function for non-GM pulse. Aborting!')
    if Tp is None:
        Tp=RF1._Tp*1000
    tvec=RF1.get_tvec(tp=Tp)
    axdict={'ph':0,'amp':1,'gm':3}
    ylabs={'ph':'Phase (deg)','amp':'Amplitude (arb units)','gm':'Gradient (G/cm)'}
    if mode=='all':
        if RF1.isGM:
            axlist=['ph','amp','gm']
        else:
            axlist=['ph','amp']
    else:
        axlist=[mode]
    f1,ax1=plt.subplots(1,len(axlist))
    f1.set_size_inches([3*len(axlist),3])
    for eachax,axnm in zip(ax1,axlist):
        eachax.plot(tvec,RF1.waveform[:,axdict[axnm]],**kwargs)
        eachax.set_xlabel('Time (ms)')
        eachax.set_ylabel(ylabs[axnm])
    f1.set_facecolor('w')
    f1.tight_layout()
    return f1,ax1

def rf_refocusedComponent(RF1,tp=5,flipAngle=180,fspan=10):
    [mvx,fvecx]=rf_blochSim(RF1,tp,fspan=fspan,f0=0,peakB1=RF1.tw1/tp*(flipAngle/180),ph=0,npts=10000,M0=np.r_[1,0,0],display_output=False)
    [mvy,fvecy]=rf_blochSim(RF1,tp,fspan=fspan,f0=0,peakB1=RF1.tw1/tp*(flipAngle/180),ph=0,npts=10000,M0=np.r_[0,1,0],display_output=False)
    fxx=mvx[0,:]
    fxy=mvx[1,:]
    fyx=mvy[0,:]
    fyy=mvy[1,:]
    Imag=0.5*np.sqrt((fxx-fyy)**2+(fxy+fyx)**2)
    Iph=0.5*np.arctan((fxy+fyx)/(fyy-fxx))*180/np.pi
    f1,ax1=plt.subplots(2,1)
    ax1[0].plot(fvecx,Imag)
    ax1[0].set_ylabel('Magnitude')
    ax1[0].set_title('Refocused Component')
    ax1[1].plot(fvecx,Iph)
    if RF1.isGM:
        ax1[1].set_xlabel('Frequency (kHz)')
    else:
        ax1[1].set_xlabel('Position (cm)')
    ax1[1].set_ylabel('Phase (deg)')
    return Imag,Iph

def rf_resample(RF_in,N):
    # Matlab code only deals with the case where time samples are equal. I have
    # replicated this as best possible (resample_poly uses a zero-phase 
    # low-pass FIR filter but assumes equidistant points) but also added a
    # method to try to deal with unequal spacings (relatively untested).
    # Thought about using the expanded wf for this but it's debatable then how
    # to interpret N. Probably best just to not use this function for non-uniform
    # timesteps
    P=N
    Q=RF_in.npts
    RFcomplex=RF_in.get_complex_wf()
    tspace=RF_in.waveform[:,2]
    newwf=np.zeros([N,RF_in.waveform.shape[1]])
    if np.all(tspace==tspace[0]):
        RFresample=resample_poly(RFcomplex,P,Q,padtype='line')
        newwf[:,0]=np.round(10000*phase(RFresample)*180/np.pi)/10000
        newwf[:,1]=np.round(10000*np.abs(RFresample))/10000
        newwf[:,2]=tspace[0]*np.ones([N,])
        if RF_in.isGM:
            newwf[:,3]=resample_poly(RF_in.waveform[:,3],P,Q,padtype='line')
    else:
        print('WARNING: time intervals for waveform are not uniform. Using cubic spline interpolation.')
        tvec=RF_in.get_tvec()
        x=np.linspace(0,Q,Q)
        xnew=np.linspace(0,Q,N)
        f=interp1d(x,tspace,kind='nearest')
        new_tspace=f(xnew)
        tvec_new=(np.cumsum(new_tspace)-new_tspace[0])/sum(new_tspace)*RF_in._Tp
        cubspl=CubicSpline(tvec,RFcomplex)
        RFresample=cubspl(tvec_new)
        newwf[:,0]=np.round(10000*phase(RFresample)*180/np.pi)/10000
        newwf[:,1]=np.round(10000*np.abs(RFresample))/10000
        newwf[:,2]=new_tspace
        if RF_in.isGM:
            cubspl2=CubicSpline(tvec,RF_in.waveform[:,3])
            newwf[:,3]=cubspl2(tvec_new)
    if not RF_in.isPhsMod:
        newwf[:,0]=np.round(newwf[:,0])
        newwf[:,0]=newwf[:,0]+(newwf[:,0]==-180)*360
    RF_out=RF_in.copy()
    RF_out._waveform=newwf
    # The frequency profiles, etc. should be the same, although the on-resonance 
    # evolution in RF_out.plot_mvec doesn't have the right number of points.
    return RF_out
        
def rf_scaleGrad(RF_in,scale):
    if not isinstance(RF_in,RF_pulse):
        raise TypeError('ERROR: the input RF pulse must be of type RF_pulse. Try using io_loadRFwaveform to load a file. Aborting!')
    if not RF_in.isGM:
        raise FidAException('ERROR: the input RF pulse must be a gradient-modulated pulse. Aborting!')
    RF_out=RF_in.copy()
    RF_out.waveform[:,3]=RF_out.waveform[:,3]*scale
    RF_out._tbw=RF_out._tbw/scale
    # Need to change the frequency vector to reflect the narrower bandwidth. But
    # if pulse is not centered at 0 frequency than this will also change (because
    # the amount of accumlated extra phase due to the gradient changes). I 
    # considered altering the phase part of the waveform in order to keep f0 the
    # same but I think that it's a bad idea to do "hidden" things to the waveform.
    # The function is called scaleGrad so it should just scale the gradient and,
    # whatever results from that, changes in bandwidth or central frequency, is
    # what happens. So I just issue a warning that f0 will be shifted before
    # changing it.
    if RF_out.f0!=0:
        print('WARNING: centre frequency is non-zero. The Mz-frequency profile will shift with gradient scaling!')
        RF_out._f0=RF_out._f0/scale
    #bw=RF_out.bw
    RF_out._fvec=RF_in._fvec+(RF_out.f0-RF_in.f0)/1000#np.linspace(-bw+RF_out.f0/1000,bw+RF_out.f0/1000,len(RF_out._fvec))
    return RF_out

def rf_sinc(lobes,npts,ptype,tp=5,df=0,filter_flag=False,gamma=GAMMA_DICT['1H'],suppress_plots=False):
    tps=tp/1000
    # bandwidth depends on number of lobes
    # Oh, actually, here you can't really do df from the start because you don't
    # have a real time vector, so the 2*pi*df*t calculation is messed up
    # The Matlab code claims to return AMPINT but never calculates it. Seems to
    # be copied from rf_gauss code.
    if lobes < 1:
        raise ValueError('ERROR: sinc pulse must have at least 1 lobe! Aborting!')
    else:
        # Not totally clear to me how this relates to time but, as with gauss, 
        # choosing to make t negative will affect the relationship between 
        # lobes and bandwdith
        t=np.linspace(-1*(0.5+lobes/2),0.5+lobes/2,npts)
    # Will do the f0 shift at the end
    AMfunc=np.sinc(t)
    # Filter the using a cosine filter to minimize any ringing artefacts
    if filter_flag:
        print('Cosine filtering is on to minimize ringing. Check plot to ensure filter is not negative at the tails. Change lobes if so.')
        attn=float(input('By what factor would you like to attenuate the edges of the pulse? '))
        fcos=np.cos(t*np.pi/(np.pi/np.arccos(attn)*np.amax(t)))
        plt.figure()
        plt.plot(t,fcos)
        AMfunc=AMfunc*fcos
    rfwf=np.ones([npts,3])
    rfwf[:,0]=180*(AMfunc<0)
    rfwf[:,1]=np.abs(AMfunc)
 
    # Just generate a new RF_pulse object rather than running calculations individually
    rf_out=RF_pulse(rfwf,ptype,tps,f0=0,gamma=gamma,suppress_plots=True)
    # Shift central frequency and (if indicated) plot
    rf_out.f0=df
    if not suppress_plots:
        f1,ax1=plt.subplots(2,1)
        f1.set_size_inches([4,7])
        rf_out.plot_w1_profile(ax1=ax1[0])
        rf_out.plot_freq_profile(ax1=ax1[1])
        f1.tight_layout()
    
    AMPINT=rf_out.get_ampint(AMfunc)
    return rf_out, AMPINT

def rf_single2dualBand(rf_in,df=0,tp=None):
    # There is simplification to be done between this, dual band and combined wf.
    rf2=rf_freqshift(rf_in,F=df,Tp=tp)
    rf_out,AMPINT=rf_combineRF(rf_in, rf2)
    return rf_out,AMPINT
    
def rf_timeReverse(RF_in):
    if not isinstance(RF_in,RF_pulse):
        raise TypeError('ERROR: the input RF pulse must be of type RF_pulse. Try using io_loadRFwaveform to load a file. Aborting!')
    RF_out=RF_in.copy()
    RF_out._waveform=RF_out.waveform[::-1,:]
    # RF_out.rfCentre will automatically be re-calculated
    return RF_out

def rf_verse(RF_in,alpha):
    # Note that I haven't actually checked the timestep issue here but it's 
    # written like this in the Matlab code, explicity referencing the timesteps
    # so I'm hoping it checks out. At the end, it claims to have resampled 
    # everything back onto a linear time waveform, so we should just be able
    # to use a vector of ones there.
    if RF_in.f0!=0:
        print('WARNING: attempting to use rf_verse on off-resonance waveform. Shifting on-resonance to apply gradient, then will shift back.')
        RFtmp=rf_freqshift(RF_in,-1*RF_in.f0)
    else:
        RFtmp=RF_in.copy()
    newwf=RFtmp.waveform
    if newwf.shape[1]<4:
        raise FidAException('ERROR: Input waveform must already have a gradient waveform')
    if len(alpha)!=newwf.shape[0]:
        raise FidAException('ERROR: Gradient waveform does not match the length of the input RF pulse waveform. ABORTING!')
    newwf[:,1]=newwf[:,1]*alpha
    newwf[:,2]=newwf[:,2]/alpha
    newwf[:,3]=newwf[:,3]*alpha
    # Verify that the new time-step function has no negative values
    minduration=np.amin(newwf[:,2])
    if minduration<0:
        raise ValueError('ERROR: Resulting duration cannot be negative.')
    # Now resample all of the waveforms back onto a linear time waveform
    t_nonlin=np.cumsum(newwf[:,2])
    t_lin=np.linspace(t_nonlin[0],t_nonlin[-1],len(t_nonlin))
    # newwf is 2D but interp1d accepts this and creates interpolation along the specified dimension
    f=interp1d(t_nonlin,newwf,axis=0,kind='linear')
    rf2=f(t_lin)
    # But the timestep part of the waveform should be ones
    rf2[:,2]=np.ones([len(t_lin)])
    # Need to check that this works for off-resonance pulses or else add a warning
    if RF_in.f0!=0:
        dt=RF_in._Tp/np.sum(rf2[:,2])
        phaseRamp=(np.cumsum(rf2[:,3]*rf2[:,2])-rf2[0,3]*rf2[0,2])*(RF_in.gamma/10)*dt*RF_in.f0*360
        rf2[:,0]=rf2[:,0]+phaseRamp
        rf_out=RF_pulse(rf2,RF_in.pulse_type,Tp=RF_in._Tp,f0=RF_in.f0,gamma=RF_in.gamma)
    else:
        rf_out=RF_pulse(rf2,RF_in.pulse_type,Tp=RF_in._Tp,f0=0,gamma=RF_in.gamma)
    return rf_out


if __name__ == '__main__':
    """
    for debugging
    """
    from pyFidA.fidA_io import io_loadRFwaveform, io_readpta, io_readRFtxt
    import time
    pname='/Users/nearlabmacbook1/Documents/PythonScripts/pyFidA/exampleData/rfPulses'
    ptype='inv'
    newf0=1000
    Tp=5/1000
    
    RFtest=io_loadRFwaveform(os.path.join(pname,'GOIA_tthk0.01_R120.txt'),ptype)
    tmprf=RFtest.get_expanded_wf()
    #print(RFtest.isAdiabatic)
    #RFtest2=io_loadRFwaveform(os.path.join(pname,'sampleExcPulse.pta'),ptype)
    
    #rf_off=RFtest.copy()
    #rf_off.f0=800
    #RF3=rf_verse(rf_off,2*np.ones([RFtest.npts,]))
    #rf_combined,ampint=rf_combineRF(RFtest2,rf_off)
    
    #print(rf_off.get_ampint())
    #tmpwf=RFtest.waveform
    #tmpwf[5,2]=4
    #rflong=RF_pulse(tmpwf,ptype,Tp)
    #check=RFtest.get_expanded_wf()
    #print("original wf length: {:d}, OG expanded wf length: {:d}".format(RFtest.waveform.shape[0],check.shape[0]))
    #check2=rflong.get_expanded_wf()
    #print("long wf length: {:d}, long expanded wf length: {:d}".format(rflong.waveform.shape[0],check2.shape[0]))
                      
    # RFtest=io_loadRFwaveform(os.path.join(pname,'GOIA_tthk0.01_R120.txt'),ptype)
    # RF1_gmod_fnew=RFtest.copy()
    # RF1_gmod_fnew.f0=newf0
    # rf_gmod_fnew=RF1_gmod_fnew.waveform.copy()
    # rf_gmod=RFtest.waveform.copy()
    # RFtest2=io_loadRFwaveform(os.path.join(pname,'sampleExcPulse.pta'),type_p=ptype)
    # RF1_nograd_fnew=RFtest.copy()
    # RF1_nograd_fnew.f0=newf0
    # rf_nograd_fnew=RF1_nograd_fnew.waveform.copy()
    # rf_nograd=RFtest.waveform.copy()
    # Tp=5/1000
    # f2,ax2=plt.subplots(2,2)
    # tic=time.perf_counter()
    # tw1_nograd,B1vec_nograd,w1_profile_nograd=_calc_tw1max(ptype,rf_nograd,Tp,f0=0)
    # toc=time.perf_counter()
    # t_f0_nograd=toc-tic
    # tic=time.perf_counter()
    # tw1_gmod,B1vec_gmod,w1_profile_gmod=_calc_tw1max(ptype,rf_gmod,Tp,f0=0)
    # toc=time.perf_counter()
    # t_f0_gmod=toc-tic
    # ax2[0,0].plot(B1vec_nograd,w1_profile_nograd[2,:])
    # ax2[1,0].plot(B1vec_gmod,w1_profile_gmod[2,:])
    # tic=time.perf_counter()
    # tbw_nograd,fvec_nograd,pulse_freq_profile_nograd=_calc_tbw(ptype,rf_nograd,Tp,f0=0,w1max=tw1_nograd/Tp)
    # toc=time.perf_counter()
    # tbw_f0_nograd=toc-tic
    # tic=time.perf_counter()
    # tbw_gmod,fvec_gmod,pulse_freq_profile_gmod=_calc_tbw(ptype,rf_gmod,Tp,f0=0,w1max=tw1_gmod/Tp)
    # toc=time.perf_counter()
    # tbw_f0_gmod=toc-tic
    # ax2[0,1].plot(fvec_nograd,pulse_freq_profile_nograd[2,:])
    # ax2[1,1].plot(fvec_gmod,pulse_freq_profile_gmod[2,:])
    # tic=time.perf_counter()
    # tw1_nograd_fnew,B1vec_nograd_fnew,w1_profile_nograd_fnew=_calc_tw1max(ptype,rf_nograd_fnew,Tp,f0=newf0)
    # toc=time.perf_counter()
    # t_fnew_nograd=toc-tic
    # tic=time.perf_counter()
    # tw1_gmod_fnew,B1vec_gmod_fnew,w1_profile_gmod_fnew=_calc_tw1max(ptype,rf_gmod_fnew,Tp,f0=newf0)
    # toc=time.perf_counter()
    # t_fnew_gmod=toc-tic
    # ax2[0,0].plot(B1vec_nograd_fnew,w1_profile_nograd_fnew[2,:])
    # ax2[1,0].plot(B1vec_gmod_fnew,w1_profile_gmod_fnew[2,:])
    # tic=time.perf_counter()
    # tbw_nograd_fnew,fvec_nograd_fnew,pulse_freq_profile_nograd_fnew=_calc_tbw(ptype,rf_nograd_fnew,Tp,f0=newf0,w1max=tw1_nograd_fnew/Tp)
    # toc=time.perf_counter()
    # tbw_fnew_nograd=toc-tic
    # tic=time.perf_counter()
    # tbw_gmod_fnew,fvec_gmod_fnew,pulse_freq_profile_gmod_fnew=_calc_tbw(ptype,rf_gmod_fnew,Tp,f0=newf0,w1max=tw1_gmod_fnew/Tp)
    # toc=time.perf_counter()
    # tbw_fnew_gmod=toc-tic
    # ax2[0,1].plot(fvec_nograd_fnew,pulse_freq_profile_nograd_fnew[2,:])
    # ax2[1,1].plot(fvec_gmod_fnew,pulse_freq_profile_gmod_fnew[2,:])
    # print('w1max f0=0 no grad: '+str(tw1_nograd/Tp))
    # print('w1max f0=0 with grad: '+str(tw1_gmod/Tp))
    # print('w1max f0='+str(newf0)+' no grad: '+str(tw1_nograd_fnew/Tp))
    # print('w1max f0='+str(newf0)+' with grad: '+str(tw1_gmod_fnew/Tp))
    # print('bw f0=0 no grad: '+str(tbw_nograd/Tp))
    # print('bw f0=0 with grad: '+str(tbw_gmod/Tp))
    # print('bw f0='+str(newf0)+' no grad: '+str(tbw_nograd_fnew/Tp))
    # print('bw f0='+str(newf0)+' with grad: '+str(tbw_gmod_fnew/Tp))
    # print('Time for tw1 with f0, no grad: '+str(t_f0_nograd))
    # print('Time for tw1 with f0, gradient: '+str(t_f0_gmod))
    # print('Time for tw1 with f1000, no grad: '+str(t_fnew_nograd))
    # print('Time for tw1 with f1000, gradient: '+str(t_fnew_gmod))
    # print('Time for tbw with f0, no grad: '+str(tbw_f0_nograd))
    # print('Time for tbw with f0, gradient: '+str(tbw_f0_gmod))
    # print('Time for tbw with f1000, no grad: '+str(tbw_fnew_nograd))
    # print('Time for tbw with f1000, gradient: '+str(tbw_fnew_gmod))
    
    
    #RFtest=io_loadRFwaveform(os.path.join(pname,'sampleRefocPulse.pta'),type_p='inv')
    #RFtest.plot_freq_profile()
    #RFtest.f0=1200
    #RFtest.plot_freq_profile()
    #tbw,fvec2,pulse_freq_profile2=_calc_tbw(RFtest.pulse_type,RFtest.waveform,RFtest._Tp,1200,RFtest.get_w1max(RFtest._Tp))
    #plt.figure()
    #plt.plot(fvec2,pulse_freq_profile2[2,:])
    # tp=5
    # tpw1new,B1vec,w1_profile=_calc_tw1max(RFtest.pulse_type,RFtest.waveform,tp/1000,f0=1000)
    # tbw,fvec,pulse_freq_profile=_calc_tbw(RFtest.pulse_type,RFtest.waveform,tp/1000,1000,tpw1new/(tp/1000))
    # # This frequency profile doesn't change even if you enter a different f0??
    # # Note that it does for gradient-modulated
    # plt.figure()
    # plt.plot(fvec,pulse_freq_profile[2,:])
    
    #[mv,sc]=rf_blochSim(RFtest,tp=5,fspan=10,f0=0,ph=0,M0=np.r_[0,0,1],display_output=True)