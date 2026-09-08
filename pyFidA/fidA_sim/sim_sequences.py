#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:40:34 2026
pyFidA.fidA_sim.sim_sequences.py

@author: Colleen Bailey (@cbailey@sri.utoronto.ca), based on Matlab code by 
Jamie Near and Robin Simpson

Functions to simulate the results of pulse sequences using the density matrix
formalism. For the basic operators that make up the pulse sequences 
(excitation, evolution, etc., see pyFidA.fidA_sim.sim_operators.py.

Functions:
    sim_cosy,
    sim_laser,
    sim_megapress,
    sim_megapress_shaped,
    sim_megapress_shapedEdit,
    sim_megapress_shapedRefoc,
    sim_megaspecial_shaped,
    sim_onepulse,
    sim_onepulse_arbPh,
    sim_onepulse_delay,
    sim_onepulse_shaped,
    sim_press,
    sim_press_shaped,
    sim_press_shaped_phCyc,
    sim_semiLASER_shaped,
    sim_semiLASER_shaped_phCyc,
    sim_spinecho,
    sim_spinecho_shaped,
    sim_spinecho_xN,
    sim_steam,
    sim_steam_shaped
"""
import numpy as np
from .sim_operators import sim_COF, sim_Hamiltonian, sim_excite, sim_excite_arbPh, \
    sim_evolve, sim_gradSpoil, sim_rotate, sim_readout, sim_shapedRF
from pyFidA.fidA_rf import rf_scaleGrad, rf_timeReverse

def sim_cosy(npts,sw,Bfield,linewidth,spinSys,npts2,centerFreq=4.65,centerFreq_label=None):
    """
    Simulates an ideal 2D COSY sequence with indirect dimension having npts2 
    delays (spectral width is assumed to be the same for both dimensions)

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum, direct dimension.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    npts2 : int
        Number of points in the indirect dimension. Spectral width is assumed
        to be the same as the direct dimension, meaning that the delays run
        from 0 to npts2/sw and have resolution 1/sw.
    centerFreq : centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq. 

    Returns
    -------
    out1 : list of pyFidA.FID objects
        Output spectra, with each spectrum in the indirect dimension as a 
        separate list element.

    """
    df=sw/npts2
    # delay_vec in ms, divide by 1000 to get in microseconds before sending to sim_evolve
    delay_vec=np.r_[0:npts2/df:1/df]
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    out1=list()
    for delay in delay_vec:
        d3=sim_evolve(d2,H1,delay)
        d3=sim_excite(d3,H1,whichax='x',anglein=90)
        outtmp,dfinal=sim_readout(d3,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=0,center_freq_ppm=centerFreq_label)
        outtmp.sequence='cosy'
        outtmp.sim='ideal'
        outtmp.te=delay
        out1.append(outtmp)
    ### END PULSE SEQUENCE
    return out1

def sim_laser(npts,sw,Bfield,linewidth,spinSys,TE,centerFreq=4.65,centerFreq_label=None):
    """
    Simulates an ideal LASER experiment with total echo time TE and six
    equally spaced echoes.
    
    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    TE : float
        Echo time in ms.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq. 

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the LASER sequence.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    # Assume equal delays between all echoes (divide by 6), and convert from ms to s.
    tau=TE/6/1000
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,tau/2) #Evolve by tau/2
    for refoc_idx in range(5):
        d2=sim_rotate(d2,H1,180,'y') #180 refocusing pulse about y axis.
        d2=sim_evolve(d2,H1,tau) #Evolve by tau
    d2=sim_rotate(d2,H1,180,'y') #Sixth 180 refocusing pulse about y axis.
    d2=sim_evolve(d2,H1,tau/2) #Evolve by tau/2
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='laser'
    out1.sim='ideal'
    out1.te=TE
    return out1

def sim_megapress(npts,sw,Bfield,linewidth,spinSys,taus,refoc1Flip,refoc2Flip,editFlip,centerFreq=3,centerFreq_label=None):
    """
    Simulates an ideal MEGAPRESS experiment with instantaneous localization and
    editing pulses. Provides the ability to specify the flip angle of each
    refocusing pulse and editing pulse on each spin in the spin system.
    
    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    taus : list or numpy array
        Pulse sequence timing vector:
            taus[0]: time in ms from first 90 to 180
            taus[1]: time in ms from 1st 180 to 1st edit pulse
            taus[2]: time in ms from 1st edit pulse to 2nd 180
            taus[3]: time in ms from 2nd 180 to 2nd edit pulse
            taus[4]: time in ms from 2nd edit pulse to ADC
    refoc1Flip : float or numpy array or list
        Flip angles(s) for first refocusing pulse in the sequence. Can be the
        same for all spins, specified for each non-interacting part of the spin
        system or unique for each spin.
    refoc2Flip : float or numpy array or list
        Flip angles(s) for second refocusing pulse in the sequence. Can be the
        same for all spins, specified for each non-interacting part of the spin
        system or unique for each spin.
    editFlip : float or numpy array or list
        Flip angles(s) for edit pulses in the sequence (same flip angle for
        both pulses). Can be the same for all spins, specified for each
        non-interacting part of the spin system or unique for each spin.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 3 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq. 

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the MEGA-PRESS sequence.

    """
    
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    taus=[tval/1000 for tval in taus]
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,taus[0]) #Evolve
    for tval,mp_pulse in zip(taus[1:],[refoc1Flip,editFlip,refoc2Flip,editFlip]):
        d2=sim_rotate(d2,H1,mp_pulse,'y') #Refocusing or editing pulse about y axis.
        d2=sim_evolve(d2,H1,tval) #Evolve
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='megapress'
    out1.sim='ideal'
    out1.te=sum(taus)
    return out1

def sim_megapress_shaped(npts,sw,Bfield,linewidth,spinSys,taus,editPulse,editTp,editPh1,editPh2,refPulse,refTp,dx,dy,Gx,Gy,refPh1,refPh2,centerFreq=3,centerFreq_label=None):
    """
    Simulates the MEGA-PRESS sequence with shaped localization and editing 
    pulses. Enables choice of the timings of all of the rf pulses as well as 
    the choice of the phase of both the editing pulse and the refocusing 
    pulses. This allows phase cycling of the editing and refocusing pulses by 
    repeating simulations with different editing pulse phases, which is 
    necessary to remove phase artefacts from the editing pulses. For the 
    editing pulses, an eight step phase cycling scheme is typically sufficient, 
    where the first editing pulse is cycled by 0 and 90 degrees, and the second 
    editing pulse is cycled by 0, 90, 180 and 270 degrees, and all phase cycles 
    should be added together to remove unwanted coherences. For the refocusing 
    pulses, a four step phase cycling scheme is typically sufficient, where 
    both refocusing pulses are phase cycled by 0 and 90 degrees, and the phases 
    are combined in the following way:
        signal = ([0 90] - [0 0]) + ([90 0] - [90 90])
    where, in [X Y], X is the phase of the first refocusing pulse and Y is the 
    phase of the second refocusing pulse
    
    Note that this code only simulates one subspectrum at a time (edit-on or 
    edit-off). The difference spectrum can be obtained by simulating one of
    each, and then subtracting.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    taus : list or numpy array
        Pulse sequence timing vector:
            taus[0]: time in ms from first 90 to 180
            taus[1]: time in ms from 1st 180 to 1st edit pulse
            taus[2]: time in ms from 1st edit pulse to 2nd 180
            taus[3]: time in ms from 2nd 180 to 2nd edit pulse
            taus[4]: time in ms from 2nd edit pulse to ADC
    editPulse : pyFidA.RF_pulse object
        RF pulse for the editing pulses (obtained using pyFidA.io_loadRFwaveform).
    editTp : float
        Duration of editing pulse in ms.
    editPh1 : float
        Phase of the first editing pulse in degrees.
    editPh2 : float
        Phase of the second editing pulse in degrees.
    refPulse : pyFidA.RF_pulse object
        RF pulse for the refocusing pulses (obtained using pyFidA.io_loadRFwaveform).
    refTp : float
        Duration of refocusing pulse in ms.    
    dx : float
        Position offset in x-direction in cm (corresponding to first refocusing 
        pulse).
    dy : float
        Position offset in y-direction in cm (corresponding to second 
        refocusing pulse).
    Gx : float
        Gradient strength for first selective refocusing pulse in G/cm.
    Gy : float
        Gradient strength for second selective refocusing pulse in G/cm.
    refPh1 : float
        Phase of the first refocusing pulse in degrees.
    refPh2 : float
        Phase of the second refocusing pulse in degrees.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 3 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.
        
    Raises
    ------
    ValueError
        Delay values cannot be negative after subtracting pulse durations.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the MEGA-PRESS sequence.
        
    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    # Calculate new delays by subtracting the pulse durations from the taus vector
    delays=[(tval-pulseval/2)/1000 for tval,pulseval in zip(taus,[refTp,refTp+editTp,editTp+refTp,refTp+editTp,editTp])]
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following taus are too short: ' + str([didx for didx,dval in enumerate(delays) if dval<0]) +'.')
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,delays[0]) #Evolve by delays[0]
    d2=sim_shapedRF(d2,H1,refPulse,refTp,180,90+refPh1,dx,Gx) #1st shaped 180 degree refocusing pulse
    d2=sim_evolve(d2,H1,delays[1]) #Evolve by taus[1]
    d2=sim_shapedRF(d2,H1,editPulse,editTp,180,90+editPh1) #1st shaped editing pulse rotation
    d2=sim_evolve(d2,H1,delays[2]) #Evolve by delays[2]
    d2=sim_shapedRF(d2,H1,refPulse,refTp,180,90+refPh2,dy,Gy) #2nd shaped 180 degree refocusing pulse
    d2=sim_evolve(d2,H1,delays[3]) #Evolve by delays[3]
    d2=sim_shapedRF(d2,H1,editPulse,editTp,180,90+editPh2) #2nd shaped editing pulse rotation
    d2=sim_evolve(d2,H1,delays[4]) #Evolve by delays[4]
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='megapress'
    out1.sim='shaped'
    out1.te=sum(taus)
    return out1

def sim_megapress_shapedEdit(npts,sw,Bfield,linewidth,spinSys,taus,editPulse,editTp,editPh1,editPh2,centerFreq=3,centerFreq_label=None):
    """
    This function simulates the MEGA-PRESS sequence with instantaneous
    localization pulses and shaped editing pulses. Enables choice of the 
    timings of all of the rf pulses as well as the choice of the phase of the
    editing pulse. This allows phase cycling of the editing pulses. For the 
    editing pulses, an eight step phase cycling scheme is typically sufficient, 
    where the first editing pulse is cycled by 0 and 90 degrees, and the second 
    editing pulse is cycled by 0,90,180, and 270 degrees, and all phase cycles 
    should be added together to remove unwanted coherences.
    
    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    taus : list or numpy array
        Pulse sequence timing vector:
            taus[0]: time in ms from first 90 to 180
            taus[1]: time in ms from 1st 180 to 1st edit pulse
            taus[2]: time in ms from 1st edit pulse to 2nd 180
            taus[3]: time in ms from 2nd 180 to 2nd edit pulse
            taus[4]: time in ms from 2nd edit pulse to ADC
    editPulse : pyFidA.RF_pulse object
        RF pulse for the editing pulses (obtained using pyFidA.io_loadRFwaveform).
    editTp : float
        Duration of editing pulse in ms.
    editPh1 : float
        Phase of the first editing pulse in degrees.
    editPh2 : float
        Phase of the second editing pulse in degrees.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 3 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.
    
    Raises
    ------
    ValueError
        Delay values cannot be negative after subtracting pulse durations.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the MEGA-PRESS sequence.
            
    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    # Calculate new delays by subtracting the pulse durations from the taus vector
    delays=[taus[0]/1000]+[(tval-editTp/2)/1000 for tval in taus[1:]]
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following taus are too short: ' + str([didx for didx,dval in enumerate(delays) if dval<0]) +'.')
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,delays[0]) #Evolve by delays[0]
    d2=sim_rotate(d2,H1,180,'y') #1st instantaneous 180 degree refocusing pulse about y
    d2=sim_evolve(d2,H1,delays[1]) #Evolve by taus[1]
    d2=sim_shapedRF(d2,H1,editPulse,editTp,180,90+editPh1) #1st shaped editing pulse rotation
    d2=sim_evolve(d2,H1,delays[2]) #Evolve by delays[2]
    d2=sim_rotate(d2,H1,180,'y') #2nd instantaneous 180 degree refocusing pulse
    d2=sim_evolve(d2,H1,delays[3]) #Evolve by delays[3]
    d2=sim_shapedRF(d2,H1,editPulse,editTp,180,90+editPh2) #2nd shaped editing pulse rotation
    d2=sim_evolve(d2,H1,delays[4]) #Evolve by delays[4]
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='megapress'
    out1.sim='shaped'
    out1.te=sum(taus)
    return out1

def sim_megapress_shapedRefoc(npts,sw,Bfield,linewidth,spinSys,taus,editFlip,refPulse,refTp,dx,dy,Gx,Gy,refPh1,refPh2,centerFreq=3,centerFreq_label=None):
    """
    This function simulates the MEGA-PRESS sequence with shaped localization 
    pulses and instantaneous editing pulses. Enables choice of the timings of 
    all of the rf pulses as well as the choice of the phase of both the editing 
    pulse and the refocusing pulses. This allows phase cycling of the editing 
    and refocusing pulses. For the editing pulses, an eight step phase cycling 
    scheme is typically sufficient, where the first editing pulse is cycled by 
    0 and 90 degrees, and the second editing pulse is cycled by 0, 90, 180, and 
    270 degrees, and all phase cycles should be added together to remove 
    unwanted coherences. For the refocusing pulses, a four step phase cycling 
    scheme is typically sufficient, where both refocusing pulses are phase 
    cycled by 0 and 90 degrees, and the phases are combined in the following 
    way:
        signal = ([0 90] - [0 0]) + ([90 0] - [90 90])
    where, in [X Y], X is the phase of the first refocusing pulse and Y is the
    phase of the second refocusing pulse.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    taus : list or numpy array
        Pulse sequence timing vector:
            taus[0]: time in ms from first 90 to 180
            taus[1]: time in ms from 1st 180 to 1st edit pulse
            taus[2]: time in ms from 1st edit pulse to 2nd 180
            taus[3]: time in ms from 2nd 180 to 2nd edit pulse
            taus[4]: time in ms from 2nd edit pulse to ADC
    editFlip : float or numpy array or list
        Flip angles(s) for edit pulses in the sequence (same flip angle for
        both pulses). Can be the same for all spins, specified for each
        non-interacting part of the spin system or unique for each spin.
    refPulse : pyFidA.RF_pulse object
        RF pulse for the refocusing pulses (obtained using pyFidA.io_loadRFwaveform).
    refTp : float
        Duration of refocusing pulse in ms.    
    dx : float
        Position offset in x-direction in cm (corresponding to first refocusing 
        pulse).
    dy : float
        Position offset in y-direction in cm (corresponding to second 
        refocusing pulse).
    Gx : float
        Gradient strength for first selective refocusing pulse in G/cm.
    Gy : float
        Gradient strength for second selective refocusing pulse in G/cm.
    refPh1 : float
        Phase of the first refocusing pulse in degrees.
    refPh2 : float
        Phase of the second refocusing pulse in degrees.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 3 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Raises
    ------
    ValueError
        Delay values cannot be negative after subtracting pulse durations.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the MEGA-PRESS sequence.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    # Calculate new delays by subtracting the pulse durations from the taus vector
    delays=[(tval-refTp/2)/1000 for tval in taus[:-1]]+[taus[-1]/1000]
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following taus are too short: ' + str([didx for didx,dval in enumerate(delays) if dval<0]) +'.')
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,delays[0]) #Evolve by delays[0]
    d2=sim_shapedRF(d2,H1,refPulse,refTp,180,90+refPh1,dx,Gx) #1st shaped 180 degree refocusing pulse
    d2=sim_evolve(d2,H1,delays[1]) #Evolve by delays[1]
    d2=sim_rotate(d2,H1,editFlip,'y') #1st editing pulse rotation
    d2=sim_evolve(d2,H1,delays[2]) #Evolve by delays[2]
    d2=sim_shapedRF(d2,H1,refPulse,refTp,180,90+refPh2,dy,Gy) #2nd shaped 180 degree refocusing pulse
    d2=sim_evolve(d2,H1,delays[3]) #Evolve by delays[3]
    d2=sim_rotate(d2,H1,editFlip,'y') #2nd editing pulse rotation
    d2=sim_evolve(d2,H1,delays[4]) #Evolve by delays[4]
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='megapress'
    out1.sim='shaped'
    out1.te=sum(taus)
    return out1

def sim_megaspecial_shaped(npts,sw,Bfield,linewidth,spinSys,taus,editPulse,editTp,editPh1,editPh2,refPulse,refTp,dx,Gx,refPh,centerFreq=3,centerFreq_label=None):
    """
    Simulates the MEGA-SPECIAL sequence with shaped localization and editing 
    pulses. Enables choice of the timings of all of the rf pulses as well as 
    the choice of the phase of both the editing and refocusing pulses. This 
    allows phase cycling of the editing and refocusing pulses. For the editing 
    pulses, an eight step phase cycling scheme is typically sufficient, where 
    the first editing pulse is cycled by 0 and 90 degrees, and the second 
    editing pulse is cycled by 0, 90, 180, and 270 degrees. For the refocusing 
    pulse, a two step phase cycling scheme is typically sufficient, where the 
    refocusing pulses is phase cycled by 0 and 90 degrees.
    
    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    taus : list or numpy array
        Pulse sequence timing vector:
            taus[0]: time in ms from first 90 to 1st edit pulse
            taus[1]: time in ms from 1st edit pulse to 180
            taus[2]: time in ms from 180 to 2nd edit pulse
            taus[3]: time in ms from 2nd edit pulse to ADC
    editPulse : pyFidA.RF_pulse object
        RF pulse for the editing pulses (obtained using pyFidA.io_loadRFwaveform).
    editTp : float
        Duration of editing pulse in ms.
    editPh1 : float
        Phase of the first editing pulse in degrees.
    editPh2 : float
        Phase of the second editing pulse in degrees.
    refPulse : pyFidA.RF_pulse object
        RF pulse for the refocusing pulses (obtained using pyFidA.io_loadRFwaveform).
    refTp : float
        Duration of refocusing pulse in ms.    
    dx : float
        Position offset in x-direction in cm (corresponding to refocusing pulse).
    Gx : float
        Gradient strength for selective refocusing pulse in G/cm.
    refPh : float
        Phase of the refocusing pulse in degrees.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 3 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Raises
    ------
    ValueError
        Delay values cannot be negative after subtracting pulse durations.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the MEGA-SPECIAL sequence.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    # Calculate new delays by subtracting the pulse durations from the taus vector
    delays=[(tval-pulsedur/2)/1000 for tval,pulsedur in zip(taus,[editTp,editTp+refTp,refTp+editTp,editTp])]
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following taus are too short: ' + str([didx for didx,dval in enumerate(delays) if dval<0]) +'.')
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,delays[0]) #Evolve by delays[0]
    d2=sim_shapedRF(d2,H1,editPulse,editTp,180,90+editPh1) #1st shaped editing pulse
    d2=sim_evolve(d2,H1,delays[1]) #Evolve by delays[1]
    d2= sim_shapedRF(d2,H1,refPulse,refTp,180,90+refPh,dx,Gx) #1st shaped 180 degree refocusing pulse
    d2=sim_evolve(d2,H1,delays[2]) #Evolve by delays[2]
    d2=sim_shapedRF(d2,H1,editPulse,editTp,180,90+editPh2) #2nd shaped editing pulse rotation
    d2=sim_evolve(d2,H1,delays[3]) #Evolve by delays[4]
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='megaspecial'
    out1.sim='shaped'
    out1.te=sum(taus)
    return out1

def sim_onepulse(npts,sw,Bfield,linewidth,spinSys,anglein=90,ph1=None,centerFreq=4.65,centerFreq_label=None):
    """
    Simulates a pulse-acquire experiment with an ideal (instantaneous) 
    excitation pulse and an assumed Lorentzian lineshape.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    anglein : float or numpy array or list, optional
        Flip angles(s) for excitation pulse in the sequence. Can be the same 
        for all spins, specified for each non-interacting part of the spin 
        system or unique for each spin. The default is 90.
    ph1 : float, optional
        Phase of excitation pulse. The default is None, which excites along x.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using pulse-acquire sequence.

    """
    if ph1 is None:
        excite_func=sim_excite
        excite_kwargs={'whichax':'x','anglein':anglein}
    else:
        excite_func=sim_excite_arbPh
        excite_kwargs={'ph_ax':ph1,'anglein':anglein}
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=excite_func(d1,H1,**excite_kwargs)
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label)
    ### END PULSE SEQUENCE
    out1.sequence='onepulse'
    out1.sim='ideal'
    return out1

def sim_onepulse_arbPh(npts,sw,Bfield,linewidth,spinSys,anglein=90,ph1=0,centerFreq=4.65,centerFreq_label=None):
    """
    Simulates a pulse-acquire experiment with an ideal (instantaneous) 
    excitation pulse and an assumed Lorentzian lineshape. This function is a
    wrapper for sim_onepulse, although both allow excitation pulse phase; it
    simply allows the function names from Matlab to be preserved.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    anglein : float or numpy array or list, optional
        Flip angles(s) for excitation pulse in the sequence. Can be the same 
        for all spins, specified for each non-interacting part of the spin 
        system or unique for each spin. The default is 90.
    ph1 : float, optional
        Phase of excitation pulse. The default is None, which excites along x.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using pulse-acquire sequence.

    """
    out1=sim_onepulse(npts,sw,Bfield,linewidth,spinSys,anglein=anglein,ph1=ph1,centerFreq=centerFreq,centerFreq_label=centerFreq_label)
    return out1

def sim_onepulse_delay(npts,sw,Bfield,linewidth,spinSys,delay,anglein=90,ph1=None,centerFreq=4.65,centerFreq_label=None):
    """
    Simulates a pulse-acquire experiment with an ideal (instantaneous) 
    excitation pulse and an assumed Lorentzian lineshape. A delay is included
    before the ADC to simulate the effect of a first order phase shift.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    delay : float
        Delay in ms between excitation pulse and ADC readout.
    anglein : float or numpy array or list, optional
        Flip angles(s) for excitation pulse in the sequence. Can be the same 
        for all spins, specified for each non-interacting part of the spin 
        system or unique for each spin. The default is 90.
    ph1 : float, optional
        Phase of excitation pulse. The default is None, which excites along x.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using pulse-acquire sequence.

    """
    if ph1 is None:
        excite_func=sim_excite
        excite_kwargs={'whichax':'x','anglein':anglein}
    else:
        excite_func=sim_excite_arbPh
        excite_kwargs={'ph_ax':ph1,'anglein':anglein}
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=excite_func(d1,H1,**excite_kwargs)
    d2=sim_evolve(d2,H1,delay/1000)
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label)
    ### END PULSE SEQUENCE
    out1.sequence='onepulse'
    out1.sim='ideal'
    out1.te=delay
    return out1

def sim_onepulse_shaped(npts,sw,Bfield,linewidth,spinSys,RF1,tp,phCyc,dfdx=0,G=None,centerFreq=4.65,centerFreq_label=None):
    """
    This function simulates the effect of a frequency-selective or slice-
    selective excitation, followed immediately by the acquisition window.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    RF1 : pyFidA.RF_pulse object
        RF pulse for the excitation (obtained using pyFidA.io_loadRFwaveform).
    tp : float
        RF pulse duration in ms.
    phCyc : float
        Phase of excitation pulse in degrees
    dfdx : float, optional
        If simulating a frequency-selective pulse, this argument should be the
        frequency offset in Hz. If simulating a slice-selective pulse, this
        argument should be the position offset in cm. The default is 0.
    grad : float, optional
        Gradient strength for slice-selective pulse in G/cm. Only used if 
        RFpulse.isGM=False. The default is None.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using pulse-acquire sequence.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_shapedRF(d1,H1,RF1,tp,90,phCyc,dfdx,G)
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label)
    ### END PULSE SEQUENCE
    out1.sequence='onepulse'
    out1.sim='shaped'
    out1.te=tp/2
    return out1

def sim_press(npts,sw,Bfield,linewidth,spinSys,tau1,tau2,centerFreq=4.65,centerFreq_label=None):
    """
    Simulates an ideal PRESS experiment with first echo time tau1 and a second
    echo time tau2.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    tau1 : float
        Echo time in ms of first PRESS spin echo.
    tau2 : float
        Echo time in ms of second PRESS spin echo.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using PRESS sequence.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,tau1/2000) #Evolve by tau1/2
    d2=sim_rotate(d2,H1,180,'y') #First 180 refocusing pulse about y axis.
    d2=sim_evolve(d2,H1,(tau1+tau2)/2000) #Evolve by (tau1+tau2)/2
    d2=sim_rotate(d2,H1,180,'y') #Second 180 refocusing pulse about y axis
    d2=sim_evolve(d2,H1,tau2/2000) #Evolve by tau2/2
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='press'
    out1.sim='ideal'
    out1.te=tau1+tau2
    return out1

def sim_press_shaped(npts,sw,Bfield,linewidth,spinSys,tau1,tau2,RF1,tp,dx,dy,Gx,Gy,flipAngle=180,centerFreq=4.65,centerFreq_label=None):
    """
    Simulates a PRESS experiment where the excitation is simulated as an 
    instantaneous rotation and the refocusing pulses are simulated as shaped
    rotations. It employs coherence selection to only include desired coherence
    orders (rather than requiring separate simulations with phase cycling). The 
    code simulates the spectrum at a given point in space (dx,dy), given the 
    slice selection gradients Gx and Gy. To simulate the PRESS experiment, the 
    simulation must be repeated at various points in space and the resulting 
    spectra added together.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    tau1 : float
        Echo time in ms of first PRESS spin echo.
    tau2 : float
        Echo time in ms of second PRESS spin echo.
    RF1 : pyFidA.RF_pulse object
        RF pulse for refocusing (obtained using pyFidA.io_loadRFwaveform).
    tp : float
        RF pulse duration in ms.
    dx : float
        Position offset in x-direction in cm (corresponding to first refocusing 
        pulse).
    dy : float
        Position offset in y-direction in cm (corresponding to second 
        refocusing pulse).
    Gx : float
        Gradient strength for first selective refocusing pulse in G/cm OR, in 
        the case that RF1 is gradient-modulated, Gx is a scaling factor to
        achieve the desired slice thickness in the x-direction.
    Gy : float
        Gradient strength for secondselective refocusing pulse in G/cm OR, in 
        the case that RF1 is gradient-modulated, Gy is a scaling factor to
        achieve the desired slice thickness in the y-direction.
    flipAngle : float, optional
        Flip angle of the refocusing pulse in degrees. The default is 180.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Raises
    ------
    ValueError
        Delay values cannot be negative after subtracting pulse durations.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the PRESS sequence.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    delays=[(tau1-tp)/1000,(tau2-tp)/1000]
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following taus are too short: ' + str([didx for didx,dval in enumerate(delays) if dval<0]) +'.')
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_COF(H1,d2,-1) # Keep only -1 coherences
    d2=sim_evolve(d2,H1,delays[0]/2) #Evolve by delays[0]/2
    d2=sim_shapedRF(d2,H1,RF1,tp,flipAngle,90,dx,Gx) #First shaped 180 refocusing pulse
    d2=sim_COF(H1,d2,1) # Keep only +1 coherences
    d2=sim_evolve(d2,H1,sum(delays)/2) #Evolve by (delays[0]+delays[1])/2
    d2=sim_shapedRF(d2,H1,RF1,tp,flipAngle,90,dy,Gy)
    d2=sim_COF(H1,d2,-1) # Keep only -1 coherences
    d2=sim_evolve(d2,H1,delays[1]/2) #Evolve by delays[1]/2
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='press'
    out1.sim='shaped'
    out1.te=tau1+tau2
    return out1

def sim_press_shaped_phCyc(npts,sw,Bfield,linewidth,spinSys,tau1,tau2,RF1,tp,dx,dy,Gx,Gy,phCyc1,phCyc2,flipAngle=180,centerFreq=4.65,centerFreq_label=None):
    """
    This function is deprecated. Use sim_press_shaped, which nulls undesirable 
    coherences rather than this function, which requires simulating spectra for 
    each phase cycle step and then combining.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    delays=[tau1-tp,tau2-tp]
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following taus are too short: ' + str([dval<0 for dval in delays]) +'.')
    delays=[dval/1000 for dval in delays]
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,delays[0]/2) #Evolve by delays[0]/2
    d2=sim_shapedRF(d2,H1,RF1,tp,flipAngle,90+phCyc1,dx,Gx) #First shaped 180 refocusing pulse
    d2=sim_evolve(d2,H1,sum(delays)/2) #Evolve by (delays[0]+delays[1])/2
    d2=sim_shapedRF(d2,H1,RF1,tp,flipAngle,90+phCyc2,dy,Gy)
    d2=sim_evolve(d2,H1,delays[1]/2) #Evolve by delays[1]/2
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='press'
    out1.sim='shaped'
    out1.te=tau1+tau2
    return out1

def sim_semiLASER_shaped(npts,sw,Bfield,linewidth,spinSys,te,RF1,tp,dx,dy,Gx,Gy,flipAngle=180,centerFreq=2.3,centerFreq_label=None):
    """
    Simulates the semi-LASER experiment as described by Oz et al. (2018). The
    excitation pulse is simulated as an instantaneous rotation and the two 
    pairs of slice-selective refocusing adiabatic pulses are simulated as
    shaped RF pulses. Unwanted coherences are removed using coherence order 
    filtering (rather than having to simulate phase cycling). Spectrum is 
    simulated at a point (dx,dy) given the values of the slice-selective
    gradients Gx and Gy. The code accepts gradient-modulated pulses and uses
    Gx and Gy as scaling factors to achieve the desired slice thickness in 
    that case. To fully simulate the sLASER experiment, the simulation must be
    run at various dx and dy, then summed and scaled together for a final
    spectrum representing a voxel.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    te : float
        Echo time in ms of the sLASER experiment.
    RF1 : pyFidA.RF_pulse object
        RF pulse for refocusing (obtained using pyFidA.io_loadRFwaveform).
    tp : float
        RF pulse duration in ms.
    dx : float
        Position offset in x-direction in cm (corresponding to first refocusing 
        pulse).
    dy : float
        Position offset in y-direction in cm (corresponding to second 
        refocusing pulse).
    Gx : float
        Gradient strength for first selective refocusing pulse in G/cm OR, in 
        the case that RF1 is gradient-modulated, Gx is a scaling factor to
        achieve the desired slice thickness in the x-direction.
    Gy : float
        Gradient strength for secondselective refocusing pulse in G/cm OR, in 
        the case that RF1 is gradient-modulated, Gy is a scaling factor to
        achieve the desired slice thickness in the y-direction.
    flipAngle : float, optional
        Flip angle of the refocusing pulse in degrees. The default is 180.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 2.3 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Raises
    ------
    ValueError
        Delay values cannot be negative after subtracting pulse durations.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the sLASER sequence.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    if RF1.isGM:
        RFX=rf_scaleGrad(RF1,Gx)
        RFY=rf_scaleGrad(RF1,Gy)
        Gx=None
        Gy=None
    else:
        RFX=RF1
        RFY=RF1
    if te/4<tp:
        raise ValueError('ERROR! The duration of the refocusing pulse cannot be longer than a quarter of the echo time! ABORTING!!')
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    for loopct,(cor_keep,delay_time,rf_pul,position,grad) in enumerate(zip([-1,1,-1,1,-1],[(te/4-tp)/2,te/4-tp,te/4-tp,te/4-tp,(te/4-tp)/2],[RFX,RFX,RFY,RFY,0],[dx,dx,dy,dy,0],[Gx,Gx,Gy,Gy,0])):
        d2=sim_COF(H1,d2,cor_keep) # Keep only certain coherences
        d2=sim_evolve(d2,H1,delay_time/1000) # Evolve during delay
        if loopct<4:
            d2=sim_shapedRF(d2,H1,rf_pul,tp,flipAngle,0,position,grad) #Next shaped 180 degree adiabatic refocusing pulse
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='semi-LASER'
    out1.sim='shaped'
    out1.te=te
    return out1

def sim_semiLASER_shaped_phCyc(npts,sw,Bfield,linewidth,spinSys,te,RF1,tp,dx,dy,Gx,Gy,ph1,ph2,ph3,ph4,flipAngle=180,centerFreq=2.3,centerFreq_label=None):
    """
    This function is deprecated. Use sim_semiLASER_shaped, which nulls 
    undesirable coherences rather than this function, which requires simulating 
    spectra for each phase cycle step and then combining.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    if RF1.isGM:
        RFX=rf_scaleGrad(RF1,Gx)
        RFY=rf_scaleGrad(RF1,Gy)
        Gx=None
        Gy=None
    else:
        RFX=RF1
        RFY=RF1
    if te/4<tp:
        raise ValueError('ERROR! The duration of the refocusing pulse cannot be longer than a quarter of the echo time! ABORTING!!')
    tau1=(te/4-tp)/2
    tau2=te/4-tp
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,tau1/1000) #Evolve by tau1
    d2=sim_shapedRF(d2,H1,RFX,tp,flipAngle,ph1,dx,Gx) #1st shaped 180 degree adiabatic refocusing pulse along X gradient
    d2=sim_evolve(d2,H1,tau2/1000) #Evolve by tau2
    d2=sim_shapedRF(d2,H1,RFX,tp,flipAngle,ph2,dx,Gx) #2nd shaped 180 degree adiabatic refocusing pulse along X gradient
    d2=sim_evolve(d2,H1,tau2/1000) #Evolve by tau2
    d2=sim_shapedRF(d2,H1,RFY,tp,flipAngle,ph3,dy,Gy) #3rd shaped 180 degree adiabatic refocusing pulse along Y gradient
    d2=sim_evolve(d2,H1,tau2/1000) #Evolve by tau2
    d2=sim_shapedRF(d2,H1,RFY,tp,flipAngle,ph4,dy,Gy) #4th shaped 180 degree adiabatic refocusing pulse along Y gradient
    d2=sim_evolve(d2,H1,tau1/1000) #Evolve by tau1
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='semi-LASER'
    out1.sim='shaped'
    out1.te=te
    return out1

def sim_spinecho(npts,sw,Bfield,linewidth,spinSys,tau,centerFreq=4.65,centerFreq_label=None):
    """
    Simulates a spin-echo experiment with instantaneous RF pulses.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    tau : float
        Echo time in ms.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the spin-echo sequence.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,tau/2000) #Evolve by tau/2
    d2=sim_rotate(d2,H1,180,'y') #180 refocusing pulse about y axis.
    d2=sim_evolve(d2,H1,tau/2000) #Evolve by tau/2
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='spinecho'
    out1.sim='ideal'
    out1.te=tau
    return out1

def sim_spinecho_shaped(npts,sw,Bfield,linewidth,spinSys,TE,RF1,Tp,pos,grad,ph,centerFreq=4.65,centerFreq_label=None):
    """
    Simulates a localized spin-echo experiment with a shaped RF refocusing
    pulse. Allows phase cycling of the refocusing pulses, which can be combined
    to remove unwanted coherences. For the refocusing pulse, a two-step phase
    cycling scheme is typically sufficient, where the refocusing pulse is
    phase-cycled by 0 and 90 degrees, the phases are combined by subtraction.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    TE : float
        Echo time in ms.
    RF1 : pyFidA.RF_pulse object
        RF pulse for refocusing (obtained using pyFidA.io_loadRFwaveform).
    Tp : float
        Duration of the refocusing pulse in ms.
    pos : float
        Position offset in cm in the direction corresponding to the refocusing 
        pulse.
    grad : float
        Gradient strength for the selective refocusing pulse in G/cm.
    ph : float
        Phase of the refocusing pulse in degrees.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Raises
    ------
    ValueError
        Refocusing pulse duration cannot be longer than echo time.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the spin-echo sequence.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    delay=TE-Tp
    if delay<0:
        raise ValueError('ERROR!  TE is too short.')
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,delay/2000) #Evolve by delay/2
    d2=sim_shapedRF(d2,H1,RF1,Tp,180,90+ph,pos,grad) #shaped 180 degree refocusing pulse about y axis.
    d2=sim_evolve(d2,H1,delay/2000) #Evolve by delay/2
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='spinecho'
    out1.sim='shaped'
    out1.te=TE
    return out1

def sim_spinecho_xN(npts,sw,Bfield,linewidth,spinSys,tau,Nechoes=10,centerFreq=4.65,centerFreq_label=None):
    """
    Simulates a multi-echo spin echo experiment with Nechoes as the number of
    instantaneous RF pulses.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    tau : float
        Echo time in ms.
    Nechoes : int, optional
        Number of spin echoes. The default is 10.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the multi-echo spin echo sequence.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    delay=tau/(2*Nechoes)
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    for echoct in range(Nechoes):
        d2=sim_evolve(d2,H1,delay/1000) #Evolve by delay
        d2=sim_rotate(d2,H1,180,'y') #180 refocusing pulse about y axis.
        d2=sim_evolve(d2,H1,delay/1000) #Evolve by delay
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='spinecho_x'
    out1.sim='ideal'
    out1.te=tau
    return out1

def sim_steam(npts,sw,Bfield,linewidth,spinSys,te,tm,centerFreq=4.65,centerFreq_label=None):
    """
    Simulates the STEAM sequence using ideal (instantaneous) RF pulses. To 
    remove unwanted coherences, coherence order filtering is employed.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    te : float
        Echo time in ms.
    tm : float
        Mixing time in ms.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the STEAM sequence.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_COF(H1,d2,1) #Select coherence order 1
    d2=sim_evolve(d2,H1,te/2000) #Evolve by te/2
    d2=sim_rotate(d2,H1,-90,'x') #Second 90 degree pulse around x axis
    d2=sim_COF(H1,d2,0) # Select coherence order 0
    d2=sim_evolve(d2,H1,tm/1000) #Evolve by tm delay
    d2=sim_rotate(d2,H1,90,'x') #Third 90 degree pulse around x axis
    d2=sim_COF(H1,d2,-1) # Select coherence order -1
    d2=sim_evolve(d2,H1,te/2000) #Evolve by te/2
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='steam'
    out1.sim='ideal'
    out1.te=te
    out1.tm=tm
    return out1

def sim_steam_shaped(npts,sw,Bfield,linewidth,spinSys,te,tm,RFpulse,tp,dx,dy,Gx,Gy,flipAngle=90,centerFreq=4.65,centerFreq_label=None):
    """
    Simulates the STEAM experiment. The initial excitation is simulated as an
    instantaneous rotation, and the subsequent 90 degree pulses are simulated
    as shape rotations. It employs coherence selection to null undesired 
    coherences, rather than needing to simulate phase cycling. Spectrum is 
    simulated at a point (dx,dy) given the values of the slice-selective
    gradients Gx and Gy. To fully simulate the STEAM experiment, the simulation 
    must be run at various dx and dy, then summed and scaled together for a 
    final spectrum representing a voxel.

    Parameters
    ----------
    npts : int
        Number of points in the fid/spectrum.
    sw : float
        Spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Full width at half maximum of the simulated peaks, in Hz.
    spinSys : list of dicts
        Spin system containing the name, chemical shifts, J-couplings and 
        scaling factor for each part of the spin system. Non-interacting parts 
        of the spin system should be split into separate dicts (the list 
        elements), in order to speed up calculations. Fully interacting spin
        systems will still be a list, but with only one dict element.
    te : float
        Echo time in ms.
    tm : float
        Mixing time in ms.
    RFpulse : pyFidA.RF_pulse object
        RF pulse for refocusing pulses (obtained using pyFidA.io_loadRFwaveform).
    tp : float
        RF pulse duration in ms.
    dx : float
        Position offset in cm in the x-direction (corresponding to the first
        refocusing pulse).
    dy : float
        Position offset in cm in the y-direction (corresponding to the second
        refocusing pulse).
    Gx : float
        Gradient strength for the first selective refocusing pulse in G/cm.
    Gt : float
        Gradient strength for the second selective refocusing pulse in G/cm.
    flipAngle : float, optional
        Flip angle of refocusing pulses in degrees The default is 90.
    centerFreq : float, optional
        Center frequency (in ppm) of the spectrum that defines the chemical 
        shift values in spinSys. The default is 4.65 ppm.
    centerFreq_label : float, optional
        Center frequency of the simulated spectrum in ppm. The values for the 
        center frequencies of the spin system and simulated spectrum are 
        allowed to be different in case the chemical shifts used in spinSys
        are relative to a different 0 ppm reference than the simulated 
        spectrum. This is unlikely but possible. The default is None, which 
        will use the value of centerFreq.

    Raises
    ------
    ValueError
        RF pulse duration must be shorter than echo time and mixing time.

    Returns
    -------
    out1 : pyFidA.FID object
        Spectrum simulated using the STEAM sequence.

    """
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    # It is common for STEAM sequences to use an asymmetric RF pulse and time-
    # reverse it. Second pulse should be max-phase and third should be 
    # min-phase (so long tails are during TM period and TE can be minimized).
    if RFpulse.rfCentre>0.5:
        RF1=rf_timeReverse(RFpulse)
        RF2=RFpulse
    else:
        RF1=RFpulse
        RF2=rf_timeReverse(RFpulse)
    if te<(2*RF1.rfCentre*tp):
        raise ValueError('ERROR! TE cannot be less than the duration of the RF pulse. ABORTING')
    if tm<(2*RF2.rfCentre*tp):
        raise ValueError('ERROR! Mixing time cannot be less than the duration of the RF pulse. ABORTING')
    delays=[te-(RF1.rfCentre*tp*2),tm-(RF2.rfCentre*tp*2)]
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following timings are too short: ' + str([didx for didx,dval in enumerate(delays) if dval<0]) +'.')
    delays=[dval/1000 for dval in delays]
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_COF(H1,d2,1) # Keep only +1 coherences
    d2=sim_evolve(d2,H1,delays[0]/2) #Evolve by delays[0]/2
    d2=sim_gradSpoil(d2,H1,[Gx,0,0],[dx,dy,0],tp*RF1.rfCentre) # Pre-wind for 2nd 90. Matlab notes that this only seems to work with Gx positive, which doesn't make sense
    d2=sim_shapedRF(d2,H1,RF1,tp,flipAngle,90,dx,Gx) #First shaped pulse (2nd 90)
    d2=sim_COF(H1,d2,0) # Keep only 0-order coherences
    d2=sim_evolve(d2,H1,delays[1]/2) #Evolve by delays[1]/2
    d2=sim_shapedRF(d2,H1,RF2,tp,flipAngle,90,dy,Gy) #2nd shaped RF (3rd 90)
    d2=sim_gradSpoil(d2,H1,[0,Gy,0],[dx,dy,0],tp*RF2.rfCentre) # Rewind for 3rd 90 pulse. Matlab says RF1.rfCentre but surely it should be RF2?
    d2=sim_COF(H1,d2,-1) # Keep only -1 coherences
    d2=sim_evolve(d2,H1,delays[1]/2) #Evolve by delays[1]/2
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='steam'
    out1.sim='shaped'
    out1.te=te
    return out1