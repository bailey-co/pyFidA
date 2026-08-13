#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:40:34 2026
sim_sequences

@author: nearlabmacbook1
"""
import numpy as np
from .sim_operators import sim_COF, sim_Hamiltonian, sim_excite, sim_excite_arbPh, \
    sim_evolve, sim_gradSpoil, sim_rotate, sim_readout, sim_shapedRF
from pyFidA.fidA_rf import rf_scaleGrad, rf_timeReverse

def sim_laser(npts,sw,Bfield,linewidth,spinSys,TE,centerFreq=4.65,centerFreq_label=None):
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    # Assume equal delays between all echoes (divide by 6), and convert from ms to s.
    tau=TE/6/1000
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,tau/2) #Evolve by tau/2
    d2=sim_rotate(d2,H1,180,'y') #First 180 refocusing pulse about y axis.
    d2=sim_evolve(d2,H1,tau) #Evolve by tau
    d2=sim_rotate(d2,H1,180,'y') #2nd 180 refocusing pulse about y axis.
    d2=sim_evolve(d2,H1,tau) #Evolve by tau
    d2=sim_rotate(d2,H1,180,'y') #3rd 180 refocusing pulse about y axis.
    d2=sim_evolve(d2,H1,tau) #Evolve by tau
    d2=sim_rotate(d2,H1,180,'y') #4th 180 refocusing pulse about y axis.
    d2=sim_evolve(d2,H1,tau) #Evolve by tau
    d2=sim_rotate(d2,H1,180,'y') #5th 180 refocusing pulse about y axis.
    d2=sim_evolve(d2,H1,tau) #Evolve by tau
    d2=sim_rotate(d2,H1,180,'y') #Sixth 180 refocusing pulse about y axis.
    d2=sim_evolve(d2,H1,tau/2) #Evolve by tau/2
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='laser'
    out1.sim='ideal'
    out1.te=TE #Note te in ms here. Not sure if that works with lcm write functions
    return out1

def sim_megapress(npts,sw,Bfield,linewidth,spinSys,taus,refoc1Flip,refoc2Flip,editFlip,centerFreq=3,centerFreq_label=None):
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    taus=[tval/1000 for tval in taus]
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,taus[0]) #Evolve by taus[0]
    d2=sim_rotate(d2,H1,refoc1Flip,'y') #First refocusing pulse about y axis.
    d2=sim_evolve(d2,H1,taus[1]) #Evolve by taus[1]
    d2=sim_rotate(d2,H1,editFlip,'y') #First editing pulse about y axis
    d2=sim_evolve(d2,H1,taus[2]) #Evolve by taus[2]
    d2=sim_rotate(d2,H1,refoc2Flip,'y') #Second refocusing pulse about y axis.
    d2=sim_evolve(d2,H1,taus[3]) #Evolve by taus[3]
    d2=sim_rotate(d2,H1,editFlip,'y') #Second editing pulse about y axis
    d2=sim_evolve(d2,H1,taus[4]) #Evolve by taus[4]
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='megapress'
    out1.sim='ideal'
    out1.te=sum(taus)
    return out1

def sim_megapress_shaped(npts,sw,Bfield,linewidth,taus,spinSys,editPulse,editTp,editPh1,editPh2,refPulse,refTp,dx,dy,Gx,Gy,refPh1,refPh2,centerFreq=3,centerFreq_label=None):
    # Note that I have changed the order of arguments from Matlab in order to be
    # consistent with the order for sim_shapedRF and especially to be consistent
    # with other pulse sequences, where dx, dy, etc are listed before Gx,Gy. 
    # There are a few sequences (megapress, megaspecial and spinecho) where the
    # gradients were listed first, but then others (press, steam, semiLASER) 
    # where the position is first. It felt more important to be consistent 
    # across functions within Python than to stay consistent with Matlab. I 
    # opted for the argument order in sim_shapedRF, which puts position before
    # gradient.
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    # Default is to set 3 ppm GABA resonance to the center??
    
    # Calculate new delays by subtracting the pulse durations from the taus vector
    delays=[0]*len(taus)
    delays[0]=taus[0]-refTp/2
    delays[1]=taus[1]-(refTp+editTp)/2
    delays[2]=taus[2]-(editTp+refTp)/2
    delays[3]=taus[3]-(refTp+editTp)/2
    delays[4]=taus[4]-editTp/2
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following taus are too short: ' + str([dval<0 for dval in delays]) +'.')
    delays=[dval/1000 for dval in delays]
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

def sim_megapress_shapedEdit(npts,sw,Bfield,linewidth,taus,spinSys,editPulse,editTp,editPh1,editPh2,centerFreq=3,centerFreq_label=None):
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    # Default is to set 3 ppm GABA resonance to the center??
    
    # Calculate new delays by subtracting the pulse durations from the taus vector
    delays=[0]*len(taus)
    delays[0]=taus[0]
    delays[1]=taus[1]-editTp/2
    delays[2]=taus[2]-editTp/2
    delays[3]=taus[3]-editTp/2
    delays[4]=taus[4]-editTp/2
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following taus are too short: ' + str([dval<0 for dval in delays]) +'.')
    delays=[dval/1000 for dval in delays]
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

def sim_megapress_shapedRefoc(npts,sw,Bfield,linewidth,taus,spinSys,editFlip,refPulse,refTp,dx,dy,Gx,Gy,refPh1,refPh2,centerFreq=3,centerFreq_label=None):
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    # Default is to set 3 ppm GABA resonance to the center??
    
    # Calculate new delays by subtracting the pulse durations from the taus vector
    delays=[0]*len(taus)
    delays[0]=taus[0]-refTp/2
    delays[1]=taus[1]-refTp/2
    delays[2]=taus[2]-refTp/2
    delays[3]=taus[3]-refTp/2
    delays[4]=taus[4]
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following taus are too short: ' + str([dval<0 for dval in delays]) +'.')
    delays=[dval/1000 for dval in delays]
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

def sim_megaspecial_shaped(npts,sw,Bfield,linewidth,taus,spinSys,editPulse,editTp,editPh1,editPh2,refPulse,refTp,dx,Gx,refPh,centerFreq=3,centerFreq_label=None):
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    # Default is to set 3 ppm GABA resonance to the center??
    
    # Calculate new delays by subtracting the pulse durations from the taus vector
    delays=[0]*len(taus)
    delays[0]=taus[0]-editTp/2
    delays[1]=taus[1]-(editTp+refTp)/2
    delays[2]=taus[2]-(refTp+editTp)/2
    delays[3]=taus[3]-editTp/2
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following taus are too short: ' + str([dval<0 for dval in delays]) +'.')
    delays=[dval/1000 for dval in delays]
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
    # Note that I deal with the center frequency slightly differently here 
    # compared to Matlab. Rather than change spinSys.shifts, the shifts are 
    # altered in sim_Hamiltonian and those new shifts get incorporated into the
    # Hamiltonian object created. It is also used in sim_readout in order to 
    # define the ppm range for the FID object.
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
    out1=sim_onepulse(npts,sw,Bfield,linewidth,spinSys,anglein=anglein,ph1=ph1,centerFreq=centerFreq,centerFreq_label=centerFreq_label)
    return out1

def sim_onepulse_delay(npts,sw,Bfield,linewidth,spinSys,delay,centerFreq=4.65,centerFreq_label=None):
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    d2=sim_excite(d1,H1,whichax='x',anglein=90)
    d2=sim_evolve(d2,H1,delay/1000)
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label)
    ### END PULSE SEQUENCE
    out1.sequence='onepulse'
    out1.sim='ideal'
    out1.te=delay
    return out1

def sim_cosy(npts,sw,Bfield,linewidth,spinSys,npts2,centerFreq=4.65,centerFreq_label=None):
    df=sw/npts2
    # delay_vec in ms, divide by 1000 to get in microseconds before sending to sim_evolve
    #delay_vec=np.r_[0:npts2/df:1/df]
    delay_vec=(np.r_[3:200*16+3:200]+200*8)/1e6
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

def sim_onepulse_shaped(npts,sw,Bfield,linewidth,spinSys,RF1,tp,phCyc,dfdx=0,G=None,centerFreq=4.65,centerFreq_label=None):
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    # Note that Matlab works out separate cases for slice-selective vs 
    # frequency-selective excitation but I can just pass the values for G and
    # dfdx to sim_shapedRF and it will work out what to do, so don't need
    # separate cases in this function.
    H1,d1=sim_Hamiltonian(spinSys,Bfield,center_freq_ppm=centerFreq)
    ### BEGIN PULSE SEQUENCE
    # Note that I have removed an added 90 degrees on the phase cycle here so that
    # the function is analogous to sim_onepulse_arbPh
    d2=sim_shapedRF(d1,H1,RF1,tp,90,phCyc,dfdx,G)
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label)
    ### END PULSE SEQUENCE
    out1.sequence='onepulse'
    out1.sim='shaped'
    out1.te=tp/2
    return out1

def sim_press(npts,sw,Bfield,linewidth,spinSys,tau1,tau2,centerFreq=4.65,centerFreq_label=None):
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
    out1.te=tau1+tau2 #Note te in ms here. Not sure if that works with lcm write functions
    return out1

def sim_press_shaped(npts,sw,Bfield,linewidth,spinSys,tau1,tau2,RF1,tp,dx,dy,Gx,Gy,flipAngle=180,centerFreq=4.65,centerFreq_label=None):
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    # Changed the order of the Gx, dx arguments from Matlab sim_press_shaped
    # function because the order here doesn't match the order in sim_megapress_shaped
    # and that's annoying.
    delays=[tau1-tp,tau2-tp]
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following taus are too short: ' + str([dval<0 for dval in delays]) +'.')
    delays=[dval/1000 for dval in delays]
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
    # This could almost be the same function as sim_press_shaped_phCyc with
    # phCyc1=0 and phCyc2=0 by default. Except that this one doesn't have the
    # sim_COF steps and I'm not sure why those are there. Is it just to save time?
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
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    # Changed the order of the Gx, dx arguments from Matlab sim_press_shaped
    # function because the order here doesn't match the order in sim_megapress_shaped
    # and that's weird.
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
    d2=sim_COF(H1,d2,-1) # Keep only -1 coherences
    d2=sim_evolve(d2,H1,tau1/1000) #Evolve by tau1
    d2=sim_shapedRF(d2,H1,RFX,tp,flipAngle,0,dx,Gx) #1st shaped 180 degree adiabatic refocusing pulse along X gradient
    d2=sim_COF(H1,d2,1) # Select coherence order 1
    d2=sim_evolve(d2,H1,tau2/1000) #Evolve by tau2
    d2=sim_shapedRF(d2,H1,RFX,tp,flipAngle,0,dx,Gx) #2nd shaped 180 degree adiabatic refocusing pulse along X gradient
    d2=sim_COF(H1,d2,-1) # Select coherence order -1
    d2=sim_evolve(d2,H1,tau2/1000) #Evolve by tau2
    d2=sim_shapedRF(d2,H1,RFY,tp,flipAngle,0,dy,Gy) #3rd shaped 180 degree adiabatic refocusing pulse along Y gradient
    d2=sim_COF(H1,d2,1) # Select coherence order 1
    d2=sim_evolve(d2,H1,tau2/1000) #Evolve by tau2
    d2=sim_shapedRF(d2,H1,RFY,tp,flipAngle,0,dy,Gy) #4tg shaped 180 degree adiabatic refocusing pulse along Y gradient
    d2=sim_COF(H1,d2,-1) # Select coherence order -1
    d2=sim_evolve(d2,H1,tau1/1000) #Evolve by tau1
    out1,dfinal=sim_readout(d2,H1,npts,sw=sw,linewidth=linewidth,rcvPhase=90,center_freq_ppm=centerFreq_label) #Readout along y (90 degree phase)
    ### END PULSE SEQUENCE
    out1.sequence='semi-LASER'
    out1.sim='shaped'
    out1.te=te
    return out1

def sim_semiLASER_shaped_phCyc(npts,sw,Bfield,linewidth,spinSys,te,RF1,tp,dx,dy,Gx,Gy,ph1,ph2,ph3,ph4,flipAngle=180,centerFreq=2.3,centerFreq_label=None):
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
    out1.te=tau #Note te in ms here. Not sure if that works with lcm write functions
    return out1

def sim_spinecho_shaped(npts,sw,Bfield,linewidth,spinSys,TE,RF1,Tp,pos,grad,ph,centerFreq=4.65,centerFreq_label=None):
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
    out1.te=tau #Note te in ms here. Not sure if that works with lcm write functions
    return out1

def sim_steam(npts,sw,Bfield,linewidth,spinSys,te,tm,centerFreq=4.65,centerFreq_label=None):
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
    # Changed the order of the Gx, dx arguments from Matlab sim_press_shaped
    # function because the order here doesn't match the order in sim_megapress_shaped
    # and that's weird.
    if centerFreq_label is None:
        centerFreq_label=centerFreq
    if RFpulse.rfCentre>0.5:
        RF1=rf_timeReverse(RFpulse)
        RF2=RFpulse
    else:
        RF1=RFpulse
        RF2=rf_timeReverse(RFpulse)
    if te<(2*RF1.rfCentre*tp):
        raise ValueError('ERROR! TE cannot be less than the duration of the RF pulse. ABORTING')
    if tm<(2*RF2.rfCentre*tp):
        raise ValueError('ERROR! Echo time 2 cannot be less than the duration of the RF pulse. ABORTING')
    delays=[te-(RF1.rfCentre*tp*2),tm-(RF2.rfCentre*tp*2)]
    if any([dval<0 for dval in delays]):
        raise ValueError('ERROR! The following timings are too short: ' + str([dval<0 for dval in delays]) +'.')
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