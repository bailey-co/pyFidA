#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:14:29 2022
pyFidA.fidA_sim.sim_operators.py

@author: Colleen Bailey (@cbailey@sri.utoronto.ca), based on Matlab code by Jamie Near

Operations to generate spin Hamiltonians and density matrices from spin system
information, and then simulate their changes through basic functions: rf pulse 
effects, evolution and spoiling. Also contains code for generating a spectrum
from a density matrix and Hamiltonian.

For functions that combine these operations into larger pulse sequence 
simulations, see pyFidA.fidA_sim.sim_sequences.py.

Functions:
    * check_angle_format
    * sim_COF
    * sim_coherenceOrder
    * sim_dAdd
    * sim_dMul
    * sim_dDiv
    * sim_evolve
    * sim_excite
    * sim_excite_arbPh
    * sim_gradSpoil
    * sim_Hamiltonian
    * sim_readout
    * sim_rotate
    * sim_rotate_arbPh
    * sim_shapedRF
    * sim_spoil
"""

import numpy as np
from scipy.linalg import expm
from pyFidA.fidA_common import Hamiltonian, FID
from pyFidA import io_loadRFwaveform


def check_angle_format(anglein,Hlist):
    # This code is common to sim_rotate, sim_rotate_arbPh, sim_excite and 
    # sim_excite_arbPh so I've moved it to a separate function. You need to
    # have Hlist as an input argument so that you can know the length of each
    # part of the spin system though.
    if hasattr(anglein,'__iter__'):#list or np.array
        angleout=list()
        if len(anglein)==len(Hlist):
            for listct,(eachang,hmat) in enumerate(zip(anglein,Hlist)):
                if hasattr(eachang,'__iter__'):#note that these should be np.array. Attempt to convert if list
                    if len(eachang)==len(hmat.shifts):
                        angleout.append(np.array(eachang))#np.array takes care of case where anglein is list of lists (can be list on outer part but must be np.array on innner)
                    else:
                        raise ValueError('ERROR: For anglein as list of iterables, each list element must have the same length as "shifts" in the corresponding element of Hlist.')
                else:#element of anglein is a scalar, make into np.array
                    angleout.append(eachang*np.ones(len(hmat.shifts)))
        else:
            raise ValueError('ERROR: For anglein as list, it must have the same length as the list of Hamiltonians for the spin system, Hlist.')
    else:#anglein is scalar. Create list of angle arrays that match length of shifts for elements of Hlist.
        angleout=[anglein*np.ones(len(hmat.shifts)) for hmat in Hlist]
    return angleout

def sim_COF(Hlist,d_in,order):
    d_out=list()
    for eachd,hmat in zip(d_in,Hlist):
        mask1=(hmat.coherenceOrder==order)
        d_out.append(eachd*mask1)
    return d_out

def sim_coherenceOrder(spinSys):
    if type(spinSys) is dict:
        spinSys=[spinSys]
    outlist=list()
    for eachpart in spinSys:
        p0=np.array([[0,-1],[1,0]])
        p=p0.copy()
        for spinct in range(len(spinSys.shifts)-1):
            p=np.kron(np.ones([2,2]),p)+np.kron(p0,np.ones(p.shape))
        outlist.append(p)
    return outlist

def sim_dAdd(d1,d2,factor=1):
    # I'm not running all of the checks that are in Matlab because I think that
    # this should be sufficient. Could consider making a class for the density
    # matrix that has __add__, __mul__, etc. functions but not sure that it's
    # worth it.
    if type(d1) is list and type(d2) is list:
        d_out=[d1val+d2val*factor for d1val,d2val in zip(d1,d2)]
    else:
        d_out=d1+d2*factor
    return d_out

def sim_dMul(d_in,factor):
    if type(d_in) is list:
        d_out=[dval*factor for dval in d_in]
    else:
        d_out=factor*d_in
    return d_out

def sim_dDiv(d_in,factor):
    if type(d_in) is list:
        d_out=[dval/factor for dval in d_in]
    else:
        d_out=d_in/factor
    return d_out

def sim_evolve(d_in,Hlist,t):
    d_out=list()
    for dmat,Hmat in zip(d_in,Hlist):
        # Could consider defining the __mul__ operator for Hamiltonian objects 
        # to do matrix multiplication, in order to simplify code below, but 
        # probably better to stay consistent with numpy notation
        p=expm(1j*Hmat.HAB*t)
        d_out.append(np.dot(np.dot(p.conj().T,dmat),p))
    return d_out

def sim_excite(d_in,Hlist,whichax,anglein=90):
    angle=check_angle_format(anglein, Hlist)
    
    if whichax.lower()=='x':
        whichdir=0
    elif whichax.lower()=='y':
        whichdir=1
    d_out=list()
    for eachd,hmat,eachang in zip(d_in,Hlist,angle):
        alpha=np.where(hmat.shifts>=30,0,eachang*np.pi/180)
        # alpha will broadcast to match size of Ix or Iy, then can add over axis=2
        excite=np.sum(alpha*hmat.Imats[whichdir],axis=2)
        # Note that the eigenvalue that you get from linalg.eig are ordered 
        # differently and produce a different result than for Matlab. I'm not 
        # sure if this is just the bit of imaginary component creating problems
        # but using eigh, for Hermitian, seems to more closely back the Matlab
        # work. So just need to ensure that excite is always Hermitian?? I guess 
        # that you could run a check on whether excite is Hermitian to be safe 
        # Actually, I guess that I can work this out because Imats is made up of
        # the 4 base matrices in various combinations (kronecker) and those matrices
        # are themselves all Hermitian so I would guess that the kronecker product
        # is also Hermitian then??
        # but also this might be something known.
        [D,U]=np.linalg.eigh(excite)
        d1=np.diag(np.exp(-1j*D))
        d2=np.diag(np.exp(1j*D))
        mat1=np.dot(np.dot(U,d1),U.conj().T)
        mat2=np.dot(np.dot(U,d2),U.conj().T)
        d_out.append(np.dot(np.dot(mat1,eachd),mat2))
    return d_out

def sim_excite_arbPh(d_in,Hlist,ph_ax,anglein=90):
    angle=check_angle_format(anglein, Hlist)
    
    d_out=list()
    for eachd,hmat,eachang in zip(d_in,Hlist,angle):
        alpha=np.where(hmat.shifts>=30,0,eachang*np.pi/180)
        # alpha will broadcast to match size of Ix or Iy, then can add over axis=2
        #excite=np.sum(alpha*hmat.Imats[0],axis=2)
        excite=np.dot(hmat.Imats[0],alpha)
        # In Matlab, phase*pi/180 is multiplied on the matrix for each spin, before
        # addition, so you actually multiply by nspins*phase*pi/180, which doesn't
        # seem quite right. I think that it should be pulled out of the sum so that
        # the angle is just multiplied in once. This is how you can get ph_ax=90
        # to match up with sim_excite with whichax='y'. So I've changed that here
        # in Python from what was done in Matlab. (I ran with the ph_ax factor in
        # and outside of the sum and it seemed to come out the same)
        Rz=np.sum(ph_ax*np.pi/180*hmat.Imats[2],axis=2)
        # In Matlab, these expressions are wrong. They are listed as expm(1*Rz)
        # and expm(1*excite) so they are missing the imaginary component.
        p=expm(1j*Rz)
        q=expm(1j*excite)
        mat1=np.dot(np.dot(p.conj().T,q.conj().T),p)
        mat2=np.dot(np.dot(p.conj().T,q),p)
        d_out.append(np.dot(np.dot(mat1,eachd),mat2))
    return d_out

def sim_gradSpoil(d_in,Hlist,gradvec,posvec,dur):
    # This definitely needs to be tested. I know know how gradvec gets multiplied
    # in. Need to find if there is somewhere that this function is called??
    gamma=Hlist[0]._gamma
    # Convert gradvec into T/m
    gradvec=gradvec/100
    # Convert posvec into m
    posvec=posvec/100
    # Convert dur into s
    dur=dur/1000
    # Calculate the actual gradient field at point [x,y,z] in units of Tesla
    # Oh, I see, gradvec and posvec are both 3-element vectors for x,y and z 
    # directions (the above said that but I didn't realize it wasn't a 2D matrix)
    # so B_gradTotal is scalar (and thus angle1 is scalar)
    B_gradTotal=np.dot(gradvec,posvec)
    # The phase in radians generated by the gradient pulse
    angle1=2*np.pi*gamma*B_gradTotal*dur
    d_out=list()
    for eachd,hmat in zip(d_in,Hlist):
        # Same issue with whether angle1 should be in the sum or out. I ran with 
        # the angle both inside and outside and it comes out the same so I guess
        # it doesn't matter, although I find that a bit strange because for
        # sim_excite, it seems to give differences. Maybe z is special.
        spoil=angle1*np.sum(hmat.Imats[2],axis=2)
        p=expm(1j*spoil)
        d_out.append(np.dot(np.dot(p.conj().T,eachd),p))
    return d_out

def sim_Hamiltonian(spinSys,Bfield,nucleus='1H',center_freq_ppm=4.65):
    # I've got a Hamiltonian object and this creates a list of Hamiltonian objects
    # that corresponds to the spin system (for spin systems like GPC where there
    # are independent parts and you can make the math go faster by having them
    # separated out and then just combine them at the end). It could be worth 
    # considering having a Hamiltonian list object that subclasses list and
    # gives you, like, the total number of spins, or maybe functions to combine
    # the spins later??? Actually though there don't seem to be many use cases
    # for that since the Hamiltonian is relatively simple and it's the density
    # matrix that evolves (might be more useful to have an object for a list of
    # density matrix components).
    if type(spinSys) is not list: # dict, only a single set of spins for this metabolite. Make list for iterable calls below.
        spinSys=[spinSys]
    # Note that all Hamiltonians and density matrices generated this way will be
    # lists, even if they only have one element.
    hamlist=list()
    dlist=list()
    for sysct,each_syspart in enumerate(spinSys):
        newsys=each_syspart.copy()
        newsys['shifts']=each_syspart['shifts']-center_freq_ppm
        hamlist.append(Hamiltonian(newsys,Bfield,nucleus))
        dlist.append(hamlist[sysct].initd.copy())
    return hamlist,dlist

def sim_readout(d_in,Hlist,npts,sw,linewidth,rcvPhase=0,shape='L',center_freq_ppm=4.65,Rval=0.5):
    d_out=list()
    out_parts=list()
    deltat=1/sw
    if shape.lower()=='l' or shape.lower()=='lorentz':
        # For linewidth in Hz, we need to convert to multiply by 2*pi to get
        # in rad/s before combining with time. But also the t2 uses HWHM or 
        # linewidth/2, so 2*pi/2 gives the pi factor that you see in t2.
        t2=1/(np.pi*linewidth)
        decay=np.exp(-np.r_[:npts]*deltat/t2)
    elif shape.lower()=='g' or shape.lower()=='gauss':
        # I think Matlab might have calculated 1/sigma and not sigma???
        sigma=linewidth/np.sqrt(2*np.log(2))*np.pi
        decay=np.exp(-1*(np.r_[:npts]*deltat)**2/(2*sigma**2))
    elif shape.lower()=='lg':
        t2=1/(np.pi*linewidth)
        sigma=linewidth/np.sqrt(2*np.log(2))*np.pi
        decay=Rval*np.exp(-np.r_[:npts]*deltat/t2)+(1-Rval)*np.exp(-1*(np.r_[:npts]*deltat)**2/(2*sigma**2))
    else:
        raise ValueError('ERROR: Shape not recognized! Must be "L", "G", or "LG".')
    for sysct,(eachd,hmat) in enumerate(zip(d_in,Hlist)):
        # Note sure if I should use eigh for Hermitian again
        [D,U]=np.linalg.eig(hmat.HAB)
        val=2**(2-hmat.nspins)
        Fxy=hmat.F[0]+1j*hmat.F[1]
        phase_comp=np.exp(1j*rcvPhase*np.pi/180)
        fidtmp=np.zeros([npts],dtype=complex)
        for kct in range(npts):
            d1=np.diag(np.exp(-1j*kct*D*deltat))
            d=np.dot(np.dot(U,d1),U.conj().T)
            dmat=np.dot(np.dot(d,eachd),d.conj().T)
            fidtmp[kct]=np.trace(np.dot(dmat,Fxy)*phase_comp)
        out_parts.append(val*fidtmp*decay)
    d_out=sim_evolve(d_in,Hlist,npts*deltat)
    # Combine the output FIDs from the different parts of the spin system
    # There is an apostrophe in the Matlab code that I missed for the longest 
    # time (this is conjugate transpose in Matlab. Since this is a 1D array,
    # that's just the conjugate). That's why I had spectra that were reversed
    # in the left-right direction.
    outfids=sum(out_parts,start=0*out_parts[0]).conj()
    
    # Can have te and tr as input arguments??
    # I can have saved the nucleus in the Hamiltonian for use here??? Here it needs to be a list
    out1=FID(outfids,sw,1*Hlist[0]._Bfield*Hlist[0]._gamma,te=0,tr=0,sequence='simulated',nucleus=[hmat._nucleus for hmat in Hlist],dims=['t'],center_freq_ppm=center_freq_ppm)
    return out1,d_out
        
def sim_rotate(d_in,Hlist,anglein=90,whichax='x'):
    # This seems verysimilar to excite, but it doesn't diagonalize, which I am
    # guessing means that this takes slightly longer. Also, this technically
    # allows z rotation, which you wouldn't typically get with an rf pulse
    
    angle=check_angle_format(anglein, Hlist)
    if whichax.lower()=='x':
        whichdir=0
    elif whichax.lower()=='y':
        whichdir=1
    elif whichax.lower()=='z':
        whichdir=2
    
    d_out=list()
    for eachd,hmat,eachang in zip(d_in,Hlist,angle):
        alpha=np.where(hmat.shifts>=30,0,eachang*np.pi/180)
        # alpha will broadcast to match size of Ix or Iy, then can add over axis=2
        rotMat=np.sum(alpha*hmat.Imats[whichdir],axis=2)
        # In Matlab, phase*pi/180 is multiplied on the matrix for each spin, before
        # addition, so you actually multiply by nspins*phase*pi/180, which doesn't
        # seem quite right. I think that it should be pulled out of the sum so that
        # the angle it just multiplied in once. This is how you can get ph_ax=90
        # to match up with sim_excite with whichax='y'. So I've changed that here
        # in Python from what was done in Matlab
        p=expm(1j*rotMat)
        d_out.append(np.dot(np.dot(p.conj().T,eachd),p))
    return d_out

def sim_rotate_arbPh(d_in,Hlist,anglein=90,ph_ax=0):
    # This code is basically the same as sim_excite_arbPh but the expm matrix
    # exponentials have the 1j factor in so there is presumably a phase difference?
    angle=check_angle_format(anglein, Hlist)
    
    d_out=list()
    for eachd,hmat,eachang in zip(d_in,Hlist,angle):
        alpha=np.where(hmat.shifts>=30,0,eachang*np.pi/180)
        # alpha will broadcast to match size of Ix or Iy, then can add over axis=2
        rotMat=np.sum(alpha*hmat.Imats[0],axis=2)
        # In Matlab, phase*pi/180 is multiplied on the matrix for each spin, before
        # addition, so you actually multiply by nspins*phase*pi/180, which doesn't
        # seem quite right. I think that it should be pulled out of the sum so that
        # the angle it just multiplied in once. This is how you can get ph_ax=90
        # to match up with sim_excite with whichax='y'. So I've changed that here
        # in Python from what was done in Matlab.
        Rz=ph_ax*np.pi/180*np.sum(hmat.Imats[2],axis=2)
        p=expm(1j*Rz)
        q=expm(1j*rotMat)
        mat1=np.dot(np.dot(p.conj().T,q.conj().T),p)
        mat2=np.dot(np.dot(p.conj().T,q),p)
        d_out.append(np.dot(np.dot(mat1,eachd),mat2))
    return d_out

def sim_shapedRF(d_in,Hlist,RFpulse,Tp,flipAngle,ph1=0,dfdx=0,grad=None,**pulse_kwargs):
    # If RFpulse is a string, need to load
    # I haven't included any extra arguments like Tp or nucleus here. Maybe I
    # should. They would have to be arguments for sim_shapedRF and then be
    # passed to these functions, but I think that gets confusing because then 
    # sim_shapedRF has a bunch of arguments that aren't used in the "normal"
    # case of passing RFpulse. For example, you pass an RFpulse that is designed
    # for 1H and pass a nucleus of 13C and that nucleus argument is never used.
    # The nucleus/gamma has very limited use anyway - it's just used to calculate
    # the slice thickness from tbw for gradient-modulated pulses. Hardly seems
    # worth the confusion. One thing that you could do is have **pulse_kwargs as
    # the final argument in sim_shapedRF and then pass those on to RFpulse and
    # that's probably clearer, but then I think that you want to pass Tp on 
    # explicitly as well.
    # OH. Wait. That's not exactly true. It is passed to the BlochSimulator and 
    # it is used to calculate gradlist for gradient-modulated case. But then 
    # gradlist is used to calculate phi for the z-rotation during evolution. The
    # question then is whether the resulting tthk in cm*s could be obtained from
    # a simple scaling by gamma or 1/gamma, or whether you need the gradient
    # to be in the correct units. I think that you must need gradient in the 
    # correct units because it's combined with f in Hz (ie. gyromagnetic ratio
    # is already incorporated) but I haven't tested.
    # NOTE: Because there are a number of matrix multiplications, it is desirable
    # to avoid using the expanded waveform and use a more abbreviated form (if
    # it exists). Hence why dt is calculated from RFpulse.waveform[:,2] rather
    # than calling the expanded waveform
    if type(RFpulse) is str:
        if flipAngle<=110:
            RFpulse=io_loadRFwaveform(RFpulse,'exc',Tp=Tp,**pulse_kwargs)
        else:
            RFpulse=io_loadRFwaveform(RFpulse,'inv',Tp=Tp,**pulse_kwargs)
        # There is an issue here in Matlab. When RFpulse is a string, it decides
        # Whether to make an exc or inv pulse based on the flipAngle. But then 
        # it runs the w1max calculation with Tp*90, ie. assuming an exc type.
        # If RFpulse is not a string but an RF struct, it runs with Tp*180, so
        # assuming an inv pulse, even though no check is run on RF_struct.type
    #RFpulse should be the pulse already. I don't know why the thing beside
    #Tp is different for loading from a file??
    if RFpulse.pulse_type=='exc':
        w1max=(RFpulse.tw1*flipAngle)/(Tp*90)
    elif RFpulse.pulse_type=='inv' or RFpulse.pulse_type=='ref':
        w1max=(RFpulse.tw1*flipAngle)/(Tp*180)
    else:
        # One option here is to run the calibration for w1 for a 180 and then use that
        raise ValueError('ERROR: Unknown pulse type. Only types exc, inv and ref types are implemented.')
    # Now I need to work out whether the pulse is frequency selective or spatially selective
    if not RFpulse.isGM:
        if grad is None or grad==0:
            grad=0
            simType='f'
        else:
            simType='g'
            grad=grad*0.01 #convert from G/cm to T/m
    else:#pulse is gradient-modulated
        if grad is not None and grad!=0:
            simType='g'
            grad=RFpulse.waveform[:,3]*0.01
        else:
            raise ValueError('ERROR! You cannot supply a gradient-modulated RFpulse AND specify the gradient strength.')
    dfdx=dfdx/100 #convert from cm to m
    # Define the properties of the refocusing rf pulse. (Note that some of this
    # could maybe be moved to the class??)
    Tp=Tp/1000
    rfph=RFpulse.waveform[:,0]*np.pi/180
    dt=Tp*RFpulse.waveform[:,2]/np.sum(RFpulse.waveform[:,2])
    d_out=list()
    for eachd,hmat in zip(d_in,Hlist):
        zeta=rfph+ph1*np.pi/180
        B1max=w1max*1000/hmat._gamma #convert from w1max in kHz to B1max in Tesla
        rfB1=B1max*RFpulse.waveform[:,1]/np.amax(RFpulse.waveform[:,1])
        Rz=np.zeros([2**hmat.nspins,2**hmat.nspins,len(rfB1)])
        Ry=np.zeros([2**hmat.nspins,2**hmat.nspins,len(rfB1)])
        Rx=np.zeros([2**hmat.nspins,2**hmat.nspins,len(rfB1)])
        for spinct in range(hmat.nspins):
            # These all have length RF and apply at one particular shift in the spin system
            # Matlab saves the value for each nspin but I don't think that there is any point
            if simType=='g':
                Beff_tmp=np.sqrt(rfB1**2+(grad*dfdx+hmat._Bfield*hmat.shifts[spinct]/1e6)**2)
                alpha_tmp=np.arctan2(grad*dfdx+hmat._Bfield*hmat.shifts[spinct]/1e6,rfB1)
            elif simType=='f':
                Beff_tmp=np.sqrt(rfB1**2+(dfdx/hmat._gamma+hmat._Bfield*hmat.shifts[spinct]/1e6)**2)
                alpha_tmp=np.arctan2(dfdx/hmat._gamma+hmat._Bfield*hmat.shifts[spinct]/1e6,rfB1)
            theta_tmp=2*np.pi*hmat._gamma*Beff_tmp*dt
            # Now loop through the rf pulse. With list comprehensions, Rz, Rx etc
            # are lists of 3x3 rotation matrices and the list has length RF
            # Note that this could be replaced with a single Rmats of length 3
            # to correspond to x, y, and z directions but this creates lists of 
            # lists of matrices, which, no.
            Rz=Rz+np.array([zval*hmat.Imats[2][:,:,spinct] for zval in zeta]).transpose([1,2,0])
            Ry=Ry+np.array([yval*hmat.Imats[1][:,:,spinct] for yval in alpha_tmp]).transpose([1,2,0])
            Rx=Rx+np.array([xval*hmat.Imats[0][:,:,spinct] for xval in theta_tmp]).transpose([1,2,0])
        dtmp=eachd.copy()
        for rfct,rfval in enumerate(rfB1):
            p=expm(1j*hmat.HABJonly*dt[rfct])
            q=expm(1j*Rz[:,:,rfct])
            r=expm(1j*Ry[:,:,rfct])
            s=expm(1j*Rx[:,:,rfct])
            mat1=np.dot(np.dot(np.dot(np.dot(np.dot(p.conj().T,q.conj().T),r.conj().T),s.conj().T),r),q)
            mat2=np.dot(np.dot(np.dot(np.dot(np.dot(q.conj().T,r.conj().T),s),r),q),p)
            dtmp=np.dot(np.dot(mat1,dtmp),mat2)
        d_out.append(dtmp)
    return d_out

def sim_spoil(d_in,Hlist,anglein):
    # The code here is the same as for sim_rotate but with whichax='z' and 
    # angle is assumed to be a scalar. But sim_rotate will deal with scalars.
    d_out=sim_rotate(d_in,Hlist,anglein=anglein,whichax='z')
    return d_out

