#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:14:29 2022
pyFidA.fidA_sim.sim_operators.py

@author: Colleen Bailey (@cbailey@sri.utoronto.ca), based on Matlab code by 
Jamie Near and Robin Simpson

Operations to generate spin Hamiltonians and density matrices from spin system
information, and then simulate their changes through basic functions: rf pulse 
effects, evolution and spoiling. Also contains code for generating a spectrum
from a density matrix and Hamiltonian.

For functions that combine these operations into larger pulse sequence 
simulations, see pyFidA.fidA_sim.sim_sequences.py.

Functions:
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
from pyFidA.fidA_common import Hamiltonian, FID, FidAWarning
from pyFidA import io_loadRFwaveform
import warnings

def check_angle_format(anglein,Hlist):
    """
    Checks the type and length of the variable anglein and adjusts it to match
    the format of the spin system represented by Hlist, if needed. Code is used 
    in sim_rotate, sim_rotate_arbPh, sim_excite and sim_excite_arbPh so was 
    moved to its own function.

    Parameters
    ----------
    anglein : list OR np.ndarray OR float
        Angle(s) to be reformatted. If anglein is a scalar, then it is expanded
        so that angleout has the same number for all spins in the spin system.
        If anglein is a list/array of scalars, it will be expanded to apply
        the same flip angle to each spin within a part of the spin system and
        the parts of the spin system have different values. If anglein is a 
        list/array of lists/arrays, then angleout is a copy of anglein 
        (provided that the lengths match the spin system sizes in Hlist), with 
        a unique flip angle for every spin in each part of the spin system.
    Hlist : list of pyFidA.Hamiltonian objects
        Each list element is the Hamiltonian for a separable part of the spin
        system. Only the shifts portion of each Hamiltonian is used, to expand
        anglein to match its length, if needed.

    Raises
    ------
    ValueError
        If anglein is an iterable but its length does not match the length of 
        Hlist OR if one of the list elements is itself an iterable, but its 
        length doesn't match that of the corresponding Hamiltonian shifts 
        array.

    Returns
    -------
    angleout : list of np.ndarray elements
        anglein, expanded to match the size of the spin system.

    """
    if hasattr(anglein,'__iter__'): #list or np.array
        angleout=list()
        if len(anglein)==len(Hlist):
            for listct,(eachang,hmat) in enumerate(zip(anglein,Hlist)):
                if hasattr(eachang,'__iter__'): #elements can be used as-is. Forcing conversion to np.array in the case of lists
                    if len(eachang)==len(hmat.shifts):
                        angleout.append(np.array(eachang))
                    else:
                        raise ValueError('ERROR: For anglein as list of iterables, each list element must have the same length as "shifts" in the corresponding element of Hlist.')
                else: #element of anglein is a scalar, make into np.array
                    angleout.append(eachang*np.ones(len(hmat.shifts)))
        else:
            raise ValueError('ERROR: For anglein as list, it must have the same length as the list of Hamiltonians for the spin system, Hlist.')
    else: #anglein is scalar. Create list of angle arrays that match length of shifts for elements of Hlist.
        angleout=[anglein*np.ones(len(hmat.shifts)) for hmat in Hlist]
    return angleout

def sim_COF(Hlist,d_in,order):
    """
    Nulls the signal from any undesired coherences in a spin system. Desired
    coherences are determined through extended phase graph analysis and pulse
    sequence design.

    Parameters
    ----------
    Hlist : list of pyFidA.Hamiltonian objects
        Hamiltonian operators for each part of the spin system.
    d_in : list of np.ndarray
        The input density matrices for each part of the spin system.
    order : int
        Desired coherence order that you wish to keep signal from.

    Returns
    -------
    d_out : list of np.ndarray
        Output density matrices with only desired coherences.

    """
    d_out=list()
    for eachd,hmat in zip(d_in,Hlist):
        mask1=(hmat.coherenceOrder==order)
        d_out.append(eachd*mask1)
    return d_out

def sim_coherenceOrder(spinSys):
    """
    Creates a list of coherence order matrices for each element of the density
    matrix of the spin system. These can then be used to null signal 
    incoherences during pulse sequence simulations.

    Parameters
    ----------
    spinSys : dict or list of dicts
        Spin system to generate coherence matrix for.

    Returns
    -------
    outlist : list of numpy arrays
        Coherence order matrix (or matrices, if spinSys has multiple elements)
        for the spin system.

    """
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
    """
    Add together two density matrices. Needed because density matrix outputs 
    from simulations are lists (even in single-element cases), the "+" operator
    will be interpreted as appending, not addition. The case of arrays rather
    than lists is also handled, although the + and - operators work in that 
    case.

    Parameters
    ----------
    d1 : list of numpy arrays
        First input density matrix to be added
    d2 : list of numpy arrays
        Second input density matrix to be added
    factor : float, optional
        Mainly used to indicate addition (1) or subtraction (-1) but any 
        scaling factor for the second density matrix is a valid input. The 
        default is 1.

    Returns
    -------
    d_out : list of numpy arrays
        Sum (or difference) of d1 and d2.

    """
    if type(d1) is list and type(d2) is list:
        d_out=[d1val+d2val*factor for d1val,d2val in zip(d1,d2)]
    else:
        d_out=d1+d2*factor
    return d_out

def sim_dDiv(d_in,factor):
    """
    Divide a density matrix by a scalar. Needed because density matrix 
    outputs from simulations are lists (even in single-element cases), so the 
    "/" operator will not work. The case where d_in is an array is also 
    handled, although the / operator works in that case.

    Parameters
    ----------
    d_in : list of numpy arrays
        Input density matrix to be divided
    factor : float
        Scalar factor to divide by.

    Returns
    -------
    d_out : list of numpy arrays
        Result of d_in / factor.

    """
    if type(d_in) is list:
        d_out=[dval/factor for dval in d_in]
    else:
        d_out=d_in/factor
    return d_out

def sim_dMul(d_in,factor):
    """
    Multiply a density matrix by a scalar. Needed because density matrix 
    outputs from simulations are lists (even in single-element cases), so the 
    "*" operator will be interpreted as a list expansion, not multiplication. 
    The case where d_in is an array is also handled, although the * operator
    works in that case.

    Parameters
    ----------
    d_in : list of numpy arrays
        Input density matrix to be multiplied
    factor : float
        Scalar multiplication factor.

    Returns
    -------
    d_out : list of numpy arrays
        Result of d_in * factor.

    """
    if type(d_in) is list:
        d_out=[dval*factor for dval in d_in]
    else:
        d_out=factor*d_in
    return d_out

def sim_evolve(d_in,Hlist,t):
    """
    Simulate the free evolution of the spin system under the effects of 
    chemical shift and J-coupling

    Parameters
    ----------
    d_in : list of numpy arrays
        Input density matrices.
    Hlist : list of pyFidA.Hamiltonian objects
        Hamiltonian operator(s) for the spin system(s).
    t : float
        Duration of evolution, in seconds.

    Returns
    -------
    d_out : list of numpy arrays
        Output density matrices following free evolution.

    """
    d_out=list()
    for dmat,Hmat in zip(d_in,Hlist):
        p=expm(1j*Hmat.HAB*t)
        d_out.append(np.dot(np.dot(p.conj().T,dmat),p))
    return d_out

def sim_excite(d_in,Hlist,whichax='x',anglein=90):
    """
    Simulate the effect of an ideal (instantaneous) excitation pulse on the
    density matrix.

    Parameters
    ----------
    d_in : list of numpy arrays
        Input density matrices.
    Hlist : list of pyFidA.Hamiltonian objects
        Hamiltonian operator(s) for the spin system.
    whichax : 'x' or 'y', optional
        Axis of rotation. The default is 'x'.
    anglein : float or numpy array or list, optional
        Flip angle of excitation in degrees. The angle can be the same for 
        every part of the spin system, or different for different spins, 
        depending on the type/format that anglein is entered in. See docstring
        of check_angle_format for more details. The default is 90.

    Returns
    -------
    d_out : list of numpy arrays
        Output density matrices following excitation pulse.

    """
    angle=check_angle_format(anglein, Hlist)
    
    if whichax.lower()=='x':
        whichdir=0
    elif whichax.lower()=='y':
        whichdir=1
    d_out=list()
    for eachd,hmat,eachang in zip(d_in,Hlist,angle):
        alpha=np.where(hmat.shifts>=30,0,eachang*np.pi/180)
        # np.dot(a,b) will do a sum product over the last axis of a and b when b is 1D. (Slightly faster than np.sum(b*a),axis=2)
        excite=np.dot(hmat.Imats[whichdir],alpha)
        # Note that the eigenvalue that you get from linalg.eig are ordered 
        # differently and produce a different result than for Matlab. Should
        # still work fine but np.linalg.eigh seems to match Matlab so using
        # that.
        [D,U]=np.linalg.eigh(excite)
        d1=np.diag(np.exp(-1j*D))
        d2=np.diag(np.exp(1j*D))
        mat1=np.dot(np.dot(U,d1),U.conj().T)
        mat2=np.dot(np.dot(U,d2),U.conj().T)
        d_out.append(np.dot(np.dot(mat1,eachd),mat2))
    return d_out

def sim_excite_arbPh(d_in,Hlist,ph_ax,anglein=90):
    """
    Simulates the effect of an ideal (instantaneous) excitation pulse on the
    density matrix. The phase of the excitation pulse can be arbitrarily 
    chosen. To achieve an arbitrary phase, there is a rotation about z by
    -1*phase, then a rotation about x by anglein, then a rotation back around
    z by phase.

    Parameters
    ----------
    d_in : list of numpy arrays
        Input density matrices.
    Hlist : list of pyFidA.Hamiltonian objects
        Hamiltonian operator(s) for the spin system.
    ph_ax : float
        Phase of rotation in degrees. 0='x', 90='y', etc.
    anglein : float or numpy array or list, optional
        Flip angle of excitation in degrees. The angle can be the same for 
        every part of the spin system, or different for different spins, 
        depending on the type/format that anglein is entered in. See docstring
        of check_angle_format for more details. The default is 90.

    Returns
    -------
    d_out : list of numpy arrays
        Output density matrices following excitation pulse.

    """
    angle=check_angle_format(anglein, Hlist)
    
    d_out=list()
    for eachd,hmat,eachang in zip(d_in,Hlist,angle):
        alpha=np.where(hmat.shifts>=30,0,eachang*np.pi/180)
        # np.dot(a,b) will do a sum product over the last axis of a and b when b is 1D. (Slightly faster than np.sum(b*a),axis=2)
        excite=np.dot(hmat.Imats[0],alpha)
        Rz=np.sum(ph_ax*np.pi/180*hmat.Imats[2],axis=2)
        # In Matlab, these expressions are missing the imaginary component.
        p=expm(1j*Rz)
        q=expm(1j*excite)
        # Breaking the matrix multiplications up into parts for readability
        mat1=np.dot(np.dot(p.conj().T,q.conj().T),p)
        mat2=np.dot(np.dot(p.conj().T,q),p)
        d_out.append(np.dot(np.dot(mat1,eachd),mat2))
    return d_out

def sim_gradSpoil(d_in,Hlist,gradvec,posvec,dur):
    """
    Simulate the effect of a rectangular spoiler gradient with a given 
    amplitude, duration and direction at a particular point in space.

    Parameters
    ----------
    d_in : list of numpy arrays
        Input density matrices.
    Hlist : list of pyFidA.Hamiltonian objects
        Hamiltonian operator(s) for the spin system.
    gradvec : 3-element numpy array
        Vector of the gradient spoiler amplitudes np.r_[Gx,Gy,Gz] in G/cm.
    posvec : 3-element numpy array
        Position vector of the spins of interest np.r_[x,y,z] in cm.
    dur : float
        Duration of the gradient pulse in ms.

    Returns
    -------
    d_out : list of numpy arrays
        Output density matrices following the spoiler gradient.

    """
    gamma=Hlist[0]._gamma # in Hz/T
    gradvec=gradvec/100 # Convert gradvec into T/m
    posvec=posvec/100 # Convert posvec into m
    dur=dur/1000 # Convert dur into s
    # Calculate the actual gradient field at point [x,y,z] in units of Tesla
    # Since gradvec and posvec are both 3-element vectors, the dot product
    # B_gradTotal is a scalar
    B_gradTotal=np.dot(gradvec,posvec)
    # The phase in radians generated by the gradient pulse (also scalar)
    angle1=2*np.pi*gamma*B_gradTotal*dur
    d_out=list()
    for eachd,hmat in zip(d_in,Hlist):
        spoil=angle1*np.sum(hmat.Imats[2],axis=2)
        p=expm(1j*spoil)
        d_out.append(np.dot(np.dot(p.conj().T,eachd),p))
    return d_out

def sim_Hamiltonian(spinSys,Bfield,nucleus='1H',center_freq_ppm=4.65):
    """
    Create the Hamiltonian and density matrix for spin system. In cases 
    where spin system is a list of dicts (eg. because spin system is 
    separable), a list of Hamiltonians and a list of density matrices will be 
    generated, each element representing a separable part of the spin system.
    (Where a system is separable, it is better to separate it into parts to
    reduce the size of matrices and thus the time spent on matrix 
    multiplications and matrix exponential calculations in simulations)

    Parameters
    ----------
    spinSys : dict or list
        Dictionary containing the name, shifts, J-couplings and scaleFactor of
        the spin system, or a list of such dicts representing separable parts
        of a spin system.
    Bfield : float
        Magnetic field strength in Tesla.
    nucleus : string, optional
        Nucleus used to identify the gyromagnetic ratio for calculating the 
        frequency from the Bfield. The default is '1H'.
    center_freq_ppm : float, optional
        Center frequency in ppm, used to adjust the positions of the shifts in 
        spinSys before generating the Hamiltonian. The default is 4.65 ppm.

    Returns
    -------
    Hlist : list of pyFidA.Hamiltonian objects
        Hamiltonians for each part of spinSys. Even if spinSys is a dict, 
        Hlist will be a list. All pyFidA simulation operators expect 
        Hamiltonians in a list, even if it only has a single element.
    dlist : list of numpy arrays
        The equilibrium density matrix for spinSys, with each separable part
        as its own element of the list. Even if spinSys is a dict, dlist will
        be a list since lists are expected as input for all pyFidA simulation
        functions.

    """
    if type(spinSys) is not list: # dict case converted to list
        spinSys=[spinSys]
    Hlist=list()
    dlist=list()
    for sysct,each_syspart in enumerate(spinSys):
        newsys=each_syspart.copy()
        newsys['shifts']=each_syspart['shifts']-center_freq_ppm
        Hlist.append(Hamiltonian(newsys,Bfield,nucleus))
        dlist.append(Hlist[sysct].initd.copy())
    return Hlist,dlist

def sim_readout(d_in,Hlist,npts,sw,linewidth,rcvPhase=0,shape='L',center_freq_ppm=4.65,Rval=0.5):
    """
    Simulate an ADC readout of the transverse magnetization during the free
    evolution of the spin system under the effects of chemical shift and
    J-coupling.

    Parameters
    ----------
    d_in : list of numpy arrays
        Input density matrices.
    Hlist : list of pyFidA.Hamiltonian objects
        Hamiltonian operator(s) for the spin system.
    npts : int
        Number of readout points
    sw : float
        Spectral width in Hz.
    linewidth : float
        Full width at half maximum of spectral peaks, in Hz.
    rcvPhase : float, optional
        Receiver phase, in degrees. The default is 0.
    shape : string, optional
        Shape to use for line broadening function. Options are 'L' for 
        Lorentzian, 'G' for Gaussian, and 'LG' for Lorentz-Gauss mixture. 
        The default is 'L'.
    center_freq_ppm : float, optional
        Center frequency of the spectrum in ppm. The default is 4.65 ppm.
    Rval : float between 0 and 1, optional
        Fraction of Lorentz-Gauss mixture that is made up of each peak shape.
        A value of 0 is fully Gaussian and a value of 1 is fully Lorentzian.
        This number is only used when shape='LG'. The default is 0.5.

    Returns
    -------
    out1 : pyFidA.FID object
        Simulated spectrum resulting from the readout.
    d_out : list of numpy arrays
        Output density matrix following readout.

    """
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
        # I think Matlab might have calculated 1/sigma and not sigma?
        sigma=linewidth/np.sqrt(2*np.log(2))*np.pi
        decay=np.exp(-1*(np.r_[:npts]*deltat)**2/(2*sigma**2))
    elif shape.lower()=='lg':
        t2=1/(np.pi*linewidth)
        sigma=linewidth/np.sqrt(2*np.log(2))*np.pi
        decay=Rval*np.exp(-np.r_[:npts]*deltat/t2)+(1-Rval)*np.exp(-1*(np.r_[:npts]*deltat)**2/(2*sigma**2))
    else:
        raise ValueError('ERROR: Shape not recognized! Must be "L", "G", or "LG".')
    for sysct,(eachd,hmat) in enumerate(zip(d_in,Hlist)):
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
    # Note that you need to take the complex conjugate for the data to be
    # correct in the spectral domain.
    outfids=sum(out_parts,start=0*out_parts[0]).conj()
    out1=FID(outfids,sw,1*Hlist[0]._Bfield*Hlist[0]._gamma,te=0,tr=0,sequence='simulated',nucleus=Hlist[0]._nucleus,dims=['t'],center_freq_ppm=center_freq_ppm)
    return out1,d_out
        
def sim_rotate(d_in,Hlist,anglein=90,whichax='x'):
    """
    Simulates the effect of an ideal (instantaneous) rotation on the densty 
    matrix. Similar to sim_excite but allows z rotation and doesn't 
    diagonalize. Code is repeated rather than used as a implementing one 
    function as wrapper for the other in case diagonalization is faster.

    Parameters
    ----------
    d_in : list of numpy arrays
        Input density matrices.
    Hlist : list of pyFidA.Hamiltonian objects
        Hamiltonian operator(s) for the spin system.
    anglein : float or numpy array or list, optional
        Flip angle of excitation in degrees. The angle can be the same for 
        every part of the spin system, or different for different spins, 
        depending on the type/format that anglein is entered in. See docstring
        of check_angle_format for more details. The default is 90.
    whichax : 'x' or 'y' or 'z'
        Axis of rotation. The default is 'x'.
        

    Returns
    -------
    d_out : list of numpy arrays
        Output density matrix following rf rotation.

    """
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
        rotMat=np.dot(hmat.Imats[whichdir],alpha)
        p=expm(1j*rotMat)
        d_out.append(np.dot(np.dot(p.conj().T,eachd),p))
    return d_out

def sim_rotate_arbPh(d_in,Hlist,anglein=90,ph_ax=0):
    """
    Simulates the effect of an ideal (instantaneous) rotation on the density
    matrix. The phase of the rf pulse can be arbitrarily chosen and is 
    implemented by rotating around z by the angle -ph_ax, then exciting around 
    x by anglein, then rotating back around z by angle ph_ax. Similar to 
    sim_excite_arbPh but allows z rotation and doesn't diagonalize.

    Parameters
    ----------
    d_in : list of numpy arrays
        Input density matrices.
    Hlist : list of pyFidA.Hamiltonian objects
        Hamiltonian operator(s) for the spin system.
    anglein : float or numpy array or list, optional
        Flip angle of excitation in degrees. The angle can be the same for 
        every part of the spin system, or different for different spins, 
        depending on the type/format that anglein is entered in. See docstring
        of check_angle_format for more details. The default is 90.
    ph_ax : float, optional
        Phase of rotation in degrees. 0 = 'x', 90 = 'y'. The default is 0.

    Returns
    -------
    d_out : list of numpy arrays
        Output density matrix following rf rotation.

    """
    angle=check_angle_format(anglein, Hlist)
    d_out=list()
    for eachd,hmat,eachang in zip(d_in,Hlist,angle):
        alpha=np.where(hmat.shifts>=30,0,eachang*np.pi/180)
        rotMat=np.dot(hmat.Imats[0],alpha)
        Rz=ph_ax*np.pi/180*np.sum(hmat.Imats[2],axis=2)
        p=expm(1j*Rz)
        q=expm(1j*rotMat)
        mat1=np.dot(np.dot(p.conj().T,q.conj().T),p)
        mat2=np.dot(np.dot(p.conj().T,q),p)
        d_out.append(np.dot(np.dot(mat1,eachd),mat2))
    return d_out

def sim_shapedRF(d_in,Hlist,RFpulse,Tp,flipAngle,ph1=0,dfdx=0,grad=None,**pulse_kwargs):
    """
    Simulates the effect of a shaped rf pulse on the density matrix. The 
    temporal shape of the refocusing pulses is modelled as a series of N
    rotations about the effective RF field (made up of a composite rotation
    of -alpha around Y, -zeta around Z, 2*pi*gamma*Beff*dt around X, then 
    rotate back around Z by zeta and Y by alpha), where N is the number of 
    time points in the RF waveform. The angle alpha is the angle between the
    transverse plane and Beff, while zeta is the azimuthal angle (the phase of
    the rf). The code allows gradient-modulated pulses, as long as no value is
    given for grad, a constant gradient.
    
    Parameters
    ----------
    d_in : list of numpy arrays
        Input density matrices.
    Hlist : list of pyFidA.Hamiltonian objects
        Hamiltonian operator(s) for the spin system.
    RFpulse : file OR pyFidA.RF_pulse object
        The RF pulse to use for the simulation or the filename of an RF pulse
        to be loaded by pyFidA.io_loadRFwaveform.
    Tp : float
        Pulse duration in ms.
    flipAngle : float
        flipAngle of the RF pulse in degrees.
    ph1 : float, optional
        Phase of the RF pulse in degrees. The default is 0.
    dfdx : float, optional
        If simulating a frequency-selective pulse, this argument should be the
        frequency offset in Hz. If simulating a slice-selective pulse, this
        argument should be the position offset in cm. The default is 0.
    grad : float, optional
        Gradient strength in G/cm. Only used if RFpulse.isGM=False. The 
        default is None.
    **pulse_kwargs : optional
        Additional keyword arguments to be passed to io_loadRFwaveform if 
        RFpulse is a filename (offres, nucleus, suppress_plots).

    Raises
    ------
    ValueError
        Cannot enter grad value for gradient-modulated pulse.

    Returns
    -------
    d_out : list of numpy arrays
        Output density matrix following shaped RF pulse excitation/refocusing.

    """
    if type(RFpulse) is str:
        RFpulse=io_loadRFwaveform(RFpulse,flipAngle,Tp=Tp,**pulse_kwargs)
    # Allows pulses to be used for different flip angles than they may have
    # been generated for. (Correction from Matlab, which only corrects for 
    # inv case (but runs no check that flip angle is 180)
    if RFpulse.flipAngle!=flipAngle:
        warnings.warn('WARNING: Requested flip angle does not match angle used to generate RF pulse. Results may be inaccurate, especially for adiabatic case.',FidAWarning)
    w1max=RFpulse.tw1*flipAngle/(Tp*RFpulse.flipAngle)
    # Now I need to work out whether the pulse is frequency selective or spatially selective
    if not RFpulse.isGM:
        if grad is None or grad==0:
            grad=0
            simType='f'
        else:
            simType='g'
            grad=grad*0.01 #convert from G/cm to T/m
    else: #pulse is gradient-modulated
        if grad is None or grad==0:
            simType='g'
            grad=RFpulse.waveform[:,3]*0.01
        else:
            raise ValueError('ERROR! You cannot supply a gradient-modulated RFpulse AND specify the gradient strength.')
    dfdx=dfdx/100 #convert from cm to m
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
            if simType=='g':
                Beff_tmp=np.sqrt(rfB1**2+(grad*dfdx+hmat._Bfield*hmat.shifts[spinct]/1e6)**2)
                alpha_tmp=np.arctan2(grad*dfdx+hmat._Bfield*hmat.shifts[spinct]/1e6,rfB1)
            elif simType=='f':
                Beff_tmp=np.sqrt(rfB1**2+(dfdx/hmat._gamma+hmat._Bfield*hmat.shifts[spinct]/1e6)**2)
                alpha_tmp=np.arctan2(dfdx/hmat._gamma+hmat._Bfield*hmat.shifts[spinct]/1e6,rfB1)
            theta_tmp=2*np.pi*hmat._gamma*Beff_tmp*dt
            # Now loop through the rf pulse. With list comprehensions, Rz, Rx etc
            # are lists of 3x3 rotation matrices and the list has length RF
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
    """
    Simulates the effects of rotation about the z-axis. This function is a 
    wrapper that calls sim_rotate with whichax='z'.

    Parameters
    ----------
    d_in : list of numpy arrays
        Input density matrices.
    Hlist : list of pyFidA.Hamiltonian objects
        Hamiltonian operator(s) for the spin system.
    anglein : float
        Spoil angle in degrees. (Note that numpy arrays and lists are allowed,
        if different spoiling is needed for different parts of the spin system,
        but the common case is the same value for all parts of the spin system.
        The default is 90.

    Returns
    -------
    d_out : list of numpy arrays
        Output density matrix following z-rotation.

    """
    d_out=sim_rotate(d_in,Hlist,anglein=anglein,whichax='z')
    return d_out

