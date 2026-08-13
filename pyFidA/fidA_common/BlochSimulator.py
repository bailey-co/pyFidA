#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:10:29 2026
pyFidA.fidA_common.BlochSimulator.py

@author: Colleen Bailey (cbailey@sri.utoronto.ca), based on Matlab code by 
    Jamie Near (the Bloch Simulator Matlab code was developed by Martyn Klassen)

Contains the BlochSimulator class, which can be initialized with an RF waveform
and other values to run through a numerical simulation of how magnetization
evolves over time based on the Bloch equations. The results of the simulation
are saved in the BlochSimulator.mvec and BlochSimulator.final_M0 attributes.

I have attempted to replicate the Matlab code, which avoids re-calculating some
matrix multiplications and exponentials. However, this Python code still appears
to run slightly slower than Matlab.
"""

import numpy as np
from pyFidA.fidA_common import GAMMA_DICT, FidAException

class BlochSimulator(object):
    def __init__(self,rfwf2, pulsewidth, f0, B1, phase=0, M0=np.r_[0,0,1], T1=np.inf, T2=np.inf, gamma=GAMMA_DICT['1H']):
        """
        Bloch equation simulator.

        Parameters
        ----------
        rfwf2 : n x 3 or n x 4 numpy array
            Radio frequency pulse with n time steps along the rows. The columns
            define the phase (in degrees), amplitude, duration (relative. 
            Adjusted to sum to pulsewidth) and (optionally) gradient (in G/cm)
            at each time point.
        pulsewidth : float
            Duration of the pulse in ms.
        f0 : float or array-like (a max of 1 of f0 or B1 can be array-like)
            Frequencies (in kHz) that the Bloch simulation should be run for.
        B1 : float or array-like (a max of 1 of f0 or B1 can be array-like)
            RF powers (in kHz) that the Bloch simulation should be run for.
        phase : float, optional
            The phase at which the RF pulse is applied. The default is 0.
        M0 : 3-element array, optional
            Vector describing the initial magnetization in the x, y and z 
            directions. Must be normalized to have magnitude 1. The default is 
            np.r_[0,0,1].
        T1 : float, optional
            T1 relaxation time constant in ms. The default is np.inf (no T1 decay).
        T2 : float, optional
            T2 relaxation time constant in ms. The default is np.inf (no T2 decay).
        gamma : float, optional
            Gyromagnetic ratio of the nucleus of interest for the simulation
            in MHz/T. This value is only used in the case of gradient-modulated
            pulses, to convert gradients in G/cm to a dephasing for that nucleus
            at a particular time step. For non-gradient-modulated pulses, the
            value is irrelevant. The default is GAMMA_DICT['1H']=42.577.

        Returns
        -------
        None.
        The results of the simulation are available in the attributes
        BlochSimulator.mvec (for magnetization vector evolution over time), 
        BlochSimulator.finalM0 (a 3 x m vector of the magnetization at the final
        time point for all m frequency points (if f0 is array-like) or B1 
        powers (if B1 is array-like)). BlochSimulator.finalMz and the complex-
        valued BlochSimulator.finalMxy are also accessible as attributes after
        simulation.

        """
        rfwf=rfwf2.copy()
        # T1 and T2 are in the same units as dt/pulsewidth, ie. ms
        self._tct=0
        self._mct=0
        # Only one of f0 or B1 is iterable/a numpy array, or both can be scalar
        # but they can't both be iterable.
        if hasattr(B1,'__iter__'):
            if hasattr(f0,'__iter__'):
                raise FidAException('ERROR: Only one of f0 and B1 can be a vector')
            self.scantype='b'
        elif hasattr(f0,'__iter__'):
            self.scantype='f'
        else:
            self.scantype='t'
        # If no time steps, assume equal temporal spacing
        if rfwf.shape[1]==2:
            rfwf=np.concatenate((rfwf,np.ones([rfwf.shape[0],1])),axis=1)
        # Cut out points where duration is 0
        rfwf=rfwf[rfwf[:,2]!=0,:]
        # This differs slightly from Matlab in that it explicitly tests that the 
        # timesteps are positive integers, rather than just checking that they're
        # positive and then rounding.
        if np.amin(rfwf[:,2])<1 or not all([np.isclose(round(eachval),eachval) for eachval in rfwf[:,2]]):
            raise FidAException('ERROR: Pulse step duration should be zero or positive int')
        self.steplist=np.int32(np.round(rfwf[:,2]))
        self.phaselist=(rfwf[:,0]+phase)*np.pi/180
        # Normalize amplitudes
        self.amplist=rfwf[:,1]/np.amax(np.abs(rfwf[:,1]))
        self.gradflag=False
        if rfwf.shape[1]==4 and any(rfwf[:,3]!=0):
            # Convert from G/cm to kHz/cm
            self.gradlist=rfwf[:,3]*(gamma/10)
            self.gradflag=True
        elif rfwf.shape[1]>4:
            raise FidAException('ERROR: Too many columns in rf waveform. Max is 4')
        self.B1max=2*np.pi*B1
        self.initf=2*np.pi*f0
        self.dt=pulsewidth/np.sum(self.steplist)
        try:
            self.R1=1/T1
        except ZeroDivisionError:
            self.R1=np.inf
        try:
            self.R2=1/T2
        except ZeroDivisionError:
            self.R2=np.inf
        if np.sum(M0*M0)<1.00001:
            self.initM0=M0
            self.M0=M0
        else:
            raise FidAException('ERROR: Initial magnetization is greater than 1.0')
            
        # Since dt, R2 and R1 are the same for every time step, _tct, the
        # matrices using just those (Amat, Bmat and _phi) can be calculated 
        # here be calculated here. phi will be adjusted for gradient-modulated
        # waveforms in the self.phi property.
        self.Amat=np.array([[np.exp(-self.dt*self.R2),0,0],
                    [0,np.exp(-self.dt*self.R2),0],
                    [0,0,np.exp(-self.dt*self.R1)]])
        self.Bmat=np.array([0,0,1-np.exp(-self.dt*self.R1)])
        self._phi=self.initf*self.dt
        
        # Initialize main outputs
        self.mvec=list()
        self.finalM0=None
        # Run the simulation
        self.run_allsteps()
    @property
    def phi(self):
        if self.gradflag:
            return self.initf*self.dt*self.gradlist[self._tct]
        else:
            return self._phi
    @property
    def Rz1(self):
        # The initial matrix to align the magnetization based on the phase of
        # the first RF pulse point. (Written generally for _tct but only ever)
        # called when self._tct=0
        theta=-1*self.phaselist[self._tct]
        Rz1=np.array([[np.cos(theta),np.sin(theta),0],
                      [-1*np.sin(theta),np.cos(theta),0],
                      [0,0,1]])
        return Rz1
    @property
    def Rx(self):
        # RF rotation matrix
        theta=self.B1max*self.amplist[self._tct]*self.dt
        if self.scantype=='b':
            Rx=np.zeros([len(theta),3,3])
            Rx[:,0,0]=np.ones([len(theta),])
            Rx[:,1,1]=np.cos(theta)
            Rx[:,1,2]=np.sin(theta)
            Rx[:,2,1]=-1*np.sin(theta)
            Rx[:,2,2]=np.cos(theta)
        else:
            Rx=np.array([[1,0,0],
                         [0,np.cos(theta),np.sin(theta)],
                         [0,-1*np.sin(theta),np.cos(theta)]])
        return Rx
    @property
    def Rz2(self):
        """
        Rotation about z for the current time step, _tct (and part of the next
        time step). Note that this has three components, as explained more 
        fully in calc_rotmat_step: free precession (and possibly a phase 
        accumulation due to a gradient), a reversal of the RF pulse phase for 
        this time step and then applying the RF pulse phase for the next time 
        step. (On the last time step, only the final phase value is needed 
        rather than the phase difference). The T1 and T2 relaxation effects are 
        added in later in the calc_rotmat function rather than in this matrix 
        calculation

        Returns
        -------
        Rz2 : m x 3 x 3 or 3 x 3 numpy array
            Matrix for the combined z-rotations described above. When f0 is an
            m-element vector, the rotation matrix for each frequency is held
            in the first dimension. In the case where f0 is not a vector, the
            returned array is 3 x 3.

        """
        if self._tct+1<len(self.phaselist):
            if self._mct==0:
                theta=self.phi+self.phaselist[self._tct]-self.phaselist[self._tct+1]
            else:
                theta=self.phi
        else:
            theta=self.phi+self.phaselist[self._tct]
        if self.scantype=='f':
            Rz2=np.zeros([len(theta),3,3])
            Rz2[:,0,0]=np.cos(theta)
            Rz2[:,0,1]=np.sin(theta)
            Rz2[:,1,0]=-1*np.sin(theta)
            Rz2[:,1,1]=np.cos(theta)
            Rz2[:,2,2]=np.ones([len(theta),])
        else:
            Rz2=np.array([[np.cos(theta),np.sin(theta),0],
                      [-1*np.sin(theta),np.cos(theta),0],
                      [0,0,1]])
        return Rz2
    def calc_rotmat_step(self):
        """
        Sets self.M0 for the current time step self._tct. The rotation matrix 
        for each timestep can be obtained by rotating around z by the phase of 
        the RF pulse, rotating around x based on the B1 applied during that 
        step, calculating the free precession for that step (including 
        potentially any gradient applied during the RF pulse) and then rotating
        back around z by the negative of the RF pulse at that time point before 
        moving to the next time step. However, if you consider consecutive 
        steps, there are three rotations around z in a row 
        (precession_i, -phase_i, phase_i+1). The full angle can be obtained by 
        adding these angles together and then creating a single matrix for a 
        rotation about z by that angle, without doing three more 
        computationally expensive matrix rotations separately. The only 
        exception is the last step, where there is no next i+1 phase and you 
        just have to rotate back by the final phase value.

        Some rotation matrices are m x 3 x 3 in the case where f0 or B1 were
        entered as an m-element array. This was originally accounted for in
        this code by using broadcasting and using Einstein summation to get
        the matrix multiplications for all m points. This looked elegant but 
        was taking a long time to run. It is better to make use of the a priori 
        knowledge about the rotation matrices (that some elements are always 0 
        while others will be vectors related to the sine or cosine of an angle) 
        to limit calculations, as is done in Matlab. This is now implemented 
        here and is indeed much faster.

        """
        if self.scantype=='b':
            theta=self.B1max*self.amplist[self._tct]*self.dt
            mtmp=np.cos(theta)*self.M0[1,:]+np.sin(theta)*self.M0[2,:]
            self.M0[2,:]=-1*np.sin(theta)*self.M0[1,:]+np.cos(theta)*self.M0[2,:]
            self.M0[1,:]=mtmp
            # Apply T1 and T2 relaxation for this timestep through Amat/Bmat
            Rztmp=np.matmul(self.Amat,self.Rz2)
            self.M0=np.matmul(Rztmp,self.M0)
            self.M0[2,:]=self.M0[2,:]+self.Bmat[2]
        elif self.scantype=='f':
            if self._tct+1<len(self.phaselist):
                if self._mct==0:
                    theta=self.phi+self.phaselist[self._tct]-self.phaselist[self._tct+1]
                else:
                    theta=self.phi
            else:
                theta=self.phi+self.phaselist[self._tct]
            # Apply T1 and T2 relaxation for this timestep through Amat/Bmat
            mtmp=np.matmul(np.matmul(self.Amat,self.Rx),self.M0)
            self.M0[2,:]=mtmp[2,:]+self.Bmat[2]
            self.M0[0,:]=np.cos(theta)*mtmp[0,:]+np.sin(theta)*mtmp[1,:]
            self.M0[1,:]=-1*np.sin(theta)*mtmp[0,:]+np.cos(theta)*mtmp[1,:]
        else: # Case where f0 and B1 are both scalars and rotation matrices are just 3x3
            self.M0=np.matmul(self.Amat,np.matmul(self.Rz2,np.matmul(self.Rx,self.M0)))+self.Bmat
    def run_allsteps(self):
        # re-initialize some variables in case this function is called multiple
        # times. eg. after changing one variable. _tct are the time steps 
        # outlined in rfwf[:,2]. Often these are all 1 but, if not, _mct will 
        # repeat calculations for larger values without recalculating every
        # matrix (since values are unchanged).
        self._tct=0
        self._mct=0
        self.mvec=list()
        self.M0=self.initM0
        # Set the initial phase since the rotation matrix for each step starts
        # with the RF. This is a 3x3 matrix so result with be 3x1 and can be
        # replicated for every initf or B1max (since initM0 is the same for all
        # of them)
        self.M0=np.matmul(self.Rz1,self.initM0)
        if self.scantype=='f':
            self.M0=np.tile(self.M0,[len(self.initf),1]).T
        elif self.scantype=='b':
            self.M0=np.tile(self.M0,[len(self.B1max),1]).T
            
        # Calculate the magnetization at each timestep. Note that the last time
        # step is treated slightly differently but this is dealt with in the
        # Rz2 matrix, above.
        for stepct,stepval in enumerate(self.steplist):
            for mct in range(0,stepval):
                self.calc_rotmat_step()
                self._mct=self._mct+1
            self._tct=self._tct+1
            self._mct=0
            self.mvec.append(self.M0.copy())
        self.finalM0=self.M0.copy()
        self.mvec=np.array(self.mvec)
    @property
    def finalMz(self):
        return self.finalM0[2,:]
    @property
    def finalMxy(self):
        return np.abs(self.finalM0[0,:]+1j*self.finalM0[1,:])
    @property
    def tvec(self):
        return self.dt*(np.cumsum(self.steplist)-self.steplist[0])