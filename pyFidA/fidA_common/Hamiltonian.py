#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:58:48 2026
pyFidA.fidA_common.Hamiltonian.py

@author: Colleen Bailey (cbailey@sri.utoronto.ca), based on Matlab code by 
    Jamie Near

Contains the Hamiltonian class, which can be used to generate the HAB matrix, 
with and without J-coupling, for a particular spin system. Currently, this
object is only used in the fidA_sim sub-module of pyFidA and instances of the
Hamiltonian object are typically created with the function sim_Hamiltonian
rather than initializing an instance with a direct call to Hamiltonian.__init__.

The four I-matrices that define the basis set for a spin-1/2 system are also 
defined in this file.
"""

import numpy as np
from pyFidA.fidA_common import GAMMA_DICT

# Basis states for spin-1/2
I0=np.eye(2)
Ix=(1/2)*np.array([[0,1],[1,0]])
Iy=(1j/2)*np.array([[0,1],[-1,0]])
Iz=(1/2)*np.array([[-1,0],[0,1]])
base_I=[Ix,Iy,Iz]
        
class Hamiltonian(object):
    def __init__(self,spin_dict,Bfield,nucleus='1H'):
        """
        Hamiltonian for a spin system. When a molecule's spin system contains 
        non-interacting parts that can be separated (eg. the NH2 protons in Gln 
        are not J-coupled to the other protons in the molecule), a separate 
        Hamiltonian instance can be generated for each part and placed into a 
        list using sim_Hamiltonian. 
        The fidA_sim functions anticipate a list of Hamiltonian objects 
        generated in this way (even if the list only has length 1 and contains 
        all info for the molecule in one system) rather than a Hamiltonian 
        object directly. Placing non-interacting spins in separate list 
        elements speeds up calculations.

        Parameters
        ----------
        spin_dict : dict
            Contains the spin system information in four parts:
                * 'name': a string that describes the spin system
                * 'shifts': an n-element array of frequencies, in ppm, for 
                    all n nuclei in the spin system. If there is only 1 element
                    it can be written as a float.
                * 'J': an n x n array containing the information about J-coupling
                    between spins (only upper triangle info needs to be filled.
                    If there is only 1 spin, this can be int 0)
                * 'scaleFactor': float defining the relative amplitude of this 
                    part of the spin system relative to others. (eg. the 
                    trimethyl protons in PCh can be represented by a single
                    chemical shift but have a scaleFactor of 9 compared to the
                    protons in the rest of the molecule)
        Bfield : float
            Magnetic field strength, in Tesla.
        nucleus : string, optional
            String that identifies the nucleus of the spin system. This 
            identifies the gyromagnetic ratio to be used for frequency 
            calculations. The default is '1H', which defines 
            self._gamma=42.577*1e6.

        Returns
        -------
        None.
        Some relevant attributes of the Hamiltonian that can be referenced are:
            * initd: the initial density matrix for the system
            * HAB: matrix describing precession for each spin, including J-coupling
            * HABJonly: matrix describing just the J-couplings between spins
            * nspins: number of spins in the system
            * omega0: main precession frequency for the nucleus in rads (corresponds to 0 ppm offset)
            * coherenceOrder(): a function to calculate the coherence order for the spin system
        """
        self.name=spin_dict['name']
        self.shifts=spin_dict['shifts']
        self.J=2*np.pi*spin_dict['J']
        self.scaleFactor=spin_dict['scaleFactor']
        self._nucleus=nucleus
        self._gamma=GAMMA_DICT[nucleus]*1e6 # in Hz
        self._Bfield=Bfield
        self._nspins=len(self.shifts)
        self._omega0=-2*np.pi*self._Bfield*self._gamma
        # self.Imats is a list [Ix, Iy, Iz]
        self.Imats=self.initialize_Imats()
        # Similarly, Fx, Fy, Fz are the 3 elements of the list self.F
        self.F=[np.sum(imat,axis=2) for imat in self.Imats]
        # Store initial density matrix for the spin system, including scaleFactor
        self.initd=self.F[2]*self.scaleFactor
        # Shifts is 1D with length nspins but will be broadcast to match the 
        # size of Imats[2], which has last dimension nspins. Then you can sum 
        # along this axis to get HAB
        self.HAB=self._omega0/1e6*np.sum(self.shifts*self.Imats[2],axis=2)
        # Now need the J-coupling calculation over all i and j combos
        self.HABJonly=np.zeros_like(self.HAB)
        for spin1 in range(self._nspins):
            for spin2 in range(spin1,self._nspins):
                IS=sum([np.matmul(self.Imats[dirct][:,:,spin1],self.Imats[dirct][:,:,spin2]) for dirct in range(3)])
                # Note that J is already multiplied by 2*pi above so no need to do it again here.
                JIS=self.J[spin1,spin2]*IS
                self.HABJonly=self.HABJonly+JIS
                self.HAB=self.HAB+JIS
        self.coherenceOrder=self.coherenceOrder()
    @property
    def nspins(self):
        return self._nspins
    @property
    def omega0(self):
        return self._omega0
    @property
    def shifts_rads(self):
        return self.shifts*(self.omega0/1e6)
    def coherenceOrder(self):
        p0=np.array([[0,-1],[1,0]])
        p=p0.copy()
        for spinct in range(self.nspins-1):
            p=np.kron(np.ones([2,2]),p)+np.kron(p0,np.ones(p.shape))
        return p
    def initialize_Imats(self):
        # Storing the I matrices as a list containing Ix, Iy, Iz
        Imats=list()
        for dirct,base_i in enumerate(base_I):
            # Iy is imaginary.
            if dirct==1:
                Imats.append(np.zeros([2**self.nspins,2**self.nspins,self.nspins],dtype=np.complex128))
            else:
                Imats.append(np.zeros([2**self.nspins,2**self.nspins,self.nspins]))
            # Loop through all spins
            for spinct in range(self.nspins):
                if spinct==0:
                    temp_I=base_i.copy()
                else:
                    temp_I=I0.copy()
                # loop through spins again starting at 2nd
                for pos in range(1,self.nspins):
                    if spinct==pos:
                        temp_I=np.kron(temp_I,base_i)
                    else:
                        temp_I=np.kron(temp_I,I0)
                Imats[dirct][:,:,spinct]=temp_I.copy()
        return Imats