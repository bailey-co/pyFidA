#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:43:55 2026
pyFidA.fidA_sim.sim_makebasis.py

@author: Colleen Bailey (@cbailey@sri.utoronto.ca), based on Matlab code by Jamie Near

High-level functions for simulating basis sets, including reading in metabolite
information and simulating a basis set for writing to LCModel.

Functions:
    * read_all_spinsys_matlab
    * read_spinsys_matlab
    * sim_lcmrawbasis
"""

import importlib.resources as importlib_resources
import os
from scipy.io import loadmat
from pyFidA.fidA_io import io_writelcmraw
from pyFidA.fidA_sim.sim_sequences import sim_spinecho, sim_press, sim_steam, sim_laser

def read_all_spinsys_matlab(fname=None):
    """
    Imports the spinSystems.mat file that contains spin systems for many
    metabolites.

    Parameters
    ----------
    fname : string or path object, optional
        Filename of the .mat file to be loaded. If None (the default), function
        will look in pyFidA.assets.metabolites for spinSystems.mat and try to
        import that file.

    Returns
    -------
    fullDict : dict
        Dictionary for the form {'sysAla': ala_dict, 'sysAsc': asc_dict,...}.
        Each metabolite is a separate dict entry and the corresponding value
        is either a dict (in the case of inseparable spin systems) with the 
        spin system information (name, peak positions (shifts), J-couplings 
        and relative scaling factor) or a list of dicts (in the case where
        parts of the spin system can be separated)

    """
    if fname is None:
        pname=importlib_resources.files('pyFidA')
        fname=os.path.join(pname,'assets','metabolites','spinSystems.mat')
    # A dict with keys sysAla, sysAsc, etc. where the values are either dicts
    # of 'J', 'shifts', etc OR a list of such dicts when the spin system is broken up.
    fullDict={kn:vn for kn,vn in loadmat(fname,simplify_cells=True).items() if kn.startswith('sys')}
    return fullDict

def read_spinsys_matlab(fname='list'):
    """
    Import the .mat file for a spin system into a dict

    Parameters
    ----------
    fname : string or path object, optional
        Filename of the .mat file to import. If the filename, as entered, 
        doesn't exist, function will try to prepend the location of pyFidA's
        metabolites folder so that users can enter just 'Ala.mat', 'GPC.mat', 
        etc. For a list of available metabolites, enter 'list' in place of a
        filename and the list of spin system files available from pyFidA's
        metabolites folder will be printed and returned.

    Returns
    -------
    sysmet : dict or list of dicts
        For an inseparable spin system, spin system parameters in the format:
            {'name': str,
             'shifts': 1D np.ndarray,
             'J': 2D np.ndarray,
             'scaleFactor': int}
        For a separable spin system, each part of the system is its own dict 
        entry in a list, with the dicts having the above format.
        
    OR
    
    flist : list
        If fname=='list' then the output is a list of all available metabolites
        in pyFidA's metabolites folder. This list is also printed when running
        in list mode.

    """
    # If run in list mode
    if fname=='list':
        pname=importlib_resources.files('pyFidA')
        pname=os.path.join(pname,'assets','metabolites')
        flist=[fn for fn in os.listdir(pname) if fn.endswith('.mat')]
        print('Available metabolites: '+ str(flist))
        return flist
    # If filename doesn't exist, try to append entered fname to pyFidA/assets/metabolites
    elif not os.path.exists(fname):
        pname=importlib_resources.files('pyFidA')
        fname=os.path.join(pname,'assets','metabolites',fname)
        # loadmat will contain a number of variable, including __header__ and __version___.
        # Want to select just the spin system, which always starts with 'sys'
        sysmet=[vn for kn,vn in loadmat(fname,simplify_cells=True).items() if kn.startswith('sys')][0]
        return sysmet

def sim_lcmrawbasis(npts,sw,Bfield,linewidth,seq,metab,tau1,tau2=None,addref='n',makeraw='y',fname=None):
    """
    Generate an LCModel .RAW file to be used as an individual metabolite basis 
    spectrum in an LCModel basis set. The relevant characteristics of the 
    acquisition can be specified with the input parameters.

    Parameters
    ----------
    npts : int
        Number of points in fid/spectrum.
    sw : float
        Desired spectral width in Hz.
    Bfield : float
        Main magnetic field strength in Tesla.
    linewidth : float
        Spectral linewidth in Hz.
    seq : str
        Pulse sequence ('se' for Spin Echo, 'p' for PRESS, 'st' for STEAM, or
        'l' for LASER).
    metab : str
        Metabolite spin system to load. For a list of available spin systems,
        type "pyFidA.read_spinsys_matlab('list')".
    tau1 : float
        First echo time in ms.
    tau2 : float, optional
        For PRESS (seq='p'), second echo time in ms. For STEAM (seq='st'), 
        mixing time in ms. Not used in SE or LASER. The default is None, which 
        will fail fro PRESS and STEAM cases.
    addref : 'y' or 'n', optional
        Whether to add a reference peak at 0 ppm (for use in LCModel 
        makebasis). The default is 'n'.
    makeraw : 'y' or 'n', optional
        Whether or not to write the .RAW file (vs just returning the FID 
        object with the simulated spectrum). The default is 'y'.
    fname : str or path, optional
        Filename to write the raw file to. The default is None, which will save
        a file with the name metab.RAW in the current directory.

    Raises
    ------
    ValueError
        Sequence specified by seq is not recognized.

    Returns
    -------
    RF : npts x 2 np.ndarray, optional
        Array where columns are the real and imaginary parts of the fid. Only
        returned in the case where makeraw='y', to stay consistent with Matlab
    out1 : FID object
        Simulated spectrum for the specified metabolite and sequence.

    """
    if metab.endswith('.mat'):
        metab=metab[:metab.rfind('.')]
    spinSys=read_spinsys_matlab(metab+'.mat')
    if spinSys is dict:
        spinSys=[spinSys]
    tot_spins=sum([len(eachpart.shifts) for eachpart in spinSys])
    print('simulating metabolite {:s} with {:d} spins... Please wait...'.format(metab,tot_spins))
    spinlist=[spinSys]
    out1=list()
    if addref.lower()=='y':
        sysRef=[{'name':'Ref_0ppm','shifts':[0],'J':[0],'scaleFactor':1}]
        spinlist=[spinSys,sysRef]
    for eachsys in spinlist:
        # default shift reference (water=4.65 ppm) and center frequency (4.65 ppm) used.
        if seq.lower()=='se':
            out1.append(sim_spinecho(npts,sw,Bfield,linewidth,eachsys,tau1))
        elif seq.lower()=='p':
            out1.append(sim_press(npts,sw,Bfield,linewidth,eachsys,tau1,tau2))
        elif seq.lower()=='st':
            out1.append(sim_steam(npts,sw,Bfield,linewidth,eachsys,tau1,tau2))
        elif seq.lower()=='l':
            out1.append(sim_laser(npts,sw,Bfield,linewidth,eachsys,tau1))
        else:
            raise ValueError('ERROR:  Sequence {:s} not recognized!!!'.format(seq))
    out1=sum(out1,start=0*out1[0])
    if makeraw.lower()=='y':
        if fname is None:
            fname=metab+'.RAW'
        RF=io_writelcmraw(out1,fname,metab)
        return RF,out1
    else:
        return out1
    
def sim_make2DSimPlot(*args,**kwargs):
    """
    A similar function to this already exists in the fidA_display module: 
    disp_lcm_ridgeplot. It may be possible to write a wrapper to call that 
    unction from here in the future. However, because it uses seaborn and may
    use other packages in the future, which are not yet specified in the pyFidA
    dependencies, the fidA_display module is not yet set up to work on pyFidA 
    import.
    Therefore, this function currently just displays a message explaining this.

    """
    print('Note: Please use fidA_disp.disp_lcm_ridgeplot instead')
    