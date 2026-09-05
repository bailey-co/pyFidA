#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:52:34 2026
pyFidA.fidA_processing.op_peak_fitting.py

@author: Colleen Bailey (@cbailey@sri.utoronto.ca), based on Matlab code by Jamie Near

Functions for fitting MR spectra, including lineshapes, integration and fitting 
functions:
    * op_creFit
    * op_gauss_linbas
    * op_gauss
    * op_integrate
    * op_lorentz_linbas
    * op_lorentz
    * op_peakFit
    * op_voigt_linbas
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import voigt_profile
from pyFidA.fidA_common import FidAException, FidAWarning
from .curvefit_tools import nlinfit
from .op_common_processing import add_phase, op_addphase,freqrange,op_freqrange
import warnings

def op_creFit(indat,ph0=0,ph1=0,ppmmin=2.9,ppmmax=3.15,peaktype='lorentz2',parameter_bounds=False,disp_result='partial'):
    """
    parsFitHz=op_creFit(indat,ph0=0,ph1=0,ppmmin=2.9,ppmmax=3.15,peaktype='lorentz',parameter_bounds=False,disp_result='partial')
    
    Fits the Creatine region of the spectrum to a single peak (Lorentzian 
    lineshape by default). Also displays the fit by default. This function is a 
    wrapper for op_peakFit with convenient defaults for fitting the 2.9-3.15 ppm
    region. If you need a more complex fit, use op_peakFit instead.

    Parameters
    ----------
    indat : FID object
        Input data with Creatine peak to be fitted.
    ph0 : float, optional
        Zeroth order phase (in degrees) to add to the spectrum before fitting. 
        The default is 0.
    ph1 : float, optional
        Amount of 1st order phase shift (specified as horizontal shift in 
        seconds in the time domain) to add to the spectrum before fitting. The 
        default is 0.
    ppmmin : float, optional
        Lower bound of the frequency range to include in the peak fit. The default is 2.9.
    ppmmax : float, optional
        Upper bound of the frequency range to include in the peak fit. The default is 3.15.
    peaktype : str, optional
        Defines the lineshape of the peak to be fit. Possibilities are 
        'lorentz2' (which fits baseline offset but not slope) and 'lorentz' 
        (fits baseline offset and slope). In contrast to Matlab fid-A, the 
        baseline parameters are complex and the default is to fit the baseline
        offset only (without baseline slope). Because the ppm range is so small,
        the baseline parameters are quite sensitive to noise and large baseline
        slopes that do not agree with the baseline of the full spectrum can 
        result. Therefore, the default is 'lorentz2'.
    parameter_bounds : boolean, list, tuple or scipy.optimize Bounds object, optional
        Upper and lower bounds of the parameters passed to op_peakFit. True will
        create the default upper and lower parameter bounds defined in op_peakFit.
        False will use -inf and +inf. A list, tuple or Bounds object will 
        contain the values for the lower and upper bounds, as describe in 
        op_peakFit. The default is False.
    disp_result : boolean or 'partial', optional
        Describes how the spectrum and fit are plotted, as well as whether the 
        fit parameters and area estimate are printed. If False, no plot or text 
        is output. If True, the full spectrum/fit are plotted and text is outputted. 
        If 'partial' the region of the spectrum/fit between ppmmin and ppmmax is
        plotted and text for the fit parameters and area are outputted. The 
        default is 'partial'.

    Returns
    -------
    parsFitHz : list
        List of fit parameters (amplitude, FWHM in Hz, ppm0 peak position in ppm,
        bsaeline slope, baseline intercept). Note that the FWHM is returned in 
        Hz even though op_lorentz_linbas uses the FWHM in ppm. If you need the
        FWHM in ppm or want to send parsFitHz to op_lorentz_linbas to get a FID
        object of the fit, you can divide parsFitHz[1]/(indat.txfreq/1e6) or 
        use op_peakFit with return_pars_in_Hz=False instead of using op_creFit.
    """
    # As with op_alignScans, the main question here is if the data have more
    # than one dimension. Other checks just provide more specific info on what's
    # wrong.
    if indat.ndim!=1:
        if indat.flags['isFourSteps']:
            raise FidAException('ERROR: Must have combined subspecs in order to do this! ABORTING')
        if 'averages' in indat:
            raise FidAException('ERROR: Must have averaged in order to do this! ABORTING')
        if 'coils' in indat:
            raise FidAException('ERROR: Must have added receivers in order to do this! ABORTING')
        raise FidAException('ERROR: For fitting, data can only have one dimension. Use op_squeeze() to remove singleton dimensions. Your data are: ({:s})'.format(', '.join(indat.dims)))
    phased_spec=op_addphase(indat,ph0,ph1)
    # Initial estimate of the baseline slope in the part of the spectrum where 
    # the peak is. The fit can be quite sensitive to baseline parameters
    ind_vals=np.logical_and(np.greater(indat.ppm,ppmmin),np.less(indat.ppm,ppmmax))
    ppm_part=indat.ppm[ind_vals]
    spec_part=phased_spec.specs[ind_vals]
    base_slope=(spec_part[0]-spec_part[-1])/(ppm_part[0]-ppm_part[-1])
    # amp, FWHM (in Hz), ppm0, base_slope, base_off
    if peaktype=='lorentz':
        parsGuess=[np.amax(np.real(spec_part)),
                  3,
                  ppm_part[np.argmax(np.real(spec_part))], # in Matlab, this is hard-coded as 3.02 and ppmmin and ppmmax are also hard-coded
                  base_slope,
                  spec_part[0]] # phase is added above and not fit for
    elif peaktype=='lorentz2':
        parsGuess=[np.amax(np.real(spec_part)),
                  3,
                  ppm_part[np.argmax(np.real(spec_part))], # in Matlab, this is hard-coded as 3.02 and ppmmin and ppmmax are also hard-coded
                  spec_part[0]]
    else:
        raise Exception('ERROR: peaktype not recognized. Must be one of "lorentz" or "lorentz2.')
    outdat,parsFitHz,resids=op_peakFit(indat,ppmmin=ppmmin,ppmmax=ppmmax,parsGuess=parsGuess,peaktype=peaktype,parameter_bounds=parameter_bounds,show_plot=disp_result)
    if disp_result: # Plots are already done in op_peakFit. Now print parameters.
        print('Fit: '+', '.join(['{:3.2f}'.format(eachpar) for eachpar in parsFitHz]))
        print('Guess: '+', '.join(['{:3.2f}'.format(eachpar) for eachpar in parsGuess]))
    return parsFitHz

def op_gauss_linbas(pars,ppm):
    """
    y=op_gauss_linbas(pars,ppm)
    Generates a Gaussian peak or peaks and (optionally) a baseline and phasing 
    for the parameters pars over the frequencies in ppm. Multiple Gaussian 
    peaks can be generated by providing amplitude, FWHM and center frequency 
    paramaters as lists or numpy arrays. This function can be called with 
    pyFidA.curvefit_tools.nlinfit as a wrapper for scipy.optimize.curve_fit 
    in order to fit data to a series of Gaussian peaks. See op_peakFit for an 
    example or the notes in curvefit_tools for the details of how this done.

    Parameters
    ----------
    pars : list
        List of parameters to be fitted in the order 
        [amplitude, fwhm (in ppm), center frequency (in ppm), baseline slope, baseline offset, phase shift (in degrees)].
        The baseline slope, baseline offset and phase shift parameters are 
        optional and are scalars. The amplitude, fwhm and center frequency are
        required and can be scalars (for a single peak) or lists/numpy arrays 
        (for multiple peaks)
    ppm : numpy array
        Frequency axis vector (in ppm). Technically the units of the ppm vector 
        just need to match those of the fwhm and center frequency.

    Returns
    -------
    y : numpy array
        1D vector of the y-values specifying the Gaussian lineshape(s) 
        corresponding to the x-values in ppm.

    """
    # If pars is entered as a 2D numpy array, convert to list
    pars=list(pars)
    if len(pars)<3:
        raise ValueError('ERROR: pars must have length of at least 3 for amplitude, FWHM and center frequency.')
    else:
        if not hasattr(pars[0],'__iter__'): #single peak case with amplitude, etc entered as scalars. Convert to lists to avoid errors when iterating.
            pars=[[eachvar] for eachvar in pars[:3]]+[eachvar for eachvar in pars[3:]]
        for varct in range(3,len(pars)):
            if type(pars[varct]) is list:
                print('WARNING: Lists not allowed for baseline, offset or phase due to redundancy in fit parameters. Using first value only.')
                pars[varct]=pars[varct][0]
        # baseline parameters and phase shift are optional. Initialize to 0 if
        # no value given.
        full_pars=pars+[0]*(6-len(pars))
        [amp,fwhm,ppm0,base_slope,base_off,ph0]=full_pars
    # Warn if units appear to be Hz (value assumes 1H nucleus)
    if fwhm[0]>0.3:
        warnings.warn('WARNING: FWHM in op_gauss_linbas should be entered in ppm. Your value of {:3.3f} may be in Hz!'.format(fwhm[0]),FidAWarning)
    # Second parameter is FWHM but the equation below uses sigma. 
    sigma=[fv/2/np.sqrt(2*np.log(2)) for fv in fwhm]
    y=np.zeros([len(amp),len(ppm)],dtype=complex)
    for act,aval in enumerate(amp):
        y[act,:]=np.exp(-1*(ppm-ppm0[act])**2/2/sigma[act]**2)
        y[act,:]=y[act,:]/np.amax(np.abs(y[act,:]))*aval
    bas=base_off+base_slope*ppm
    y=np.sum(y,axis=0)
    y=add_phase(y+bas,ph0)
    return y.squeeze()

def op_gauss(pars,ppm):
    """
    yvals=op_gauss(pars,ppm)
    Wrapper function for op_gauss_linbas that fixes the baseline slope to zero.
    This is useful if you want to fit a function with a flat baseline that has
    an intercept but no slope. All other aspects of op_gauss_linbas apply: 
    baseline intercept and phase shift are optional parameters; peak 
    amplitudes, fwhm and center frequencies may be supplied as scalars for a 
    single peak or as lists/numpy arrays for multiple peaks.

    Parameters
    ----------
    pars : list
        List of parameters to be fitted in the order 
        [amplitude, fwhm (in ppm), center frequency (in ppm), baseline offset, phase shift (in degrees)].
        The baseline offset and phase shift parameters are optional and are 
        scalars. The amplitude, fwhm and center frequency are required and can 
        be scalars (for a single peak) or lists/numpy arrays (for multiple peaks)
    ppm : numpy array
        Frequency axis vector (in ppm). Technically the units of the ppm vector 
        just need to match those of the fwhm and center frequency.

    Returns
    -------
    yvals : numpy array
        1D vector of the y-values specifying the Gaussian lineshape(s) 
        corresponding to the x-values in ppm.

    """
    pars=list(pars)
    if len(pars)<3:
        raise ValueError('ERROR: pars must have length of at least 3 for amplitude, FWHM and center frequency.')
    elif len(pars)==3:
        pars=pars+[0,0,0]
    else:
        pars=pars[:3]+[0]+pars[3:] # This will put whatever comes after the first 3 elements of pars on the end, whether that's nothing, or the remaining 1-2 parameters. Any other missing parameters will be dealt with in op_gauss_linbas
    yvals=op_gauss_linbas(pars, ppm)
    return yvals

def op_integrate(indat,ppmmin,ppmmax,mode='re'):
    """
    intvals = op_integrate(indat,ppmmin,ppmmax,mode='re')
    Basic peak integration over a specified frequency range. By default, this 
    function integrates under the real part of the curve, but it can also be 
    made to integrate the imaginary part or the magnitude part by changing the
    "mode" parameter.

    Parameters
    ----------
    indat : FID object
        input data in the FID object format.
    ppmmin : float
        Min of the frequency range (in ppm) in which to calculate the integral.
    ppmmax : float
        Max of the frequency range (in ppm) in which to calculate the integral.
    mode : string ('re','im' or 'mag'), optional
        Selects whether the integral is performed onthe real, imaginary or
        magnitude part. The default is 're'.
        
    Returns
    -------
    intvals : float
        Estiamted area under the curve for the desired frequency range.

    """
    whichpts=np.flatnonzero(np.logical_and(indat.ppm>ppmmin,indat.ppm<ppmmax))
    if mode=='re':
        intvals=np.sum(np.real(indat.specs[whichpts,...]),axis=indat.dims['t'])
    elif mode=='im':
        intvals=np.sum(np.imag(indat.specs[whichpts,...]),axis=indat.dims['t'])
    elif mode=='mag':
        intvals=np.sum(np.abs(indat.specs[whichpts,...]),axis=indat.dims['t'])
    else:
        raise ValueError("ERROR: Mode not recognized. Must be 're','im' or 'mag'. Aborting!")
    return intvals

def op_lorentz_linbas(pars,ppm):
    """
    y=op_lorentz_linbas(pars,ppm)
    Generates a Lorentzian peak or peaks and (optionally) a baseline and phasing 
    for the parameters pars over the frequencies in ppm. Multiple Lorentzian 
    peaks can be generated by providing amplitude, FWHM and center frequency 
    paramaters as lists or numpy arrays. This function can be called with 
    pyFidA.curvefit_tools.nlinfit as a wrapper for scipy.optimize.curve_fit 
    in order to fit data to a series of Lorentzian peaks. See op_peakFit for an 
    example or the notes in curvefit_tools for the details of how this done.

    Parameters
    ----------
    pars : list
        List of parameters to be fitted in the order 
        [amplitude, fwhm (in ppm), center frequency (in ppm), baseline slope, baseline offset, phase shift (in degrees)].
        The baseline slope, baseline offset and phase shift parameters are 
        optional and are scalars. The amplitude, fwhm and center frequency are
        required and can be scalars (for a single peak) or lists/numpy arrays 
        (for multiple peaks)
    ppm : numpy array
        Frequency axis vector (in ppm). Technically the units of the ppm vector 
        just need to match those of the fwhm and center frequency.

    Returns
    -------
    y : numpy array
        1D vector of the y-values specifying the Lorentzian lineshape(s) 
        corresponding to the x-values in ppm.

    """
    pars=list(pars) # In case parameters are entered as 2D numpy array
    if len(pars)<3:
        raise ValueError('ERROR: pars must have length of at least 3 for amplitude, FWHM and center frequency.')
    else:
        if not hasattr(pars[0],'__iter__'): # single peak case with amplitude etc entered as scalars. Convert to lists so that code can loop through each peak.
            pars=[[eachvar] for eachvar in pars[:3]]+[eachvar for eachvar in pars[3:]]
        for varct in range(3,len(pars)):
            if type(pars[varct]) is list:
                print('WARNING: Lists not allowed for baseline, offset or phase due to redundancy in fit parameters. Using first value only.')
                pars[varct]=pars[varct][0]
        # amplitude, fwhm and ppm0 are required. Any optional parameters not entered after those will be set to 0.
        full_pars=pars+[0]*(6-len(pars))
        [amp,fwhm,ppm0,base_slope,base_off,ph0]=full_pars
    # op_lorentz_linbas divides by pi in Matlab but op_lorentz divides by 2 and this
    # is what gives FWHM in ppm.
    if fwhm[0]>0.3:
        warnings.warn('WARNING: FWHM in op_lorentz_linbas should be entered in ppm. Your value of {:3.3f} may be in Hz!'.format(fwhm[0]),FidAWarning)
    # In Matlab, op_lorentz_linbas divides by pi but op_lorentz divides by 2. If
    # fwhm is in ppm then the gamma parameter is just fwhm/2, which is what I've
    # done here for both op_lorentz_linbas and op_lorentz.
    gamma=[fv/2 for fv in fwhm]
    y=np.zeros([len(amp),len(ppm)],dtype=complex)
    for act,aval in enumerate(amp):
        # Not sure that this sqrt(2/pi) scaling is correct but shouldn't matter 
        # since result is divided by max before scaling by amp.
        y[act,:]=np.sqrt(2/np.pi)*(gamma[act]-1j*(ppm-ppm0[act]))/(gamma[act]**2+(ppm-ppm0[act])**2)
        y[act,:]=y[act,:]/np.amax(np.abs(y[act,:]))*aval
    bas=base_off+base_slope*ppm
    y=np.sum(y,axis=0)
    y=add_phase(y+bas,ph0)
    return y.squeeze()
    
def op_lorentz(pars,ppm):
    """
    yvals=op_lorentz(pars,ppm)
    Wrapper function for op_lorentz_linbas that fixes the baseline slope to 
    zero. This is useful if you want to fit a function with a flat baseline 
    that has an intercept but no slope. All other aspects of 
    op_lorentz_linbas apply: baseline intercept and phase shift are optional 
    parameters; peak amplitudes, fwhm and center frequencies may be supplied as 
    scalars for a single peak or as lists/numpy arrays for multiple peaks.

    Parameters
    ----------
    pars : list
        List of parameters to be fitted in the order 
        [amplitude, fwhm (in ppm), center frequency (in ppm), baseline offset, phase shift (in degrees)].
        The baseline offset and phase shift parameters are optional and are 
        scalars. The amplitude, fwhm and center frequency are required and can 
        be scalars (for a single peak) or lists/numpy arrays (for multiple peaks)
    ppm : numpy array
        Frequency axis vector (in ppm). Technically the units of the ppm vector 
        just need to match those of the fwhm and center frequency.

    Returns
    -------
    yvals : numpy array
        1D vector of the y-values specifying the Lorentzian lineshape 
        corresponding to the x-values in ppm.

    """
    pars=list(pars) # In case parameters are entered as 2D numpy array
    if len(pars)<3:
        raise ValueError('ERROR: pars must have length of at least 3 for amplitude, FWHM and center frequency.')
    elif len(pars)==3:
        pars=pars+[0,0,0]
    else:
        pars=pars[:3]+[0]+pars[3:] # This will put whatever comes after the first 3 elements of pars on the end, whether that's nothing, or the remaining 1-2 parameters. Extra stuff will be dealt with in op_lorentz_linbas
    yvals=op_lorentz_linbas(pars, ppm)
    return yvals

def op_peakFit(indat,ppmmin=0,ppmmax=4.2,parsGuess=None,peaktype='lorentz',parameter_bounds=False,real_ydata=False,show_plot=False,return_pars_in_Hz=True):
    """
    Perform a fit of a region of a spectrum to one or more peaks with a 
    specified lineshape. This function is able to deal with complex data and
    complex parameters, as well as multiple peaks. It is intended to be 
    flexible and some examples of fitting with different specifications will be
    outlined in a future tutorial.
    
    Parameters
    ----------
    indat : FID object
        Input data with spectrum to be fit.
    ppmmin : float, optional
        Lower frequency limit for fitting. The default is 0.
    ppmmax : float, optional
        Upper frequency limit for fitting. The default is 4.2.
    parsGuess : list or numpy array, optional
        Initial parameter guess with format dependent on fitting function 
        defined by peaktype. eg. for peaktype='lorentz', the format should match
        that of pars from op_lorentz_linbas EXCEPT that the fwhm is given in Hz 
        (it is converted here in op_peakFit):
        [amplitude, fwhm in Hz, frequency of peak(s), baseline slope, baseline offset, phase].
        For the lineshape functions in this toolbox (op_lorentz_linbas, 
        op_gauss_linbas, op_voigt_linbas, op_lorentz), more information is 
        available in the docstring of those functions, particularly about pars. 
        Generally, the baseline slope, baseline offset and phase shift 
        parameters are optional and are scalars, while the amplitude, fwhm and 
        center frequency are required and can be scalars (for a single peak) or 
        lists/numpy arrays (for multiple peaks). Parameters can be complex but
        the initial guess must explicitly enter a complex number or array (
        ie. something that passes np.iscomplexobj(mypar)=True).
        The default of None will attempt to set reasonable starting values for
        the function defined in peaktype, like:
        amp=np.amax(np.abs(indat.specs)),
        fwhm=5,
        ppm0=ppm[np.argmax(np.abs(specs))]
        0+0j for the baseline slope and baseline intercept
        0 for phase.
        If you want to fit without phase or baseline parameters, you will need 
        to enter a value for parsGuess that excludes these optional parameters,
        starting at the end (see tutorial)
    peaktype : str or function. optional
        The function that defines the lineshape used to generate the y-values 
        for fitting. Implemented string options are:
            'lorentz': op_lorentz_linbas
            'lorentz2': op_lorentz
            'gauss': op_gauss_linbas
            'voigt': op_voigt_linbas
        Users can also provide a function name directly, as long as its inputs
        are of the form [pars, ppm], although features like automatically 
        generating parameter_bounds and converting linewidths from Hz to ppm 
        will not be available. The default is 'lorentz'.
    parameter_bounds : boolean or list, numpy array or scipy's Bounds type, optional
        Describes how to deal with the bounds on the parameters. Options are:
            True - default values are defined based on certain assumptions eg.
                   amplitudes are between 0 and 2*np.amax*np.abs(indat.specs),
                   the center frequencies are between the ppmmin and ppmmax values
                   etc. This is only implemented for the 4 peaktype values 
                   identified with strings, above.
            False - bounds are -infinity to +infinity
            list, numpy array or Bounds object - explicit values for the bounds 
                of each parameters; should be a two-element tuple/list with
                each element having the same format as parsGuess. This will be 
                passed into scipy.optimize.curve_fit (after reformatting for 
                the multi-peak case)
            The default is False.
    real_ydata : boolean, optional
        Whether to take just the real part of the spectrum in indat for fitting. 
        The default is False.
    show_plot : boolean or str 'partial', optional
        Describes whether or how to plot the data and output fit information.
        True - Output the data, initial guess and fit across the full range of 
            indat. Also output text about the area under the curve.
        False - No plots or text output
        'partial' - Output the data, initial guess and fit across the range of 
            indat defined by ppmmin and ppmmax. Also output text about the 
            initial guess, fit and area under the curve.
        The default is False.
    return_pars_in_Hz : boolean, optional
        Describes whether fwhm parameters in the output parsFit are returned in
        Hz (True) or ppm (False). This exists because users are more likely to 
        know linewidth in Hz and this is how fwhm is entered in parsGuess, so
        returning a value in Hz makes the most sense. However, fitting functions
        like op_lorentz_linbas take fwhm in ppm (because they don't have access
        to transmit frequency to convert from Hz). It is sometimes useful to
        take the returned parameters from a fit and send them to a lineshape as
        op_lorentz_linbas(parsFit,indat.ppm) to generate the y-values 
        separately. Setting this parameter to False will return parsFit with 
        fwhm in ppm so that it can used with the lineshape functions if needed.
        The default is True.

    Returns
    -------
    outdat : FID object
        Fitted lineshape as the spectrum in a FID object.
    parsFit : list
        Fit parameters. Format is generally the same as parsGuess, but setting
        return_pars_in_Hz=False in the input arguments will convert any fwhm
        parameters in the list from Hz to ppm before return, as described above.
    resids : FID object
        Residuals from the fit (difference between data and fit) in a FID object.

    """
    
    def Hz_to_ppm(old_parList,whichvars,send_warning=True):
        """
        Function to convert fwhm parts of old_parList from Hz to ppm. The fwhm
        parameters are defined by whichvars. Can be used both for parsGuess and
        for parts of parameter_bounds.

        Parameters
        ----------
        old_parList : list
            List of all fit parameters in the format of parsGuess with the
            parameters defined by whichvars in Hz.
        whichvars : list of ints
            Indices of any parameters in list to be converted from Hz to ppm.
            Note that this must be in list format even if there is only a single
            fwhm value to convert.
        send_warning : boolean, optional
            Describes whether to throw a warning for fwhm values that may have
            been entered in ppm. This allows the warning to be turned off when
            passing the fit parameter bounds. Note that this is just a check 
            that the fwhm is >0.1 so it's geared for 1H spectra and not perfect.
            The default is True.

        Returns
        -------
        parList : list
            List of all fit parameters in the format of parsGuess with the
            parameters defined by whichvars now converted to ppm to send to the
            lineshape function for fitting.

        """
        parList=old_parList.copy()
        # Basic check that ppm is in correct range (1H case is the only one implemented right now)
        for varnum in whichvars:
            if send_warning:
                try: # list/np.array case
                    if parList[varnum][0]<0.1 and indat.nucleus[0]=='1H':
                        warnings.warn('WARNING: FWHM should be entered in Hz. Your value of {:3.3f} may be in ppm!'.format(parList[varnum][0]),FidAWarning)
                except TypeError: # int/float case
                    if parList[varnum]<0.1 and indat.nucleus[0]=='1H':
                        warnings.warn('WARNING: FWHM should be entered in Hz. Your value of {:3.3f} may be in ppm!'.format(parList[varnum]),FidAWarning)
            if type(parList[varnum]) is list:
                parList[varnum]=[pval/(indat.txfreq/1e6) for pval in parList[varnum]]
            else: # This should work for either np.ndarray or float/int scalars
                parList[varnum]=parList[varnum]/(indat.txfreq/1e6)
        return parList
    
    def ppm_to_Hz(old_parList,whichvars):
        """
        Function to convert fwhm parts of old_parList from ppm to Hz. The fwhm
        parameters are defined by whichvars. Used to convert the fwhm parts of
        parsFit back to Hz unless return_pars_in_Hz=False. No need for a 
        send_warning option since these are internal functions and values will
        already have been checked when Hz_to_ppm was called.

        Parameters
        ----------
        old_parList : list
            List of all fit parameters in the format of parsGuess with the
            parameters defined by whichvars in ppm.
        whichvars : list of ints
            Indices of any parameters in list to be converted from ppm to Hz.

        Returns
        -------
        parList : list
            List of all fit parameters in the format of parsGuess with the
            parameters defined by whichvars now converted to Hz for return.

        """
        parList=old_parList.copy()
        for varnum in whichvars:
            if type(parList[varnum]) is list:
                parList[varnum]=[pval*(indat.txfreq/1e6) for pval in parList[varnum]]
            else: # This should work for either np.ndarray or float/int scalars
                parList[varnum]=parList[varnum]*(indat.txfreq/1e6)
        return parList
    
    if indat.ndim!=1:
        raise FidAException('ERROR: For fitting, data can only have one dimension. Use op_squeeze() to remove singleton dimensions. Your data are: ({:s})'.format(', '.join(indat.dims)))

    in_range=op_freqrange(indat,ppmmin,ppmmax)
    specs=in_range.specs
    ppm=in_range.ppm

    # Set up initial parsGuess if not entered (cases with str values linked to functions only)
    default_FWHM=5 # A default guess for proton.
    if peaktype=='lorentz':
        fitfunc=op_lorentz_linbas
        minpars=3
        FWHMpars=[1]
        if parsGuess is None:
            # amp, fwhm (in Hz here but will be converted later), ppm0, baseline slope, baseline offset, phase
            if real_ydata:
                parsGuess=[np.amax(np.abs(specs)),default_FWHM,ppm[np.argmax(specs)]]+[0,0,0]
            else:
                parsGuess=[np.amax(np.abs(specs)),default_FWHM,ppm[np.argmax(specs)]]+[0j,0j,0]
    elif peaktype=='lorentz2':
        fitfunc=op_lorentz
        minpars=3
        FWHMpars=[1]
        if parsGuess is None:
            # amp, fwhm (in Hz here but will be converted later), ppm0, baseline offset, phase
            if real_ydata:
                parsGuess=[np.amax(np.abs(specs)),default_FWHM,ppm[np.argmax(specs)]]+[0,0]
            else:
                parsGuess=[np.amax(np.abs(specs)),default_FWHM,ppm[np.argmax(specs)]]+[0j,0]
    elif peaktype=='gauss':
        fitfunc=op_gauss_linbas
        minpars=3
        FWHMpars=[1]
        if parsGuess is None:
            # amp, fwhm (in Hz here but will be converted later), ppm0, baseline slope, baseline offset, phase
            if real_ydata:
                parsGuess=[np.amax(np.abs(specs)),default_FWHM,ppm[np.argmax(specs)]]+[0,0,0]
            else:
                parsGuess=[np.amax(np.abs(specs)),default_FWHM,ppm[np.argmax(specs)]]+[0j,0j,0]
    elif peaktype=='voigt':
        fitfunc=op_voigt_linbas
        minpars=4
        FWHMpars=[1,2]
        if parsGuess is None:
            # amp,fwhm_gauss,fwhm_lor,ppm0,base_slope,base_off,ph0
            if real_ydata:
                parsGuess=[np.amax(np.abs(specs)),default_FWHM,default_FWHM,ppm[np.argmax(specs)]]+[0,0,0]
            else:
                parsGuess=[np.amax(np.abs(specs)),default_FWHM,default_FWHM,ppm[np.argmax(specs)]]+[0j,0j,0]
    elif callable(peaktype):
        fitfunc=peaktype
        if parsGuess is None:
            parsGuess=[1]*peaktype.__code__.co_argcount
        FWHMpars=[]
    else:
        raise ValueError("Variable peaktype must be either 'lorentz','lorentz2','gauss', 'voigt' or callable function.")
    
    # Set the default parameter bounds if parameter_bounds=True or set to +/- inf 
    # if parameter_bounds=False. Otherwise use parameter_bounds values entered.
    if type(parameter_bounds) is bool:
        # First make a list the same length as parsGuess (ignoring if individual elements are lists themselves for now)
        # Complex parameters that need infinite bounds can be entered as np.inf and will be adjusted by nlinfit
        lb=[-np.inf]*len(parsGuess)
        ub=[np.inf]*len(parsGuess)
        # Now set some parameters to more limited range for single peak if parameter_bounds=True
        # If parameter_bounds=False then the above infinite bounds will be used
        if parameter_bounds:
            if callable(peaktype):
                warnings.warn("WARNING: Cannot construct parameter bounds for user-defined lineshape function. Enter paramter_bounds explicitly if needed. Using infinite bounds.")
            else:
                fwhm_max=0.7*(indat.txfreq/1e6)
                if peaktype=='voigt':
                    lb[:minpars]=[0,1e-4,1e-4,ppmmin]
                    ub[:minpars]=[2*np.amax(np.abs(indat.specs)),fwhm_max,fwhm_max,ppmmax]
                else:
                    lb[:minpars]=[0,1e-4,ppmmin]
                    ub[:minpars]=[2*np.amax(np.abs(indat.specs)),fwhm_max,ppmmax]
                if len(parsGuess)==minpars+3:
                    lb[-1]=-np.pi
                    ub[-1]=np.pi
        # Now check if multi-peak case (variable is iterable type) and expand the bounds for those if so
        for parnum,eachpar in enumerate(parsGuess):
            if hasattr(eachpar,'__iter__'):
                lb[parnum]=[lb[parnum]]*len(eachpar)
                ub[parnum]=[ub[parnum]]*len(eachpar)
        parameter_bounds=(lb,ub)
    
    # Convert fwhm in Hz to ppm for both parsGuess and parameter_bounds
    parsGuess=Hz_to_ppm(parsGuess, whichvars=FWHMpars,send_warning=True)
    if hasattr(parameter_bounds,'lb'): #scipy.optimize Bounds object type
        parameter_bounds.lb=Hz_to_ppm(parameter_bounds.lb, whichvars=FWHMpars,send_warning=False)
        parameter_bounds.ub=Hz_to_ppm(parameter_bounds.ub, whichvars=FWHMpars,send_warning=False)
    else: #bounds are list, tuple or numpy array of [lower,upper] bounds
        pb1=Hz_to_ppm(parameter_bounds[0], whichvars=FWHMpars,send_warning=False)
        pb2=Hz_to_ppm(parameter_bounds[1], whichvars=FWHMpars,send_warning=False)
        parameter_bounds=(pb1,pb2)

    # Finally ready to fit!
    yGuess=fitfunc(parsGuess,indat.ppm)
    # Complex data are dealt with in nlinfit wrapper, but can opt to fit just the real part
    if real_ydata:
        specs=np.real(specs)
    parsFit=nlinfit(ppm,specs,fitfunc,parsGuess,bounds=parameter_bounds)
    # Convert FWHM in ppm back to Hz
    parsFit_Hz=ppm_to_Hz(parsFit,whichvars=FWHMpars)
    yFit=fitfunc(parsFit,indat.ppm)

    # Plotting (if showplot=True or show_plot='partial')
    if type(show_plot) is bool:
        if show_plot: # True, show the full plot over the full range
            f1,ax1=plt.subplots(1,1)
            ax1.plot(indat.ppm,np.real(indat.specs),'.',label='data')
            ax1.plot(indat.ppm,np.real(yGuess),':',label='guess')
            ax1.plot(indat.ppm,np.real(yFit),'-',label='fit')
            if type(parsFit[0]) is list:
                print('Area under the fitted curve is: '+str([pv1*pv2 for pv1,pv2 in zip(parsFit[0],parsFit[1])]))
            else:
                print('Area under the fitted curve is: '+str(parsFit[0]*parsFit[1]))
    elif show_plot=='partial':
        f1,ax1=plt.subplots(1,1)
        ax1.plot(ppm,np.real(specs),'.',label='data')
        yGuess_part=freqrange(yGuess,indat.ppm,ppmmin,ppmmax)[1]
        yFit_part=freqrange(yFit,indat.ppm,ppmmin,ppmmax)[1]
        ax1.plot(ppm,np.real(yGuess_part),':',label='guess')
        ax1.plot(ppm,np.real(yFit_part),'-',label='fit')
        ax1.legend()
        if type(parsFit[0]) is list:
            print('Area under the fitted curve is: '++str([pv1*pv2 for pv1,pv2 in zip(parsFit[0],parsFit[1])]))
        else:
            print('Area under the fitted curve is: '+str(parsFit[0]*parsFit[1]))
    else:
        print("show_plot argument value not recognized. Must be True, False or 'partial'.")

    # Final spectrum and parameter return
    outdat=indat.copy()
    outdat.specs=yFit
    resids=indat-outdat
    if return_pars_in_Hz:
        return outdat,parsFit_Hz,resids
    else:
        return outdat,parsFit,resids

def op_voigt_linbas(pars,ppm):
    """
    Generates a voigt lineshape (single or multiple peaks) and (optionally) a 
    baseline and phasing for the parameters pars over the frequencies in ppm. 
    Multiple peaks can be generated by providing amplitude, FWHM and center
    frequency paramaters as lists or numpy arrays. This function can be 
    called with pyFidA.curvefit_tools.nlinfit as a wrapper for 
    scipy.optimize.curve_fit in order to fit data to a series of Lorentzian 
    peaks. See op_peakFit for an example or the notes in curvefit_tools for the 
    details of how this done.
    Note that, in Matlab, a function called  op_voigt_linbas function is 
    contained in the op_peakFit.m file and called in the main function, but the
    math in the code and parameters actually define a Lorentzian lineshape. 
    Here we use scipy's voigt_profile function with the parameters in pars and
    this is one of several options that can be called from op_peakFit by setting
    peaktype='voigt' in the input arguments

    Parameters
    ----------
    pars : list
        List of parameters to be fitted in the order 
        [amplitude, fwhm_gauss (in ppm), fwhm_lorentz (in ppm), center frequency (in ppm), baseline slope, baseline offset, phase shift (in degrees)].
        The baseline slope, baseline offset and phase shift parameters are 
        optional and are scalars. The amplitude, fwhms and center frequency are
        required and can be scalars (for a single peak) or lists/numpy arrays 
        (for multiple peaks)
    ppm : numpy array
        Frequency axis vector (in ppm). Technically the units of the ppm vector 
        just need to match those of the fwhms and center frequency.

    Returns
    -------
    y : numpy array
        1D vector of the y-values specifying the Voigt lineshape(s) 
        corresponding to the x-values in ppm.

    """
    pars=list(pars) # In case parameters are entered as 2D numpy array
    if len(pars)<4:
        raise ValueError('ERROR: pars must have length of at least 4 for amplitude, fwhm_gauss, fwhm_lorentz and center frequency.')
    if not hasattr(pars[0],'__iter__'): #type(pars[0]) is int or float for 1-peak case:
        pars=[[eachvar] for eachvar in pars[:4]]+[eachvar for eachvar in pars[4:]]
    for varct in range(4,len(pars)):
        if type(pars[varct]) is list:
            print('WARNING: Lists not allowed for baseline, offset or phase due to redundancy in fit parameters. Using first value only.')
            pars[varct]=pars[varct][0]
    # Any optional parameters not included in pars set to 0 by default.
    full_pars=pars+[0]*(7-len(pars))
    [amp,fwhm_gauss,fwhm_lor,ppm0,base_slope,base_off,ph0]=full_pars
    # Convert fwhm_lorentz to gamma (the HWHM) and convert fwhm_gauss to sigma 
    # (standard deviation) before sending to voigt_profile. Note that FWHM 
    # units are ppm, not Hz as Matlab code suggests. Throwing a warning if fwhm
    # appears to be in the wrong units.
    if fwhm_lor[0]>0.3 or fwhm_gauss[0]>0.3:
        warnings.warn('WARNING: FWHM in op_voigt_linbas should be entered in ppm. Your values of {:3.3f}, {:3.3f}  may be in Hz!'.format(fwhm_lor[0],fwhm_gauss[0]),FidAWarning)
    gamma=[fv/2 for fv in fwhm_lor]
    sigma=[fv/2/np.sqrt(2*np.log(2)) for fv in fwhm_gauss]
    y=np.zeros([len(amp),len(ppm)],dtype=complex)
    for act,aval in enumerate(amp):
        y[act,:]=voigt_profile(ppm-ppm0[act],sigma[act],gamma[act])
        y[act,:]=y[act,:]/np.amax(np.abs(y[act,:]))*aval
    bas=base_off+base_slope*ppm
    y=np.sum(y,axis=0)
    y=add_phase(y+bas,ph0)
    return y.squeeze()
