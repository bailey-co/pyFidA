#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 10:01:18 2025

@author: Colleen Bailey
Functions and wrappers that alter scipy.optimize.curve_fit to make it more 
flexible for use with pyFidA and the expected spectral data needed for fitting:

1. Re-orders complex data (as expected for NMR or MRS spectra) into a single
re-valued vector with the real component of the data followed by imaginary
component. The entered function (expected to produce complex data for 
comparison) will also be re-ordered in this way so that all data is used for
fitting by scipy.optimize.curve_fit. The code will still work if the data 
entered for comparison are real (but will throw a warning if the fitting
function produce complex data and just take the real part of that output).

2. With the make_flattening_functions function, there are a few features added:
    * Fitting functions can have parameters that are iterables to represent 
    multiple peaks. For example, op_lorentz([1,5/300,2], ppm) gives a FID with 
    a spectrum that has a peak with amplitude 1, FWHM 5 Hz (for a 300 MHz 
    scanner) at 2 ppm. But op_lorentz([[1,2], [5/300, 5/300], [2, 3.02]], ppm)
    describes a spectrum with 2 peaks (at 2 ppm and 3.02 ppm). Longer lists are
    possible depending on how many peaks are being fit for, so the number of
    parameters will vary depending on the number of peaks. However, for fitting,
    scipy.optimize.curve_fit needs a single list of values and a function that
    takes parameters in that format. This module allows for that (more details
    are available in the make_flattening_functions docstring)
    * Fitting functions can have complex parameter values (eg. baseline offset). 
    Parameters need to be real-valued for direct use with 
    scipy.optimize.curve_fit, but this module checks the initial parameter 
    guess and, if any parameters are complex values in that guess, it will fit 
    the real and imaginary parts to get a final complex value for that 
    parameter.
    * This function also deals with fitting functions that output complex data
    by wrapping the function in code that reshapes the output into a 1D real-
    valued vector that can be used with scipy.optimize.curve_fit (which 
    expects real-valued data and functions that output same)

3. With the alter_func_args function: fitting accepts lineshape functions that 
take arguments in the same format as Matlab's lineshape functions, eg. 
op_lorentz_linbas(pars,ppm), and creates a similar function with arguments
reformatted for use with scipy.optimize.curve_fit. This allows users with 
existing Matlab code to copy it over to pyFidA without major adjustments.
    
If you want to make sue of these features and have a function that accepts
arguments in the format described above (multiple peaks as iterables, complex-
valued parameters, in the order [pars, ppm]), you use:
"from curvefit_tools import nlinfit""
and then you should be able to use nlinfit similar to how you would in Matlab.

(Extra arguments related to other inputs of scipy.optimize.curve_fit will be
passed through from nlinfit (eg. the algorithm used for optimization) but they
must follow Python's structure rather than Matlab's though, and the arguments
must be entered as keyword arguments even though curve_fit allows some of them
to be entered as args. Note that the Jacobian functionality is untested.)

For a larger list of examples, see pyFidA/docs/tutorials/Peak Fitting Examples.ipynb.
eg.
ppmvec=np.linspace(11.3,-2,2048)
multi_parvec=[[1.5,2],[5/300,2/300],[2,6],0,0.1] #FWHM in ppm to generate spectrum
multi_lorentz_peak=op_lorentz_linbas(multi_parvec, ppmvec)+0.02*np.random.randn(len(ppmvec))+1j*0.02*np.random.randn(len(ppmvec))
# generates a "spectrum" with two peaks (their amplitudes, fwhm and frequency 
# positions are given by the first three lists in multi_parvec) and then the 
# baseline slope is 0 and the offset is 0.1. If you have imported nlinfit, you 
# can then fitthis function by
multi_parvec=[[1.5,2],[5,2],[2,6],0,0.1] #adjusting the FWHM values to Hz
lbs=[[0]*len(multi_parvec[0]),[0]*len(multi_parvec[1]),[np.amin(ppmvec)]*len(multi_parvec[2]),0,-1]
ubs=[[5]*len(multi_parvec[0]),[2]*len(multi_parvec[1]),[np.amax(ppmvec)]*len(multi_parvec[0]),1,1]
parsFit=nlinfit(ppmvec, multi_lorentz_peak, op_lorentz_linbas, multi_parvec,bounds=(lbs,ubs),real_nest=True)
"""
from scipy.optimize import curve_fit, Bounds
import numpy as np
import functools
import warnings
from pyFidA.fidA_common.common_functions import FidAWarning

def alter_func_args(funcnm):
    """
    Decorator that takes a function that has input arguments [parlist,xdata] 
    and generates a new function that can be used with scipy.optimize.curve_fit
    (with arguments [xdata,par1,par2,...].

    Parameters
    ----------
    funcnm : function of the form funcnm(parlist,xdata)
        Function whose arguments are to be altered.

    Returns
    -------
    wrapper : A function with the form func(xdata,*parlist)
        Decorated function where arguments are reversed and the second argument
        (a list) is unpacked before calling.
    """
    @functools.wraps(funcnm)
    def wrapper(xdata,*parlist):
        yvals=funcnm(parlist,xdata)
        return yvals
    return wrapper

def make_flattening_functions(multipeak_func,parlist_shaped,real_nest):
    """
    Decorator to generate three functions, which will flatten variable lists,
    handle complex variables, and complex function outputs, so that the 
    function and variable lists can be used with scipy.optimize.curve_fit
    (after calling alter_func_args). Details of the 3 functions can be found in
    the Returns explanation, below.

    Parameters
    ----------
    multipeak_func : function of the form multipeak_func(shaped_list,xdat)
        Function whose arguments are to be reformatted. The name multipeak_func 
        is from its initial use fitting spectra composed of multiple Lorentzian
        peaks but it can be any function where parameters are grouped into 
        lists and/or have complex values. This function may also return
        complex values.
    parlist_shaped : list
        A list of variables. Each entry in the list may be an int, float, list
        or numpy array. This is used to generate the flattening and reshaping
        functions based on the size and type of each entry.
    real_nest : boolean
        Describes whether the ydata that was passed to curvefit_tools.nlinfit 
        for comparison with the function output was complex data that has been
        altered to a 1D array of the form [np.real(ydata),np.imag(ydata)].
        If multipeak_func returns complex y-values and real_nest=True, then the
        function returned in function_wrapper will also alter the y-values into
        a 1D array of this format. In contrast, if multipeak_func returns 
        complex numbers but real_nest=False, then a FidAWarning is thrown and 
        only the real portion of the complex output will be returned for 
        comparison to the real-valued experimental ydata. (If multipeak_func 
        does not return complex data then the value of real_nest is irrelevant
        and y-values are simply returned as-is because that is already a 1D 
        real-valued array)

    Returns
    -------
    function_wrapper : function of the form function_wrapper(flatlist,xdat)
        Decorated version of the input multipeak_func where the first argument 
        is a flattened real-valued list, ie. the list does not contain any 
        entries that are themselves lists and any complex parameters have been 
        split into separate elements of flatlist for the real and imaginary 
        parts.
    flattening_wrapper : A function of the form flattening_wrapper(shaped_list1)
        Takes shaped_list1 and flattens it such that any entries that are lists
        are appended into one long list of separate ints of floats. If any of
        the elements of shaped_list are complex, they are split into separate
        elements for the real and imaginary parts. This function can be used
        to reshape the original parameter list for use with function_wrapper,
        but also to reshape the parameter bounds before passing to scipy.
        optimize.curve_fit. The output of the flattening_wrapper function,
        flat_list, is checked to ensure that its length matches the total
        number of elements in shaped_list1, accounting for complex elements.
    reshaping_wrapper : A function of the form reshaping_wrapper(flat_list1)
        Takes flat_list1 and reshapes it to match the original format of 
        parlist_shaped: grouping elements back into lists or numpy arrays, and
        reforming complex numbers from real and imaginary components. The 
        output of the reshaping_wrapper function, shaped_list, is checked to
        ensure that the total number of elements matches the number in 
        parlist_shaped from the original make_flattening functions call.
    """
    iter_vec=list()
    n_els=list()
    # Since iterables are assumed to be parameters for separate peaks, it is
    # assumed that, if one element's parameter guess is complex, every
    # parameter in that list ought to be complex. In general, peak parameters 
    # (amplitude, FWHM, center frequency) will be real, so a list of parameters
    # is not expected to be complex, let alone a mix of complex and real 
    # values. But trying to be somewhat general, here.
    complex_els=list()
    for eachit in parlist_shaped:
        if hasattr(eachit,'__iter__'):
            iter_vec.append(True)
            if np.iscomplexobj(eachit):
                n_els.append(2*len(eachit))
                complex_els.append(True)
            else:
                n_els.append(len(eachit))
                complex_els.append(False)
        else:
            iter_vec.append(False)
            if np.iscomplexobj(eachit):
                n_els.append(2)
                complex_els.append(True)
            else:
                n_els.append(1)
                complex_els.append(False)
    tot_els=sum(n_els)
        
    @functools.wraps(multipeak_func)
    def function_wrapper(flatlist,xdat):
        # function_wrapper needs to take a flatlist for to be used with
        # scipy.optimize.curve_fit, but we need to reshape this list before
        # sending it to the old function to get our yvals. Can just use the  
        # wrapper function to do reshape
        reshaped_list=reshaping_wrapper(flatlist)
        yvals=multipeak_func(reshaped_list,xdat)
        if np.iscomplexobj(yvals):
            if not real_nest:
                warnings.warn('WARNING: Function produces complex output but data to compare with appear to be real. Taking real part of function output',FidAWarning)
                yvals=np.real(yvals)
            else:
                yvals=np.concatenate([np.real(yvals),np.imag(yvals)])
        return yvals
    
    def flattening_wrapper(shaped_list1):
        if hasattr(shaped_list1,'__iter__'):
            flat_list=list()
            for varct,eachvar in enumerate(shaped_list1):
                # If it's an iterable
                if iter_vec[varct]:
                    if complex_els[varct]:
                        for eachit in eachvar:
                            # Bounds are sometimes infinite but you can't have complex np.inf so just repeat for each part
                            if np.isinf(eachit):
                                flat_list.append(eachit)
                                flat_list.append(eachit)
                            else:
                                flat_list.append(np.real(eachit))
                                flat_list.append(np.imag(eachit))
                    else: # iterable, not complex
                        for eachit in eachvar:
                            flat_list.append(eachit)
                else: # not iterable
                    if complex_els[varct]:
                        # Bounds for complex numbers
                        if np.isinf(eachvar):
                            flat_list.append(eachvar)
                            flat_list.append(eachvar)
                        else:
                            flat_list.append(np.real(eachvar))
                            flat_list.append(np.imag(eachvar))
                    else: # not iterable, not complex
                        flat_list.append(eachvar)
            # If parlist_shaped is longer than flat_list, raise an error (if
            # parlist_shaped is shorter than flat_list, you'll get an 
            # IndexError above)
            if len(flat_list)!=tot_els:
                raise ValueError('ERROR: Number of elements in list does not match what is expected. Please check that your input list matches the shape of '+str(parlist_shaped))
            return flat_list
        else:
            print('WARNING: item sent to flattener is not a list: '+str(shaped_list1))
            return shaped_list1
    
    def reshaping_wrapper(flat_list1):
        shaped_list=list()
        # Find the points to divide up flat_list1 into groups
        idxpts=[0]+list(np.cumsum(n_els))
        if len(flat_list1)!=tot_els:
            raise ValueError('ERROR: Number of elements in list is wrong. Input should have length '+str(tot_els))
        for act,(start_idx,end_idx) in enumerate(zip(idxpts[:-1],idxpts[1:])):
            if iter_vec[act]:
                # Preserves iterable type (ndarray vs list)
                if isinstance(parlist_shaped[act],np.ndarray):
                    if complex_els[act]:
                        shaped_list.append(np.array(flat_list1[start_idx:end_idx:2])+1j*np.array(flat_list1[start_idx+1:end_idx:2]))
                    else:
                        shaped_list.append(np.array(flat_list1[start_idx:end_idx]))
                else: # assumes list type for other iterables
                    if complex_els[act]:
                        shaped_list.append([flat_list1[curr_idx]+1j*flat_list1[curr_idx+1] for curr_idx in range(start_idx,end_idx,2)])
                    else:
                        shaped_list.append(list(flat_list1[start_idx:end_idx]))
            else: # floats and ints
                if complex_els[act]:
                    shaped_list.append(flat_list1[start_idx]+1j*flat_list1[start_idx+1])
                else:
                    shaped_list.append(flat_list1[start_idx])
        return shaped_list
    return function_wrapper,flattening_wrapper,reshaping_wrapper

def nlinfit(xdata,ydata,funcnm,parlist_of_lists,**kwargs):
    """
    Fitting function that mimics Matlab's nlinfit and adds some new
    functionality to handle complex ydata, complex parameter values and 
    functions where some parameters are lists or numpy arrays (eg. fitting
    multiple peaks with pyFidA.op_lorentz). Example usage can be seen in 
    pyFidA.op_peakFit.

    Parameters
    ----------
    xdata : array_like
        The independent variable for the fitting function.
    ydata : array_like
        The dependent data that will be compared to funcnm(pars,xdata).
    funcnm : function of the form funcnm(parlist_of_lists,xdata)
        The first argument must be a list of parameters to fit (entries within 
        pars may be grouped into lists for convenience and can be complex). The 
        second argument must be the independent variable.
    parlist_of_lists : list
        Initial guess for the fit parameters. Entries within the list may
        be grouped into lists/arrays of varying size if that is how funcnm
        expects them, or entries may be scalars. The entries will be flattened 
        before being passed to scipy.optimize.curve_fit. Note that 
        parlist_of_lists must itself be a list, rather than a numpy array 
        (if parlist_of_lists is a numpy array, it cannot correctly handle
        complex numbers or elements that are arrays/lists).
    **kwargs : 
        Keyword arguments passed to scipy.optimize.curve_fit. Note that these 
        include the parameters listed after p0 in curve_fit's documentation as
        well as the **kwargs passed to leastsq. However, there are a couple of
        changes worth noting.
        1. sigma can be None, scalar or array-like of the same shape as 
        parlist_of_lists. In the last case, it will be flattened the same way
        as parlist_of_lists.
        2. bounds can be left out, or a 2-tuple of array-like elements with the 
        same shape as parlist_of_lists or an instance of Bounds class. If
        entered, the lower bounds and upper bounds will be flattened in the 
        same way as parlist_of_lists. The Bounds case isn't well-tested and the 
        keep_feasible property may fail.
        3. jac is untested. It seems like you should be able to wrap it in the
        same way as funcnm to deal with the flattening and argument swaps, so 
        that it responds correctly to the xdata and list passed into nlinfit. 
        So this is what I have done but I don't generally use the Jacobian and
        so have not tested it.

    Returns
    -------
    reformatted_parsFit : list
        Optimal values for the parameters of funcnm so that the sum of the 
        squared residuals of funcnm(pars,xdata) - ydata is minimized. The list 
        is formatted to match parlist_of_lists.
        Note that scipy.optimize.curve_fit returns pcov (and later versions of 
        scipy give the option to return additional arguments, but these are not 
        passed back as output arguments from nlinfit at the moment because the 
        co-variance matrix doesn't seem to be called or used for Matlab fid-A 
        calls).

    """
    if type(parlist_of_lists) is not list:
        raise TypeError('ERROR: parameters must be entered as type list (but can contain elements that are arrays, lists or scalars)')
    # Note that np.iscomplex will not return True for any elements that are
    # just 0j. However, np.iscomplexobj will return True even if 0j is the only
    # complex number in the array
    if np.iscomplexobj(ydata):
        real_nest=True
        ydata=np.concatenate([np.real(ydata),np.imag(ydata)])
    else:
        real_nest=False
    # Create the function for use with scipy.optimize.curve_fit, as well as the
    # functions that can reshape the parameter list for use with it.
    flattened_func,flatten_vars,shape_vars=make_flattening_functions(funcnm,parlist_of_lists,real_nest)
    func_reformatted_args=alter_func_args(flattened_func)
    # Apply the variable-flattening function to parameter bounds
    if 'bounds' in kwargs.keys():
        # Covers the case where bounds are entered as an object of class Bounds rather than as an iterable. Largely untested
        if isinstance(kwargs['bounds'],Bounds):
            kwargs['bounds'].lb=flatten_vars(kwargs['bounds'].lb)
            kwargs['bounds'].ub=flatten_vars(kwargs['bounds'].ub)
            try:
                kwargs['bounds']=flatten_vars(kwargs['bounds'].keep_feasible)
            except:
                print('WARNING: Cannot reshape keep_feasible. Setting to false')
                kwargs['bounds'].keep_feasible=False
        # bounds as arrays. Can't write to existing tuple so have to make a new list and convert
        else:
            newbounds=list()
            newbounds.append(flatten_vars(kwargs['bounds'][0]))
            newbounds.append(flatten_vars(kwargs['bounds'][1]))
            kwargs['bounds']=tuple(newbounds)
    if 'sigma' in kwargs.keys():
        kwargs['sigma']=flatten_vars(kwargs['sigma'])
        # Note that there is a case where sigma can be MxM covariance matrix that I haven't dealt with.
        # Not sure how to check for it.
    if 'jac' in kwargs.keys():
        warnings.warn('WARNING: Jacobian not tested for wrapping functions in curvefit_tools. Set to None if you get an error.',FidAWarning)
        flattened_jac,*_=make_flattening_functions(kwargs['jac'],parlist_of_lists,real_nest)
        jac_reformatted_args=alter_func_args(flattened_jac)
        kwargs['jac']=jac_reformatted_args
    parsFit, pcov=curve_fit(func_reformatted_args, xdata, ydata, flatten_vars(parlist_of_lists), **kwargs)
    reformatted_parsFit=shape_vars(parsFit)
    return reformatted_parsFit
