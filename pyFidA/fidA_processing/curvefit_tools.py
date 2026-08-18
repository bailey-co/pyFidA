#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 10:01:18 2025

@author: Colleen Bailey
Functions and wrappers that alter scipy.optimize.curve_fit to behave more like
Matlab's nlinfit. This is a result of trying to make peak-generating functions
like op_lorentz_linbas(pars,ppm) match the setup in Matlab's fid-A (same order
and type of arguments), even though the main non-linear curve-fitting algorithm
in Python, scipy.optimize.curve_fit, expects the function arguments in a different
order and format, namely func(xdata,par1,par2,...). There has also been 
functionality added so that multiple peaks with the same lineshape can be entered
as lists/arrays within the main parameter list (eg. a list of amplitudes that
has npeak elements in it, followed by a list of FWHM with npeak elements, etc.) 
This module also allows complex parameters to be input (eg. for baseline offset)
These changes are accomplished by decorating the original function to (1) 
flatten the input variable into a single list of floats/ints, (2) split any
complex parameters into real and imaginary parts, (3) reverse the order of the 
xdata and pars arguments so they are as curve_fit expects and (3) unpacking the 
list of parameters so that they are accepted as separate input arguments.
Additional functions for flattening and reshaping arrays of the same size as
the parameters are also generated so that (1) sigma and bounds can also be 
entered in the same format as the original function's variable and then flattened
and passed on to curve_fit, and (b) the parameters that are returned are 
reshaped back to the original parameter list size. Some attempt to get things
into the right format is also made, although this has only been tested on lists
and numpy arrays.
If you want to fit a spectrum with some lineshape function that you have that
is like the existing Matlab fid-A lineshape functions setups, you can start with
"from curvefit_tools import nlinfit""
and then you should be able to use nlinfit similar to how you would in Matlab.
(Extra arguments related to other parts of curve_fit and the least squares
algorithm follow Python's structure rather than Matlab's though, and the 
arguments MUST be entered as keyword arguments even though curve_fit allows some
of them to be entered as args. Also the Jacobian is not tested at all.)

For a larger list of examples, a tutorial will be developed.
eg.
ppmvec=np.linspace(11.3,-2,2048)
multi_parvec=[[1.5,2],[5/300,2/300],[2,6],0,0.1] #FWHM in ppm to generate spectrum
multi_lorentz_peak=op_lorentz_linbas(multi_parvec, ppmvec)+0.02*np.random.randn(len(ppmvec))+1j*0.02*np.random.randn(len(ppmvec))
generates a "spectrum" with two peaks (their amplitudes, fwhm and frequency 
positions are given by the first three lists in multi_parvec) and then the baseline
slope is 0 and the offset is 0.1. If you have imported nlinfit, you can then fit
this function by
multi_parvec=[[1.5,2],[5,2],[2,6],0,0.1] #adjusting the FWHM values to Hz
lbs=[[0]*len(multi_parvec[0]),[0]*len(multi_parvec[1]),[np.amin(ppmvec)]*len(multi_parvec[2]),0,-1]
ubs=[[5]*len(multi_parvec[0]),[2]*len(multi_parvec[1]),[np.amax(ppmvec)]*len(multi_parvec[0]),1,1]
parsFit=nlinfit(ppmvec, multi_lorentz_peak, op_lorentz_linbas, multi_parvec,bounds=(lbs,ubs),real_nest=True)
where bounds is optional.
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
    (with arguments [xdata,par1,par2,...]

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
    Decorator to generate three functions, as described in Returns section, below

    Parameters
    ----------
    multipeak_func : function of the form multipeak_func(shaped_list,xdat)
        Function whose arguments are to be altered. The name multipeak_func is
        from its initial use fitting spectra composed of multiple Lorentzian
        peaks but it can be any function where variables are grouped into lists
        (and even function where there is no list, in which case flattening
        will just return the original list). (Note that shaped_list here
        is the same as parlist_shaped above, but I gave them different names to
        be clear that it is not the "saved" parlist_shaped from the decorator
        that is being sent through but rather something sent into whatever is
        returned from function_wrapper)
    parlist_shaped : list
        A list of variables. Each entry in the list may be an int, float, list
        or numpy array. This is used to generate the flattening and reshaping
        functions based on the size and type of each entry.
    real_nest : boolean
        Describes whether the ydata that was passed to nlinfit for comparison
        with the function output was complex data that has been altered to
        a 1D array of the form [np.real(ydata),np.imag(ydata)]. If 
        multipeak_func returns complex y-values and real_nest is True, then the
        function returned in function_wrapper will also alter the y-values into
        a 1D array of this format. In contrast, multipeak_func returns complex
        numbers but real_nest is False, then a FidAWarning is thrown and only 
        the real portion of the complex output will be returned for comparison 
        to the real-valued experimental ydata. (If multipeak_func does not 
        return complex data then y-values are simply returned as-is because
        that is already a 1D real-valued array)

    Returns
    -------
    function_wrapper : function of the form func(flatlist,xdat)
        Decorated function where the first argument is flattened list, ie. the
        list does not contain any entries that are themselves lists and any 
        complex parameters have been split into separate elements of flatlist
        for the real and imaginary parts.
    flattening_wrapper : A function of the form func(shaped_list1)
        Takes shaped_list1 and flattens it such that any entries that are lists
        are appended into one long list of separate ints of floats. If any of
        the elements of shaped_list are complex, The 
        flat_list is returned after a basic check that its length matches that 
        for the flattened parlist_shaped sent in during the initial decorator 
        call.
    reshaping_wrapper : A function of the form func(flat_list1)
        Takes flat_list1 and reshapes it to match the original shape of 
        parlist_shaped. The function returns the shaped list. A basic check is 
        run to check that the total number of elements in flat_list1 matches 
        the total number of elements in the parlist_shaped variable sent in 
        during the initial decorator call. For any iterables that are created 
        in shaped_list, an effort is made to match the iterable type in the 
        original list (only list and numpy arrays are implemented).
    """
    iter_vec=list()
    #type_vec=list()
    n_els=list()
    # Since iterables are assumed to be parameters for separate peaks, it is
    # assumed that, if the one element's parameter guess is complex, every 
    # parameter in that list outght to be complex. In general, peak parameters 
    # (amplitude, FWHM, center frequency) will be real, so a list of parameters
    # within the full list should not be complex at all, let alone a mix of
    # complex and real values. The parameters that I expect to be complex are 
    # all scalars. But trying to be somewhat general, here.
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
        #type_vec.append(eachit.__class__.__name__)
    tot_els=sum(n_els)
        
    @functools.wraps(multipeak_func)
    def function_wrapper(flatlist,xdat):
        # Need to reshape the flatlist before sending into the old function. Can
        # just use the wrapper function for this
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
        # adding some extra cases to deal with bounds, etc that are scalars
        if hasattr(shaped_list1,'__iter__'):
            flat_list=list()

            for varct,eachvar in enumerate(shaped_list1):
                if iter_vec[varct]:
                    if complex_els[varct]:
                        for eachit in eachvar:
                            if np.isinf(eachit):
                                flat_list.append(eachit)
                                flat_list.append(eachit)
                            else:
                                flat_list.append(np.real(eachit))
                                flat_list.append(np.imag(eachit))
                    else:
                        for eachit in eachvar:
                            flat_list.append(eachit)
                else:
                    if complex_els[varct]:
                        if np.isinf(eachvar):
                            flat_list.append(eachvar)
                            flat_list.append(eachvar)
                        else:
                            flat_list.append(np.real(eachvar))
                            flat_list.append(np.imag(eachvar))
                    else:
                        flat_list.append(eachvar)
            # If parlist_shaped doesn't match the size of whatever is entered
            # for shaped_list1, there are two possibilities. If parlist_shaped 
            # is shorter than shaped_list1, you'll get an IndexError from iter_vec[varct]
            # in the loop above. If parlist_shaped is longer than shaped_list1,
            # this check on the length will stop that. (There are weird cases
            # where you might have a list of lists with length 2 and 3 vs a list
            # with lengths 3 and 2 and no error is thrown but, at that point, I
            # don't really know what people would be entering as bounds or sigmas)
            if len(flat_list)!=tot_els:
                raise ValueError('ERROR: Number of elements in list does not match what is expected. Please check that your input list matches the shape of '+str(parlist_shaped))
            return flat_list
        else:
            print('WARNING: item sent to flattener is not a list: '+str(shaped_list1))
            return shaped_list1
    
    def reshaping_wrapper(flat_list1):
        shaped_list=list()
        idxpts=[0]+list(np.cumsum(n_els))
        if len(flat_list1)!=tot_els:
            raise ValueError('ERROR: Number of elements in list is wrong. Input should have length '+str(tot_els))
        # Preserves iterable type (ndarray vs list - are there other iterable
        # types to consider?). Doesn't do any thing with item types (eg. int vs
        # float). Which I think should generally be fine?
        for act,(start_idx,end_idx) in enumerate(zip(idxpts[:-1],idxpts[1:])):
            if iter_vec[act]:
                if isinstance(parlist_shaped,np.ndarray):#type_vec[act]=='ndarray':
                    if complex_els[act]:
                        shaped_list.append(np.array(flat_list1[start_idx:end_idx:2]+1j*flat_list1[start_idx+1:end_idx:2]))
                    else:
                        shaped_list.append(np.array(flat_list1[start_idx:end_idx]))
                else: # assumes list type for other iterables
                    if complex_els[act]:
                        shaped_list.append(list(flat_list1[start_idx:end_idx:2]+1j*flat_list1[start_idx+1:end_idx:2]))
                    else:
                        shaped_list.append(list(flat_list1[start_idx:end_idx]))
            else: # floats and ints
                if complex_els[act]:
                    shaped_list.append(flat_list1[start_idx]+1j*flat_list1[start_idx+1])
                else:
                    shaped_list.append(flat_list1[start_idx])
        if isinstance(parlist_shaped,np.ndarray):
            shaped_list=np.array(shaped_list)
        return shaped_list
    return function_wrapper,flattening_wrapper,reshaping_wrapper

def nlinfit(xdata,ydata,funcnm,parlist_of_lists,**kwargs):
    """
    reformatted_parsFit=nlinfit(xdata,ydata,funcnm,parlist_of_lists,**kwargs)
    Mimics Matlab's nlinfit by expecting the function for fitting to have the
    form funcnm(parlist,xdata). It creates a new function (see curvefit_tools
    module help text above for more explanation) func_reformatted_args(xdata,par1,par2,...) 
    where par1, par2,... are the individual elements of parlist. This allows
    scipy.optimize.curve_fit to be called for the new function.
    There is also an additional wrapper that allows parlist to have entries that
    are themselves lists (eg. [list_of_amplitudes, list_of_fwhms] for fitting
    multiple peaks) and flattens these entries, as well as any sigma or bounds
    with the same list of list format, before sending them to curve_fit, which
    requires any lists to be flat.
    Several Matlab fid-A peak functions (eg. op_lorentz_linbas) are set up to 
    take 2D arrays, although it's not clear that nlinfit actually would work in 
    these cases and they don't allow for combining lists of varying lengths or
    lists with scalars. So there is some functionality here that may not 
    translate to Matlab.

    Parameters
    ----------
    xdata : array_like
        The independent variable for the fitting function.
    ydata : array_like
        The dependent data that will be compared to funcnm(pars,xdata).
    funcnm : function
        The model function funcnm(pars,xdata). The first argument must be a list
        of parameters to fit (entries within pars may be grouped into lists for 
        convenience). The second argument must be the independent variable.
    parlist_of_lists : array-like
        Initial guess for the fit parameters. Entries within the array/list can
        be grouped into lists of varying size for convenience, or may be 
        scalars or combinations of these. The entries will be flattened before
        being passed to curve_fit.
    **kwargs : 
        Keyword arguments passed to scipy.optimize.curve_fit. (Note that these 
        include the parameters listed after p0 in curve_fit's documentation as
        well as the **kwargs passed to leastsq. However, there are a couple of
        changes worth noting.
        1. sigma can be None, scalar or array-like of the same shape as 
        parlist_of_lists. In the last case, it will be flattened the same way
        as parlist_of_lists.
        2. bounds can be left out, a 2-tuple of array-like of the same shape as
        parlist_of_lists or an instance of Bounds class. In the latter two cases,
        the lower bounds and upper bounds will be flattened to match the flattened
        way as parlist_of_lists. The Bounds case isn't well-tested and the 
        keep_feasible property may fail.
        3. jac is untested. It seems like you should be able to wrap it in the
        same way to deal with the flattening and argument swaps, so that it
        responds correctly to the xdata and list passed into nlinfit. So this
        is what I have done but I don't generally use the Jacobian to test it.
        I have added a warning so that users know to use with caution.

    Returns
    -------
    reformatted_parsFit : array-like
        Optimal values for the parameters so that the sum of the squared 
        residuals of funcnm(pars,xdata) - ydata is minimized. The list is shaped
        to match the shape of parlist_of_lists and to match the types of any
        iterable entries (list of numpy array).
        Note that curve_fit returns pcov (and later versions of scipy than I
        have give the option to return additional arguments, but these are not 
        passed back as output arguments in nlinfit at the moment because the 
        co-variance matrix doesn't seem to be called or used for Matlab fid-A 
        calls.

    """
    # Note that np.iscomplex will not return True for any elements that are just
    # 0j. However, np.iscomplexobj will return True even if 0j is the only complex
    # number in the array
    if np.iscomplexobj(ydata):
        real_nest=True
        ydata=np.concatenate([np.real(ydata),np.imag(ydata)])
    else:
        real_nest=False
    flattened_func,flatten_vars,shape_vars=make_flattening_functions(funcnm,parlist_of_lists,real_nest)
    # For each parameter in parlist of lists, if it's scalar AND complex-valued,
    # split it into two parameters and make a new function that accepts the new
    # parameter list
    
    func_reformatted_args=alter_func_args(flattened_func)
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
        # bounds as arrays. Ugh. Can't write to tuple so have to make a new bounds variable and then assign
        else:
            newbounds=list()
            newbounds.append(flatten_vars(kwargs['bounds'][0]))
            newbounds.append(flatten_vars(kwargs['bounds'][1]))
            kwargs['bounds']=newbounds
    if 'sigma' in kwargs.keys():
        kwargs['sigma']=flatten_vars(kwargs['sigma'])
        # Note that there is a case where sigma can be MxM covariance matrix that I haven't dealt with.
        # Not sure how to check for it.
    if 'jac' in kwargs.keys():
        print('WARNING: Jacobian not tested for wrapping functions in curvefit_tools. Set to None if you get an error.')
        flattened_jac,*_=make_flattening_functions(kwargs['jac'],parlist_of_lists)
        jac_reformatted_args=alter_func_args(flattened_jac,real_nest=real_nest)
        kwargs['jac']=jac_reformatted_args
    parsFit, pcov=curve_fit(func_reformatted_args, xdata, ydata, flatten_vars(parlist_of_lists),**kwargs)
    #parsFit, pcov=curve_fit(func_reformatted_args, xdata, ydata, flatten_vars(parlist_of_lists))
    reformatted_parsFit=shape_vars(parsFit)
    return reformatted_parsFit
