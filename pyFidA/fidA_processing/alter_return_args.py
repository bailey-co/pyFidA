#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup that defines how return arguments are dealt with in pyFidA.

Python is unable to vary its outputs based on the function call in the way that
Matlab is. This module contains code to help users change how output arguments
are returned, including:
    1. Setting _use_default, which determines whether all arguments are returned
    from a function (_use_default=True) or only the first argument 
    (_use_default=False) upon importing the pyFidA module.
    2. The alter_return_args decorator, which can be applied to any function as
    long as an input argument (named return_extra_args, by convention) is added 
    as the last input arguments of that function. This variable can be used to
    override other indicators of how to deal with the return arguments for a
    single function call.
    eg. op_autophase(myfid, return_extra_args=False) will
    return just a FID object (and not the zero-th order phase), regardless of
    the value of _use_default or whether allow_chaining() has been called 
    beforehand.
    3. The allow_chaining() and stop_chaining() functions that change the 
    output argument return behaviour for blocks of code.

Currently, this code is only used in the fidA_processing toolbox on selected
functions, but similar code could be implemented for other modules. (It may
also be possible to import the functions below into other modules, but it
depends on the order that the modules are loaded in the __init__ file. This
could result in circular imports in some cases.)

For more details and examples, see the "Behaviour of functions that return 
multiple output arguments" section in docs/Matlab_differences_basic.md

eg.
import pyFidA
phased_fid, ph0 = pyFidA.op_autophase(myfid)
pyFidA.allow_chaining()
pyFidA.op_plotspec(pyFidA.op_autophase(myfid))

Created on Fri Jun  5 14:37:41 2026
@author: Colleen Bailey
"""

import functools
from pyFidA.fidA_common import ReturnBehaviour

# The default behaviour on import is set by this line, which assigns the 
# initial value of use_default._return_args. Users can change this line if they 
# want a different default behaviour after importing pyFidA.
# For a default that returns all output arguments: 
#    _use_default=ReturnBehaviour(True)
# For a default that returns only the first output argument: 
#    _use_default=ReturnBehaviour(False)
_use_default=ReturnBehaviour(True)

def allow_chaining():
    """
    Changes the behaviour of selected functions in the fidA_processing module
    so that only the first output argument (usually the processed FID object)
    is returned. The behaviour persist for all subsequent function calls until
    stop_chaining() is called or the pyFidA module is reloaded. This means 
    that functions can be easily chained together like:
    pyFidA.allow_chaining()
    pyFidA.op_plotspec(pyFidA.op_autophase(pyFidA.op_averaging(my_fid)))
    """
    _use_default._return_args=False
    
def stop_chaining():
    """
    Changes the behaviour of selected functions in the fidA_processing module
    to return all output arguments. The behaviour persist for subsequent 
    function calls, until allow_chaining() is called or the pyFidA module is
    reloaded. This means that functions cannot be chained together:
    pyFidA.stop_chaining()
    phased_fid, ph0 = pyFidA.op_autophase(myfid)
    pyFidA.op_plotspec(phased_fid)
    """
    _use_default._return_args=True
    
def alter_return_args(funcnm):
    """
    Decorator applied to functions to change how output arguments are returned.
    The function being decorated (funcnm) should have a final input argument
    that is used to define the return behaviour. This final input argument is
    named return_extra_args by convention but can technically have any name.
    The only requirements are that it is the final argument and that it has a 
    default value of None (this is what lets the allow_chaining() and 
    stop_chaining() methods work, as well as applying the default behaviour 
    when pyFidA is loaded). Setting the value of this final argument to True 
    or False in any individual function call will override everything else to 
    return all output arguments or return only the first output argument, 
    respectively.

    Parameters
    ----------
    funcnm : function
        Function to be decorated.

    Returns
    -------
    wrapper : function
        Decorated version of funcnm, affected by allow_chaining, stop_chaining,
        and the value of funcnm's final input argument.
    """
    # save the number of input arguments in the function (and their names in 
    # the case of keyword arguments) for later comparisons.
    n_args=funcnm.__code__.co_argcount
    varnames=funcnm.__code__.co_varnames[:n_args]
    @functools.wraps(funcnm)
    def wrapper(*args,**kwargs):
        # if all possible input arguments are given by the user with no keyword arguments, the last argument is the return argument flag
        if (len(args)==n_args and args[-1] is not None): 
            return_flag=args[-1]
        # if return argument flag is entered as keyword argument
        elif (varnames[-1] in kwargs.keys() and kwargs[varnames[-1]] is not None):
            return_flag=kwargs[varnames[-1]]
        # No value is given for the return argument flag, either as argument or keyword argument
        else:
            return_flag=_use_default()
            kwargs[varnames[-1]]=return_flag
        # Call the function and get all return arguments. Assign them to 
        # a single variable, outargs
        outargs=funcnm(*args,**kwargs)
        # Then decide how many arguments to return based on the return_flag value found above
        if return_flag:
            return outargs
        else:
            try:
                return outargs[0]
            except IndexError:
                print('WARNING: The function you called has been decorated by alter_return_args decorator. Multiple output arguments were expected for this function but only one was returned. This could indicate a problem with the decorator.')
                return outargs
    return wrapper    
