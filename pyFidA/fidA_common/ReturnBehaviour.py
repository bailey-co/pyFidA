#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyFidA.fidA_common.ReturnBehaviour.py

Code to assist in determining how arguments are returned for selected 
functions. For more explanation, see the 
pyFidA.fidA_processing.alter_return_args.py module. Currently, this class is 
only used to change how return arguments behave in the fidA_processing module, 
but it could be imported and used in other modules, although each module 
should likely use a separate instance to avoid conflicts.

Created on Thu Jun  4 10:15:53 2026
@author: Colleen Bailey (cbailey@sri.utoronto.ca)
"""

class ReturnBehaviour(object):
    """
    A class to hold the default behaviour of return arguments, used in the
    pyFidA.fidA_processing toolbox. If ReturnBehaviour._return_args=True then 
    all arguments from a function are returned by default. If 
    ReturnBehaviour._return_args=False, then only the first argument (usually 
    the FID object that the function created) is returned for functions with
    the @alter_return_args decorator. For more information, see the 
    documentation in pyFidA.fidA_processing.alter_return_args.py.
    """
    def __init__(self,return_flag=True):
        self._return_args=return_flag
    def __call__(self):
        return self._return_args