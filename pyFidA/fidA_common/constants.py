#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:48:14 2026
pyFidA.fidA_common.constants.py

@author: Colleen Bailey (cbailey@sri.utoronto.ca)

Functions that are used across multiple sub-modules in the pyFidA module:
    GAMMA_DICT - a dictionary of gyromagnetic ratio values for different 
    nuclei in MHz/T. Used to calculate resonance frequencies for FID objects,
    RF pulses, etc.
"""

GAMMA_DICT={'1H':42.577,'2H':6.536,'13C':10.7084,'19F':40.078,'23Na':11.262,'31P':17.235}


