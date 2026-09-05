#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:46:32 2026

@author: nearlabmacbook1
"""

# Note that this ordering needs to be preserved with GAMMA_DICT imported first
# since it is used in later imports. eg. BlochSimulator import from 
# pyFidA.fidA_common without referencing the constants toolbox
# phase is imported here but won't be re-exported at the top level of pyFidA
# because of potential namespace clobbering issues
from pyFidA.fidA_common.constants import GAMMA_DICT
from pyFidA.fidA_common.common_functions import (
    fid_from_specs,
    spec_from_fids,
    phase,
    FidAWarning,
    FidAException,
    )

from pyFidA.fidA_common.BlochSimulator import BlochSimulator
from pyFidA.fidA_common.RF_pulse import RF_pulse, FidAWarningRF
# Not importing _calc_tw1max, _calc_tbw and estimate_f0. These are used in some
# other modules but will be imported from pyFidA.fidA_common.RF_pulse to keep 
# them more private and to be clear that they're associated with RF_pulse
from pyFidA.fidA_common.FID import FID
from pyFidA.fidA_common.Hamiltonian import Hamiltonian
from pyFidA.fidA_common.ReturnBehaviour import ReturnBehaviour


__all__ = [
    'BlochSimulator',
    'FID',
    'GAMMA_DICT',
    'Hamiltonian',
    'ReturnBehaviour',
    'RF_pulse',
    'fid_from_specs',
    'spec_from_fids',
    'phase',
    'FidAWarning',
    'FidAWarningRF',
    'FidAException'
    ]