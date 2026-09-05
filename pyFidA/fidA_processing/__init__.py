#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:28:30 2026
pyFidA.fidA_processing.__init__.py

@author: Colleen Bailey (@cbailey@sri.utoronto.ca)

Note that the order of import matters because some modules reference others.
alter_return_args is needed for any module where the decorator is used.
op_common_processing contains the functions most often used in other modules.
"""

from pyFidA.fidA_processing.alter_return_args import (
    allow_chaining,
    stop_chaining,
    )

from pyFidA.fidA_processing.op_common_processing import (
    add_phase,
    add_phase1,
    op_addphase,
    op_ampScale,
    op_addScans,
    op_subtractScans,
    op_freqrange,
    freqrange,
    op_zeropad,
    )

from pyFidA.fidA_processing.op_concatenation import (
    op_concatAverages,
    op_concatFreq,
    op_concatSubspecs,
    )

from pyFidA.fidA_processing.op_splitspecs import (
    op_takeaverages,
    op_takecoils,
    op_takeextras,
    op_takesubspec,
    )

from pyFidA.fidA_processing.op_main_processing import (
    op_addphaseSubspec,
    op_addrcvrs,
    op_alignAllScans,
    op_alignAllScans_fd,
    op_alignAverages,
    op_alignAverages_fd,
    op_alignISIS,
    op_alignMPSubspecs,
    op_alignMPSubspecs_fd,
    op_alignScans,
    op_alignScans_fd,
    op_alignrcvrs,
    op_arsos,
    op_autophase,
    op_averaging,
    op_blockAvg,
    op_combineRcvrs,
    op_combinesubspecs,
    op_complexConj,
    op_dccorr,
    op_downsamp,
    op_ecc,
    op_ecc_klose,
    op_fddccorr,
    op_filter,
    op_fourStepCombine,
    op_freqAlignAverages,
    op_freqAlignAverages_fd,
    op_freqshift,
    op_freqshiftSubspec,
    op_getcoilcombos,
    op_getcoilcombos_specReg,
    op_HSVDfit,
    op_leftshift,
    op_matchLW,
    op_median,
    op_movef0,
    op_phaseAlignAverages,
    op_phaseAlignAverages_fd,
    op_ppmref,
    op_removeWater,
    op_rmNworstaverages,
    op_rmbadaverages,
    op_rmworstaverage,
    op_squeeze,
    op_timerange,
    op_unfilter,
    op_zerotrim,
    )

from pyFidA.fidA_processing.op_peak_simulation import (
    op_addNoise,
    op_gaussianPeak,
    op_lorentzianPeak,
    op_makeECArtifact,
    op_makePhaseDrift,
    op_makeFreqDrift,
    )

from pyFidA.fidA_processing.op_peak_fitting import (
    op_creFit,
    op_gauss_linbas,
    op_gauss,
    op_integrate,
    op_lorentz_linbas,
    op_lorentz,
    op_peakFit,
    op_voigt_linbas,
    )

from pyFidA.fidA_processing.op_plotting import (
    op_plotfid,
    op_plotspec,
    )

from pyFidA.fidA_processing.op_metrics import (
    op_getLW,
    op_getPeakHeight,
    op_getSNR,
    op_relyTest,
    )

