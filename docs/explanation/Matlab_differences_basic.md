# Differences between pyFidA and Matlab fid-A for basic users
This document explains the main differences between pyFidA and Matlab fid-A that are relevant for most users who will be using the functions to import data, process data and simulate spectra. If you are planning to extend pyFidA or need to understand more about the underlying code, you should also read about [the differences between pyFidA and Matlab fid-A for programmers](Matlab_differences_programmers.md)

Where possible, I have kept the function names, inputs and outputs aligned with Matlab. This should allow users who have analysis code in Matlab to easily switch over. The current version was created based on the Matlab fid-A that existed on Github on May 21, 2025, without MRSI functionality.
* [Basics](#basics)
* [Processing Module](#processing)
* [Simulation Module](#simulation)
* [RF Pulses](#rf)

<a name="basics"></a>
## Basics

### Calling functions
By importing the module with a \* import, the functions become available directly in the namespace and can be called in the same way that they would in Matlab.
```python
from pyFidA import *
fname='mydata/ge_directory/P12345.7'
[metfid,reffid]=io_loadspec_GE(fname)
avg_met=op_averaging(metfid)
```
In general, star imports of this form are not considered good form in Python because it is harder to know where functions come from and it will overwrite variables or functions from other modules with the same names. However, the pyFidA names that are imported are relatively unique (eg. the "phase" function that is used for several underlying calculations is not imported at the top level by the \* import). Nevertheless, if you are not looking to transfer over existing Matlab code, other import formats are preferable. eg.
```python
import pyFidA
fname='mydata/ge_directory/P12345.7'
metfid,reffid=pyFidA.io_loadspec_GE(fname)
avg_met=pyFidA.op_averaging(metfid)
```
However, for the remainder of this section, I will show examples that would follow after *from pyFidA import \**.

### Avoiding "in" as a variable name
Some Matlab functions use "in" as the name of an input argument and it is possible that some users copied this and used "in" as a variable name in their own code. In Python, "in" is a protected word and attempting to assign something to it will throw an error. If you see a SyntaxError or other error on a line with a variable named "in", change the variable name.

### Behaviour of functions that return multiple output arguments
The largest difference is in how Python and Matlab handle functions with multiple return arguments . In Matlab, it is possible for a function to return a different number of output arguments based on how it is called:
```matlab
% In Matlab, this returns a phased fid structure and a float value for the zero-th order phase that was calculated
[phased_fid,ph0]=op_autophase(avg_met)

% This returns just the phased fid structure. The zero-th order phase is calculated within the function but
% Matlab "knows" not to return it because there is only one variable that is assigned to the result of the 
% function call.
phased_fid=op_autophase(avg_met)

% This is particularly convenient for "chaining" multiple processing steps together into one line
op_plotspec(op_averaging(op_addrcvrs(metfid)))
```

In Python, this is not possible. A brief explanation is that, when a Python function has three output arguments, but the user calls it and assigns to just one variable name, Python will "collapse" these outputs. That one variable name will then be for a tuple that contains three elements, one for each return argument. As a result, you may be sending your data through several processing steps and you'll get an error message. However, depending on the functions used, an error may not be apparent until later steps:
```python
# In Python, this returns a phased fid structure and a float value 
# for the zero-th order phase that was calculated
[phased_fid,ph0]=op_autophase(avg_met)

# This returns a tuple with two elements: phased_fid=(fid_structure,ph0). No
# error is thrown at this stage.
phased_fid=op_autophase(avg_met)

# The error shows up when you try to used phased_fid as an input to another
# function expecting phased_fid to just be an object containing the spectrum 
# but instead it tries to operate on the full tuple returned in the previous step
op_plotspec(phased_fid)
> "AttributeError: 'tuple' object has no attribute 'nucleus'"
```
Other errors are possible, depending on what the later function is trying to do with the tuple, which can make this problem hard to identify. However, errors of type AttributeError (above) and ValueError (shown below) are most common.

In the case of functions where you want to return, say, 2 out of the 5 possible output arguments, you may get a ValueError. For example, op_combineRcvrs has 5 possible output arguments, but you may only be interested in the first two. In this case, Python can't create a single tuple for all 5 output arguments because there are two variables to assign things to. So the error is thrown when it happens and may be easier to spot than in the case where an unexpected tuple of all output arguments is returned to one variable.
```python
[out_met, out_w]=op_combineRcvrs(in_met,in_w)
> "ValueError: too many values to unpack (expected 2)"
```

Because this behaviour is inherent to the way that Python functions work, there is no way to exactly replicate the Matlab behaviour. "Extra" return arguments must be dealt with explicitly, either by providing variable names for them in your code, or by somehow telling pyFidA and/or its functions when you don't want these extra outputs. I have designed a few options. Which is most suitable depends on how often you make use of the extra return arguments.
1. For processing functions with outputs beyond just the processed spectrum, there is an input argument called return_extra_args. Calling the function with return_extra_args=False will return only the first argument (typically the processed spectrum).
```python
# The default behaviour, returning all arguments to two separate variables
[phased_fid,ph0]=op_autophase(avg_met)

# Using the input argument to only return the autophased FID object
phased_fid=op_autophase(avg_met, return_extra_args=False)

# Now multiple processing steps can be chained together on one line without error
# because we can restrict a function to one output
averaged_fid=op_averaging(op_addrcvrs(metfid,return_extra_args=False))
```
2. If you find these input arguments cumbersome or you anticipate running a number of lines of code where you know that you will only use the first output argument, you can use allow_chaining() to tell pyFidA that it should only return the first output argument with the processed data in subsequent functions. Note that this only applies to selected pyFidA functions (the ones that have the "return_extra_args" input argument at the end). It does not apply to all Python functions or even all pyFidA functions. It is specifically designed to allow chaining together of processing functions. To end this behaviour and start returning all output arguments from functions, enter stop_chaining().
Note that you can still use the return_extra_args input argument to override the temporary behaviour set by allow_chaining() and stop_chaining()
```python
# After importing pyFidA, the default behaviour is to return all arguments
coil_combined_fid,fids_presum,specs_presum,coilcombos=op_addrcvrs(metfid)

# We can switch to single output arguments without using the return_extra_args input
# argument by calling the allow_chaining() function, which persists until stop_chaining()
# is called.
# This works over multiple lines.
allow_chaining()
coil_combined_fid=op_addrcvrs(metfid)
averaged_fid=op_averaging(coil_combined_fid)
# Or it works for chaining together on a single line
averaged_fid=op_averaging(op_addrcvrs(metfid))
# But you can turn it off for a single function call with the return_extra_args input argument
coil_combined_fid,fids_presum,specs_presum,coilcombos=op_addrcvrs(metfid,return_extra_args=True)

# After calling stop_chaining, multiple arguments are returned again
stop_chaining()
phased_average_fid,ph0=op_autophase(averaged_fid)
# This can also be overridden for a single function call with the return_extra_args 
# input argument
phased_average_fid=op_autophase(averaged_fid,return_extra_args=False)
```
3. If you think that you will almost always be ignoring the "extra" output arguments and only using the first output (the first processed spectrum), you can change the setting in your version of pyFidA so that it loads with a default behaviour that returns only the first argument on these processing functions. To do this, you need to find the alter_return_args.py file and open it. If you have imported pyFidA using "import pyFidA" you can find this file by first typing pyFidA.\__file__. This will give you the location of the pyFidA code for your current environment. Within this folder, you can navigate to pyFidA/fidA_processing and alter_return_args.py is in this folder. Near the top of the alter_return_args.py file, there is a line that says "\_use_default=ReturnBehaviour(True)". Change the word True to False (there is a comment above this line in the file explaining how users can change this line to adjust the default behaviour) and save the changes, overwriting the original file.
Once this is done (you will need to restart Python after saving changes), you no longer need to use the return_extra_args=False or the allow_chaining() function to get the single output argument behaviour. It is the default every time you import pyFidA. You can switch from the default behaviour with the input argument return_extra_args=True or calling stop_chaining()
```python
# For users who have set _use_default=ReturnBehaviour(False) in the file alter_return_args.py
from pyFidA import *

# The new default behaviour right after input
coil_combined_fid=op_addrcvrs(metfid)

# This can be swtiched in the return_extra_args input argument
coil_combined_fid,fids_presum,specs_presum,coilcombos=op_addrcvrs(metfid,return_extra_args=True)

# Or go back to returning all arguments for multiple lines with stop_chaining
stop_chaining()
coil_combined_fid,fids_presum,specs_presum,coilcombos=op_addrcvrs(metfid,return_extra_args=True)
averaged_fid=op_averaging(coil_combined_fid) # function only ever has one output argument
allow_chaining()

# Back to returning just the first output argument with the processed spectrum
phased_average_fid=op_autophase(averaged_fid)
```
4. Users can also make use of Python's ability to "pack" multiple variables/return arguments into a list with the \* operator. Although extra argument can be packed into any variable name, it is common to use the underscore '\_' to store all of the extra stuff that the user doesn't need to give a specific name to. This method is not part of pyFidA but it is the only way to return a subset of the output arguments >1. (the return_extra_args and allow_chaining() methods only deal with returning the first output vs returning all output arguments).
```python
# This does not work because there are four total output arguments in op_combineRcvrs
# Setting return_extra_args=True will try to return 4 output arguments to 2 variables.
# Setting return_extra_args=False will try to return 1 output argument to 2 variables.
out_met, out_w=op_combineRcvrs(in_met,in_w, return_extra_args=False)
> "ValueError: not enough values to unpack (expected 1, got 2)"

# This returns all 4 output arguments but the last two are "packed" into
# a single list with the variable name "_" using the * operator. This
# packing will work regardless of how many output arguments exist beyond
# the first three arguments
out_met, out_w, *_=op_combineRcvrs(in_met,in_w, return_extra_args=True)
```

<a name="processing"></a>
## Processing Module

### More consistency between various op_align functions
In Matlab, there are a large number of functions designed to align spectra. In many cases, code was repeated and input arguments are not always defined consistently between functions. Where possible, op_align functions in pyFidA have been designed to call existing functions, as appropriate, and to have similar input argument options.

For example, many functions in Matlab have a separate, corresponding function that ends in \_fd to restrict the fit to a particular frequency range. In pyFidA, the original alignment functions allow freq_range as an input argument (with default None, to use the whole spectrum) and then the corresonding \_fd function is just a wrapper that sets this freq_range variable and calls the original function.
eg. 
op_alignAllScans_fd(inlist, fmin, fmax)
will call op_alignAllScans(inlist, freq_range=[fmin,fmax]). This avoids duplicating all of the code in the \_fd functions.

In most cases, the initial spectrum or spectra that are input are rearranged into reference and float spectra in order to call op_alignScans on all of the combinations of two scans that are needed. This allows the code for the actual fitting algorithm to be located mostly in one place so that, if something needs to be changed, it can be done in this single function rather than having to be tracked down and changed everywhere. The exceptions are op_alignMPSubspecs and op_matchLW, where the underlying fitting function is different and so they cannot call op_alignScans. The function op_phaseAlignAverages does not involve fitting but rather calculates and average phase difference from the data.

In addition, some Matlab op_align functions allowed you to enter initial guesses for the fit parameters as initPars and some did not. And the default tmax value was different for different functions. Because most of the code feeds into op_alignScans, which performs these actions, most functions now have the option to enter an initial guess initPars. And the tmax default is calculated in a more consistent way between functions (based on an estimate of where the SNR falls below 5). If you want the previous default of 0.5 s that was used in some functions, enter that value manually.

In one case, the meaning of an input argument changed in order to make functions more consistent. In Matlab, the op_freqAlignAverages function had an input argument avg that defined whether the reference scan that the averages were aligned to was the overall average ('y') or the first average ('n'). However, op_freqAlignAverages is just op_alignAverages with mode='f' to restrict the fit to the frequency shift parameter only. So, in pyFidA, op_freqAlignAverages calls op_alignAverages with mode='f' BUT op_alignAverages has many more options for the reference spectrum used to align averages. These options are defined by the input argument med:
* 'a' aligns to the average
* 'y' aligns to the median
* 'n' aligns to the most representative individual average
* 'f' aligns to the first average
* 'r' aligns to a reference spectrum that is provided by the input argument ref

Note that this changes the meaning of 'y' and 'n' for op_freqAlignAverages compared to what they were in Matlab. This is partly conveyed by changing the input parameter name from 'avg' to 'med' in pyFidA.op_freqAlignAverages to match those for op_alignAverages, but wouldn't necessarily be clear for users who don't use keyword input arguments.

Output arguments for these functions also varied in whether frequency was returned before phase or vice versa in Matlab. In pyFidA, phase is always returned before frequency when both output arguments are present.
 
Finally, op_alignISIS has changed quite a bit. In pyFidA, this function aligns the second subspectrum to the first. In Matlab, the idea seemed to be to align both subspectra to some sort of combined subspectrum. But that combined subspectrum was calculated from the unaligned scan and therefore appeared to assume good alignment to start with.

<a name="lineshape"></a>
### Lineshape functions take FWHM in ppm; Peak-fitting functions take FWHM in Hz
Matlab's fid-A lineshape functions (like op_lorentz_linbas, op_gaussian, op_voigt_linbas) were inconsistent about how the units for full-width-at-half-max were defined. Some comments say that the FWHM is in Hz but really use ppm. Others use Hz but do the conversion assuming that the nucleus is 1H. Because of the way that the lineshape functions are called when fitting, they do not have access to the transmit frequency or nucleus and field strength, so they cannot easily convert between ppm and Hz without assumptions. Therefore, these functions require the FWHM to be entered in ppm because the units of the x values are in ppm.

However, because users are more likely to know their linewidth in Hz and because the peak-fitting functions (op_peakFit, op_creFit) are able to access the transmit frequency for conversion, these functions assume that the linewidth will be entered in Hz and this value is converted to ppm before fitting to the lineshape function, then the fitted parameters convert the ppm linewidth from the fit back to Hz before returning values to the user.

There is a check in the code that throws a warning if the value looks like it may be in the wrong units, but this is just based on the magnitude of the FWHM for the first peak so it isn't foolproof (particularly for non-proton spectra).

One issue with this difference is that users might want to take the output parameter parsFit from op_peakFit and feed it into a lineshape function (this value is returned from op_peakFit as outdat for the exact values in parsFit but there may be cases where users wish to generate a line to represent an error of 10% in a parameter or where the user wishes to see what a Gaussian with similar parameters to the Lorentzian might look like). In this case, the user cannot send the usual parsFit value from the peak-fitting function (in Hz) to the lineshape function (which requires an input in ppm). For this, users can run op_peakFit with the input argument returns_pars_in_Hz=False. Then parsFit will have FWHM values in ppm, which can be sent to the lineshape function.

<a name="lineshape2"></a>
### More flexibility in lineshape functions for fitting: multiple peaks, baselines, more lineshapes
The op_lorentz function in Matlab appears to be set up to accept a two-dimensional pars object, with the intention of constructing multiple peaks from each row in pars, and then summing along the rows to create the final spectrum. While this works for obtaining a simulated spectrum from a particular set of parameters, it creates instability when using the function for fitting because each peak has its own baseline parameters and phasing.

Imagine a spectrum with two peaks and zero baseline offset. This might be constructed with each peak having a baseline offset of 0, or by the first peak having a baseline of +42 and the second of -42, and so on. Similar issues exist for the baseline slope. And allowing different phases for each peak adds a large number of degrees of freedom that are not actually needed in reality, where a single phase correction may be needed for the entire spectrum, but not generally for individual peaks. There are therefore far more parameters than needed or realistic, most of which are coupled to one another.

In pyFidA, the amplitudes, linewidths and center frequencies are each entered as vectors (1D numpy arrays and Python lists are both accepted). When a vector has multiple items, the values are interpreted as belonging to multiple peaks. The parameters that come after these three vectors then represent the baseline slope, offset and phase and should be entered as scalars. These last three parameters are optional, the same as they are in Matlab, but with one value applied to the whole spectrum. In the case of fitting a single peak, the first three parameters can be entered as scalars rather than constructing single-element arrays or lists.

The op_lorentz_linbas function behaves similarly, but with an optional baseline slope parameter in the fourth position of pars.

In addition, pyFidA includes op_gauss and op_gauss_linbas functions that are not present in Matlab that behave in similar ways. There is also an op_voigt_linbas function, which is present in Matlab, but only accessible from within op_peakFit.m. This means that it is possible to fit functions with each of these lineshapes, but to exclude some parameters based on how the initial parameter guess is structured as an input argument to op_peakFit.

As in Matlab, op_peakFit includes a "peaktype" input argument that indicates the lineshape function to use for fitting. Strings ("lorentz", "lorentz2", "gauss" and "voigt"), correspond to existing pyFidA functions (op_lorentz_linbas, op_lorentz, op_gauss_linbas and op_voigt_linbas). If no initial parameter guess (parsGuess) is input, op_peakFit will try to fit a single peak, with all other baseline and phase parameters that the function includes. The initial guess is unique to each function, based on the known parameter order and certain assumption (eg. the linewidth parameter(s) are given an initial starting guess of 5 Hz). Multiple peaks must be specified as vectors in parsGuess, otherwise there is no way to know how many peaks are expected for the fitting function. If you wish to exclude baseline or phase parameters from the fit, these initial parameter estimates should also be left off the end of parsGuess. Parameters are always excluded starting from the end. If you wish to leave out the baseline slope from op_lorentz_linbas and only include a baseline offset, you must use a different function like op_lorentz.

In Matlab, op_peakFit does not include bounds on the fit parameters as a possible input argument. This is the default in pyFidA, but there is an option for users to enter bounds, or users can set parameter_bounds=True to have pyFidA construct what it considers to be reasonable bounds on the fit parameters. If fitting many peaks, one of the latter two options is recommended. This is particularly true when including phase as a fit parameter because there is parameter coupling between this value and the peak amplitudes (eg. a peak amplitude of 1 and a phase of 0 will yield the same y-value as a peak amplitude of -1 and a phase of 180). Bounds should be entered in the same format as parsGuess (ie. if there are multiple peaks, the upper bound for the amplitude of each peak should be grouped together in a vector; lower and upper bounds should be entered as a two-element tuple/list).

It is also possible to construct your own lineshape functions for use with op_peakFit, as long as they have two input arguments in the form [pars,xvals] but these will not generate a "smart" initial parsGuess (it will assign every parameter an initial guess of 1) if none is entered as an input argument. In addition the default parameter bounds from parameter_bounds=True cannot be generated since nothing is known about the expected parameters. For more, see [the lineshape section in Matlab Differences for Programmers]((./Matlab_differences_programmers.md#lineshape)).

### Convenience functions for basic math on spectra
pyFidA has implemented special methods that allow you to use some math operators in intuitive ways. Functions with the names from Matlab's fid-A like op_addScans, etc. are still available in pyFidA for users familiar with those.
```python
total_fid=fid1+fid2 # equivalent to op_addScans(fid1,fid2)
diff_fid=fid1-fid2 # equivalent to op_addScans(fid1,fid2,1)
double_fid=2*fid1 # equivalent to op_ampScale(fid1,2)
half_fid=fid1/2 # equivalent to op_ampScale(fid1,0.5)
'coils' in fid1 # equivalent to fid1.dims['coils']!=0 in Matlab (this form of the check on dims does not work in pyFidA)
fid1[:,:10] # equivalent to op_takeaverages(fid1,1:10) in Matlab if fid1.dims['averages']=1
```
The details of how this works are contained in a section of "[Matlab differences for Programmers](Matlab_differences_programmers.md)".

### More generalized processing functions
In Matlab, some processing functions do not operate on data with multiple dimensions. Sometimes this makes sense (eg. in op_alignAverages, you want data where coils have been combined before you try to align the averages) and this behaviour is preserved in pyFidA in those cases. However, for a function like op_ppmref, it may be that the user wants to know a peak position for every average or even every coil, rather than requiring the spectrum to already be averaged or coil-combined. pyFidA allows spectra with multiple dimensions to be entered in these cases.

### Fitting algorithms
For fitting, I have typically used scipy.optimize.curve_fit. This algorithm may be slightly different or have some different default options relative to Matlab's curve fitting functions. This could produce small differences in the results of functions like op_getSNR that attempt to fit a baseline.

<a name="simulation"></a>
## Simulation Module

### More consistency in the input argument order of functions with shaped waveforms in the simulation toolbox
In Matlab, there are functions with shaped waveforms that take a gradient (eg. sim_megapress_shaped). Some of these functions are called with the position argument before the gradient and some have these input arguments the other way around. In pyFidA, these functions have input arguments with position before gradient, the same order as they are sent to sim_shapedRF. For simulation code being copied from Matlab, the argument order will need to be changed for any code containing calls to: sim_megapress_shaped, sim_megapress_shapedRefoc, sim_megaspecial_shaped and sim_spinecho_shaped. Alternatively, you can make use of Python's ability to enter arguments as keyword arguments to be clear which value being entered is Gx vs what is dx, regardless of their order.

### Minor changes to phase of sim_onepulse_shaped
In Matlab, sim_onepulse is an excitation around the 'x' axis. sim_one_pulse_arbPh with an input argument of ph=0 excites around the 'x' axis. But sim_onepulse_shaped with phCyc=0 will excite around the 'y' axis because there is a 90 degree phase added to phCyc when calling sim_shapedRF. In pyFidA, I have removed this extra 90 degree addition so that these functions would produce similar results to one another when they have the same flip angle (90) and phase.

<a name="rf"></a>
## RF Pulses

### Estimation of time-w1 product
The estimation of the time-w1 product related to the desired flip angle works slightly differently than in Matlab for pulses that are not phase-modulated. In Matlab, Bloch simulations are run at a range of B1 power values to determine the magnetization at the end of the RF pulse. The longitudinal magnetization, Mz, is then plotted versus the corresponding B1 values and the user is asked to input the power corresponding to the flip angle that they want based on this visualization.

In Python, figures do not always display mid-function and not all setups can easily accept user input; it depends on the matplotlib backend. For example, standard Jupyter notebook setups run a cell at a time without stopping for user input, or some qt backends use inline plotting or figures that don't display until the current code finishes running. This means that not all users will be able to see the output in order to enter a B1 power. While it is possible to require users to work with certain backends or more interactive gui setups could be designed with other Python modules, this adds to the user requirements for a basic installation. Instead, pyFidA attempts to estimate the first Mz point that most closely corresponds to the desired flip angle (eg. the w1 value where Mz crosses 0 for a 90 degree pulse) and then outputs the plot of w1 versus Mz so that the user can double-check it (assuming the suppress_plots=False in the call to create a new instance of the RF_pulse object, the default). Users can still alter the w1 after seeing the plot with my_rf_pulse.w1=neww1.

### More properties available
TBC

List the new properties available (w1, etc)

Not sure whether it should be noted here or elsewhere, but Matlab uses "type" as the input argument for RF_pulse to define the pulse type. In Python, "type" is a special/reserved word. I've renamed it ptype/type_p/pulse_type in different areas. (I think the init call might have different options than in Matlab so it might be worth starting this section on RF_pulses with those differences and including this bit there).

### Plotting pulses
More info about the vectors generated/stored and plotting functions.

### Loading off-resonance RF pulses
Loading and manipulating off-resonance RF pulses in Matlab was not completed when this Python code was written. However, the code that was completed for loading these waveforms required users to specify that the waveform was off-resonance, and the code would then attempt to find the frequency offset and shift the waveform back on-resonance to calculate the time-w1max and time-bandwidth products. It appears that the intention was then to shift the waveform back off-resonance before saving the waveform in the RF_pulse structure in Matlab. However, this was not completed and so all off-resonance pulses that are loaded into Matlab are shifted back on-resonance upon loading (they can be shifted back off-resonace using rf_freqshift if the off-resonance frequency is known).

In pyFidA, a different approach is taken. By default, it is assumed that the pulse is on-resonance (f0=0) and the time-w1max and time-bandwidth products are calculated for this frequency. Alternatively, the user can provide a frequency offset in the input argument f0 if this value is known. The waveform in the file is then loaded and stored as-is in the RF_pulse object (ie. the off-resonance waveform is stored). The time-w1max and time-bandwidth products are calculated for this waveform at the indicated frequency value. A warning is thrown if this value appears to be incorrect (the desired flip angle cannot be achieved within a reasonable w1 range or the bandwidth cannot be calculated); a warning is also thrown in the f0=0 case if that appears to be incorrect.

If the f0 value is not known, or the user is not sure whether the pulse stored in the file is off-resonance or not, the user can load the RF_pulse object with f0=None. This tells Python to try to find the offset frequency and to calculate tw1 and tbw at that frequency. Because this is done via a loop through w1 values across a range of frequencies, each of which involves Bloch simulations of the pulse, it can take a long time to run, which is why it is not the default, but is an option for pulses where the resonance frequency is unknown.

In addition, not all functions in the Matlab RF toolbox were designed to work for gradient-modulated or off-resonance pulses, and several functions assume that the waveform has even timesteps without checking that this is the case. I have attempted to provide functionality for these cases in pyFidA, or to run a check and throw an error when a function cannot be implemented for a particular type of RF pulse.
