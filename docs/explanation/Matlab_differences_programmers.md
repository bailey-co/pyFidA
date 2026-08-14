# Differences between pyFidA and Matlab fid-A for programmers
This document explains the differences between pyFidA and Matlab fid-A that are likely to be of interest to those looking to extend the code or who need to understand what is going on "under the hood." If you will not be doing your own programming but are just looking to use pyFidA to load, process and simulate spectra, this document is likely more detailed than you need. The more relevant document for most users is [Differences between pyFidA and Matlab fid-A for basic users](Matlab_differences_basic.md) 

In general, pyFidA aims to take advantage of Python features like classes and dunder methods in order to reduce the amount of code, remove unnecessary "extra" calculations in many processing functions and to make the code more generalizable. This should make it easier to add in functionality for non-proton nuclei, MRSI capabilities, processing for 2D and 3D NMR data with indirect dimensions, or other user-defined extensions. Some familiarity with Python and numpy is assumed when outlining the changes below.
* [FID object](#fidobject)

# Using classes in place of structs
Matlab uses a struct to hold data in the form of free induction decays and spectra. Various information about this data is contained in different fields of the struct. Similarly, RF pulses are loaded into a struct that contains various information about the pulse. In Python, these Matlab structs have become classes: FID and RF_pulse. There is also a Hamiltonian class used in the simulation module, although it is much smaller.

Classes have a number of advantages over structs. A big one is that classes don't have fields (attributes) that just hold arrays and numbers. Instead, they can have functions. These functions can be used to make automatic connections between related attributes, like a free induction decay over time and its inverse Fourier transform, which produces a spectrum. Python also has an @property decorator so that these functions can be made to look like an attribute on the user side, even though they reference a function call (eg. you can type "mydata.specs" instead of "mydata.get_specs()". Using functions in this way means that related attributes do not all need to be re-calculated when one is changed, which reduces the amount of code in many processing functions. It also allows new properties to be added to the class that it might be convenient for users to have access to without having to update every processing function that would change a related attribute. This will likely become clearer with specific examples, outlined below. More details can be found in the docstrings within the module for each class.

<a name="fidobject"></a>
## 1. The FID object

### 1.1 The fids and specs attributes

In Matlab, suppose that spectral data are in a structure named mydata, which has a matrix mydata.fids with 250 separate free induction decay averages, such that it has size (2048,250). This structure will also have mydata.specs, which should be the inverse Fourier transform of mydata.fids with size (2048,250). If you call the op_averaging function, then the mydata.fids matrix will be averaged over the "averages" dimension; it then needs to run the inverse Fourier transform calculation on the new averaged free induction decay in order to get a new value for mydata.specs, which is also saved in the structure in case it is needed later (which it may or may not be).

In Python, mydata is an instance of the FID class, which stores the 250 free induction decay averages in mydata.fids, a numpy array with shape [2048,250], similar to Matlab. However, mydata.specs does not store a numpy array. Search for "def specs" in the FID.py module to see that specs is defined as a function. Above that, the @property decorator allows this to be referenced as mydata.specs without the brackets that are typically needed to call a function. Put another way, any call to "get" mydata.specs doesn't return a saved array but instead calls a function. This function takes the inverse Fourier transform of whatever numpy array is stored in mydata.fids at that moment.

This means that, if you write a processing function that alters mydata.fids, you do not need to tell that processing function to calculate and store a new value for mydata.specs. This function call is now associated with the FID object and will done automatically when mydata.specs is called. This also avoids unnecessary calls to the inverse Fourier transform function at intermediate processing steps where it may not be needed.

In cases where it is easier to work in the spectral domain (eg. first-order phase correction, as can be seen in op_addphase), mydata.specs can also be set because the "setter" part of the property decorator (@specs.setter) changes what is saved in mydata.fids. Just as in Matlab, you can type "mydata.specs=newspec" but now, behind-the-scenes, @specs.setter is a function that calculates the shifted fft of newspec and assigns the result to mydata.fids for any future reference. So, again, there is no need to explicitly write the new value of mydata.fids in your own processing function after you have set mydata.specs; the FID class does this work internally.

You will still need to change the spectral width or center frequency if the assignment to myfid.specs has a different frequency axis, just as you would in Matlab. There is no way for the setter to know if the ppm range has changed so this cannot be done automatically. However, there are other automatic links between other properties in the FID class that should make this change simpler, as described in the next sections.

### 1.2 Bo, spectralwidth, txfreq, center_freq_ppm, spectralwidthppm, dwelltime, t, ppm

These properties all relate to each other. Thus, only some need to be stored as values; the remaining ones can be functions that run the appropriate calculation when called. I have chosen to assign the spectralwidth, center_freq_ppm and txfreq values at initialization. The others make use of the property decorator so that they are calculated automatically from these and GAMMA (see next subsection), but they also have setters so that a user can set a new dwelltime and this will alter the underlying spectralwidth value stored in the object instance.

In summary, parameters related to frequency and time are:
* spectralwidth (assigned value directly)
* center_freq_ppm (assigned value directly)
* txfreq (assigned value directly)
* Bo (calculated from txfreq and GAMMA),
* spectralwidthppm (calculated from spectralwidth and txfreq), 
* dwelltime (calculated from spectralwidth).
* t vector (calculated from dwelltime and number of points in the first dimension of fids)
* ppm vector (calculated from center_freq_ppm, spectralwidth and number of points in the first dimension of fids. There are two semi-private variables \_ppmmin and \_ppmmax that are used for an intermediate stage of the calculation)

The vectors t and ppm are read-only and cannot be changed through a setter. These vectors are dependent on multiple other parameters and it isn't clear which should be adjusted when a new vector is entered. Instead, functions like op_freqrange calculate the spectralwidth and center_freq_ppm corresponding to the new spectrum and set these. Others are adjusted automatically; eg. any zero-padding of the fid (adding zeros to the end of the time dimension) will automatically be incorporated into the t vector because it uses the number of points in the fid attribute (and the inverse when truncating the fid in the time dimension). If you try to set t or ppm directly (or even \_ppmmin), you will get "AttributeError: can't set attribute." To change these vectors, the programmer needs to set the underlying related values explicitly, like center_freq_ppm.

### 1.3 A note about GAMMA

The gyromagnetic ratio GAMMA is determined from the nucleus assigned when an instance of the FID object is created ("1H" by default). The GAMMA value is pulled from a GAMMA_DICT dictionary of constants that is contained in fidA_common (and imported into other modules as needed):
```python
# newfid.GAMMA will give 42577000 because this is 1e6 times the value in GAMMA_DICT['1H']
newfid=pyFidA.FID(fid_data,spectralwidth=4000,txfreq=300.32e6,dims=['t','averages'],nucleus=['1H'])

# newfid.GAMMA will give 42577000 because the default nucleus is 1H
newfid=pyFidA.FID(fid_data,spectralwidth=4000,txfreq=300.32e6,dims=['t','averages'])

# newfid.GAMMA will give 17235000 because this is 1e6 times the value in GAMMA_DICT['31P']
newfid=pyFidA.FID(fid_data,spectralwidth=4000,txfreq=300.32e6,dims=['t','averages'],nucleus=['31P'])
```
The intention here was to allow for expansion of fidA to other nuclei relatively easily by adding gyromagnetic ratios to GAMMA_DICT as needed without the need for the user to enter them and then referencing the GAMMA attribute during processing. However, it seemed wise to allow the user to set this value in exceptional cases where they might be imaging a nucleus not contained in the dictionary. Therefore, GAMMA is currently designed with a setter so that the semi-private attribute self.\_GAMMA can be altered if needed.

### 1.4 Other read-only properties

Several other values that need to be set explicitly when they change in Matlab fid-A are instead read-only properties that call a function and therefore do not need to be updated explicitly in processing functions:
* sz: returns mydata.fids.shape
* ndim: returns the length of mydata.fids.shape
* averages: returns mydata.sz[mydata.\_dimlist.index('averages')] or 1 if there is no averages dimension
* subspecs: returns mydata.sz[mydata.\_dimlist.index('subspecs')] or 1 if there is no subspecs dimension
* rawAverages: typically the number of averages when the FID instance is created (stored on init and usually unchanged during processing)
* rawSubspecs: typically the number of subspecs when the FID instance is created (stored on init and usually unchanged during processing)

It is probably also worth noting that Matlab used "subSpecs" in the dims structure, "subspecs" when referencing the number of subspecs through "mydata.subspecs" and "rawSubspecs" for mydata.rawSubspecs. This inconsistent capitalization was difficult to remember. In pyFidA, it is "subspecs" except for rawSubspecs, where the first S is capitalized, similar to rawAverages vs averages.

### 1.5 1D arrays and singleton mydata.fids dimensions

Python allows 1D arrays (eg. mydata.fids.sz can return [2048,]) so you can have just the time dimension. Matlab is designed to have a minimum of 2-dimensions, eg. (2048,1), and Matlab fid-A typically assigns this singleton dimension to averages.

Instead of keeping the Matlab approach with a minimum requirement for two dimensions, it is preferable to remove singleton dimensions. This simplifies some processing functions and checks (eg. in Matlab, you not only have to check whether mydata.dims.averages has a non-zero value, but you also have to check whether size of that dimension is greater than one).

While it is possible to create an instance of the FID object with singleton dimenions, eg. mydata.fids.shape=[2048,1] and mydata.dims={'t':0,'averages':1}, this is not recommended and the singleton dimension may not be preserved through all processing steps. eg. When an array is sliced to return a single average like op_takaverages(myfid,4), the returned FID object has the averages dimension removed. You can use the op_squeeze(myfid) function to remove singleton dimensions.

### 1.6 The dims attribute

Quite a long explanation follows, but the short version is that you can get information about the dimensions of the fid (time, coils, averages, etc.) using dot notation in pyFidA in a way that looks similar to Matlab. eg. myfid.dims.t will return the index that corresponds to the time dimension in myfid.fids. However, the underlying class that stores the dimensional information and needs to be altered if the dimensions are changed is a list and semi-private attribute, myfid._dimlist. This information is made into the read-only property myfid.dims, which is a dictionary that allows items to be accessed via dot notation to replicate the Matlab form of the calls (so myfid.dims['t'] is equivalent to myfid.dims.t). Items that are not present will throw an error rather than returning -1 (Matlab returns 0 for missing dimensions but the equivalent for 0-indexed arrays in Python would be -1).

I will explain a little about the underlying reason behind the different setup in pyFidA, then outline the details of how it works, but the main takeaway is given in the paragraph above.

The dims structure in Matlab includes a bunch of dimensions that may not be included in the data and assigns them a value of 0. This is done, in part, because the dims attribute is sometimes used to check what dimensions are present in the data or whether something needs to be done in a processing function (eg. if mydata.dims.coils==0 in Matlab then there is no data for separate coils and you can't do coil combination). There are several issues with this:
* It is harder to extend the code to account for new dimensions in the future. For example, if there are 2D NMR data with an indirect dimension, or if 2D MRSI data are handled by adding 'x' and 'y' dimensions then, based on the convention above, these dimensions ought to be present in dims but return value 0 in cases where they are not present. Every function that creates a new dims attribute would then need to be updated to set these values any time someone adds a new dimension to pyFidA that isn't part of the current dims struct.
* When creating a new instance of a FID, you have to tell it what dimensions the array has. This means that you either need to remember the names of all of the non-existent dimensions so that you can assign them a value (0 in Matlab) OR create a function that will add on any "missing" values to the dims attribute any time you create a new FID. I did both of these when I started translating code from Matlab to Python before deciding to just change the way that dimensions worked altogether.
* Adding and removing dimensions is more complicated than necessary. For example, if you perform coil combination, the dimensionality of the fids array is reduced and you need to adjust mydata.dims to reflect the new order of dimensions. This involves setting mydata.dims.coils=0 but also finding any values in mydata.dims that were higher than the old mydata.dims.coils value and shifting them each down one.
* Reverse lookups are complicated. It is easy to find out which dimension is the averages dimensions from mydata.dims.averages. However, it is harder to find out "What is in the second dimension?" You need to search through all of the values in mydata.dims and see if any of them are 2.
* Python uses 0-indexed arrays and negative indexes and slices have meaning. The zero-indexed arrays are not a problem in themselves. You could create the dimensional information such that mydata.dims.t=0 and then assign the next dimensions in order, then use the value -1 instead of 0 to indicate that a dimension does not exist. However, when combined with the negative indices, there is high potential for errors that may not be easily identified.

    eg. suppose mydata.fids has size [2048,256] where these dimensions are time and averages. If I want to know how many coils the data comes from, I could use mydata.sz[mydata.dims['coils']]. What I want is for this to throw an error (or possibly tell me 1) because the data don't have information for separate coils. Instead, mydata.dims['coils'] returns -1 and mydata.sz[-1]=256, because the number of averages are returned as the size of the last dimension. And there is no indication in the code that I now have a coils value that is not a coils value at all!

    Of course, checks can be run at the start of the function but the fundamental issue is that we are co-opting integer values to try to indicate that something does not exist. In Matlab, an environment geared toward numbers, this may make some sense. However, Python has several options that are better-suited for this purpose: the 'in' word can be used to check whether something exists in an object, the None object can be used to indicate something isn't there, or you can simply make use of Python's error handling to either throw a KeyError/AttributeError for something that does not exist (or catch it in an Exception if you want to provide a more detailed error message or handle the problem another way).

Looking at the above list of concerns, a list seems to be a much better way of storing this information. I have named this list with the semi-private variable name \_dimlist for reasons that I'll explain below. This solves almost all of the above issues:
* If you need to add on functionality for dimensions that have not yet been considered or explicitly programmed into pyFidA (eg. indirect dimensions or MRSI), you can just decide on a name for these dimensions and add them to the list in cases where that data is present. In cases where these dimensions are not present, they simply do not appear in the list. While you will need to reference these new dimensions in functions that are affected by them, there is no need to add them to some sort of default dict and assign them a value of -1 in every existing function that re-creates the dims dict.
* You can obtain information about the association between a dimension's name/identifier and its value in either direction relatively easily using built-in list functions: eg. mydata.\_dimlist.index('averages') or mydata.\_dimlist[1].
* Dimensions can easily be added or removed from the list using mydata.\_dimlist.remove('averages') or mydata.\_dimlist.insert('coils',-1) (or whatever the correct index is for the new dimension). There is no need to update the dimensions "above" averages when averages is added or removed from the list because their position in the list automatically adjusts.
* Related to the above, it is easier to spot potential mistakes in the dimensional information. Firstly referencing a list index means that the value returned will always be an integer. It is also not possible for two names to refer to the same value, which can happen with the dims struct/dict if there is an error updating dimensions after insertion/removal. It is possible to have a dimension name repeated in the list, although this is easier to spot when printing out the string representation of the FID object than it is with the information in a dict format. Finally, checking consistency between the data array and dimension list is a simple and relatively intuitive matter of comparing mydata.sz==len(mydata.\_dimlist), whereas the dims dict/struct requires you to remove all -1 values before checking the length.
* An additional feature is that a list makes it a simple matter to construct a property for the \_\_contains\_\_ method (these double-underscore or dunder methods are explained more below) that uses the "in" keyword to check whether a dimension is present in the data:
```python
# To check whether the data have information from multiple coils
if 'coils' in mydata:
	print('There are {:d} coils in this data.'.format(mydata.sz[mydata._dimlist.index('coils')]))

# The above uses a dunder method so that the "in" object checks mydata._dimlist and is equivalent to
if 'coils' in mydata._dimlist:
	print('Same as above.')
```
* If you use the list format to look up a dimension, attempting to reference an index that does not exist will throw an error.
```python
print(mydata)
> "FID has fids size (4096, 16) and dimensions: t, averages"

mydata.\_dimlist.index('coils')
> "ValueError: 'coils' is not in list"
```

As mentioned above, the more Matlab-like format of the dimension information still exists and can be referenced through mydata.dims.averages or mydata.dims['averages']. This dims attribute was originally constructed as a dict because this is such a standard Python object for holding this kind of data and it provided all of the features that the Matlab struct used. However, it has since been updated to a class that allows the dot notation (in addition to the existing mydata.dims['averages'] dict notation) for some backward compatibility for those transferring functions from Matlab.

The information is accessible from mydata.dims as a read-only property, a function that creates a dict of (key, value) pairs from the values and indexes in mydata.\_dimlist. Thus, any update to mydata.\_dimlist is automatically reflected in mydata.dims. This dict is recreated from the list every time mydata.dims is called so there is some small loss in efficiency compared to storing the dimensional information in the dict directly. However, because I made the decision to shift from a dict to a list for storing the dimensional information after substantial pyFidA coding had already been done, much of the existing code uses the dims dict. This format is also more readable than mydata.\_dimlist.index('averages') in many cases.

The read-only nature prevents inexperienced users from altering the dimensional information unintentionally. If you need to alter the dimensional information in your own functions, you need to change the underlying list (likely with the insert/remove methods for lists, as described above). This list is identified as a semi-private variable by its single underscore in order to make clear that it should only be altered with caution, by those who understand what updating the dimensional information means.

The dictionary is constructed such that it only contains items in the list. Dimensions that do not exist will return a KeyError rather than a value of -1. For this reason, checks about whether a dimension exists should be run using the "if dimname in mydata" construction. Trying to use something like the Matlab check of "if mydata.dims[dimname]==-1" will result in a KeyError and stop the code running. Because mydata.averages and mydata.subspecs are now properties that return 1 in the case of *either* singleton dimensions or when there is no averages/subspecs dimensions, it is also possible to run checks like "if mydata.averages>1" in some cases.

A final note: when creating an instance of the FID object, the dimensional information should be input as a list, not a dict. In the \_\_init\_\_ method of the FID object, you can see that the input argument is named dims but it is assigned to self.\_dimlist. Despite its somewhat confusing name, it is not assigned to the dims dict directly (because the dict is a read-only property) and so should not be entered in a dict format.

### 1.7 The flags attribute and error/warning checks for processing functions

As with dims, the flags attribute in pyFidA is basically a dict, although I have added functionality to use dot notation. So you can call either mydata.flags['averaged'] or mydata.flags.averaged. My code uses the dictionary notation and the dot notation functionality was only added after-the-fact for other users more used to the Matlab fid-A notation.

I have updated the flags in the processing functions in the same way as is done in Matlab, so that this information should remain updated in case others require this information. However, it appears that the flags are largely redundant with the dimensional information in many cases. eg. you can use mydata.flags['addedrcvrs'] to check if coil combination is complete before aligning averages. However, it is just as easy to check "if 'coils' not in mydata" because the lack of a coil dimension indicates that coils have already been combined (or that information wasn't present to start with) and it's okay to proceed with aligning averages.

There are a few exceptions: mydata.flags['zeropadded'] is more easily checked with a flag rather than searching through the end of the time dimension of mydata.fids to try to see if all of the values are 0. And there is no easy way to check from the data whether it has been filtered or downsampled. However, these flag values are rarely if ever called and don't even seem to be consistently updated by all processing functions. In cases related to averaging, subspecs and coils, it is recommended to use the dimensional information rather than the flags for checks because there are more safeguards in place to ensure that it is being properly updated or not improperly overwritten, keeping it consistent with the data. 

### 1.8 Convenience methods (adding and scaling spectra, selecting slices of the fid)

Python classes allow dunder methods (double underscore methods, sometimes called magic methods) to make use of common operators that may be more intuitive. For example, the \_\_add\_\_ method can be used to describe what Python should do when the '+' operator is used with objects from your class. The FID class has several of these dunder methods, which can be used in place of fidA functions (the fidA_processing functions are still available for those familiar with them from Matlab).

1.8.1 fid1 + fid2

```python
newfid = fid1 + fid2

# This is the equivalent of
new_fid=pyFidA.op_addscans(fid1,fid2)

# But it makes repeated calls more intuitive
new_fid=fid1+fid2+fid2+fid4

# You can also make use of Python's sum() function if you provide an appropriate "zero" value
summed_fid=sum(list_of_fids, start=0*list_of_fids[0])
```

1.8.2 fid1 - fid2

fid1 - fid2 is the equivalent of op_addscans(fid1, fid2, 1) where the "1" argument indicates subtraction.

1.8.3 fid1 + 3.5

The assumed intention here is that the scalar value is to be added as a DC offset in the spectral domain. That is, the above statement would be equivalent to op_dccorr(fid1,'v',-3.5). Put another way, it creates a fid object with a spectrum equal to fid1.specs+3.5. Likewise, fid1 - 3.5 would subtract 3.5 from every point in the frequency spectrum.

1.8.4 2\*fid1

```python
newfid = 2*fid1

# This is an amplitude scaling equivalent to
newfid = op_ampScale(2,fid1)
```
The \_\_div\_\_ method is also implemented so that you can use fid1/2 or 0.5\*fid1. The method is also implemented for elementwise multiplication of two fid objects - fid1\*fid2 - although there isn't a particular use case for this (it is more just for completeness).

1.8.5 'coils' in fid1

This uses the \_\_container\_\_ method to check whether 'coils' is one of the dimensions in fid1.\_dimlist. Returns a boolean True/False answer. This method of checking whether dimensions are part of the fid is the main recommended method in pyFidA. The Matlab process of checking whether fid1.dims['coils']==-1 will not work because of changes in how the dims attribute is structured (explained above).

1.8.6 slicing, eg. myfid[:,5:8]

In Python, you can define a slicing operation using the \_\_getitem\_\_ dunder method in order to return parts of that object. With arrays, this is used to select a subset of the data in that array. For the FID object, myfid[:,5:8] will return a new FID object where newfid.fids=myfid.fids[:,5:8]. In that particular example, newfid has the same number of dimensions as myfid because 5:8 means that the second dimension still exists but now has a size of 3.

However, in the case where a slice is a single int, the default numpy slicing behaviour is to reduce the array size. So if array1.shape returns [2048,6] and array2=array1[:,0], then array2.shape returns [2048,]. Note that this is not the case for a slice that includes :, even if that slice only has size 1. That is, if array2=array1[:,:1] then array2.shape will return [2048,1]. This behaviour from numpy has been kept in the \_\_getitem\_\_ function of the FID object . As a result, in the case of an int, myfid[:,5] will remove that dimension from myfid so that myfid.dims will just return {'t':0}. This is done automatically, and accounts for cases with multiple ints, eg. fid2[:,0,2] will return an object where fid2.ndim is 1 and fid2.dims is {'t':0}. This allows for consistency for users familiar with numpy slicing and it also seems to make the most sense since the point of picking out 1 slice is often to operate on it as its own object, without singleton dimensions (eg. for plotting).

In some cases, slicing might be more convenient than functions like op_takaverages, op_takesubspec, etc., although the \_\_getitem\_\_ call requires you to know which dimension corresponds to the averages or subspecs in order to slice it. Functions like op_takeaverages are also designed to remove dimensions when a particular dimension is an integer instead of a range. This \_\_getitem\_\_ slicing is also useful when, for example, you want to iterate through each average to analyze frequency drift, or if you want to group subsets of averages together for a windowed average over time.
```python
# iterate through some averages to do something
for specct in range(myfid.averages):
	newfid=myfid[:,specct]
	# do something. Plotting averages in separate figures
	f1,ax1=plt.subplots(1,1)
	pyFidA.op_plotspec(newfid,plotax=ax1)
	
# Slicing also allows plotting within the same axes for every 25th average, for example
pyFidA.op_plotspec(myfid[:,::25])

# Take a windowed average (something similar can be done with op_blockAvg but slicing 
# may be more intuitive for those familiar with numpy)
newfid=myfid.copy()
for avg_ct in range(myfid.sz[1]-10):
	newfid[:,avg_ct]=pyFidA.op_averaging(myfid[:,avg_ct:avg_ct+10)
```

Note that flags are *not* adjusted in the current slicing behaviour so, even if you eliminate the averages dimension by selecting the first average of 250, myfid.flags['averaged'] will still return False.

Similarly, a \_\_setitem\_\_ dunder method is defined for uses like myfid[:,5:8]=myarray, which would be equivalent to myfid.fids[:,5:8]=myarray. This, of course, requires that myarray is the correct size to fit in this slice of myfid, or else an error will be thrown. The FID slice can be set using either a numpy array or another FID object. eg. if you want to want to phase just one subspec (and subspecs were the second dimension), you could run myfid[:,1]=pyFidA.op_addphase(myfid[:,1],10). The op_addphase function returns a FID object, but \_\_setitem\_\_ will take the fids attribute from that returned object and assign it to myfid.fids[:,1].

# The RF_pulse object
I covered most of the stuff about interactive plotting and the time-w1 estimation in Matlab_differences_basic.md. Not sure if there's anything that needs to be added here.

# Peak-fitting functions
Some differences are covered in the [Matlab Differences for Basic Users](./Matlab_differences_basic.md#lineshape). This section will cover how the translation from Matlab was implemented, for users who with to implement their own lineshape or fitting functions.

## The curvefit_tools module for fitting argument order
Matlab fid-A uses Matlab's nlinfit function for op_peakFit. Any lineshape function that is used with nlinfit (eg. op_lorentz) is expected to have two arguments, with the lineshape parameters coming before the x-values (frequencies):
```matlab
% In Matlab
y=op_lorentz(pars,ppm)
```

The main equivalent to nlinfit in pyFidA is scipy.optimize.curve_fit, but this expects arguments in the opposite order, with x-values coming before parameters in the lineshape function:
```python
# in Python
y=pyFidA.op_lorentz(ppm,pars)
```

While this is not important for users coming directly to pyFidA, it would mean that any users copying scripts from Matlab that contained lineshape function calls would need to swap the argument order. Instead, pyFidA has a curvefit_tool.py module with an nlinfit function that swaps the order of the arguments around before calling scipy.optimize.curvefit. This means that users can make use of lineshape functions that are ordered in the same way as Matlab (pars before ppm). If you are constructing your own lineshape functions to use with op_peakFit, they should follow this argument order. And, if you are constructing your own peak-fitting function that makes use of existing pyFidA lineshapes, you should import nlinfit from pyFidA.fidA_processing.curvefit_tools.py and use this as the fitting function.

## The curvefit_tools module for fitting multiple peaks
As described in [Matlab Differences for Basic Users](./Matlab_differences_basic.md#lineshape2), op_peakFit can fit multiple peaks with a given lineshape by entering vectors (1D numpy arrays or lists) for the amplitudes, linewidths and center frequencies of each peak. Each element in a vector will represent one peak. Because scipy.optimize.curvefit requires all parameters to be entered individually (ie. pars is one flat array where each element is a scalar), this functionality is also contained within the curvefit_tools module.

The documentation in the module contains the full details. A summary is that, when nlinfit is called, it first takes the lineshape function (eg. op_lorentz) and parameter list that are entered, and it creates three new functions. The first is a new version of the lineshape function that can accept a flattened list of input parameters. The second is a function that can reformat the original parameter list, where some elements are vectors that represent multiple peaks, into a flat list. This new lineshape function and flat list are the ones sent to scipy.optimize.curve_fit to obtain the fit parameters and the new lineshape function is able to reshape the flat list into the format accepted by the original lineshape function (eg. op_lorentz), which is then called. The resulting fit parameters are then reshaped back into the original shape, with vectors representing the amplitudes/linewidths/centre frequencies for multiple peaks, by the third function that is constructed form the original lineshape function and parameter list. These second and third functions are also used to reshape any parameter bounds that are specified, meaning that the bounds should be entered by the user in the same format as the initial parameter guess (with vectors specifying multiple peaks); bounds are optional.

This means that you can create your own lineshape functions that will fit multiple peaks simply by using the nlinfit function from the pyFidA.fidA_processing.curvefit_tools.py module. The amplitudes, etc. for each peak just need to be entered as lists or numpy arrays and nlinfit will flatten them before calling the fitting function, then re-format the flattened list back to its original format, with any lists or numpy arrays contained within it. This assumes, of course, that you have constructed your lineshape function to deal with these vectors appropriately. See op_lorentz_linbas for an example. Your lineshape functions can also be used with op_peakFit in many cases, although some alterations to op_peakFit will be needed for full functionality (see next section).

## Using op_peakFit with your own lineshape functions (including Hz to ppm conversion)
As in Matlab, op_peakFit includes a "peaktype"
As described in [Matlab Differences for Basic Users](./Matlab_differences_basic.md#lineshape), lineshape functions cannot easily convert between ppm and Hz because they do not have access to the transmit frequency of the spectrum. This can be a challenge as linewidth is often reported in Hz and users may expect linewidth parameters from fitting to be in Hz, but the x-values used in fitting functions are in ppm. However, op_peakFit does have access

## Minor changes in peak-fitting functions to avoid repeated code
Where possible, pyFidA functions are wrappers to avoid repeating code and so that changes can be more easily implemented in one place. For example, op_creFit is a wrapper of op_peakFit with certain initial parameter guesses and constraints. The op_lorentz function is a wrapper of op_lorentz_linbas with the slope value fixed to 0. And so on.

# Return arguments
(I think some of this is covered in basics so mostly you just want to reference that. But you can have some explanation of how programmers can add it to their own functions here).

For return args: Users familiar with Python programming can add this functionality to their own functions by importing and applying the @alter_return_args decorator to their function and then adding a variable with default "None" to the end of the input arguments. The exact name of the input argument does not matter (by convention, I have used "return_extra_args") but it MUST be the last argument. The remainder of the function is the same and the function should return all output arguments in this part of the code (it is the decorator that decides how many are returned to the user). Also note that the default value of the last input argument should be None. This is what allows its default behaviour to be "set" by allow_chaining()
eg.
@alter_return_args
def my_func(input1,input2,input3=0,return_extra_args=None)

Possible that Return Behaviour should just be defined in processing rather than in the common module, although I can see the case for implementing similar behaviour in fidA_sim if it's relevant.
