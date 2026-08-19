# pyFidA
pyFidA is a Python package for loading, processing, displaying, and simulating magnetic resonance spectroscopy data.

It is based on the [Matlab toolbox fid-A](https://github.com/CIC-methods/FID-A), translated to Python with permission from Jamie Near. The code is a work in progress, as is the documentation.

## Features
- Reading and writing data
	- Reads MRS data files from common MRI vendors (GE, Philips, Siemens, Bruker)
	- Writes to LCModel and jmrui formats
	- Reads LCModel output file formats (.coord, .table, .print)
	- Reads and writes RF pulse files from common MRI vendors
- MRS data processing
	- Aligining averages, removing bad averages, frequency shifting, etc.
	- Combining subspectra from spectral editing techniques
	- Calculation of SNR, linewidth, etc.
	- Basic peak fitting
- RF toolbox
	- Simulating RF pulse effects on magnetization
	- Analyzing RF power and bandwidth
	- Design of gradient-modulated waveforms
- Simulation toolbox
	- Simulate MRS spectra for a variety of pulse sequences
	- Allows ideal and numerical RF pulses

A Jupyter notebook demonstrating some of the processing features is available [in the tutorials](docs/tutorials/MRS_processing.ipynb).

Not yet implemented: MRSI, GPU acceleration (and exampleRunScripts generally)

There is limited functionality for non-proton spectroscopy. Users can enter the nuclei when loading data and the gyromagnetic ratio for that nucleus will be loaded and used. However, the functions have not been tested on non-proton data. Many function defaults (center frequency of 4.65 ppm, creatine peak position, displaying a spectral region between 0 and 4.5 ppm, etc) are designed for use with proton spectra and will need to be set explicitly for other spectra.

## Installation

pyFidA is not yet available through a package management system. The github repository can be cloned or downloaded, then added to your path. The dependency requirements are listed in pyproject.toml and include: numpy, scipy, matplotlib, pandas and spec2nii. Several tutorials are available in Jupyter notebook so this package is also recommended if you want to follow those. If you install [spec2nii](https://github.com/wtclarke/spec2nii) and matplotlib, this should cover all of the other dependencies needed for pyFidA. An example step-by-step process that _may_ work for [Anaconda](https://anaconda.org) users is:

1. In your shell:
```
conda create -n pyFidA
conda activate pyFidA
conda install -c conda-forge spec2nii matplotlib jupyter
```

2. Download the above pyFidA repository as a zip folder (or clone it with git). Unzip the folder into a known location on your computer. For example, place it in home/yourname/Documents/PythonScripts.

3. Open your Python interface and, before running anything else, run the following:
```python
import sys
sys.path.append('home/yourname/Documents/PythonScripts/pyFidA')
import pyFidA
```

If you are using a Python interface that allows startup scripts (eg. Spyder), you can add this code (or just the first two lines of it) to the startup script to avoid running it every time. Note that, within the pyFidA folder that you downloaded, there is a second folder labelled pyFidA with the code in it. You want sys.path.append to be just above this second pyFidA code folder (ie. the path should include the top-level pyFidA folder).

A similar process is possible with pip, installing the dependencies listed above and downloading the code.

## Getting Started
Documentation is still a work in progress. Tutorials will be available as [Jupyter notebooks in the tutorials section](docs/tutorials):
* [Example processing of raw data set](docs/tutorials/MRS_processing.ipynb)
* [Examples of using the peak-fitting tools](<docs/tutorials/Peak Fitting Examples.ipynb>)

## Changes from Matlab fid-A

For users familiar with Matlab fid-A, basic data processing functions can be called in the same way. Some minor differences are explained in [basic differences](docs/explanation/Matlab_differences_basic.md).

For those interested in programming, there are several important underlying changes, with Matlab structs being converted to Python objects. These differences are outlined in the [differences for programmers](docs/explanation/Matlab_differences_programmers.md). In particular, there is an explanation of the [FID class](docs/explanation/Matlab_differences_programmers.md#fidobject) that replaces Matlab's struct for holding spectral data.

## License
This project is licensed under a [BSD 3-Clause License](LICENSE.txt).

## References
- [fid-A for Matlab](https://github.com/CIC-methods/FID-A)
- [Matlab fid-A pdf documentation](https://github.com/CIC-methods/FID-A/blob/master/FID-A_Documentation/FID-A_Manual.pdf)
- [spec2nii](https://github.com/wtclarke/spec2nii) used for loading spectral data from many vendors into a niftimrs format that is translated into the FID class for use in pyFidA