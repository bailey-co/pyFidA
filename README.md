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

## Changes from Matlab fid-A

For users familiar with Matlab fid-A, basic data processing functions can be called in the same way. Some minor differences are explained in [basic differences](docs/explanation/Matlab_differences_basic.md).

For those interested in programming, there are several important underlying changes, with Matlab structs being converted to Python objects. These differences are outlined in the [differences for programmers](docs/explanation/Matlab_differences_programmers.md). In particular, there is an explanation of the [FID class](docs/explanation/Matlab_differences_programmers.md#fidobject) that replaces Matlab's struct for holding spectral data.

## Installation

pyFidA is not yet available through a package management system. The github repository needs to be cloned and added to your path before it can be imported. The dependency requirements for pyFidA include: numpy, scipy, matplotlib, pandas and spec2nii. If you install [spec2nii](https://github.com/wtclarke/spec2nii), this should cover all of the other dependencies needed for pyFidA.

## License
I think, by default, this shows up in a separate tab on github, but you can also link the file here

## References
- [fid-A for Matlab](https://github.com/CIC-methods/FID-A)
- [Matlab fid-A pdf documentation](https://github.com/CIC-methods/FID-A/blob/master/FID-A_Documentation/FID-A_Manual.pdf)
- [spec2nii](https://github.com/wtclarke/spec2nii) used for loading spectral data from many vendors into a niftimrs format that is translated into the FID class for use in pyFidA