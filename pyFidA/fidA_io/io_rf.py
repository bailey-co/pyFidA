# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 16:59:34 2025
pyFidA.fidA_io.fidA_io_rf.io_rf.py

@author: Colleen Bailey

RF input/output functions.
"""

import datetime
from pathlib import Path
import numpy as np
from pyFidA.fidA_common import RF_pulse, GAMMA_DICT
from pyFidA.fidA_common.RF_pulse import estimate_f0

def io_readpta(fname):
    with open(fname,'r') as f:
        linelist=[line.rstrip() for line in f]
    info=dict()
    rf_out=list()
    for line in linelist:
        if ':' in line:
            dkey,dval=line.split(':')
            try:
                info[dkey.strip()]=int(dval)
            except ValueError:
                try:
                    info[dkey.strip()]=float(dval)
                except ValueError:
                    info[dkey.strip()]=dval.strip()
        elif ';' in line:
            d1,d2=line.split(';')
            rf_out.append([float(dval) for dval in d1.split()][::-1])
        else:
            if not line.strip()=='': # empty line
                print('WARNING: Line without format found: '+line)
    rf_out=np.array(rf_out)
    rf_out=np.concatenate((rf_out,np.ones([rf_out.shape[0],1])),axis=1)
    rf_out[:,0]=rf_out[:,0]*180/np.pi
    return rf_out,info

def io_readRF(fname):
    # Keeping it simple and not reading the header into an info variable since
    # Matlab doesn't return this info
    with open(fname,'r') as f:
        linelist=[line.rstrip() for line in f if not line.startswith('#')]
    rf_out=list()
    for line in linelist:
        if line.strip()=='': # empty or blank line. Skip.
            pass
        else:
            lineparts=line.split()
            # The only file example that I have, I can't tell which of the columns 
            # is amplitude vs phase, so relying on the Matlab code.
            rf_out.append([float(dval) for dval in lineparts])
    rf_out=np.array(rf_out)
    return rf_out

def io_readRFBruk(fname):
    with open(fname,'r') as f:
        #linelist=[line.rstrip() for line in f if (line[:2]!="$$" and not (line[:2]=="##" and line[2]!="$"))]
        linelist=[line.rstrip() for line in f]
    def get_par(varnm):
        varline=[line for line in linelist if line.startswith(varnm)][0]
        dkey,dval=varline.split('=')
        try:
            dval=int(dval)
        except ValueError:
            try:
                dval=float(dval)
            except ValueError:
                dval=dval.strip() #string case
        return dval
    # We read this info in but never use it?
    pulse_type=get_par('##$SHAPE_EXMODE')
    flipangle=get_par('##$SHAPE_TOTROT')
    R=get_par('##$SHAPE_BWFAC')
    integ=get_par('##$SHAPE_INTEGFAC')
    npts=get_par('##NPOINTS')
    startline=[lct for lct,line in enumerate(linelist) if line.startswith('##XYPOINTS')][0]
    rf_out=np.zeros([3,npts])
    # I need an actual rf file to check this. Do I need to add 1 and start after
    # the XYPOINTS line, or do I need to deal with the XYPOINTS line?
    for rfct,rfline in enumerate(linelist[startline+1:startline+1+npts]):
        d1,d2=rfline.split(',')
        rf_out[0,rfct]=float(d1)
        rf_out[1,rfct]=float(d2)
        rf_out[2,rfct]=1
    return rf_out.T

def io_readRFtxt(fname,col_order=['a','p','t','g']):
    # Jamie has a much more complicated col_order, where amplitude and
    # phase are reversed and it allows for empty columns. Why not just transpose?
    # It seems like maybe amplitude and phase are reverse by default but not
    # going to implement just yet until I can confirm.
    # For column order, phase and amplitude are mandatory. Time steps are 
    # optional but will be added as a vector of ones if not present. Gradient is
    # optional and will only be added if present in col_order
    with open(fname,'r') as f:
        linelist=[line.rstrip() for line in f]
    rftmp=list()
    for line in linelist:
        if line.strip()=='': # empty or blank line. Skip.
            pass
        else:
            lineparts=line.split()
            rftmp.append([float(dval) for dval in lineparts])
    rftmp=np.array(rftmp)
    if len(col_order)!=rftmp.shape[1]:
        print('WARNING: col_order vector does not match length of rf waveform. Truncating col_order!!! Check that RFpulse waveform is in order phase, amplitude, time and (optional) gradient in final result!')
        col_order=col_order[:rftmp.shape[1]]
    rf_out=list()
    rf_out.append(rftmp[:,col_order.index('p')])
    rf_out.append(rftmp[:,col_order.index('a')])
    if 't' in col_order:
        rf_out.append(rftmp[:,col_order.index('t')])
    else:
        rf_out.append(np.ones([rftmp.shape[0]]))
    if 'g' in col_order:
        rf_out.append(rftmp[:,col_order.index('g')])
    rf_out=np.array(rf_out).transpose()
    return rf_out

def io_loadRFwaveform(fname,type_p,off_res=0,Tp=None,col_order=None,nucleus='1H',suppress_plots=False):
    """
    General function for reading in RF pulses. A function is called depending on
    the suffix of the filename. There are several differences from Matlab. 
    Firstly, the estimate of tw1 and tbw are moved to the RF_pulse object.
    Secondly, off-resonance pulse waveforms are not adjusted back to f0=0 with 
    a phase ramp as in Matlab. Instead, the waveform from the file is used and
    the offset frequency needs to be given (or can be calculated) and the Bloch
    simulations are centered around this frequency

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.
    type_p : TYPE
        DESCRIPTION.
    off_res : TYPE, optional
        DESCRIPTION. The default is 0.
    Tp : TYPE, optional
        DESCRIPTION. The default is None.
    col_order : TYPE, optional
        DESCRIPTION. The default is None.
    nucleus : TYPE, optional
        DESCRIPTION. The default is '1H'.
    suppress_plots : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    FileNotFoundError
        DESCRIPTION.
    ValueError
        DESCRIPTION.

    Returns
    -------
    rf_struct : TYPE
        DESCRIPTION.

    """
    # For .txt files, column order can vary. In addition, time and gradient columns
    # may or may not be present. This is the only type for which col_order is 
    # currently implemented.
    
    # Simplify for numerical cases of type_p that are exact 90 or 180 flip angles
    if type_p==90:
        type_p='exc'
    elif type_p==180:
        type_p='inv'
    if isinstance(fname,str) or isinstance(fname,Path):
        fname=Path(fname)
        endstr=fname.suffix#fname[fname.rfind('.'):]
        if endstr=='.pta':
            print('Siemens format .pta RF pulse file detected!! Loading waveform now.')
            rf,info1=io_readpta(fname)
        elif endstr=='.RF':
            print('Varian/Agilent format .RF RF pulse file detected!! Loading waveform now.')
            rf=io_readRF(fname)
        elif endstr=='.inv' or endstr=='.rfc' or endstr=='.exc':
            print('Bruker format {:s} RF pulse file detected!! Loading waveform now.'.format(endstr))
            rf=io_readRFBruk(fname)
        elif endstr=='.txt':
            if col_order is None:
                col_order=['a','p','t','g']
            print('Basic .txt format RF pulse file detected!! Loading waveform now. Assumed column order: '+str(col_order))
            rf=io_readRFtxt(fname,col_order=col_order)
        else:
            raise FileNotFoundError('ERROR: RF Pulse file extension not recognized.  Aborting!')
    else: #if it's not a filename, it should be an array with 2, 3 or 4 columns with the rf waveform info
        if fname.ndim==2:
            print('Input is an array already in the workspace. Creating waveform')
            if fname.shape[1]==3 or fname.shape[1]==4:
                rf=fname
            elif fname.shape[1]==2:
                rf=np.concatenate((fname,np.ones([fname.shape[0],1])),axis=1)
            else:
                raise ValueError('ERROR: Input array for name must have 2 to 4 columns')
        else:
            raise ValueError('ERROR: RF pulse must be a 2D array.')
    # Assume a 5 ms rf pulse for tw1 and tbw calculations
    if Tp is None:
        Tp=0.005
    if off_res:
        # Matlab frequency shifts the pulse to 0 Hz but I think that it makes more sense
        # to keep the waveform intact and set f0. If the user wants to make it into
        # an on-resonance pulse then they can call rf_struct.f0=0 after loading
        f0=estimate_f0(rf,Tp,gamma=GAMMA_DICT[nucleus],w1_start=0.03,w1_step=0.02)
    else:
        f0=0
    
    rf_struct=RF_pulse(rf,type_p,Tp,f0=f0,gamma=GAMMA_DICT[nucleus],suppress_plots=suppress_plots)
    return rf_struct
    
def io_writepta(rf_in,outfile,ignore_adiabatic=False,refgrad=None):
    # Note that there's no option to write a gradient file if there is a gradient
    # in the rf waveform
    outfile=Path(outfile)
    if rf_in.isGM:
        print('WARNING: Attempting to write gradient-modulated waveform. Only rf portion will be written and some calculations assume no gradient. Gradient waveform must be written separately.')
    expanded_wf=rf_in.get_expanded_wf()
    B1INT=np.sum(expanded_wf[:,1])
    rfnew=np.zeros([3,expanded_wf.shape[0]])
    rfnew[0,:]=expanded_wf[:,1]/np.amax(expanded_wf[:,1])
    rfnew[1,:]=np.mod(expanded_wf[:,0]*np.pi/180,2*np.pi)
    # By using expanded_wf, all timepoints should be equal, so cumulative sum is
    # just a linear thing.
    rfnew[2,:]=np.r_[:expanded_wf.shape[0]]
    # For simplicity, make a complex-valued version of the pulse (I already have this as an RFpulse method)
    RF_complex=rf_in.get_complex_wf(expanded=True)
    # Calculate the power integratl (POWERINT). This is used for calculating the SAR
    POWERINT=np.sum((np.real(RF_complex)**2)+(np.imag(RF_complex)**2))
    # Calculate the magnitude/absolute integral (ABSINT)
    ABSINT=np.sum(np.abs(RF_complex))
    # Amplitude integral (AMPINT)
    AMPINT=rf_in.get_ampint(ignore_adiabatic=ignore_adiabatic)
    # Reference gradient is used for slice-selective pulses. It refers to the
    # gradient required to excite a 10 mm slice if the pulse duration is 5.12 ms.
    # Given that the time-bandwidth product is included in the RF_pulse object,
    # the refgrad value can be calculated here, but the option is provided to
    # override and enter it as an input argument, in case a different refgrad
    # value is desired
    if refgrad is None:
        print('Using refgrad calculated from rf_in._tbw. A manual reference gradient in mT/m can be entered as input argument "refgrad" to override.')
        # rf_in._tbw is unitless (ms*kHz) / (5.12e-3 s * gamma MHz/T * 1e6 Hz/MHz * 0.01 m) for a 10 mm slice
        # So need to multiply by 1000 mT/T to get a final value in mT/m
        refgrad=1000*rf_in._tbw/(5.12e-3 * rf_in.gamma*1e6 * 0.01)
    # Now ready to write to pta file for Siemens
    with open(outfile,'w') as f:
        f.write('PULSENAME: \t{:s}'.format(outfile.name))
        f.write('\nCOMMENT: \tRF pulse generated using the pyFidA toolkit.')
        f.write('\nREFGRAD: \t{:5.6f}'.format(refgrad))
        f.write('\nMINSLICE: \t1.00000000')
        f.write('\nMAXSLICE: \t200.0000000')
        f.write('\nAMPINT: \t{:5.6f}'.format(AMPINT))
        f.write('\nPOWERINT: \t{:5.6f}'.format(POWERINT))
        f.write('\nABSINT: \t{:5.6f}'.format(ABSINT))
        f.write('\n\n')
        for eachline in rfnew.T:
            f.write('{:1.9f} {:1.9f} ; ({:1.0f})\n'.format(*eachline))
    return rfnew

def io_writeRF(rf_in,outfile):
    # Note that there's no option to write a gradient file if there is a gradient
    # in the rf waveform
    outfile=Path(outfile)
    if rf_in.isGM:
        print('WARNING: Attempting to write gradient-modulated waveform. Only rf portion will be written and some calculations assume no gradient. Gradient waveform must be written separately.')
    expanded_wf=rf_in.get_expanded_wf()
    rfnew=np.zeros([3,expanded_wf.shape[0]])
    rfnew[0,:]=expanded_wf[:,1]/np.amax(expanded_wf[:,1])
    rfnew[1,:]=np.mod(expanded_wf[:,0]*np.pi/180,2*np.pi)
    # By using expanded_wf, all timepoints should be equal, so cumulative sum is
    # just a linear thing.
    rfnew[2,:]=np.r_[:expanded_wf.shape[0]]
    # Matlab leaves a bunch of calculations in the code that aren't actually 
    # used. I've remove them here.
    
    # For some reason, Matlab then makes a "pulse" matrix and this 1023 factor 
    # is in there.
    pulse=np.zeros([2,expanded_wf.shape[0]])
    pulse[0,:]=expanded_wf[:,0] # No conversion to radians in code
    pulse[1,:]=1023*np.abs(expanded_wf[:,1])/np.amax(np.abs(expanded_wf[:,1]))
    # Now ready to write to pta file for Siemens
    with open(outfile,'w') as f:
        f.write('# RF pulse generated using the pyFidA toolkit.\n')
        f.write('# number of points = \t{:5.2f}\n'.format(pulse.shape[1]))
        f.write('# TYPE \tselective\n')
        f.write('# MODULATION \tamplitude\n')
        f.write('# INTEGRAL \t-1\n')
        f.write('# ************************************************\n')
        for eachline in pulse.T:
            f.write('{:8.3f} {:9.3f} ; 1.0\n'.format(*eachline))
    return rfnew

def io_writeRFbruk(rf_in,outfile):
    outfile=Path(outfile)
    if rf_in.isGM:
        print('WARNING: Attempting to write gradient-modulated waveform. Only rf portion will be written and some calculations assume no gradient. Gradient waveform must be written separately.')
    expanded_wf=rf_in.get_expanded_wf()
    rfnew=np.zeros([expanded_wf.shape[0],2])
    rfnew[:,0]=100*expanded_wf[:,1]/np.amax(expanded_wf[:,1])
    rfnew[:,1]=np.mod(expanded_wf[:,0],360)
    # In Matlab, the min is taken across the second dimension but that can't 
    # be right. More strange is that I think that you want to write the real
    # and imaginary components but this is phase and amplitude? (Edit: I
    # pulled some Bruker RF pulses off the scanner and it appears to just be
    # amplitude in the 0th column and phase in the 1st. That's based on square.100
    # from the lists/wave). Similarly for Gaus1.1000 (neither of these have any
    # phase so...). I still think the minx and maxx have to be across rows though.
    # I also checked it on the pulses that we wrote to the method file for
    # comp_MRS
    minx=np.amin(rfnew[:,0])
    maxx=np.amin(rfnew[:,0])
    miny=np.amin(rfnew[:,1])
    maxy=np.amin(rfnew[:,1])
    # Fill in the INTEGFAC value. It appears that this is the integral of the 
    # shaped pulse (presumably its amplitude) relative to that of a square pulse
    # of the same length. This should make it equivalent to AMPINT. However,
    # in https://www.pascal-man.com/pulseprogram/avance3/topspin_2_1/shape_tool.pdf
    # they talk about the max power as gamma*B1max/2 and seem to say that this is
    # gamma*B1/2*pi (for a square pulse of the same length) / integral ratio. Since
    # the amplitude is normalized to 1 (or, actually, 100 for Bruker but the
    # square wave calculation is for a square pulse amplitude normalized to 1.
    # ie. gamma*B1/2*pi = 1/(Tp*360/FA) for a square pulse.
    INTEGFAC=rf_in.get_ampint()
    currtime=datetime.datetime.now()
    thisdate=str(currtime.date()).replace('-','/')
    thistime=currtime.time()
    ptype_dict={'exc':'Excitation','ref':'Refocusing','inv':'Inversion'}
    with open(outfile) as f:
        f.write('##TITLE= {:s}'.format(outfile.name))
        f.write('\n##JCAMP-DX= 5.00 Bruker JCAMP library')
        f.write('\n##DATA TYPE= Shape Data')
        f.write('\n##ORIGIN= pyFidA')
        f.write('\n##OWNER= nmrsu')
        f.write('\n##DATE= {:s}'.format(thisdate))
        f.write('\n##TIME= {:02d}:{:02d}:{:02d}'.format(thistime.hour,thistime.minute,thistime.second))
        f.write('\n##MINX= {:2.6e}'.format(minx))
        f.write('\n##MAXX= {:2.6e}'.format(maxx))
        f.write('\n##MINY= {:2.6e}'.format(miny))
        f.write('\n##MAXY= {:2.6e}'.format(maxy))
        # Options for excitation mode are Excitation, Inversion, Refocusing, Universal, Universal180, Decoupling, Adiabatic, CompositeAdiabatic, Bib and Gradient
        # It seems as though basic pulses like Gauss and sine can be universal?
        f.write('\n##$SHAPE_EXMODE= {:s}'.format(ptype_dict[rf_in.pulse_type]))
        if rf_in.pulse_type=='inv' or rf_in.pulse_type=='ref':
            f.write('\n##$SHAPE_TOTROT= 18.00000e+1')
        elif rf_in.pulse_type=='exc':
            f.write('\n##$SHAPE_TOTROT= 90.00000e0')
        else: # Any numerical value
            f.write('\n##$SHAPE_TOTROT= {:2.6e}'.format(rf_in.pulse_type))
        f.write('\n##$SHAPE_BWFAC= {:2.6e}'.format(rf_in.tbw))
        f.write('\n##$SHAPE_INTEGFAC= {:2.6e}'.format(INTEGFAC))
        f.write('\n##$SHAPE_REPHFAC=')
        # There are actually a bunch of different parameters for adiabatic pulses
        # (based on what shows up in ShapeTool). Coming back to this later.
        if rf_in.isAdiabatic:
            f.write('\n##$SHAPE_TYPE=adiabatic')
            print('WARNING: Writing adiabatic waveforms not yet tested for Bruker.')
        else:
            f.write('\n##$SHAPE_TYPE= {:s}'.format(ptype_dict[rf_in.pulse_type]))
        f.write('\n##$SHAPE_MODE= 0')
        f.write('\n##NPOINTS= {:d}'.format(rfnew.shape[0]))
        f.write('\n##XYPOINTS= (XY..XY)')
        f.write('\n')
        for eachline in rfnew:
            f.write('{:1.6e}, {:1.6e}\n'.format(*eachline))
        f.write('##END')
    return rfnew

def io_writeRFtxt(rf_in,outfile):
    if rf_in.waveform.shape==3:
        str_fmt='{:5.5f}  {:5.5f}  {:5.5f}\n'
    elif rf_in.waveform.shape==4:
        str_fmt='{:5.5f}  {:5.5f}  {:5.5f}  {:5.5f}\n'
    with open(outfile) as f:
        for eachline in rf_in.waveform:
            f.write(str_fmt.format(*eachline))


if __name__ == '__main__':
    """
    for debugging
    """
    import matplotlib.pyplot as plt
    from pyFidA.fidA_common import BlochSimulator
    
    pname='/Users/nearlabmacbook1/Documents/PythonScripts/pyFidA/exampleData/rfPulses'
    #RFtest=io_loadRFwaveform(os.path.join(pname,'sampleExcPulse.pta'),'exc')
    #print(RFtest.get_ampint())
    #RFtest,info1=io_readpta(os.path.join(pname,'sampleExcPulse.pta'))
    #RFtest=io_readRFtxt(os.path.join(pname,'GOIA_tthk0.01_R120.txt'))
    #RFtest=io_readRF(os.path.join(pname,'sampleAFPpulse_HS2_R15.RF'))
    # I tested a pulse with a few phase changes (sampleExcPulse.pta). I haven't
    # tested anything gradient modulated yet. No idea what to expect there.
    # Edit: tested on GOIA_tthk0.01_R120.txt. It ran. And it did indeed seem to
    # produce something very thin with B1=0.14, if that was the goal.
    # Also tried sampleAFPpulse_HS2_R15.RF, which is presumably an adiabatic 
    # pulse and has more complex phase stuff. And it seems to do something reasonable
    # for a 5 ms pulse with B1=0.3 and it move off-resonance stuff in a fairly
    # wide range, like -1 to 1 Hz??? Or is that wide?
    # bvec=np.linspace(0.01,0.5,100)
    # bloch1=BlochSimulator(RFtest,5,0,bvec)
    # plt.figure()
    # plt.plot(bvec,bloch1.finalM0[2,:])
    # plt.figure()
    # plt.plot([bm[50][2] for bm in bloch1.mvec])
    # fvec=np.linspace(-5,5,1000)
    # bloch1=BlochSimulator(RFtest,5,fvec,0.3)
    # plt.figure()
    # plt.plot(fvec,bloch1.finalM0[2,:])
    # plt.figure()
    # plt.plot([bm[500][2] for bm in bloch1.mvec])
    fname='/Users/nearlabmacbook1/Documents/PythonScripts/pyFidA/exampleData/rfPulses/sech30.5.hwt'
    rfwf=io_readRFBruk(fname)
    #rf_complex=RF[:,0]*np.exp(1j*RF[:,1]*np.pi/180)
    #AMPINT=np.sum(np.abs(rf_complex))/np.amax(np.abs(rf_complex))/rf_complex.shape[0]
    #print(AMPINT)