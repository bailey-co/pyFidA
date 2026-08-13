#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:02:41 2026

@author: Colleen Bailey
Input and output functions for LCMOdel
"""

from pathlib import Path
import numpy as np
from scipy.fft import fftshift
from pyFidA.fidA_common import FID, fid_from_specs, GAMMA_DICT

def io_readlcmcoord_getBackground(fname,part,nucleus=None):
    # Differs from Matlab in that the Python io_readlcmcoord function already 
    # reads in the whole coord file and I can just use this as a wrapper to
    # extract what I need
    
    # The Matlab options for part are slightly different than the strings that 
    # I used in the io_readlcmcoord lcm_dict, so this will convert where they
    # are different
    partdict={'bg':'bgrd','sp':'data'}
    try:
        partstr=partdict[part]
    except KeyError:
        partstr=part
    outdat=io_readlcmcoord(fname,partstr,nucleus=nucleus)
    return outdat

def io_readlcmcoord(fname,metab=None,nucleus=None):
    # So, I've already written this but there is a version in Matlab designed to
    # obtain a FID structure for a particular metabolite, and then a separate
    # function for the background. I've adapted this function so that it 
    # returns the full dictionary if there is no metabolite specified. If there
    # is a metabolite, it takes that spectral data and tried to make it into a 
    # FID object. Note that the background is not subtracted off of the metabolite
    # fits.
    lcm_dict=dict()
    info_dict=dict()
    # Note that using f.read() and then splitting on '\n' seems to work okay here
    # but, for other lcm files, eg. '/Users/nearlabmacbook1/Documents/BrukerData/ADrats/ved_basis/basis-sets/jn_press_7T_11ms.basis'
    # there are variables that run multiple lines and aren't correctly sorted.
    # For these, I've used f.readlines() and then split on ',', which is maybe
    # one more step than you need. But maybe not because the data sections don't
    # end with a comma and the header identifying sections don't end with a header either.
    with open(fname) as f:
        coord_text=f.read()
    div_strs=[' points on ppm-axis = NY','NY phased data points follow',
              'NY points of the fit to the data follow','NY background values follow',
              ' Conc. = ']
    coord_lines=coord_text.split('\n')
    nlines=int(coord_lines[2].split()[0])
    info_dict['names']=[eachlines.split()[3] for eachlines in coord_lines[4:4+nlines-1]]
    info_dict['conc']=[float(eachlines.split()[0]) for eachlines in coord_lines[4:4+nlines-1]]
    info_dict['SD_perc']=[int(eachlines.split()[1][:-1]) for eachlines in coord_lines[4:4+nlines-1]]
    info_dict['SNR']=int(coord_lines[4+nlines].split()[-1])
    tmpline=coord_lines[4+nlines]
    info_dict['FWHM']=float(tmpline[tmpline.index('=')+1:tmpline.index('ppm')])
    tmpline=coord_lines[4+nlines+1]
    info_dict['shift']=float(tmpline[tmpline.index('=')+1:tmpline.index('ppm')])
    tmpline=coord_lines[4+nlines+2]
    info_dict['phase']=float(tmpline[tmpline.index('deg')+3:tmpline.index('deg/ppm')])
    for divct,(divstr,partstr) in enumerate(zip(div_strs[:-1],['ppm','data','fit','bgrd'])):
        tmppts=coord_text[coord_text.find(divstr)+len(divstr):coord_text.find(div_strs[divct+1])]
        # one possibility is to save each of these as a FID object so that you can use the same
        # functions and stuff that you do for raw data but not sure how useful this is. Mostly
        # it might work for plotting but also you often want to plot multiple metabolites as a 
        # ridgeplot and I already have a function for that.
        if partstr=='bgrd': # Need to cut metabolite name after split. Can't specify metabolite in string because depends on basis set
            lcm_dict[partstr]=np.array([float(numstr) for numstr in tmppts.split()[:-1]])
        else:
            lcm_dict[partstr]=np.array([float(numstr) for numstr in tmppts.split()])
    # split up text into chunks for each metabolite
    tmpconc=coord_text.split('Conc. = ')
    npts=len(lcm_dict['ppm'])+1
    for concct,eachconc in enumerate(tmpconc[1:]):
        metname=tmpconc[concct].split()
        metname=metname[-1].strip()
        lcm_dict[metname]=np.array([float(numstr) for numstr in eachconc.split()[1:npts]])
    # Note that there is inconsistent spacing around the = sign so can't include '=' in search string
    varsplit=coord_text.split('hzpppm')
    varsplit=varsplit[-1]
    varsplit=varsplit[varsplit.find('=')+1:].split()[0]
    # Sometimes I write append to my control files and the same value appears twice.
    # It should be the last instance that matters.
    txfreq=float(varsplit)*1e6
    varsplit=coord_text.split('echot')
    varsplit=varsplit[-1]
    varsplit=varsplit[varsplit.find('=')+1:].split()[0]
    te=float(varsplit)
    tr=-1
    sequence='lcmcoord'
    if metab is not None:
        fids=fid_from_specs(lcm_dict[metab])
        spectralwidth=np.abs(lcm_dict['ppm'][0]-lcm_dict['ppm'][-1])*txfreq/1e6
        center_freq=np.abs(lcm_dict['ppm'][0]-lcm_dict['ppm'][-1])/2+min(lcm_dict['ppm'][0],lcm_dict['ppm'][-1])
        if nucleus is None:
            print('No nucleus entered. Assuming proton.')
            nucleus='1H'
        outdat=FID(fids,spectralwidth,txfreq,te,tr,sequence=sequence,dims=['t'],nucleus=[nucleus],center_freq_ppm=center_freq)
        return outdat
    else:
        return lcm_dict, info_dict
    
def io_loadlcmdetail(fname):
    # It looks like this is meant to be for the LCModel .print file.
    with open(fname) as f:
        detail_text=f.readlines()
    linestart=[fnum for fnum,line in enumerate(detail_text) if 'Correlation coefficients' in line][0]
    # The table starts two lines after the 'Correlation coefficients' header
    met_header=detail_text[linestart+2].split()
    # The file only holds 17 metabolites per row. So, if there are more than 17 metabolites
    # we need to read the next line. To know whether you need to go down another row,
    # you can check whether the first metabolite in the next row is equal to the 
    # second metabolite in the original line (it's a correlation matrix so the
    # first metabolite is not repeated)
    keep_going=True
    linect=1
    while keep_going:
        new_met=detail_text[linestart+2+linect].split()
        if new_met[0]==met_header[1]:
            keep_going=False
        else:
            met_header=met_header+new_met
            linect=linect+1
    # Matlab said something about the last metabolite name being missing from the file.
    num_metabs=len(met_header)+1
    corrMatrix=np.zeros([num_metabs,num_metabs])
    # Populate the lower part of the table. If there are more than 17 metabolites,
    # You need to read in multiple lines.
    max_cols=17
    new_linect=linestart+2+linect #This takes us to the next line after the end of the correlation coefficient table header
    # First metabolite doesn't have a line in the correlation matrix because it's just the diagonal element
    corrMatrix[0,0]=0.5
    for metct in range(1,num_metabs):
        lines_to_read=(metct)//max_cols+1
        next_line=[]
        # Note that this is untested for more than 34 metabolites (when we would get to the next table chunk) but should work
        for modct in range(lines_to_read):
            next_line=next_line+detail_text[new_linect].split()
            new_linect=new_linect+1
        currmet=next_line[0]
        float_line=[float(eachval) for eachval in next_line[1:]]+[0.5]
        corrMatrix[metct,:len(float_line)]=float_line
    metabs=met_header+[currmet]
    # Then we need to reflect these up into the upper part of the matrix
    corrMatrix=corrMatrix+corrMatrix.T
    return metabs,corrMatrix

def io_readlcmraw(fname,nucleus=None,imagingfreq=None,bandwidth=None,center_freq_ppm=None,do_conj=False):
    # Several problems trying to use spec2nii.other_formats.lcm_raw. (1) the
    # reader for the header assumes that there are spaces around the = so you
    # can use split() but this isn't necessarily the case. You can write RAW 
    # files without those spaces. In addition, spec2nii reads in the echo time
    # but does not actually transfer that to the nifti_mrs object for some
    # reason. There doesn't seem to be anywhere to set the center frequency
    # (hzpppm is read into the header as a center frequency but this is actually
    # the transmit frequency). Also, the code suggests that the data need to be
    # conjugated but, for the example RAW file that I have, this isn't the case.
    # Anyway, it seemed more straightforward just to read the info in myself 
    # and put it into the FID without going via nifti_mrs.
    fname=Path(fname)
    header=list()
    data=list()
    # Header is defined by a line that starts with '$' (can be more afterward)
    # and ends with '$END' but there can be more than one header section so
    # this variable keeps track of whether you are in a header section or not.
    in_header=False
    with open(fname) as f:
        for line in f:
            if '$' in line:
                in_header=True
            if in_header and '$' not in line:
                header.append(line)
            elif not in_header:
                data.append([float(eachval) for eachval in line.split()])
            if '$END' in line:
                in_header=False
    data=np.array(data)
    # Matlab suggests that LCModel raw needs to be conjugated after reading in
    # but, on the file that I have, that's not true. Have made it an input option.
    data=(data[:,0]+1j*data[:,1])
    if do_conj:
        data=data.conj()
    # process header
    header_dict=dict()
    for line in header:
        if '=' in line:
            lineproc=[eachel.strip() for eachel in line.split('=')]
        header_dict[lineproc[0]]=lineproc[-1]
    # Need to put header values into the right units
    # There are a few possible header names for the dwell time
    for kval in ['dwelltime','deltat','badelt']:
        try:
            spectralwidth=1/float(header_dict['bandwidth'])
        except KeyError:
            spectralwidth=bandwidth
    if spectralwidth is None:
        raise ValueError('ERROR: No spectral width found in file header. Please enter a value in Hz as input argument bandwidth.')
    try:
        txfreq=float(header_dict['hzpppm'])*1e6
    except KeyError:
        if imagingfreq is None:
            raise ValueError('ERROR: No transmit frequency found in file header. Please enter a value for imagingfreq in Hz as input argument bandwidth.')
        else:
            txfreq=imagingfreq #assumes entered in Hz
    try:
        te=float(header_dict['deltat'])/1e3
    except KeyError:
        te=-1
    tr=-1
    sequence='lcmraw_file'
    # If there are spatial dimensions, then you'll want to add those on to both
    # data via a reshape and the dims dictionary. Then allow an affine transform
    # as nifti_mrs does.
    if nucleus is None:
        print('No nucleus entered. Assuming hydrogen')
        nucleus=['1H']
    else:
        if type(nucleus) is not list:
            nucleus=[nucleus]
    if center_freq_ppm is None:
        if nucleus==['1H']:
            center_freq_ppm=4.65
        else:
            center_freq_ppm=0
    out1=FID(data,spectralwidth,txfreq,te,tr,sequence,dims=['t'],nucleus=nucleus,center_freq_ppm=center_freq_ppm)
    return out1

def io_readlcmraw_basis(fname,nucleus=['1H'],center_freq_ppm=4.65,return_info_dict=False,do_conj=True):
    # I think it would be handy to read everything in and split on "$END". You 
    # could probably even use this for io_readlcmraw above. Then, if that chunk
    # starts with a line that starts with "$", it's header. Otherwise, it's data
    # (For this, there will be multiple bits of data with their own headers at 
    # the start)
    # I think lcmodel only really takes 1H and automatically puts your center 
    # frequency at 4.65, so I've allowed the user to set something different but
    # I don't throw any warnings if they're not given and these defaults are used
    fname=Path(fname)
    with open(fname) as f:
        text_lines=f.readlines()
    startpts=[sval for sval,line in enumerate(text_lines) if ('$' in line and '$END' not in line) or ('$' not in line and '$END' in text_lines[sval-1])]
    endpts=startpts[1:]+[len(text_lines)]
    split_lines=[text_lines[sidx:eidx] for sidx,eidx in zip(startpts,endpts)]
    split_dict=dict()
    metlist=list()
    for eachchunk in split_lines:
        if '$END' in eachchunk[-1]:
            # So, some things are longer than one line. You want to first join all of
            # the data together and then split on ','
            headlist=''.join(eachchunk[1:-1]).split(',')
            lcmdict=process_lcmraw_header(headlist)
            if eachchunk[0].strip() in ['$SEQPAR','$BASIS1']:
                split_dict[eachchunk[0].strip()]=lcmdict
            else:
                # NMUSED comes before basis but the easiest access to the metabolite data is in basis
                if '$NMUSED' in eachchunk[0]:
                    saved_dict=lcmdict.copy()
                else:
                    # Do I want ID or metabo?? Either way, need to get rid of quotation marks
                    metnm=(lcmdict['METABO'].strip()[1:-1]).strip()
                    metlist.append(metnm)
                    split_dict[metnm]={'$NMUSED':saved_dict,eachchunk[0].strip():lcmdict}
        else:
            dlist=''.join(eachchunk).split()
            # Note that this is the raw data. Needs to be re-ordered into complex and shifted before converting to fid
            dvals=np.array([float(dval) for dval in dlist])
            split_dict[metnm]['DATA']=fftshift((dvals[::2]+1j*dvals[1::2]))
            if do_conj:
                split_dict[metnm]['DATA']=split_dict[metnm]['DATA'].conj()
    txfreq=split_dict['$SEQPAR']['HZPPPM']*1e6
    linewidth=split_dict['$SEQPAR']['FWHMBA']*txfreq/1e6
    te=split_dict['$SEQPAR']['ECHOT']# units?
    sequence=split_dict['$SEQPAR']['SEQ']
    spectralwidth=1/split_dict['$BASIS1']['BADELT']
    outdict=dict()
    if type(nucleus) is not list:
        nucleus=[nucleus]
    for metnm in metlist:
        outdict[metnm]=FID(fid_from_specs(split_dict[metnm]['DATA']),spectralwidth,txfreq,te,sequence=sequence,dims=['t'],nucleus=nucleus,center_freq_ppm=center_freq_ppm)
        outdict[metnm].linewidth=linewidth
    if return_info_dict:
        return outdict,split_dict
    else:
        return outdict
    
def io_readlcmraw_dotraw(fname,nucleus='1H',center_freq_ppm=4.65,do_conj=True):
    fname=Path(fname)
    with open(fname) as f:
        text_lines=f.readlines()
    # Should only be on $END, with data starting afterward
    endidx=[linect for linect,line in enumerate(text_lines) if '$END' in line][-1]
    header,data=text_lines[:endidx+1],text_lines[endidx+1:]
    header_dict=process_lcmraw_header(header)
    # Some things from the header have units and so are still strings. Need to do float(var.split()[0])
    get_headerval=lambda kval: float(header_dict[kval].split()[0])
    spectralwidth=get_headerval('Sweep Width')
    vectorsize=int(get_headerval('Vector Size'))
    Bo=get_headerval('B0 Field')
    hzpppm=Bo*GAMMA_DICT[nucleus]
    # Should I attempt to read in a TE? There isn't one present in the .RAW file example that I have but maybe it's possible?
    data=np.array([[float(eachval.strip()) for eachval in eachline.strip().split()] for eachline in data])
    # Apparently this data is the fids??
    data=data[:,0]+1j*data[:,1]
    if do_conj:
        data=data.conj()
    outdat=FID(data,spectralwidth,hzpppm*1e6,sequence='lcmraw.RAW file',dims=['t'],nucleus=[nucleus],center_freq_ppm=center_freq_ppm)
    return outdat
        
def process_lcmraw_header(headlist):
    headdict=dict()
    for eachline in headlist:
        if '=' in eachline:
            lineparts=eachline.strip().split('=')#eachline.strip().strip(',').split('=')
            kval=lineparts[0].strip()
            try:
                vval=float(lineparts[-1])
            except ValueError:
                vval=lineparts[-1].strip()
            headdict[kval]=vval
    return headdict
    
def io_readlcmtab(fname):
    info_dict=dict()
    with open(fname) as f:
        text_lines=f.read()
    text_lines=text_lines.split('\n')
    startline=[linect for linect,line in enumerate(text_lines) if line.startswith('$$MISC')][0]
    nlines=int(text_lines[startline].split()[1])
    tablines=text_lines[startline+1:startline+1+nlines]
    info_dict=dict()
    for eachline in tablines:
        lineparts=eachline.strip().split('=')
        leftside=[part.split()[-1].strip() for part in lineparts[:-1]]
        rightside=[part.split()[0].strip() for part in lineparts[1:]]
        # for loop won't run if no '=' because leftside=[]
        for ls,rs in zip(leftside,rightside):
            try:
                info_dict[ls]=float(rs)
            except ValueError:
                info_dict[ls]=rs
    return info_dict

def io_writelcm(infid,outfile,te=None,vol=8.0,tramp=1.0,doConj=True,isSeqacq=False):
    # Matlab runs a number of checks but the main question seems to be whether
    # only in.fids(:,1) is written to file, so the only check you really need 
    # is that there s one dimension. However, I've set it up so that it still 
    # saves if there's more data and just throws a warning, which might be a bad
    # idea. It won't be possible to recover what the other dimensions are from
    # the info that gets saved here.
    # Note that you probably want to set the default for doConj to False since
    # this is only needed for Bruker/Philips and Canon/Toshiba scanners. However
    # I have it as True since I'm usually making Basis sets for Bruker.
    if infid.ndim!=1:
        if np.prod(infid.sz)==infid.sz[0]:
            tmpfid=infid.fids.flatten()
        else:
            raise TypeError('ERROR: io_writelcm is only intended to write 1D fids. This fid has {:d} dimensions.'.format(infid.ndim))
    else:
        tmpfid=infid.fids.copy()
    RF=np.zeros([tmpfid.shape[0],2])
    # There are differences between vendors here that I haven't checked on.
    # Only really done with Bruker PV6.0.1
    RF[:,0]=np.real(tmpfid)
    RF[:,1]=np.imag(tmpfid)
    with open(outfile,'w') as f:
        f.write(' $SEQPAR')
        if te is None:
            f.write('\n echot= {:4.3f}'.format(infid.te))
        else:
            f.write('\n echot= {:4.3f}'.format(te))
        # Some changes here from Matlab to make use of the fid attributes
        f.write("\n seq= '{:s}'".format(infid.sequence))
        f.write('\n hzpppm= {:5.6f}'.format(infid.txfreq/1e6))
        f.write('\n NumberOfPoints= {:d}'.format(infid.fids.shape[0]))
        f.write('\n dwellTime= {:5.6f}'.format(infid.dwelltime))
        f.write('\n $END')
        f.write('\n $NMID')
        # These two flags default to False if they're not present in the LCModel
        # $NMID section, but they may need to be set for Basis files used to fit
        # certain scanners. For Bruker, Toshiba/Canon and Philips, the data need
        # to be complex conjugated before fitting (the parameter name in LCModel
        # is bruker=T even though it applies to other scanners). For very old 
        # Bruker systems, those that pre-date the Avance series, the real and 
        # imaginary data are acquired sequentially rather than simultaneously.
        # If you have old data like this, you will need to set isSeqacq=True
        # at the input in order for seqacq to be correct in $NMID. See the 
        # LCModel manual for more info.
        # NOTE: I've set these up here in case you're writing a raw file, but
        # the comments in Matlab suggest that this function might be more likely
        # to be used to write a PRESS spectrum and io_writelcmraw would be used
        # to write a .RAW file from simulation for a basis set.
        if doConj:
            f.write('\n bruker=T')
        if isSeqacq:
            f.write('\n seqacq=T')
        #fmtdata='(2E14.5)'
        fmtdata='(2E15.6)'
        f.write("\n id='ANONYMOUS ', fmtdat='{:s}'".format(fmtdata))
        f.write('\n volume={:4.3e}'.format(vol))
        f.write('\n tramp={:3.2f}'.format(tramp))
        f.write('\n $END\n')
        for eachct in range(RF.shape[0]):
            # Space after ":" in string form deals with hanging negative signs
            f.write('  {: 7.6e}  {: 7.6e}\n'.format(RF[eachct,0],RF[eachct,1]))
    return RF
            
def io_writelcmraw(indat,outfile,metab,comment='',doConj=True,isSeqacq=False):
    # Not sure what the difference between io_writelcmraw and io_writelcm is 
    # intended to be. I think io_writelcm is mainly intended for writing data
    # and io_writelcmraw is intended for writing the metabolites that make up
    # basis sets.
    if indat.ndim!=1:
        # Can be more than one dimension if all dimensions after the first are 1
        if np.prod(indat.sz)==indat.sz[0]:
            tmpfid=indat.fids.flatten()
        else:
            raise TypeError('ERROR: io_writelcm is only intended to write 1D fids. This fid has {:d} dimensions.'.format(indat.ndim))
    else:
        tmpfid=indat.fids.copy()
    # This is all quite confusing. Matlab writes RF' and this operator is the 
    # conjugate transpose. BUT earlier, op_complexConj is called, so you're 
    # actually doing two conjugations (ie. getting back to where you started)
    # and just writing the original file. BUT, I have a doConj input argument
    # which doesn't change the data written, but does set bruker=T so that 
    # LCModel will take the complex conjugate of the basis set files. Basically,
    # I need to check that Matlab and pyFidA output the same numbers (and then
    # I can leave the bruker=T parameter to be set by the user, if desired).
    #tmpfid=np.conj(tmpfid)
    RF=np.zeros([tmpfid.shape[0],2])
    RF[:,0]=np.real(tmpfid)
    RF[:,1]=np.imag(tmpfid)
    with open(outfile,'w') as f:
        f.write(' This RAW file was created using pyFidA, with the spectral')
        f.write('\n simulation adapted from the Matlab tool developed by Robin Simpson')
        f.write('\n and Jamie Near, FMRIB 2010')
        f.write('\n\n Experiment Name : {:s} {:4.3f}_{:s}Sim'.format(indat.sequence,indat.te,indat.sim))
        f.write('\n Comment : {:s}'.format(comment))
        f.write('\n\n User defined parameters:')
        f.write('\n\n Sweep Width = {:1.4f} Hz'.format(indat.spectralwidth))
        f.write('\n Vector Size = {:d} points'.format(indat.fids.shape[0]))
        # Note that I don't seem to have saved linewidth in my simulation files but
        # I guess I should make it possible
        # f.write('\n Apodization = {:1.4f} Hz'.format(indat.linewidth))
        f.write('\n B0 Field   = {:1.4f} T'.format(indat.Bo))
        f.write("\n\n $NMID ={:s}".format(metab))
        fmtdata='(2E16.6)'
        f.write("\n FMTDAT='{:s}'".format(fmtdata))
        # NOTE: I've used the LCModel parameters here rather than the way that
        # Matlab does it. Matlab takes the complex conjugate of the data entered
        # because whatever data was used for testing noted that the Matlab fidA
        # structure seemed to be the complex conjugate of what LCModel expected.
        # However, the LCModel manual explains that this is scanner dependent.
        # Bruker, Philips and Toshiba/Canon data are in complex conjugate relative
        # to other scanners. This is noted by setting bruker=T in the $NMID section
        # of the .RAW file (the parameter is named bruker regardless of whether
        # you're setting it for use with Philips, etc. data. seqacq is only 
        # relevant for old Bruker systems. If these lines aren't present in the
        # $NMID section of the file, their values default to False, so they're
        # only written if needed/set by input arguments.
        if doConj:
            f.write('\n bruker=T')
        if isSeqacq:
            f.write('\n seqacq=T')
        f.write('\n volume={:1.5e}'.format(1.0))
        f.write('\n tramp={:1.5f}'.format(1.0))
        f.write('\n $END')
        for eachct in range(RF.shape[0]):
            f.write('\n  {: 1.6e}  {: 1.6e}'.format(RF[eachct,0],RF[eachct,1]))
        f.write('\n')
    return RF