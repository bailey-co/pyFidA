#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:51:37 2026
pyFidA.fidA_io.io_fidspec.py

@author: Colleen Bailey (cbailey@sri.utoronto.ca), based on Matlab code by 
    Jamie Near
    
This module contains functions for reading spectral data from various MRI 
vendors into pyFidA's FID object. It also allows writing to jmrui and the
standardized nifti_mrs format. In fact, most formats are imported into 
nifti_mrs format first using the pre-existing spec2nii toolbox.

For reading and writing to/from lcmodel formats, see io_lcmodel.py.
For reading and writing of RF pulses, see io_rf
"""

import argparse
import os
from pathlib import Path
import sys
import numpy as np
from spec2nii.GE.ge_read_pfile import Pfile
from pyFidA.fidA_common import FID, fid_from_specs, spec_from_fids

def io_loadjmrui(fname,affine=None):
    # It looks like there is a difference of a complex conjugate between what 
    # Matlab loads for the fid and what spec2nii loads. I've written 
    # io_readjmrui so that it should agree with Matlab but it's not called here.
    
    from spec2nii.jmrui import jmrui_format
    # Totally untested because I don't have data in this format.
    fname=Path(fname)
    args=make_args(file=fname,affine=affine,fout=None)
    imout,f_out=jmrui_format(args)
    out_fidA=nii_to_fidA(imout[0])
    return out_fidA

def io_loadspec_jmrui(fname,affine=None):
    # Just making this version so that it matches the other loadspec calls. 
    # Leaving io_loadjmrui for those more familiar with Matlab.
    out_fidA=io_loadjmrui(fname,affine=affine)
    return out_fidA

def io_readjmrui(fname):
    # This function lines up with the Matlab call to io_readjmrui, the result
    # of which is passed on to io_loadjmrui. It's not clear to me that you need
    # both functions (load basically creates FID object from the rf array that
    # this function produces by adding in the txfreq, etc info). The version
    # of io_loadjmrui that is in this module doesn't call this io_readjmrui 
    # function, in part because this file only deals with text files, whereas 
    # spec2nii has the capability to load .mrui files.
    from spec2nii.jmrui import readjMRUItxt
    # Note that, in the readjMRUItxt function, the conjugate of the data is taken
    # before return. This doesn't seem to happen in the Matlab code so may want
    # to double check.
    rf_jmrui,info_dict=readjMRUItxt(fname)
    # Actually, what Matlab returns is the complex data in the first column and
    # the frequency domain data in the second column. The assumption seems to
    # be that the data in rf_jmrui is 1D
    if rf_jmrui.squeeze().ndim!=1:
        print('WARNING: Size of original file is '+str(rf_jmrui.shape)+'. Reshaping to 1D with length'+str(len(rf_jmrui.T.flatten()))+'.')
        rf_jmrui=rf_jmrui.T.flatten()
    rf_jmrui=rf_jmrui.conj()
    rf_out=np.zeros([len(rf_jmrui,2)],dtype=np.complex)
    rf_out[:,0]=rf_jmrui.copy()
    rf_out[:,1]=spec_from_fids(rf_jmrui)
    return rf_out,info_dict

def io_loadspec_data(fname,list_name=None, aux_name=None,special_flag=None):
    # This is for Philips .data and .list files
    # Note that this differs quite a bit from Matlab, which has the user enter
    # sw,Larmor,subspecs and (optionally) te and tr as input arguments and does
    # not use an auxiliary file. The argument special_flag here can be 'hyper'
    # although this flag is also activated if 'hyper' is found in the ProtocolName
    # tag of the auxiliary (dicom) file
    from spec2nii.Philips.philips_data_list import read_data_list_pair
    
    fname=Path(fname)
    if list_name is None:
        list_name=Path.joinpath(fname.parent,fname.stem+'.list')
    # I'm not quite sure what to expect for aux_file. Matlab code seems to 
    # suggest that the .list file is the spar file (like, that's what the variable
    # name indicates even though it loads fname.list). Whereas spec2nii seems to
    # expect a third file that's either spar or dicom. I guess in part this is
    # because spec2nii uses the auxiliary file to get the dwelltime, which is
    # used to fill out parameters in the eventual nifti_mrs header. So I think
    # that the best thing is to allow the user to enter it, try fname.spar if it's
    # missing, then try fname.dcm. This is awkward because I have no idea if the
    # auxiliary file has the same stem as the data_name or not, so I'll just output
    # the text to note what's being tried.
    if aux_name is None:
        print('No auxiliary file name given. Trying '+fname.stem+'.spar')
        if Path.exists(Path.joinpath(fname.parent,fname.stem+'.spar')):
            aux_name=Path.joinpath(fname.parent,fname.stem+'.spar')
        else:
            print('No .spar file found as auxiliary. Trying dicom.')
            if Path.exists(Path.joinpath(fname.parent,fname.stem+'.dcm')):
                aux_name=Path.joinpath(fname.parent,fname.stem+'.dcm')
            else:
                print('No dicom file found with .dcm extension. Please enter aux_name explicitly in io_loadspec_data')
                raise FileNotFoundError()
    imout,f_out=read_data_list_pair(fname,list_name,aux_name,special_flag)
    print('{:d} items found in dataset.'.format(len(imout)))
    # It's not totally clear to me but I think imout here returns a list of
    # nifti_mrs objects that can potentially include the reference data. I'm going
    # to attempt to convert everything and then I'll return as a tuple so that
    # the user can unpack as they wish.
    out_ims=list()
    for fn,eachim in zip(f_out,imout):
        print('Converting {:s}'.format(fn))
        out_ims.append(nii_to_fidA(eachim))
    if len(out_ims)==1:
        return out_ims[0]
    else:
        return tuple(out_ims)

def io_loadspec_GE(fname):
    # Note that, with the test data, frequency spectrum seems to be reversed
    # Suggests complex conjugate issue.
    from spec2nii.GE.ge_pfile import _process_svs_pfile, _process_mrsi_pfile
    # The original read_pfile function that is downloaded with spec2nii creates
    # an error when the psdname in the header contains the full path name 
    # (eg. /usr/g/bin/repress7 as in the fidA exampleData for metapress, P21504.7)
    # There is a bit written in the code to correct this for gaba, but not for
    # other scan types (line 183 in the version of ge_read_pfile.py that I have, 
    # in Pfile.get_mapper() but there is similar code in ge_pfile's 
    # _process_svs_pfile). I have therefore created a class NewPfile (code in the 
    # fidA_common module) that inherits from Pfile but overwrites its read_header 
    # method by calling the original Pfile class's read_header but then looking 
    # at hdr.rhi_psdname and removing the extra bit of the path if it's there.
    # However, this means that you can use ge_read_pfile.read_pfile because
    # that is what creates the original Pfile object, whereas we want to create
    # a NewPfile object and then do the rest of the processing.

    fname=Path(fname)
    # Right now, fn_out is unused in terms of actually writing files but I do
    # use the default names created to provide info on what has been loaded.
    pfile=NewPfile(fname)
    if pfile.is_svs:
        data,fname_suffix=_process_svs_pfile(pfile)
    else:
        data, fname_suffix=_process_mrsi_pfile(pfile)

    fnames=[fname.stem+fns for fns in fname_suffix]
    print('{:d} items found in pfile.'.format(len(data)))
    # note that data is typically a list of len(2) for the metabolite (water
    # suppressed) and reference (water unsuppressed) data, but I am not sure if
    # there are cases where this could vary (no reference scan, multiple 
    # metabolite scans) so I've left it general and will just attempt to convert
    # every item in the list.
    out_ims=list()
    for fn,eachim in zip(fnames,data):
        # Could be worth having a try catch block here for nii_to_fidA but I'm
        # not sure what the most likely error type is if conversion fails.
        print('Converting {:s}'.format(fn))
        out_ims.append(nii_to_fidA(eachim))
    if len(out_ims)==1:
        return out_ims[0]
    else:
        return tuple(out_ims)
    
# Dealing with the GE pfile reader in spec2nii, which does not account for longer
# filenames in rhi_psdname
class NewPfile(Pfile):
    def read_header(self):
        Pfile.read_header(self)
        tmphdr=self.hdr
        psd=tmphdr.rhi_psdname.decode('utf-8').lower()
        # Note that you don't want to use os.sep because it's not the separator
        # on the file system where the code is being run but rather the separator
        # on the system where the pfile was made. Most likely this will be unix
        # (ie. '/') but could be Windows ('\\')
        if '/' in psd:
            print('WARNING: p-file header psd contains full path name, {:s}. Truncating to {:s}.'.format(psd,psd[psd.rfind('/')+1:]))
            psd = psd[psd.rfind('/')+1:]
        elif '\\' in psd:
            print('WARNING: p-file header psd contains full path name, {:s}. Truncating to {:s}.'.format(psd,psd[psd.rfind('\\')+1:]))
            psd = psd[psd.rfind('\\')+1:]
        tmphdr.rhi_psdname=psd.encode()
        self.hdr = tmphdr
    
def io_loadspec_IMA(fname,tag=None,fname_out=None,verbose=False):
    from spec2nii.Siemens.dicomfunctions import multi_file_dicom
    fname=Path(fname)
    # I don't know that I need all of this because we already know it's Siemens (this is just for the auto parser)
    # The main things that are done are to set tag to None and voi to False and 
    # voi is the tag that tells it whether to save a voi mask so I don't even think you need that.
    if fname.is_dir():
        files_in=sorted(fname.rglob('*.IMA')) + sorted(fname.rglob('*.ima'))+sorted(fname.rglob('*.dcm'))
        if len(files_in)==0:
            print('No .ima or .dcm files found. Searching for other file types.')
            files_in=sorted([fn for fn in fname.iterdir() if fn.is_file()])
    else:
        files_in=[fname]
    imout,f_out=multi_file_dicom(files_in,fname_out,tag,verbose)
    
    print('{:d} items found in pfile.'.format(len(imout)))
    out_ims=list()
    for fn,eachim in zip(f_out,imout):
        # Could be worth having a try catch block here for nii_to_fidA but I'm
        # not sure what the most likely error type is if conversion fails.
        print('Converting {:s}'.format(fn))
        out_ims.append(nii_to_fidA(eachim))
    if len(out_ims)==1:
        return out_ims[0]
    else:
        return tuple(out_ims)
    
def io_loadspec_niimrs(fname, validate_on_creation=True,**kwargs):
    # Not sure if other NIFTI_MRS options are needed. eg. affine transform? I
    # don't think that matters too much to me right now since I'm not set up
    # for MRSI but it would be good to make it easier to implement in the future
    # It looks like almost nothing else is used if the thing passed to NIFTI_MRS
    # is a filename. The one thing might be allowing the loading of a json
    # sidecar with loadMeta. Passing whatever is extra on through although I think
    # loadMeta is meant to be in args not kwargs so it might take testing whether
    # something with a json sidecar can be loaded
    from nifti_mrs.nifti_mrs import NIFTI_MRS
    out_nii=NIFTI_MRS(fname,validate_on_creation=validate_on_creation,**kwargs)
    out_im=nii_to_fidA(out_nii)
    return out_im

def io_loadspec_rda(rda_filename,fname_out=None,verbose=False):
    from spec2nii.Siemends.rda import convert_rda
    rda_filename=Path(rda_filename)
    out_nii,f_out=convert_rda(rda_filename.file,fname_out,verbose)
    # Pretty sure there is only one file expected here but convert_rda returns
    # two lists so might as well loop to extract that info just in case.
    out_ims=list()
    for fn,eachim in zip(f_out,out_nii):
        print('Converting {:s}'.format(fn))
        out_ims.append(nii_to_fidA(eachim))
    if len(out_ims)==1:
        return out_ims[0]
    else:
        print('WARNING: More than one FID object found. Returning all.')
        return tuple(out_ims)
    
def io_loadspec_sdat(fname,shape_mat=None,tags=["DIM_DYN",None,None],special_flag=None,fname_out=None):
    # Python has sdat, spar (although assumption seems to be that you only need one
    # filename and the spar/sdat stems should be the same), tags (optional, defaults ["DIM_DYN",None,None]), 
    # shape(optional, None), special (optional, None). Quite
    # similar to data/list format in some ways and to twix default in others.
    from spec2nii.Philips.philips import read_sdat_spar_pair
    fname=Path(fname)
    # I think the big question here is whether I need subspecs. shape_mat and/or
    # tags can be used to override the expected shape on loading. Presumably you
    # could allow the user to enter subspecs and infer shape_mat/tags from that
    # but this is less flexible. So I've removed the subspecs input argument that
    # you see in Matlab.
    # Note also that scans containing 'hyper' or 'hyper-ref' in the scan_id part
    # of the spar file will be loaded as those special cases even when special_flag
    # is set to None (at least that's how I read the code) but the special_flag
    # just gives an additional way to flag files that need special loading.
    # There doesn't seem to be any way to force a normal load if your scan_id
    # includes 'hyper' in it but presumably that's unlikely.
    if fname.with_suffix('.spar').isfile() and fname.with_suffix('.sdat').isfile():
        sdat_file=fname.with_suffix('.sdat')
        spar_file=fname.with_suffix('.spar')
    else:
        print('ERROR: Could not find matching sdat/spar pair with '+fname.name)
        raise FileNotFoundError()
    shape_mat=shape_mat
    special_flag=special_flag
    out_nii,f_out=read_sdat_spar_pair(sdat_file, spar_file,shape_mat,tags,fname_out,special_flag)
    print('{:d} items found in spar/sdat files.'.format(len(out_nii)))
    out_ims=list()
    for fn,eachim in zip(f_out,out_nii):
        print('Converting {:s}'.format(fn))
        out_ims.append(nii_to_fidA(eachim))
    if len(out_ims)==1:
        return out_ims[0]
    else:
        return tuple(out_ims)
    
def io_loadspec_twix(fname,dim_overrides=None,multiraid=None,quiet=False,verbose=False,remove_os=False,subspecs=None):
    # Note that I tried this on some of Peter's special data and it works but gives
    # you the subspecs as averages. Seems like this is how the scanner saves
    # the on/off data, along the averages dimension. So you need to make some
    # assumptions and then create a case for special data (see Matlab io_loadspectwix)
    # The same header format seems to exist ie. you want to check
    # twixObj[multiraid-1].hdr.Config.SequenceFileName to see if the various strings
    # are in there.
    # Alternatively, it may be possible to use dim_overrides to get the subspecs
    # separated out if you know the data format. However, It's not totally clear
    # to me that this will work without a data reshaping.
    
    # Don't think that there is a use case for the view mode so I'm omitting 
    # that argument and just using eval/process_twix.
    from mapvbvd import mapVBVD
    from spec2nii.Siemens.twixfunctions import process_twix
    
    # It seems like Matlab suggests that some sequences may have water
    # references in the file and others don't. It isn't clear to me whether
    # nifti_mrs deals with this in the same way (or at all?). It's possible 
    # that the imout returned is a list with more than one element and one is
    # the reference data
    fname=Path(fname)
    twixObj = mapVBVD(fname, quiet=quiet)
    # Dealing with multi-RAID files
    if isinstance(twixObj, list):
        if multiraid is None:
            print("WARNING: Multiraid file detected but no multiraid index given. Assuming last element is data of interest. If incorrect, enter a value for multiraid in io_loadspec_twix input arguments.")
            multiraid=len(twixObj)
        twixObj = twixObj[multiraid - 1]
    
    # One thing that's confusing about the dim_overrides is that they'll be the 
    # tags for nifti_mrs, not the dimension names for fidA. So I will want to
    # be clear about that in the help and/or create a dict to convert between
    # them (which I already have for nii_to_fidA)
    if dim_overrides is None:
        dim_overrides={'dims': (None,None,None),'tags': (None,None,None)}
    elif len(dim_overrides['dims'])>3:
        print('WARNING: Only three dimensions can be set via dim_overrides. Further dimensions not overriden')
    elif len(dim_overrides['dims'])<3: #Add on None so don't get an IndexError
        dim_overrides['dims']=tuple(list(dim_overrides['dims'])+[None]*(3-len(dim_overrides['dims'])))
        dim_overrides['tags']=tuple(list(dim_overrides['tags'])+[None]*(3-len(dim_overrides['tags'])))
    # Second input argument is the output filename. None will use the default
    imout, f_out = process_twix(twixObj, None, fname.name,'image',dim_overrides,quiet,verbose,remove_os)
    print('{:d} items found in pfile.'.format(len(imout)))
    # note that data is typically a list of len(2) for the metabolite (water
    # suppressed) and reference (water unsuppressed) data, but I am not sure if
    # there are cases where this could vary (no reference scan, multiple 
    # metabolite scans) so I've left it general and will just attempt to convert
    # every item in the list.
    out_ims=list()
    for fn,eachim in zip(f_out,imout):
        print('Converting {:s}'.format(fn))
        tmp_fid=nii_to_fidA(eachim)
        if subspecs is not None:
            if 'subspecs' not in tmp_fid:
                new_shape=list(tmp_fid.sz)
                if subspecs>1:
                    # So, this is okay if averages is the last dimension. But you could
                    # generalize for averages inthe middle
                    new_shape[tmp_fid.dims['averages']]=new_shape[tmp_fid.dims['averages']]//subspecs
                    new_shape=new_shape+[subspecs]
                    new_fid=np.reshape(tmp_fid.fids,new_shape)
                    tmp_fid.fids=new_fid
                    tmp_fid._dimlist=tmp_fid._dimlist+['subspecs']
                    #tmp_fid.averages and tmp_fid.subspecs will be automatically re-calculated because they are FID object properties
            else:
                print('WARNING: Subspecs already in dims. Ignoring user-defined subspecs')
        out_ims.append(tmp_fid)
    if len(out_ims)==1:
        return out_ims[0]
    else:
        return tuple(out_ims)
    
def io_loadspec_varian(fname,tag="DIM_DYN",dump_headers=True,fileout=None):
    # Note sure if dump_headers should be true or not
    from spec2nii.varian_importer import read_varian
    fname=Path(fname)
    args=make_args(file=fname,tag6=tag,dump_headers=dump_headers,fileout=fileout)
    out_nii,f_out=read_varian(args)
    print('{:d} items found in spar/sdat files.'.format(len(out_nii)))
    out_ims=list()
    for fn,eachim in zip(f_out,out_nii):
        print('Converting {:s}'.format(fn))
        out_ims.append(nii_to_fidA(eachim))
    if len(out_ims)==1:
        return out_ims[0]
    else:
        return tuple(out_ims)
    
#######################################################################
# I want to make the spec2nii bruker stuff somewhat available because it 
# probably deals with different verrsions of ParaVision better, but it needs 
# the adjustment to deal with raw data format. So some of the previous Bruker 
# loading options are still present here even though they partly cover cases
# that are also covered by spec2nii written.

def io_loadspec_bruk(fname, load_ref=False, do_leftshift=True, fill_truncated_data=True):
    """
    A lot of this is similar to what is done in bruker.py in the read_bruker() 
    function, but that did not have a case for loading raw data (which has only
    been tested for PV6.0.1) so I have added that in here. In addition, there
    were several things that were not working correctly for me or that I wanted
    options for. eg. read_bruker was returning 1980 points in my 2048 point file,
    which is because of Bruker having 68 points that you need to remove or leftshift
    but in the Matlab fidA code for Bruker, the leftshift is followed by a zero
    fill to bring back to the original 2048 size. So I've set that up as the 
    default here, although it can be changed.

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.
    fill_truncated_data : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    out1 : TYPE
        DESCRIPTION.

    """
    from brukerapi.dataset import Dataset
    import spec2nii.bruker as brkr
    from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext, gen_nifti_mrs
    # I'm not sure if this is the right thing to check. Some previous ParaVision
    # versions seem to use the naming convention fid.raw. However, brukerapi.dataset
    # seems to just use the file stem so I think it is better to stick with that convention
    # This version of the rawdata file is only tested on PV6.0.1. My understanding
    # is that PV5.1 does not save receive coil data separately, so this case is
    # set to 1 receiver in the override_path
    # It looks like, once I have a package set up, I will be able to do the 
    # importlib.resources thing below, but it will be my package (pyFidA.bruker_properties or whatever)
    # that I am looking for instead of spec2nii
    module_path=os.path.dirname(sys.modules[__name__].__file__)
    if 'rawdata' in str(fname):
        bruker_properties_path2=os.path.join(module_path,'..','bruker_properties','bruker_properties_rawdata_CB.json')
        bruker_override_path2 = os.path.join(module_path,'..','bruker_properties','bruker_rawdata_override_CB.json')
    else:
        bruker_properties_path2=os.path.join(module_path,'..','bruker_properties','bruker_properties_CB.json')
        bruker_override_path2 = os.path.join(module_path,'..','bruker_properties','bruker_fid_override_CB.json')
        
    
    d2=Dataset(fname,property_files=[bruker_override_path2,bruker_properties_path2],parameter_files=['method'])
    # For whatever reason, I can't get d2 into the correct shape for the rawdata case using the parameters
    # in the json. Seems easier to just do it afterward. It loads with shape_storage but this just seems
    # based on block_size and block_count.
    # Looks like maybe I fixed this at some point but didn't note it down??
    # The data size info is in the schemas.py file and class SchemaRawdata has
    # layout job_desc[0]/2, channels, job_desc[3]
    # I think job_desc is read in from properties_rawdata_core.json in the brukerapi.config
    # folder. I don't seem to have overwritten it in my own. And the exact way
    # to read this in seems to vary with ParaVision type.
    # It appears that the rawdata type in schemas.py does not utilize the permute
    # variable/operation, so I could permute manually to put the channels last,
    # which seems to be what Jamie suggests for ordering in Matlab. However,
    # the GE code already puts coils before averages so maybe it doesn't matter.
    # I've commented out for now so that it remains [t,coils,averages] but, if
    # I change, I'll have to remember to change dim_type in bruker_rawdata_override_CB.json
    #if d2.type=='rawdata':
        #d2.data=np.transpose(np.reshape(d2.data,[d2.channels,-1,d2.block_count]),[1,2,0])
        #d2.data=np.transpose(d2.data,[0,2,1])
        # Potentially I should remove singleton dimensions here and adjust dim_type
        # to match, but I'm also rethinking that whole thing right now.
    # Then this is basically copied from spec2nii.bruker._proc_dataset, but that requires args to set
    # the output filename. Also, this gives me flexibility to set some parameters for rawdata
    # since spec2nii is only set up to read 2dseq and fid, really.
    # merge 2dseq complex frame group if present
    # I've added in the part for when the 2dseq data is not complex (which seems
    # to be the case for svs in PV6.0.1. Not sure if 2dseq is really only 
    # intended for mrsi maybe?)
    if d2.is_complex and d2.type == '2dseq':
        d2 = brkr.FrameGroupMerger().merge(d2, 'FG_COMPLEX')
    # actually, if it's a 2dseq for spectroscopic data, you are loading the spectral
    # data, not the fid. so you need to fft. It will then be complex
    elif not d2.is_complex and d2.type=='2dseq':
        d2.data=fid_from_specs(np.flipud(d2.data))
        d2.is_complex=True
    # Bruker raw and fid data have "junk" at the start that needs to be cut out
    # In Matlab's io_loadspec_bruk for fidA, the data are zero-filled back to 
    # the original data size. That is the default behaviour here, but both the
    # leftshift and zero fill can be left out. Note that we do not set the flag
    # for leftshift to True because that is intended for processing operations
    # for first order phase correction, not the standard data correction.
    if (d2.type=='rawdata' or d2.type=='fid') and do_leftshift:
        # Remove points acquired before echo
        d2.data = d2.data[d2.points_prior_to_echo:, ...]
        if fill_truncated_data:
            pad_vals=[[0,d2.points_prior_to_echo] if dct==0 else [0,0] for dct in range(len(d2.data.shape))]
            d2.data=np.pad(d2.data, pad_width=pad_vals)
    if d2.is_svs:
        data=d2.data
        # This can probably done more neatly with np.newaxis
        data = np.expand_dims(np.expand_dims(np.expand_dims(data, axis=0), axis=0), axis=0)
    elif d2.is_mrsi:
        data=d2.data
        # push the spectral dimension to position 2
        data = np.moveaxis(data, 0, 2)
        # add empty dimensions to push the spectral dimension to the 3rd index
        data = np.expand_dims(data, axis=2)
    else:
        data = d2.data
    
    # get properties
    properties = d2.to_dict()
    # Orientation information
    if d2.type == 'fid':
        orientation = brkr.NIFTIOrient(brkr._fid_affine_from_params(d2))
    # added this in to match fid type, but not sure if that is correct. Should only matter for mrsi?
    elif d2.type == 'rawdata':
        orientation = brkr.NIFTIOrient(brkr._fid_affine_from_params(d2))
    else:
        orientation = brkr.NIFTIOrient(np.reshape(np.array(properties['affine']), (4, 4)))
    # Meta data. Setting dump=True, whereas it was a command line argument in the original
    if d2.type == 'fid':
        meta = brkr._fid_meta(d2, dump=True)
    if d2.type == 'rawdata':
        meta = brkr._fid_meta(d2, dump=True)
    else:
        meta = brkr._2dseq_meta(d2, dump=True)
    # Dwelltime - original code in brukerapi's _proc_dataset call had a factor of 2
    # to resolve because, for some reason, dwell_s was being calculated as 1/sw_h/2
    # But I changed so that dwell_s is 1/sw_h, so you don't need the factor 2 that
    # was inexplicably in there (maybe it applies for non-spectroscopic data or
    # other versions of ParaVision???)
    dwelltime = d2.dwell_s
    # And then we do the stuff to make nii
    im1=gen_nifti_mrs_hdr_ext(data, dwelltime, meta, orientation.Q44, no_conj=True)
    # Annoyingly, fsl.data.image seems to remove singleton dimensions (other than x,y,z)
    # from data but doesn't remove them from the header. Doing it manually here. I'm
    # assuming that these dimensions would be the ones at the end. (It might also be 
    # possible to do via error-checking in nii_to_fidA but the issue seems to be with 
    # the Bruker loading for fid files and maybe reference data (rawdata is okay) so 
    # I'm putting it here. Can move to nii_to_fidA if it's an issue for other file types)
    if len(im1.image.shape)<np.max(im1._hdr_ext.ndim):
        for idx in range(len(im1.image.shape),im1._hdr_ext.ndim):
            im1._hdr_ext.remove_dim_info(idx-4)
    # Then we convert nifti_mrs to fidA
    out1=nii_to_fidA(im1)
    if d2.type=='rawdata' and d2.channels==1:
        if 'coils' in out1:
            out1._dimlist.remove('coils')
        if 'averages' not in out1:
            out1._dimlist.insert(1,'averages')
    # Then I need to do this for the refscan file. OR could just leave to the user
    # since your input argument is a file, not a directory (as it is in Matlab)
    return out1

def io_loadspec_brukNMR(fname,spectrometer=True,try_raw=False,return_info_dict=False,ADC_OFFSET=None):
    # Note that this is only set up to read single voxel data
    # In the case where spectrometer=True, the value of try_raw doesn't really
    # matter. At least for TopSpin 3.2, the data is contained in the fid file
    # for 1D and the ser file for 2D. There is no other file with the fids, even
    # if you have multiple averages. This is because of how TopSpin writes from
    # the buffer to the file, just adding onto what's there rather than appending
    # new data. So there is no "raw" data saved separately unless you actually
    # alter the pulse program to save individual averages as a pseudo-2D file.
    # Basically, I think that it's fair to say that fid isn't really a "raw" 
    # file because it contains the combined averages (same as it does in PV6.0.1)
    # but it's also the only file that there is.
    # Also, if you enter the full filename, then whether it's a raw file or not
    # will determine try_raw
    from brukerapi.jcampdx import JCAMPDX
    fname=Path(fname)
    if fname.is_dir():
        inDir=fname
        # Need to go through and add in the cases from below (if try_raw, etc.)
        # First try to load fid.raw file. If that does not work, use regular fid
        if try_raw:
            # fid is last so that, if nothing else is found, it loads this, even 
            # if it's averaged. Could create problems for n_av calculation
            for fname_try in ['ser','rawdata.job0','fid.raw','fid']:
                if inDir.joinpath(fname_try).is_file():
                    fname=inDir.joinpath(fname_try)
                    break
        else:
            for fname_try in ['fid','ser']: # including ser even though it's in raw files above because it's the format for 2D regardless (unless you load the spectral data, which is done with io_loadspec_irBruk)
                if inDir.joinpath(fname_try).is_file():
                    fname=inDir.joinpath(fname_try)
                    break
    else:
        inDir=fname.parent
        if fname.name in ['rawdata.job0','fid.raw']:
            try_raw=True
        else:
            try_raw=False
        # Can just leave fname as the full file name for later
    # if we made it through the loop and still don't have a file, raise error
    if not fname.is_file():
        print('No raw file found in {:s}. Please enter full filename for input fname.'.format(str(inDir)))
        raise FileNotFoundError()
        
    # Now try to load the data
    fid_data=np.fromfile(fname,dtype=np.int32)
    # sometimes getting a 90 degree phase difference, which suggests these
    # were backwards. Going to swap them and see
    real_fid = fid_data[::2]
    imag_fid = fid_data[1::2]
    fids_raw=real_fid+1j*imag_fid
        
    if spectrometer:
        if ADC_OFFSET is None:
            ADC_OFFSET=68
        jc1=JCAMPDX(os.path.join(inDir,'acqu'))
        dic1=jc1.get_parameters()
        spectralwidth=dic1['SW_h'].value
        txfrq=dic1['SFO1'].value*1e6
        offset=dic1['O1'].value/dic1['SFO1'].value
        whichnuc=[dic1['NUC1'].value[1:-1]]
        te=-1
        # Assumption is the D1 is your TR, which is true for basic Bruker pulse
        # programs. The D array doesn't seem to convert correctly and is 
        # throwing an error when calling .values, but you can get at D1 by 
        # calling val_str and dividing it up, then converting to float
        dlist=dic1['D'].val_str
        tr=float(dlist.split()[1])*1000
        sequence=dic1['PULPROG'].value[1:-1]
        info_dict=dic1
        coil_dim=-1; ncoil=1
        # We can find npts from TD/2 and then everything else is put into the 
        # second dimension, even if it's 3D NMR, etc. This second dimension is
        # assumed to be the averages if TD0>1 (for some pseudo-2D sequences that
        # I wrote to save chunks of data separately) and is put in to "extras"
        # for all other cases. There is no way to know what the extra dimensions
        # refer to because it just depends on the pulse program and variables
        # used in it so users will have to manually reshape the fid and assign
        # the dims if they don't want things in "extras"
        npts=int(dic1['TD'].value/2)
        n_av=dic1['TD0'].value
        # Otherwise, if they don't look like averages, stick the extra dimensions
        # in the "extras"
        n_extra=int(len(fids_raw)/npts/ncoil/n_av)
    else:
        jc1=JCAMPDX(os.path.join(inDir,'method'))
        dic_method=jc1.get_parameters()
        # Note that PV6.0.1 at least seems to have an acqu file that you could
        # read in if you wanted, but you would still want to read in the method
        # file to get TE and TR.
        jc2=JCAMPDX(os.path.join(inDir,'acqp'))
        dic_acqp=jc2.get_parameters()
        # I believe all method parameters start with PVM except a few (overlap
        # in one I checked was TITLE, JCAMPDX, DATATYPE, ORIGIN, and OWNER, where
        # values match between the two dicts/files) so it should be okay to 
        # merge the dictionaries via unpacking
        info_dict={**dic_acqp,**dic_method}
        if ADC_OFFSET is None:
            ADC_OFFSET=dic_method['PVM_DigShift'].value
        spectralwidth=dic_method['PVM_SpecSWH'].value
        txfrq=dic_method['PVM_FrqRef'].value[0]*1e6#dic_acqp['BF1'].value*1e6
        whichnuc=[dic_method['PVM_Nucleus1Enum'].value[1:-1]]#[dic_acqp['NUC1'][1:-1]]
        # In MRI case, the water peak is centered and O1 is 0. However, there is an offset that appears to be stored in PVM_FrqWorkPpm
        offset=dic_method['PVM_FrqWorkPpm'].value[0]
        te=dic_method['PVM_EchoTime'].value
        tr=dic_method['PVM_RepetitionTime'].value
        sequence=dic_method['Method'].value
        # If you've loaded the fid rather than the raw file, then the fid should
        # only have length npts with no other dimensions
        npts=dic_method['PVM_SpecMatrix'].value
        if try_raw:
            # At least for PV6.0.1, the order seems to be t, averages, coils; but I'm transposing to get t, coils, averages, extras. Will assume anything else is at the end.
            ncoil=dic_method['PVM_EncNReceivers'].value
            n_av=dic_method['PVM_NAverages'].value
        # Dimensions should agree but, if not, fids_raw will be reshaped along "extras"
        n_extra=int(len(fids_raw)/npts/ncoil/n_av)
        if n_extra>1:
            print('WARNING: Unexpected dimensions found. Dimensions may be incorrectly assinged')
    # Truncate/leftshift and then reshape
    fids_raw=np.reshape(fids_raw,[-1,npts]).T
    fids_trunc=fids_raw[ADC_OFFSET:,...]
    padw=np.zeros([fids_trunc.ndim,2],dtype=int)
    padw[0,1]=ADC_OFFSET
    fids=np.pad(fids_trunc, pad_width=padw)
    fids=np.squeeze(np.transpose(np.reshape(fids,[npts,n_av,ncoil,n_extra]),[0,2,1,3]))
    dimlist=[dimnm for dimnm,dimsz in zip(['t','averages','coils','extras'],[npts,n_av,ncoil,n_extra]) if dimsz>1]
    fid1=FID(fids,spectralwidth,txfrq,te,tr,sequence,dims=dimlist,nucleus=whichnuc,center_freq_ppm=offset)
    
    # NOW TRY LOADING IN THE REFERENCE SCAN DATA (IF IT EXISTS)
    # If it's a spectrometer, the reference is a separate scan in its own folder
    # that needs to be loaded separately in its own call to io_loadspec_brukerNMR
    # with that filename
    if spectrometer or dic_method['PVM_RefScanYN'].value.lower()=='no':
        reffid=0
    else:
        # Note that there is a PVM_RefScanPC parameter that I think might be a
        # phase correcton vector
        # I actually think fid.ref might be the Navigator scans and maybe 
        # shouldn't be here???
        for fname_try in ['fid.refscan','fid.ref']:
            if inDir.joinpath(fname_try).is_file():
                refname=inDir.joinpath(fname_try)
                break
        if refname.is_file():
            # load file directly
            all_fid=np.fromfile(refname,dtype=np.int32)
            real_fid = all_fid[::2]
            imag_fid = all_fid[1::2]
            # This data seems to be coil-combined whereas, in the PVM_RefScan from
            # the method file, it's not for PV6.0.1. However, my recollection is
            # that storing the refscan in the method file is a thing from PV5 and
            # that there the coil-uncombined info is not available so maybe that
            # doesn't matter?? However, this does leave the question of how averages
            # are stored. I suspect, in the file, the averaging is already done.
            # Will run a final check and assign to n_extra for now
            n_av=1; ncoil=1
        else:
            # if no file, try to load from the PVM_RefScan variable in the 
            # methods file (I believe this is the only option in PV5 although
            # the variable still exists in PV6.0.1). In PV6.0.1, the data appear
            # to have shape [coil, 2*t]. Presumably there would also be individual
            # averages if PVM_RefScanNA>1. However, I seem to recall that PV5
            # does not save coil data separately so I assume that ncoil=1 above.
            # And if ncoil!=1 then presumably you have separate coil data for
            # the reference scan as well? I've tried to set the code up to
            # work in either case.
            all_fid=dic_method['PVM_RefScan'].value
            # NOTE: Will have to change this if averages is last dimension
            real_fid = all_fid[...,::2]
            imag_fid = all_fid[...,1::2]
            n_av=dic_method['PVM_RefScanNA'].value
            ncoil=1
        ref_raw=np.reshape(real_fid+1j*imag_fid,[-1,npts]).T
        n_extra=int(len(ref_raw)/npts/ncoil/n_av)
        if n_extra>1:
            print('WARNING: Unexpected dimensions found for reference scan. Dimensions may be incorrectly assinged')
        ref_trunc=ref_raw[ADC_OFFSET:,...]
        # Can use the same padw as before
        fids2=np.pad(ref_trunc, pad_width=padw)
        fids2=np.squeeze(np.transpose(np.reshape(fids2,[npts,n_av,ncoil,n_extra]),[0,2,1,3]))
        dimlist=[dimnm for dimnm,dimsz in zip(['t','averages','coils','extras'],[npts,n_av,ncoil,n_extra]) if dimsz>1]
        reffid=FID(fids2,spectralwidth,txfrq,te,tr,sequence,dims=dimlist,nucleus=whichnuc,center_freq_ppm=offset)
    
    # I haven't accounted for navigators in the raw files and not sure if that
    # might cause an issue. (You can probably add an amount onto leftshift in
    # some cases but not if the navigator doesn't occur for every fid)
    return_vals=[fid1]
    if reffid!=0:
        return_vals=return_vals+[reffid]
    if return_info_dict:
        return_vals=return_vals+[info_dict]
    if len(return_vals)==1:
        return return_vals[0]
    else:
        return tuple(return_vals)

def io_loadspec_irBruk(fname,return_info_dict=False):
    # Note that this is only set up to read single voxel data. No 2D option to
    # read in 2rr files at the moment.
    from brukerapi.jcampdx import JCAMPDX
    
    fname=Path(fname)
    if fname.is_dir():
        fileDir=fname
        if fileDir.joinpath('pdata','1','1r').is_file():
            fname=fileDir.joinpath('pdata','1','1r')
            # top level directory containing method/acqu files
            topDir=fileDir
            # directory containing 1i/1r files. Assume folder 1 if input directory
            # is top-level directory for scan
            fileDir=topDir.joinpath('pdata','1')
        else:
            fname=fileDir.joinpath('1r')
            topDir=fileDir.parent.parent
    else:
        fileDir=fname.parent
        topDir=fileDir.parent.parent
        # Can just leave fname as the full file name for later
    # if we made it through the loop and still don't have a file, raise error
    if not fname.is_file():
        print('No 1r file found in {:s}. Please enter full filename for input fname.'.format(str(fileDir)))
        raise FileNotFoundError()
        
    # Now try to load the data
    # I followed the recon in Matlab with -1j
    real_dat = np.fromfile(fileDir.joinpath('1r'),dtype=np.int32)
    imag_dat = np.fromfile(fileDir.joinpath('1i'),dtype=np.int32)
    specs_raw=real_dat-1j*imag_dat
    fid_dat=fid_from_specs(specs_raw)
        
    # From Matlab, parameters are read in from method and acqp but, in theory, 
    # some parameters could be changed for recon. eg. could do zero-padding and
    # this changes the number of points. In this case, I think all fids will be
    # 1D so can jsut use length of fid_data for npts but there may be options 
    # to move the center frequency for display. Better to read numbers in from 
    # procs where available, but things like the TE, TR and sequence name still 
    # need to be read in from files with acquisition parameters.
    # NOTE: Matlab reads in averages from the methods file and sets raw_Averages
    # to that values and averages to 1. Not sure what the point of that is. I
    # think, in some cases, rawAverages was being used to contain both on/off
    # editing repeats and averages, which were separated out later. I'm not sure 
    # rawAverages is actually used for anything. (I see in the notes for Matlab
    # io_loadspec_GE is that it is specificed that rawAverages is the original
    # averages and won't change whereas averages could, so I could see making 
    # them distinct but it's more for info)
    # It's maybe a worthwhile thing that I could read in the averages or NS
    # parameters and set rawAverages to that when I load a fid file or 1i/1r 
    # that are already averaged. But you could also get this info on the actual
    # number of averages that contributed to the spectrum from from FID.nii_mrs['hdr_ext']
    # (except that of course I haven't created the nii_mrs header or header extension
    # here, but I can look at that)
    jc1=JCAMPDX(os.path.join(fileDir,'procs'))
    dic_proc=jc1.get_parameters()
    spectralwidth=dic_proc['SW_p'].value
    txfrq=dic_proc['SF'].value*1e6
    whichnuc=[dic_proc['AXNUC'].value[1:-1]]
    npts=dic_proc['SI'].value
    # Dimensions should agree but, if not, fid_dat will be reshaped along "extras"
    n_extra=int(len(fid_dat)/npts)
    if n_extra>1:
        print('WARNING: Unexpected dimensions found. Dimensions may be incorrectly assinged')
        fid_dat=np.reshape(fid_dat,[npts,n_extra])
        dimlist=['t','extras']
    else:
        dimlist=['t']
    # instead of having spectrometer as an input argument, I'm going to try to
    # open the method file. If there is no method file (as in spectrometer/
    # TopSpin case), I'll use acqu to get parameters instead (and assume te DNE)
    try:
        jc2=JCAMPDX(os.path.join(topDir,'method'))
        spectrometer=False
    except FileNotFoundError:
        jc2=JCAMPDX(os.path.join(topDir,'acqu'))
        spectrometer=True
    if spectrometer:
        dic_acq=jc2.get_parameters()
        offset=dic_acq['O1'].value/dic_acq['SFO1'].value
        te=-1
        dlist=dic_acq['D'].val_str
        tr=float(dlist.split()[1])*1000
        sequence=dic_acq['PULPROG'].value[1:-1]
    else:
        jc2=JCAMPDX(os.path.join(topDir,'method'))
        dic_acq=jc2.get_parameters()
        # In MRI case, the water peak is centered and O1 is 0. However, there is an offset that appears to be stored in PVM_FrqWorkPpm
        offset=dic_acq['PVM_FrqWorkPpm'].value[0]
        te=dic_acq['PVM_EchoTime'].value
        tr=dic_acq['PVM_RepetitionTime'].value
        sequence=dic_acq['Method'].value
    info_dict={**dic_acq,**dic_proc}
    fid1=FID(fid_dat,spectralwidth,txfrq,te,tr,sequence,dims=dimlist,nucleus=whichnuc,center_freq_ppm=offset)
    
    # Getting rid of reference data since there is no ir data for the reference spectrum afaik
    if return_info_dict:
        return fid1,info_dict
    else:
        return fid1

def io_writejmrui(indat,outfile,scanner='TrioTim',addinfo=''):
    io_writespec_jmrui(indat,outfile,scanner=scanner,addinfo=addinfo)
    
def io_writespec_jmrui(indat,outfile,scanner='TrioTim',addinfo=''):
    # This is copied from Matlab. Note that spec2nii seems to suggest that there
    # are two possible jmrui formats: one for .txt and one for .mrui. The reader
    # for things ending .mrui seems to expect a 13-line header but I don't think 
    # those fields in spec2nii.jmrui line up with the ones listed here from the 
    # Matlab file. Maybe the assumption is that you are writing a .txt file (
    # since that's the only one Matlab reads in) and so the header may line up 
    # with that but I haven't checked.
    datasets=1
    RF=np.zeros([indat.sz[0],4])
    RF[:,0]=np.real(indat.fids)
    RF[:,1]=np.imag(indat.fids)
    RF[:,2]=np.real(indat.specs)
    RF[:,3]=np.imag(indat.specs)
    header=['jMRUI Data Textfile']
    header.append('\n\nFilename: {:s}'.format(outfile))
    header.append('\n\nPointsInDataset: {:d}'.format(RF.shape[0]))
    header.append('\nDatasetsInFile: {:d}'.format(datasets))
    header.append('\nSamplingInterval: {:4.6E}'.format(indat.dwelltime*1000))
    header.append('\nZeroOrderPhase: {:1.0E}'.format(0))
    header.append('\nBeginTime: {:1.0E}'.format(0))
    header.append('\nTransmitterFrequency: {:4.6E}'.format(indat.txfreq))
    header.append('\nMagneticField: {:4.6E}'.format(indat.Bo))
    # copied from Matlab. Should this be indat.nucleus??
    header.append('\nTypeOfNucleus: {:1.0E}'.format(0))
    header.append('\nNameOfPatient: {:s}'.format('No Name'))
    header.append('\nDateOfExperiment: {:%Y-%m-%d}'.format(indat.date))
    header.append('\nSpectrometer: {:s}'.format(scanner))
    header.append('\nAdditionalInfo: {:s}\n\n\n'.format(addinfo))
    header.append('Signal and FFT\n')
    header.append('sig(real)\tsig(imag)\tfft(real)\tfft(imag)\n')
    header.append('Signal 1 out of {:d} in file\n'.format(datasets))
    with open(outfile,'w') as f:
        f.writelines(header)
        # note that Matlab writes RF' and the ' operator is the conjugate 
        # transpose. Since I'm writing here by row, I don't think I need to transpose
        # but I would need to conjugate to match Matlab (I wonder if this is why 
        # the conj issues for lcm files??). Currently not doing conjugate because 
        # it seems to result in some files not being read correctly
        for eachline in RF.conj():
            f.write('{:1.8E}\t{:1.8E}\t{:1.8E}\t{:1.8E}\t\n'.format(*eachline))
            
def io_writespec_niimrs(indat,outfile,write_json=False,dim_tags=None):
    # Inverse of io_loadspec_niimrs.
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    import json
    outfile=Path(outfile)
    # There's probably a better way to do this but, currently, there are far
    # more NIFTI tags for dimensions than fid-A tags (eg. indirect dimensions,
    # various tags for different subspectra, user-defined tags, etc.). When 
    # imported, things like indirect or user-defined dimensions are all currently
    # assigned to extras. Here, I've arbitrarily chosen to assign back to DIM_USER_0
    # since there's no way to know what these are (but you could keep the nifti_mrs
    # tags now that I've generalized a lot of these functions. That is, they 
    # should deal with arbitrary tags by just applying operations across all of that
    # dimension). Then, the only info lost is whether subspecs is DIM_EDIT vs
    # DIM_ISIS, which MIGHT be obtainable from isFourSteps???. However, note that
    # dim_tags can be entered manually to override on save. This is only for
    # automatic detection. And, if you're just using this to store FID objects
    # and read them back in later, this should work.
    reverse_dimtag_dict={'coils':'DIM_COIL','averages':'DIM_DYN','subspecs':'DIM_EDIT','extras':'DIM_USER_0'}
    
    if indat.isMRSI:
        raise TypeError('ERROR: Writing for MRSI data not yet implemented')
    # dim_tags can be entered manually
    if dim_tags is None:
        dim_tags=[None,None,None]
        # Will need to add a check in here for MRSI at some point
        if indat.ndim>1:
            for dimct,eachdim in enumerate(indat._dimlist[1:]):
                dim_tags[dimct]=reverse_dimtag_dict[eachdim]
    newdat=np.expand_dims(indat.fids,axis=[0,1,2])
    nii_obj=gen_nifti_mrs(newdat,indat.dwelltime,indat.txfreq/1e6,nucleus=indat.nucleus[0],dim_tags=dim_tags)
    if indat.te is not None:
        nii_obj.hdr_ext.set_standard_def('EchoTime',indat.te/1000)
    if indat.tr is not None:
        nii_obj.hdr_ext.set_standard_def('RepetitionTime',indat.tr/1000)
    if indat.sequence!='None':
        nii_obj.hdr_ext.set_standard_def('SequenceName',indat.sequence)
    nii_obj.save(outfile)
    if write_json:
        json_file=outfile.parent.joinpath(outfile.stem+'.json')
        with open(json_file,'w') as f:
            json.dump(json.loads(nii_obj.header.extensions[0].get_content()),f,indent=4)
    
def make_args(**kwargs):
    parser1=argparse.ArgumentParser()
    for kw in kwargs.keys():
        parser1.add_argument('-'+kw)
    parser1.set_defaults(**kwargs)
    args=parser1.parse_args()
    return args

def nii_to_fidA(out1):
    # It looks like the data are in out1.image.data and dimensions are 
    # [cols,rows,slices,numTimePts, numCoils,numSpecPts] where row,cols,slice are all 1 for SVS data
    hdr=out1.header
    hdr_ext=out1.hdr_ext.to_dict()
    fids=out1.image.data
    dt=out1.dwelltime
    sw=1/dt
    # nifti_mrs data blocks are up-to-7-dimensional blocks of complex floating
    # point numbers. Dimensions 1-4 are compulsory as the 3 spatial dimensions
    # (x,y,z) and the spectral time domain. Dimensions 5-6 are optional and have
    # variable definition so will need to be read from the header. In addition,
    # these will need to be translated to fidA dimension names, but there are 
    # more nifti names than (current) fid-A names.
    nifti_to_fidA_dimnames={'DIM_COIL':'coils','DIM_DYN':'averages','DIM_EDIT':'subspecs','DIM_ISIS':'subspecs','DIM_INDIRECT_0':'extras','DIM_INDIRECT_1':'extras','DIM_INDIRECT_2':'extras','DIM_PHASE_CYCLE':'extras','DIM_MEAS':'extras','DIM_USER_0':'extras','DIM_USER_1':'extras','DIM_USER_2':'extras'}
    dimlist=['x','y','z','t']
    full_dimorder=['t','coils','averages','subspecs','extras','x','y','z']
    for dimval in out1.dim_tags:
        if dimval is not None:
            dimlist.append(nifti_to_fidA_dimnames[dimval])
    # Note that this check for singleton dimensions deals with the SVS case 
    # (since x,y,z are compulsory and are included above and are in fids before
    # squeeze is called). But there is no warning if you try to load MRSI data
    # even though pyFidA isn't yet set up for that.
    final_dimlist=[dimnm for dimnm in full_dimorder if dimnm in dimlist and fids.shape[dimlist.index(dimnm)]>1]
    fids=np.squeeze(fids)
    if 'EchoTime' in hdr_ext:
        te=hdr_ext['EchoTime']*1000
    else:
        te=None
    if 'RepetitionTime' in hdr_ext:
        tr=hdr_ext['RepetitionTime']*1000
    else:
        tr=None
    if 'SequenceName' in hdr_ext:
        seq_name=hdr_ext['SequenceName']
    else:
        seq_name='None'
    outnii=FID(fids,sw,out1.spectrometer_frequency[0]*1e6,te,tr,seq_name,dims=final_dimlist,hdr=hdr,hdr_ext=hdr_ext,nucleus=out1.nucleus)
    return outnii
    
if __name__=='__main__':
    pass