# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:13:32 2022


fidA_display.py
Colleen Bailey, Sunnybrook Research Institute

A set of convenience functions for displaying fidA data, basis sets, etc. This
module is optional and requires seaborn and pandas to run some functions. 

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyFidA import fidA_io as fio
import seaborn as sns
import functools
import os

# Note that io_readlcmcoord is in the fidA_io module. This module is for display
# functions only

def make_df_from_flist(flist,fcontains='',fsuff='.csv',add_cols=None,pname='.',construct_cols=None,idcol='fct',idpre=''):
    """
    Create a pandas DataFrame from a list of folders containing LCModel .csv or .coord 
    outputs. Each LCModel file contains the concentrations and % SD for all
    metbolites for one dataset, so they need to be concatenated together. Additional
    and constructed columns can be added as outlined below, as well as setting
    the index column.
    
    eg. df_pre=make_df_from_flist(flist,fcontains='20240916_pre',pname='lcm-out',add_cols={'timept':'pre'},construct_cols={'id':['fct','timept']},idcol='id',idpre='Rat')
    will create a DataFrame from all of the folders in flist present in the 'lcm-out'
    directory if those folders have .csv files containing the text '20240916_pre'.
    A column for the treatment timepoint 'timept' is added with value 'pre' and
    an 'id' column is created with the prefix "Rat", the count of the file within
    flist and the timept 'pre', which is then set as the index.
    
    Initially, I had planned to allow flist to be lists of lists that could be 
    iterated through with different timepoints or other column values added but, 
    given the variety of use cases, I think it is best to let the user do this 
    and concatenate for their particular data structure.
    eg. df_pre=make_df_from_flist(flist,fcontains='20240916_pre',pname='lcm-out',add_cols={'timept':'pre'},construct_cols={'id':['fct','timept']},idcol='id',idpre='Rat')
        df_post=make_df_from_flist(flist,fcontains='20240916_post',pname='lcm-out',add_cols={'timept':'post'},construct_cols={'id':['fct','timept']},idcol='id',idpre='Rat')
        df_full=pd.concat([df_pre,df_post],axis=0)

    Parameters
    ----------
    flist : list
        The list of folders containing the files to concatenate. Typically,
        one folder is for one animal or dataset.
    fcontains : str, optional
        The string to use to identify the correct file in cases where multiple
        files with the same file extension are present in the same folder. This 
        assumes that the string is unique, or at least that the file you want 
        will be the first instance returned. The default is '', which will return 
        all files that end in fsuff in a list and use the first.
    fsuff : str, optional
        The string found at the end of the file, likely the file extension.
        Initially, this was set up for the LCModel .csv files, but it can also 
        be used for .coord files that contain the SNR, etc. The default is '.csv'.
    add_cols : dict, optional
        A dictionary of column names and values to add to the DataFrame where 
        each column has the same value for every item in flist. The default is None.
    pname : directory path, optional
        Path to location of flist directories to be opened. The default is '.', 
        which uses the current working directory.
    construct_cols : dict, optional
        Each key is the name of a new column to be constructed. Each value is a
        list of existing column names to be concatenated together using '_'.join,
        meaning that the columns must be strings. Useful for constructing a
        column to make the index column since each individual csv file loads with
        and index of '0' and retains this by default when concatenated. The columns
        'fname' and 'fct' containing the values of flist and the position of each
        item within it are automatically constructed and are unique identifiers
        but may be inconvenient for being too long (fname) or not descriptive enough
        (fct). The default is None.
    idcol : str, optional
        String corresponding to the column name to use as the index. The default 
        is 'fct'.
    idpre : str, optional
        A string that can be pre-pended to the value of idcol. Useful if one wants
        the indexes to read 'Mouse_1','Mouse_2',etc. The default is ''.

    Returns
    -------
    df1 : Pandas DataFrame
        The concatenated DataFrame with len(flist) rows, containing the information
        from each file in flist.

    """
    df1=pd.DataFrame()
    for fct,ftmp in enumerate(flist):
        fname=[fn for fn in os.listdir(os.path.join(pname,ftmp)) if fcontains in fn and fn.endswith(fsuff)]
        if fsuff=='.csv':
            try:
                dftmp=pd.concat([pd.Series(str(fct+1),name='fct'),pd.read_csv(os.path.join(pname,ftmp,fname[0])),pd.Series(ftmp,name='fname')],axis=1)
                df1=pd.concat([df1,dftmp],axis=0)
            except IndexError:
                print('No file with '+fcontains+' found in '+ftmp)
        elif fsuff=='.coord':
            try:
                kvals=['fct','fname','SNR','FWHM','shift','phase']
                lcm_dict,info_dict=fio.io_readlcmcoord(os.path.join(pname,ftmp,fname[0]))
                info_dict.update({'fct':str(fct+1),'fname':fname[0]})
                # items in dict need to be list for DataFrame.from_dict to work
                dftmp=pd.DataFrame.from_dict({kn:[info_dict[kn]] for kn in kvals})
                df1=pd.concat([df1,dftmp],axis=0)
            except FileNotFoundError:
                print('No file with '+fcontains+' found in '+ftmp)
        else:
            print('Not set up to run for files with extension '+fsuff)
            break
    # These are extra columns with constant value for every item in flist
    if add_cols:
        for kn,vn in add_cols.items():
            df1[kn]=vn
    # These are columns constructed from existing column values concatenated together
    if construct_cols:
        # a dict where each key is the name of a new column and vn is a list of existing column names (with string values)
        for kn,vn in construct_cols.items():
            df1[kn]=['_'.join(df1[construct_cols[kn]].iloc[eachrow]) for eachrow in range(df1.shape[0])]
    # The column names for the metabolites sometimes seem to have whitespace on the ends
    df1=df1.rename(columns={cn1:cn1.strip() for cn1 in df1.columns})
    if idcol:
        df1.loc[:,idcol]=[idpre+idval for idval in df1.loc[:,idcol]]
        df1.set_index(idcol,inplace=True)
    return df1

def disp_lcm_spectra(lcm_dict,ax1=None,axlims=[4.2,0.5],whichmets=None,plot_resids=True,resid_offset=None,plot_bgrd=True,tstr='',ylab='Signal',xlab='Chemical Shift (ppm)',
                     kwdata={'color':'k','ls':'-','lw':0.5},kwfit={'color':'r','ls':'-','lw':1.0},kwbgrd={'color':'darkgrey','ls':':','lw':1.0},kwmet={'color':'b','ls':'-','lw':1.0},kwresids={'color':'k','ls':'-','lw':0.5}):
    # uses lcm_dict from fidA_io.io_readlcmcoord(fname)
    if ax1 is None:
        f1,ax1=plt.subplots(1,1)
        f1.set_size_inches(7,5.5)
    ax1.plot(lcm_dict['ppm'],lcm_dict['data'],**kwdata)
    ax1.plot(lcm_dict['ppm'],lcm_dict['fit'],**kwfit)
    if plot_bgrd:
        ax1.plot(lcm_dict['ppm'],lcm_dict['bgrd'],**kwbgrd)
    if whichmets is not None:
        for metct,metnm in enumerate(whichmets):
            #ax1.plot(lcm_dict['ppm']-0.1*metct*np.max(lcm_dict['fit']),lcm_dict[metnm]-lcm_dict['bgrd'],**kwmet)
            ax1.plot(lcm_dict['ppm'],lcm_dict[metnm],**kwmet)
    if plot_resids:
        if resid_offset is None:
            resid_offset=-0.25*max(lcm_dict['fit'])
        ax1.plot(lcm_dict['ppm'],lcm_dict['fit']-lcm_dict['data']+resid_offset,**kwresids)
    ax1.set_xlim(axlims)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    ax1.set_title(tstr)
    return ax1

def use_fixed_errorbars(funcnm):
    """
    This is a decorator function that can be applied to certain seaborn plotting
    functions (eg. lineplot, relplot, as shown below in the relplot_fixed_err 
    function) that creates errorbars that are defined by a column identified 
    by the 'yerrs' argument (either a percentage of the y-value or an absolute
    value, depending on the yerr_is_percent flag). This is done by "hacking" 
    the dataframe and adding the upper and lower bounds to the column name 
    defined in the 'y' arguments such that there are three data points for each 
    x-value (the original y-value, the upper bound and the lower bound) and 
    then setting errorbar=('pi',100) to create error bars that span the range 
    from lower to upper bound.
    Note that, because of the way that the hack is implemented, this only makes
    sense when there is one y-value for each x/hue/row/column combination (ie.
    each y point on any graph is one number directly from the dataframe, not a 
    mean or median value obtained by combining y-values at a particular x-value).
    A check is run to try to ensure this is the case and the 'estimator' argument 
    is fixed to median to try to prevent odd behaviour in other cases.
    The function funcnm must have data, x and y arguments and any call to the 
    decorated function must have values for these arguments.

    Parameters
    ----------
    funcnm : function
        The seaborn function to be adapted to use fixed errorbar values. 
        Currently have tested on sns.lineplot and sns.relplot. Initially I 
        thought that I might want to use it for scatterplot but actually
        scatterplot plots all points separately and has no estimator or errobar-
        associated parameters. If you want a plot with points that represent 
        the y-values and errorbars that represent uncertainty, then you 
        actually want to call lineplot and use the err_style='bars' argument.

    Returns
    -------
    wrapper
        The wrapper function that is returned is the original function with the 
        alterations applied to make the errorbar plotting work.
        Example of applying the decorator:
        @use_fixed_errorbars
        def relplot_fixed_err(yerrs=None, data=None, **kwargs):
            g1=sns.relplot(data, **kwargs)
            return g1
        Creates a function relplot_fixed_err that applies this decorator's 
        functionality to sns.relplot
    """
    @functools.wraps(funcnm)
    def wrapper(yerrs=None,yerr_is_percent=True,data=None,x=None,y=None,**kwargs):
        # Note that I've set this up so that yerrs, yerr_is_percent, data, x and 
        # y are required. Other arguments passed to funcnm are optional but 
        # must be passed as keyword arguments (this is actually not that 
        # different from relplot and lineplot, where everything other than data 
        # is a keyword argument. Technically, relplot and lineplot can be 
        # called where data is a wideform pandas dataframe and the x and y 
        # column identifiers are not required. However, I don't regularly use 
        # wideframe data so I have only tested on longform and I'm not sure 
        # what alterations would be needed otherwise. Therefore data must be 
        # longform and x and y are required arguments.
        try:
            # If someone enters the estimator argument, it should be ignored. 
            # If errorbars are symmetric then mean and median estimators would 
            # be identical, but the code does also allow a list of two column 
            # names for yerrs, which would indicate upper and lower errorbars 
            # that could be asymmetric. In this case, the estimator must be
            # median to plot the correct original data location.
            kwargs.pop('estimator')
            print("WARNING: Estimator argument cannot be set with use_fixed_errorbars decorator. Using median estimator instead.")
            kwargs['estimator']='median'
        except KeyError:
            kwargs['estimator']='median'
        # Check that the y-values are unique (once split over subplots, hues, 
        # etc). I have not really tested that this works for all grouping 
        # variables since I only tend to use hue, row and col, but it should.
        alist=[akey for akey in ['hue','row','col','size','style'] if akey in kwargs.keys()]
        df2=data.groupby(alist)
        countcheck=all([xv==yv for xv,yv in zip(df2[x].count().values,[len(dval) for dval in df2[x].unique().values])])
        if not countcheck:
            print('WARNING: This wrapper function is only designed to work when each x value has only one corresponding y-value. You may have more.')
        # yerrs is either a column name (str) or a list of 2 column names that 
        # define the locations of the upper and lower errors (or just one if
        # errors are symmetric)
        if isinstance(yerrs, str):
            if yerr_is_percent:
                y_upper=data[y]+data[y]*data[yerrs]/100
                y_lower=data[y]-data[y]*data[yerrs]/100
            else:
                y_upper=data[y]+data[yerrs]
                y_lower=data[y]-data[yerrs]
        elif isinstance(yerrs,list) and len(yerrs)==2:
            if yerr_is_percent:
                y_lower=data[y]-data[y]*data[yerrs[0]]/100
                y_upper=data[y]+data[y]*data[yerrs[1]]/100
            else:
                y_lower=data[y]-data[yerrs[0]]
                y_upper=data[y]+data[yerrs[1]]
        else:
            print('yerr must be either str or a 2-element list designating the column name(s) used to define errors')
        d2=data.copy()
        d2[y]=y_upper
        d3=data.copy()
        d3[y]=y_lower
        data=pd.concat([data,d2,d3],ignore_index=True)
        kwargs['errorbar']=('pi',100)
        result=funcnm(data=data,x=x,y=y,**kwargs)
        return result
    return wrapper

@use_fixed_errorbars
def relplot_fixed_err(yerrs=None,yerr_is_percent=True,data=None,**kwargs):
    """
    relplot_fixed_err(yerrs=None, yerr_is_percent=True, data=None, x=None, y=None, **kwargs) 
    is a version of the seaborn relplot function that applies the 
    use_fixed_errorbars decorator in order to plot errorbars based on a column 
    name defined by 'yerrs'. The original relplot function plots errorbars 
    based on the distribution of sets of y-values in the dataframe (when there 
    are multiple y-values for a particular x-value) but use_fixed_errorbars 
    assumes one y-value per x-value (a check is run in the decorator for this) 
    and "hacks" the existing errobrar functionality. See the use_fixed_errorbar 
    function for more explanation.

    Parameters
    ----------
    yerrs : str or 2-element list
        Identifies the column(s) to use for the errorbars. In the case of a 
        string input, this is simply the column name in data. In the case of a 
        list, the two elements are strings identifying the column names 
        corresponding to the lower and upper error values, respectively.
    yerr_is_percent : boolean, optional
        Identifies whether yerrs should be interpreted as a percentage of the 
        y-value (True) or as an absolute value below/above the y-value (False). 
        The default is True.
    data : long form pandas dataframe
        Contains the data to be plotted. Note that the original relplot function
        allows wideform data where the column names that indicate the x and y
        data are not required, but this modified relplot function has not been
        set up or tested for this (mainly because I don't use wideform data).
        Therefore data must be long form and x and y are required when 
        relplot_fixed_err is called.
    **kwargs : keyword arguments
        Dictionary of keyword arguments to be passed through. Note that yerrs,
        data, x and y are all required to run the function. Remaining kwargs
        such as hue or col will be passed through from the relplot_fixed_err 
        call. Any "estimator" kwarg is replaced with 'median' to avoid plotting
        mistakes when the upper and lower errors are asymmetric.

    Returns
    -------
    g1 : seaborn.axisgrid.FacetGrid
        The figure object containing the relplot output.

    """
    g1=sns.relplot(data,**kwargs)
    return g1
    
    
@use_fixed_errorbars
def lineplot_fixed_err(yerrs=None,yerr_is_percent=True,data=None,**kwargs):
    """
    lineplot_fixed_err(yerrs, yerr_is_percent=True, data, x=None, y=None, **kwargs) 
    is a version of the seaborn lineplot function that applies the 
    use_fixed_errorbars decorator in order to plot errorbars based on a column 
    name defined by 'yerrs'. The original lineplot function plots errorbars 
    based on the distribution of sets of y-values in the dataframe (when there 
    are multiple y-values for a particular x-value) but use_fixed_errorbars 
    assumes one y-value per x-value (a check is run in the decorator for this) 
    and "hacks" the existing errobrar functionality. See the use_fixed_errorbar 
    function for more explanation.

    Parameters
    ----------
    yerrs : str or 2-element list
        Identifies the column(s) to use for the errorbars. In the case of a 
        string input, this is simply the column name. In the case of a list,
        the two elements are strings identifying the column names corresponding
        to the lower and upper error values, respectively.
    yerr_is_percent : boolean, optional
        Identifies whether yerrs should be interpreted as a percentage of the 
        y-value (True) or as an absolute value below/above the y-value (False). 
        The default is True.
    data : long form pandas dataframe
        Contains the data to be plotted. Note that the original lineplot 
        function allows wideform data where the column names that indicate the 
        x and y data are not required, but this modified lineplot function has 
        not been set up or tested for this (mainly because I don't use wideform 
        data). Therefore data must be long form and x and y are required when 
        lineplot_fixed_err is called.
    **kwargs : keyword arguments
        Dictionary of keyword arguments to be passed through. Note that yerrs,
        data, x and y are all required to run the function. Remaining kwargs
        such as hue or style will be passed through from the lineplot_fixed_err 
        call. Any "estimator" kwarg is replaced with 'median' to avoid plotting
        mistakes when the upper and lower errors are asymmetric.

    Returns
    -------
    g1 : matplotlib.axes._subplots.AxesSubplot
        The axis subplot object containing the lineplot output.

    """
    g1=sns.lineplot(data,**kwargs)
    return g1

def disp_lcm_ridgeplot(lcm_dict,metlist=None,xlims=[4.2,0.2],ylims=None,plot_MMs='combined',offset=0.0,figname=None,figheight=5,figwidth=7):
    """
    disp_lcm_ridgeplot(lcm_dict,metlist=None,xlims=[4.2,0.2],ylims=None,plot_MMs='combined',offset=0.0,figname=None,figheight=5,figwidth=7)
    This function creates a graphical version of as basis set (as defined by a
    dictionary lcm_dict obtained from fio.io_readlcmcoord(fname) function on the
    .coord output from any lcmodel fit). This is useful for demonstrating
    metabolite positions or the components of the basis set. Note that the function
    name uses the term ridgeplot because that is where the initial idea came 
    from, but plotting is actually done by combining seaborn's FacetGrid and 
    lineplot functions. The colours vary and are generated by a palette that is
    hard-coded below. Labels in corresponding colours are added to the right of 
    each metabolite.

    Parameters
    ----------
    lcm_dict : dictionary of arrays
        A dictionary of arrays defining the metabolite spectrum for each 
        metabolite, as well as the ppm and bgrd.
    metlist : List, optional
        A list of metabolite names corresponding to the dictionary key values to
        be plotted. The default is None, which plots all metabolites, but 
        macromolecular plotting is determined by the plot_MMs argument.
    xlims : tuple or list of 2 items, optional
        The x-limit on the graph, corresponding to the ppm values to plot between.
        The default is [4.2,0.5].
    ylims : tuple or list of 2 items, optional
        The x-limit on the graph, corresponding to the ppm values to plot between.
        The default is None, which plots between 0 and the max NAA value
    plot_MMs : string, optional
        Defines how to plot the macromolecules, which are defined by starting 
        with 'MM'. The options are 'no' (do not plot any macromolecules), 'combined'
        (add together all macromolecules for a single line in the plot) and 'yes'
        or any other value, which plots all MMs in the dictionary. The default 
        is 'combined'.
    offset : float, optional
        The amount to offset the plot for each metabolite in the x-direction, 
        which can improve the ease of seeing some metabolites. This value is 
        scaled so that a value of -0.05 moves subsequent plots 5% of the axis
        distance toward the negative direction (ie. to the right for conventional
        spectra that run from 4.5 to 0 ppm. The default is 0.0.
    figname : string filename, optional
        The filename to save the figure to. The default is None, where the figure
        is not saved.
    figheight : float, optional
        Height of the figure (in inches, for my seaborn setup). The default is 5.
    figwidth : float, optional
        Height of the figure (in inches, for my seaborn setup). The default is 7.

    Returns
    -------
    g1 : seaborn.axisgrid.FacetGrid
        The figure object containing the metabolite plots

    """
    if ylims is None:
        ylims=[0,np.max(lcm_dict['NAA']-lcm_dict['bgrd'])]
    # if no metabolites are given, plot everything (except macromolecules, whose
    # plotting depends on the value of plot_MMs). This does mean removing the ppm
    # bgrd, data and fit arrays in lcm_dict that are obtained from the .coord file
    if metlist is None:
        metlist=[mname for mname in lcm_dict.keys() if mname not in ['bgrd','ppm','data','fit']]
    if plot_MMs=='no':
        metlist=[mname for mname in metlist if not mname.startswith('MM')]
    elif plot_MMs=='combined':
        MMlist=[mname for mname in metlist if mname.startswith('MM')]
        metlist=[mname for mname in metlist if not mname.startswith('MM')]
        try:
            lcm_dict['MM_tot']=np.zeros_like(lcm_dict[MMlist[0]])
            for mnm in MMlist:
                lcm_dict['MM_tot']=lcm_dict['MM_tot']+lcm_dict[mnm]
            lcm_dict['MM_tot']=lcm_dict['MM_tot']/len(MMlist)
            metlist=['MM_tot']+metlist
        except IndexError:
            print('Warning: no macromolecules in basis set')
    
    df_lcm=pd.DataFrame(columns=['ppm','Conc','metabolite'])
    for metct,metab in enumerate(metlist):
        tmpdict=pd.DataFrame.from_dict({'ppm':lcm_dict['ppm']+metct*offset,'Conc':lcm_dict[metab]-lcm_dict['bgrd'],'metabolite':[metab]*len(lcm_dict['ppm'])})
        df_lcm=pd.concat([df_lcm,tmpdict], axis=0,ignore_index=True)
    sns.set_theme(style='white', rc={'axes.facecolor':(0,0,0,0)})
    pal = sns.cubehelix_palette(len(metlist), rot=-.25, light=.7)
    #pal = sns.color_palette("Set2", 12)
    g1=sns.FacetGrid(df_lcm,row='metabolite',hue='metabolite',palette=pal)
    g1.fig.set_figheight(figheight)
    g1.fig.set_figwidth(figwidth)
    g1.map(sns.lineplot,'ppm','Conc',lw=1.5,clip_on=False)
    xwidth=abs(xlims[1]-xlims[0])+len(metlist)*abs(offset)
    stepct=offset/(xwidth)
    if offset<0:
        xbit=len(metlist)*stepct
    else:
        xbit=0
    for metct,metlab in enumerate(metlist):
        ax=g1.facet_axis(metct, 0)
        ax.text(1.0-stepct*metct+xbit,0.2,metlab,color=pal[metct],fontsize='medium',
                ha='left',va='center',transform=ax.transAxes)
    g1.fig.subplots_adjust(hspace=-0.6,bottom=0.15,top=1.0,right=0.90,left=0.05)
    g1.set_titles('')
    currticks=np.r_[min(xlims):max(xlims):0.5]
    g1.set(yticks=[],ylabel='',ylim=ylims,xlim=xlims,xticks=currticks+offset*len(metlist),xticklabels=['{:3.2f}'.format(currt) for currt in currticks])
    g1.set_xlabels('ppm',x=0.5-stepct*len(metlist)/2)
    g1.despine(bottom=True, left=True)
    if figname:
        g1.fig.savefig(figname,dpi=300)
    return g1


if __name__ == '__main__':
    """
    for debugging
    """
    
    #pname='/Users/nearlabmacbook1/Documents/BrukerData/StressMice/baseline'
    #with open(os.path.join(pname,'flist_hippo_female')) as f:
    #    flist2=f.readlines()
    #flist2=[fn.split(os.sep)[0] for fn in flist2]
    #df1=make_df_from_flist(flist2,fcontains='20250303_hippo',fsuff='.csv',add_cols={'timept':'baseline','region':'hippocampus','sex':'female'},pname=os.path.join(pname,'lcm-out'),construct_cols={'id':['fname','timept','sex']},idcol='id')
    #df2=make_df_from_flist(flist2,fcontains='20250303_hippo',fsuff='.coord',add_cols={'timept':'baseline','region':'hippocampus','sex':'female'},pname=os.path.join(pname,'lcm-out'),construct_cols={'id':['fname','timept','sex']},idcol='id')
    #lcm_dict, info_dict=fio.io_readlcmcoord(os.path.join(pname,'lcm-out',flist2[0],'20250303_hippo.coord'))
    #disp_lcm_spectra(lcm_dict,ax1=None,axlims=[4.2,0.5],whichmets=['PCh','GPC'])
    
    pname='/Users/nearlabmacbook1/Documents/BrukerData/StressMice'
    df2=pd.DataFrame()
    for timept in ['baseline','2week','6week']:
        with open(os.path.join(pname,timept,'flist_hippo_female')) as f:
            flist2=f.readlines()
        flist2=[fn.split(os.sep)[0] for fn in flist2]
        df1=make_df_from_flist(flist2,pname=os.path.join(pname,timept,'lcm-out'),fcontains='20250303_hippo',add_cols={'timept':timept})
        df1['mouse_id']=['_'.join(tval.split('_')[5:8]) for tval in df1['fname']]
        df1['genotype']=['WT']*4+['KO']*4+['WT']*4+['KO']*3
        df2=pd.concat([df2,df1.copy()],ignore_index=True)
    df2['y_const']=0.5
    df2['y_const2']=10
    #g2=relplot_fixed_err(yerrs='NAA %SD',yerr_is_percent=True,data=df2,x='timept',y='NAA',hue='mouse_id',col='genotype',kind='line')
    #g2=relplot_fixed_err(yerrs=['y_const2','y_const'],yerr_is_percent=False,data=df2,x='timept',y='NAA',hue='mouse_id',col='genotype',kind='line')
    g2=lineplot_fixed_err(yerrs='NAA %SD',data=df2[df2.loc[:,'genotype']=='WT'],x='timept',y='NAA',hue='mouse_id')
    
    # fname='/Users/nearlabmacbook1/Documents/BrukerData/StressMice/2week/lcm-out/20230715_121511_768_wang_stress_c640_mR_2week_1_3/20250303_hippo.coord'
    # lcdict1,infodict1=fio.io_readlcmcoord(fname)
    # g1=disp_lcm_ridgeplot(lcdict1,offset=-0.05,plot_MMs='combined')#,figname='/Users/nearlabmacbook1/Documents/Analysis/ADrats/Figures/RidgePlot.png')