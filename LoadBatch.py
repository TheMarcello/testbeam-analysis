import numpy as np # NumPy
import matplotlib.pylab as plt # Matplotlib plots
import matplotlib.colors as colors
# import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import mpl_scatter_density
import pandas as pd # Pandas
import uproot
import pickle
import logging

# import awkward as ak
# import mplhep as hep
# import argparse     # to get arguments from command line executing file.py
import os # read directories etc.
from scipy.signal import find_peaks, gaussian
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import pylandau
# import re
# import copy
from wrapt_timeout_decorator import timeout

from SensorClasses import *


PIXEL_SIZE = 0.0184 #mm


def get_DUTs_from_dictionary(dictionary, oscilloscope):
    """
    Small function to add the dut only if there is a 'board' and voltage>0
    dictionary contains all the sensor.__dict__
    """
    DUTs = []
    if dictionary is not None:
        for i,ch in enumerate(('Ch2','Ch3','Ch4')):
            if (dictionary[(oscilloscope,ch)]['board'] != 'no_board') and (dictionary[(oscilloscope,ch)]['voltage'] != 0):
                DUTs.append(i+1)
    else:
        logging.warning(f"Empty dictionary to extract DUTs from: NO DUTs")
    return DUTs

def my_gauss(x, A, mu, sigma, background=0):

    """
    Custom normal distribution function + uniform background.
    default value of background is zero, so if 
    """
    return A * np.exp(-0.5*(x-mu)**2/sigma**2) + background


def read_pickle(file):
    """
    Read the '.pickle' file containing all the list of sensors for each batch, oscilloscope, channel
    
    Arguments
    ----------
    file:           file path to the .pickle c

    Returns
    pickle_dict:    (usually dict) contained in the pickle file
    -------
    """
    with open(file, 'rb') as f:
        pickle_dict = pickle.load(f)
    return pickle_dict


def load_batch(batch_number, oscilloscope, branches=None,
              data_path=f"/home/marcello/Desktop/Radboud_not_synchro/Master_Thesis/Data_TestBeam/2023_May/"):
    """"
    Load all the data from the .root file of one batch and one oscilloscope into a pandas.Dataframe. \n
    Arguments
    ----------
    batch_number:       number of the batch
    oscilloscope:       string for oscilloscope: 'S1' or 'S2'
    branches:           branches of the data of the .root file to load (not all, so it's lighter)
    data_path:          default path of the directory where to find the data (of both oscilloscopes)
    
    Returns
    -------
    df:                 pandas.Dataframe with all the required data
    """
    if branches is None:
        branches = ["eventNumber", "Xtr", "Ytr", "pulseHeight", "charge", "noise", "timeCFD20", "timeCFD50", "timeCFD70"]
    columns_to_remove = ["Xtr_4","Xtr_5","Xtr_6","Xtr_7","Ytr_4","Ytr_5","Ytr_6","Ytr_7"]
    logging.info(f"Loading batch {batch_number} \t Oscilloscope {oscilloscope}")    
    dir_path = os.path.join(data_path,oscilloscope)
    file_path = f"tree_May2023_{oscilloscope}_{batch_number}.root"
    try:
        df = root_to_df(os.path.join(dir_path, file_path), branches)
    except FileNotFoundError:
        logging.error(f"File of batch {batch_number} not found")
        return
    df = df.drop(columns=columns_to_remove)
    return df
    

### this IS NOT necessary anymore
# ### pretty ugly but no alternatives right now
# def get_transimpedance(batch, oscilloscope):
#     """
#     Pretty ugly function that links each batch to the transimpedance value of the board \
#     it mounted on.
#     """
#     four_ch = 10700
#     single_ch = 4700
#     none = -1
#     if batch>=100 and batch<200:     ### Ch2      Ch3       Ch4
#         if oscilloscope=="S1":   return (four_ch, four_ch, four_ch)
#         elif oscilloscope=="S2": return (single_ch, single_ch, four_ch)
#     elif batch>=200 and batch<300:
#         if oscilloscope=="S1":   return (four_ch, four_ch, none)
#         elif oscilloscope=="S2": return (none, none, none)
#     elif batch>=300 and batch<400:
#         if oscilloscope=="S1":   return (four_ch, four_ch, none)
#         elif oscilloscope=="S2": return (single_ch, single_ch, four_ch)
#     elif batch>=400 and batch<500:
#         if oscilloscope=="S1":   return (four_ch, four_ch, four_ch)
#         elif oscilloscope=="S2": return (single_ch, single_ch, single_ch)
#     elif batch>=500 and batch<600:
#         if oscilloscope=="S1":   return (single_ch, single_ch, single_ch)
#         elif oscilloscope=="S2": return (single_ch, single_ch, none)
#     elif batch>=600 and batch<700:
#         if oscilloscope=="S1":   return (four_ch, four_ch, four_ch)
#         elif oscilloscope=="S2": return (single_ch, single_ch, four_ch)
#     elif batch>=700 and batch<800:
#         if oscilloscope=="S1":   return (four_ch, four_ch, four_ch)
#         elif oscilloscope=="S2": return (none, none, four_ch)
#     elif batch>=800 and batch<900:
#         if oscilloscope=="S1":   return (none, none, none)
#         elif oscilloscope=="S2": return (single_ch, single_ch, none)
#     elif batch>=900 and batch<1000:
#         if oscilloscope=="S1":   return (single_ch, single_ch, none)
#         elif oscilloscope=="S2": return (none, none, none)
#     elif batch>=1000 and batch<1100:
#         if oscilloscope=="S1":   return (four_ch, four_ch, four_ch)
#         elif oscilloscope=="S2": return (single_ch, single_ch, four_ch)
#     elif batch>=1100 and batch<1200:
#         if oscilloscope=="S1":   return (single_ch, single_ch, none)
#         elif oscilloscope=="S2": return (single_ch, single_ch, none)
#     elif batch>=1200 and batch<1300:
#         if oscilloscope=="S1":   return (single_ch, single_ch, none)
#         elif oscilloscope=="S2": return (none, none, none)
#     elif batch>=1300 and batch<1400:
#         if oscilloscope=="S1":   return (none, none, none)
#         elif oscilloscope=="S2": return (none, single_ch, none)
#     else:     ### last case, return all none
#         if oscilloscope=="S1":   return (none, none, none)
#         elif oscilloscope=="S2": return (none, none, none)


def root_to_df(file_path, branches):
    """
    Converts a file.root into pandas DataFrame unpacking branches with multiple channels \
into 'Branch_0', 'Branch_1' etc. \n
    Arguments
    ----------
    file_path:  full or relative path of the .root file with the Ntuples
    branches:   list of branches of the tree to be imported into the dataframe

    Returns
    -------
    df:         pandas DataFrame containing the branches (unpacked into multiple columns)
    """
    try:    tree = uproot.open(file_path+":tree")
    except: raise FileNotFoundError(f"no file named: {file_path}")  ### raise error if file not found
    df_ak = tree.arrays(branches, library='ak') ### changed library from pd (pandas) to ak (awkward)
    df = pd.DataFrame()  ### empty dataframe

    for name in branches: ### converts all entries into Entry_0, Entry_1 etc.
        try:              ### if the shape of the branch 'name' is larger than 1
            for idx in range(np.shape(df_ak[name])[1]):
                new_columns = pd.DataFrame(df_ak[name][:,idx])
                new_columns.columns = [f'{name}_{idx}']   ### rename it to Entry_idx
                df = pd.concat([df, new_columns], axis=1) ### add new_columns
        except:  
            new_columns = pd.DataFrame(df_ak[name])
            new_columns.columns = [f'{name}']
            df = pd.concat([df,new_columns], axis=1)
    del df_ak, tree         ### I am trying to fix the memory leak (not sure this is relevant)
    return df


def plot_histogram(data, bins='auto', poisson_err=False, error_band=False, fig_ax=None, label=None, **kwrd_arg):
    """
    Plot a simple (list of) histogram with optionally the poissonian error. \n
    Arguments
    ----------
    data:       (list of) data to plotted as histogram
    bins:       matplot bins options e.g. int (number of bins), list (bin edges)
    poisson_err: boolean for calculating and plotting the poissonian error
    error_band: plots error as a band instead of errorbar (for dense data)
    fig_ax:     (figure, axis) object so the histogram can be added to other figures
    kwrd_arg:   dictionary of other matplot parameters

    Returns
    -------
    hist:       histogram values
    bins_points: histogram bins
    info:       dictionary with the return information from plt.hist()
    fig:        matplotlib  figure object
    ax:         and axis object, so that more plots can be added on top
    """
    if fig_ax:  fig, ax = fig_ax
    else:       fig, ax = plt.subplots(figsize=(10,6), dpi=300)
    ax.grid('--')
    hist_parameters = {'histtype':'step'}
    hist_parameters.update(kwrd_arg)
    hist, bins_points, info = ax.hist(data, bins=bins, label=label, **hist_parameters)
    if (poisson_err):      ### adding the poissonian error (sqrt(hist_point)
        bins_centers = (bins_points[1:]+bins_points[:-1])/2
        errorbar_parameters = {'markersize':0, 'linewidth':0, 'alpha':0.5,'ecolor':'k', 'elinewidth':0.3, 'capsize':1, 'errorevery':1}
        errorbar_parameters.update(kwrd_arg)  ### this adds the default options (and overrides them if repeated)
        if (np.shape(np.shape(data))[0]>1): ### a bit convoluted but checks the dimensions of the data
            for single_hist in hist:     ### in case data is a list of data
                y_error = single_hist**0.5
                if error_band:  ### I just mask all the errorbars
                    filled_band_parameters = {'alpha':0.5, 'linestyle':'--'}
                    errorbar_parameters.update({'elinewidth':0,'capsize':0,'errorevery':1})     ### defaults specific to 
                    ax.fill_between(bins_centers, single_hist-y_error, single_hist+y_error, **filled_band_parameters)#, label=f"{label} error")
                ax.errorbar(bins_centers, single_hist, yerr=y_error, **errorbar_parameters)
        else:
            y_error = hist**0.5
            if error_band:      ### I just mask all the errorbars
                filled_band_parameters = {'alpha':0.5, 'linestyle':'--'}
                errorbar_parameters.update({'elinewidth':0,'capsize':0,'errorevery':1})
                ax.fill_between(bins_centers, hist-y_error, hist+y_error, **filled_band_parameters)
            ax.errorbar(bins_centers, hist, yerr=y_error, **errorbar_parameters)
    return hist, bins_points, info, fig, ax


def charge_fit(df, dut, mask, transimpedance, bins=500, p0=None, plot=True, savefig=False, **kwargs):
    """
    Function to find the best fit of the charge distribution to a Landau*Gaussian convolution

    Arguments
    ----------
    df:         (full) dataframe of the data
    dut:        dut number to be studied (1,2,3)
    mask:       boolean mask to apply to the data before plotting histogram and fitting (e.g. time_mask)
    transimpedance: transimpedance value (as df['charge_i'] needs to be divided by the transimpedance to get the actual charge)
    p0:         initial parameters of the fit
    plot:       boolean if the plot should be shown

    Returns
    -------
    param:      fit parameters (mpv, eta, sigma, A)
    covariance: covariance matrix of the fit parameters
    """
    if plot:    hist,my_bins,_,fig,ax = plot_histogram(df[f'charge_{dut}'].loc[mask]/transimpedance, bins=bins,
                                          label=f"CHARGE: Ch{dut+1}")
    else:       hist,my_bins = np.histogram(df[f'charge_{dut}'].loc[mask]/transimpedance, bins=bins)
    bins_centers = (my_bins[1:]+my_bins[:-1])/2
    bins_centers = bins_centers.astype(np.float64)
    charge_estimate = bins_centers[np.argmax(hist)]
    logging.info(f'First charge estimate: {charge_estimate}')
    if p0 is None: p0 = (np.abs(charge_estimate),1,1,np.max(hist))
    param, covariance = curve_fit(pylandau.langau, bins_centers, hist, p0=p0)
    if plot:
        ax.plot(bins_centers, pylandau.langau(bins_centers, *param),
                label=f"$MPV$: %.1f, $\eta$: %.1f, $\sigma$: %.1f, A: %.0f" %(param[0],param[1], param[2], param[3]), **kwargs)
        ax.semilogy()
        ax.set_ylim(1,1.2*np.max(hist))
        ax.legend(fontsize=16)
        if savefig:
            fig.savefig(savefig)
    return param, covariance


def extend_edges(left_edge, right_edge, fraction=0.2):
    """
    Just increase the size of the edges by a 'fraction' amount,
    maybe overkill but easy to understand function
    """
    if left_edge>right_edge:
        logging.warning("in 'extend_edges()', left_edge > right_edge")
    extra_edge = (right_edge-left_edge)*fraction
    return left_edge-extra_edge, right_edge+extra_edge


def efficiency(data, threshold, percentage=True):
    """
    Efficiency of the data: data greater than threshold value / total data. \n
    Arguments
    ----------
    data:       data to evaluate efficiency
    threshold:  thrshold value
    percentage: boolean, default=True, if the output should be in percentage or not

    Returns
    -------
    efficiency: (data>threshold/total)
    """
    factor = 1
    if percentage: factor = 100
    if data.size>0:
        return (sum(data>threshold) / data.size) * factor
    else:
        logging.warning("In efficiency(), data is empty")
        return 0

def efficiency_error(data, threshold):
    """
    Efficiency of the data, including error (adjusted as in https://arxiv.org/pdf/physics/0701199v1.pdf) \n
    Arguments
    ----------
    data:       data to evaluate efficiency
    threshold:  thrshold value

    Returns:
    --------
    efficiency: efficiency=(k+1)/(n+2)
    error:      error=variance**0.5; variance=(k+1)*(k+2)/((n+2)*(n+3)) - (k+1)**2/(n+2)**2

    """
    k = sum(data>threshold)+1
    n = data.size
    eff = (k+1)/(n+2) ### this is the mean value, most probable value is still k/n
    var = ((k+1)*(k+2)/((n+2)*(n+3)) - (k+1)**2/(n+2)**2 )
    return (eff, var**0.5)


def efficiency_k_n(k,n):
    """
    Efficiency and its error (as in https://arxiv.org/pdf/physics/0701199v1.pdf) \n
    Arguments
    ----------
    k:      (array of) int, selected elements
    n:      (array of) int, total elements

    Returns
    -------
    efficiency: efficiency=(k+1)/(n+2)
    error:      error=variance**0.5; variance=(k+1)*(k+2)/((n+2)*(n+3)) - (k+1)**2/(n+2)**2
    """
    ### this is the mean value, most probable value is still k/n
    eff = np.where(np.logical_and(k>0,n>0), (k+1)/(n+2), 0)  
    var = np.where(np.logical_and(k>0,n>0), ((k+1)*(k+2)/((n+2)*(n+3)) - (k+1)**2/(n+2)**2 ), 0)
    return (eff, var**(1/2))


def error_propagation(time_difference, time_difference_error, MCP_resolution, MCP_resolution_error):
    """
    error propagation of delta_t^2 - MCP^2, which gives the time resolution of the DUT and its uncertainty
    """
    if time_difference**2 < MCP_resolution**2:
        logging.error(f"Invalid values of either Deltat or MCP resolution: {time_difference} and {MCP_resolution}")
    z = np.sqrt(time_difference**2 - MCP_resolution**2)
    z_err = np.sqrt((time_difference**2 * time_difference_error**2 + MCP_resolution**2 * MCP_resolution_error**2) / z)
    return z, z_err


@timeout(20) ### max seconds of running
def time_limited_kde_evaluate(kde, x_axis):
    """Evaluating a kernel density estimate on the points of x_axis, it includes a timeout error if it runs too long"""
    return kde.evaluate(x_axis)

### I can actually try to use np.gradient instead of find_peaks
def find_min_btw_peaks(data, bins, peak_prominence=None, min_prominence=None, plot=True,
                       savefig=False, savefig_path='../various plots/', savefig_details='', fig_ax=None):
    """
    Finds the minimun between two peaks, using 'find_peaks()' function. \n
    Arguments
    ----------
    data:           data to be transformed into histogram and of which to find the peaks (e.g. df['pulseHeight_1'])
    bins:           matplot bins options e.g. int (number of bins), list (bin edges), str (method)
                    see https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html
    peak_prominence: "height" of the peaks compared to neighbouring data
    min_prominence:  "depth" of the min compared to neighbouring data
    plot:           boolean, default True, if the plot will to be shown
    savefig:        boolean, if the plot should be saved
    savefig_path:   path of the directory in which to save the plot
    savefig_details: additional details to add to the file name (e.g '_no_cut')
    fig_ax:         (figure, axis) objects so that the plots can be drawn there

    Returns
    -------
    x_min:          x position of the minimum
    """
    if plot:
        if fig_ax:  fig, ax = fig_ax
        else:       fig, ax = plt.subplots(figsize=(15,10), dpi=200)
        hist, bins_hist, _, fig, ax = plot_histogram(data, bins=bins, poisson_err=True, error_band=True,
                                                    fig_ax=(fig,ax))
        ax.semilogy()
        ax.set_ylim(10**(-2), 1.5*np.max(hist))
    ### use np.histogram if I don't want the plot
    else: hist, bins_hist = np.histogram(data, bins=bins)

    bins_centers = (bins_hist[1:]+bins_hist[:-1])/2
    ### Find the normalization factor so I can 'denormalize' the kde
    density_factor = sum(hist)*np.diff(bins_hist)
    kde = gaussian_kde(data)    ### KERNEL DENSITY ESTIMATE
    number_of_tries = 5
    for i in range(number_of_tries):
        try:
            smoothed_hist = time_limited_kde_evaluate(kde, bins_centers) * density_factor 
        except TimeoutError:          ### now take only half of the points to be evaluated
            logging.info(f"Evaluating kde timeout n°: {i+1}. Trying with 1/2 number of points")
            bins_centers = bins_centers[::2] 
            density_factor = density_factor[::2]
            if i==(number_of_tries-1):
                logging.warning("Giving up evaluating kde")
                return None
        except:
            logging.error(f"Unknown error evaluation kde")
            break
        else:   ### when no exception occurs
            break
    ### it plots before it tries to find peaks and/or min
    if plot:    ax.plot(bins_centers, smoothed_hist, linewidth=1, label='Smoothed hist')
    if not peak_prominence: peak_prominence = np.max(hist)/100
    if not min_prominence:  min_prominence = np.max(hist)/100
    recursion_depth = 0  # 0 or 1, not sure which one gives 'max_recursion' tries
    max_recursion = 20 # 10 or 20

    while(recursion_depth<=max_recursion):
            ### find (hopefully two) peaks and plot them
        peaks_idx, info_peaks = find_peaks(smoothed_hist, prominence=peak_prominence)
        global_max_idx = np.argmax(smoothed_hist)
        if (len(peaks_idx)==1) and (global_max_idx!=peaks_idx[0]):  ### find_peaks() does not find max values at edges,
            peaks_idx = np.append(global_max_idx, peaks_idx)        ### so I the global max (if not identical to the peak found)

        if len(peaks_idx)>=2:       ### find the minimum
            local_min, _ = find_peaks(-smoothed_hist[peaks_idx[0]:peaks_idx[1]], prominence=min_prominence)
            
        else:    ### if it doesn't work it's because only one peak was found
            logging.debug("Two peaks not found, retrying")
            recursion_depth += 1
            if recursion_depth==max_recursion:
                logging.warning(f"Two PEAKS not found after {recursion_depth} iterations")
                logging.info(": {info_peaks}")
                return None
            peak_prominence *= 0.5    ### reduce prominence if the peaks are not found
            continue
        if len(local_min)==1:
            break
        elif len(local_min)>1:
            logging.warning(f"More than one minimum found at: {[bins_centers[min_idx+peaks_idx[0]] for min_idx in local_min]}")
            break
        elif len(local_min)==0:
            recursion_depth += 1
            if recursion_depth==max_recursion:
                logging.warning(f"No MIN found after {recursion_depth} iterations")
                return None
            min_prominence *= 0.5       ### reduce prominence if the min is not found

    x_min = bins_centers[local_min[0]+peaks_idx[0]]
    if plot:
        ax.plot(bins_centers[peaks_idx], smoothed_hist[peaks_idx], 'x', markersize=10, color='k', label='Peaks')
        ax.plot(x_min, smoothed_hist[local_min[0]+peaks_idx[0]], 'o', markersize=10, color='r',
                label='Mimimum: %.1f'%x_min, alpha=.7)
        ax.legend(fontsize=16)
    if savefig: fig.savefig(f"{savefig_path}find_min_btw_peaks{savefig_details}.svg")

    return  x_min 


def find_edges(data, bins='rice', use_kde=True, plot=False):
    """
    Finds the 'edges' of the dut (sensor) using the gradient of the hits distribution. \n
    Arguments
    ----------
    data:       data to be put into histogram to find the edges
    bins:       matplot bins options e.g. int (number of bins), list (bin edges)
    use_kde:    boolean, if to use the kernel density estimate instead
    plot:       boolean, if the plot should be shown

    Returns
    -------
    left_edge:  left edge 
    right_edge: right edge
    """
    if plot:    hist, bins_points, _ = plt.hist(data, bins=bins, histtype='step')
    else:       hist, bins_points = np.histogram(data, bins=bins)  ### use np.histogram if I don't want the plot
    bins_centers = (bins_points[1:]+bins_points[:-1])/2
    if use_kde:
        kde = gaussian_kde(data)
        density_factor = sum(hist)*np.diff(bins_points)
        try:
            values = time_limited_kde_evaluate(kde, bins_centers)*density_factor
        except:
            logging.warning("in 'find_edges()': KDE timed out, using normal hist")
            values = hist
    else:
        values = hist
    left_edge = bins_centers[np.argmax(np.gradient(values))]
    right_edge = bins_centers[np.argmin(np.gradient(values))]
    return left_edge, right_edge


def rectangle_from_geometry_cut(left_edge, right_edge, bottom_edge, top_edge, **kwargs):
    """
    Makes a Rectangle (matplotlib patches) with the geometry_cut information, so that it can be plotted
    """
    ### default arguments, can be overrided by user kwargs
    default_arguments = {'facecolor':"none", 'ec':'r', 'lw':1}
    default_arguments.update(**kwargs)
    return Rectangle(xy=(left_edge,bottom_edge), width=(right_edge-left_edge), height=(top_edge-bottom_edge),
                     **default_arguments)


def geometry_mask(df, DUT_number, bins, bins_find_min='rice', only_select="normal", plot=False, use='pulseheight', time_bins=5000, fraction=0.2):
    """
    Creates a boolean mask for selecting the 2D shape of the dut (sensor) by applying a pulseHeight cut.
    If the minimum of the pulseHeight could not be found it returns all True

    Arguments
    ----------
    df:             full dataframe because it needs pulseHeight, Xtr and Ytr
    bins:           bins options for  "Xtr" and "Ytr"
    DUT_number:     number of the DUT (1,2,3), corresponding to Channels 2,3,4
    bins_find_min:  bins options for 'find_min_btw_peaks()'
    only_select:    option to select specific subselections of the 'geometry cut'
                        'center':   central area of 0.5x0.5 mm^2
                        'extended': 20% extended area (to study interpad area)
                        'normal':   full dut area 
                        'X':        only filters on one axis (X)
                        'Y':         "      "      "     "   (Y)
    plot:           plot the fit of the 'find_min_btw_peaks()' or the 'time_mask()'
    use:            option to use pulseheight or time to determine the geometry cut
                        'pulseheight'
                        'time'
    time_bins:      binning for the time (only used in case: use='time')
    fraction:       fraction of sensor width to extend the 'extended' selection

    Returns
    -------
    bool_geometry:  boolean mask of the events inside the geometry cut
    dict:           dictionary of the edges values found:
                        'left_edge'
                        'right_edge'
                        'bottom_edge'
                        'top_edge'
    """
    dut = DUT_number ### index of the DUT
    try:
        match use:
            case 'pulseheight':
                min_value = find_min_btw_peaks(df[f"pulseHeight_{dut}"], bins=bins_find_min, plot=plot)#True
                my_filter = df[f"pulseHeight_{dut}"]>min_value
            case 'time':
                my_filter = time_mask(df, dut, bins=time_bins, plot=plot)[0]
            case other:
                logging.warning(f"wrong parameter: {other}, options: 'pulseheight' or 'time' ")
        Xtr_cut = df[f"Xtr_{dut-1}"].loc[my_filter]       ### X tracks with applied pulseHeight
        Ytr_cut = df[f"Ytr_{dut-1}"].loc[my_filter]
        left_edge, right_edge = find_edges(Xtr_cut, bins=bins[0], use_kde=True)
        bottom_edge, top_edge = find_edges(Ytr_cut, bins=bins[1], use_kde=True)
    except:
        logging.error("in 'geometry_mask()', something wrong, no boolean mask")
        return pd.Series(True, index=df.index), {}  ### return all True array if there is no minimum
    match only_select:
        case "center":
            central_edge = 0.5 / 2 # 0.5mm / 2
            center = ((left_edge+right_edge)/2, (bottom_edge+top_edge)/2)  ### center of the pixel
            left_edge =     np.floor(center[0] - central_edge/PIXEL_SIZE)  ### new edges, rounded to the pixel
            right_edge =    np.ceil(center[0] + central_edge/PIXEL_SIZE)
            bottom_edge =   np.floor(center[1] - central_edge/PIXEL_SIZE)
            top_edge =      np.ceil(center[1] + central_edge/PIXEL_SIZE)
            xgeometry = np.logical_and(df[f"Xtr_{dut-1}"]>left_edge, df[f"Xtr_{dut-1}"]<right_edge)
            ygeometry = np.logical_and(df[f"Ytr_{dut-1}"]>bottom_edge, df[f"Ytr_{dut-1}"]<top_edge)
            bool_geometry = np.logical_and(xgeometry, ygeometry)
        case "extended":
            left_edge, right_edge = extend_edges(left_edge, right_edge, fraction=fraction)
            bottom_edge, top_edge = extend_edges(bottom_edge, top_edge, fraction=fraction)
            xgeometry = np.logical_and(df[f"Xtr_{dut-1}"]>left_edge, df[f"Xtr_{dut-1}"]<right_edge)
            ygeometry = np.logical_and(df[f"Ytr_{dut-1}"]>bottom_edge, df[f"Ytr_{dut-1}"]<top_edge)
            bool_geometry = np.logical_and(xgeometry, ygeometry)
        case "normal":
            xgeometry = np.logical_and(df[f"Xtr_{dut-1}"]>left_edge, df[f"Xtr_{dut-1}"]<right_edge)
            ygeometry = np.logical_and(df[f"Ytr_{dut-1}"]>bottom_edge, df[f"Ytr_{dut-1}"]<top_edge)
            bool_geometry = np.logical_and(xgeometry, ygeometry)
        case "X":
            left_edge, right_edge = extend_edges(left_edge, right_edge, fraction=fraction)
            xgeometry = np.logical_and(df[f"Xtr_{dut-1}"]>left_edge, df[f"Xtr_{dut-1}"]<right_edge)
            ygeometry = np.logical_and(df[f"Ytr_{dut-1}"]>bottom_edge, df[f"Ytr_{dut-1}"]<top_edge)
            bool_geometry = np.logical_and(xgeometry, ygeometry)
        case "Y":
            bottom_edge, top_edge = extend_edges(bottom_edge, top_edge, fraction=fraction)
            xgeometry = np.logical_and(df[f"Xtr_{dut-1}"]>left_edge, df[f"Xtr_{dut-1}"]<right_edge)
            ygeometry = np.logical_and(df[f"Ytr_{dut-1}"]>bottom_edge, df[f"Ytr_{dut-1}"]<top_edge)
            bool_geometry = np.logical_and(xgeometry, ygeometry)
        case other:
            logging.warning(f"{other} is not an option, options are 'center', 'X', 'Y', 'normal'")
            return
    return bool_geometry, {'left_edge':left_edge, 'right_edge':right_edge, 'bottom_edge':bottom_edge, 'top_edge':top_edge}


def time_mask(df, DUT_number, bins=10000, n_bootstrap=False, mask=None, p0=None, sigmas=3, CFD_DUT=20, CFD_MCP=50, plot=False, savefig=False, title_info='', fig_ax=None):
    """
    Creates a boolean mask using a gaussian+background fit of the time difference between DUT and MCP.
    The fit is done in the time window -20e3 :_: 20e3

    Arguments
    ----------
    df:         dataframe containing the 'timeCFD50_0' and 'timeCFD20_dut'
    DUT_number: number of the selected dut for the time_mask filter
    bins:       binning options for the time difference
    n_bootstrap:  integer or False, iterations to repeat bootstrap resampling
    CFD_DUT:    constant fraction discriminator for the DUT, possibles are: 20,50,70 (percentage), default=20 # reintroduced
    CFD_MCP:    constant fraction discriminator for the MCP, possibles are: 20,50,70 (percentage), dafault=50
    mask:       boolean array (only one array) to filter events where 'mask' is True (i.e. df['Xtr'].loc[mask[DUT]])
    p0:         initial parameters for the gaussian fit (A, mu, sigma, background)
    sigmas:     number of sigmas from the center to include in the time cut window
    plot:       boolean, if False: np.histogram is called instead, so that no plot is shown
    savefig:    boolean, if not False: the fig is saved at the path 'savefig' (include file name please)
    title_info: additional info to put in the title (e.g. batch number)
    Returns
    -------
    time_cut:   boolean mask of the events within the calculated time frame 
    info:       dictionary containing other useful information about the time cut:
                    'parameters':       parameters of the gaussian fit 
                    'parameters_errors':  and its uncertainty from bootstrap method (zero if no bootstrap) 
                    'covariance':       covariance   "   "   "   "
                    'covariance_errors':   and its uncertainty from bootstrap method (zero if no bootstrap)
                    'chi2_reduced':     reduced chi squared: chi^2/d.o.f.
                    'left_base':        value of the left edge of the cut
                    'right_base':       value of the right edge of the cut
    """
    colormap = ['k','b','g','r']
    dut = DUT_number
    window_limit = 20e3 #ps     ### -window_limit < delta t < +window_limit
    window_fit = np.logical_and((df[f"timeCFD{CFD_DUT}_{dut}"]-df[f"timeCFD{CFD_MCP}_0"]) > -window_limit,
                                (df[f"timeCFD{CFD_DUT}_{dut}"]-df[f"timeCFD{CFD_MCP}_0"]) < +window_limit)
    if mask is not None:
        boolean_mask = np.logical_and(window_fit, mask)
    else:
        boolean_mask = window_fit
    time_data = df[f"timeCFD{CFD_DUT}_{dut}"].loc[boolean_mask]-df[f"timeCFD{CFD_MCP}_0"].loc[boolean_mask]
    if plot:
        if fig_ax:  hist,my_bins,_,fig,ax = plot_histogram((time_data), bins=bins, poisson_err=True, fig_ax=fig_ax)
        else:       hist,my_bins,_,fig,ax = plot_histogram((time_data), bins=bins, poisson_err=True)
    else:       hist,my_bins = np.histogram((time_data), bins=bins, density=False)
    bins_centers = (my_bins[:-1]+my_bins[1:])/2
    if p0 is None: p0 = (np.max(hist), bins_centers[np.argmax(hist)], 100, np.average(hist))
    try:
        hist_error = (hist+1)**.5      ### relative (poissonian) error
        density_factor = sum(hist*np.diff(my_bins))      ### to rescale the histogram after 
        if n_bootstrap:
            logging.info(f"Starting bootstrap for time resolution error with {n_bootstrap} iterations")
            title_info = f" w/ bootstrap (n={n_bootstrap})" + title_info  ### add "w/ bootstrap" to the title
            param_list = np.zeros(shape=(n_bootstrap, len(p0)))
            covar_list = np.zeros(shape=(n_bootstrap, len(p0), len(p0)))
            for i in range(n_bootstrap):
                resampled_hist = np.random.choice(bins_centers, size=len(time_data), p=hist/np.sum(hist))
                hist_sample, _ = np.histogram(resampled_hist, bins=my_bins, density=True)
                if i==0: p0 = (np.max(hist_sample), bins_centers[np.argmax(hist_sample)], 100, np.average(hist_sample))
                ### actually including the error in the fit makes the fit worse (I guess because the 'zeros' have very small error(?) )
                param, covar = curve_fit(my_gauss, bins_centers, hist_sample, p0=p0,# sigma=hist_error/density_factor, absolute_sigma=True, 
                                         bounds=((0,-np.inf,0,0),(np.inf,np.inf,np.inf,np.inf)), nan_policy='omit')
                param_list[i] = param
                p0 = param      ### trying to set p0 to the old parameters to make convergence faster (?)
            param = param_list.mean(axis=0)
            param_error = param_list.std(axis=0)
            covar = covar_list.mean(axis=0)
            covar_error = covar_list.std(axis=0)
        else:
            param, covar = curve_fit(my_gauss, bins_centers, hist, p0=p0)
            param_error, covar_error = 0, 0
            density_factor = 1
    except:
        logging.error("in 'time_mask(): some error occurred while fitting, no time mask")
        return pd.Series(True, index=df.index), None
    logging.info(f"in 'time_mask()': Fit parameters {param}")
    # chi2_reduced = sum((hist-my_gauss(bins_centers,*param))**2/my_gauss(bins_centers,*param))/(len(hist)-len(param))
                ###      (  Y_n - f(X_n)  )**2          /    sigma_n=(f_k**.5 / (N*bin_size)**.5 )     /  (d.o.f)                       
    chi2_reduced = sum((hist - density_factor*my_gauss(bins_centers,*param))**2 / (hist_error)) / (len(hist)-len(param))
    left_base, right_base = param[1]-sigmas*np.abs(param[2]), param[1]+sigmas*np.abs(param[2])
    left_cut =  (df[f"timeCFD{CFD_DUT}_{dut}"]-df[f"timeCFD{CFD_MCP}_0"])>left_base
    right_cut = (df[f"timeCFD{CFD_DUT}_{dut}"]-df[f"timeCFD{CFD_MCP}_0"])<right_base
    time_cut = np.logical_and(left_cut, right_cut)
    if plot:
        ax.set_xlim(param[1]-20*np.abs(param[2]), param[1]+30*np.abs(param[2])) ### sligthly asymmetric to fit the legend better
        ax.set_xlabel(f"$\Delta t$ [ps] (DUT - MCP)", fontsize=14)
        ax.set_ylabel("Events", fontsize=14)
        ax.plot(bins_centers, density_factor*my_gauss(bins_centers,*param), color=colormap[dut], label=f"A: {density_factor*param[0]:.1f}, $\mu$: {param[1]:.1f}, BG:  {density_factor*param[3]:.1f}")
        ax.plot([],[], linewidth=0, label=f"$\sigma$: {param[2]:.2f} $\pm$ {param_error[2]:.2f} ps")
        ax.plot([],[], linewidth=0, label="$\chi^2_{reduced}$: "+f" {chi2_reduced:.3f}")
        ax.set_title("$\Delta$t gaussian fit"+title_info, fontsize=16)
        ax.legend(fontsize=14)
        if savefig:
            fig.savefig(savefig)
    return time_cut, {'parameters':param, 'parameters_errors':param_error, 'covariance':covar, 'covariance_errors':covar_error,
                      'chi2_reduced':chi2_reduced, 'left_base':left_base, 'right_base':right_base} # info 
    

### I want to add time_bins (now 5000)
def plot(df, plot_type, batch_object, this_scope, bins=None, bins_find_min='rice', time_bins=5000, n_DUT=None, CFD_values=None, efficiency_lim=None, extra_info=True, info=True, 
        geometry_cut="normal", mask=None, threshold_charge=4, transimpedance=None, use='pulseheight', zoom_to_sensor=False, 
        fig_ax=None, savefig=False, savefig_path='../various plots', savefig_details='', fmt='svg', title_position=None,
        **kwrd_arg):
    """
    Function to produce the plots \n
    Arguments
    ----------
    df:             FULL dataframe of the data to plot (each plot_type select the data it needs)
    plot_type:      type of plot, options are:
                        '1D_Tracks':        histogram of reconstructed tracks distribution (Xtr and Ytr) ### this is probably not useful but I leave it in
                        '2D_Tracks':        2D plot of the reconstructed tracks
                        'pulseHeight':      histogram of the pulseHeight of all channels (log scale)
                        '2D_Sensors':       pulseHeight cut plot + 2D plot of tracks with cut (highlighting the duts)
                        'Time_pulseHeight'
                        '1D_Efficiency':    projection of the efficiency on X and Y axis respectively
                        '2D_Efficiency':    2D plot of the efficiency 
                        'CFD_comparison':   grid plots with delta t histogram with different CFD values
    batch_object:   batch object (see class Batch in SensorClasses.py)
    this_scope:     oscilloscope name (either 'S1' or 'S2')
    bins:           binning options, (int,int) or (bin_edges_list, bin_edges_list), different default for each plot_type
    bins_find_min:  binning options for the find_min_btw_peaks function (in '2D_Sensors')  
    n_DUT:          number of devices under test (3 for each Scope for May 2023)
    efficiency_lim: limit of the y axis for 1D efficiency plot
    extra_info:     boolean option to have extra information on the plot        ### now only for 'Time_pulseHeight' but could be more generally useful 
    info:           boolean option for standard information 
    mask:           list of boolean arrays to plot the 2D tracks where 'mask' is True (i.e. df['Xtr'].loc[mask[DUT-1]]), LIST SHOULD BE WITH SIZE=3
    geometry_cut:   options for specific cuts
                        'center':   central area of 0.5x0.5 mm^2
                        'extended': 20% extended area (to study interpad area)
                        'normal':   full dut area 
                        'XY':        only filters on one axis (X/Y)
                        False:      no geometry cut applied
    threshold_charge: threshold charge for efficiency calculations (default 4fC)
    transimpedance: manually change the transimpedance values
    use:            option to use pulseheight or time to determine the geometry cut
                        'pulseheight'
                        'time'
    zoom_to_sensor: boolean option to set x and/or y limits of the plot to the geometry cut
    savefig:        boolean option to save the plot
    savefig_path:   folder where to save the plot
    savefig_details: optional details for the file name (e.g. distinguish cuts)
    fmt:            format of the file saved ('.jpg', '.svg', '.png' etc.)
    title_position: vertical displacement of the main title in the plot (each plot_type has its own defaults)
    Returns
    -------
    fig, axes:      figure and axis objects so that more manipulation can be done
    """
    if n_DUT is None:
        n_DUT = (1,2,3)
    match plot_type:
        case "1D_Tracks":        ### 1D tracks plots
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,6), constrained_layout=True, dpi=200, sharey='all')
            if bins is None: bins = (200,200)   ### default binning
            axes = np.atleast_1d(axes)          ### for simplicity, so I can use axes[i] for a single DUT  
            for dut in n_DUT:
                sensor_label = f"DUT: {batch_object.S[this_scope].get_sensor(f'Ch{dut+1}').name}"
                plot_histogram(df[f"Xtr_{dut-1}"], label=sensor_label, bins=bins[0], fig_ax=(fig,axes[0]), **kwrd_arg)
                plot_histogram(df[f"Ytr_{dut-1}"], label=sensor_label, bins=bins[1], fig_ax=(fig,axes[1]), **kwrd_arg)
            axes[0].set_title("X axis projection", fontsize=20)
            axes[1].set_title("Y axis projection", fontsize=20)
            for ax in axes:     ### modify all axes
                ax.legend(fontsize=14, loc='lower center')
                ax.semilogy()
                ax.set_xlabel('pixels', fontsize=20)
                ax.set_ylabel('Events (log)', fontsize=20)
            if title_position is None: title_position = 1.15
            
        case "2D_Tracks":        ### 2D tracks plots
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(nrows=1, ncols=len(n_DUT), figsize=(6*len(n_DUT),6), constrained_layout=True, sharex='all', sharey=False, dpi=200)
            if bins is None: bins = (200,200)   ### default binning
            axes = np.atleast_1d(axes)          ### for simplicity, so I can use axes[i] for a single DUT  
            for i,dut in enumerate(n_DUT):

                if mask:  hist, _, _, im = axes[i].hist2d(df[f"Xtr_{dut-1}"].loc[mask[dut-1]], df[f"Ytr_{dut-1}"].loc[mask[dut-1]], bins=bins, **kwrd_arg)
                else:       hist, _, _, im = axes[i].hist2d(df[f"Xtr_{dut-1}"], df[f"Ytr_{dut-1}"], bins=bins, **kwrd_arg)
                
                plot_title = f"Ch{dut+1}\n{batch_object.S[this_scope].get_sensor(f'Ch{dut+1}').name}"
                axes[i].set_title(plot_title, fontsize=20)
                axes[i].set_aspect('equal')
                axes[i].set_xlabel('pixels', fontsize=20)
                axes[i].set_ylabel('pixels', fontsize=20)
                secx = axes[i].secondary_xaxis('top', functions=(lambda x: x*PIXEL_SIZE, lambda y: y*PIXEL_SIZE))
                secy = axes[i].secondary_yaxis('right', functions=(lambda x: x*PIXEL_SIZE, lambda y: y*PIXEL_SIZE))
                secx.set_xlabel('mm', fontsize=20)
                secy.set_ylabel('mm', fontsize=20)
            if title_position is None: title_position = 1.05
            # fig.tight_layout(w_pad=6, h_pad=4)
            cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.1/len(n_DUT), pad=0.1) ### these numbers need adjusting
            cb.set_label(label="Reconstructed tracks", fontsize=16)

        case "pulseHeight":       ### PulseHeight plot
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,6), constrained_layout=True, dpi=200)
            if bins is None: bins = 'rice'  ### default binning
            # for i in n_DUT.insert(0,0):
            for dut in n_DUT:
                sensor_label = f"DUT: {batch_object.S[this_scope].get_sensor(f'Ch{dut+1}').name}"
                plot_histogram(df[f"pulseHeight_{dut}"], poisson_err=True, error_band=True, bins=bins, fig_ax=(fig,axes), label=sensor_label, **kwrd_arg)
            axes.semilogy()
            axes.set_xlabel("PulseHeight [mV]", fontsize=20)
            axes.set_ylabel("Events (log)", fontsize=20)
            axes.set_xlim(left=-10)
            axes.legend(fontsize=20)
            if title_position is None: title_position = 1.05
            
        case "2D_Sensors":        ### 2D tracks plots filtering noise out (also include pulseHeight plot)
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(nrows=2, ncols=len(n_DUT), figsize=(6*len(n_DUT),12), constrained_layout=True, sharex=False, sharey=False, dpi=200)
            if bins is None: bins = (200,200)   ### default binning
            # fig.tight_layout(w_pad=6, h_pad=6)
            axes = np.atleast_2d(axes.T).T      ### so I can call axes[i,j] in any case (I add .T because np.atleast_2d makes only axes[0,i] available, and I want axes[i,0])
            for i,dut in enumerate(n_DUT): 
                print(f"DUT_{dut}")                   ### BINS: scott, rice or sqrt; stone seems slow, rice seems the fastest
                minimum = find_min_btw_peaks(df[f"pulseHeight_{dut}"], bins=bins_find_min, plot=True, fig_ax=(fig,axes[0,i]), savefig=False)
                axes[0,i].set_xlabel('mV', fontsize=20)
                axes[0,i].set_ylabel('Events', fontsize=20)
                if not minimum:
                    logging.warning("in '2D_Sensors', No minimum found, no 2D plot")
                    axes[0,i].set_title(f"Ch{dut+1}\n{batch_object.S[this_scope].get_sensor(f'Ch{dut+1}').name}", fontsize=24)
                    continue
                plot_title = f"Ch{dut+1}, "+"cut:%.1f"%minimum+f"mV \n{batch_object.S[this_scope].get_sensor(f'Ch{dut+1}').name}"
                axes[0,i].set_title(plot_title, fontsize=20)
                pulseHeight_filter = df[f"pulseHeight_{dut}"]>minimum
                _,_,_,im = axes[1,i].hist2d(df[f"Xtr_{dut-1}"].loc[pulseHeight_filter], df[f"Ytr_{dut-1}"].loc[pulseHeight_filter],
                                                bins=bins, **kwrd_arg)
                axes[1,i].set_aspect('equal')
                axes[1,i].set_xlabel('pixels', fontsize=20)
                axes[1,i].set_ylabel('pixels', fontsize=20)
                secx = axes[1,i].secondary_xaxis('top', functions=(lambda x: x*PIXEL_SIZE, lambda y: y*PIXEL_SIZE))
                secy = axes[1,i].secondary_yaxis('right', functions=(lambda x: x*PIXEL_SIZE, lambda y: y*PIXEL_SIZE))
                secx.set_xlabel('mm', fontsize=20)
                secy.set_ylabel('mm', fontsize=20)
            if title_position is None: title_position = 1.05
            cb = fig.colorbar(im, ax=axes[1], fraction=0.1/len(n_DUT), pad=0.1)   ### this colorbar only counts the last 'im', not good
            cb.set_label(label="Reconstructed tracks", fontsize=16)


        case "Time_pulseHeight":
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(figsize=(8*len(n_DUT),8), ncols=len(n_DUT), dpi=300, subplot_kw={'projection':'scatter_density'}) 
            xlim = (-8e3,-3e3)
            if bins is None: bins = 10000  ### default binning
            axes = np.atleast_1d(axes) ### for simplicity, so I can use axes[i] for a single DUT  ### for simplicity, so I can use axes[i] for a single DUT 

            for i,dut in enumerate(n_DUT):
                if mask:
                    time_array = np.array(df[f'timeCFD20_{dut}'].loc[mask[dut-1]]-df[f'timeCFD50_0'].loc[mask[dut-1]])
                    pulseheight_array = np.array(df[f'pulseHeight_{dut}'].loc[mask[dut-1]])
                else:
                    time_array = np.array(df[f'timeCFD20_{dut}']-df[f'timeCFD50_0'])
                    pulseheight_array = np.array(df[f'pulseHeight_{dut}'])

                ### I am starting to think that this part should not be here at all
                ### only calculate pulseheight and time cut if asked
                if info==True:
                    pulse_cut = find_min_btw_peaks(df[f"pulseHeight_{dut}"], bins=bins_find_min, plot=False)
                    if pulse_cut is None:
                        pulse_cut = 0
                    else:
                        axes[i].axhline(pulse_cut, color='r', label="PulseHeight cut value: %.1f mV"%pulse_cut)
                    time_info = time_mask(df, dut, bins=bins, plot=False)[1]
                    if time_info is not None:
                        left_base, right_base = time_info['left_base'], time_info['right_base']
                        axes[i].axvline(left_base, color='g', alpha=.9, label="Time cut: %.0fps$<\Delta t<$ %.0fps"%(left_base, right_base))
                        axes[i].axvline(right_base, color='g', alpha=.9)
                        axes[i].legend(fontsize=16, loc='best', framealpha=0) ### only show legend if there is something inside

                axes[i].set_xlabel(f"$\Delta t$ [ps] (DUT {dut} - MCP)", fontsize=16)
                axes[i].set_ylabel(f"PulseHeight [mV]", fontsize=16)
                axes[i].grid('--')
                plot_title = f"DUT: {batch_object.S[this_scope].get_sensor(f'Ch{dut+1}').name}"
                axes[i].set_title(plot_title, fontsize=16)

                im = axes[i].scatter_density(time_array, pulseheight_array, cmap='Blues', norm=colors.LogNorm(vmin=1e-2, vmax=1e3, clip=True))
                axes[i].set_xlim(xlim)
                ylim = axes[i].get_ylim()
                axes[i].set_ylim(ylim[0],ylim[1]*1.1)  ### to leave space for the legend

                if extra_info and time_info is not None:
                    total = len(time_array)/100  ### so I get percentage directly
                    axes[i].annotate(f"%.3f"%(len(time_array[np.logical_and(time_array<left_base, pulseheight_array<pulse_cut)])/total)+"%", ((xlim[0]+left_base)/2, (2*ylim[0]+pulse_cut)/3), fontsize=16)
                    axes[i].annotate(f"%.3f"%(len(time_array[np.logical_and(time_array>right_base, pulseheight_array<pulse_cut)])/total)+"%", ((right_base+xlim[1])/2, (2*ylim[0]+pulse_cut)/3), fontsize=16)
                    axes[i].annotate(f"%.3f"%(len(time_array[np.logical_and(np.logical_and(time_array>left_base, time_array<right_base), pulseheight_array<pulse_cut)])/total)+"%", ((right_base+left_base)/2, (ylim[0]+pulse_cut)/2), fontsize=16)

                    axes[i].annotate(f"%.3f"%(len(time_array[np.logical_and(time_array<left_base, pulseheight_array>pulse_cut)])/total)+"%", ((xlim[0]+left_base)/2, (pulse_cut+ylim[1])/2), fontsize=16)
                    axes[i].annotate(f"%.3f"%(len(time_array[np.logical_and(time_array>right_base, pulseheight_array>pulse_cut)])/total)+"%", ((right_base+xlim[1])/2, (pulse_cut+ylim[1])/2+20), fontsize=16)
                    axes[i].annotate(f"%.3f"%(len(time_array[np.logical_and(np.logical_and(time_array>left_base, time_array<right_base), pulseheight_array>pulse_cut)])/total)+"%", ((right_base+left_base)/2, (pulse_cut+ylim[1])/2+50), fontsize=16)

            for ax in axes:
                ax.sharey(axes[0])
            cb = fig.colorbar(im, ax=axes)
            cb.set_label(label="Events density", fontsize=14)
            if title_position is None: title_position = 1.05


        case "1D_Efficiency":
            if bins is None: bins = (200)       ### default binning
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(nrows=2, ncols=len(n_DUT), figsize=(6*len(n_DUT),12), constrained_layout=True, sharex=False, sharey=True, dpi=200)
            axes = np.atleast_2d(axes.T).T  ### add an empty axis so I can call axes[i,j] in any case           
            # fig.tight_layout(w_pad=6, h_pad=10)
            if efficiency_lim is None: ylim = (0.4, 1)
            else: ylim = efficiency_lim
            for i,dut in enumerate(n_DUT):
                if geometry_cut in ('center', 'normal', 'extended'):
                    geo_mask, edges = geometry_mask(df, DUT_number=dut, bins=bins, bins_find_min=bins_find_min, only_select=geometry_cut, use=use)
                for coord_idx, XY in enumerate(('X','Y')):  # coord = ['X','Y']
                    if geometry_cut=='XY':
                        geo_mask, edges = geometry_mask(df, DUT_number=dut, bins=bins, bins_find_min=bins_find_min, only_select=XY, use=use)
                    if geometry_cut and mask: bool_mask = np.logical_and(mask[dut-1],geo_mask)
                    elif geometry_cut:  bool_mask = geo_mask  ### this is a boolean mask of the selected positions                
                    elif mask:          bool_mask = mask[dut-1]
                    else:       bool_mask = pd.Series(True,index=df.index)
                    if transimpedance is None:
                        transimpedance = batch_object.S[this_scope].get_sensor(f'Ch{dut+1}').transimpedance
                    events_above_threshold = df[f"charge_{dut}"].loc[bool_mask]/transimpedance > threshold_charge
                    above_threshold = np.logical_and(bool_mask, events_above_threshold)
                    
                    total_events_in_bin, bins_edges = np.histogram(df[f"{XY}tr_{dut-1}"].loc[bool_mask], bins=bins[coord_idx])
                    events_above_threshold_in_bin, _  = np.histogram(df[f"{XY}tr_{dut-1}"].loc[above_threshold], bins=bins[coord_idx])

                    bins_centers = (bins_edges[:-1]+bins_edges[1:])/2
                    eff, err = efficiency_k_n(events_above_threshold_in_bin, total_events_in_bin)
                    axes[coord_idx,i].plot(bins_centers, eff, label=f"Ch{dut+1}", drawstyle='steps-mid')
                    sigma_coeff = 1
                    axes[coord_idx,i].errorbar(bins_centers, eff, yerr=sigma_coeff*err, elinewidth=1, markersize=0, linewidth=0, capsize=1.5,
                                label=f"error: {sigma_coeff}$\sigma$")
                    plot_title = f"{XY} axis projection, Ch{dut+1}\n{batch_object.S[this_scope].get_sensor(f'Ch{dut+1}').name}"
                    axes[coord_idx,i].set_title(plot_title, fontsize=20, y=1.05)
                    axes[coord_idx,i].set_xlabel(f"{XY} position (pixels)", fontsize=20)
                    axes[coord_idx,i].set_ylabel("Efficiency", fontsize=20)
                    axes[coord_idx,i].set_ylim(ylim)
                    if zoom_to_sensor and geometry_cut:
                        try:
                            if XY=='X':     axes[coord_idx,i].set_xlim(edges['left_edge'],edges['right_edge'])
                            elif XY=='Y':   axes[coord_idx,i].set_xlim(edges['bottom_edge'],edges['top_edge'])
                        except:
                            logging.error("in plot(), could not set limits to geometry_cut")
                    ### calculate average only between limits of the plot
                    between_edges = np.logical_and(bins_centers>axes[coord_idx,i].get_xlim()[0], bins_centers<axes[coord_idx,i].get_xlim()[1])
                    if extra_info:  ### include (or not) extra information
                        efficiency_bar = np.average(eff[between_edges]) ### horizontal line at this efficiency %
                        axes[coord_idx,i].axhline(efficiency_bar, label=f"Average efficiency: %.2f"%(efficiency_bar*100)+'%', color='r', alpha=0.4, linewidth=2)
                    axes[coord_idx,i].grid('--')
                    axes[coord_idx,i].legend(fontsize=14)
            if title_position is None: title_position = 1.1
            # savefig_details += f'(geometry cut using {use})'
    

        case "2D_Efficiency":  
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(nrows=1, ncols=len(n_DUT), figsize=(6*len(n_DUT),6), constrained_layout=True, sharex=False, sharey=False, dpi=200)
            if bins is None: bins = (200,200)       ### default binning
            # fig.tight_layout(w_pad=10)
            axes = np.atleast_1d(axes)      ### for simplicity, so I can use axes[i] for a single DUT 
            for i,dut in enumerate(n_DUT):
                if geometry_cut and mask: 
                    geo_mask, edges = geometry_mask(df, DUT_number=dut, bins=bins, bins_find_min=bins_find_min, only_select=geometry_cut, use=use)    ### this is a boolean mask of the selected positions
                    bool_mask = np.logical_and(mask[dut-1],geo_mask)
                elif geometry_cut: bool_mask, edges = geometry_mask(df, DUT_number=dut, bins=bins, bins_find_min=bins_find_min, only_select=geometry_cut, use=use)
                elif mask:    bool_mask = mask[dut-1]
                else:       bool_mask = pd.Series(True,index=df.index) 
                if transimpedance is None:
                    transimpedance = batch_object.S[this_scope].get_sensor(f'Ch{dut+1}').transimpedance
                total_events_in_bin, x_edges, y_edges, _ = axes[i].hist2d(df[f"Xtr_{dut-1}"].loc[bool_mask], df[f"Ytr_{dut-1}"].loc[bool_mask], bins=bins)
                events_above_threshold = df[f"charge_{dut}"].loc[bool_mask]/transimpedance > threshold_charge
                above_threshold = np.logical_and(bool_mask, events_above_threshold)  ### I THINK THIS IS REDUNDANT, maybe not ???
                events_above_threshold_in_bin, _, _, _ = axes[i].hist2d(df[f"Xtr_{dut-1}"].loc[above_threshold], df[f"Ytr_{dut-1}"].loc[above_threshold], bins=bins)
                efficiency_map = np.divide(events_above_threshold_in_bin, total_events_in_bin, where=total_events_in_bin!=0,
                                        out=np.zeros_like(events_above_threshold_in_bin))*100 # in percentage
                axes[i].clear()
                im = axes[i].imshow(efficiency_map.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], ### extent is full data range
                        aspect='equal', vmin=0, vmax=100)   # aspect='equal' or 'auto'?
                if zoom_to_sensor and geometry_cut:         ### this only sets the data shown
                    if edges:
                        axes[i].set_xlim(edges['left_edge'],edges['right_edge'])
                        axes[i].set_ylim(edges['bottom_edge'],edges['top_edge'])
                    else:
                        logging.warning(f"Geometry cut failed, no edges set as limits")
                
                # axes[i].grid('--')
                plot_title = f"Ch{dut+1}\n{batch_object.S[this_scope].get_sensor(f'Ch{dut+1}').name}"
                axes[i].set_title(plot_title, fontsize=20)
                axes[i].set_xlabel('X Position (pixels)', fontsize=20)
                axes[i].set_ylabel('Y Position (pixels)', fontsize=20)
            
                secx = axes[i].secondary_xaxis('top', functions=(lambda x: x*PIXEL_SIZE, lambda y: y*PIXEL_SIZE))
                secy = axes[i].secondary_yaxis('right', functions=(lambda x: x*PIXEL_SIZE, lambda y: y*PIXEL_SIZE))
                secx.set_xlabel('mm', fontsize=20)
                secy.set_ylabel('mm', fontsize=20)
            cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.1/len(n_DUT), pad=0.2)
            cb.set_label(label="Efficiency (%)", fontsize=16)
            if title_position is None: title_position = 1.2
            # savefig_details += f'(geometry cut using {use})' 

        case "CFD_comparison":
            if bins is None: bins = (200,200)       ### default binning
            if CFD_values is None: CFD_values = (20,50,70)
            axes_size = len(CFD_values)

            window_limit = 20e3
            xlim = (-7e3,-4.5e3)
            MCP_voltage = batch_object.S[this_scope].get_sensor('Ch1').voltage
            match MCP_voltage:  ### last MCP_voltage entry
                case 2500: 
                    MCP_resolution = 36.52 # +/- 0.81
                    MCP_error = 0.81
                case 2600: 
                    MCP_resolution = 16.48 # +/- 0.57
                    MCP_error = 0.57
                case 2800: 
                    MCP_resolution = 3.73  # +/- 1.33
                    MCP_error = 1.33
                case other: logging.error("Incorrect MCP voltage")

            dut = n_DUT
            fig, axes = plt.subplots(figsize=(6*axes_size,4*axes_size), nrows=axes_size, ncols=axes_size, dpi=300)
            
            ### I NEED TO RETURN THIS VALUES
            time_resolution_table = []
            chi2_table = []

            for i, ax in enumerate(axes.flatten()):
                CFD_MCP = CFD_values[i//axes_size]
                CFD_DUT = CFD_values[i%axes_size]

                if mask is not None:    dut_cut = mask[dut-1]
                else:                   dut_cut = pd.Series(True, index=df.index)

                window_fit = np.logical_and((df[f"timeCFD{CFD_DUT}_{dut}"]-df[f"timeCFD{CFD_MCP}_0"])> -window_limit,
                                        (df[f"timeCFD{CFD_DUT}_{dut}"]-df[f"timeCFD{CFD_MCP}_0"])< +window_limit)
                dut_cut = np.logical_and(window_fit, dut_cut)

                time_dict = time_mask(df, dut, bins=time_bins, n_bootstrap=50, plot=True, mask=dut_cut, CFD_DUT=CFD_DUT, CFD_MCP=CFD_MCP, 
                                      title_info=f'\n CFD DUT:{CFD_DUT}% CFD MCP:{CFD_MCP}%', fig_ax=(fig,ax))[1]

                param, param_err = time_dict['parameters'], time_dict['parameters_errors']

                time_resolution_table.append(error_propagation(param[2], param_err[2],
                                                            MCP_resolution, MCP_error))
                # time_resolution_table.append(np.sqrt(param[2]**2-MCP_resolution**2))
                chi2_table.append(time_dict['chi2_reduced'])

                ax.set_xlim(xlim)

            fig.tight_layout(w_pad=4, h_pad=4)
            savefig_details += f' {this_scope} dut: {dut}'
            if title_position is None: title_position = 1.1

            ### I should have each 'case' provide its own final figure title name
            # sensor_name = batch_object.get_sensor(f'Ch{dut+1}').name
            # fig.suptitle(f"Time resolution fit, after applying cuts \
            # \n Batch: {batch_object.batch_number}, Oscilloscope: {this_scope}, Ch{dut+1}: {sensor_name}",y=1.1 , fontsize=20)


        case other:
            logging.error(f"""{other} not a plot option. Options are:
            '1D_Tracks', '2D_Tracks', 'pulseHeight', '2D_Sensors', '1D_Efficiency', '2D_Efficiency', 'Time_pulseHeight', 'CFD_comparison' """)
            return
    
    ### why did I even put this here? maybe to avoid duplication, but it seems dumb now
    fig.suptitle(f"{plot_type}, batch: {batch_object.batch_number} {savefig_details}", fontsize=24, y=title_position)

    if savefig: 
        file_name = f"{plot_type}_{batch_object.batch_number}{savefig_details}.{fmt}"
        fig.savefig(os.path.join(savefig_path, file_name), bbox_inches="tight")
    return fig, axes

