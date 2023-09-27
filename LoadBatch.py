import numpy as np # NumPy
import matplotlib.pylab as plt # Matplotlib plots
import matplotlib.patches as mpatches
import pandas as pd # Pandas
import uproot
import pickle

import awkward as ak
import mplhep as hep
# import argparse     # to get arguments from command line executing file.py
import os # read directories etc.
from scipy.signal import find_peaks, gaussian
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import pylandau
import re
import copy


def root_to_df(file_path, branches):
    """
    Converts a file.root into pandas DataFrame unpacking branches with multiple channels \
into 'Branch_0', 'Branch_1' etc. \n
    Parameters
    ----------
    file_path:  full or relative path of the .root file with the Ntuples
    branches:   list of branches of the tree to be imported into the dataframe

    Returns
    -------
    df:         pandas DataFrame containing the branches (unpacked into multiple columns)
    """
    tree = uproot.open(file_path+":tree")
    df_ak = tree.arrays(branches, library='ak') # changed library from pd (pandas) to ak (awkward)
    df = pd.DataFrame()  # empty dataframe

    for name in branches: # converts all entries into Entry_0, Entry_1 etc.
        try:              # if the shape of the branch 'name' is larger than 1
            for idx in range(np.shape(df_ak[name])[1]):
                new_columns = pd.DataFrame(df_ak[name][:,idx])
                new_columns.columns = [f'{name}_{idx}']   # rename it to Entry_idx
                df = pd.concat([df, new_columns], axis=1) # add new_columns
        except:  
            new_columns = pd.DataFrame(df_ak[name])
            new_columns.columns = [f'{name}']
            df = pd.concat([df,new_columns], axis=1)
    return df


def plot_histogram(data, poisson_err=False, bins='auto', **kwrd_arg):
    """
    Plot a simple (list of) histogram with optionally the poissonian error. \n
    Parameters
    ----------
    data:       (list of) data to plotted as histogram
    poisson_err: boolean for calculating and plotting the poissonian error
    bins:       matplot bins options e.g. int (number of bins), list (bin edges)
    kwrd_arg:   dictionary of other matplot parameters

    Returns
    -------
    fig, ax:    matplot figure and axis, so that more plots can be added on top
    """
    fig, ax = plt.subplots(figsize=(10,6), dpi=300)
    ax.grid('--')
    hist_points, bins_points, _ = ax.hist(data, bins=bins, histtype='step', **kwrd_arg)
    if (poisson_err):      # adding the poissonian error (sqrt(hist_point)
        bins_centers = (bins_points[1:]+bins_points[:-1])/2
        if (np.shape(np.shape(data))[0]>1): # a bit convoluted but checks the dimensions of the data
            for single_hist in hist_points:     # in case data is a list of data
                ax.errorbar(bins_centers, single_hist, yerr=single_hist**0.5, elinewidth=2, marker='.', linewidth=0, alpha=0.7)
        else:
            ax.errorbar(bins_centers, hist_points, yerr=hist_points**0.5, elinewidth=2, marker='.', linewidth=0, alpha=0.7)
    return fig, ax


def add_histogram(ax, data, bins='auto',  **kwrd_arg):
    """
    Adds a histogram to an axis. No returns. \n
    Parameters
    ----------
    ax:         matplot axis object (to which the histogram will be added)
    data:       data to be plotted
    bins:       matplot bins options e.g. int (number of bins), list (bin edges)
    kwrd_arg:   dictionary of other matplot parameters
    """
    ax.hist(data, bins=bins, histtype='step',  **kwrd_arg)


def find_min_btw_peaks(data, bins, prominence, distance, plot=True):
    """
    Finds the minimun between two peaks, using 'find_peaks()' function. \n
    Parameters
    ----------
    data:       data to be transformed into histogram and of which to find the peaks
    bins:       matplot bins options e.g. int (number of bins), list (bin edges)
    prominence: "height" of the peak compared to neighbouring data
    distance:   min distance between peaks
    plot:       boolean, default False, if the plot will to be shown

    Returns
    -------
    x_min:      x position of the minimum
    """
    hist, bins_hist, _ = plt.hist(data, bins=bins, histtype='step')
    bins_centers = (bins_hist[1:]+bins_hist[:-1])/2
    peaks_idx, _ = find_peaks(hist, prominence=prominence, distance=distance)
    plt.plot(bins_centers[peaks_idx], hist[peaks_idx], 'x', markersize=10, color='k', label='Peaks')
    try:
        global_min, _ = find_peaks(-hist[peaks_idx[0]:peaks_idx[-1]], prominence=prominence, distance=distance)
    except:
        print("Didn't find 2 peaks")
        return
    try: x_min = bins_centers[global_min+peaks_idx[0]][0]
    except:
        print("No x_min found")
        return
    plt.plot(x_min, hist[global_min+peaks_idx[0]][0], 'o', markersize=10, color='b', label='Mimimum: %.1f'%x_min)
    if not plot: plt.close()
#     y_min = hist[global_min+peaks_idx[0]][0]
    return  x_min#, y_min


def find_min_kde(data, x_axis):
    """
    Finds the minimum between two peaks, using a normal kernel distribution 'scipy.stats.gaussian_kde()' \n
    Parameters
    ----------
    data:       data to find the minimum
    x_axis:     array, x_axis to evaluate the gaussian_kde

    Returns
    -------
    minimum:    x position of the minimum (evaluated on the x_axis)
    """
    kde = gaussian_kde(dataset=data.to_numpy()).evaluate(x_axis)
    peaks = find_peaks(kde)[0]
    try: minimun = find_peaks(-kde[peaks[0]:peaks[1]])
    except: 
        print("Didn't find 2 peaks")
        return
    return x_axis[minimun[0]+peaks[0]]


def find_edges(data, bins='auto', plot=False):
    """
    Finds the 'edges' of the sensor using the gradient of the hits distribution. \n
    Parameters
    ----------
    data:       data to be put into histogram to find the edges
    bins:       matplot bins options e.g. int (number of bins), list (bin edges)
    plot:       boolean, if the plot should be shown

    Returns
    -------
    left_edge:  left edge 
    right_edge: right edge
    """
    hist, bins_points, _ = plt.hist(data, bins=bins, histtype='step')
    bins_centers = (bins_points[1:]+bins_points[:-1])/2
    left_edge = bins_centers[np.argmax(np.gradient(hist))]
    right_edge = bins_centers[np.argmin(np.gradient(hist))]
    if not plot: plt.close()
    return left_edge, right_edge


def my_and(x,y):
    """I need a 'and' function for combining dataframes"""
    return x and y


def efficiency(data, threshold, percentage=True):
    """
    Efficiency of the data: data greater than threshold value / total data. \n
    Parameters
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
    return (sum(data>threshold) / data.size) * factor


def efficiency_error(data, threshold):
    """
    Efficiency of the data, including error (adjusted as in https://arxiv.org/pdf/physics/0701199v1.pdf) \n
    Parameters
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
    eff = (k+1)/(n+2) # this is the mean value, most probable value is still k/n
    var = ((k+1)*(k+2)/((n+2)*(n+3)) - (k+1)**2/(n+2)**2 )
    return (eff, var**0.5)


def efficiency_k_n(k,n):
    """
    Efficiency and its error (as in https://arxiv.org/pdf/physics/0701199v1.pdf) \n
    Parameters
    ----------
    k:      int, selected elements
    n:      int, total elements

    Returns
    -------
    efficiency: efficiency=(k+1)/(n+2)
    error:      error=variance**0.5; variance=(k+1)*(k+2)/((n+2)*(n+3)) - (k+1)**2/(n+2)**2
    """
    eff = (k+1)/(n+2) # this is the mean value, most probable value is still k/n
    var = ((k+1)*(k+2)/((n+2)*(n+3)) - (k+1)**2/(n+2)**2 )
    return (eff, var**(1/2))


def read_sensors(sensor_file):
    """
    Read the '.pickle' file containing all the list of sensors for each batch, oscilloscope, channel
    
    Parameters
    ----------
    sensor_file:    file path to the .pickle containing the list of sensors

    Returns
    -------
    """
    with open(sensor_file, 'rb') as f:
        sensor_list = pickle.load(f)
    return sensor_list


def plot(df, plot_type, batch, sensors, *,
         bins=(200,200), n_DUT=3, savefig=False, savefig_path='../various plots', savefig_details='',
         **kwrd_arg):
    """
    Function to produce the plots \n
    Parameters
    ----------
    df:             dataframe of the data to plot
    plot_type:      type of plot, options are:
                        '2D_Tracks':    2D plot of the reconstructed tracks
                        '1D_Tracks':    histogram of reconstructed tracks distribution (Xtr and Ytr)
    batch:          batch number
    sensors:        dictionary of the sensors in this batch
    bins:           binning options, (int,int) or (bin_edges_list, bin_edges_list)
    n_DUT:          number of devices under test (3 for each Scope for May 2023)
    savefig:        boolean option to save the plot
    savefig_path:   folder where to save the plot
    savefig_details: optional details for the file (e.g. distinguish cuts)
    """
    match plot_type:        
        case "2D_Tracks":        # 2D tracks plots
            fig, axes = plt.subplots(nrows=1, ncols=n_DUT, figsize=(15,6), sharex='all', sharey='all', dpi=200)
            fig.tight_layout(w_pad=6, h_pad=4)
            for i in range(n_DUT):
                hist, _, _, _, = axes[i].hist2d(df[f"Xtr_{i}"], df[f"Ytr_{i}"], bins=bins, **kwrd_arg)
                if sensors: axes[i].set_title(f"Ch{i+2}\n({sensors['Ch'+f'{i+2}']})")
                else: axes[i].set_title(f"Ch{i+2}")
                axes[i].set_aspect('equal')
                axes[i].set_xlabel('pixels', fontsize=20)
                axes[i].set_ylabel('pixels', fontsize=20)
                
        case "1D_Tracks":        # 1D tracks plots
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5), dpi=200, sharex='all', sharey='all')
            for i in range(n_DUT):
                add_histogram(axes[0], df[f"Xtr_{i}"], label=f"Xtr_{i}", **kwrd_arg)
                add_histogram(axes[1], df[f"Ytr_{i}"], label=f"Ytr_{i}", **kwrd_arg)
                axes[0].legend(fontsize=16)
                axes[1].legend(fontsize=16)
                
    fig.suptitle(f"{plot_type}, batch: {batch}", fontsize=24, y=1.1)

    # Efficiency plot

    # Efficiency Xtr Ytr

    if savefig: fig.savefig(f"{savefig_path}/{plot_type}_{batch}{savefig_details}.jpg")



# if __name__ == '__main__':
#     main()