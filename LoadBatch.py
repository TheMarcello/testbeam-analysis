import numpy as np # NumPy
import matplotlib.pylab as plt # Matplotlib plots
# import matplotlib.patches as mpatches
import pandas as pd # Pandas
import uproot
import pickle

# import awkward as ak
# import mplhep as hep
# import argparse     # to get arguments from command line executing file.py
import os # read directories etc.
from scipy.signal import find_peaks, gaussian
# from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
# import pylandau
# import re
# import copy
from wrapt_timeout_decorator import timeout


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
    return df


def plot_histogram(data, bins='auto', poisson_err=False, error_band=False, fig_ax=None, label=None, **kwrd_arg):
    """
    Plot a simple (list of) histogram with optionally the poissonian error. \n
    Parameters
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
    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=(10,6), dpi=300)
    ax.grid('--')
    hist, bins_points, info = ax.hist(data, bins=bins, histtype='step', label=label)#, **kwrd_arg)
    if (poisson_err):      ### adding the poissonian error (sqrt(hist_point)
        bins_centers = (bins_points[1:]+bins_points[:-1])/2
        errorbar_parameters = {'markersize':0, 'linewidth':0, 'alpha':0.5,'ecolor':'k', 'elinewidth':0.3, 'capsize':1, 'errorevery':2}
        errorbar_parameters.update(kwrd_arg)  ### this overrides the default options
        if (np.shape(np.shape(data))[0]>1): ### a bit convoluted but checks the dimensions of the data
            for single_hist in hist:     ### in case data is a list of data
                y_error = single_hist**0.5
                if error_band:  ### I just mask all the errorbars
                    filled_band_parameters = {'alpha':0.5, 'linestyle':'--'}
                    errorbar_parameters.update({'elinewidth':0,'capsize':0,'errorevery':1})
                    ax.fill_between(bins_centers, single_hist-y_error, single_hist+y_error, **filled_band_parameters)
                ax.errorbar(bins_centers, single_hist, yerr=single_hist**0.5, **errorbar_parameters)
        else:
            y_error = hist**0.5
            if error_band:      ### I just mask all the errorbars
                filled_band_parameters = {'alpha':0.5, 'linestyle':'--'}
                errorbar_parameters.update({'elinewidth':0,'capsize':0})
                ax.fill_between(bins_centers, hist-y_error, hist+y_error, **filled_band_parameters)
            ax.errorbar(bins_centers, hist, yerr=y_error, **errorbar_parameters)
    return hist, bins_points, info, fig, ax


@timeout(20) ### max seconds of running
def time_limited_kde_evaluate(kde, x_axis):
    """Evaluating a kernel density estimate on the points of x_axis, it includes a timeout error if it runs too long"""
    return kde.evaluate(x_axis)


def find_min_btw_peaks(data, bins, peak_prominence=None, min_prominence=None, plot=True,
                       savefig=False, savefig_path='../various plots/', savefig_details='', fig_ax=None):#, rec_depth=0):
    """
    Finds the minimun between two peaks, using 'find_peaks()' function. \n
    Parameters
    ----------
    data:           data to be transformed into histogram and of which to find the peaks (e.g. df['pulseHeight_1'])
    bins:           matplot bins options e.g. int (number of bins), list (bin edges)
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
    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=(15,10), dpi=200)
    ax.semilogy()

    # hist, bins_hist, _ = ax.hist(data, bins=bins, histtype='step')#, density=True)
    hist, bins_hist, _, fig, ax = plot_histogram(data, bins=bins, fig_ax=(fig,ax))
    bins_centers = (bins_hist[1:]+bins_hist[:-1])/2
        ### I try to find the normalization factor so I can 'denormalize' the kde
    density_factor = sum(hist)*np.diff(bins_hist)
        ### Use kernel density estimate instead
    kde = gaussian_kde(data)
    number_of_tries = 5
    for i in range(number_of_tries):
        try:
            smoothed_hist = time_limited_kde_evaluate(kde, bins_centers) * density_factor 
        except:
            print(f"Evaluating kde timeout n°: {i+1}. Trying with 1/2 number of points")
            bins_centers = bins_centers[::2] ### now take only half of points to be evaluated
            density_factor = density_factor[::2]
            if i==(number_of_tries-1):
                print("Giving up estimating kde")
                return None
        else:
            break

    if not peak_prominence: peak_prominence = np.max(hist)/100
    if not min_prominence: min_prominence = np.max(hist)/100
    ### find (hopefully two) peaks and plot them
    peaks_idx, info_peaks = find_peaks(smoothed_hist, prominence=peak_prominence)
    ax.plot(bins_centers, smoothed_hist, linewidth=1, label='Smoothed hist')
    ax.plot(bins_centers[peaks_idx], smoothed_hist[peaks_idx], 'x', markersize=10, color='k', label='Peaks')
    
    rec_depth = 0
    ### the recursion makes it repeat kde.evaluate(), which is very slow, let's just try a loop
    while(rec_depth<10):
        if len(peaks_idx)>=2: ### find the minimum
            local_min, _ = find_peaks(-smoothed_hist[peaks_idx[0]:peaks_idx[1]], prominence=min_prominence)
        else:    ### if it doesn't work it's because only one peak was found
            # print(f"Two peaks not found, retrying...")
            ax.clear()
            peak_prominence *= 0.7    ### reduce prominence if the peaks are not found
            rec_depth += 1
            if rec_depth==10:
                print(f"Two peaks not found after {rec_depth} iterations \n info:{info_peaks}")
                return None
            continue

        if len(local_min)==1:
            break
        elif len(local_min)==0:
            print(f"No minimum found, retrying...")
            ax.clear()
            min_prominence *= 0.7     ### reduce prominence if the min is not found
            continue
        elif len(local_min)>1:
            print(f"More than one minimum found at: {[bins_centers[min_idx+peaks_idx[0]] for min_idx in local_min]}")
            break

    x_min = bins_centers[local_min[0]+peaks_idx[0]]
    ax.plot(x_min, smoothed_hist[local_min[0]+peaks_idx[0]], 'o', markersize=10, color='r',
            label='Mimimum: %.1f'%x_min, alpha=.7)
    ax.legend(fontsize=16)
    if savefig: fig.savefig(f"{savefig_path}find_min_btw_peaks{savefig_details}.jpg")
    if not plot: plt.close()
    return  x_min 


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
    eff = (k+1)/(n+2) ### this is the mean value, most probable value is still k/n
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
    eff = (k+1)/(n+2) ### this is the mean value, most probable value is still k/n
    var = ((k+1)*(k+2)/((n+2)*(n+3)) - (k+1)**2/(n+2)**2 )
    return (eff, var**(1/2))


def read_pickle(file):
    """
    Read the '.pickle' file containing all the list of sensors for each batch, oscilloscope, channel
    
    Parameters
    ----------
    sensor_file:    file path to the .pickle c

    Returns
    pickle_dict:    (usually dict) contained in the pickle file
    -------
    """
    with open(file, 'rb') as f:
        pickle_dict = pickle.load(f)
    return pickle_dict


def plot(df, plot_type, batch, *, sensors=None, bins=200, bins_find_min='rice', n_DUT=3,
         savefig=False, savefig_path='../various plots', savefig_details='', fig_ax=None,
         **kwrd_arg):
    """
    Function to produce the plots \n
    Parameters
    ----------
    df:             FULL dataframe of the data to plot (each plot_type select the data it needs)
    plot_type:      type of plot, options are:
                        '2D_Tracks':    2D plot of the reconstructed tracks
                        '1D_Tracks':    histogram of reconstructed tracks distribution (Xtr and Ytr)
                        'pulseHeight':  histogram of the pulseHeight of all channels (log scale)
                        '2D_Sensors':   2D plot of tracks with pulseHeight cut (highlighting the sensors)
    batch:          batch number
    sensors:        dictionary of the sensors in this batch
    bins:           binning options, (int,int) or (bin_edges_list, bin_edges_list), can be different for different plot_type
    bins_find_min:  binning options for the find_min_btw_peaks function (in '2D_Sensors')  
    n_DUT:          number of devices under test (3 for each Scope for May 2023)
    savefig:        boolean option to save the plot
    savefig_path:   folder where to save the plot
    savefig_details: optional details for the file name (e.g. distinguish cuts)

    Returns
    -------
    fig, axes:        figure and axis objects so that more manipulation can be done
    """
    match plot_type:        
        case "2D_Tracks":        ### 2D tracks plots
            fig, axes = plt.subplots(nrows=1, ncols=n_DUT, figsize=(15,6), sharex='all', sharey='all', dpi=200)
            fig.tight_layout(w_pad=6, h_pad=4)
            for i in range(n_DUT):
                hist, _, _, _, = axes[i].hist2d(df[f"Xtr_{i}"], df[f"Ytr_{i}"], bins=bins, **kwrd_arg)
                if sensors: axes[i].set_title(f"Ch{i+2}\n({sensors[f'Ch{i+2}']})")
                else: axes[i].set_title(f"Ch{i+2}")
                axes[i].set_aspect('equal')
                axes[i].set_xlabel('pixels', fontsize=20)
                axes[i].set_ylabel('pixels', fontsize=20)
                
        case "1D_Tracks":        ### 1D tracks plots
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6), dpi=200, sharey='all')
            for i in range(n_DUT):
                plot_histogram(df[f"Xtr_{i}"], label=f"Xtr_{i}", bins=bins[0], fig_ax=(fig,axes[0]), **kwrd_arg)
                plot_histogram(df[f"Ytr_{i}"], label=f"Ytr_{i}", bins=bins[1], fig_ax=(fig,axes[1]), **kwrd_arg)
            axes[0].legend(fontsize=16)
            axes[1].legend(fontsize=16)
            # axes[0].grid('--')    # this is already in plot_histogram
            # axes[1].grid('--')
        
        case "pulseHeight":       ### PulseHeight plot
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15,10), dpi=200)
            for i in range(n_DUT+1):
                plot_histogram(df[f"pulseHeight_{i}"], poisson_err=True, error_band=True, bins=bins, fig_ax=(fig,axes), label=f"sensor: {sensors[f'Ch{i+1}']}", **kwrd_arg)
            axes.semilogy()
            axes.set_xlabel("PulseHeight [mV]", fontsize=20)
            axes.set_ylabel("Events (log)", fontsize=20)
            axes.set_title(f"PulseHeight (no cut), batch {batch}, bins {bins}", fontsize=24, y=1.05)
            axes.set_xlim(left=-10)
            axes.legend(fontsize=20)
            # axes.grid('--') # this is already in plot_histogram
            
        case "2D_Sensors":        ### 2D tracks plots filtering some noise out (pulseHeight cut)
            fig, axes = plt.subplots(nrows=2, ncols=n_DUT, figsize=(20,12), sharex=False, sharey=False, dpi=200)
            fig.tight_layout(w_pad=6, h_pad=4)
            for i in range(n_DUT):
                print(f"DUT_{i}")                   ### BINS: scott, rice or sqrt; stone seems slow, rice seems the fastest
                minimum = find_min_btw_peaks(df[f"pulseHeight_{i+1}"], bins=bins_find_min, plot=True, fig_ax=(fig,axes[0,i]),
                                             savefig=False, savefig_details=f"_{batch}_DUT{i}"+savefig_details)
                axes[0,i].set_xlabel('mV')
                axes[0,i].set_ylabel('Events')
                if not minimum:
                    print("No minimum found, no 2D plot")
                    plot_histogram(df[f"pulseHeight_{i+1}"], bins=bins_find_min, poisson_err=True, error_band=True, fig_ax=(fig, axes[0,i]))
                    axes[0,i].semilogy()
                    continue
                pulseHeight_filter = np.where(df[f"pulseHeight_{i+1}"]>minimum)
                hist, _, _, _, = axes[1,i].hist2d(df[f"Xtr_{i}"].iloc[pulseHeight_filter], df[f"Ytr_{i}"].iloc[pulseHeight_filter],
                                                bins=bins, **kwrd_arg)
                if sensors: axes[1,i].set_title(f"Ch{i+2}, "+"cut: %.1f"%minimum+f"mV \n({sensors[f'Ch{i+2}']})")
                else: axes[1,i].set_title(f"Ch{i+2}")
                axes[1,i].set_aspect('equal')
                axes[1,i].set_xlabel('pixels', fontsize=20)
                axes[1,i].set_ylabel('pixels', fontsize=20)
                    
        case other:
            print("""No plot_type found, options are:
            '2D_Tracks', '1D_Tracks', 'pulseHeight', '2D_Sensors' """)
            return
        
    fig.suptitle(f"{plot_type}, batch: {batch} {savefig_details}", fontsize=24, y=1.15)

    if savefig: fig.savefig(f"{savefig_path}/{plot_type}_{batch}{savefig_details}.jpg", bbox_inches="tight")
    return fig, axes

    # Efficiency 2D plot

    # Efficiency Xtr Ytr

# if __name__ == '__main__':
#     main()