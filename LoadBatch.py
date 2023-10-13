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


PIXEL_SIZE = 0.0185 #mm

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
    del df_ak, tree         ### I am trying to fix the memory leak (not sure this is relevant)
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
        errorbar_parameters = {'markersize':0, 'linewidth':0, 'alpha':0.5,'ecolor':'k', 'elinewidth':0.3, 'capsize':1, 'errorevery':5}
        errorbar_parameters.update(kwrd_arg)  ### this adds the default options (and overrides them if repeated)
        if (np.shape(np.shape(data))[0]>1): ### a bit convoluted but checks the dimensions of the data
            for single_hist in hist:     ### in case data is a list of data
                y_error = single_hist**0.5
                if error_band:  ### I just mask all the errorbars
                    filled_band_parameters = {'alpha':0.5, 'linestyle':'--'}
                    errorbar_parameters.update({'elinewidth':0,'capsize':0,'errorevery':1})     ### defaults specific to 
                    ax.fill_between(bins_centers, single_hist-y_error, single_hist+y_error, **filled_band_parameters)#, label=f"{label} error")
                ax.errorbar(bins_centers, single_hist, yerr=single_hist**0.5, **errorbar_parameters)
        else:
            y_error = hist**0.5
            if error_band:      ### I just mask all the errorbars
                filled_band_parameters = {'alpha':0.5, 'linestyle':'--'}
                errorbar_parameters.update({'elinewidth':0,'capsize':0,'errorevery':1})
                ax.fill_between(bins_centers, hist-y_error, hist+y_error, **filled_band_parameters)
            ax.errorbar(bins_centers, hist, yerr=y_error, **errorbar_parameters)
    return hist, bins_points, info, fig, ax


@timeout(20) ### max seconds of running
def time_limited_kde_evaluate(kde, x_axis):
    """Evaluating a kernel density estimate on the points of x_axis, it includes a timeout error if it runs too long"""
    return kde.evaluate(x_axis)


def find_min_btw_peaks(data, bins, peak_prominence=None, min_prominence=None, plot=True,
                       savefig=False, savefig_path='../various plots/', savefig_details='', fig_ax=None):#, recursion_depth=0):
    """
    Finds the minimun between two peaks, using 'find_peaks()' function. \n
    Parameters
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
    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=(15,10), dpi=200)
    # hist, bins_hist, _, fig, ax = plot_histogram(data, bins=bins, fig_ax=(fig,ax))
    hist, bins_hist, _, fig, ax = plot_histogram(data, bins=bins, poisson_err=True, error_band=True,
                                                 fig_ax=(fig,ax))
    ax.semilogy()
    ax.set_ylim(10**(-2), 1.5*np.max(hist))
    bins_centers = (bins_hist[1:]+bins_hist[:-1])/2
        ### Find the normalization factor so I can 'denormalize' the kde
    density_factor = sum(hist)*np.diff(bins_hist)
    ### Use KERNEL DENSITY ESTIMATE
    kde = gaussian_kde(data)
    number_of_tries = 5
    for i in range(number_of_tries):
        try:
            smoothed_hist = time_limited_kde_evaluate(kde, bins_centers) * density_factor 
        except:          ### now take only half of the points to be evaluated
            print(f"Evaluating kde timeout n°: {i+1}. Trying with 1/2 number of points")
            bins_centers = bins_centers[::2] 
            density_factor = density_factor[::2]
            if i==(number_of_tries-1):
                print("Giving up evaluating kde")
                return None
        else:
            break
    ### it plots even if it cannot find the peaks and/or min
    ax.plot(bins_centers, smoothed_hist, linewidth=1, label='Smoothed hist')
    if not peak_prominence: peak_prominence = np.max(hist)/100
    if not min_prominence: min_prominence = np.max(hist)/100
    recursion_depth = 0  # 0 or 1, not sure which one gives 'max_recursion' tries
    max_recursion = 20
    ### 
    while(recursion_depth<max_recursion):
            ### find (hopefully two) peaks and plot them
        peaks_idx, info_peaks = find_peaks(smoothed_hist, prominence=peak_prominence)
        global_max_idx = np.argmax(smoothed_hist)
        if (len(peaks_idx)==1) and (global_max_idx!=peaks_idx[0]):  ### find_peaks() does not find max values at edges,
            peaks_idx = np.append(global_max_idx, peaks_idx)        ### so I the global max (if not identical to the peak found)
            
        if len(peaks_idx)>=2:       ### find the minimum
            local_min, info_min = find_peaks(-smoothed_hist[peaks_idx[0]:peaks_idx[1]], prominence=min_prominence)
            
        else:    ### if it doesn't work it's because only one peak was found
            # print(f"Two peaks not found, retrying...")
            recursion_depth += 1
            if recursion_depth==max_recursion:
                print(f"Two PEAKS not found after {recursion_depth} iterations \n INFO :{info_peaks}")
                return None
            peak_prominence *= 0.5    ### reduce prominence if the peaks are not found
            continue
        if len(local_min)==1:
            break
        elif len(local_min)>1:
            print(f"More than one minimum found at: {[bins_centers[min_idx+peaks_idx[0]] for min_idx in local_min]}")
            break
        elif len(local_min)==0:
            recursion_depth += 1
            # print(f"No minimum found, retrying...")
            if recursion_depth==max_recursion:
                print(f"No MIN found after {recursion_depth} iterations")# \n INFO :{info_min}")
                return None
            min_prominence *= 0.5       ### reduce prominence if the min is not found

    x_min = bins_centers[local_min[0]+peaks_idx[0]]
    ax.plot(bins_centers[peaks_idx], smoothed_hist[peaks_idx], 'x', markersize=10, color='k', label='Peaks')
    ax.plot(x_min, smoothed_hist[local_min[0]+peaks_idx[0]], 'o', markersize=10, color='r',
            label='Mimimum: %.1f'%x_min, alpha=.7)
    ax.legend(fontsize=16)
    if savefig: fig.savefig(f"{savefig_path}find_min_btw_peaks{savefig_details}.jpg")
    if not plot: plt.close()
    return  x_min 


def find_edges(data, bins='rice', use_kde=True, plot=False):
    """
    Finds the 'edges' of the sensor using the gradient of the hits distribution. \n
    Parameters
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
    hist, bins_points, _ = plt.hist(data, bins=bins, histtype='step')
    bins_centers = (bins_points[1:]+bins_points[:-1])/2
    if use_kde:
        kde = gaussian_kde(hist)
        try:
            values = time_limited_kde_evaluate(kde)
        except:
            print("in 'find_edges()': KDE timed out, using normal hist")
            values = hist
    else:
        values = hist
    left_edge = bins_centers[np.argmax(np.gradient(values))]
    right_edge = bins_centers[np.argmin(np.gradient(values))]
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


def geometry_mask(df, bins, bins_find_min, DUT_number):
    """Geometry boolean mask
    """
    i = DUT_number
    min_value = find_min_btw_peaks(df[f"pulseHeight_{i+1}"], bins=bins_find_min, plot=False)
    pulseHeight_filter = np.where(df[f"pulseHeight_{i+1}"]>min_value)
    Xtr_cut = df[f"Xtr_{i}"].iloc[pulseHeight_filter]       ### X tracks with applied pulseHeight
    Ytr_cut = df[f"Ytr_{i}"].iloc[pulseHeight_filter]
    left_edge, right_edge = find_edges(Xtr_cut, bins=bins[0], use_kde=True, plot=False)
    bottom_edge, top_edge = find_edges(Ytr_cut, bins=bins[1], use_kde=True, plot=False)
    xgeometry = np.logical_and(df[f"Xtr_{i}"]>left_edge, df[f"Xtr_{i}"]<right_edge)
    ygeometry = np.logical_and(df[f"Ytr_{i}"]>bottom_edge, df[f"Ytr_{i}"]<top_edge)
    bool_geometry = np.logical_and(xgeometry, ygeometry)
    return bool_geometry


def plot(df, plot_type, batch, *, sensors=None, bins=None, bins_find_min='rice', n_DUT=3,
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
                        '2D_Sensors':   pulseHeight cut plot + 2D plot of tracks with cut (highlighting the sensors)
    batch:          batch number
    sensors:        dictionary of the sensors in this batch
    bins:           binning options, (int,int) or (bin_edges_list, bin_edges_list), different default for each plot_type
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
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(nrows=1, ncols=n_DUT, figsize=(15,6), sharex='all', sharey='all', dpi=200)
            fig.tight_layout(w_pad=6, h_pad=4)
            if not bins: bins = (200,200)   ### default binning
            for i in range(n_DUT):
                hist, _, _, _, = axes[i].hist2d(df[f"Xtr_{i}"], df[f"Ytr_{i}"], bins=bins, **kwrd_arg)
                if sensors: axes[i].set_title(f"Ch{i+2}\n({sensors[f'Ch{i+2}']})")
                else: axes[i].set_title(f"Ch{i+2}")
                axes[i].set_aspect('equal')
                axes[i].set_xlabel('pixels', fontsize=20)
                axes[i].set_ylabel('pixels', fontsize=20)
                secx = axes[i].secondary_xaxis('top', functions=(lambda x: x*PIXEL_SIZE, lambda x: x*PIXEL_SIZE))
                secy = axes[i].secondary_yaxis('right', functions=(lambda x: x*PIXEL_SIZE, lambda x: x*PIXEL_SIZE))
                secx.set_xlabel('mm', fontsize=20)
                secy.set_ylabel('mm', fontsize=20)

        case "1D_Tracks":        ### 1D tracks plots
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,6), dpi=200, sharey='all')
            if not bins: bins = (200,200)   ### default binning
            for i in range(n_DUT):
                plot_histogram(df[f"Xtr_{i}"], label=f"Xtr_{i}", bins=bins[0], fig_ax=(fig,axes[0]), **kwrd_arg)
                plot_histogram(df[f"Ytr_{i}"], label=f"Ytr_{i}", bins=bins[1], fig_ax=(fig,axes[1]), **kwrd_arg)
            for ax in axes:     ### modify both axes
                ax.legend(fontsize=16)
                ax.semilogy()
                ax.set_xlabel('pixels', fontsize=20)
                ax.set_ylabel('Events (log)', fontsize=20)

        case "pulseHeight":       ### PulseHeight plot
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15,10), dpi=200)
            if not bins: bins = 'rice'
            for i in range(n_DUT+1):
                if sensors: sensor_label=f"sensor: {sensors[f'Ch{i+1}']}"
                else: sensor_label=f'Ch{i+1}'
                plot_histogram(df[f"pulseHeight_{i}"], poisson_err=True, error_band=True, bins=bins, fig_ax=(fig,axes), label=sensor_label, **kwrd_arg)
            axes.semilogy()
            axes.set_xlabel("PulseHeight [mV]", fontsize=20)
            axes.set_ylabel("Events (log)", fontsize=20)
            axes.set_title(f"PulseHeight (no cut), batch {batch}, bins {bins}", fontsize=24, y=1.05)
            axes.set_xlim(left=-10)
            axes.legend(fontsize=20)
            
        case "2D_Sensors":        ### 2D tracks plots filtering some noise out (pulseHeight cut)
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(nrows=2, ncols=n_DUT, figsize=(20,12), sharex=False, sharey=False, dpi=200)
            if not bins: bins = (200,200)   ### default binning
            fig.tight_layout(w_pad=6, h_pad=4)
            for i in range(n_DUT):
                print(f"DUT_{i}")                   ### BINS: scott, rice or sqrt; stone seems slow, rice seems the fastest
                minimum = find_min_btw_peaks(df[f"pulseHeight_{i+1}"], bins=bins_find_min, plot=True, fig_ax=(fig,axes[0,i]),
                                             savefig=False, savefig_details=f"_{batch}_DUT{i}"+savefig_details)
                axes[0,i].set_xlabel('mV')
                axes[0,i].set_ylabel('Events')
                if not minimum:
                    print("No minimum found, no 2D plot")
                    axes[0,i].set_title(f"Ch{i+2} \n({sensors[f'Ch{i+2}']})")
                    continue
                if sensors: axes[0,i].set_title(f"Ch{i+2}, "+"cut: %.1f"%minimum+f"mV \n({sensors[f'Ch{i+2}']})")
                else: axes[0,i].set_title(f"Ch{i+2}")
                pulseHeight_filter = np.where(df[f"pulseHeight_{i+1}"]>minimum)
                axes[1,i].hist2d(df[f"Xtr_{i}"].iloc[pulseHeight_filter], df[f"Ytr_{i}"].iloc[pulseHeight_filter],
                                                bins=bins, **kwrd_arg)
                axes[1,i].set_aspect('equal')
                axes[1,i].set_xlabel('pixels', fontsize=20)
                axes[1,i].set_ylabel('pixels', fontsize=20)

        case "1D_Efficiency":
            for key, value in kwrd_arg.items():
                match key:
                    case 'threshold_charge': threshold_charge=value
                    case 'transimpedance':   transimpedance=value
                    case other: print(f"invalid argument: {other}")
            coord = ['X','Y']
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(nrows=2, ncols=n_DUT, figsize=(20,12), sharex=False, sharey=False, dpi=200)
            for i in range(n_DUT):
### I can probably make this part into a function: 'geometry_mask(df, bins, bins_find_min)'
### that returns bool_geometry
                bool_geometry = geometry_mask(df, bins, bins_find_min, DUT_number=i)    ### this is a boolean mask of the selected positions
                geometry = np.where(bool_geometry)   ### this is the array of indices of the selected values
            ### Create a boolean mask for events above the threshold
                events_above_threshold = df[f"charge_{i}"].iloc[geometry]/transimpedance > threshold_charge
            ### Calculate the number of events above threshold in each bin
                for coord_idx, XY in enumerate(coord):
                    above_threshold = np.where(np.logical_and(bool_geometry, events_above_threshold))
                    total_events_in_bin, bins_edges, _, _, _ = plot_histogram(df[f"{XY}tr_{i}"].iloc[geometry], bins=bins[coord_idx], fig_ax=(fig, axes[coord_idx,i]))
                    events_above_threshold_in_bin, _, _, _, _ = plot_histogram(df[f"{XY}tr_{i}"].iloc[above_threshold], bins=bins[coord_idx], fig_ax=(fig, axes[coord_idx,i]))
                    axes[coord_idx, i].clear()
                    bins_centers = (bins_edges[:-1]+bins_edges[1:])/2
                    eff, err = efficiency_k_n(events_above_threshold_in_bin, total_events_in_bin)
                    axes[coord_idx,i].step(bins_centers, eff, where='mid', label=f"Ch{i+1}")
                    # sigma_coeff = 1
                    # axes[coord_idx,i].errorbar(bins_centers, eff, yerr=sigma_coeff*err, elinewidth=1.5, markersize=0, linewidth=0,
                    #             label=f"error: {sigma_coeff}$\sigma$")
                    axes[coord_idx,i].set_title(f"{XY} axis projection of efficiency", fontsize=24, y=1.05)
                    axes[coord_idx,i].set_xlabel(f"{XY} position (pixels)", fontsize=20)
                    axes[coord_idx,i].set_ylabel("Efficiency", fontsize=20)
                    # axes[coord_idx,i].legend()
                    if XY=='X': axes[coord_idx,i].set_xlim(left_edge,right_edge)
                    if XY=='Y': axes[coord_idx,i].set_xlim(bottom_edge,top_edge)
                    # axes[coord_idx,i].set_ylim(bottom_edge,top_edge)

        case "2D_Efficiency":
            if fig_ax:  fig, axes = fig_ax
            else:       fig, axes = plt.subplots(nrows=1, ncols=n_DUT, figsize=(15,6), sharex=False, sharey=False, dpi=200)
            fig.tight_layout(w_pad=6, h_pad=4)
            for key, value in kwrd_arg.items():
                match key:
                    case 'threshold_charge': threshold_charge=value
                    case 'transimpedance':   transimpedance=value
                    case other: print(f"invalid argument: {other}")
            for i in range(n_DUT):
                min_value = find_min_btw_peaks(df[f"pulseHeight_{i+1}"], bins=bins_find_min, plot=False)
                pulseHeight_filter = np.where(df[f"pulseHeight_{i+1}"]>min_value)
                Xtr_cut = df[f"Xtr_{i}"].iloc[pulseHeight_filter]       ### X tracks with applied pulseHeight
                Ytr_cut = df[f"Ytr_{i}"].iloc[pulseHeight_filter]
                left_edge, right_edge = find_edges(Xtr_cut, bins=bins[0], use_kde=True, plot=False)
                bottom_edge, top_edge = find_edges(Ytr_cut, bins=bins[1], use_kde=True, plot=False)
                xgeometry = np.logical_and(df[f"Xtr_{i}"]>left_edge, df[f"Xtr_{i}"]<right_edge)
                ygeometry = np.logical_and(df[f"Ytr_{i}"]>bottom_edge, df[f"Ytr_{i}"]<top_edge)
                bool_geometry = np.logical_and(xgeometry, ygeometry)    ### this is a boolean mask of the selected positions
                geometry = np.where(bool_geometry)   ### this is the array of indices of the selected values
                total_events_in_bin, x_edges, y_edges, _ = axes[i].hist2d(df[f"Xtr_{i}"].iloc[geometry], df[f"Ytr_{i}"].iloc[geometry], bins=bins)
        ### Create a boolean mask for events above the threshold
                events_above_threshold = df[f"charge_{i+1}"].iloc[geometry]/transimpedance > threshold_charge
        ### Calculate the number of events above threshold in each bin
                above_threshold = np.where(np.logical_and(bool_geometry, events_above_threshold))
                events_above_threshold_in_bin, _, _, _ = axes[i].hist2d(df[f"Xtr_{i}"].iloc[above_threshold], df[f"Ytr_{i}"].iloc[above_threshold], bins=bins)
                efficiency_map = np.divide(events_above_threshold_in_bin, total_events_in_bin,
                                        where=total_events_in_bin!=0,
                                        out=np.zeros_like(events_above_threshold_in_bin))*100 # in percentage
                axes[i].clear()
                # fig = plt.figure(figsize=(7.28, 6), dpi=200)    ### I need to be able to plot 3 in the same
                axes[i].imshow(efficiency_map.T, origin='lower',# extent=[left_edge, right_edge, bottom_edge, top_edge],
                        aspect='equal', vmin=0, vmax=100)
                # axes[i].colorbar(label='Efficiency')
                axes[i].set_xlabel('X Position', fontsize=20)
                axes[i].set_ylabel('Y Position', fontsize=20)
                # axes = None


        case other:
            print("""No plot_type found, options are:
            '2D_Tracks', '1D_Tracks', 'pulseHeight', '2D_Sensors' """)
            return
    
    fig.suptitle(f"{plot_type}, batch: {batch} {savefig_details}", fontsize=24, y=1.15)
    plt.show()
    if savefig: fig.savefig(f"{savefig_path}/{plot_type}_{batch}{savefig_details}.jpg", bbox_inches="tight")
    return fig, axes

    # Efficiency Xtr Ytr


    # Efficiency 2D plot


# if __name__ == '__main__':
#     main()