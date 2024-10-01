import numpy as np # NumPy
import matplotlib.pylab as plt # Matplotlib plots
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import mpl_scatter_density     # density scatter plots
import pandas as pd # Pandas
import uproot
import pickle
import logging
import argparse     # to get arguments from command line executing file.py

import os # read directories etc.
from scipy.signal import find_peaks, gaussian
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import pylandau

import time
from timeout_decorator import timeout

from importlib import reload # to reload modules
import LoadBatch
reload(LoadBatch)
from LoadBatch import *
from SensorClasses import *

### use the argparse package to parse command line arguments
parser = argparse.ArgumentParser(description='Plot a single batch, given batch number and other arguments')
parser.add_argument('-batch', type=int, help='Batch number')
parser.add_argument('-S', type=bool, default=True, help='SAVE option to save plots (or not)')
parser.add_argument('-fit_charge', default=False, help='if charge should be fitted (with ROOT)')
parser.add_argument('-CFD', default=False, help='option for CFD comparison plots')
# parser.add_argument('-d', help='Output directory')
args = parser.parse_args()

this_batch = args.batch
fit_charge = args.fit_charge
SAVE = args.S
CFD_comparison = args.CFD

# if args.d:
#     dir_path = args.d
# else:
dir_path = f'/home/marcello/Desktop/Radboud_not_synchro/Master_Thesis/various plots/all batches/{this_batch}'



pd.set_option('display.max_columns', None)
# logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] - %(message)s')
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - \t %(message)s')

### choose the bins so that they match the MIMOSA pixels (which are just the coordinates)

### Load the dictionary of sensor names and runs
dict_of_batches = read_pickle("/home/marcello/Desktop/Radboud_not_synchro/Master_Thesis/testbeam-analysis/dict_of_batches.pickle")

ROOT_fit_dir = f"/home/marcello/Desktop/Radboud_not_synchro/Master_Thesis/testbeam-analysis/ROOT Langaus fit/"

# list_of_batches = list(dict_of_batches.keys())
# list_of_batches.sort()


start = time.time()
### show all information about the batch
SAVE = True
show_plot = False


colormap = ['k','b','g','r']

binning_method = 'rice'
threshold_charge = 4 #fC
charge_bins = 500
eff_lim = (0.4,1)
time_bins = 4000
window_limit = 20e3
n_bootstrap = 50
CFD_values = (20, 50, 70)
axes_size = len(CFD_values)

# for this_batch in [100, 101, 199]:

# display(dict_of_batches[this_batch].__dict__)
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

info_dict = {}
### show all informations about each sensor
for S in ['S1','S2']:
    if dict_of_batches[this_batch].S[S] is None:
        logging.warning(f"No oscilloscope {S} in batch {this_batch}")
        info_dict[(S,ch)] = None
    else:
        for ch, sensor in dict_of_batches[this_batch].S[S].channels.items():
    #         print(f"{S}, {ch}:", sensor.__dict__)
            info_dict[(S,ch)] = sensor.__dict__
info_df = pd.DataFrame(info_dict)
# display(info_df)
# Export DataFrame to a CSV file
info_df.to_csv(os.path.join(dir_path,f'table_data_{this_batch}.csv'), index=True)

print("Batch: ", this_batch)

### save in a presentation folder
# dir_path = pres_path

df = {}  # dictionary containing the two dataframes of the two oscilloscopes
# 'pulseheight' or 'time'
if this_batch in [502, 601, 602, 603, 604, 605, 901, 902, 1001, 1002]:
    use_for_geometry_cut = 'time'
else:
    use_for_geometry_cut = 'pulseheight' 

these_bins = bins_dict[this_batch] #bins1    ### custom bins around the sensors

for S in ['S1','S2']: #"S2" ### the two scopes
    print(S)
    
    DUTs = get_DUTs_from_dictionary(info_dict,S)
    if not DUTs: continue   ### if there are no DUTs in this oscilloscope just go to next

    df[S] = load_batch(this_batch,S)
    print(f'MCP: {dict_of_batches[this_batch].S[S].channels["Ch1"].voltage} V, angle: {dict_of_batches[this_batch].angle}°', 'temperature:%.2f°C'%dict_of_batches[this_batch].temperature)

    ## show full area
    plot(df[S], "2D_Tracks", dict_of_batches[this_batch], S, bins=large_bins,
            n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} all tracks (no cut)', savefig_path=dir_path, fmt='png', show_plot=show_plot)  
    
    ### I PUT THE TRANSIMPEDANCE TO 4700 MANUALLY
    my_transimpedance = 4700 #4700 or 10700

    ###[ ... if dut in DUTs else None for dut in [1,2,3]], it avoids calculating the cuts for the channels with no dut
    geo_cuts = [geometry_mask(df[S], DUT_number=dut, bins=these_bins, bins_find_min='rice', use=use_for_geometry_cut)[0] if dut in DUTs else None for dut in [1,2,3]]
    central_sensor_area_cuts = [geometry_mask(df[S], DUT_number=dut, bins=these_bins, bins_find_min='rice', only_select='center', use=use_for_geometry_cut)[0] if dut in DUTs else None for dut in [1,2,3]]
    time_cuts = [time_mask(df[S], dut, bins=time_bins, mask=geo_cuts[dut-1], n_bootstrap=False, plot=False, savefig=os.path.join(dir_path,f'time_plot_with_geo_cuts_{S}_{this_batch}_DUT{dut}.png'))[0] if dut in DUTs else None for dut in [1,2,3]]
    charge_cut = [df[S][f'charge_{dut}']/my_transimpedance>threshold_charge if dut in DUTs else None for dut in [1,2,3]]

    ### time resolution fit
    [time_mask(df[S], dut, bins=time_bins, n_bootstrap=n_bootstrap, mask=np.logical_and(geo_cuts[dut-1],charge_cut[dut-1]), plot=True,
                                    savefig=os.path.join(dir_path,f'time_plot_with_geo_cuts_{S}_{this_batch}_DUT{dut}.png'))[1]['parameters'] if dut in DUTs else None for dut in [1,2,3]]
    ### time resolution fit with central area cuts
    [time_mask(df[S], dut, bins=time_bins, n_bootstrap=n_bootstrap, mask=np.logical_and(central_sensor_area_cuts[dut-1],charge_cut[dut-1]), plot=True, title_info=' center cut',
                                    savefig=os.path.join(dir_path,f'time_plot_with_center_cuts_{S}_{this_batch}_DUT{dut}.png'))[1]['parameters'] if dut in DUTs else None for dut in [1,2,3]]

    ### I still need pulseHeight cut for the charge fit
    mins = [find_min_btw_peaks(df[S][f"pulseHeight_{dut}"], bins='rice', plot=False) if dut in DUTs else None for dut in [1,2,3]]
    pulse_cuts = [df[S][f'pulseHeight_{dut}']>mins[dut-1] if dut in DUTs else None for dut in [1,2,3]]
    for dut in [1,2,3]:
        if (pulse_cuts[dut-1] is not None) and (np.alltrue(pulse_cuts[dut-1]==False)):
            pulse_cuts[dut-1] = pd.Series(True, index=df[S].index)
        if use_for_geometry_cut == "time":   ### also NOT APPLY a pulseHeight cut if I choose time for geometry cut
            pulse_cuts[dut-1] = pd.Series(True, index=df[S].index)
    ### charge distribution with cuts saved into a file and the fitted with ROOT Landau*Gauss convolution
    all_cuts = [np.logical_and(np.logical_and(geo_cuts[dut-1], time_cuts[dut-1]), pulse_cuts[dut-1]) 
                if dut in DUTs else None for dut in [1,2,3]]
    
    if fit_charge:
        for dut in DUTs:
            np.savetxt(os.path.join(dir_path, f"charge_data_all_cuts_{this_batch}_{S}_{dut}.csv"),
                        df[S][f'charge_{dut}'].loc[all_cuts[dut-1]]/my_transimpedance, delimiter=',')
            os.chdir(ROOT_fit_dir)
            run_root_string = f'root -b -q "charge_fit.C({this_batch},\\"{S}\\",{dut})"'
            os.system(run_root_string)
    
## highlight the sensors with pulseHeight cut
    plot(df[S], "2D_Sensors", dict_of_batches[this_batch], S, bins=these_bins, bins_find_min=binning_method,
            n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} (pulseHeight cut)', savefig_path=dir_path, fmt='png')    
    ### highlight the sensors with time cut
    plot(df[S], "2D_Tracks", dict_of_batches[this_batch], S, bins=these_bins, mask=time_cuts,
            n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} (w. time cut)', savefig_path=dir_path, fmt='png')
    # delta time vs pulseHeight w/ info
    plot(df[S], "Time_pulseHeight", dict_of_batches[this_batch], S, bins=time_bins,
            n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} ', savefig_path=dir_path, fmt='png')
    ### delta time vs pulseHeight no info
    plot(df[S], "Time_pulseHeight", dict_of_batches[this_batch], S, bins=time_bins, info=False, extra_info=False,
            n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} no info', savefig_path=dir_path, fmt='png') 
    ### delta time vs pulseHeight central area no info
    plot(df[S], "Time_pulseHeight", dict_of_batches[this_batch], S, bins=time_bins, info=False, extra_info=False, mask=central_sensor_area_cuts,
            n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} central area', savefig_path=dir_path, fmt='png')
    ### efficiency projection whole sensor (zooomed)
    plot(df[S], "1D_Efficiency", dict_of_batches[this_batch], S, threshold_charge=threshold_charge, transimpedance=my_transimpedance, geometry_cut='normal', use=use_for_geometry_cut, zoom_to_sensor=True, efficiency_lim=eff_lim,
        bins=these_bins, bins_find_min=binning_method, n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} threshold charge {threshold_charge}fC', savefig_path=dir_path)
    ### with time cut in the center (zoomed)
    plot(df[S], "1D_Efficiency", dict_of_batches[this_batch], S, threshold_charge=threshold_charge, transimpedance=my_transimpedance, geometry_cut='center', use=use_for_geometry_cut, mask=time_cuts, zoom_to_sensor=True, efficiency_lim=eff_lim,
        bins=these_bins, bins_find_min=binning_method, n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} threshold charge {threshold_charge}fC (center and time cut)', savefig_path=dir_path)
    ### 2D efficiency
    plot(df[S], "2D_Efficiency", dict_of_batches[this_batch], S, threshold_charge=threshold_charge, transimpedance=my_transimpedance, geometry_cut='normal', use=use_for_geometry_cut, zoom_to_sensor=True,
        bins=these_bins, bins_find_min=binning_method, n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} thresh charge {threshold_charge}fC', savefig_path=dir_path, fmt='png')
    ### with time cut and zoomed
    plot(df[S], "2D_Efficiency", dict_of_batches[this_batch], S, threshold_charge=threshold_charge, transimpedance=my_transimpedance, geometry_cut='normal', use=use_for_geometry_cut, mask=time_cuts, zoom_to_sensor=True,
        bins=these_bins, bins_find_min=binning_method, n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} thresh charge {threshold_charge}fC (center and time cut)', savefig_path=dir_path, fmt='png')    
    plt.close('all')

if CFD_comparison:
    ### CFD values comparison with normal geo cuts
        CFD_mask = [np.logical_and(charge_cut[dut-1], geo_cuts[dut-1]) if dut in DUTs else None for dut in [1,2,3]]
        for dut in DUTs:
            fig, _ = plot(df[S], 'CFD_comparison', dict_of_batches[this_batch], S, n_DUT=dut, CFD_values=CFD_values, mask=CFD_mask, time_bins=time_bins,
                    savefig=SAVE, savefig_path=dir_path, savefig_details=f" geo cuts")
            plt.close(fig)
    ### CFD values comparison with central area cuts (less statistics)
        CFD_mask = [np.logical_and(charge_cut[dut-1], central_sensor_area_cuts[dut-1]) if dut in DUTs else None for dut in [1,2,3]]
        for dut in DUTs:
            fig, _ = plot(df[S], 'CFD_comparison', dict_of_batches[this_batch], S, n_DUT=dut, CFD_values=CFD_values, mask=CFD_mask,
                    savefig=SAVE, savefig_path=dir_path, savefig_details=f" central area cuts")
            plt.close(fig)
    
stop = time.time()
print(f"TOTAL TIME: {(stop-start)//60:.0f} min and {(stop-start)%60:.2f} sec")

