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


def analysis_batch(this_batch, batch_object, S, n_DUT=None, do_plots=True, SAVE=True, CFD_comparison=False, fit_charge=False, return_results=True):
    
    ### I want to keep this specific directory for many reasons (fit charge, all plots etc.)
    dir_path = f'/home/marcello/Desktop/Radboud_not_synchro/Master_Thesis/various plots/all batches/{this_batch}'
    ROOT_fit_dir = f"/home/marcello/Desktop/Radboud_not_synchro/Master_Thesis/testbeam-analysis/ROOT_Langaus_fit/Charge_fit_results"
    if not os.path.exists(dir_path):
        logging.warning(f"in analysis_batch(), the path: {dir_path} does not exist, creating directory")
        os.mkdir(dir_path)
    start = time.time()

    ### show all information about the batch
    binning_method = 'rice'
    threshold_charge = 4 #fC
    eff_lim = (0.4,1)
    time_bins = 4000
    n_bootstrap = False
    my_transimpedance = 4700 #4700 or 10700   ### I PUT THE TRANSIMPEDANCE TO 4700 MANUALLY
    these_bins = bins_dict[this_batch] #bins1    ### custom bins around the sensors

    if this_batch in [502, 601, 602, 603, 604, 605, 901, 902, 1001, 1002]:
        use_for_geometry_cut = 'time'
    else:
        use_for_geometry_cut = 'pulseheight' 
    ### I could print this info into the logs
    logging.info(f"in analysis_batch(), analysing Batch: {this_batch}, {S}")

    info_df = pd.read_pickle(os.path.join(dir_path,f'table_data_{this_batch}.pickle'))
    if not n_DUT:
        DUTs = get_DUTs_from_dictionary(info_df,S)
    else:
        DUTs = n_DUT
    if not DUTs:     ### if there are no DUTs in this oscilloscope
        logging.error("in analysis_batch(), No DUTs selected, no analysis or plot performed")
        return dict()
    
    df = {}  ### having a dictionary in this function is now useless (only one S at a time) but changing everything is not worth the risk
    df[S] = load_batch(this_batch,S)
    logging.info(f'MCP: {batch_object.S[S].channels["Ch1"].voltage} V, angle: {batch_object.angle}°, temperature: {batch_object.temperature:.2f}°C')
    
    ### initilizing the dictionary for the results:
    results_dictionary = {dut:{'comments':set()} for dut in DUTs}
### ALL OF THE CUTS
    ### [ ... if dut in DUTs else None for dut in [1,2,3]], it avoids calculating the cuts for the channels with no dut
    geo_cuts = [geometry_mask(df[S], DUT_number=dut, bins=these_bins, bins_find_min='rice', use=use_for_geometry_cut)[0] if dut in DUTs else None for dut in [1,2,3]]
    central_sensor_area_cuts = [geometry_mask(df[S], DUT_number=dut, bins=these_bins, bins_find_min='rice', only_select='center', use=use_for_geometry_cut)[0] if dut in DUTs else None for dut in [1,2,3]]
    time_cuts = [time_mask(df[S], dut, bins=time_bins, mask=geo_cuts[dut-1], n_bootstrap=False, plot=False, savefig=os.path.join(dir_path,f'time_plot_with_geo_cuts_{S}_{this_batch}_DUT{dut}.png'))[0] if dut in DUTs else None for dut in [1,2,3]]
    charge_cuts = [df[S][f'charge_{dut}']/my_transimpedance>threshold_charge if dut in DUTs else None for dut in [1,2,3]]

    ### I still need pulseHeight cut for the charge fit
    mins = [find_min_btw_peaks(df[S][f"pulseHeight_{dut}"], bins='rice', plot=False) if dut in DUTs else None for dut in [1,2,3]]
    ### also check that the pulseHeight is > pedestal + 3*noise
    pulse_noise_cuts = [df[S][f'pulseHeight_{dut}']>(df[S][f"pedestal_{dut}"]+3*df[S][f"noise_{dut}"]) if dut in DUTs else None for dut in [1,2,3]]
    pulse_cuts = [np.logical_and(pulse_noise_cuts[dut-1], df[S][f'pulseHeight_{dut}']>mins[dut-1]) if dut in DUTs else None for dut in [1,2,3]]
    for dut in (1,2,3):
        if (pulse_cuts[dut-1] is not None) and (np.all(pulse_cuts[dut-1]==False)):
            pulse_cuts[dut-1] = pd.Series(True, index=df[S].index)
        if use_for_geometry_cut == "time":   ### also NOT APPLY a pulseHeight cut if I choose time for geometry cut
            pulse_cuts[dut-1] = pd.Series(True, index=df[S].index)

    ### charge distribution with cuts saved into a file and the fitted with ROOT Landau*Gauss convolution
    all_cuts = [np.logical_and(np.logical_and(np.logical_and(geo_cuts[dut-1], time_cuts[dut-1]), pulse_cuts[dut-1]), pulse_noise_cuts[dut-1])
                if dut in DUTs else None for dut in [1,2,3]]
    
    ### I also do the fit between 3 sigmas because the tails make it worse (aka also time_cut)
    time_fit_cuts = [np.logical_and(np.logical_and(pulse_noise_cuts[dut-1], charge_cuts[dut-1]), time_cuts[dut-1])
                        if dut in DUTs else None for dut in [1,2,3]]

### FIT OF THE CHARGE 
    if fit_charge:
        for dut in DUTs:
            np.savetxt(os.path.join(dir_path, f"charge_data_all_cuts_{this_batch}_{S}_{dut}.csv"),
                        df[S][f'charge_{dut}'].loc[all_cuts[dut-1]]/my_transimpedance, delimiter=',')
            os.chdir(ROOT_fit_dir)
            run_root_string = f'root -b -q "charge_fit.C({this_batch},\\"{S}\\",{dut})"'
            os.system(run_root_string)
    # else:
    #     pass
            
### ALL OF THE PLOTS
    ### time resolution fit
    for dut in DUTs:
        time_mask(df[S], dut, bins=100, n_bootstrap=n_bootstrap, mask=np.logical_and(geo_cuts[dut-1], time_fit_cuts[dut-1]), plot=do_plots,
                                    savefig=os.path.join(dir_path,f'time_plot_with_geo_cuts_{S}_{this_batch}_DUT{dut}.png'))[1]['parameters']
        
    [time_mask(df[S], dut, bins=100, n_bootstrap=n_bootstrap, mask=np.logical_and(geo_cuts[dut-1], time_fit_cuts[dut-1]), plot=do_plots,
                                    savefig=os.path.join(dir_path,f'time_plot_with_geo_cuts_{S}_{this_batch}_DUT{dut}.png'))[1]['parameters'] if dut in DUTs else None for dut in [1,2,3]]
    ### time resolution fit with central area cuts
    [time_mask(df[S], dut, bins=100, n_bootstrap=n_bootstrap, mask=np.logical_and(central_sensor_area_cuts[dut-1], time_fit_cuts[dut-1]), plot=do_plots, title_info=' center cut',
                                    savefig=os.path.join(dir_path,f'time_plot_with_center_cuts_{S}_{this_batch}_DUT{dut}.png'))[1]['parameters'] if dut in DUTs else None for dut in [1,2,3]]
    if do_plots:
        ## show full area
        plot(df[S], "2D_Tracks", batch_object, S, bins=large_bins,
                n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} all tracks (no cut)', savefig_path=dir_path, fmt='png', show_plot=True)  
        ### highlight the sensors with pulseHeight cut
        plot(df[S], "2D_Sensors", batch_object, S, bins=these_bins, bins_find_min=binning_method,
                n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} (pulseHeight cut)', savefig_path=dir_path, fmt='png')    
        ### highlight the sensors with time cut
        plot(df[S], "2D_Tracks", batch_object, S, bins=these_bins, mask=time_cuts,
                n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} (w. time cut)', savefig_path=dir_path, fmt='png')
        # delta time vs pulseHeight w/ info
        plot(df[S], "Time_pulseHeight", batch_object, S, bins=time_bins,
                n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} ', savefig_path=dir_path, fmt='png')
        ### delta time vs pulseHeight no info
        plot(df[S], "Time_pulseHeight", batch_object, S, bins=time_bins, info=False, extra_info=False,
                n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} no info', savefig_path=dir_path, fmt='png') 
        ### delta time vs pulseHeight central area no info
        plot(df[S], "Time_pulseHeight", batch_object, S, bins=time_bins, info=False, extra_info=False, mask=central_sensor_area_cuts,
                n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} central area', savefig_path=dir_path, fmt='png')
        ### efficiency projection whole sensor (zooomed)
        plot(df[S], "1D_Efficiency", batch_object, S, threshold_charge=threshold_charge, transimpedance=my_transimpedance, geometry_cut='normal', use=use_for_geometry_cut, zoom_to_sensor=True, efficiency_lim=eff_lim,
            bins=these_bins, bins_find_min=binning_method, n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} threshold charge {threshold_charge}fC', savefig_path=dir_path)
        ### with time cut in the center (zoomed)
        plot(df[S], "1D_Efficiency", batch_object, S, threshold_charge=threshold_charge, transimpedance=my_transimpedance, geometry_cut='center', use=use_for_geometry_cut, mask=time_cuts, zoom_to_sensor=True, efficiency_lim=eff_lim,
            bins=these_bins, bins_find_min=binning_method, n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} threshold charge {threshold_charge}fC (center and time cut)', savefig_path=dir_path)
        ### 2D efficiency
        plot(df[S], "2D_Efficiency", batch_object, S, threshold_charge=threshold_charge, transimpedance=my_transimpedance, geometry_cut='normal', use=use_for_geometry_cut, zoom_to_sensor=True,
            bins=these_bins, bins_find_min=binning_method, n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} thresh charge {threshold_charge}fC', savefig_path=dir_path, fmt='png')
        ### with time cut and zoomed
        plot(df[S], "2D_Efficiency", batch_object, S, threshold_charge=threshold_charge, transimpedance=my_transimpedance, geometry_cut='normal', use=use_for_geometry_cut, mask=time_cuts, zoom_to_sensor=True,
            bins=these_bins, bins_find_min=binning_method, n_DUT=DUTs, savefig=SAVE, savefig_details=f' {S} thresh charge {threshold_charge}fC (center and time cut)', savefig_path=dir_path, fmt='png')    
        plt.close('all')

### CFD COMPARISON PLOTS
    if CFD_comparison:
        CFD_values = (20, 50, 70)
    ### CFD values comparison with normal geo cuts
        CFD_mask = [np.logical_and(time_fit_cuts[dut-1], geo_cuts[dut-1]) if dut in DUTs else None for dut in [1,2,3]]
        for dut in DUTs:
            fig, _ = plot(df[S], 'CFD_comparison', batch_object, S, n_DUT=dut, CFD_values=CFD_values, mask=CFD_mask, time_bins=100,
                    savefig=SAVE, savefig_path=dir_path, savefig_details=f" geo cuts",fmt='png')
            plt.close(fig)
    ### CFD values comparison with central area cuts (less statistics)
        CFD_mask = [np.logical_and(time_fit_cuts[dut-1], central_sensor_area_cuts[dut-1]) if dut in DUTs else None for dut in [1,2,3]]
        for dut in DUTs:
            fig, _ = plot(df[S], 'CFD_comparison', batch_object, S, n_DUT=dut, CFD_values=CFD_values, mask=CFD_mask, time_bins=100,
                    savefig=SAVE, savefig_path=dir_path, savefig_details=f" central area cuts",fmt='png')
            plt.close(fig)

    for dut in DUTs:
        ch = f'Ch{dut+1}'
        results_dictionary[dut]['name'] = batch_object.S[S].get_sensor(ch).name
        results_dictionary[dut]['board'] = batch_object.S[S].get_sensor(ch).board
        results_dictionary[dut]['voltage'] = batch_object.S[S].get_sensor(ch).voltage
        results_dictionary[dut]['current'] = batch_object.S[S].get_sensor(ch).current
        results_dictionary[dut]['fluence'] = batch_object.S[S].get_sensor(ch).fluence
        results_dictionary[dut]['MCP_voltage'] = batch_object.S[S].get_sensor('Ch1').voltage
        MCP_voltage = batch_object.S[S].get_sensor('Ch1').voltage
        results_dictionary[dut]['temp_A'] = batch_object.S[S].tempA
        results_dictionary[dut]['temp_B'] = batch_object.S[S].tempB
        results_dictionary[dut]['angle'] = batch_object.angle
        results_dictionary[dut]['humidity'] = batch_object.humidity
        results_dictionary[dut]['temperature'] = batch_object.temperature
        try:
            charge_fit_file = f"charge_fit_results_{this_batch}_{S}_{dut}.csv"
            charge_fit_df = pd.read_csv(os.path.join(ROOT_fit_dir,charge_fit_file), skiprows=1)
            results_dictionary[dut]['charge'] = charge_fit_df["MPV"].iloc[0]
            results_dictionary[dut]['charge_error'] = charge_fit_df["MPV_error"].iloc[0]
        except FileNotFoundError:
            logging.error("in analysis_batch(), charge file not found")
            results_dictionary[dut]['charge'], results_dictionary[dut]['charge_error'] = -1, 0
            results_dictionary[dut]['comments'].add("Charge fit file not found")
        except:
            logging.error("in analysis_batch(), unknown error when loading charge fit")
            results_dictionary[dut]['charge'], results_dictionary[dut]['charge_error'] = -1, 0
            results_dictionary[dut]['comments'].add("Charge fit error")
            ### read the results from the file
        match MCP_voltage:  ### matche the MCP voltage to its time resolution
            case 2500: 
                MCP_resolution, MCP_error = 36.52, 0.81  # 36.52 +/- 0.81 ps
            case 2600: 
                MCP_resolution, MCP_error = 16.48, 0.57  # 16.48 +/- 0.57 ps
            case 2800: 
                MCP_resolution, MCP_error = 3.73, 1.33   # 3.73 +/- 1.33 ps
            case other: logging.error(f"in analysis_batch(), incorrect MCP voltage: {other}")    
        try:
            time_dict = time_mask(df[S], dut, bins=100, n_bootstrap=False, plot=do_plots, mask=np.logical_and(np.logical_and(central_sensor_area_cuts[dut-1],charge_cuts[dut-1]),time_cuts[dut-1]))[1]
            time_resolution, time_res_err = error_propagation(time_dict['parameters'][2], time_dict['parameters_errors'][2], MCP_resolution, MCP_error)
        except:
            logging.error(f"in analysis_batch(), Time fit error")
            results_dictionary[dut]['comments'].add("Time fit error")
            time_resolution, time_res_err = 0, 0
        logging.info(f"in analysis_batch(), Time resolution: {time_resolution:.2f}ps +/- {time_res_err:.2f}ps, (MCP: {MCP_resolution}ps)")
        ### efficiency only calculated with center and time cuts
        results_dictionary[dut]['efficiency'] = efficiency(df[S][f"charge_{dut}"].loc[np.logical_and(central_sensor_area_cuts[dut-1],time_cuts[dut-1])], threshold=threshold_charge, percentage=False)

    stop = time.time()

    print(f"TOTAL TIME: {(stop-start)//60:.0f} min and {(stop-start)%60:.2f} sec")

    
    if return_results:
        ### I need to update all the code in Analysis_and_comparison_UTSC for the fact that I have 3 (or different number) of DUTS
        return results_dictionary



def main():
        ### use the argparse package to parse command line arguments
    parser = argparse.ArgumentParser(description='Plot a single batch, given batch number and other arguments')
    parser.add_argument('-batch', type=int, help='Batch number')
    parser.add_argument('-S', type=bool, default=True, help='SAVE option to save plots (or not)')
    parser.add_argument('-fit_charge', default=False, help='if charge should be fitted (with ROOT)')
    parser.add_argument('-CFD', default=False, help='option for CFD comparison plots')
    # parser.add_argument('-d', default=False, help='Output directory')
    args = parser.parse_args()

    this_batch = args.batch
    fit_charge = args.fit_charge
    SAVE = args.S
    CFD_comparison = args.CFD

    pd.set_option('display.max_columns', None)
    # logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] - %(message)s')
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - \t %(message)s')

    ### Load the dictionary of sensor names and runs
    dict_of_batches = read_pickle("/home/marcello/Desktop/Radboud_not_synchro/Master_Thesis/testbeam-analysis/dict_of_batches.pickle")
    ### I want to keep this specific directory for many reasons (fit charge, all plots etc.)
    dir_path = f'/home/marcello/Desktop/Radboud_not_synchro/Master_Thesis/various plots/all batches/{this_batch}'

    info_dict = {}    ### show all informations about each sensor
    for S in ['S1','S2']:
        if dict_of_batches[this_batch].S[S] is None:
            logging.warning(f"No oscilloscope {S} in batch {this_batch}")
            info_dict[(S,ch)] = None
        else:
            for ch, sensor in dict_of_batches[this_batch].S[S].channels.items():
                info_dict[(S,ch)] = sensor.__dict__
    info_df = pd.DataFrame(info_dict)
    # Export DataFrame to a CSV file
    info_df.to_pickle(os.path.join(dir_path,f'table_data_{this_batch}.pickle'))

    for S in ('S1','S2'):
        ### in main() I only want to do the plots
        DUTs = get_DUTs_from_dictionary(info_df)
        if not DUTs:
            continue

        analysis_batch(this_batch, dict_of_batches[this_batch], S, n_DUT=DUTs, do_plots=True, SAVE=SAVE, CFD_comparison=CFD_comparison, fit_charge=fit_charge, return_results=False)

### I need a main() function so I can import it as a module and call the analysis_batch()
if __name__ == "__main__":
    main()