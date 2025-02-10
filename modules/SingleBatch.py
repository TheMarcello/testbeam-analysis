### can be run individually with:
### python3 -m modules.SingleBatch --batch <_c> 
### optional flags: --SAVE, --show_plots, --fit_charge, --CFD_matrix, --new_results
### warning: --show_plots plots are distorted

import numpy as np # NumPy
import matplotlib.pylab as plt # Matplotlib plots
import mpl_scatter_density     # density scatter plots
import pandas as pd # Pandas

import logging
import argparse     # to get arguments from command line executing file.py
import os # read directories etc.
import sys # to pass the argv to the main() function

import time
# from timeout_decorator import timeout

from importlib import reload # to reload modules

# import LoadBatch
# reload(LoadBatch)
from .LoadBatch import *
from .SensorClasses import Batch, Sensor, Oscilloscope


def analysis_batch(this_batch, batch_object, S, n_DUT=None, CFD_comparison=False, do_plots=True, fit_charge=False, new_results=False, only_center=True, SAVE=True, show_plot=False, time_bins=5000, time_bins_fine=100, results_name="Results_dictionary.pickle", results_path=None, return_results=True, dir_path=None, ROOT_fit_dir=None):
    """
    Performs analysis of one batch: one (or more) duts in a single oscilloscope. Plots a lot of different quantities and performs some fits.
    Also returns a dictionary with all the calculations performed. \n
    Arguments
    ---------
    this_batch:     batch number
    batch_object:   batch object (see class Batch in SensorClasses.py)
    S:              oscilloscope name (either 'S1' or 'S2')
    n_DUT:          number of devices under test (3 for each oscilloscope for May 2023)
    CFD_comparison: 3x3 plot of delta_t distributions with different CFD values for the dut and MCP, to compare them
    do_plots:       boolean option to create the plots (not necessary to obtain the results)
    fit_charge:     boolean option to call charge_fit.C to perform the Langau*Gauss fit of the charge, if False the results are read from files saved previously
    new_results:    boolean option to replace the old results with newly calculated ones
    only_center:    use only the central area (0.5x0.5mm^2) to calculate the time resolution
    SAVE:           boolean option to save the plots
    show_plot:      boolean option to show the plots, could raise warnings for too many figures open (they can be saved if this is false)
    time_bins:      number of bins for the plots of gaussian fit
    time_bins_fine: number of bins for calculating the time resolution (after cuts)
    results_name:   name of the picke file with the previous results
    results_path:   path to the directory containing the pickle file (if existing) with the previous results, to be loaded and updated (if new_results==True)
    return_results: boolean option to return a dictionary of results ### right now this is not useful because the dictionary is created anyways
    dir_path:       path to save all the plots, if not
    ROOT_fit_dir:   path to the folder of the charge_fit.C file (which performs the fit using ROOT)
    Returns
    -------
    results_dictionary:     dictionary of a dictionary of the results, structure:
        {<dut_number>:  {'name':'sensor_name', 
                'board':'board_mounted', 
                'voltage':<voltage_value>,
                'comments':{'Time fit error'},
                'etc':'other_info'}}
    """
    
    ### I want to keep this specific directory for many reasons (fit charge, all plots etc.)
    if results_path is None:    results_path = "/home/marcello/Desktop/Radboud_not_synchro/Master_Thesis/testbeam-analysis/files/"
    if dir_path is None:    dir_path = f'/home/marcello/Desktop/Radboud_not_synchro/Master_Thesis/various plots/all batches/{this_batch}'
    if ROOT_fit_dir is None:    ROOT_fit_dir = f"/home/marcello/Desktop/Radboud_not_synchro/Master_Thesis/testbeam-analysis/ROOT_Langaus_fit/"

    if not os.path.exists(dir_path):
        logging.warning(f"in analysis_batch(), the path: {dir_path} does not exist, creating directory")
        os.mkdir(dir_path)
    start = time.time()


    ### setting all the options for the batch
    binning_method = 'rice'
    threshold_charge = 4 #fC
    eff_lim = (0.4,1)
    # time_bins = 5000 ### maybe 5000 instead of 4000?
    n_bootstrap = False
    my_transimpedance = 4700 #4700 or 10700   ### I PUT THE TRANSIMPEDANCE TO 4700 MANUALLY
    these_bins = bins_dict[this_batch] #bins1    ### custom bins around the sensors
    ### the pulseHeight cut of these batches failed too often
    if this_batch in [502, 505, 601, 602, 603, 604, 605, 901, 902, 1001, 1002]:
        use_for_geometry_cut = 'time'
    else:
        use_for_geometry_cut = 'pulseheight' 
    logging.info(f"in analysis_batch(), analysing Batch: {this_batch}, {S}\n bins for pulseHeight minimum: {binning_method}, bins for time plots: {time_bins}, bins for time resolution: {time_bins_fine} threshold charge: {threshold_charge}fC, bootstrap: {n_bootstrap}")


    ### the table has to be generated (in main()), if they don't exist already
    info_df = pd.read_pickle(os.path.join(dir_path,f'table_data_{this_batch}.pickle'))
    if not n_DUT:   DUTs = get_DUTs_from_dictionary(info_df,S)
    else:           DUTs = n_DUT
    if not DUTs:     ### if there are no DUTs in this oscilloscope
        logging.error("in analysis_batch(), No DUTs selected, no analysis or plot performed")
        return dict()
    
    df = {}  ### having a dictionary in this function is now useless (only one S at a time) but changing everything is not worth the risk
    df[S] = load_batch(this_batch,S)
    ### I can add a check to make sure that batch_object and this_batch are from the same batch
    if batch_object.batch_number != this_batch:
        logging.error(f"In analysis_batch(), batch_number: {batch_object.batch_number} and this_batch: {this_batch} DO NOT MATCH")
    logging.info(f'MCP: {batch_object.S[S].channels["Ch1"].voltage} V, angle: {batch_object.angle}°, temperature: {batch_object.temperature:.2f}°C')
    
### ALL OF THE CUTS
    ### [ ... if dut in DUTs else None for dut in [1,2,3]]  avoids calculating the cuts for the channels with no dut
    geo_cuts = [geometry_mask(df[S], DUT_number=dut, bins=these_bins, bins_find_min=binning_method, use=use_for_geometry_cut)[0] if dut in DUTs else None for dut in [1,2,3]]
    central_sensor_area_cuts = [geometry_mask(df[S], DUT_number=dut, bins=these_bins, bins_find_min=binning_method, only_select='center', use=use_for_geometry_cut)[0] if dut in DUTs else None for dut in [1,2,3]]
    time_cuts = [time_mask(df[S], dut, bins=time_bins, mask=geo_cuts[dut-1], n_bootstrap=False, show_plot=False, savefig=os.path.join(dir_path,f'time_plot_with_geo_cuts_{S}_{this_batch}_DUT{dut}.png'))[0] if dut in DUTs else None for dut in [1,2,3]]
    charge_cuts = [df[S][f'charge_{dut}']/my_transimpedance>threshold_charge if dut in DUTs else None for dut in [1,2,3]]

    ### I still need pulseHeight cut for the charge fit
    mins = [find_min_btw_peaks(df[S][f"pulseHeight_{dut}"], bins=binning_method, show_plot=False) if dut in DUTs else None for dut in [1,2,3]]
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
            # os.chdir(os.path.join(ROOT_fit_dir,'..')) ### I shouldn't change folder
            run_root_string = f'root -b -q "{ROOT_fit_dir}charge_fit.C({this_batch},\\"{S}\\",{dut})"'
            os.system(run_root_string)
            
### ALL OF THE PLOTS
    if do_plots:
        ## show full area
        plot(df[S], "2D_Tracks", batch_object, S, bins=large_bins,
                n_DUT=DUTs, show_plot=show_plot, savefig=SAVE, savefig_details=f' {S} all tracks (no cut)', savefig_path=dir_path, fmt='png')  
        ### highlight the sensors with pulseHeight cut
        plot(df[S], "2D_Sensors", batch_object, S, bins=these_bins, bins_find_min=binning_method,
                n_DUT=DUTs, show_plot=show_plot, savefig=SAVE, savefig_details=f' {S} (pulseHeight cut)', savefig_path=dir_path, fmt='png')    
        ### highlight the sensors with time cut
        plot(df[S], "2D_Tracks", batch_object, S, bins=these_bins, mask=time_cuts,
                n_DUT=DUTs, show_plot=show_plot, savefig=SAVE, savefig_details=f' {S} (w. time cut)', savefig_path=dir_path, fmt='png')
        # delta time vs pulseHeight w/ info
        plot(df[S], "Time_pulseHeight", batch_object, S, bins=time_bins,
                n_DUT=DUTs, show_plot=show_plot, savefig=SAVE, savefig_details=f' {S} ', savefig_path=dir_path, fmt='png')
        ### delta time vs pulseHeight no info
        plot(df[S], "Time_pulseHeight", batch_object, S, bins=time_bins, info=False, extra_info=False,
                n_DUT=DUTs, show_plot=show_plot, savefig=SAVE, savefig_details=f' {S} no info', savefig_path=dir_path, fmt='png') 
        ### delta time vs pulseHeight central area no info
        plot(df[S], "Time_pulseHeight", batch_object, S, bins=time_bins, info=False, extra_info=False, mask=central_sensor_area_cuts,
                n_DUT=DUTs, show_plot=show_plot, savefig=SAVE, savefig_details=f' {S} central area', savefig_path=dir_path, fmt='png')
        ### efficiency projection whole sensor (zooomed)
        plot(df[S], "1D_Efficiency", batch_object, S, threshold_charge=threshold_charge, transimpedance=my_transimpedance, geometry_cut='normal', use=use_for_geometry_cut, zoom_to_sensor=True, efficiency_lim=eff_lim,
            bins=these_bins, bins_find_min=binning_method, n_DUT=DUTs, show_plot=show_plot, savefig=SAVE, savefig_details=f' {S} threshold charge {threshold_charge}fC', savefig_path=dir_path)
        ### with time cut in the center (zoomed)
        plot(df[S], "1D_Efficiency", batch_object, S, threshold_charge=threshold_charge, transimpedance=my_transimpedance, geometry_cut='center', use=use_for_geometry_cut, mask=time_cuts, zoom_to_sensor=True, efficiency_lim=eff_lim,
            bins=these_bins, bins_find_min=binning_method, n_DUT=DUTs, show_plot=show_plot, savefig=SAVE, savefig_details=f' {S} threshold charge {threshold_charge}fC (center and time cut)', savefig_path=dir_path)
        ### 2D efficiency
        plot(df[S], "2D_Efficiency", batch_object, S, threshold_charge=threshold_charge, transimpedance=my_transimpedance, geometry_cut='normal', use=use_for_geometry_cut, zoom_to_sensor=True,
            bins=these_bins, bins_find_min=binning_method, n_DUT=DUTs, show_plot=show_plot, savefig=SAVE, savefig_details=f' {S} thresh charge {threshold_charge}fC', savefig_path=dir_path, fmt='png')
        ### with time cut and zoomed
        plot(df[S], "2D_Efficiency", batch_object, S, threshold_charge=threshold_charge, transimpedance=my_transimpedance, geometry_cut='normal', use=use_for_geometry_cut, mask=time_cuts, zoom_to_sensor=True,
            bins=these_bins, bins_find_min=binning_method, n_DUT=DUTs, show_plot=show_plot, savefig=SAVE, savefig_details=f' {S} thresh charge {threshold_charge}fC (center and time cut)', savefig_path=dir_path, fmt='png')    
        # plt.close('all')

### CFD COMPARISON PLOTS
    if CFD_comparison:
        CFD_values = (20, 50, 70)
    ### CFD values comparison with normal geo cuts
        CFD_mask = [np.logical_and(time_fit_cuts[dut-1], geo_cuts[dut-1]) if dut in DUTs else None for dut in [1,2,3]]
        for dut in DUTs:
            fig, _ = plot(df[S], 'CFD_comparison', batch_object, S, n_DUT=dut, CFD_values=CFD_values, mask=CFD_mask, time_bins=time_bins_fine,
                    savefig=SAVE, savefig_path=dir_path, savefig_details=f" geo cuts",fmt='png')
            plt.close(fig)
    ### CFD values comparison with central area cuts (less statistics)
        CFD_mask = [np.logical_and(time_fit_cuts[dut-1], central_sensor_area_cuts[dut-1]) if dut in DUTs else None for dut in [1,2,3]]
        for dut in DUTs:
            fig, _ = plot(df[S], 'CFD_comparison', batch_object, S, n_DUT=dut, CFD_values=CFD_values, mask=CFD_mask, time_bins=time_bins_fine,
                    savefig=SAVE, savefig_path=dir_path, savefig_details=f" central area cuts",fmt='png')
            plt.close(fig)

    try:
        results_dictionary = read_pickle(os.path.join(results_path, results_name))
    except FileNotFoundError:
        logging.warning(f"The file {results_name} does not exist, creating a new dictionary")
        ### initilizing the dictionary for the results:
        results_dictionary = {dut:{'comments':set()} for dut in DUTs}
    except Exception as e:
        logging.error(f"In analysis_batch(), unknow error: {e}")

    if new_results:
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
                charge_fit_df = pd.read_csv(os.path.join(ROOT_fit_dir,"Charge_fit_results",charge_fit_file), skiprows=1)
                results_dictionary[dut]['charge'] = charge_fit_df["MPV"].iloc[0]
                results_dictionary[dut]['charge_error'] = charge_fit_df["MPV_error"].iloc[0]
                
            except FileNotFoundError:
                logging.error("in analysis_batch(), charge file not found")
                results_dictionary[dut]['charge'], results_dictionary[dut]['charge_error'] = -1, 0
                results_dictionary[dut]['comments'].add("Charge fit file not found")
            except Exception as e:
                logging.errorf(f"in analysis_batch(), raised error {e} when loading charge fit")
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
                ### for all sensors that are irradiated set CFD for the DUT to 70%
                if results_dictionary[dut]['fluence'] != 0:         CFD_DUT = 70
                elif results_dictionary[dut]['fluence'] == 0:       CFD_DUT = 20
                else:        logging.error(f"Value of fluence not recognized:{results_dictionary[dut]['fluence']}, unable to set CFD value")
                ### option to make the geometrical cut be only the central 0.5xo.5mm^2 OR the full surface of the dut
                if only_center: 
                    this_mask = np.logical_and(central_sensor_area_cuts[dut-1], time_fit_cuts[dut-1])
                else:
                    this_mask = np.logical_and(geo_cuts[dut-1],time_fit_cuts[dut-1])

                time_dict = time_mask(df[S], dut, bins=100, n_bootstrap=n_bootstrap, show_plot=show_plot, mask=this_mask, CFD_DUT=CFD_DUT,
                                    title_info=' center cut', savefig=os.path.join(dir_path,f'time_plot_with_center_cuts_{S}_{this_batch}_DUT{dut}.png'))[1]
                time_resolution, time_res_err = error_propagation(time_dict['parameters'][2], time_dict['parameters_errors'][2], MCP_resolution, MCP_error)
            except Exception as e:
                logging.error(f"in analysis_batch(), Time fit error: {e}")
                results_dictionary[dut]['comments'].add(f"Time fit error ({e})")
                time_resolution, time_res_err = 0, 0
            
            results_dictionary[dut]['time_resolution'], results_dictionary[dut]['time_res_err'] = time_resolution, time_res_err
            logging.info(f"in analysis_batch(), Time resolution: {time_resolution:.2f}ps +/- {time_res_err:.2f}ps, (MCP: {MCP_resolution}ps)")
            ### efficiency only calculated with center and time cuts
            results_dictionary[dut]['efficiency'],_ = efficiency_error(df[S][f"charge_{dut}"].loc[np.logical_and(central_sensor_area_cuts[dut-1],time_cuts[dut-1])], threshold=threshold_charge)
    else:
        logging.info("In analysis_batch(), loading existing data and NOT calculating new results")

    
    with open(os.path.join(results_path, results_name), 'wb') as f:
        pickle.dump(results_dictionary, f)
    logging.info(f"in analysis_batch")
    stop = time.time()

    print(f"TOTAL TIME: {(stop-start)//60:.0f} min and {(stop-start)%60:.2f} sec")

    
    if return_results:
        return results_dictionary



def main(argv):
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - \t %(message)s')

        ### use the argparse package to parse command line arguments
    parser = argparse.ArgumentParser(description='Plot a single batch, given batch number and other arguments')
    parser.add_argument('--batch', type=int, help='Batch number')
    parser.add_argument('--do_plots', action='store_true', help='if False, no plots are produced')
    parser.add_argument('--fit_charge', action='store_true', help='if charge should be fitted (with ROOT)')
    parser.add_argument('--CFD_matrix', action='store_true', help='option for CFD comparison plots')
    parser.add_argument('--new_results', action='store_true', help='choice to recalculate the data or just read from existing files')
    parser.add_argument('--SAVE', action='store_true', help='SAVE option to save plots (or not)')
    parser.add_argument('--show_plots', action='store_true', help='option to show plots (or not)')
    # parser.add_argument('-d', default=False, help='Output directory')
    args = parser.parse_args(argv[1:])

    this_batch = args.batch
    do_plots = args.do_plots
    fit_charge = args.fit_charge
    CFD_comparison = args.CFD_matrix
    new_results = args.new_results
    SAVE = args.SAVE
    show_plot = args.show_plots
    logging.info(f"in main() of Single_batch.py, parsed arguments:  batch: {this_batch}, SAVE: {SAVE}, do_plots: {do_plots}, show_plot: {show_plot}, fit_charge: {fit_charge}, CFD_comp: {CFD_comparison}, new_results: {new_results}")

    if this_batch is None:
        logging.error(f"in main() of Singe_batch, no batch number has been passed as argument, run e.g. 'python3 Single_batch.py --batch 301 --SAVE' ")
        return 1
    ### Load the dictionary of sensor names and runs
    dict_of_batches = read_pickle("./files/dict_of_batches.pickle")
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
        DUTs = get_DUTs_from_dictionary(info_df,S)
        if not DUTs:
            continue
        analysis_batch(this_batch, dict_of_batches[this_batch], S, n_DUT=DUTs, do_plots=do_plots, show_plot=show_plot, SAVE=SAVE, CFD_comparison=CFD_comparison, fit_charge=fit_charge, new_results=new_results, return_results=False)

### I need a main() function so I can import it as a module and call the analysis_batch()
if __name__ == "__main__":
    main(sys.argv)