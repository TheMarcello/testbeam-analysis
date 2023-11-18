import numpy as np # NumPy
import matplotlib.pylab as plt # Matplotlib plots
import pandas as pd # Pandas
# import uproot
# import pickle

# import os # read directories etc.
from scipy.signal import find_peaks, gaussian
from scipy.stats import gaussian_kde


class Sensor:
    """
    Class to describe the DUT that is being studied in a single batch
    ------------
    name:           name of the sensor
    board:          name of the board on which the sensor is mounted
    dut_position:   position of the sensor (1-5)
    fluence:        radiation given to the sensor [units?]
    transimpedance: transimpedance, depends on which board (to calculate charge) [units?]
    voltage:        voltage of the sensor [V]
    """
    def __init__(self, name, dut_position, transimpedance, voltage, angle=0, board='no board', fluence=-1):
        self.name = name
        self.angle = angle
        self.board = board
        self.dut_position = dut_position
        self.fluence = fluence
        self.transimpedance = transimpedance
        self.voltage = voltage


class Oscilloscope:
    """
    The single oscilloscope containing the four channels with the four sensors
    ------------
    name:           name of the oscilloscope ('S1' or 'S2' usually)
    channels:       dictionary of the 4 channels: {'Ch1':sensor1, etc.}
    ------------
    add_sensor:     add a Sensor to the specified channel
    """
    def __init__(self, name, sensor1=None, sensor2=None, sensor3=None, sensor4=None):
        self.name = name
        self.channels = {'Ch1':sensor1, 'Ch2':sensor2, 'Ch3':sensor3, 'Ch4':sensor4}
    
    def add_sensor(self, channel, sensor):
        self.channels[channel] = sensor


class Batch:
    """
    Batch class, includes all relevant information of the batch, such as
    ------------- 
    batch_number:   batch number
    angle:          angle to the beam   [°degrees] ### I should put it into the sensor
    runs:           list of run numbers belonging to the same batch
    tempA:          temperature of thermometer A [°C]
    tempB:          temperature of thermometer B [°C]
    S1, S2:         Oscilloscope objects 1 and 2

    get_fluence_boards():
    """
    def __init__(self, batch_number, angle, runs, temperatureA, temperatureB, S1, S2):#):
        self.batch_number = batch_number
        self.angle = angle
        self.runs = runs
        self.tempA = temperatureA
        self.tempB = temperatureB
        self.S = {'S1':S1, 'S2':S2}
        self.get_fluence_boards()

    def get_fluence_boards(self):
        """
        Map of each board names for each batch
        same thing for the fluence
        """
    #     four_ch = 10700
    #     single_ch = 4700
#         if S=="S1": scope.channels['Ch1'].fluence, scope.channels['Ch2'].fluence, \
#                     scope.channels['Ch3'].fluence, scope.channels['Ch4'].fluence = (0, 0, 0, 0)
#         elif S=="S2":   scope.channels['Ch1'].fluence, scope.channels['Ch2'].fluence, \
#                         scope.channels['Ch3'].fluence, scope.channels['Ch4'].fluence = (0, 0, 0, 0)
        none = ' '
        batch = self.batch_number        
        for S,scope in self.S.items():
               ### default to zero because most of them are unirradiated and not angled
            fluences = (0, 0, 0, 0)     
            # angles = (0, 0, 0, 0)     ### I am not keeping this because it's inconsistent, I keep the RUNLOG angles
            ### maybe I can avoid some repetition
            if batch>=100 and batch<200:     ### Ch2      Ch3       Ch4
                if S=="S1":     boards = (none, 'CERN-1','CERN-1','CERN-1')
                elif S=="S2":   boards = (none, 'JSI-B14', 'JSI-B12', 'CERN-1')
            elif batch>=200 and batch<300:
                if S=="S1":     boards = (none, 'CERN-1', 'CERN-1', none)
                elif S=="S2":   boards = (none, none, none, none)
            elif batch>=300 and batch<400:
                if S=="S1":     boards = (none, 'CERN-3', 'CERN-3', none)
                elif S=="S2":   boards =  (none, 'JSI-B14', 'JSI-B12', 'CERN-1')
            elif batch>=400 and batch<500:
                if S=="S1":     boards = (none, 'CERN-3', 'CERN-3', 'CERN-1')
                elif S=="S2":   boards= (none, 'JSI-B14', 'JSI-B12', 'CERN-1')
            elif batch>=500 and batch<600:
                if S=="S1":
                    boards = (none, 'JSI PP4', 'JSI B7', 'JSI B13')
                    fluences = (0, '6.5E14 p', '8.00E+14', '2.50E+15')
                elif S=="S2":   
                    boards = (none, 'JSI B6', 'JSI PP1', none)
                    fluences = (0, '1E14 P', '1.50E+15', 0)
            elif batch>=600 and batch<700:
                if S=="S1": 
                    boards = (none, 'CERN2', 'CERN2', 'CERN2')
                    fluences = (0, '1.50E+15', '1.50E+15', '1.50E+15')
                elif S=="S2":   
                    boards = (none, 'JSI B7', 'JSI B5', 'CERN2')
                    fluences = (0, '8.00E+14', '2.50E+15', '1.50E+15')
            elif batch>=700 and batch<800:
                if S=="S1":
                    # angles = (0, 14, 14, 14)
                    boards = (none, 'CERN2', 'CERN2', 'CERN2')
                    fluences = (0, '1.50E+15', '1.50E+15', '1.50E+15')
                elif S=="S2":  
                    # angles = (0, 0, 0, 14)
                    boards = (none, none, none, 'CERN2')
                    fluences = (0, 0, 0, '1.50E+15')
            elif batch>=800 and batch<900:
                if S=="S1": boards = (none, none, none, none)
                elif S=="S2":  
                    # angles = (0, 12.5, 14, 0)
                    boards = (none, 'JSI B7', 'JSI B6', none)
                    fluences = (0, '8.00E+14', '1E14 P', 0)
            elif batch>=900 and batch<1000:
                if S=="S1": 
                    # angles = (0, 12.5, 14, 0)
                    boards = (none, 'JSI PP4', 'JSI B13', none)
                    fluences = (0, '6.5E14 p', '2.50E+15', 0)
                elif S=="S2":   boards= (none, none, none, none)
            elif batch>=1000 and batch<1100:
                if S=="S1":
                    # angles = (0, 14, 14, 14)
                    boards = (none, 'CERN2', 'CERN2', 'CERN2')
                    fluences = (0, '1.50E+15', '1.50E+15', '1.50E+15')
                elif S=="S2":
                    # angles = (0, 12.5, 14, 14)
                    boards = (none, 'JSI B7', 'JSI B6', 'CERN2')
                    fluences = (0, '8.00E+14', '1E14 P', '1.50E+15')
            elif batch>=1100 and batch<1200:
                if S=="S1":
                    # angles = (0, 12.5, 14, 0)
                    boards = (none, 'JSI PP4', 'JSI B13', none)
                    fluences = (0, '6.5E14 p', '2.50E+15', 0)
                elif S=="S2":
                    # angles = (0, 12.5, 14, 0)
                    boards = (none, 'JSI B7', 'JSI B6', none)
                    fluences = (0, '8.00E+14', '1E14 P', 0)
            elif batch>=1200 and batch<1300:
                if S=="S1": 
                    boards = (none, 'JSI B5', 'JSI PP1', none)
                    fluences = (0, '2.50E+15', '1.50E+15', 0)
                elif S=="S2":   boards = (none, none, none, none)
            elif batch>=1300 and batch<1400:
                if S=="S1": boards = (none, none, none, none)
                elif S=="S2":   
                    boards = (none, none, 'JSI B2', none)
                    fluences = (0, 0, '2.00E+14', 0)
            else:     ### last case, return all none
                if S=="S1": boards = (none, none, none, none)
                elif S=="S2":   boards =(none, none, none, none)
                
            scope.channels['Ch1'].board, scope.channels['Ch2'].board, \
                            scope.channels['Ch3'].board, scope.channels['Ch4'].board = boards 
            scope.channels['Ch1'].fluence, scope.channels['Ch2'].fluence, \
                            scope.channels['Ch3'].fluence, scope.channels['Ch4'].fluence = fluences 
