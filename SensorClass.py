import numpy as np # NumPy
import matplotlib.pylab as plt # Matplotlib plots
import pandas as pd # Pandas
# import uproot
# import pickle

# import os # read directories etc.
from scipy.signal import find_peaks, gaussian
from scipy.stats import gaussian_kde




class Batch:
    """
    Batch class, includes all relevant information of the batch, such as
    ------------- 
    batch_number:   batch number
    angle:          angle to the beam   [°degrees]
    runs:           list of run numbers belonging to the same batch
    tempA:          temperature of thermometer A [°C]
    tempB:          temperature of thermometer B [°C]
    S1, S2:         Oscilloscope objects 1 and 2
    """
    def __init__(self, batch_number, angle, runs, temperatureA, temperatrureB, S1, S2):
        self.batch_number = batch_number
        self.angle = angle
        self.runs = runs
        self.tempA = temperatureA
        self.tempB = temperatrureB
        self.S1 = S1
        self.S2 = S2
        
        
class Oscilloscope:
    """
    The single oscilloscope containing the four channels with the four sensors
    ------------
    name:           name of the oscilloscope ('S1' or 'S2' usually)
    channels:       dictionary of the 4 channels: {'Ch1':sensor1, etc.}
    """
    def __init__(self, name, sensor1, sensor2, sensor3, sensor4): ### or **sensors ??? = {'Ch1':sensor1, 'Ch2':sensor2 etc.}
        self.name = name
        self.channels = {'Ch1':sensor1, 'Ch2':sensor2, 'Ch3':sensor3, 'Ch4':sensor4}

        
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
    def __init__(self, name, board, dut_position, fluence, transimpedance, voltage):
        self.name = name
        self.board = board
        self.dut_position = dut_position
        self.fluence = fluence
        self.transimpedance = transimpedance
        self.voltage = voltage


# sensor1 = Sensor('USTC','CERN1', 0, 1, 4700, -80)
# sensor1.name

# my_scope = Oscilloscope('S1', sensor1, sensor1, sensor1, sensor1)
# my_scope.channels['Ch1'].name

# my_batch = Batch(999, 0, [9998,9999], -30, my_scope, my_scope)
# # my_batch.angle
# my_batch.S1.channels['Ch1'].transimpedance