import numpy as np # NumPy
import matplotlib.pylab as plt # Matplotlib plots
import pandas as pd # Pandas
import uproot
import pickle

import os # read directories etc.
from scipy.signal import find_peaks, gaussian
from scipy.stats import gaussian_kde




class Batch:
    """
    Batch class, includes all relevant information of the batch, such as
    batch_number: 
    angle:   
    runs
    temperature
    S1, S2:       
    """
    def __init__(self, batch_number, angle, runs, temperature, S1, S2):
        self.batch_number = batch_number
        self.angle = angle
        self.runs = runs
        self.temperature = temperature
        self.S1 = S1
        self.S2 = S2
        
        
class Oscilloscope:
    """
    The single oscilloscope containing the four channels with the four sensors
    """
    def __init__(self, name, sensor1, sensor2, sensor3, sensor4): ### or **sensors ??? = {'Ch1':sensor1, 'Ch2':sensor2 etc.}
        self.name = name
        self.channels = {'Ch1':sensor1, 'Ch2':sensor2, 'Ch3':sensor3, 'Ch4':sensor4}#, etc.}# dict of channels

        
class Sensor:
    """
    Class to describe the DUT that is being studied in a single batch
    """
    def __init__(self, name, board, dut_position, fluence, transimpedance, voltage):
        self.name = name
        self.board = board
        self.dut_position = dut_position
        self.transimpedance = transimpedance
        self.fluence = fluence
        self.voltage = voltage


# sensor1 = Sensor('USTC','CERN1', 0, 1, 4700, -80)
# sensor1.name

# my_scope = Oscilloscope('S1', sensor1, sensor1, sensor1, sensor1)
# my_scope.channels['Ch1'].name

# my_batch = Batch(999, 0, [9998,9999], -30, my_scope, my_scope)
# # my_batch.angle
# my_batch.S1.channels['Ch1'].transimpedance