# import numpy as np # NumPy
# import matplotlib.pylab as plt # Matplotlib plots
# import pandas as pd # Pandas
import logging


NO_BOARD = 'no_board' ### default value for when the board info is missing (might change later)

class Sensor:
    """
    Class to describe the DUT that is being studied in a single batch
    ------------
    name:           name of the sensor
    dut_position:   position of the sensor (1-5)
    voltage:        voltage of the sensor [V]
    current:        *measured* current [A]
    board:          name of the board on which the sensor is mounted
    fluence:        radiation given to the sensor [units?]
    transimpedance: transimpedance, it depends on the board (used to calculate charge) [units?]
    """
    def __init__(self, name=NO_BOARD, dut_position=0, voltage=0, current=0, board=NO_BOARD, fluence=-1, transimpedance=-1):
        self.name = name
        self.board = board
        self.dut_position = dut_position
        self.fluence = fluence
        self.transimpedance = transimpedance
        self.voltage = voltage
        self.current = current

class Oscilloscope:
    """
    The single oscilloscope containing the four channels with the four sensors
    ------------
    name:           name of the oscilloscope ('S1' or 'S2' usually)
    channels:       dictionary of the 4 channels: {'Ch1':sensor1, etc.}
    runs:           list of runs (some runs where excluded so they could be different btw oscilloscopes)
    tempA:          list of temperatures, thermometer A ( " " )
    tempB:          list of temperatures, thermometer B ( " " )
    ------------
    add_sensor:     add a Sensor to the specified channel
    get_sensor:     get a Sensor from the specified channel
    """
    def __init__(self, name, runs, temperatureA=None, temperatureB=None, sensor1=None, sensor2=None, sensor3=None, sensor4=None):
        self.name = name
        self.channels = {'Ch1':sensor1 if sensor1 else Sensor(),
                         'Ch2':sensor2 if sensor2 else Sensor(),
                         'Ch3':sensor3 if sensor3 else Sensor(),
                         'Ch4':sensor4 if sensor4 else Sensor()}
        self.runs = runs
        self.tempA = temperatureA if temperatureA is not None else []
        self.tempB = temperatureB if temperatureB is not None else []

    def add_sensor(self, channel, sensor):
        self.channels[channel] = sensor
        ### should I add the "set_fluence()" things here??
        ### so that it only gets initialised when I add the sensor?
        ### yes, I think that's better but I don't know how easy it is to change
            
    def get_sensor(self, ch):
        match ch:
            case 'Ch1' | 'ch1' |'Ch_1' | 'ch_1' : return self.channels['Ch1']
            case 'Ch2' | 'ch2' |'Ch_2' | 'ch_2' : return self.channels['Ch2']
            case 'Ch3' | 'ch3' |'Ch_3' | 'ch_3' : return self.channels['Ch3']
            case 'Ch4' | 'ch4' |'Ch_4' | 'ch_4' : return self.channels['Ch4']
            case other:
                logging.error(f"Wrong argument in get_sensor(): {other}")


class Batch:
    """
    Batch class, includes all relevant information of the batch, such as
    ------------- 
    batch_number:   batch number
    angle:          angle to the beam   [°degrees] ### I should put it into the sensor
    humidity:       average humidity [%] inside the cooling box
    temperature:    average temperature [°C]
    S:              dictionary of the two oscilloscopes
        {'S1': Oscilloscope1, 'S2': Oscilloscope2}

    set_fluence_boards():   sets board names and fluences (only for __init__)
    """
    def __init__(self, batch_number, angle=0, humidity=0, temperature_avg=0, S1=None, S2=None):#):
        self.batch_number = batch_number
        self.angle = angle
        self.humidity = humidity
        self.temperature = temperature_avg
        self.S = {'S1':S1 if S1 else Oscilloscope('S1', []),    ### this is a bit overly nested but it's useful for a loop like: "S in ['S1','S2']:"
                  'S2':S2 if S2 else Oscilloscope('S2', [])} 
        self.set_fluence_boards()
        self.set_transimpedance()

    
    def set_fluence_boards(self):
        """
        ONLY FOR INITIALIZATION
        Map of each board names for each batch
        same thing for the fluence
        """
        none = NO_BOARD
        batch = self.batch_number        
        for S,scope in self.S.items():
               ### default to zero because most of them are unirradiated and not angled
            fluences = (0, 0, 0, 0)     
            # angles = (0, 0, 0, 0)     ### I am not using this because it's inconsistent, I keep the RUNLOG angles
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
                logging.warning(f"in set_fluence_boards(), Batch:{batch} does not have any board names assigned")
                if S=="S1": boards = (none, none, none, none)
                elif S=="S2":   boards =(none, none, none, none)
            
            if scope is not None:
                scope.channels['Ch1'].board, scope.channels['Ch2'].board, \
                            scope.channels['Ch3'].board, scope.channels['Ch4'].board = boards 
                scope.channels['Ch1'].fluence, scope.channels['Ch2'].fluence, \
                            scope.channels['Ch3'].fluence, scope.channels['Ch4'].fluence = fluences 
            else:
                logging.warning(f"in set_fluence_boards(), (at least) one oscilloscope NOT provided in initialization of {S} in batch {self.batch_number}, boards and fluence NOT set")
            
            
    def set_transimpedance(self): ### move it after set_fluence_boards() afterwards
        """
        ONLY FOR INITIALIZATION
        assigns transimpedance to each sensor depending on the board
        """
        single_ch_transimpedance = 4700 #mV*ps/fC (I think)
        four_ch_transimpedance = 4700 ### actually this was wrong, they are all 4700
        for S,scope in self.S.items():
            if scope is not None:
                for ch, sensor in scope.channels.items():
                    if 'CERN' in sensor.board and 'CERN-4' not in sensor.board: ### boards CERN-1,CERN-2,CERN-3
                        sensor.transimpedance = four_ch_transimpedance
                    elif sensor.board == NO_BOARD:                              ### no board name
                        logging.info(f"in set_transimpedance(): No board name assigned: no transimpedance set")
                    elif sensor.board is not None and sensor.board != NO_BOARD: ### all the other options
                        sensor.transimpedance = single_ch_transimpedance
                    else:                                                       ### probably board name is None
                        logging.error("set_transimpedance(): Invalid board name")
            else:
                logging.error(f"No oscilloscope provided in initialization of {S} in batch {self.batch_number}s, transimpedance NOT set")

