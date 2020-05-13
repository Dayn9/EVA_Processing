import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import pyxdf # pip install pyxdf

# %% READ-ME
# this script is used for post analysis of EMG signals and to test out processing techniques

# the command (pip install pyxdf) is necessary to make use of the pyxdf library
# recommended: use the cell structure to load xdf once and avoid loading times

FILENAME = 'r2.xdf' #ex: 'block_T1.xdf'
Hz = 2000

# %% load in the XDF file from current directory
# SOURCE (modified):
# https://github.com/xdf-modules/xdf-python/blob/d642dbf86f17b8dd94cce56ff339dd57e6d3774a/example/example.py

logging.basicConfig(level=logging.INFO)  # Use logging.INFO to reduce output.
fname = os.path.abspath(os.path.join(os.path.dirname(__file__), FILENAME)) # gets xdf from CURRENT DIRECTORY
streams, fileheader = pyxdf.load_xdf(fname)

print("Found {} streams:".format(len(streams)))
for ix, stream in enumerate(streams):
    print("Stream {}: {} - type {} - uid {} - shape {} at {} Hz (effective {} Hz)".format(
        ix + 1, stream['info']['name'][0],
        stream['info']['type'][0],
        stream['info']['uid'][0],
        (int(stream['info']['channel_count'][0]), len(stream['time_stamps'])),
        stream['info']['nominal_srate'][0],
        stream['info']['effective_srate'])
    )
    if any(stream['time_stamps']):
        print("\tDuration: {} s".format(stream['time_stamps'][-1] - stream['time_stamps'][0]))
print("Done.")

# %% Data Extraction
# Extract the raw EMG data from the streams

CHANNELS = 2 #default to 2

emgData = [] #(largeNum,)
emgTimes = [] #(largeNum, CHANNELS)

smooth = []
smoothTimes = []

#find the EMG stream
for ix, stream in enumerate(streams):
    if(stream['info']['type'][0] == 'EMG'):
        CHANNELS = int(stream['info']['channel_count'][0])
        emgTimes = stream['time_stamps']
        emgData = stream['time_series']
    if(stream['info']['type'][0] == 'Smooth EMG'):
        CHANNELS = int(stream['info']['channel_count'][0])
        smoothTimes = stream['time_stamps']
        smooth = stream['time_series']
        

# %% Filtering

def bandpass(data):
    x = np.zeros(CHANNELS) # High Pass Result
    e = np.zeros(CHANNELS) # Low Pass Result
    od = np.zeros(CHANNELS) # old data
    
    filtered = []
    
    #loop over each time step
    for d in data:
        #loop over channels at timestep
        for i in range(CHANNELS):
            # High pass
            x[i] = 0.8642 * (x[i] + d[i] - od[i])
            #Low pass
            e[i] = 0.611 * x[i] + e[i] * (1 - 0.611)
            #store old data
            od[i] = d[i]
        
        #add to new filtered data 
        filtered.append(e.copy())   
    
    return filtered

# %% Smoothing

def sliding_rms(data, window):
    smoothed = []    
    rms = [[],[]] # 2 lists (used as queues)
    
    print("Smoothing...")
    for d in data:
        #dequeue oldest data when full
        if(len(rms[0]) == window):
            del rms[0][0] 
            del rms[1][0]
        #enqueue newest data     
        rms[0].append(abs(d[0]))
        rms[1].append(abs(d[1]))
        
        smoothed.append([sum(rms[0])/window, sum(rms[1])/window])
        #smoothed.append([max(rms[0]), max(rms[1])])
                
    return smoothed

# =============================================================================
        
def sliding_rms_weighted(data, window):
    smoothed = []    
    rms = [[],[]] # 2 lists (used as queues)
    
    window = round(window)
    
    #initialize linear weights  domain: [0, window) -> range: [0, 2)
    weights = np.zeros(window)
    for i in range(window):
        weights[i] = (i * 2/ window)
    
    print("Smoothing...")
    for d in data:
        #dequeue oldest data when full
        if(len(rms[0]) == window):
            del rms[0][0] 
            del rms[1][0]
        #enqueue newest data     
        rms[0].append(abs(d[0]))
        rms[1].append(abs(d[1]))
        
        smoothed.append([sum(rms[0]), sum(rms[1])])
        
    return smoothed    

# %% Calibration

# Goal of calibration is to best approximate these variables for rest, flex, and ext
rest_avg = np.zeros(CHANNELS) 
flex_max = np.zeros(CHANNELS) 
ext_max = np.zeros(CHANNELS) 

def calibrate(data):
    
    seconds = 5
    samples = Hz * seconds # samples in calibration
    
# CALIBRATION METHODS
    
    def calibrate_rest(d):
        for i in range(CHANNELS):
            rest_avg[i] += abs(d[i]) / samples
   
    def calibrate_flex(d):
        for i in range(CHANNELS):
            #flex_max[i] = max(abs(d[i]), flex_max[i])
            flex_max[i] += abs(d[i]) / samples

    def calibrate_ext(d):
        for i in range(CHANNELS):
            #ext_max[i] = max(abs(d[i]), ext_max[i])
            ext_max[i] += abs(d[i]) / samples
    
    # Calibration Process
    # 10000 comes from 2000 Hz * 5 seconds
    for d in data[0:samples]:
        calibrate_rest(d)
        
    for d in data[samples:samples*2]:
        calibrate_flex(d)
        
    for d in data[samples*2:samples*3]:
        calibrate_ext(d)
    
    
    #DEBUG info and pluts
    print('REST AVG', rest_avg)
    print('FLEX MAX', flex_max)
    print('EXT MAX', ext_max)
    
    
# %% Normalization

def normalize(data):
    normalized = [[],[]]
    
    print("Normalizing...")

    for d in data:  
        #normalize data
        f_norm = (d[0] - rest_avg[0]) / (flex_max[0] - rest_avg[0])
        e_norm = (d[1] - rest_avg[1]) / (ext_max[1] - rest_avg[1])
    
        #clamp between 0(rest) and 1(max)
        normalized[0].append(max(min(f_norm, 1), 0)) #flex norm
        normalized[1].append(max(min(e_norm, 1), 0)) #ext norm

    return normalized

# %% RUN functions and Plot results
plt.figure()    
plt.title("EMG")
plt.plot(emgTimes, emgData)    

#filter
emgData_filtered = bandpass(emgData)
   
plt.figure()    
plt.title("FILTERED")
plt.plot(emgTimes, emgData_filtered)
 
#smooth
emgData_smooth = sliding_rms_weighted(emgData_filtered, 0.75 * Hz)
calibrate(emgData_smooth) #calibrate on the smoothed data

plt.figure()
plt.title("SMOOTHED EXPECTED")
plt.plot(emgTimes, emgData_smooth)
plt.axhline(rest_avg[0])
plt.axhline(rest_avg[1])
plt.axhline(flex_max[0])
plt.axhline(ext_max[1])

plt.figure()
plt.title("SMOOTHED ACTUAL")
plt.plot(smoothTimes, (smooth.T[0:2]).T)

#normalize
emgData_normalized = normalize(emgData_smooth)

plt.figure()
plt.title("FLEX NORMALIZED")
plt.plot(emgTimes, emgData_normalized[0])
plt.fill_between(emgTimes, emgData_normalized[0])
plt.axhline(0)

plt.figure()
plt.title("EXT NORMALIZED")
plt.plot(emgTimes, emgData_normalized[1])
plt.fill_between(emgTimes, emgData_normalized[1])
plt.axhline(0)