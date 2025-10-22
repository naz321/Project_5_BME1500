import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
import os

##### CHANGE THIS TO YOUR DIRECTORY #####

path = "/Users/naziba/Desktop/Project_5_BME1500/shareable_dataset/neurons" # Enter the path where the .smr files are located on your computer
filename = "neuron_002.smr" # Name of the file including the extension (.smr)

##### IMPORT .SMR DATA STRUCTURE INTO PYTHON (DO NOT CHANGE) #####
os.chdir(path)
reader = neo.io.Spike2IO(filename)
block = reader.read(lazy=False)[0]
segments = block.segments[0]

analogsignal = np.array(segments.analogsignals[0],dtype='float64').transpose()[0] # raw analog waveform (unfiltered)
spike_times = np.array(segments.events[0],dtype='float64') # spike timing array (i.e. exact time reletive to the start of the recording when a spike fired an action potential) 
sampling_frequency = float(segments.analogsignals[0].sampling_rate) # the number of samples per second (in Hz)
time = np.arange(0,len(analogsignal))/sampling_frequency # time vector

# Feature extraction 

# Firing rate is the average number of spikes per second::
recording_duration = time[-1]  # seconds
firing_rate = len(spike_times) / recording_duration
print("Firing Rate (Hz):", firing_rate)

# ISI and CV:
# the ISI is the time difference between consecutive spikes.
# the CV tells you how regular or irregular the firing is.  A low CV (close to 0) indicates a regular firing pattern, 
# while a high CV (around or above 1) suggests an irregular, more random pattern. 
isis = np.diff(spike_times)
cv = np.std(isis) / np.mean(isis)
print("ISI mean (s):", np.mean(isis))
print("CV of ISI:", cv)

# Burst Index:
burst_threshold = 0.01  # seconds
bursts = isis < burst_threshold
burst_index = np.sum(bursts) / len(isis)
print("Burst Index:", burst_index)

##### PLOT SIGNALS #####

fig, ax = plt.subplots(2, sharex = True, )
fig.suptitle(filename, fontsize=16)    

ax[0].plot(time,analogsignal,'green')
ax[1].eventplot(spike_times, color='black')
ax[1].set_xlabel("Time (s)")

plt.show()

