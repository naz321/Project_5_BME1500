import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
import os

##### CHANGE THIS TO YOUR DIRECTORY #####

path = "/Users/srdjansumarac/Library/CloudStorage/OneDrive-UniversityofToronto/teaching/BME1500_FALL_2022_PROJECT5_DATA/neurons-smr-format-sorted" # Enter the path where the .smr files are located on your computer
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

##### PLOT SIGNALS #####

fig, ax = plt.subplots(2, sharex = True, )
fig.suptitle(filename, fontsize=16)    

ax[0].plot(time,analogsignal,'green')
ax[1].eventplot(spike_times, color='black')
ax[1].set_xlabel("Time (s)")

plt.show()