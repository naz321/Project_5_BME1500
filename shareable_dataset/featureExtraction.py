#Attempt 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
import os
import analysis as an
import read_file as read_file

# NOTEEEE COPY THE METADATA FILE INTO THE NEURONS FOLDER OR CHANGE THE PATH FOR IT BELOW

#### DEFINING SETTINGS
DATA_DER = './neurons/'
METADATA_FILE = "./bme1500-project-5-metadata.xlsx"
OUTPUT_FILE = './myNeuronFeatures.csv'

path = "/Users/tasnianabil/Desktop/PhD Courses/BME1500/DBSClassification/neurons" # Enter the path where the .smr files are located on your computer

#filter properties
lowFilter = 300
highFilter = 3000

#prepare to collect results
allFeaturesList = []

#### LOAD METADATA
metadata_df = pd.read_excel(METADATA_FILE)

for index, row in metadata_df.iterrows():

    #### IMPORTING DATA
    fileName = row['Filename']
    print(fileName)
    fileKey = fileName.replace(".smr","")
    print(fileKey)

    #importing .smr data
    os.chdir(path)
    reader = neo.io.Spike2IO(fileName)
    block = reader.read(lazy=False)[0]
    segments = block.segments[0]    
    
    #Parameters from .smr data
    analogsignal = np.array(segments.analogsignals[0],dtype='float64').transpose()[0] # raw analog waveform (unfiltered)
    spike_times = np.array(segments.events[0],dtype='float64') # spike timing array (i.e. exact time reletive to the start of the recording when a spike fired an action potential) 
    fs = float(segments.analogsignals[0].sampling_rate) # the number of samples per second (in Hz)
    time = np.arange(0,len(analogsignal))/fs # time vector

    #### PREPROCESSING
    spikeSignal = an.butterworth_filter(analogsignal, fs, highFilter, lowFilter)

    threshold = an.get_spike_threshold(spikeSignal,3) #3x the noise?

    ## Getting indices (sample # of spikes)
    spikeIndices = an.get_spike_indices(spikeSignal, threshold, fs,inverted=False)

    ## EXTRACTING BASIC FEATURES
    features = {
        'file_key': fileKey,
        'target': row['Target'],
        'hemisphere': row['Hemisphere'],
        'neuron': row['Neuron']
    }

    features['firing_rate'] = an.get_FR(spikeIndices, fs)
    features['burst_index'] = an.get_BI(spikeIndices)
    features['cv'] = an.get_CV(spikeIndices)

    ## EXTRACTING OSCILLATORY FEATURES
    burstDurations = an.get_spiketrain_burstduration(spikeIndices, fs)
    spikePower = an.get_spiketrain_power(spikeIndices, fs)

    features['burst_durations'] = burstDurations
    features['spike_power'] = spikePower

    allFeaturesList.append(features)

print("\...Feature Extraction complete!")

if allFeaturesList:
    finalFeatureMatrix = pd.DataFrame(allFeaturesList)
    finalFeatureMatrix.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved feature matrix to {OUTPUT_FILE}")
    print("Here's a sample of your data (first 5 rows):")
    print(finalFeatureMatrix.head())