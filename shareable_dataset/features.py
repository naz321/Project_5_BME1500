import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
import os

path = "/Users/naziba/Desktop/Project_5_BME1500/shareable_dataset/neurons" # Enter the path where the .smr files are located on your computer

results = []

for filename in glob.glob(os.path.join(path, "*.smr")):
    reader = neo.io.Spike2IO(filename)
    block = reader.read(lazy=False)[0]
    segments = block.segments[0]

    analogsignal = np.array(segments.analogsignals[0]).T[0]
    spike_times = np.array(segments.events[0])
    sampling_frequency = float(segments.analogsignals[0].sampling_rate)
    time = np.arange(0, len(analogsignal)) / sampling_frequency

    # Compute metrics
    recording_duration = time[-1]
    firing_rate = len(spike_times) / recording_duration
    isis = np.diff(spike_times)
    cv = np.std(isis) / np.mean(isis)
    burst_index = np.sum(isis < 0.01) / len(isis)

    results.append({
        "File ": os.path.basename(filename),
        "FiringRate_Hz ": firing_rate,
        "ISI_Mean_s ": np.mean(isis),
        "CV_ISI ": cv,
        "BurstIndex ": burst_index
    })

df = pd.DataFrame(results)
df.to_csv("neuron_features.csv", index=False)
print(df.head())
