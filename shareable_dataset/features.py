import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
import os

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
import os
from scipy.signal import welch

# Extract LFP metrics #
def extract_lfp_metrics(lfp_signal, fs):
    """
    Extract common LFP spectral features from a 1D LFP signal.
    """
    # Power Spectral Density (Welch’s method)
    freqs, psd = welch(lfp_signal, fs=fs, nperseg=int(fs*2))  # 2-second window

    # Define frequency bands (Hz)
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (13, 30),
        'gamma': (30, 90)
    }

    features = {}

    # Compute band power using area under the PSD curve
    for band_name, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.trapz(psd[idx_band], freqs[idx_band])
        features[f'{band_name}_power'] = band_power

    # Add general metrics
    features['total_power'] = np.trapz(psd, freqs)
    features['signal_rms'] = np.sqrt(np.mean(lfp_signal**2))
    features['variance'] = np.var(lfp_signal)

    return features

# Main #
path = "/Users/naziba/Desktop/Project_5_BME1500/shareable_dataset/neurons"  # Folder with .smr files
results = []

for filename in glob.glob(os.path.join(path, "*.smr")):
    reader = neo.io.Spike2IO(filename)
    block = reader.read(lazy=False)[0]
    segments = block.segments[0]

    # Extract raw analog signal (LFP)
    analogsignal = np.array(segments.analogsignals[0]).T[0]
    spike_times = np.array(segments.events[0])
    sampling_frequency = float(segments.analogsignals[0].sampling_rate)
    time = np.arange(0, len(analogsignal)) / sampling_frequency

    # Spike-based metrics #
    recording_duration = time[-1]
    firing_rate = len(spike_times) / recording_duration
    isis = np.diff(spike_times)
    cv = np.std(isis) / np.mean(isis)
    burst_index = np.sum(isis < 0.01) / len(isis)

    # LFP-based metrics #
    lfp_features = extract_lfp_metrics(analogsignal, sampling_frequency)

    # Combine all features #
    feature_dict = {
        "Filename": os.path.basename(filename),
        "FiringRate_Hz": firing_rate,
        "ISI_Mean_s": np.mean(isis),
        "CV_ISI": cv,
        "BurstIndex": burst_index
    }
    feature_dict.update(lfp_features)

    results.append(feature_dict)

# Save all results #
df = pd.DataFrame(results)
df.to_excel("neuron_features_with_lfp.xlsx", index=False)
# df.to_csv("neuron_features_with_lfp.csv", index=False)
print(df.head())

