import mne
import numpy as np
import os
import pickle
from tqdm import tqdm

def readGDF(fileName):
    raw = mne.io.read_raw_gdf(fileName, preload=True)
    
    # Apply TUEV-style filtering
    raw.filter(l_freq=0.1, h_freq=75.0)
    raw.notch_filter(50.0)
    
    # Resample to 200 Hz (TUEV standard)
    raw.resample(200)
    
    signals = raw.get_data(units='uV')  # Get data in microvolts
    times = raw.times
    events, event_id = mne.events_from_annotations(raw)
    
    print(f"Signals shape: {signals.shape}, Events shape: {events.shape}")
    print(f"Event IDs: {event_id}")
    
    return [signals, times, events, event_id, raw]

def BuildEvents(signals, times, events, event_id, raw):
    fs = 200.0  # New sampling rate (TUEV standard)
    numChan = signals.shape[0]
    numPoints = signals.shape[1]
    numEvents = len(events)
    
    # 5-second window (TUEV standard)
    window_samples = int(fs * 5)
    
    features = np.zeros([numEvents, numChan, window_samples], dtype=np.float64)
    labels = np.zeros([numEvents, 1], dtype=np.int32)
    
    for i, event in enumerate(events):
        start = event[0]
        end = start + window_samples
        if end > numPoints:
            start = numPoints - window_samples
            end = numPoints
        features[i] = signals[:, start:end]
        labels[i, 0] = event[2]
    
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    return [features, labels]

def process_graz_dataset(BaseDir, OutDir):
    print("BaseDir: ", BaseDir)
    gdf_files = [f for f in os.listdir(BaseDir) if f.endswith('.gdf')]
    
    print(f"Found {len(gdf_files)} .gdf files")
    
    for fname in tqdm(gdf_files):
        print(f"\nProcessing file: {fname}")
        try:
            [signals, times, events, event_id, raw] = readGDF(os.path.join(BaseDir, fname))
            features, labels = BuildEvents(signals, times, events, event_id, raw)
            
            for idx, (feature, label) in enumerate(zip(features, labels)):
                sample = {
                    "signal": feature,
                    "offending_channel": np.zeros((1,)),  # Placeholder, as we don't have this info for Graz 2a
                    "label": int(label) - 1,  # Adjusting labels to be 0-indexed
                }
                with open(os.path.join(OutDir, f"{fname.split('.')[0]}-{idx}.pkl"), 'wb') as f:
                    pickle.dump(sample, f)
            
            print(f"Processed {len(features)} events from {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")
            continue

if __name__ == "__main__":
    BaseDir = "/root/LaBraM/datasets/BCICIV_2a_gdf"
    OutDir = "/root/LaBraM/datasets/BCICIV_2a_gdf/processed"
    
    if not os.path.exists(OutDir):
        os.makedirs(OutDir)
    
    process_graz_dataset(BaseDir, OutDir)