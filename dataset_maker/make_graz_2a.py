import mne
import numpy as np
import os
import pickle
from tqdm import tqdm

def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)

def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
    fs = 200.0
    [numChan, numPoints] = signals.shape
    # for i in range(numChan):  # standardize each channel
    #     if np.std(signals[i, :]) > 0:
    #         signals[i, :] = (signals[i, :] - np.mean(signals[i, :])) / np.std(signals[i, :])
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])  # channel that had the detected thing
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):  # for each event
        chan = int(EventData[i, 0])  # chan is channel
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        # print (offset + start - 2 * int(fs), offset + end + 2 * int(fs), signals.shape)
        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels]

def readGDF(fileName):
    raw = mne.io.read_raw_gdf(fileName, preload=True)
    raw.filter(l_freq=0.1, h_freq=75.0)
    raw.notch_filter(50.0)
    raw.resample(200)
    
    signals = raw.get_data()[:22]
    times = raw.times
    events, _ = mne.events_from_annotations(raw)
    
    return [signals, times, events]

def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir):
    for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname.endswith('.gdf'):
                print("\t%s" % fname)
                try:
                    [signals, times, events] = readGDF(os.path.join(dirName, fname))
                    signals, offending_channels, labels = BuildEvents(signals, times, events)

                    for idx, (signal, offending_channel, label) in enumerate(zip(signals, offending_channels, labels)):
                        sample = {
                            "signal": signal,
                            "offending_channel": offending_channel,
                            "label": int(label) - 1,  # Adjusting labels to be 0-indexed
                        }
                        save_pickle(sample, os.path.join(OutDir, f"{fname.split('.')[0]}-{idx}.pkl"))

                except Exception as e:
                    print(f"Error processing {fname}: {str(e)}")
                    continue

    return Features, Labels, OffendingChannels

# Set up directories
root = "/root/LaBraM/datasets/BCICIV_2a_gdf"
train_out_dir = os.path.join(root, "processed_train")
eval_out_dir = os.path.join(root, "processed_eval")
if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)

# Process training data
fs = 200
TrainFeatures = np.empty((0, 22, fs))  # 22 channels, 1 second at 200 Hz
TrainLabels = np.empty([0, 1])
TrainOffendingChannel = np.empty([0, 1])
load_up_objects(root, TrainFeatures, TrainLabels, TrainOffendingChannel, train_out_dir)

# Process evaluation data
EvalFeatures = np.empty((0, 22, fs))  # 22 channels, 1 second at 200 Hz
EvalLabels = np.empty([0, 1])
EvalOffendingChannel = np.empty([0, 1])
load_up_objects(root, EvalFeatures, EvalLabels, EvalOffendingChannel, eval_out_dir)

print(f"Processed {len(os.listdir(train_out_dir))} training samples and {len(os.listdir(eval_out_dir))} evaluation samples.")