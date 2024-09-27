#transfer to train, eval, and test

import mne
import numpy as np
import os
import pickle
from tqdm import tqdm

rootOld = "/root/LaBraM/datasets/BCICIV_2a_gdf"
root = "/root/LaBraM/share/graz_2a"
seed = 4523
np.random.seed(seed)

train_files = os.listdir(os.path.join(rootOld, "processed_train"))
print("train files", len(train_files))

train_sub = list(set([f.split("_")[0] for f in train_files]))
print("train sub", len(train_sub))
test_files = os.listdir(os.path.join(rootOld, "processed_eval"))

val_sub = np.random.choice(train_sub, size=int(
    len(train_sub) * 0.2), replace=False)
train_sub = list(set(train_sub) - set(val_sub))
val_files = [f for f in train_files if f.split("_")[0] in val_sub]
train_files = [f for f in train_files if f.split("_")[0] in train_sub]

for file in train_files:
    os.system(f"cp {os.path.join(rootOld, 'processed_train', file)} {os.path.join(root, 'processed', 'processed_train')}")
for file in val_files:
    os.system(f"cp {os.path.join(rootOld, 'processed_train', file)} {os.path.join(root, 'processed', 'processed_eval')}")
for file in test_files:
    os.system(f"cp {os.path.join(rootOld, 'processed_eval', file)} {os.path.join(root, 'processed', 'processed_test')}")
