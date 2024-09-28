#transfer to train, eval, and test

import mne
import numpy as np
import os
import pickle
from tqdm import tqdm

import os
import shutil
import numpy as np

root = "/root/LaBraM/datasets/BCICIV_2a_gdf"
processed_dir = os.path.join(root, "processed")

# Create directories if they don't exist
for subdir in ["processed_train", "processed_eval", "processed_test"]:
    os.makedirs(os.path.join(processed_dir, subdir), exist_ok=True)

# Get all .pkl files in the processed directory
all_files = [f for f in os.listdir(processed_dir) if f.endswith('.pkl')]

# Initial segmentation
train_files = []
test_files = []

for file in all_files:
    source = os.path.join(processed_dir, file)
    if file[3] == 'E':
        test_files.append(file)
    elif file[3] == 'T':
        train_files.append(file)

# Split train files into train and validation
np.random.seed(4523)
train_subjects = list(set([f.split("_")[0] for f in train_files]))
val_subjects = np.random.choice(train_subjects, size=int(len(train_subjects) * 0.2), replace=False)
train_subjects = list(set(train_subjects) - set(val_subjects))

val_files = [f for f in train_files if f.split("_")[0] in val_subjects]
train_files = [f for f in train_files if f.split("_")[0] in train_subjects]

# Move files to their respective directories
for file in train_files:
    shutil.move(os.path.join(processed_dir, file), os.path.join(processed_dir, "processed_train", file))

for file in val_files:
    shutil.move(os.path.join(processed_dir, file), os.path.join(processed_dir, "processed_eval", file))

for file in test_files:
    shutil.move(os.path.join(processed_dir, file), os.path.join(processed_dir, "processed_test", file))

print(f"Processed {len(all_files)} files.")
print(f"Train files: {len(os.listdir(os.path.join(processed_dir, 'processed_train')))}")
print(f"Eval files: {len(os.listdir(os.path.join(processed_dir, 'processed_eval')))}")
print(f"Test files: {len(os.listdir(os.path.join(processed_dir, 'processed_test')))}")
