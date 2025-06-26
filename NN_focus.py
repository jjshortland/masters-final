import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import librosa
import numpy as np
import os

metadata = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset/metadata_split.csv')
train_metadata = metadata[metadata['split'] == 'train']
val_metadata = metadata[metadata['split'] == 'val']
test_metadata = metadata[metadata['split'] == 'test']

def extract_mfcc(filepath, sr=16000, n_mfcc=13):
    y, sr = librosa.load(filepath, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate((mfcc, delta, delta2), axis=0)
    return combined

base_dir = '/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset'

def extract_and_pad_set(df, base_dir, T_max):
    X, y = [], []
    for _, row in df.iterrows():
        label = row['label']
        filepath = os.path.join(base_dir, label, row['filepath'])
        try:
            mfcc = extract_mfcc(filepath)
            T = mfcc.shape[1]
            if T > T_max:
                mfcc = mfcc[:, :T_max]
            else:
                mfcc = np.pad(mfcc, ((0, 0), (0, T_max - T)), mode='constant')
            X.append(mfcc)
            y.append(1 if label == 'speech' else 0)
        except Exception as e:
            print(f"Failed to process {filepath}: {e}")
    return np.stack(X), np.array(y)

print(extract_and_pad_set(train_metadata, base_dir, 0))