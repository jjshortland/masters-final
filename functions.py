import librosa
import numpy as np
import os

def extract_mfcc(filepath, sr=16000, n_mfcc=13):
    y, sr = librosa.load(filepath, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate((mfcc, delta, delta2), axis=0)
    mfcc_mean = np.mean(combined, axis=1)
    mfcc_std = np.std(combined, axis=1)
    features = np.concatenate([mfcc_mean, mfcc_std])
    return features

def prepare_set(df, base_dir):
    x = []
    y = []
    for _, row in df.iterrows():
        label = row['label']
        filepath = os.path.join(base_dir, row['filepath'])
        try:
            features = extract_mfcc(filepath)
            x.append(features)
            y.append(1 if label == 'speech' else 0)
        except Exception as e:
            print(f'Failed to process {filepath}: {e}.')
    return np.array(x), np.array(y)