import joblib
import pandas as pd
import librosa
import numpy as np
import os

loaded_model = joblib.load('vad_svm_model.joblib')
loaded_scaler = joblib.load('vad_scaler.joblib')

metadata = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_paula_recordings/metadata.csv')
base_dir = '/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_paula_recordings'

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
    for _, row in df.iterrows():
        filepath = os.path.join(base_dir, row['file_name'])
        try:
            features = extract_mfcc(filepath)
            x.append(features)
        except Exception as e:
            print(f'Failed to process {filepath}: {e}.')
    return np.array(x)


X = prepare_set(metadata, base_dir)
X = loaded_scaler.transform(X)
y_pred = loaded_model.predict(X)

print(X.shape)
print(y_pred.shape)

metadata['predicted_label'] = y_pred
metadata.to_csv(
    '/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_paula_recordings/metadata.csv', index=False)


