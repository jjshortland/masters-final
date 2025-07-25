import joblib
import pandas as pd
import numpy as np
import librosa
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

loaded_model = joblib.load('vad_svm_model.joblib')
loaded_scaler = joblib.load('vad_scaler.joblib')

metadata = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_freesound_dataset/metadata.csv')
test_metadata = metadata[metadata['split'] == 'test']


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


base_dir = '/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_freesound_dataset'


def prepare_set(df):
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


X_test, y_test = prepare_set(test_metadata)

X_test = loaded_scaler.transform(X_test)
y_pred = loaded_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print("SVM Evaluation on Test Set:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")