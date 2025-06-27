import os
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from realistic_test import build_realistic_test_set

metadata = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_freesound_dataset/metadata.csv')


def evaluate_model(df, n_runs=10, speech_ratio=0.1, random_seed=None):
    all_metrics = []
    for n in range(n_runs):
        metadata = build_realistic_test_set(df, speech_ratio)
        train_metadata = metadata[metadata['split'].isin(['train', 'val'])]
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
            X = []
            y = []
            for _, row in df.iterrows():
                label = row['label']
                filepath = os.path.join(base_dir, row['filepath'])
                try:
                    features = extract_mfcc(filepath)
                    X.append(features)
                    y.append(1 if label == 'speech' else 0)
                except Exception as e:
                    print(f'Failed to process {filepath}: {e}.')
            return np.array(X), np.array(y)

        X_train, y_train = prepare_set(train_metadata)
        X_test, y_test = prepare_set(test_metadata)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        param_grid = {
            'C': [1.0],
            'gamma': ['scale'],
            'kernel': ['rbf']
        }

        grid = GridSearchCV(SVC(), param_grid, scoring='f1', cv=5)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        metrics_dict = {
            'run': n+1,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label=1),
            'recall': recall_score(y_test, y_pred, pos_label=1),
            'f1': f1_score(y_test, y_pred, pos_label=1)
        }

        all_metrics.append(metrics_dict)

    return pd.DataFrame(all_metrics)

df_results = evaluate_model(metadata, 50)
print(df_results.to_string(index=False))
print(df_results.describe())