import os
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metadata = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_freesound_dataset/metadata.csv')
train_metadata = metadata[metadata['split'].isin(['train', 'val'])]
test_metadata = metadata[metadata['split'] == 'test']

def classifier_test(classifier):
    def extract_mfcc(filepath, sr=16000, n_mfcc=13):
        y, sr = librosa.load(filepath, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.concatenate((mfcc, delta, delta2), axis=0)
        mfcc_mean = np.mean(combined, axis=1)
        return mfcc_mean

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

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print(f"{classifier} on Test Set:")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")

svc = SVC(kernel="rbf", C=1.0, gamma="scale")
classifier_test(svc)

ran_f = RandomForestClassifier(n_estimators=100, random_state=42)
classifier_test(ran_f)

log_r = LogisticRegression(max_iter=1000)
classifier_test(log_r)

knn = KNeighborsClassifier(n_neighbors=5)
classifier_test(knn)

mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000)
classifier_test(mlp)