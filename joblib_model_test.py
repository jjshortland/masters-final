import joblib
import pandas as pd
from functions import prepare_set
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

loaded_model = joblib.load('vad_svm_model.joblib')
loaded_scaler = joblib.load('vad_scaler.joblib')

metadata = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_freesound_dataset/metadata.csv')
test_metadata = metadata[metadata['split'] == 'test']


base_dir = '/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_freesound_dataset'


X_test, y_test = prepare_set(test_metadata, base_dir)

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