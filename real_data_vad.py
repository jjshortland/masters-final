from functions import prepare_set_filename
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# train_metadata = pd.read_csv('/Users/jamesshortland/Desktop/labels/training_dataset.csv')
train_metadata = pd.read_csv('/Users/jamesshortland/Desktop/labels/unbalanced_paula_training.csv')
test_metadata = pd.read_csv('/Users/jamesshortland/Desktop/labels/test_dataset.csv')
# test_metadata = pd.read_csv('/Users/jamesshortland/Desktop/labels/test_balanced.csv')

base_dir = '/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_paula_recordings'

X_train, y_train = prepare_set_filename(train_metadata, base_dir)
X_test, y_test = prepare_set_filename(test_metadata, base_dir)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'C': [1],
    'gamma': ['scale'],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, scoring='f1', cv=5)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print("SVM Evaluation on Test Set:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
