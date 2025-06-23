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
    mfcc_mean = np.mean(combined, axis=1)
    return mfcc_mean

base_dir = '/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset'

def prepare_set(df):
    X = []
    y = []
    for _, row in df.iterrows():
        label = row['label']
        filepath = os.path.join(base_dir, label, row['filepath'])
        try:
            features = extract_mfcc(filepath)
            X.append(features)
            y.append(1 if label == 'speech' else 0)
        except Exception as e:
            print(f'Failed to process {filepath}: {e}.')
    return np.array(X), np.array(y)

X_train, y_train = prepare_set(train_metadata)
X_val, y_val = prepare_set(val_metadata)
X_test, y_test = prepare_set(test_metadata)

class VADNet(nn.Module):
    def __init__(self, input_dim=39):
        super(VADNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.classifier(x)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

model = VADNet(input_dim=39)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_outputs = model(val_X)
            loss = criterion(val_outputs, val_y)
            val_loss += loss.item()

            predicted = (val_outputs > 0.5).float()
            correct += (predicted == val_y).sum().item()
            total += val_y.size(0)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {running_loss:.4f} | Val Loss: {val_loss:.4f} "
          f"| Val Acc: {correct / total:.4f}")
