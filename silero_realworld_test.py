from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = load_silero_vad()
metadata = pd.read_csv('/Users/jamesshortland/Desktop/labels/complete_annotations.csv')
metadata = metadata[metadata['label'] != 'flagged']

base_dir = Path('/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_paula_recordings')

y_true = []
y_pred = []

for _, row in metadata.iterrows():
    label = row['label']
    filename = row['filename']
    full_path = base_dir / filename

    try:
        wav = read_audio(str(full_path), sampling_rate=16000)
        timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

        predicted_label = "speech" if timestamps else "non_speech"

        y_true.append(label)
        y_pred.append(predicted_label)

    except Exception as e:
        print(f'Error processing {full_path}: {e}')

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label='speech')
recall = recall_score(y_true, y_pred, pos_label='speech')
f1 = f1_score(y_true, y_pred, pos_label='speech')

print("Silero VAD Evaluation:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")