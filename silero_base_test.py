from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = load_silero_vad()

metadata = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset/metadata_split.csv')
#metadata = metadata[(metadata['split'] == 'test') & (metadata['verified'] == True)]

y_true = []
y_pred = []

base_path = Path('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset')

for _, row in metadata.iterrows():
    label = row['label']
    filename = row['filepath']
    full_path = base_path/label/filename

    try:
        wav = read_audio(str(full_path), sampling_rate=16000)
        timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

        predicted_label = "speech" if timestamps else "non_speech"

        y_true.append(label)
        y_pred.append(predicted_label)

    except Exception as e:
        print(f'Error processing {filename}: {e}')

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label='speech')
recall = recall_score(y_true, y_pred, pos_label='speech')
f1 = f1_score(y_true, y_pred, pos_label='speech')

print("Silero VAD Evaluation:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

results_df = metadata.copy()
results_df = results_df.reset_index(drop=True)
results_df["predicted_label"] = y_pred
results_df["true_label"] = y_true

incorrect_df = results_df[results_df["predicted_label"] != results_df["true_label"]]

false_negatives = incorrect_df[
    (incorrect_df["true_label"] == "speech") & (incorrect_df["predicted_label"] == "non_speech")]

print(f"\nMisclassified clips: {len(incorrect_df)} total")
print(f"False negatives (missed speech): {len(false_negatives)}")

print('Saving results to CSV...')

incorrect_df.to_csv("silero_base_misclassified.csv", index=False)
false_negatives.to_csv("silero_base_false_negatives.csv", index=False)
