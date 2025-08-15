import webrtcvad
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pydub import AudioSegment
from pathlib import Path


vad = webrtcvad.Vad()
vad.set_mode(3)

metadata = pd.read_csv('/Users/jamesshortland/Desktop/labels/complete_annotations.csv')
metadata = metadata[metadata['label'] != 'flagged']

# metadata = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset/metadata_split.csv')
# metadata = metadata[(metadata['split'] == 'test') & (metadata['verified'] == True)]

def is_speech(filepath):
    try:
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

        raw_audio = audio.raw_data
        frame_bytes = int(16000*30/1000) * 2
        frames = [raw_audio[i:i+frame_bytes] for i in range(0, len(raw_audio), frame_bytes)]

        speech_frames = [vad.is_speech(frame, 16000) for frame in frames if len(frame) == frame_bytes]
        return any(speech_frames)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

y_true = []
y_pred = []

# base_path = Path('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset')
base_path = Path('/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_paula_recordings')

for _, row in metadata.iterrows():
    label = row['label']
    # filename = row['filepath']
    filename = row['filename']
    # full_path = base_path/label/filename
    full_path = base_path/filename

    predicted = 'speech' if is_speech(full_path) else 'non_speech'

    y_pred.append(predicted)
    y_true.append(label)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label='speech')
recall = recall_score(y_true, y_pred, pos_label='speech')
f1 = f1_score(y_true, y_pred, pos_label='speech')

print("WebRTC VAD Evaluation:")
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

incorrect_df.to_csv("webrtc_real_data_misclassified.csv", index=False)
false_negatives.to_csv("webrtc_real_data_false_negatives.csv", index=False)



