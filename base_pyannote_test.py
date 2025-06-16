from pyannote.audio import Pipeline
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pipeline = Pipeline.from_pretrained(
    'pyannote/voice-activity-detection',
    use_auth_token=''
)

metadata = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset/metadata_split.csv')
#metadata = metadata[(metadata['split'] == 'test') & (metadata['verified'] == True)]

y_true = []
y_pred = []

base_path = Path('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset')

for _, row in metadata.iterrows():
    label = row['label']
    filename = row['filepath']
    full_path = base_path / label / filename

    try:
        wav_path = str(full_path)
        vad_result = pipeline(wav_path)

        is_speech = not vad_result.get_timeline().empty()

        predicted_label = 'speech' if is_speech else 'non_speech'

        y_pred.append(predicted_label)
        y_true.append(label)

    except Exception as e:
        print(f'Error processing {filename}: {e}')

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label='speech')
recall = recall_score(y_true, y_pred, pos_label='speech')
f1 = f1_score(y_true, y_pred, pos_label='speech')

print("Pyannote VAD Evaluation:")
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

incorrect_df.to_csv("base_pyannote_misclassified.csv", index=False)
false_negatives.to_csv("base_pyannote_false_negatives.csv", index=False)