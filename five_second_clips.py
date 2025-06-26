import pandas as pd
import os
from pydub import AudioSegment

original_metadata = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset/metadata_split.csv')
base_dir = '/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset'
output_folder = '/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_freesound_dataset'
metadata_rows = []
chunk_length = 5000

for _, row in original_metadata.iterrows():
    label = row['label']
    filepath = os.path.join(base_dir, label, row['filepath'])
    split = row['split']

    audio = AudioSegment.from_file(filepath)

    for i, start in enumerate(range(0, len(audio), chunk_length)):
        chunk = audio[start:start+chunk_length]
        if len(chunk) < chunk_length:
            continue

        chunk_filename = f"{os.path.splitext(row['filepath'])[0]}_chunk_{i}.wav"
        out_path = os.path.join(output_folder, label, chunk_filename)
        chunk.export(out_path, format="wav")
        metadata_rows.append({
            'label': label,
            'verified': 'no',
            'filepath': os.path.join(label, chunk_filename),
            'split': split
        })

new_metadata = pd.DataFrame(metadata_rows)
new_metadata.to_csv(os.path.join(output_folder, 'metadata.csv'), index=False)






