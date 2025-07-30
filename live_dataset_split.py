import pandas as pd
import os
from pydub import AudioSegment
from pathlib import Path

metadata = []
chunk_length = 5000
base_folder = Path('/Users/jamesshortland/PycharmProjects/Masters_Final/20250705_PaulaRecordings')
output_folder = '/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_paula_recordings'

for m4a_file in base_folder.glob('*.m4a'):
    audio = AudioSegment.from_file(m4a_file)
    for i, start in enumerate(range(0, len(audio), chunk_length)):
        chunk = audio[start:start + chunk_length]
        if len(chunk) < chunk_length:
            chunk = audio[-chunk_length:]

        chunk_file_name = f"{m4a_file.stem}_chunk{i+1}.wav"
        outpath = os.path.join(output_folder, chunk_file_name)
        chunk.export(outpath, format='wav')
        metadata.append({
            'predicted_label': None,
            'actual_label': None,
            'file_name': chunk_file_name,
            'verified': 'no'
        })

new_metadata = pd.DataFrame(metadata)
new_metadata.to_csv(os.path.join(output_folder, 'metadata.csv'), index=False)

