import pandas as pd
from pathlib import Path
import shutil

metadata = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset/metadata.csv')
metadata = metadata[metadata['verified'] == 'True']

base_path = Path('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset')
destination_base_path = Path('/Users/jamesshortland/Desktop/freesound_speech_dataset')

for _, row in metadata.iterrows():
    label = row['label']
    filename = row['filepath']
    full_path = base_path/label/filename
    destination_path = destination_base_path/label

    shutil.copy(full_path, destination_path)

metadata_destination = destination_base_path/'metadata.csv'
metadata.to_csv(metadata_destination)
