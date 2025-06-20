import pandas as pd
from pydub import AudioSegment
import numpy as np
from pathlib import Path

metadata = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset/metadata_split.csv')

def preprocess_clips(filepath):
    audio = AudioSegment.from_file(filepath)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio = audio.apply_gain(-audio.max_dBFS)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
    return samples, 16000

base_path = Path('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset')

# for _, row in metadata.iterrows():
#     label = row['label']
#     filename = row['filepath']
#     full_path = base_path/label/filename
#
#     print(preprocess_clips(full_path))

n_mfcc = 13
sample_rate = 16000