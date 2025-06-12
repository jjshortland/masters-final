import os
import pandas as pd

non_speech_path = '/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset/non_speech'
speech_path = '/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset/speech'


non_speech_entries = os.listdir(non_speech_path)

non_speech_df = pd.DataFrame({
    'filepath': non_speech_entries,
    'label': 'non_speech',
    'verified': 'False'
})

speech_entries = os.listdir(speech_path)

speech_df = pd.DataFrame({
    'filepath': speech_entries,
    'label': 'speech',
    'verified': 'False'
})

df = pd.concat([non_speech_df, speech_df], ignore_index=True)

df.to_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/freesound_dataset/metadata_redo.csv', index=False)

