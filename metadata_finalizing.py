import pandas as pd

file_name = '/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_freesound_dataset/metadata.csv'

df = pd.read_csv(file_name)
df = df[df['verified'].astype(str) == 'True']
df.to_csv(file_name)