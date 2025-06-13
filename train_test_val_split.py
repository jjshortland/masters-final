import pandas as pd
import numpy as np

df = pd.read_csv('freesound_dataset/metadata.csv')

train_percent = 0.8
test_percent = 0.1
val_percent = 0.1

df = df[df['verified'] == "True"].reset_index(drop=True)

def split_classes(df_class, label_name):
    df_class = df_class.sample(frac=1, random_state=42).reset_index(drop=True)
    n_total = len(df_class)
    n_train = int(n_total*train_percent)
    n_test = int(n_total*test_percent)

    print(n_total, n_train, n_test)


    df_class.loc[:n_train, 'split'] = 'train'
    df_class.loc[n_train:n_train+n_test, 'split'] = 'test'
    df_class.loc[n_train+n_test:, 'split'] = 'val'

    print(f"{label_name} → {n_train} train, {n_test} test, {n_total - n_train - n_test} val")
    return df_class

df_speech = df[df['label'] == 'speech']
df_speech = split_classes(df_speech, 'speech')

df_non_speech = df[df['label'] == 'non_speech']
df_non_speech = split_classes(df_non_speech, 'non speech')

df_split = pd.concat([df_speech, df_non_speech])
df_split = df_split.sample(frac=1, random_state=42).reset_index(drop=True)

df_split.to_csv('freesound_dataset/metadata_split.csv', index=False)

print('Dataset split complete!')



