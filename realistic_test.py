import pandas as pd

df = pd.read_csv('/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_freesound_dataset/metadata.csv')
def build_realistic_test_set(df, speech_ratio=0.1):
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    test_speech = test_df[test_df['label'] == 'speech']
    test_non_speech = test_df[test_df['label'] == 'non_speech']
    non_speech_size = len(test_non_speech)
    speech_clips_wanted = int((speech_ratio * non_speech_size) / (1 - speech_ratio))

    speech_subset = test_speech.sample(n=speech_clips_wanted)
    balanced_test_df = pd.concat([speech_subset, test_non_speech, train_df, val_df]).sample(frac=1).reset_index(drop=True)
    return balanced_test_df

