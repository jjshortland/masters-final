import pandas as pd

big_df = pd.read_csv('/Users/jamesshortland/Desktop/labels/complete_annotations.csv')
big_df['id_num'] = big_df['filename'].str.extract(r'_(\d{2})-')[0].astype(int)

speech_train = big_df[
    (big_df['id_num'].isin([11, 12, 13])) &
    (big_df['label'] == 'speech')].reset_index(drop=True)

non_speech_train = big_df[
    (big_df['id_num'].isin([11, 12, 13])) &
    (big_df['label'] == 'non_speech')
].sample(
    n=len(speech_train),
    random_state=42).reset_index(drop=True)

train_balanced = pd.concat([speech_train, non_speech_train]).sample(frac=1, random_state=42).reset_index(drop=True)

test_dataset = big_df[big_df['id_num'].isin([14,15,16])].reset_index(drop=True)

speech_test = big_df[
    (big_df['id_num'].isin([14, 15, 16])) &
    (big_df['label'] == 'speech')].reset_index(drop=True)

non_speech_test = big_df[
    (big_df['id_num'].isin([14, 15, 16])) &
    (big_df['label'] == 'non_speech')
].sample(
    n=len(speech_test),
    random_state=42).reset_index(drop=True)

test_balanced = pd.concat([speech_test, non_speech_test]).sample(frac=1, random_state=42).reset_index(drop=True)


train_balanced.to_csv('/Users/jamesshortland/Desktop/labels/training_dataset.csv')
test_dataset.to_csv('/Users/jamesshortland/Desktop/labels/test_dataset.csv')
test_balanced.to_csv('/Users/jamesshortland/Desktop/labels/test_balanced.csv')