import pandas as pd

big_df = pd.read_csv('/Users/jamesshortland/Desktop/labels/complete_annotations.csv')
big_df['id_num'] = big_df['filename'].str.extract(r'_(\d{2})-')[0].astype(int)

train_df = big_df[big_df['id_num'].isin([11, 12, 13]) & (big_df['label'] != 'flagged')].reset_index(drop=True)

train_df.to_csv('/Users/jamesshortland/Desktop/labels/unbalanced_paula_training.csv')