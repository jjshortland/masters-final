import pandas as pd
from pathlib import Path

file_folder = Path('/Users/jamesshortland/Desktop/labels')

complete_df = pd.concat(
    (pd.read_csv(csv) for csv in file_folder.glob('*.csv')),
    ignore_index=True
)

complete_df['id_num'] = complete_df['filename'].str.extract(r'_(\d{2})-')[0].astype(int)

complete_df['chunk_num'] = complete_df['filename'].str.extract(r'_chunk(\d+)')[0].astype(int)

complete_df = complete_df.sort_values(['id_num', 'chunk_num']).drop(columns=['id_num', 'chunk_num'])

complete_df.to_csv('/Users/jamesshortland/Desktop/labels/complete_annotations.csv', index=False)
