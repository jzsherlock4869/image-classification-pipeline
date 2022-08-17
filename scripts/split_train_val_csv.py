from email import header
from sklearn.model_selection import train_test_split
import pandas as pd


csv_path = '/path/to/train_val.csv'
imgpath_colname = 'image_path'
label_colname = 'label'
val_ratio = 0.2  # 20% of all samples split for evaluation, others for training

# output split train/val csv in the same folder, with postfix
train_csv_path = csv_path[:-4] + '_train.csv'
val_csv_path = csv_path[:-4] + '_val.csv'

df = pd.read_csv(csv_path)
train_df, val_df = train_test_split(df, test_size=val_ratio)
train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)
print('train val csv file split done')
print('ori csv path: ', csv_path)
print('train csv path: ', train_csv_path)
print('val csv path: ', val_csv_path)