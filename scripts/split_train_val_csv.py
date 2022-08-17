from email import header
from sklearn.model_selection import train_test_split
import pandas as pd


csv_path = '/home/jzsherlock/my_lab/datasets/DRAC22/train_val/dataset_B/b_iqa_label.csv'
imgpath_colname = 'image name'
label_colname = 'image quality level'
val_ratio = 0.2

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