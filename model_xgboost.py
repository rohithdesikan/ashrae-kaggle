# %%
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd
import os
import time
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage


def localpath(fn):
    datadir = os.path.join(os.getcwd(), 'datasets', fn)
    return datadir


def rmsle(y_pred, y_test):
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


BUCKET_NAME = 'ashrae-kaggle-bucket'


# pd.options.display.float_format = '{:.0f}'.format

# %%
s = time.time()
table = pq.read_table(localpath('joined_df.parquet'))
df_orig = table.to_pandas()
e = time.time()
print('Time taken to read in joined file:', e-s)

# %%
# Data Engineering
df = df_orig[df_orig['site_id'] == 0].copy()
df[['site_id', 'building_id', 'meter']] = df[[
    'site_id', 'building_id', 'meter']].astype('int32')
df[['air_temperature', 'dew_temperature', 'wind_speed', 'meter_reading']] = df[[
    'air_temperature', 'dew_temperature', 'wind_speed', 'meter_reading']].astype('float32')
df['primary_use'] = df['primary_use'].astype('category')
df.drop('timestamp', axis=1, inplace=True)
cat_cols = ['site_id', 'building_id', 'meter', 'primary_use']
dfcat = pd.get_dummies(df, columns=cat_cols)

# %%
# Set up training and test data
y = df['meter_reading'].values
df.drop('meter_reading', axis=1, inplace=True)
train_cols = dfcat.columns.values.tolist()
full_arr = np.array(dfcat)

# %%
# arr = sparse.csr_matrix(full_arr)

# %%
X_train, X_val, y_train, y_val = train_test_split(
    full_arr, y, test_size=0.2, random_state=42)


# %%
