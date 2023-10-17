import pandas as pd
import numpy as np
import dask.dataframe as dd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

path = 'E:\CICIoT2023'

df = dd.read_csv(path + '\*.csv')

print("Number of rows: ", len(df))
labels, uniques = pd.factorize(df['label'])

# sampled_rows = df.sample(frac=3/len(df)).compute()

# print("Sample: ", sampled_rows)
print(f'There are {len(uniques)} labels: {uniques}')




