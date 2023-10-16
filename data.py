import pandas as pd
import glob
import psutil

path = 'E:\ML-Research\CICIoT2023'

filenames = glob.glob(path + "\*.csv")

chunks = []

# Get the current memory usage
memory = psutil.virtual_memory()
print("Used memory before loading data:", memory.used)

for file in filenames:
    for chunk in pd.read_csv(file, chunksize=50000):
        chunks.append(chunk)

memory = psutil.virtual_memory()
print("Used memory after loading data:", memory.used)

# Concatenate all chunks into a single DataFrame
finalcsv = pd.concat(chunks, ignore_index=True)

#Number of rows in finalcsv
row_count = finalcsv.shape[0]
print("Number of labeled attacks in the dataset: ", row_count)