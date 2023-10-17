import pandas as pd
import numpy as np
import dask.dataframe as dd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

path = 'E:\CICIoT2023'

filenames = glob.glob(path + "\*.csv")

chunks = []

for file in filenames:
    for chunk in pd.read_csv(file, chunksize=20000):
        chunks.append(chunk)

# Concatenate all chunks into a single DataFrame
finalcsv = pd.concat(chunks, ignore_index=True)

#Convert the labels into numbers (Random Forest can't predict text)
labels, uniques = pd.factorize(finalcsv['label'])

#Number of rows in finalcsv
row_count = finalcsv.shape[0]
print("Number of labeled attacks in the dataset: ", row_count)
print("Sample: ", finalcsv.sample())
print("Unique labels: ", uniques)

#Seperate the features and the labels
X = finalcsv.iloc[:, :46].values #the first 46 columns
y = finalcsv.iloc[:, 46].values #the 47th label column

#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=52, test_size=0.20, shuffle=True) #Shuffle the data each time with a random seed in an 80/20 split

#Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Train the model
classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=32)
classifier.fit(X_train, y_train)

#Predict
y_pred = classifier.predict(X_test)

#Convert labels back into text
reverselabel = dict(zip(range(3), uniques))
y_test = np.vectorize(reverselabel.get)(y_test)
y_pred = np.vectorize(reverselabel.get)(y_pred)

#Creating the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=["Actual Attack"], colnames=['Predicted Attack']))






