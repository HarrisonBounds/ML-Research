import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

path = 'E:\CICIoT2023'
fraction = 0.1
df = dd.read_csv(path + '\*.csv').sample(frac=fraction)

#column that contains the labels
label_column = 'label'

X_train, X_test, y_train, y_test = train_test_split(df, df[label_column], test_size=0.20, shuffle=True)

#Drop the labels from the training set
X_train = X_train.drop(label_column, axis=1)
X_test = X_test.drop(label_column, axis=1)

#Covert the text labels to numbers so the model can understand it
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)

#Train the model
print("Training")
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=50)
classifier.fit(X_train, y_train_encoded)

print("Predicting")
y_pred = classifier.predict(X_test)

#Evaluate model
accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred, average='weighted')
recall = recall_score(y_test_encoded, y_pred, average='weighted')
f1 = f1_score(y_test_encoded, y_pred, average='weighted')

print("=============================================================")

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)

def view_data():
    #Checking the results
    print("Total rows: ", len(df))
    print("X_train shape: ", X_train.shape[0].compute())
    print("X_test shape: ", X_test.shape[0].compute())
    print("y_train shape: ", y_train.shape[0].compute())
    print("y_test shape: ", y_test.shape[0].compute())

    print("Examples\n")
    print("X_train example: ", X_train.head())
    print("X_test example: ", X_test.head())
    print("y_train example: ", y_train.head())
    print("y_test example: ", y_test.head())







