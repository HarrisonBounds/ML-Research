import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def encode_labels(y_train, y_test):
    #Covert the text labels to numbers so the model can understand it
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.fit_transform(y_test)

    class_names = label_encoder.classes_

    print("Class names: ", class_names)

    return y_train_encoded, y_test_encoded, class_names


def binary_classification(X_train, y_train, X_test, y_test, num_trees):
    positive_class = "BenignTraffic"

    #Create binary labels where benign is 1 and malicious is 0
    y_train_binary = (y_train == positive_class).astype(int)
    y_test_binary = (y_test == positive_class).astype(int)

    #Training binary classifier
    binary_classifier = RandomForestClassifier(n_estimators=num_trees, criterion='entropy', class_weight='balanced', random_state=50)
    binary_classifier.fit(X_train, y_train_binary)

    # Predict using the binary classifier
    y_pred_binary = binary_classifier.predict(X_test)
    y_pred_prob_binary = binary_classifier.predict_proba(X_test)[:, 1]  # Use the probability for the positive class

    return y_test_binary, y_pred_binary, y_pred_prob_binary


def multi_class_classification(X_train, y_train_encoded, X_test, num_trees):
    print("Training...")
    start_time = time.time()
    classifier = RandomForestClassifier(n_estimators=num_trees, criterion='entropy')
    classifier.fit(X_train, y_train_encoded)
    end_time = time.time()

    print(f'It took {(end_time-start_time)/60} minutes to train')

    print("Predicting...")
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    y_pred_prob = classifier.predict_proba(X_test)
    end_time = time.time()

    print(f'It took {(end_time-start_time)/60} minutes to predict')

    return y_pred, y_pred_prob

def evaluate(y_test, y_pred, y_pred_prob, class_names):
    print("=====================================================\n")
    print("Overall")
    print("=========")

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall Score: {recall}')
    print(f'f1 Score: {f1}\n')

    for i, c in enumerate(class_names):
        class_accuracy = accuracy_score(y_test == i, y_pred == i)
        class_f1 = f1_score(y_test == i, y_pred == i)
        class_recall = recall_score(y_test == i, y_pred == i)
        class_precision = precision_score(y_test == i, y_pred == i)

        print(f'Class: {c}')
        print(f'  Accuracy: {class_accuracy}')
        print(f'  Precision: {class_precision}')
        print(f'  Recall: {class_recall}')
        print(f'  f1: {class_f1}\n')
        

def show_confusion_matrix(y_test, y_pred, class_names, num_trees):
    confusion_mat = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(16, 12))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for IoT Data using Random Forest Classification with {num_trees} Trees')
    plt.show()

def main():
    iot_path = "E:\\CICIoT2023\\*.csv"
    label_column = 'label'
    num_trees = 100
    fraction = 0.1

    df = dd.read_csv(iot_path).sample(frac=fraction)

    df = df.compute()

    X_train, X_test, y_train, y_test = train_test_split(df.drop(label_column, axis=1), df[label_column], test_size=0.20, shuffle=True)

    #Get the encoded labels
    y_train_encoded, y_test_encoded, class_names = encode_labels(y_train, y_test)

    #Model you want to run
    y_pred, y_pred_prob = multi_class_classification(X_train, y_train_encoded, X_test, num_trees)

    #Evaluate the model
    evaluate(y_test_encoded, y_pred, y_pred_prob, class_names)

    #Display confusion matrix
    show_confusion_matrix(y_test_encoded, y_pred, class_names, num_trees)

if __name__ == "__main__":
    main()
