import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality

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
    #classifier = RandomForestClassifier(n_estimators=num_trees, criterion='entropy')
    classifier = xgb.XGBClassifier()
    classifier.fit(X_train, y_train_encoded)
    end_time = time.time()

    print(f'It took {(end_time-start_time)/60} minutes to train')

    print("Predicting...")
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    end_time = time.time()

    print(f'It took {(end_time-start_time)/60} minutes to predict')

    return y_pred

def evaluate(y_test, y_pred, class_names):
   #Evaluate the model
    print(classification_report(y_test, y_pred, target_names=class_names))
        

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
    fraction = 0.01
    random_seed = 22

    df = dd.read_csv(iot_path).sample(frac=fraction)

    df = df.compute()

    df[label_column] = df[label_column].apply(lambda x: 'malicious' if x != 'BenignTraffic' else x)

    X_train, X_test, y_train, y_test = train_test_split(df.drop(label_column, axis=1), df[label_column], test_size=0.20, shuffle=True, random_state=random_seed)

    #Get the encoded labels
    y_train_encoded, y_test_encoded, class_names = encode_labels(y_train, y_test)

    #Model you want to run
    y_pred = multi_class_classification(X_train, y_train_encoded, X_test, num_trees)

    #Evaluate the model
    evaluate(y_test_encoded, y_pred, class_names)

    synthetic_data = pd.DataFrame()

    print("Using CT GAN")
    # for name in class_names:
    #     df_clipped = df[df[label_column] == name]
        #print(f'Generating {data["num_generated_rows"]} samples of synthetic data for {name}...')
        
    #Convert dataframe into metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    synthesizer = CTGANSynthesizer(metadata, verbose=True, epochs=200)
    synthesizer.fit(df)

    fake_data = synthesizer.sample(num_rows=len(df))                                                                                                                                        

    # Concatenate synthetic_data with the previous data frames
    #synthetic_data = pd.concat([synthetic_data, fake_data], ignore_index=True)

    synthetic_data = fake_data

    X_train_syn = synthetic_data.drop(label_column, axis=1)
    y_train_syn = synthetic_data[label_column]

    le = LabelEncoder()

    y_train_syn_enc = le.fit_transform(y_train_syn)

    
    run_diagnostic(
            real_data=df,
            synthetic_data=fake_data,
            metadata=metadata,
            verbose=True)
        
    evaluate_quality(
        real_data=df,
        synthetic_data=fake_data,
        metadata=metadata,
        verbose=True)


    y_pred_syn = multi_class_classification(X_train_syn, y_train_syn_enc, X_test, num_trees)

    evaluate(y_test_encoded, y_pred_syn, class_names)

    #Display confusion matrix
    #show_confusion_matrix(y_test_encoded, y_pred, class_names, num_trees)

if __name__ == "__main__":
    main()
