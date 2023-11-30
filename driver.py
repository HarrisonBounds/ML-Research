#import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
#from dask_ml.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import sys
sys.path.append('E:\ML-Research\CTABGAN')
from CTABGAN import ctabgan


def encode_labels(y_train, y_test):
    #Covert the text labels to numbers so the model can understand it
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.fit_transform(y_test)

    class_names = label_encoder.classes_

    print("Class names: ", class_names)

    return y_train_encoded, y_test_encoded, class_names

def generate_data(df, gan_model, num_generated_rows, label_column, path):
    if gan_model == 'ctgan':
        #Convert dataframe into metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        #Generate synthetic data
        print("Generating Synthetic Data...")
        synthesizer = CTGANSynthesizer(metadata, verbose=True)
        synthesizer.fit(df)

        synthetic_data = synthesizer.sample(num_rows=num_generated_rows)
    elif gan_model == 'ctabgan':
        ctabgan_instance = ctabgan.CTABGAN(path, 0.2, [], [], [], ['pslist.nproc', 'pslist.nppid', 'pslist.avg_threads',	'pslist.nprocs64bit', 'pslist.avg_handlers', 'dlllist.ndlls',
                                                    'dlllist.avg_dlls_per_proc',	'handles.nhandles',	'handles.avg_handles_per_proc',	'handles.nport',	'handles.nfile',	
                                                    'handles.nevent',	'handles.ndesktop',	'handles.nkey',	'handles.nthread',	'handles.ndirectory',	'handles.nsemaphore',	
                                                    'handles.ntimer',	'handles.nsection',	'handles.nmutant',	'ldrmodules.not_in_load',	'ldrmodules.not_in_init',	
                                                    'ldrmodules.not_in_mem',	'ldrmodules.not_in_load_avg',	'ldrmodules.not_in_init_avg',	'ldrmodules.not_in_mem_avg',	
                                                    'malfind.ninjections',	'malfind.commitCharge',	'malfind.protection',	'malfind.uniqueInjections',	'psxview.not_in_pslist',	
                                                    'psxview.not_in_eprocess_pool',	'psxview.not_in_ethread_pool',	'psxview.not_in_pspcid_list',	'psxview.not_in_csrss_handles',	
                                                    'psxview.not_in_session',	'psxview.not_in_deskthrd',	'psxview.not_in_pslist_false_avg',	'psxview.not_in_eprocess_pool_false_avg', 'psxview.not_in_ethread_pool_false_avg',	
                                                    'psxview.not_in_pspcid_list_false_avg',	'psxview.not_in_csrss_handles_false_avg	psxview.not_in_session_false_avg',	'psxview.not_in_deskthrd_false_avg',	
                                                    'modules.nmodules',	'svcscan.nservices',	'svcscan.kernel_drivers	svcscan.fs_drivers',	'svcscan.process_services',	'svcscan.shared_process_services',	
                                                    'svcscan.interactive_process_services',	'svcscan.nactive',
                                                    'callbacks.ncallbacks',	'callbacks.nanonymous',	'callbacks.ngeneric'], { "Classification": label_column }, 100)

        ctabgan_instance.fit()
        synthetic_data = ctabgan_instance.generate_samples()

    else:
        synthetic_data = None

    return synthetic_data

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

    print(f'Accuracy: {accuracy}')
    # print(f'Precision: {precision}')
    # print(f'Recall Score: {recall}')
    print(f'f1 Score: {f1}')

def show_confusion_matrix(y_test, y_pred, malware_type, class_names, num_trees):
    confusion_mat = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(16, 12))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {malware_type} Random Forest Classification with {num_trees} Trees')
    plt.show()

def main():
    path = 'E:\\Malware Data Set\\Obfuscated-MalMem2022.csv'
    num_trees = 5000
    malware_type = 'Ransomware'
    gan_model = 'ctabgan'
    num_generated_rows = 3000

    df = pd.read_csv(path)

    #Split data into training and testing
    label_column = 'Category'
    class_column = 'Class'

    #Clip the random string of letters and numbers in the label column
    df['Category'] = df['Category'].str.split('-').str.slice(stop=2).str.join('-')

    #Only classify Spyware, Ransomware, and Trojan
    df = df[df[label_column] != 'Benign']

    #Only classify one class at a time
    df = df[df[label_column].str.contains(malware_type)]

    #Drop the 'Malware' column
    df = df.drop(class_column, axis=1)

    #Encode labels before use
    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])

    #Convert df to csv for ctabgan
    df.to_csv('E:\\Malware Data Set\\clipped.csv')
    clipped_path = 'E:\\Malware Data set\\clipped.csv'
    synthetic_data = generate_data(df, gan_model, 3000, label_column, clipped_path)

    #Add the synthetic data to the data frame
    df = df.append(synthetic_data, ignore_index=True)

    X = df.drop([label_column, class_column], axis=1)
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

    # Clip the "Category" values
    y_train = y_train.str.split('-').str.slice(stop=2).str.join('-')
    y_test = y_test.str.split('-').str.slice(stop=2).str.join('-')

    #Get the encoded labels
    y_train_encoded, y_test_encoded, class_names = encode_labels(y_train, y_test)

    #Model you want to run
    y_pred, y_pred_prob = multi_class_classification(X_train, y_train_encoded, X_test, num_trees)

    #Evaluate the model
    evaluate(y_test_encoded, y_pred, y_pred_prob, class_names)

    #Display confusion matrix
    show_confusion_matrix(y_test_encoded, y_pred, malware_type, class_names, num_trees)

if __name__ == "__main__":
    main()









