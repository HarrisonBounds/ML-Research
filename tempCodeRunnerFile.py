import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot
from table_evaluator import TableEvaluator
import numpy as np
import json
import sys
from dp_cgans import DP_CGAN
sys.path.append('E:\ML-Research\CasTGAN')
from CasTGAN.Model import CasTGAN
sys.path.append('E:\ML-Research\CTAB')
from CTAB.model import ctabgan
sys.path.append('E:\ML-Research\TGAN')
from TGAN.tgan.model import TGANModel
from sklearn.feature_selection import RFECV
import random


def encode_labels(y_train, y_test):
    #Covert the text labels to numbers so the model can understand it
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.fit_transform(y_test)

    return y_train_encoded, y_test_encoded


def multi_class_classification(X_train, y_train_encoded, X_test, num_trees, classifier, random_seed):
    print(f"Training using {classifier} classification...")
    
    #Classification selection
    if classifier == "RF":
        classifier = RandomForestClassifier(n_estimators=num_trees, criterion='gini', min_samples_leaf=2, max_depth=None, verbose=True, random_state=random_seed)

    elif classifier == "XGB":
        np.random.seed(random_seed)
        classifier = xgb.XGBClassifier()
    
    classifier.fit(X_train, y_train_encoded)

    print("Predicting...")
    y_pred = classifier.predict(X_test)
        
    return y_pred
    
def save_confusion_matrix(y_test, y_pred, class_names, data, classifier, original, df_train):
    cf = confusion_matrix(y_test, y_pred)
    total_class_samples = np.sum(cf, axis=1)
    cf_percentage = (cf / total_class_samples).T * 100

    plt.figure(figsize=(14, 10))
    sns.heatmap(cf_percentage, annot=True, fmt='0.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    for i, name in enumerate(class_names):
        count = (df_train[data["label_column"]] == name).sum()
        plt.text(i + 0.5, -0.5, f'num_instances: {count}', ha='center', va='center', color='red')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if original:
        plt.title(f'{data["mal_type"]} Confusion Matrix using {classifier} classification with original data')
        plt.savefig(f'evaluation\\{data["mal_type"]}\\{classifier}_original.jpg')
    else:
        plt.title(f'{data["mal_type"]} Confusion Matrix using {classifier} classification with {data["mal_type_class"]} synthetic data') 
        plt.savefig(f'evaluation\\{data["mal_type"]}\\{data["mal_type_class"]}_{data["num_generated_rows"]}_{classifier}_synthetic.jpg')

def test_synthethic(df_train, synthetic_data, metadata):
    run_diagnostic(
        real_data=df_train,
        synthetic_data=synthetic_data,
        metadata=metadata,
        verbose=True)
    
    evaluate_quality(
        real_data=df_train,
        synthetic_data=synthetic_data,
        metadata=metadata,
        verbose=True)
    
def prepare_synthetic(synthetic_data, data):
    X_train_combined = pd.DataFrame()
    y_train_combined = pd.DataFrame()

    X_train_combined = synthetic_data.drop(data["label_column"], axis=1)
    y_train_combined = synthetic_data[data["label_column"]]

    label_encoder = LabelEncoder()
    y_train_combined_encoded = label_encoder.fit_transform(y_train_combined)

    return X_train_combined, y_train_combined_encoded

# Function to identify continuous columns
def get_continuous_columns(df):
    continuous_cols = []
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):  # Check if column contains numerical data
            continuous_cols.append(col)
    return continuous_cols


def main():
    #Load config file
    with open("config.json", "r") as f:
        data = json.load(f)
    
    original = True
    random_seed = 11
    classifiers = data["classifiers"]
    y_pred = []
    y_pred_combined = []

    df = pd.read_csv(data["mal_path"])

    #Clip the random string of letters and numbers in the label column
    df['Category'] = df['Category'].str.split('-').str.slice(stop=1).str.join('-')

    #Only classify malicious data
    if data["remove_benign"]:
        df = df[df[data["label_column"]] != 'Benign']

    # if data["mal_type"]:
    #     df = df[df[data["label_column"]].str.contains(data["mal_type"])]

    #1. Split the data
    df_train, df_test = train_test_split(df, test_size=0.20, stratify=df[data["label_column"]], random_state=random_seed)

    X_train = df_train.drop(data["label_column"], axis=1)
    y_train = df_train[data["label_column"]]
    X_test = df_test.drop(data["label_column"], axis=1)
    y_test = df_test[data["label_column"]]

    #Get class_names
    class_names = df_train[data["label_column"]].unique()

    #Get the encoded labels
    y_train_encoded, y_test_encoded = encode_labels(y_train, y_test)

    #2. Run the Baseline classifier and evaluatet the model's performance
    for i, classifier in enumerate(classifiers):
        y_pred.append(multi_class_classification(X_train, y_train_encoded, X_test, data["num_trees"], classifier, random_seed))
 
        print(f'{classifier}')
        print(classification_report(y_test_encoded, y_pred[i], target_names=class_names)) 

        #Display confusion matrix
        save_confusion_matrix(y_test_encoded, y_pred[i], class_names, data, classifiers[i], original, df_train)

    #Move to data generation
    original = False



    #4. Use the training data to generate synthetic data
    synthetic_data = pd.DataFrame()
    
    for gan in data["gan_models"]:
        
        if gan == "CTAB":
            print("Using CTAB GAN")
            for name in class_names:
                count = (df_train[data["label_column"]] == name).sum()
                print(f'{name} has {count} instances of data.')

                print(f'Generating {count} for {name} class...')
                df_clipped = df_train[df_train[data["label_column"]] == name]
            
                #Convert dataframe into metadata
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(df_clipped)
                
                df_clipped.to_csv("malware_data.csv", index=False)

                ctab = ctabgan.CTABGAN("E:\\ML-Research\\malware_data.csv", test_ratio=0.2, categorical_columns=[], general_columns=['pslist.nproc', 'pslist.nppid', 'pslist.avg_threads','pslist.nprocs64bit', 'pslist.avg_handlers', 'dlllist.ndlls',
                                                            'dlllist.avg_dlls_per_proc','handles.nhandles',	'handles.avg_handles_per_proc',	'handles.nport','handles.nfile',	
                                                            'handles.nevent',	'handles.ndesktop',	'handles.nkey',	'handles.nthread',	'handles.ndirectory',	'handles.nsemaphore',	
                                                            'handles.ntimer',	'handles.nsection',	'handles.nmutant',	'ldrmodules.not_in_load',	'ldrmodules.not_in_init',	
                                                            'ldrmodules.not_in_mem',	'ldrmodules.not_in_load_avg',	'ldrmodules.not_in_init_avg',	'ldrmodules.not_in_mem_avg',	
                                                            'malfind.ninjections',	'malfind.commitCharge',	'malfind.protection',	'malfind.uniqueInjections',	'psxview.not_in_pslist', 
                                                            'psxview.not_in_eprocess_pool',	'psxview.not_in_ethread_pool',	'psxview.not_in_pspcid_list',	'psxview.not_in_csrss_handles', 
                                                            'psxview.not_in_session',	'psxview.not_in_deskthrd',	'psxview.not_in_pslist_false_avg',	'psxview.not_in_eprocess_pool_false_avg', 
                                                            'psxview.not_in_ethread_pool_false_avg', 'psxview.not_in_pspcid_list_false_avg',	'psxview.not_in_csrss_handles_false_avg	psxview.not_in_session_false_avg',	'psxview.not_in_deskthrd_false_avg',	
                                                            'modules.nmodules',	'svcscan.nservices',	'svcscan.kernel_drivers	svcscan.fs_drivers',	'svcscan.process_services',	'svcscan.shared_process_services',	
                                                            'svcscan.interactive_process_services',	'svcscan.nactive',
                                                            'callbacks.ncallbacks',	'callbacks.nanonymous',	'callbacks.ngeneric'], 
                                                            integer_columns=[], problem_type={"Classification": "Category"})


            
                ctab.fit()

                fake_data = ctab.generate_samples(data["num_generated_rows"]) 

                synthetic_data = pd.concat([synthetic_data, fake_data], ignore_index=True)
            
        
        elif gan == "CT":
            print("Using CT GAN")
            for name in class_names:
                count = (df_train[data["label_column"]] == name).sum()
                print(f'{name} has {count} instances of data.')

                df_clipped = df_train[df_train[data["label_column"]] == name]
                print(f'Generating {data["num_generated_rows"]} samples of synthetic data for {name}...')
                
                #Convert dataframe into metadata
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(df_clipped)
                
                synthesizer = CTGANSynthesizer(metadata, verbose=True, epochs=20, batch_size=1000)
                preprocessed_data = synthesizer.preprocess(df_clipped)
                synthesizer.fit(preprocessed_data)

                fake_data = synthesizer.sample(num_rows=data['num_generated_rows'], batch_size=1000)

                # Concatenate synthetic_data with the previous data frames
                synthetic_data = pd.concat([synthetic_data, fake_data], ignore_index=True)

        elif gan == "DP-CGAN":
            print("Using DP-CGAN")
            for name in class_names:
                count = (df_train[data["label_column"]] == name).sum()
                print(f'{name} has {count} instances of data.')

                df_clipped = df_train[df_train[data["label_column"]] == name]
                print(f'Generating {count} samples of synthetic data for {name}...')

                model = DP_CGAN(
                    epochs=200,
                    batch_size=1000,
                    verbose=True,
                )

                model.fit(df_clipped)

                fake_data = model.sample(count)

                synthetic_data = pd.concat([df_clipped, fake_data], ignore_index=True)

        elif gan == "TGAN":
            print("Using TGAN")

            
            for name in class_names:
                count = (df_train[data["label_column"]] == name).sum()
                print(f'{name}_synthethic has {count} instances of data.')

                df_clipped = df_train[df_train[data["label_column"]] == name]
                print(f'Generating {count} samples of synthetic data for {name}...')

                continuous_columns = get_continuous_columns(df_clipped)

                model = TGANModel(continuous_columns)

                model.fit(df_clipped)

                fake_data = model.sample(data["num_generated_rows"])

                synthetic_data = pd.concat([df_clipped, fake_data], ignore_index=True)

        elif gan == "CasTGAN":
            print("Using CasTGAN")
            for name in class_names:
                count = (df_train[data["label_column"]] == name).sum()
                print(f'{name}_synthethic has {count} instances of data.')

                df_clipped = df_train[df_train[data["label_column"]] == name]
                print(f'Generating {count} samples of synthetic data for {name}...')

                #df_clipped.to_csv("castgan_malware_data.csv", index=False)

                model = CasTGAN(batch_size=1000, verbose=True)

                model.fit(raw_dataset=df_clipped, categorical_columns=["Category"], datetime_columns=[], epochs=20)

                fake_data = model.sample(data["num_generated_rows"])

                synthetic_data = pd.concat([df_clipped, fake_data], ignore_index=True)
                

        synthetic_class_names = synthetic_data[data["label_column"]].unique()

        for name in synthetic_class_names:
            synthetic_count = (synthetic_data[data["label_column"]] == name).sum()
            print(f'{name} now has {synthetic_count} instances of synthetic data.')

        synthetic_data.to_csv("syn_data.csv", index=False)

        #5. Test how similar the synthetic data is to the real data
        test_synthethic(df_train, synthetic_data, metadata)

        #6. Prepare synthetic data for classification
        X_train_combined, y_train_combined_encoded = prepare_synthetic(synthetic_data, data)

        #7. Train classifier with the combined data amd evaluate model's performace
        for i, classifier in enumerate(classifiers):
            y_pred_combined.append(multi_class_classification(X_train_combined, y_train_combined_encoded, X_test, data["num_trees"], classifier, random_seed))        
            print(classification_report(y_test_encoded, y_pred_combined[i], target_names=class_names))
            save_confusion_matrix(y_test_encoded, y_pred_combined[i], synthetic_class_names, data, classifiers[i], original, df_train)

if __name__ == "__main__":
    main()