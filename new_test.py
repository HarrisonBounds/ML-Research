import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
import sys
sys.path.append('E:\ML-Research\CTAB_PLUS')
from CTAB_PLUS.model import ctabgan
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality

random_seed = 88
label = "Category"
real_path = "datasets\\Obfuscated-MalMem2022.csv"
fake_path = "datasets\\fake_data.csv"
processed_path = "datasets\\processed.csv"
gan = "CTAB"

df = pd.read_csv(real_path)

#Preprocessing
df['Category'] = df['Category'].str.split('-').str.slice(stop=2).str.join('-')

df = df[df["Category"] != 'Benign']

df = df[df["Category"].str.contains("Trojan")]


# df = df.drop("pslist.nprocs64bit", axis=1)
# df = df.drop("handles.nport", axis=1)
# df = df.drop("psxview.not_in_eprocess_pool", axis=1)
# df = df.drop("psxview.not_in_eprocess_pool_false_avg", axis=1)
# df = df.drop("svcscan.shared_process_services", axis=1)
# df = df.drop("callbacks.ncallbacks", axis=1)
# df = df.drop("callbacks.ngeneric", axis=1)
# # df = df.drop("modules.nmodules", axis=1)
# df = df.drop("svcscan.nservices", axis=1)
# df = df.drop("svcscan.kernel_drivers", axis=1) 
# df = df.drop("svcscan.interactive_process_services", axis=1)
# df = df.drop("callbacks.nanonymous", axis=1)


X = df.drop(label, axis=1)
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

label_encoder = LabelEncoder()

y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.fit_transform(y_test)

np.random.seed(random_seed)
classifier = xgb.XGBClassifier()

classifier.fit(X_train, y_train_enc)

y_pred = classifier.predict(X_test)

class_names = df[label].unique()

print(classification_report(y_test_enc, y_pred, target_names=class_names))

fake_data_comb = pd.DataFrame()

for name in class_names:
    if gan == "CTAB":
        df_clipped = df[df[label] == name]
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_clipped)
        df_clipped.to_csv(processed_path, index=False)

        # synthesizer = ctabgan.CTABGAN(processed_path, test_ratio=0.2, categorical_columns=['Category'], general_columns=['pslist.nproc', 'pslist.nppid', 'pslist.avg_threads','pslist.nprocs64bit', 'pslist.avg_handlers', 'dlllist.ndlls',
        #                                     'dlllist.avg_dlls_per_proc','handles.nhandles',	'handles.avg_handles_per_proc',	'handles.nport','handles.nfile',	
        #                                     'handles.nevent',	'handles.ndesktop',	'handles.nkey',	'handles.nthread',	'handles.ndirectory',	'handles.nsemaphore',	
        #                                     'handles.ntimer',	'handles.nsection',	'handles.nmutant',	'ldrmodules.not_in_load',	'ldrmodules.not_in_init',	
        #                                     'ldrmodules.not_in_mem',	'ldrmodules.not_in_load_avg',	'ldrmodules.not_in_init_avg',	'ldrmodules.not_in_mem_avg',	
        #                                     'malfind.ninjections',	'malfind.commitCharge',	'malfind.protection',	'malfind.uniqueInjections',	'psxview.not_in_pslist', 
        #                                     'psxview.not_in_eprocess_pool',	'psxview.not_in_ethread_pool',	'psxview.not_in_pspcid_list',	'psxview.not_in_csrss_handles', 
        #                                     'psxview.not_in_session',	'psxview.not_in_deskthrd',	'psxview.not_in_pslist_false_avg',	'psxview.not_in_eprocess_pool_false_avg', 
        #                                     'psxview.not_in_ethread_pool_false_avg', 'psxview.not_in_pspcid_list_false_avg',	'psxview.not_in_csrss_handles_false_avg', 	'psxview.not_in_session_false_avg',	'psxview.not_in_deskthrd_false_avg',	
        #                                     'modules.nmodules',	'svcscan.nservices',	'svcscan.kernel_drivers', 'svcscan.fs_drivers',	'svcscan.process_services',	'svcscan.shared_process_services',	
        #                                     'svcscan.interactive_process_services',	'svcscan.nactive',
        #                                     'callbacks.ncallbacks',	'callbacks.nanonymous',	'callbacks.ngeneric'], 
        #                                     integer_columns=[], problem_type={"Classification": "Category"})

        synthesizer = ctabgan.CTABGAN(raw_csv_path=processed_path, test_ratio=0.2, categorical_columns=['Category'], log_columns=[], mixed_columns=[],
                                    integer_columns=['pslist.nproc', 'pslist.nppid','pslist.nprocs64bit', 'dlllist.ndlls','handles.nhandles', 'handles.nport','handles.nfile',	
                                    'handles.nevent',	'handles.ndesktop',	'handles.nkey',	'handles.nthread',	'handles.ndirectory',	'handles.nsemaphore',	
                                    'handles.ntimer',	'handles.nsection',	'handles.nmutant',	'ldrmodules.not_in_load',	'ldrmodules.not_in_init',	
                                    'ldrmodules.not_in_mem', 'malfind.ninjections',	'malfind.commitCharge',	'malfind.protection',	'malfind.uniqueInjections',	'psxview.not_in_pslist', 
                                    'psxview.not_in_eprocess_pool',	'psxview.not_in_ethread_pool',	'psxview.not_in_pspcid_list',	'psxview.not_in_csrss_handles', 
                                    'psxview.not_in_session',	'psxview.not_in_deskthrd','modules.nmodules',	'svcscan.nservices',	'svcscan.kernel_drivers',	'svcscan.fs_drivers',	'svcscan.process_services',	'svcscan.shared_process_services',	
                                    'svcscan.interactive_process_services',	'svcscan.nactive',
                                    'callbacks.ncallbacks',	'callbacks.nanonymous',	'callbacks.ngeneric'], problem_type={"Classification": "Category"})
                
                

        synthesizer.fit()

        fake_data = synthesizer.generate_samples(len(df_clipped))

        fake_data_comb = pd.concat([fake_data_comb, fake_data], ignore_index=True)

    if gan == "CT":
        df_clipped = df[df[label] == name]
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_clipped)

        synthesizer = CTGANSynthesizer(metadata=metadata, verbose=True)

        synthesizer.fit(df_clipped)

        fake_data = synthesizer.sample(len(df_clipped))

        fake_data_comb = pd.concat([fake_data_comb, fake_data], ignore_index=True)


fake_data_comb.to_csv(fake_path, index=False)

run_diagnostic(
        real_data=df,
        synthetic_data=fake_data_comb,
        metadata=metadata,
        verbose=True)
    
evaluate_quality(
    real_data=df,
    synthetic_data=fake_data_comb,
    metadata=metadata,
    verbose=True)

X_train_syn = fake_data_comb.drop(label, axis=1)
y_train_syn = fake_data_comb[label]

y_train_syn_enc = label_encoder.fit_transform(y_train_syn)

np.random.seed(random_seed)
classifier_syn = xgb.XGBClassifier()

classifier_syn.fit(X_train_syn, y_train_syn_enc)

y_pred_syn = classifier_syn.predict(X_test)

print(classification_report(y_test_enc, y_pred_syn, target_names=class_names))
