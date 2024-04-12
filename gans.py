import json

with open("config.json", "r") as f:
    data = json.load(f)

if data["gan_model"] == "CTAB":

        num_samples = 0

        for name in class_names:
            print(f'Generating data for {name} class...')
            df_clipped = df_train[df_train[data["label_column"]] == name]
            df_clipped.to_csv("malware_data.csv", index=False)

            ctab = ctabgan.CTABGAN("E:\\ML-Research\\malware_data.csv", test_ratio=0.2, categorical_columns=['Category'], general_columns=['pslist.nproc', 'pslist.nppid', 'pslist.avg_threads','pslist.nprocs64bit', 'pslist.avg_handlers', 'dlllist.ndlls',
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

            synthetic_data = pd.concat([df_train, fake_data], ignore_index=True)

            #Convert dataframe into metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df_train)
        
    #4. Use the training data to generate synthetic data
    elif data["gan_model"] == "CT":
        synthetic_data = pd.DataFrame()

        for name in class_names:
            
            df_clipped = df_train[df_train[data["label_column"]] == name]
            print(f'There are {len(df_clipped)} rows in the {name} class')
            print(f'Generating {data["num_generated_rows"]} samples of synthetic data for {name}...')
            
            #Convert dataframe into metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df_clipped)
            
            synthesizer = CTGANSynthesizer(metadata, verbose=True, epochs=500)
            preprocessed_data = synthesizer.preprocess(df_clipped)
            synthesizer.fit(preprocessed_data)

            fake_data = synthesizer.sample(num_rows=data["num_generated_rows"], batch_size=1000)

            # Concatenate synthetic_data with the previous data frames
            synthetic_data = pd.concat([df_train, fake_data], ignore_index=True)