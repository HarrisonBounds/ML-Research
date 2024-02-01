from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import time
from CTABGAN import ctabgan


def generate_data(df, gan_model, num_generated_rows, label_column, path):
    if gan_model == 'ctgan':
        #Convert dataframe into metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        #Generate synthetic data
        print("Generating Synthetic Data...")
        start_time = time.time()
        synthesizer = CTGANSynthesizer(metadata, epochs=1000, verbose=True)
        synthesizer.fit(df)
        end_time = time.time()

        print(f'It took {(end_time-start_time)/60} minutes to generate')


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
                                                    'psxview.not_in_pspcid_list_false_avg',	'psxview.not_in_csrss_handles_false_avg', 'psxview.not_in_session_false_avg',	'psxview.not_in_deskthrd_false_avg',	
                                                    'modules.nmodules',	'svcscan.nservices',	'svcscan.kernel_drivers	svcscan.fs_drivers',	'svcscan.process_services',	'svcscan.shared_process_services',	
                                                    'svcscan.interactive_process_services',	'svcscan.nactive',
                                                    'callbacks.ncallbacks',	'callbacks.nanonymous',	'callbacks.ngeneric'], { "Classification": label_column }, 100)

        ctabgan_instance.fit()
        synthetic_data = ctabgan_instance.generate_samples()

    else:
        synthetic_data = None

    return synthetic_data
