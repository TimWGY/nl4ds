from IPython.core.display import clear_output
import os
os.system('pip install ipumspy')
clear_output()

from ipumspy import IpumsApiClient, UsaExtract, readers
IPUMS_API_KEY = input("Your IPUMS API Key: (Acquire it at https://account.ipums.org/api_keys if you don't have one yet)")
clear_output()
ipums = IpumsApiClient(IPUMS_API_KEY)

from pathlib import Path
DOWNLOAD_DIR = Path(input('Please specify the folder in which you plan to save the downloaded extract (full path): ').strip())


import pandas as pd
import numpy as np
import re

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 200)
pd.set_option('chained_assignment',None)

def flatten_list(l):
    return [item for sublist in l for item in sublist] 

from google.colab import data_table

sample_table = pd.read_html('https://usa.ipums.org/usa-action/samples/sample_ids')[-1]
sample_table['year'] = sample_table['Sample ID'].apply(lambda x: x[2:-1])
sample_table['detail'] = sample_table.apply(lambda row: row['Description'].split(row['year'],maxsplit=1)[-1].strip().strip(',').strip(), axis=1)
sample_table = sample_table.rename(columns={'Sample ID':'sample_id','Description':'description'})

meta_df = pd.read_feather('/content/nl4ds/ipums_usa_variable_metadata.feather')
preselected_fields = ['SAMPLE', 'SERIAL', 'HHWT', 'GQ', 'PERNUM', 'PERWT', 'VERSIONHIST', 'HISTID']

def select_sample(df):
    # 1. Ask the user for the year
    year = input("Which year: ")
    subset_df = df.query(f"year == '{year}'")
    # If no datasets for that year
    if subset_df.empty:
        print("No samples available for the year", year)
        return select_sample(df)
    # Reset the index of the subset_df
    subset_df = subset_df.reset_index(drop=True)
    # 2. List available details for that year
    print("\nAvailable samples for year", year, "are:")
    for index, row in subset_df.iterrows():
        print(index + 1, "-", row["detail"])
    # 3. Ask user to select a detail
    choice_index = int(input("\nPlease select a sample by entering its index (1-n): ")) - 1
    chosen_dataset = subset_df.iloc[choice_index]
    # 4. Display the year and detail choice
    print(f"You have selected the sample from {chosen_dataset['year']} with details: {chosen_dataset['detail']}")
    print(f"Sample ID: {chosen_dataset['sample_id']}")
    # 5. Return the Sample ID
    return chosen_dataset["sample_id"]

def download_extract(extract):
    ipums.download_extract(extract, download_dir=DOWNLOAD_DIR)
    print(f'The extract has been downloaded. You may find it at {DOWNLOAD_DIR}.')

def ask_wait_extract(extract):
    wait_decision = input('Would you like to wait for the extract to complete and download automatically when it is ready? [y/n]')
    if wait_decision.lower()[0] == 'y':
        print('Waiting for the extract to complete ...')
        ipums.wait_for_extract(extract)
        download_extract(extract)
    else:
        print('You can always use the command "retrieve_extract()" to check on extraction status and download the extract later.')

def submit_extract():
    print('You will build an extract by responding to a series of questions.')
    print('Let us begin with the first sample:')
    sample_id_list = []
    while True:
        sample_id = select_sample(sample_table)
        sample_id_list.append(sample_id)
        sample_id_list = sorted(sample_id_list)
        add_sample_decision = input(f'The current sample(s) selected are: {",".join(sample_id_list)}.\nDo you have another sample to add? [y/n]')
        if add_sample_decision.lower()[0]=='n':
            break

    variables_available_in_samples = set(meta_df.loc[meta_df['availability'].apply(lambda li: len(set(sample_id_list).difference(set(li)))==0 ),'variable_code'].tolist())
    while True:
        variable_error = False
        variables_list = input('Which variables would you like to export for this sample?\nPlease provide variable codes and separate them with comma.\n').replace(' ','').split(',')
        variables_list = [x.upper() for x in variables_list]
        for v in variables_list:
            if v not in variables_available_in_samples:
                force_include_variable_decision = input(f'The variable "{v}" is not available in the all the sample(s) you selected, do you still wish to proceed and add it? [y/n]')
                if force_include_variable_decision.lower()[0] == 'y':
                    print(f'The variable "{v}" has been added, but please note this variable may be showing missing values for the years/samples when it is not available.')
                else:
                    variable_error = True
                    print('Please remember to drop or revise the variable in the list you provide.')
        if not variable_error:
            break

    extract_description = input('Please describe this extract: ')
    
    filter_info_pairs = []
    start_filter_decision = input('Do you want to filter the sample by limiting it to only records with certain values for certain variables? [y/n]')
    if start_filter_decision.lower()[0]=='y':
        print('Please describe the filter(s) you wish to add one by one:')
        while True:

            while True:
                variable_error = False
                variable_code_to_filter_by = input('Which variable to filter by? Please provide one variable code first: ').upper()
                if variable_code_to_filter_by not in variables_available_in_samples:
                    print(f'The variable "{variable_code_to_filter_by}" is not available in the sample(s) you selected, please double check.')
                    variable_error = True
                if not variable_error:
                    break
            
            variable_values_to_filter_with = input(f'''Which values of this variable to include? Please use the codes instead of textual label (e.g. for CITY variable, use 4610,4611 instead of
New York,Brooklyn). Value codes should be comma separated, unless the values are a numeric range, in which case you should use "begin:end"
format (e.g. for AGE variable, use 15:64 to include values between 15 and 64, both ends inclusive). You may look up value codes at
https://usa.ipums.org/usa-action/variables/{variable_code_to_filter_by}#codes_section.\n''').replace(' ','')
            if ':' in variable_values_to_filter_with:
                range_start,range_start = variable_values_to_filter_with.split(':')
                variable_values_to_filter_with = list(map(str, range(range_start,range_start+1)))
            else:
                variable_values_to_filter_with = variable_values_to_filter_with.split(',')
            
            filter_info_pairs.append((variable_code_to_filter_by,variable_values_to_filter_with))
            if variable_code_to_filter_by not in variables_list:
                variables_list.append(variable_code_to_filter_by)
            
            end_filter_decision = input('Filter added successfully. Do you have more filters to add? [y/n]')
            if end_filter_decision.lower()[0]=='n':
                break
    print('Here is the summary of this extract:\n')
    print('------------------------------------------')
    print('Samples:'+'\n\t'+','.join(sample_id_list))
    print('Variables:'+'\n\t'+','.join(variables_list))
    print('Filters:'+'\n\t'+'\n\t& '.join([pair[0]+':'+','.join(pair[1]) for pair in filter_info_pairs]) if len(filter_info_pairs)>0 else 'None')
    print('Description:'+'\n\t'+extract_description)
    print('------------------------------------------\n')
    while True:
        create_decision = input('Please type "yes" to confirm the information above or type "no" to do it again:')
        if create_decision.lower()[0]=='n':
            return submit_extract()
        if create_decision.lower()[0]=='y':
            break
        
    extract = UsaExtract(
        sample_id_list,
        variables_list,
        data_format="csv",
        description=extract_description
    )
    
    for variable_code_to_filter_by, variable_values_to_filter_with in filter_info_pairs:
        extract.select_cases(variable_code_to_filter_by,
                             variable_values_to_filter_with,
                             general=True) # only general codes supported for now since it would be enough foor most use cases
    
    submit_decision = input('\nExtract information confirmed. Are you ready to submit this extract to IPUMS server? [y/n]')
    if submit_decision.lower()[0] == 'y':
        extract_id = ipums.submit_extract(extract)
        print(f"\nExtract submitted with id *** {extract.extract_id} ***.\n\nPlease take note of this id if you plan to retrieve it later.")
        extract_status = ipums.extract_status(collection="usa", extract=extract_id)
        print(f'Your extract is {extract_status}.')
        if extract_status == 'failed':
            print('Something went wrong. Your extract has failed.')
        else:
            ask_wait_extract(extract)
    return extract

def retrieve_extract():
    
    extract_id = input('What is the extract id: ')
    
    # TODO: check if the extract has been downloaded already
    
    try:
        is_expired = ipums.extract_is_expired(collection="usa", extract=extract_id)
    except Exception as e:
        print(str(e))
        return None
    
    if is_expired:
        resubmit_decision = input('This extract has expired. Would you like to get a new extract following the same specification? [y/n]')
        if resubmit_decision.lower()[0] == 'y':
            renewed_extract = ipums.get_extract_by_id(collection="usa", extract_id=extract_id)
            resubmitted_extract = ipums.submit_extract(renewed_extract)
            print(f'A new extract has been submitted. Your new extract id is {resubmitted_extract.extract_id}.')
            ask_wait_extract(extract)
    else:
        extract_status = ipums.extract_status(collection="usa", extract=extract_id)
        print(f'This request is {extract_status}.')
        if extract_status == 'completed':
            download_decision = input('Would you like to download this extract now? [y/n]')
            if download_decision.lower()[0] == 'y':
                extract = ipums.get_extract_by_id(collection="usa", extract_id=extract_id)
                download_extract(extract)

    return None

def load_extract():

    extract_id = input('What is the extract id: ')

    drop_preselected_fields_decision = input('Would you like to drop preselected fields? [y/n]')
    
    decode_decision = input('Would you like to decode the data, i.e. convert codes to human-readable values? [y/n]')

    print('Processing ...')

    df = pd.read_csv(DOWNLOAD_DIR / f'usa_{extract_id.zfill(5)}.csv.gz')
    ddi = readers.read_ipums_ddi(DOWNLOAD_DIR / f'usa_{extract_id.zfill(5)}.xml')
    ddi_df = pd.DataFrame(ddi.data_description)[['id','label','description','notes','codes']]

    if drop_preselected_fields_decision.lower()[0] == 'y':
        df = df.drop([x for x in df.columns if x in preselected_fields], axis=1).reset_index(drop=True)
        ddi_df = ddi_df.query('id not in @preselected_fields').reset_index(drop=True)

    ddi_df['largely_numeric'] = ddi_df['codes'].apply(lambda dic: np.mean([k==str(v) for k,v in dic.items()])>0.9)
    ddi_df['codes'] = ddi_df['codes'].apply(lambda x: {v:k for k,v in x.items()} if isinstance(x,dict) else np.nan)

    print('\nExtract loaded succesfully.')
    print('-'*140+'\n\nSamples:\n\n\t'+'\n\t'.join(ddi.samples_description)+'\n')
    print('Variables:\n')
    print(ddi_df[['id','label']])
    print('\nFilters:\n\t'+'\n\t'.join([' - '.join(tup) for tup in ddi_df[['id','notes']].query('notes != ""').values]))
    print('\n'+'-'*140+'\n')

    if decode_decision.lower()[0] == 'y':
        for _, row in ddi_df.query('codes.notnull() & ~largely_numeric').iterrows():
            df[row['id']] = df[row['id']].map(row['codes']).fillna('Not included in codes')

    return df, ddi_df

print('\nEnvironment initialization ready.\n')