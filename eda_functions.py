from IPython.display import clear_output

# ------------------------------------Import Libraries----------------------------------------

import os
from glob import glob
import pickle
import inspect
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

import re
from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import pytz
from datetime import datetime

# import statsmodels.api as sm
# import patsy
# import sklearn

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

global default_dpi
default_dpi = 120


# -------------------------------------Data Selection----------------------------------------------

# def load_census(which_year, fields = None, version_code = 'data_2021_may'):
#   '''Provide the census year you will to load (choose from any year between 1850 - 1940, except for 1890)'''
#   if str(which_year) == '1890':
#     print('Unfortunately, census records for NYC in 1890 are not available because a fire destroyed the collection. Try other years between 1850 and 1940.')
#     return None
#   filepath = '/content/drive/My Drive/'+version_code+'/census_' + str(which_year) + '.csv'
#   print('\nLoading data for census year '+str(which_year)+'...')

#   try:
#     if fields is not None:
#         data = pd.read_csv(filepath, nrows = 0)
#         available_fields = data.columns.tolist()
#         valid_fields = []
#         for field in fields:
#             if field not in available_fields:
#                 print(field+' is not in this dataset.')
#             else:
#                 valid_fields.append(field)
#         data = pd.read_csv(filepath, usecols = valid_fields)
#     else:
#         data = pd.read_csv(filepath)


#     print('\nThere are '+str(len(data))+' entries.\n')
#     print('Available columns:\n\n')
#     print_list(data.columns.tolist())
#     print('\n\n')
#     return data

#   except FileNotFoundError as e:
#     print('File Not Found! Please contact Tim at gw923@nyu.edu for acccess to certain datasets.')
#   except NameError as e:
#     print('Function not defined yet! Please check if you have run the first cell in this notebook.')

def load_census(data_file_name = 'ipums_full_count_nyc_census_decoded_1_percent_sample_20210801.csv', data_folder_path = '/content/drive/Shareddrives/Humanities Research Lab - Shanghai/colab_data/', fields = None):
  '''Provide the census year you will to load (choose from any year between 1850 - 1940, except for 1890)'''

  filepath = data_folder_path + data_file_name

  print('\nLoading data, this may take a while for full count dataset ...')

  try:
    if fields is not None:
        data = pd.read_csv(filepath, nrows = 0)
        available_fields = data.columns.tolist()
        valid_fields = []
        for field in fields:
            if field not in available_fields:
                print('The field you requested "'+field+'"" is not in this dataset.')
            else:
                valid_fields.append(field)
        data = pd.read_csv(filepath, usecols = valid_fields)
    else:
        data = pd.read_csv(filepath)

    print('\nThere are '+str(len(data))+' entries.\n')
    print('Available columns:\n\n')
    print_list(data.columns.tolist())
    print('\n\n')
    return data

  except FileNotFoundError as e:
    print('File Not Found! Please contact Tim at gw923@nyu.edu for acccess to certain datasets.')
  except NameError as e:
    print('Function not defined yet! Please check if you have run the first cell in this notebook.')

def see_value_options(data, col, top_k = None):
  val_cnt = data[col].value_counts()
  val_cnt = val_cnt/len(data)
  val_cnt_df = val_cnt.rename('Proportion').reset_index().rename(columns={'index':'Value'})
  val_cnt_df['Proportion'] = np.round(val_cnt_df['Proportion']*100, 2)
  if top_k:
    val_cnt_df = val_cnt_df[:top_k]
  return val_cnt_df

def check_parenthesis_and_replace_comma_within_parenthesis(string):

  output = []
  letters = list(string.strip())
  waiting_to_close = False
  while len(letters) > 0:
    head = letters.pop(0)
    if head == '[':
      if waiting_to_close == False:
        waiting_to_close = True
      else:
        return False
    elif head == ']':
      if waiting_to_close == True:
        waiting_to_close = False
      else:
        return False

    if (head == ',' or head == ':') and waiting_to_close == True:
      output.append('|')
    else:
      output.append(head)

  if waiting_to_close == False:
    return ''.join(output)
  else:
    return False

def replace_first_occurence_of_sign(string, sign, replacement):

  first_position_of_sign = string.index(sign)
  new_string = string[:first_position_of_sign] + replacement + string[first_position_of_sign + len(sign):]
  return new_string

def check_for_criteria_type(string, data, sign, alternative_sign, valid_cols):

  sign = ' ' + sign.strip() + ' '
  alternative_sign = ' ' + alternative_sign.strip() + ' '
  left_side_of_sign_or_alt_sign_is_valid_col = ((string.split(sign, maxsplit=1)[0].strip() in valid_cols) or (string.split(alternative_sign, maxsplit=1)[0].strip() in valid_cols))
  if (sign in string or alternative_sign in string) and left_side_of_sign_or_alt_sign_is_valid_col:
    string = replace_first_occurence_of_sign(string, sign, alternative_sign)
    col = string.split(alternative_sign.strip())[0].strip()
    value = string.split(alternative_sign.strip())[1].strip()
    if sign == ' is in ' or sign == ' is not in ':
      assert(value[0] == '[' and value[-1] == ']')
      value = [option.strip().strip('"').strip("'") for option in value[1:-1].split('|')] # handle the is-in strings like ['a', 'b', 'c'] or ["d", "e"]
    else:
      value = float(value) if value.isnumeric() else value
    return build_criteria(col, value, data, sign=sign)
  else:
    return None

def build_criteria_from_string(string, data):

  valid_cols = data.columns.tolist()

  criteria = check_for_criteria_type(string, data, ' is not in ', ' is not in ', valid_cols)
  if isinstance(criteria, pd.Series):
    return criteria

  criteria = check_for_criteria_type(string, data, ' is not ', ' != ', valid_cols)
  if isinstance(criteria, pd.Series):
    return criteria

  criteria = check_for_criteria_type(string, data, ' is in ', ' is in ', valid_cols)
  if isinstance(criteria, pd.Series):
    return criteria

  criteria = check_for_criteria_type(string, data, ' is ', ' = ', valid_cols)
  if isinstance(criteria, pd.Series):
    return criteria


def build_criteria(col, value, data, sign=' is '):

  if sign == ' is ' or sign == ' is not ':
    if value == 'MISSING':
      output = data[col].isnull()
    else:
      output = data[col] == value

  elif sign == ' is in ' or sign == ' is not in ':

    if pd.api.types.is_numeric_dtype(data[col].dtype) and len(value) == 2:
      output = (data[col] >= float(value[0])) & (data[col] < float(value[1]))
    else:
      output = data[col].isin(value)

  if ' not ' in sign:
    output = ~output

  return output

def get_multiple_criteria(string, data):
  if ' is in ' or 'is not in' in string:
    string = check_parenthesis_and_replace_comma_within_parenthesis(string)
  multiple_criteria = [c.strip() for c in string.split(',') if c.strip()!='']
  multiple_criteria = [c + ' = 1' if (' = ' not in c) and (' is ' not in c) and (c in data.columns) else c for c in multiple_criteria]
  multiple_criteria_filters = [build_criteria_from_string(c, data) for c in multiple_criteria]
  combined_filter = pd.Series([True] * len(data))
  for filter in multiple_criteria_filters:
    combined_filter = combined_filter & filter
  return combined_filter


def select(criteria_string, data):
  '''Provide a comma separated criteria_string, and specify which dataframe (df) to select from'''
  data = data.reset_index(drop=True).copy()
  criteria_filter = get_multiple_criteria(criteria_string, data)
  return data[criteria_filter].copy()

def select_data(criteria_string, data):
  '''Older name for the select fucntion, plan to retire the old name gradually, keeping this to support older codes'''
  return select(criteria_string, data)

# ----------------------------------- Value Selection Based on Keywords ------------------------------------

def get_values_that_covers_threshold_percentage(col, data, thres = 0.9, order = 'most_first' ):

  ser = data.copy()[col].value_counts()
  cumsum_ser = (ser/ser.sum()).cumsum()
  top_values = cumsum_ser[cumsum_ser<=thres].index.tolist()
  if order == 'most_first':
    return top_values
  elif order == 'alphabetical':
    return sorted(top_values)
  else:
    return None


def check_contain(value, contain_list):
  contain_or_not = True
  for keyword in contain_list:
    if keyword not in value:
      contain_or_not = False
      break
  return contain_or_not

def check_not_contain(value, not_contain_list):
  not_contain_or_not = True
  for keyword in not_contain_list:
    if keyword in value:
      not_contain_or_not = False
      break
  return not_contain_or_not

def filter_by_keyword(input_list, contain = '', not_contain = '', case_important = False):

  output_list = []

  if not case_important:
    contain = contain.lower()
    not_contain = not_contain.lower()

  contain = [x.strip() for x in contain.split(',') if x!='']
  not_contain = [x.strip() for x in not_contain.split(',') if x!='']

  for value in input_list:
    if not case_important:
      processed_value = value.lower()
    else:
      processed_value = value

    if check_contain(processed_value, contain) and check_not_contain(processed_value, not_contain):
      output_list.append(value)

  return output_list

def filter_values(data, col, contain = '', not_contain = '' , coverage = 'auto', case_important = False, return_list = True, order = 'most_first'):
  if coverage == 'auto':
    if data[col].nunique()>100:
      input_list = get_values_that_covers_threshold_percentage(col, data, order = order)
    else:
      input_list = get_values_that_covers_threshold_percentage(col, data, thres = 1.0, order = order)
  elif coverage == 'full':
    input_list = get_values_that_covers_threshold_percentage(col, data, thres = 1.0, order = order)

  input_list = [item for item in input_list if isinstance(item,str)]

  output_list = filter_by_keyword(input_list, contain = contain, not_contain = not_contain, case_important = case_important)

  if return_list:
    return output_list
  else:
    print_list(output_list)

def show_filter_values(data, col, contain = '', not_contain = '' , coverage = 'auto', case_important = False, return_list = False, order = 'most_first'):
  return filter_values(data = data, col = col, contain = contain, not_contain = not_contain , coverage = coverage, case_important = case_important, return_list = return_list, order = order)

def change_values(data, orig_col, change_from, change_to, new_col = ''):
  if change_to == 'MISSING':
    change_to = np.nan
  if isinstance(change_from,list):
    pass
  else:
    change_from = [x.strip() for x in change_from.replace('\n','').split(',')  if x.strip()!='']
  if new_col == '':
    print(f'The column "{orig_col}" is changed.')
    new_col = orig_col
  else:
    print(f'A new column "{new_col}" is created to reflect the change.')
  data[new_col] = data[orig_col].apply(lambda x: change_to if x in change_from else x)

def filter_and_change_values(data, orig_col, contain = '', not_contain = '', change_to = None, new_col = '', coverage = 'auto', case_important = False):
  if change_to == None:
    print('Parameter "change_to" is not specificed.')
    return
  change_from = filter_values(data, orig_col,  contain = contain, not_contain = not_contain, coverage = coverage, case_important = case_important)
  print('Changing from:',)
  print_list(change_from,indent = 2)
  print('\nTo:\n  '+change_to+'\n')
  change_values(data, orig_col, change_from, change_to)
  return

# -------------------------------------Smart Data Description-------------------------------------------

def describe(col, data, top_k=-1, thres=90, return_full=False, plot_top_k=-1, plot_type='', bins=-1, show_graph = True, year = None):

  year_info = (" - Year "+str(year)) if year is not None else ""

  if data[col].isnull().mean() > 0:
    print(f"Notice: {np.round(data[col].isnull().mean()*100,3)}% of the entries have no records for this field.\n")

  data_numeric_columns = data.dtypes[data.dtypes.apply(lambda x: np.issubdtype(x, np.number))].index.tolist()

  if col in data_numeric_columns:
    if bins == -1:
      print(f'Change the default width of histogram bars by setting "bins = <a number>".\n')
      bins = 50
    plt.figure(figsize=(9, 6), dpi=default_dpi)
    plt.hist(data[col].dropna(), bins=bins)
    plt.title(f"Distribution of the {col}" + year_info)
    basic_stats = data[col].dropna().describe().reset_index()
    basic_stats.columns = ['Field', 'Value']
    basic_stats.Field = ['Total Count', 'Mean', 'Standard Deviation', 'Minimum', 'Value at 25% Percentile', 'Median (50% Percentile)', 'Value at 75% Percentile', 'Maximum']
    basic_stats.loc[basic_stats.Field.isin(['Mean', 'Standard Deviation']), 'Value'] = basic_stats.loc[basic_stats.Field.isin(['Mean', 'Standard Deviation']), 'Value'].apply(lambda x: np.round(x, 2))
    basic_stats.loc[~basic_stats.Field.isin(['Mean', 'Standard Deviation']), 'Value'] = basic_stats.loc[~basic_stats.Field.isin(['Mean', 'Standard Deviation']), 'Value'].apply(int).apply(str)
    return basic_stats

  ser = data[col].value_counts()
  ser.name = 'Absolute Number'

  percentage_ser = np.round(ser / len(data) * 100, 2)
  percentage_ser.name = 'Proportion in Data (%)'

  cumsum_percentage_ser = percentage_ser.cumsum()
  cumsum_percentage_ser.name = 'Cumulative Proportion (%)'

  full_value_counts_df = pd.concat([ser, percentage_ser, cumsum_percentage_ser], axis=1)

  if plot_top_k > top_k:
    top_k = plot_top_k

  if top_k == -1:
    top_k = sum(cumsum_percentage_ser <= thres) + 1
    top_k = max(5, top_k)
    top_k = min(20, top_k)

  value_counts_df = full_value_counts_df if return_full else full_value_counts_df[:top_k]

  if top_k < len(full_value_counts_df) and not return_full:
    print(f'{len(full_value_counts_df)-top_k} more rows are available, add "return_full = True" if you want to see all.\n')

  if show_graph:
    plot_top_k = 10 if plot_top_k == -1 else plot_top_k
    graph_df = value_counts_df['Proportion in Data (%)'][:plot_top_k].copy()

    if plot_type == '':
      plot_type = 'bar' if graph_df.sum() < thres else 'pie'

    if plot_type == 'pie':

      fig, ax = plt.subplots(figsize=(9, 6), dpi=default_dpi, subplot_kw=dict(aspect="equal"))

      graph_df = graph_df.sort_index() # lock the order of indices so color coding is consistent across multiple pie chart of the same subject
      values = graph_df.values.tolist()
      names = graph_df.index.tolist()

      def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%".format(pct, absolute)

      wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values), textprops=dict(color="w"))

      for w in wedges:
        w.set_edgecolor('white')

      ax.legend(wedges, names,
                title="Categories",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.8, 1))

      plt.setp(autotexts, size=12, weight="bold")

      ax.set_title(f"Relative Proportion of Top {len(graph_df)} {col}" if len(graph_df) < len(full_value_counts_df) else f"Proportion of {col}" + year_info)

    if plot_type == 'bar':
      plt.figure(figsize=(9, 6), dpi=default_dpi)
      graph_df.plot(kind='bar')
      plt.title(f"Barplot of the Top {len(graph_df)} {col} - (y axis shows percentage)"+ year_info)

    print()

  return value_counts_df


# # -----------------------------Correlation & Regression------------------------------------------

def show_corr(cols, data):

  if isinstance(cols, str):
    cols = [col.strip() for col in cols.strip().split(',')]
  try:
    corr_df = data[cols].copy().corr()
  except KeyError as e:
    print('Variable "' + re.findall(r"\[\'(.*?)\'\]", str(e))[0] + '" not found, check your spelling please.')
    return

  cmap = sns.diverging_palette(10, 130, as_cmap=True)  # red green

  corr = corr_df.values
  np.fill_diagonal(corr, np.nan)
  corr = np.triu(corr, k=1)
  corr[corr == 0] = np.nan
  labels = corr_df.columns

  plt.figure(figsize=(5, 4), dpi=default_dpi)
  sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, xticklabels=labels, yticklabels=labels)


# def run_regression(design, data=regression_df):
#   variables = design.replace('~', ' ').replace('+', ' ').split()
#   selected_data = data[variables].copy()
#   scaled_data = pd.DataFrame(sklearn.preprocessing.RobustScaler().fit_transform(X=selected_data), columns=selected_data.columns)
#   y, X = patsy.dmatrices(design, data=scaled_data, return_type='dataframe')   # Split data columns
#   mod = sm.OLS(y, X)
#   res = mod.fit()
#   print(res.summary())


# -------------------- Analytics utilities ----------------------

def proportion(small_data, big_data, rounding = 3):
  return round(len(small_data)/len(big_data), rounding)

# -------------------- Programming utilities ----------------------

def flatten_list(l):
    return [item for sublist in l for item in sublist]

# -------------------- I/O utilities ----------------------

def print_list(list_to_print, indent=0, line_width=90):
  line_length = 0
  print(' ' * indent, end='')
  for val in list_to_print:
    val = str(val)
    line_length += len(val)
    print(val, end=', ')
    if line_length > line_width:
      print()
      print(' ' * indent, end='')
      line_length = 0
  print()

def save_graph(filename = '', quality = 'HD', padding = 0.3, transparent = False):

  orig_filename = filename

  if not os.path.exists('saved_graphs'):
    os.mkdir('saved_graphs')

  if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png'):
    filename = filename+'.png'

  if filename == '':
    image_paths = [p.split('/')[-1].split('.')[0] for p in glob.glob('./saved_graphs/*')]
    available_indices = [int(p) for p in image_paths if p.isnumeric()]
    if len(available_indices) == 0:
      next_index = 1
    else:
      next_index = max(available_indices)+1
    filename = str(next_index).zfill(2)+'.png'

  if quality == 'SD':
    dpi = 90
  elif quality == 'HD':
    dpi = 150
  elif quality == 'Best':
    dpi = 300

  if orig_filename!='':
    plt.suptitle(orig_filename)

  plt.savefig('./saved_graphs/'+filename, dpi=dpi, bbox_inches='tight', transparent=transparent, pad_inches=padding)


# -------------------- Uncategorized utilities ----------------------

##############################################################################

def check_non_numeric_value_in_column(data, col_name):
    """
    Return the set of non-numeric values in the column of the data
    """
    return set([n for n in data[col_name].tolist() if isinstance(n, str) and not n.isnumeric()])


def split_and_strip(string, sep=',', strip_char=' '):
    """
    Split a string and strip the space around each part
    """
    return [part.strip(strip_char) for part in string.split(sep)]


def split_and_pad(string, sep=',', pad=1):
    """
    Split a string and pad some space around each part
    """
    return [' ' * pad + part + ' ' * pad for part in split_and_strip(string, sep)]


def get_local_variables(filter_=True):
    """
    # Reference https://stackoverflow.com/a/18425523
    Gets the name and definition of the local variables.
    :param: filter_ (Boolean): whether or not the variables starting with "_" need to be filtered out
    :return: dic
    """
    callers_local_vars = dict(inspect.currentframe().f_back.f_locals.items())
    if filter_:
        var_keys = list(callers_local_vars.keys())
        for key in var_keys:
            if key.startswith('_'):
                del callers_local_vars[key]
    return callers_local_vars


def retrieve_name(var):
    """
    # Reference https://stackoverflow.com/a/40536047
    Gets the name of the variable
    :param: var: variable to get name from
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

def list_or_make_list(li):

    li_name = retrieve_name(li)

    if isinstance(li, list):  # Check if the columns parameter is List
        return li
    elif isinstance(li, str):  # If not, make the single column a list
        return [li]
    else:
        print('[Error] The "'+li_name+'" variable you provided is of wrong data type.')
        raise

def validate_columns(data, columns, suppress_warning = False):
    """
    [Need update] Validate if the columns provided exist in the data
    :param: data (Boolean), columns (List or String)
    :return: None
    """

    columns = list_or_make_list(columns)

    # Convert the list of columns we want to operate on and the list of columns in the data into sets
    columns = set(columns)
    valid_columns = set(data.columns.tolist())

    # Compare the difference and get the list of columns that we want to operate on but do not exist in the data
    non_covered_columns = sorted(list(columns.difference(valid_columns)))

    # If this list is not empty, report an error
    if len(non_covered_columns) > 0:
        if not suppress_warning:
            print('[Error] The following columns that you provided are not in the data:\n' + ', '.join(non_covered_columns))
    # In the future, there should be some fuzzy matching or inference techniques to suggest the correct column name
    # For now, just return False
        return False
    else:
        return True


def drop_duplicates(data, columns=False, keep='first'):
    """
    Repackage of the pandas.drop_duplicates function
    :param: data (Boolean), columns (List), keep (String)
    :return: None
    """

    # Validate columns are valid
    if validate_columns(data, columns):

        # Choose commands depending on whether subset is provided
        if columns == False:
            data.drop_duplicates(keep=keep, inplace=True)
        else:
            data.drop_duplicates(subset=columns, keep=keep, inplace=True)


def drop_missing(data, columns=False, dimension='column'):
    """
    Repackage of the pandas.dropna function
    :param: data (Boolean), columns (List), dimension (String)
    :return: None
    """

    # Validate columns are valid
    if validate_columns(data, columns):

        # Convert dimension to axis
        axis = 1 if dimension.lower() == 'column' else 0 if 'row' else -1
        if axis == -1:
            print('[Error] The dimension provided is invalid, choose from "column" and "row".')
            raise

        # Choose commands depending on whether subset is provided
        if columns == False:
            data.dropna(axis=axis, inplace=True)
        else:
            data.drop_duplicates(subset=columns, axis=axis, inplace=True)


def drop_columns(data, columns):
    """
    Repackage of the pandas.drop function
    :param: data (Boolean), columns (List or String)
    :return: None
    """

    # Validate the columns
    if validate_columns(data, columns):

        # Drop the columns inplace
        data.dropna(columns, axis=1, inplace=True)


def check_duplicated_column_names(data):
    """
    Check and print out potentially duplicated column names
    :param: data (DataFrame)
    :return: None
    """

    # Store the data name
    data_name = retrieve_name(data)

    # Use regex to find col_names that ends with .1 or .n
    col_names = data.columns.tolist()
    potential_duplicated_col_names = [col for col in col_names if len(re.findall(r'.\d+$', col)) > 0 and re.sub(r'.\d+$', '', col) in col_names]

    # Print out potential duplicated col_names and suggest commands to change them
    for col in potential_duplicated_col_names:
        print('[Warning] The column named "' + col + '" may have the same name with the column "' + re.sub(r'.\d+$', '', col) + '". Please change it based on your understanding of the data source.\n\nTry this command:\nchange_column_name(' + data_name + ',"' + col + ' -> NEW_NAME")')

    # If the list is not empty, the check fails
    if len(potential_duplicated_col_names)>0:
        return False
    else:
        return True

def change_column_name(data, change_from, change_to=''):
    """
    Change a single column's name inplace
    :param: data (DataFrame), change_from (String), change_to (String)
    :return: None
    """

    # If change_to is still its default value, then information might be contained in change_from only, try use '->' to split
    if change_to == '':
        change_from, change_to = split_and_strip(change_from, sep='->')

    # validate old column name exists and new name doesn't exist
    if validate_columns(data, change_from):
        if not validate_columns(data, change_to, suppress_warning = True):
            data.rename(columns={change_from: change_to}, inplace=True)
            if check_operation_keyword_column_names(data):
                print('[Success] Column name changed: ' + change_from + ' -> ' + change_to)
        else:
            print('[Error] The column name ' + change_to + ' already exists in the data.')
    else:
        pass



def check_operation_keyword_column_names(data, operation_keywords = ['duplicate','drop','keep']):
    """
    Check and print out column names that are operation keywords in this program
    :param: data (DataFrame)
    :return: None
    """

    # Store the data name
    data_name = retrieve_name(data)

    # Get the intersection between the operation keywords and column names in the data
    operation_keyword_column_names = sorted(list(set(operation_keywords).intersection(set(data.columns.tolist()))))

    # Print out operation_keyword_column_names and suggest commands to change them
    for col in operation_keyword_column_names:
        print('[Warning] The column name "' + col + '" is an operation keyword. Please change it to another column name.\n\nTry this command:\nchange_column_name(' + data_name + ',"' + col + ' -> NEW_NAME")')

    # If the list is not empty, the check fails
    if len(operation_keyword_column_names)>0:
        return False
    else:
        return True


def find_contains(li, keyword, ignore_case = False):
    """
    Return a list of the elements in the original list that contains the keyword
    :param: li (List)
    :return: List
    """
    if ignore_case:
        return [element for element in li if keyword.lower() in element.lower()]
    return [element for element in li if keyword in element]


##############################################################################

# For the function below, example usage is shown in the Colab Notebook: https://colab.research.google.com/drive/1P6pKdGkz1IQzHR-wzvoFaCZ4H1z-VoTL

# These functions will be improved and annotated in the future.

def report_proportion_of_missing_data(data, columns=False):
    """
    Return the proportion of missing data in different columns
    :param: data (DataFrame), columns (List)
    :return: pd.Series
    """
    if not columns:
        null_rate = data.isnull().mean()
    else:
        columns = list_or_make_list(columns)
        null_rate = data[columns].isnull().mean()
    return null_rate.sort_values(ascending=False)

def report_proportion_of_available_data(data, columns=False):
    """
    Print the proportion of available data in different columns
    :param: data (DataFrame), columns (List)
    :return: pd.Series
    """
    if not columns:
        non_null_rate = data.notnull().mean()
    else:
        columns = list_or_make_list(columns)
        non_null_rate = data[columns].notnull().mean()
    print('\nAvailability for fields:\n')
    print(non_null_rate.sort_values(ascending=False))

def report_data_type_in_dataframe(data):
    """
    Print the data types of the different columns
    :param: data (DataFrame)
    :return: pd.Series
    """
    print('\nData types of fields:\n')
    print(data.dtypes)

def profile_data_and_drop_duplicates(data):
    print('\n\n\nData Profile')
    data.drop_duplicates(subset=['RecordId'], keep='last', inplace=True)
    data.reset_index(drop=True, inplace=True)
    assert(data['RecordId'].dtype == 'int64')
    assert(data['HouseHoldId'].dtype == 'int64')
    print('\n==========================================================')
    data.info()
    print('\n==========================================================')
    report_proportion_of_available_data(data)
    print()
    report_data_type_in_dataframe(data)
    print()
    for col in data.columns.tolist():
        print('==========================================================')
        print(col)
        print()
        print(data[col].value_counts())
        print()

def is_numeric(x):
    return isinstance(x, (int, np.integer)) or isinstance(x, (float, np.floating)) or (isinstance(x,str) and x.replace('.','').replace(',','').replace(' ','').replace('/','').isnumeric())

def numeric_percentage(data, col):
    return data[col].dropna().apply(is_numeric).mean()

def check_numeric_percentage(data, after_column = None , check_all_after = False, ignore_columns = ['DataId', 'RecordId', 'HouseHoldId', 'NameId', 'Street Name', 'Street', 'House Number', 'Dwelling Number', 'Family Number', 'Line Number']):
  start_checking = False if after_column is not None else True
  potential_error_count = 0
  for col in data.columns:

      if col == after_column:
          start_checking = True
          continue

      if start_checking and col not in ignore_columns:

          col_numeric_percentage = numeric_percentage(data, col)
          if col_numeric_percentage > 0.9 and col_numeric_percentage < 1:
              print('"'+col+'"','has',str(round(100*(1-col_numeric_percentage),1))+'%','"non-numeric" values.\n')
              potential_error_count += 1
          if col_numeric_percentage > 0 and col_numeric_percentage < 0.1:
              print('"'+col+'"','has',str(round(100*(col_numeric_percentage-0),1))+'%','"numeric" values.\n')
              potential_error_count += 1

      if not check_all_after and potential_error_count == 1:
          print('[Pause] Stop checking for now.')
          break

def get_indices(boolean_series):
  return boolean_series[boolean_series].index.tolist()

def top_non_numeric_values(data, col, top_k = 10, show_all = False, return_indices = True):
    boolean_series = (~data[col].apply(is_numeric)) & (data[col].notnull())
    top_non_numeric_values = data.loc[boolean_series, col].value_counts()
    if show_all:
        print( top_non_numeric_values )
    else:
        total_rows = len(top_non_numeric_values)
        if top_k < total_rows:
            print('There are '+str(total_rows)+' rows. Use show_all = True to see full results.\n')
        print( top_non_numeric_values[:top_k] )
    if return_indices:
        return get_indices(boolean_series)

def top_numeric_values(data, col, top_k = 10, show_all = False, return_indices = True):
    boolean_series = (data[col].apply(is_numeric)) & (data[col].notnull())
    top_numeric_values = data.loc[boolean_series, col].value_counts()
    if show_all:
        print( top_numeric_values )
    else:
        total_rows = len(top_numeric_values)
        if top_k < total_rows:
            print('There are '+str(total_rows)+' rows. Use show_all = True to see full results.\n')
        print( top_numeric_values[:top_k] )
    if return_indices:
        return get_indices(boolean_series)

def show_df_at_indices(data, indices):
  return data[data.index.isin(indices)].copy()

def shift_patch_of_dataframe(data, indices, origin_col, shift, width = None, check_again = True):

    fields = data.columns.tolist()
    index_of_origin_col = fields.index(origin_col)
    if shift>=0:
        from_cols = fields[index_of_origin_col:len(fields) - shift]
        to_cols = fields[index_of_origin_col + shift:len(fields)]
    else:
        from_cols = fields[index_of_origin_col:len(fields)]
        to_cols = fields[index_of_origin_col + shift:len(fields) + shift]
    if width is not None:
        from_cols, to_cols = from_cols[:width], to_cols[:width]

    data.loc[indices, to_cols] = data.loc[indices, from_cols].to_numpy()

    data.loc[indices, origin_col] = np.nan

    print('\n[Success] Patch shifted.\n')

    if check_again:
        print('Re-running "check_numeric_percentage".\n')
        check_numeric_percentage(data, after_column = origin_col)

def drop_indices(data, indices):

    data.drop(indices, axis = 0, inplace = True)

    print('\n[Success] '+str(len(indices))+' rows dropped.\n')

def keep_only_common(data, field, common_values, placeholder = 'MISSING'): # if MISSING is a meaningful value for this field, modifier the placeholder value
    common_values = set(list(common_values))
    if placeholder not in common_values:
        common_values.add(placeholder)
    data[field] = data[field].copy().fillna(placeholder).apply(lambda x: x if x in common_values else np.nan).dropna().replace(placeholder, np.nan)
    print('The common values in column "'+field+'" are kept.')

def get_indices_of_not_belonged(data, field, value_list):
    return data.index[~data[field].isin(value_list+[np.nan])].tolist()

def try_length_is_zero(x):
  try:
    if isinstance(x,int):
      return False
    return len(x)==0
  except:
    return True

def create_mapping_from_df(dataframe, key, value, drop_nan_value = True, drop_empty_value = True):
  temp_df = dataframe[[key,value]].copy()
  if drop_empty_value:
    temp_df[value] = temp_df[value].apply(lambda x: np.nan if try_length_is_zero(x) else x)
  if drop_nan_value:
    temp_df = temp_df.dropna()
  return temp_df.set_index(key)[value].to_dict()


def time_now(timezone=None,detail_level='m',hyphen=False):
  if timezone == None:
    timezone_flag = pytz.utc
    timezone_marker = 'UTC'
  elif timezone.lower() in ['china','shanghai','beijing']:
    timezone_flag = pytz.timezone('Asia/Shanghai')
    timezone_marker = 'china'
  elif timezone.lower() in ['est','edt','us eastern','eastern','new york','newyork','ny','nyc']:
    timezone_flag = pytz.timezone('US/Eastern')
    timezone_marker = 'useastern'
  elif timezone.lower() in ['cst','cdt','us central','central']:
    timezone_flag = pytz.timezone('US/Central')
    timezone_marker = 'uscentral'
  elif timezone.lower() in ['pst','pdt','us pacific','pacific']:
    timezone_flag = pytz.timezone('US/Pacific')
    timezone_marker = 'uspacific'

  raw_time_string = str(datetime.now(timezone_flag))

  if detail_level.lower()[0] == 'd':
    output = raw_time_string.split(' ')[0]
  elif detail_level.lower()[0] == 'h':
    output = '-'.join(raw_time_string.split(':')[:1])
  elif detail_level.lower()[0] == 'm':
    output = '-'.join(raw_time_string.split(':')[:2])
  elif detail_level.lower()[0] == 's':
    output = '-'.join(raw_time_string.split(':')[:3])

  output = output.replace(' ','_')+'_'+timezone_marker

  if not hyphen:
    output = output.replace('-','')

  return output
