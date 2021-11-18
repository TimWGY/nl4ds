from IPython.display import clear_output

import pandas as pd
import numpy as np
import re
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
from itertools import combinations


def load_data(filename):
  if filename.endswith('xlsx'):
    return pd.read_excel(filename)
  elif filename.endswith('csv'):
    return pd.read_csv(filename)


def transform_column_names(data, custom_function):
  if callable(custom_function):
    column_name_mapping = dict(zip(data.columns.tolist(), [custom_function(col) for col in data.columns.tolist()]))
  elif isinstance(custom_function, dict):
    column_name_mapping = custom_function
  else:
    print('[Error] Invalid input for "custom_function" parameter.')
    raise
  data = data.rename(columns=column_name_mapping)
  return data

def perform_test(control_group_arr, intervention_group_arr, test_func, significance_threshold=0.1):

  print('Mean of Control', round(control_group_arr.mean(), 2), '\nMean of Intervention', round(intervention_group_arr.mean(), 2))
  res = test_func(control_group_arr, intervention_group_arr)
  print('Control group ' + ('<' if control_group_arr.mean() < intervention_group_arr.mean() else '>') + ' Intervention group\n' + 'p_value for significant difference:', round(res.pvalue, 3))
  significant_or_not = res.pvalue < significance_threshold
  print('The difference is ' + ('*significant*' if significant_or_not else 'insignificant') + '.')
  print()
  return significant_or_not

def pre_post_intervention_test_single_field(df, field_name, test_type=stats.mannwhitneyu, significance_threshold=0.1, group_field='group', group_indicator=('control', 'intervention'), pre_post_indicator=('pre_', 'post_')):

  print('\nDetect difference between the control group versus experimental group on their pre/post values in the field:\n"' + field_name + '"\n')

  if test_type == stats.mannwhitneyu:
    print('Mann–Whitney U test is performed to detect mean difference.\n')
  else:
    print(str(test_type.__name__) + ' test is performed to detect mean difference.\n')

  data = df.copy()
  data = data.dropna(subset=[pre_post_indicator[0] + field_name, pre_post_indicator[1] + field_name])
  number_of_dropped_entries = len(df) - len(data)
  if number_of_dropped_entries > 0:
    print(number_of_dropped_entries, 'entries are dropped due to missing values.\n')
  control_pre = data.loc[data[group_field] == group_indicator[0], pre_post_indicator[0] + field_name]
  intervention_pre = data.loc[data[group_field] == group_indicator[1], pre_post_indicator[0] + field_name]
  control_post = data.loc[data[group_field] == group_indicator[0], pre_post_indicator[1] + field_name]
  intervention_post = data.loc[data[group_field] == group_indicator[1], pre_post_indicator[1] + field_name]
  print('At the baseline:')
  pre_diff_sig_or_not = perform_test(control_pre, intervention_pre, test_type, significance_threshold=significance_threshold)
  print('At the end of study:')
  post_diff_sig_or_not = perform_test(control_post, intervention_post, test_type, significance_threshold=significance_threshold)
  return pre_diff_sig_or_not, post_diff_sig_or_not


def pre_post_intervention_test(df, field_names, test_type=stats.mannwhitneyu, significance_threshold=0.1, group_field='group', group_indicator=('control', 'intervention'), pre_post_indicator=('pre_', 'post_')):
  if isinstance(field_names, str):
    field_names = [field_names]
  for field_name in field_names:
    pre_diff_sig_or_not, post_diff_sig_or_not = pre_post_intervention_test_single_field(df, field_name, test_type=test_type, significance_threshold=significance_threshold, group_field=group_field, group_indicator=group_indicator, pre_post_indicator=pre_post_indicator)
    print('\n--------------------------------------------\n')

def run_intervention_test(data_path):

  df = pd.read_excel(data_path, sheet_name=0)
  df.columns = [col.strip() for col in df.columns]
  df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
  df = df.dropna(how='all', axis=1)
  null_count_ser = df.isnull().sum()
  if null_count_ser.sum() > 0:
    print('[Warning] Null values found in column :"' + null_count_ser.index.tolist()[np.argmax(null_count_ser)] + '"')

  print('\n--------------------------------------------\n')
  print('There are ' + str(len(df)) + ' observations.')
  print('\n--------------------------------------------\n')

  df = df[[col for col in df.columns if col.startswith('pre_') or col.startswith('post_') or col in ('id', 'group')]]

  field_names = list(set([re.sub('^post\_', '', re.sub('^pre\_', '', col)) for col in df.columns.tolist() if col != 'id' and col != 'group']))

  pre_post_intervention_test(df, field_names)

def get_correlation_significance(input_df, col1=None, col2=None):

  if col1 == None or col2 == None:
    if len(input_df.columns) != 2:
      raise
    else:
      col1, col2 = input_df.columns.tolist()

  pair_df = input_df.copy()[[col1, col2]].dropna()

  return round(pearsonr(pair_df[col1], pair_df[col2])[1], 4)


def get_combinations_of_two(list_of_elements):
  return list(combinations(list_of_elements, 2))

def show_corr(data, cols, title=None):

  if isinstance(cols, str):
    cols = [col.strip() for col in cols.strip().split(',')]
  try:
    data_df = data[cols].copy()
    corr_df = data_df.corr()
    corr_columns = cols
  except KeyError as e:
    print('Variable "' + re.findall(r"\[\'(.*?)\'\]", str(e))[0] + '" not found, check your spelling please.')
    return

  cmap = sns.diverging_palette(10, 130, as_cmap=True)  # red green

  corr = corr_df.values
  np.fill_diagonal(corr, np.nan)
  labels = corr_df.columns

  plt.figure(figsize=(8, 6), dpi=150)
  if title != None:
    plt.title(title)

  g = sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, xticklabels=labels, yticklabels=labels)

  for pair in get_combinations_of_two(corr_columns):
    p_value = get_correlation_significance(data_df, * pair)
    if p_value < 0.05:
      print(pair, p_value)
      highlight_position_1 = (corr_columns.index(pair[0]), corr_columns.index(pair[1]))
      highlight_position_2 = (corr_columns.index(pair[1]), corr_columns.index(pair[0]))
      edgecolor = 'gold' if p_value < 0.01 else 'brown'
      lw = 2 if p_value < 0.01 else 1
      g.add_patch(Rectangle(highlight_position_1, 1, 1, fill=False, edgecolor=edgecolor, lw=lw))
      g.add_patch(Rectangle(highlight_position_2, 1, 1, fill=False, edgecolor=edgecolor, lw=lw))


clear_output()
