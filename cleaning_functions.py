import os
os.system('pip install pandas')
os.system('pip install numpy')
os.system('pip install thefuzz')
os.system('pip install networkx')
os.system('pip install tqdm')
clear_output()
import pandas as pd
import numpy as np
import re
from thefuzz import process
import networkx as nx
from tqdm import tqdm
tqdm.pandas()

def get_most_frequent_term(term_group, term_to_freq_mapping):

  term_group = sorted(term_group, key=len)
  most_frequent_term = term_group[0]
  max_freq = term_to_freq_mapping[most_frequent_term]
  for term in term_group[1:]:
    term_freq = term_to_freq_mapping[term]
    if term_freq > max_freq:
      most_frequent_term = term
      max_freq = term_freq
  return most_frequent_term

def fuzzy_cluster(data, field, correct_term_min_freq=5, threshold=90):

  temp_df = data[field].dropna().value_counts().reset_index().rename(columns={'index': 'term', field: 'freq'})

  term_to_freq_mapping = temp_df.set_index('term')['freq'].to_dict()

  all_terms = temp_df.loc[temp_df['freq'] >= correct_term_min_freq, 'term'].tolist()

  temp_df['top_matches'] = temp_df['term'].progress_apply(lambda x: process.extract(x, all_terms))

  temp_df['thresholded_matches'] = temp_df['top_matches'].apply(lambda li: [t[0] for t in li if t[1] > threshold])

  G = nx.Graph()

  for row in temp_df[['term', 'thresholded_matches']].itertuples():
    term = row.term
    for match in row.thresholded_matches:
      G.add_edge(term, match)

  term_groups = [list(group) for group in list(nx.connected_components(G))]

  term_to_most_frequent_term_mapping = {}
  for term_group in term_groups:
    most_frequent_term = get_most_frequent_term(term_group, term_to_freq_mapping)
    term_to_most_frequent_term_mapping.update(dict(zip(term_group, [most_frequent_term] * len(term_groups))))

  temp_df['most_common_term'] = temp_df['term'].map(term_to_most_frequent_term_mapping)

  temp_df = temp_df.dropna(subset=['most_common_term'])

  term_correction_mapping = temp_df.set_index('term')['most_common_term'].to_dict()

  return term_correction_mapping

def make_tail_other(data, field, min_freq=None, tail_percentile=None, other_marker='other'):

  vcnts = data[field].value_counts()
  prop_vcnts = vcnts / vcnts.sum()
  prop_vcnts_cumsum = prop_vcnts.cumsum()

  # 1
  if min_freq is not None:
    preserved_values = vcnts[vcnts >= min_freq].index.tolist()  # values_above_min_freq

  # 2
  if tail_percentile is not None:
    preserved_values = prop_vcnts_cumsum[prop_vcnts_cumsum < tail_percentile].index.tolist()  # values_up_to_tail_percentile

  ## 3, not ready
  # reverse_prop_vcnts_cumsum = 1 - prop_vcnts_cumsum
  # values_where_last_is_greater_than_other_combined = prop_vcnts[prop_vcnts > reverse_prop_vcnts_cumsum].index.tolist()

  ## 4, not ready
  # proportion_change_rate = prop_vcnts.diff()
  # proportion_change_rate_rolling_mean = proportion_change_rate.rolling(len(vcnts)//10).mean().fillna(1)
  # values_before_change_rate_average_to_zero = proportion_change_rate_rolling_mean[proportion_change_rate_rolling_mean>0].index.tolist()

  preserved_values = set(preserved_values)
  preserved_values.add(np.nan)

  data[field] = data[field].apply(lambda x: x if x in preserved_values else other_marker)

  return data


# Example Usage

df['address'] = df['address'].fillna('').apply(str.lower).apply(lambda x: re.sub(r'[^a-z0-9 ]', '', x)).replace('', np.nan)

address_correction_mapping = fuzzy_cluster(df, 'address')

df['auto_cleaned_address'] = df['address'].map(address_correction_mapping, df['address'])

df = make_tail_other(df, 'auto_cleaned_address', min_freq=10)
