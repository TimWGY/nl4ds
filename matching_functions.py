from IPython.core.display import clear_output
clear_output()

print('Loading packages, this may take a minute ...')

import os
os.system('pip install pandas')
os.system('pip install numpy')
os.system('pip install python-Levenshtein')
os.system('pip install thefuzz')
os.system('pip install networkx')
os.system('pip install tqdm')
os.system('pip install sklearn')
os.system('pip install scipy')
os.system('pip install haversine')
os.system('pip install jellyfish')
os.system('pip install unidecode')
clear_output()

import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 200)
import numpy as np
import re
import json
from glob import glob
from thefuzz import process, fuzz
import networkx as nx
from sklearn.cluster import DBSCAN
import scipy
from haversine import haversine
from unidecode import unidecode
import jellyfish
from functools import partial
from tqdm import tqdm

###### GENERAL, IO, TEXTUAL UTILS ######

def flatten_list(l):
  return [item for sublist in l for item in sublist]

def unique_preserving_order(l):
  # Reference:http://www.peterbe.com/plog/uniqifiers-benchmark
  seen = set()
  seen_add = seen.add
  return [x for x in l if not (x in seen or seen_add(x))]

def decimal_floor(a, precision=0):
  # Reference: https://stackoverflow.com/a/58065394
  return np.true_divide(np.floor(a * 10**precision), 10**precision)

def load_geojson_as_pandas_df(data_file_path, long_to_lon = True, need_geometry = True):
  with open(data_file_path) as f:
      data = json.load(f)
  print('Geojson Metadata')
  for c in 'type, name, crs'.split(', '):
    print(c+':',data[c])
  df = pd.DataFrame(data['features'])
  if need_geometry:
    df['geometry'] = df['geometry'].apply(lambda geom: {'type':'Missing','coordinates':[np.nan, np.nan]} if geom is None else geom)
    df = pd.concat([pd.json_normalize( df['properties'] ), pd.json_normalize( df['geometry'] ).rename(columns = {'type':'geom_type','coordinates':'geom_coordinates'})], axis = 1)
  else:
    df = pd.json_normalize( df['properties'] )
  if long_to_lon:
    df = df.rename(columns = {'long':'lon'})
  print('\nLoaded successfully with '+str(len(df.columns))+' columns and '+str(len(df))+' rows.\n')
  return df

def fetch_value_from_row(row, candidate_cols):
  value = np.nan
  candidate_cols_copy = candidate_cols[:]
  while np.isnan(value) and len(candidate_cols_copy)>0:
    value = row[candidate_cols_copy.pop()]
    if isinstance(value, str):
      break
  return value

def fill_with_other_cols_if_na(data, col, other_cols):
  if isinstance(other_cols, str):
    other_cols = [other_cols]
  candidate_cols = other_cols[::-1]+[col]
  data[candidate_cols] = data[candidate_cols].applymap(lambda x: np.nan if x is None else x)
  data[col] = data[candidate_cols].apply(lambda row: fetch_value_from_row(row, candidate_cols), axis=1)
  return data

def apply_by_value_group(input_df, field, custom_func, show_progress=True):
  input_df = input_df.copy()
  output_df = pd.DataFrame()
  unique_values = input_df[field].unique().tolist()
  if show_progress:
    for value in tqdm(unique_values):
      input_part = input_df[input_df[field]==value].copy()
      output_part = custom_func(input_part, value)
      output_df = output_df.append(output_part, ignore_index=True)
  else:
    for value in unique_values:
      input_part = input_df[input_df[field]==value].copy()
      output_part = custom_func(input_part, value)
      output_df = output_df.append(output_part, ignore_index=True)
  return output_df

# Alias of apply_by_value_group: apply_to_each_cluster
apply_to_each_cluster = apply_by_value_group 

def clean_text(x):
  # deaccent, lower, no_special (only alphanumeric), shrink_whitespace
  output = re.sub(r'\s+',' ',re.sub('[^a-z0-9\s]','',unidecode(x).lower().replace('-',' '))).strip() if isinstance(x,str) else np.nan
  output = output if isinstance(output,str) and output != '' else np.nan
  return output

###### PHONETIC CODE UTILS ######

soundex = jellyfish.soundex # 'J412'
nysiis = jellyfish.nysiis # 'JALYF'
mr_codex = jellyfish.match_rating_codex # 'JLLFSH'
metaphone = jellyfish.metaphone # 'JLFX'
fingerprint = lambda x: ' '.join([w if w.isnumeric() else (''.join(unique_preserving_order(w))).upper() for w in x.split()]) # JELYFISH

def create_phonetic_column(data, field, phonetic_code = 'nysiis', prefix = None):
  if prefix is None:
    prefix = field + '__'
  phonetic_codes = ['nysiis','soundex','mr_codex','metaphone']
  if phonetic_code == 'all':
    for code in phonetic_codes:
      func = eval(code)
      data[prefix + code] = data[field].apply(lambda x: ' '.join([(w if w.isnumeric() else func(w)) for w in x.split()]) if isinstance(x,str) else np.nan).apply(lambda x: np.nan if not isinstance(x,str) or len(x)==1 else x) 
      # last apply function in the previous line turns single letter phonetic signature into NaN, they are too abstract and can match arbitrarily different words
  elif phonetic_code in phonetic_codes:
    code = phonetic_code
    func = eval(code)
    data[prefix + code] =   data[field].apply(lambda x: ' '.join([(w if w.isnumeric() else func(w)) for w in x.split()]) if isinstance(x,str) else np.nan).apply(lambda x: np.nan if not isinstance(x,str) or len(x)==1 else x)
  else:
    raise "[Error] Invalid phonetic code, the options are 'nysiis','soundex','mr_codex','metaphone'"
  return data

###### GEOSPATIAL CLUSTERING ######

def get_geo_dbscan_labels(data, field, radius = 50, min_samples = 2):
  clusterer = DBSCAN(eps=(radius/1000)/6371., algorithm='ball_tree', metric='haversine', min_samples=min_samples)
  coorindates_array = np.array(data[field].apply(lambda tup: np.radians(np.array(tup))).tolist())
  clusterer.fit(coorindates_array)
  clabels = clusterer.labels_
  return clabels

def create_geo_cluster_column(data, field, radius, min_samples = 2):
  spatial_cluster_id_column = 'geo_dbscan_r'+str(radius)+'_cluster_id'
  labels = get_geo_dbscan_labels(data=data, field=field, radius=radius, min_samples=min_samples)
  singleton_count = sum([x == -1 for x in labels])
  data[spatial_cluster_id_column] = labels
  data.loc[data[spatial_cluster_id_column]==-1, spatial_cluster_id_column] = sorted(list(range(-singleton_count,0,1)), reverse=True)
  data.loc[data[spatial_cluster_id_column]>=0,  spatial_cluster_id_column] = data.loc[data[spatial_cluster_id_column]>=0, spatial_cluster_id_column]+1
  return data

###### SPATIAL LOCATION AND DISTANCE UTILS ######

def get_convex_hull(point_list):
  """Return vertices of the hull, guaranteed to be in counter-clockwise order for 2D"""
  point_list = list(set(point_list))
  if len(point_list)<=2:
    return point_list
  hull_vertices_indices = [scipy.spatial.ConvexHull(point_list).vertices]
  vertices = np.array(point_list)[tuple(hull_vertices_indices)]
  return vertices

def get_max_dist_between_points(point_list, dist_metric):
  point_list = list(point_list)
  if len(point_list) == 0:
    return np.nan
  max_dist = 0
  while len(point_list)>1:
    point = point_list.pop()
    dists = [dist_metric(point, p) for p in point_list]
    max_dist = max(max_dist, max(dists))
  return max_dist

def calculate_max_dist_within_cluster(data, cluster_id_col = 'dbscan_cluster_id'):
  cluster_id_to_max_dist_within_cluster_mapping = {}
  for cluster_id in tqdm(data[cluster_id_col].tolist()):
    if cluster_id>0:
      point_list = data.loc[data[cluster_id_col]==cluster_id,'coordinates'].tolist()
      cluster_id_to_max_dist_within_cluster_mapping[cluster_id] = round(get_max_dist_between_points(get_convex_hull(point_list), haversine)*1000,1)
  data['max_dist_within_cluster'] = data[cluster_id_col].map(cluster_id_to_max_dist_within_cluster_mapping).fillna(0)
  ax = data['max_dist_within_cluster'].hist(bins=range(0,int(data['max_dist_within_cluster'].max())+1,25))
  return data

def np_median_center(points, crit=0.0001, verbose = False):
  # Reference: http://pysal.org/notebooks/explore/pointpats/centrography.html#Central-Tendency
  points = [tuple(pt) for pt in points]
  if len(set(points)) == 1:
    return points[0]
  points = np.asarray(points)
  x0, y0 = points.mean(axis=0)
  dx = np.inf
  dy = np.inf
  iteration = 0
  while np.abs(dx) > crit or np.abs(dy) > crit:
    xd = points[:, 0] - x0
    yd = points[:, 1] - y0
    d = np.sqrt(xd*xd + yd*yd)
    w = 1./d
    w = w / w.sum()
    x1 = w * points[:, 0]
    x1 = x1.sum()
    y1 = w * points[:, 1]
    y1 = y1.sum()
    dx = x1 - x0
    dy = y1 - y0
    iteration +=1 
    if verbose:
      print(x0, x1, dx, dy, d.sum(), iteration)
    x0 = x1
    y0 = y1
  return round(x1,6), round(y1,6)

###### TEXT FUZZY MATCH CLUSTERING ######

def self_fuzzy_cluster(data, field, correct_term_min_freq=1, scorer=fuzz.partial_ratio, score_cutoff=90, verbose=False):
  """For each fuzzy cluster, choose most frequent term as exemplar, if there is a tie, use the longer term."""
  temp_df = data[field].dropna().value_counts().reset_index().rename(columns = {'index':'term',field:'freq'})
  temp_df['term_length'] = temp_df['term'].apply(len)
  temp_df['freq_length_tuple'] = list(zip(temp_df['freq'], temp_df['term_length']))
  term_to_freq_length_tuple_mapping = temp_df.set_index('term')['freq_length_tuple'].to_dict()
  all_terms = temp_df.loc[temp_df['freq']>=correct_term_min_freq,'term'].tolist()
  if verbose:
    temp_df['thresholded_matches'] = temp_df['term'].progress_apply(lambda x: [term for term,freq in process.extractBests(x, all_terms, scorer=scorer, score_cutoff=score_cutoff)])
  else:
    temp_df['thresholded_matches'] = temp_df['term'].apply(lambda x: [term for term,freq in process.extractBests(x, all_terms, scorer=scorer, score_cutoff=score_cutoff)])
  G = nx.Graph()
  for row in temp_df[['term','thresholded_matches']].itertuples():
    term = row.term
    for match in row.thresholded_matches:
      G.add_edge(term, match)
  term_groups = [list(group) for group in list(nx.connected_components(G))]
  term_to_most_frequent_term_mapping = {}
  for term_group in term_groups:
    most_frequent_term = max(term_group, key = lambda x: term_to_freq_length_tuple_mapping[x])
    term_to_most_frequent_term_mapping.update( dict(zip(term_group, [most_frequent_term]*len(term_group))) )
  temp_df['most_common_term'] = temp_df['term'].map(term_to_most_frequent_term_mapping)
  temp_df = temp_df.dropna(subset=['most_common_term'])
  term_correction_mapping = temp_df.set_index('term')['most_common_term'].to_dict()
  return term_correction_mapping

def get_match_score(term1, term2):
  return fuzz.partial_ratio(term1, term2)

def create_fuzzy_cluster_column(data, text_col, scorer = get_match_score, score_cutoff = 80):
  data[text_col+'_suggested'] = data[text_col].map( self_fuzzy_cluster(data, text_col, scorer=scorer, score_cutoff=score_cutoff) )
  return data

def consolidate_suggestion(data, groupby_col, uuid_col, coord_col):
  consolidated_data = data.groupby(groupby_col).agg({uuid_col: lambda x: '|'.join(sorted(x)), coord_col: lambda x: np_median_center(list(x))}).reset_index()
  return consolidated_data[[uuid_col, groupby_col, coord_col]]

def fuzzy_cluster_and_consolidate(part, value, text_col, uuid_col, coord_col, scorer, score_cutoff):
  if value < 0:
    part = part.rename(columns = {text_col: text_col+'_suggested'})[[uuid_col, text_col+'_suggested', coord_col]]
  else:
    part = create_fuzzy_cluster_column(part, text_col = text_col, scorer = scorer, score_cutoff = score_cutoff)
    part = consolidate_suggestion(part, groupby_col = text_col+'_suggested', uuid_col = uuid_col, coord_col = coord_col)
  return part

###### TEXT FUZZY MATCH EVALUATION ######

def length_based_rescale(x, median_length, to_power = 0.9):
  return decimal_floor(np.power(np.log(x),to_power)/np.power(np.log(median_length),to_power),2)
def string_coverage_ratio(token_1, token_2):
  short, long = sorted([token_1, token_2], key = len)
  return decimal_floor(1-len(long.replace(short,''))/len(long),2)
def within_token_match_ratio(string_1, string_2, metric = 'coverage'):
  max_wtmr = 0
  for string_1_token in string_1.split():
    for string_2_token in string_2.split():
      if len(string_1_token)>=2 and len(string_2_token)>=2:
        if metric == 'coverage':
          new_wtmr = string_coverage_ratio(string_1_token, string_2_token)
        else:
          new_wtmr = metric(string_1_token, string_2_token)
        max_wtmr = max(max_wtmr, new_wtmr)
  return max_wtmr

###### ######

clear_output()
print('\nMatching module is ready. Enjoy exploring!\n')

