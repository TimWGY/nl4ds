from IPython.core.display import clear_output

#======================================= SELF FUZZY CLUSTER =======================================#

os.system('pip install python-Levenshtein')
os.system('pip install thefuzz')
os.system('pip install networkx')
os.system('pip install tqdm')
clear_output()

from thefuzz import fuzz, process
import networkx as nx
from tqdm import tqdm
tqdm.pandas()

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


#==================================================================================================#