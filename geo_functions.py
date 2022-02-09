from IPython.display import clear_output

# ------------------------------------Import Libraries----------------------------------------

import os
import re
import pandas as pd
import numpy as np

def load_raw_address_of_nyc_chinese_dataset():
  df = pd.read_excel(data_folder_path +'/'+ 'Addresses of NYC Chinese.xlsx', sheet_name=0)
  df = df[[col for col in df.columns if not 'Unnamed:' in col]].dropna(how='all')
  df = df.applymap(lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x)
  df = df[df['Address'].apply(lambda x: x != x.upper())]
  return df

def load_and_prep_address_of_nyc_chinese_dataset():

  # Read in live-updated address list excel
  df = load_raw_address_of_nyc_chinese_dataset()
  raw = df.copy()

  # Preprocess the address dataset

  # only rows with valid street name and house number information
  df['street_name'] = df['Address'].apply(lambda x: ' '.join(x.split()[1:])).apply(str.lower).apply(str.strip).replace('', np.nan)
  df['street_name'] = df['street_name'].fillna('').apply(str.strip).apply(str.upper).apply(lambda x: re.sub(r'(.*?) ST$', r'\1 STREET', x)).apply(lambda x: re.sub(r'(.*?) AVE?$', r'\1 AVENUE', x)).apply(lambda x: re.sub(r'^AVE? (.*?)', r'AVENUE \1', x)).apply(str.lower)
  df['street_name'] = df['street_name'].apply(lambda x: re.sub('^west ', 'w ', x)).apply(lambda x: re.sub('^east ', 'e ', x))
  df['street_name'] = df['street_name'].replace('', np.nan)

  df['house_number'] = df['Address'].apply(lambda x: x.split()[0]).apply(str.lower).apply(str.strip).replace('', np.nan)
  df.loc[~df['house_number'].apply(str.isnumeric), 'house_number'] = df.loc[~df['house_number'].apply(str.isnumeric), 'house_number'].apply(lambda x: x.rstrip('ab').split('-')[0].split('.')[0] if x.rstrip('ab').replace('-', '').replace('.', '').isnumeric() else np.nan)
  df = df.dropna(subset=['street_name', 'house_number'])

  df['address_for_geocoding'] = df['street_name'] + ' ' + df['house_number'].apply(str)

  # only manhattan
  df = df[df['NOT MANHATTAN'].isnull()]
  df = df[~((df['street_name'].str.endswith('bk')) | (df['street_name'].str.endswith('brooklyn')))]

  # standardize type of places
  df[['Residence', 'Business', 'Community Space', 'Restaurant']] = df[['Residence', 'Business', 'Community Space', 'Restaurant']].applymap(lambda x: 1 if x > 0 else 0)  # nan>0 evaluates to False
  df['HBCR'] = df[['Residence', 'Business', 'Community Space', 'Restaurant']].apply(lambda row: str(row['Residence']) + str(row['Business']) + str(row['Community Space']) + str(row['Restaurant']), axis=1)
  df['place_type'] = df['HBCR'].apply(lambda x: 'Home' if x == '1000' else 'Restaurant' if x == '0101' else 'Non-restaurant Business' if x == '0100' else 'Other')

  # plot year distribution
  # _ = df['YR'].hist(bins=100)

  return df, raw

# Prepare the geocoding functions


import ast

os.system('pip install python-Levenshtein')
os.system('pip install thefuzz')
from thefuzz import process
from thefuzz import fuzz
os.system('pip install tqdm')
from tqdm import tqdm
tqdm.pandas()

os.system('pip install geopy')
from geopy.distance import distance
os.system('pip install geographiclib')
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84


def haversine_distance(point_1, point_2):
  return distance(point_1, point_2).m


def get_direction(point_1, point_2):
  return geod.Inverse(*point_2, *point_1)['azi1']


def add_degree_to_azimuth(current, change):
  assert(abs(change) < 180)
  output = current + change
  if output < -180:
    output = 180 - (-output - 180)
  elif output > 180:
    output = -180 + (output - 180)
  if output == -180:
    output = -output
  return output

def get_coordinates_from_details(target_building_num, building_num_range, start_end_coordinates, segment_direction, odd_on, offset_from_road_center):

  is_odd = target_building_num % 2 == 1

  if (building_num_range[1] - building_num_range[0]) == 0:
    street_center_position = np.mean(start_end_coordinates, axis=0).tolist()
  else:
    f_pt_proportion = (target_building_num - building_num_range[0]) / (building_num_range[1] - building_num_range[0])
    t_pt_proportion = 1 - f_pt_proportion
    street_center_position = np.average(np.array(start_end_coordinates), weights=(f_pt_proportion, t_pt_proportion), axis=0).tolist()

  offset_direction = add_degree_to_azimuth(segment_direction, -90) if ((odd_on == 'left' and is_odd) or (odd_on == 'right' and not is_odd)) else add_degree_to_azimuth(segment_direction, 90)

  target_position = geod.Direct(*street_center_position, offset_direction, offset_from_road_center)
  target_point = list([round(target_position['lat2'], 6), round(target_position['lon2'], 6)])

  return target_point


global street_name_matching_mapping
street_name_matching_mapping = {}

while True:
  data_folder_choice = input('NYU or Columbia Data Folder?').lower()
  if data_folder_choice.startswith('n'):
    data_folder_path = '/content/drive/Shareddrives/Humanities Research Lab - Shanghai/colab_playground/playground_data'
    break
  elif data_folder_choice.startswith('c'):
    data_folder_path = '/content/drive/MyDrive/HNYC_2022/geocoder/data'
    break
  else:
    pass

street_segment_df = pd.read_csv(data_folder_path +'/'+ 'hnyc_street_segment_1910_v20211125.csv', converters={'building_num_range': ast.literal_eval, 'start_end_coordinates': ast.literal_eval})
global unique_street_names
unique_street_names = street_segment_df['street_name'].dropna().unique().tolist()

def get_addr_coordinates(addr):

  target_street_name, target_building_num = ' '.join(addr.split()[:-1]), addr.split()[-1]

  global unique_street_names
  global street_name_matching_mapping

  if target_street_name in unique_street_names:
    matched_target_street_name = target_street_name
  else:
    matched_target_street_name = street_name_matching_mapping.get(target_street_name, np.nan)
    if not isinstance(matched_target_street_name, str):
      best_street_name_matches = process.extractBests(target_street_name, unique_street_names, scorer=fuzz.ratio, score_cutoff=80)
      if len(best_street_name_matches) > 0:
        matched_target_street_name = best_street_name_matches[0][0]
        street_name_matching_mapping[target_street_name] = matched_target_street_name
      else:
        return (np.nan, np.nan)

  target_building_num = int(target_building_num)
  matched = street_segment_df[street_segment_df['street_name'] == matched_target_street_name].copy().apply(lambda row: get_coordinates_from_details(target_building_num, row['building_num_range'], row['start_end_coordinates'], row['segment_direction'], row['odd_on'], row['offset_from_road_center']) if target_building_num >= row['building_num_range'][0] and target_building_num <= row['building_num_range'][1] else np.nan, axis=1).dropna()
  if len(matched) > 0:
    target_coords = tuple(matched.tolist()[0])
    return (target_coords, matched_target_street_name)
  return (np.nan, np.nan)


def historical_geocode(df, street_segment_df):
  df[['coordinates', 'matched_street_name']] = pd.DataFrame(df['address_for_geocoding'].progress_apply(get_addr_coordinates).tolist(), index=df.index)
  return df


def add_back_non_geocodable_part(df, raw):
  return df.append(raw[~raw['FID'].isin(df['FID'])], ignore_index=True).sort_values('FID')


def get_gecoded_nyc_chinese_dataset():
  df, raw = load_and_prep_address_of_nyc_chinese_dataset()
  geocoded_df = historical_geocode(df, street_segment_df)
  df = add_back_non_geocodable_part(geocoded_df, raw)
  return df


# pip install folium
import folium

def show_map(data, tile_style='bw'):

  orig_data_length = len(data)
  data = data.dropna(subset=['coordinates'])
  print(orig_data_length - len(data), 'out of', orig_data_length, 'records are dropped due to the lack of cooridnates data.\nThese addresses may not be clear or may locate outside Manhattan, which is the scope of the current historical geocoder.\n')

  tiles = 'Stamen Toner' if tile_style == 'bw' else 'https://maps.nyc.gov/xyz/1.0.0/photo/1924/{z}/{x}/{y}.png8' if tile_style == 'aerial' else 'bw'

  map_center = np.array(data['coordinates'].tolist()).mean(axis=0)
  m = folium.Map(location=map_center, tiles=tiles, attr='... contributors [to be updated for public version]', control_scale=True)

  place_type_to_color_mapping = {'Home': 'green', 'Restaurant': 'orangered', 'Non-restaurant Business': 'blue'}

  for _, row in data.iterrows():
    folium.Circle(row['coordinates'], radius=1, color=place_type_to_color_mapping.get(row['place_type'], 'gold'), tooltip=f"Name: {row['Name FULL']}<br>Address: {row['Address']}<br>Type: {row['place_type']}<br>HBCR: {row['HBCR']}<br>Year: {int(row['YR'])}<br>FID: {int(row['FID'])}").add_to(m)

  return m

def get_gecoded_business_directory_dataset():
  df = pd.read_csv(data_folder_path +'/'+ 'business_directory_data_partially_geocoded_v20220205.csv')
  df['coordinates'] = df['coordinates'].fillna('np.nan').apply(eval)
  return df

def show_map_for_bus_dir(data, tile_style='bw'):

  orig_data_length = len(data)
  data = data.dropna(subset=['coordinates'])
  print(orig_data_length - len(data), 'out of', orig_data_length, 'records are dropped due to the lack of cooridnates data.\nThese addresses may not be clear or may locate outside Manhattan, which is the scope of the current historical geocoder.\n')

  tiles = 'Stamen Toner' if tile_style == 'bw' else 'https://maps.nyc.gov/xyz/1.0.0/photo/1924/{z}/{x}/{y}.png8' if tile_style == 'aerial' else 'bw'

  map_center = np.array(data['coordinates'].tolist()).mean(axis=0)
  m = folium.Map(location=map_center, tiles=tiles, attr='... contributors [to be updated for public version]', control_scale=True)

  for _, row in data.iterrows():
    folium.Circle(row['coordinates'], radius=1, color='orangered', tooltip=f"Name: {row['Name']}<br>Address: {row['Address']}<br>Year: {int(row['YR'])}<br>ID: {row['UUID']}").add_to(m)

  return m
