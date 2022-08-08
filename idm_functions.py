import os
os.system('pip list >> installed_libraries.txt')
with open('/content/installed_libraries.txt','r') as f:
  installed_libraries_string = f.read()
installed_libraries = [line.split()[0] for line in installed_libraries_string.split('\n')[2:] if line!='']

from IPython.core.display import clear_output

clear_output()

# import warnings
# warnings.filterwarnings('ignore')

import time
print('Loading packages, this may take a minute ...')
time.sleep(1)
from datetime import datetime, timedelta
from random import random

import pandas as pd
import numpy as np
import re

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon as mpb_polygon
plt.rcParams["font.serif"] = "cmr10" 

from sklearn.cluster import KMeans

import ast
import shutil
from glob import glob
import gc

import pickle
import json

from collections import Counter

import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"

import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from io import BytesIO

from shapely.geometry import Polygon as shapely_polygon
from shapely.geometry import Point as shapely_point
from shapely.geometry import LineString as shapely_line_string
from shapely.validation import make_valid as shapely_make_valid

if 'rasterio' not in installed_libraries:
  os.system('pip install rasterio')
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.plot import reshape_as_image

if 'tqdm' not in installed_libraries:
  os.system('pip install tqdm')
from tqdm import tqdm
tqdm.pandas()

if 'pyproj' not in installed_libraries:
  os.system('pip install pyproj')
import pyproj
from pyproj import Geod

if 'iteround' not in installed_libraries:
  os.system('pip install iteround')
from iteround import saferound

if 'colorsys' not in installed_libraries:
  os.system('pip install colorsys')
import colorsys

if 'unidecode' not in installed_libraries:
  os.system('pip install unidecode')
from unidecode import unidecode

if 'sklearn' not in installed_libraries:
  os.system('pip install sklearn')
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression


if 'python-Levenshtein' not in installed_libraries:
  os.system('pip install python-Levenshtein')
if 'thefuzz' not in installed_libraries:
  os.system('pip install thefuzz')
if 'networkx' not in installed_libraries:
  os.system('pip install networkx')
from thefuzz import fuzz, process
import networkx as nx

if 'fiona' not in installed_libraries:
  os.system('pip install fiona')
import fiona

import warnings
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

clear_output()

#======================================== COMMON UTILS ============================================#

def flatten_list(l):
  return [item for sublist in l for item in sublist]

def unique_preserving_order(l):
    """
    Reference:http://www.peterbe.com/plog/uniqifiers-benchmark
    """
    seen = set()
    seen_add = seen.add
    return [x for x in l if not (x in seen or seen_add(x))]

def get_outliers(data_names, data_values):
  data_names = np.array(data_names)
  data_values = np.array(data_values)
  upper_hinge = np.percentile(data_values,75)
  lower_hinge = np.percentile(data_values,25)
  iqr = upper_hinge - lower_hinge
  outlier_mask = (data_values > upper_hinge + 1.5*iqr) | (data_values < lower_hinge - 1.5*iqr)
  outlier_data_values = data_values[outlier_mask]
  outlier_data_names = data_names[outlier_mask]
  return dict(zip(outlier_data_names, outlier_data_values))

def get_non_single_elements(data, field):
  value_cnts = data[field].value_counts()
  non_single_elements = value_cnts[value_cnts>=2].index.tolist()
  return non_single_elements

def try_length_is_zero(x):
  try:
    if isinstance(x,int) or isinstance(x,float):
      return False
    return len(x)==0
  except:
    return True

def create_mapping_from_df(dataframe, key, value, drop_nan_value = True, drop_empty_value = True):
  temp_df = dataframe[[key,value]].copy()
  if drop_empty_value:
    temp_df[value] = temp_df[value].apply(lambda x: np.nan if try_length_is_zero(x) else x)
    drop_nan_value = True
  if drop_nan_value:
    temp_df = temp_df.dropna()
  return temp_df.set_index(key)[value].to_dict()

remove_special_characters_and_shrink_whitespace = lambda x: re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', x.lower())).strip() if isinstance(x,str) else ''
remove_bracket_x_pattern = lambda x: '' if re.sub(r'(\[)?x+(\])?','',x)=='' else x

#==================================================================================================#






#=========================================== MS OCR ===============================================#

need_ocr_call = input('\nDo you need OCR calls? [y/n]\n')
if need_ocr_call[0].lower() == 'y':
  os.system('pip install --upgrade azure-cognitiveservices-vision-computervision')
  clear_output()
  from azure.cognitiveservices.vision.computervision import ComputerVisionClient
  from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
  from msrest.authentication import CognitiveServicesCredentials
  computervision_client = ComputerVisionClient(input('\nEndpoint?\n'), CognitiveServicesCredentials(input('\nKey?\n')))
clear_output()

def get_ms_ocr_result(read_image_path, wait_interval=10): 

  # Open the image
  if read_image_path.endswith('.jp2'):
    image = cv2.imread(read_image_path)
    max_dimension = max(image.shape[:2])
    if max_dimension > 10000:
        target_shape = tuple(np.int0(np.floor(np.array(image.shape[:2])[::-1] / max_dimension * 10000)))
        image = cv2.resize(image, target_shape, interpolation=cv2.INTER_AREA)
    success, encoded_image = cv2.imencode('.jpg', image)
    read_image = BytesIO(encoded_image.tobytes())
  else:
    read_image = open(read_image_path, "rb")

  # Call API with image and raw response (allows you to get the operation location)
  read_response = computervision_client.read_in_stream(read_image, raw=True)
  # Get the operation location (URL with ID as last appendage)
  read_operation_location = read_response.headers["Operation-Location"]
  # Take the ID off and use to get results
  operation_id = read_operation_location.split("/")[-1]
  # Call the "GET" API and wait for the retrieval of the results
  while True:
    read_result = computervision_client.get_read_result(operation_id)
    if read_result.status.lower() not in ['notstarted', 'running']:
      break
    # print('Waiting for result...')
    time.sleep(wait_interval)
  return read_result.as_dict()

def parse_ms_ocr_result(ms_ocr_result, return_words=True, confidence_threshold=0):

  operation_result = ms_ocr_result['status']
  operation_creation_time = ms_ocr_result['created_date_time']
  operation_last_update_time = ms_ocr_result['last_updated_date_time']
  operation_api_version = ms_ocr_result['analyze_result']['version']
  operation_model_versoin = ms_ocr_result['analyze_result']['model_version']

  assert(len(ms_ocr_result['analyze_result']['read_results']) == 1)
  read_result = ms_ocr_result['analyze_result']['read_results'][0]

  result_page_num = read_result['page']
  result_angle = read_result['angle']
  result_width = read_result['width']
  result_height = read_result['height']
  result_unit = read_result['unit']
  result_lines = read_result['lines']

  if len(result_lines) == 0:  # if no lines found, return an empty components_df directly
    return pd.DataFrame(columns=['bounding_box', 'text', 'confidence', 'frame_anchor'])

  lines_df = pd.DataFrame(result_lines)

  if return_words:
    words_df = pd.DataFrame(flatten_list(lines_df['words']))
    words_df = words_df[words_df['confidence'] >= confidence_threshold]
    components_df = words_df.reset_index(drop=True)
  else:
    components_df = lines_df

  return components_df

def mark_ms_ocr_result(input_image_filepath, components_df, output_image_filepath='', fontsize=10, figsize=(20,20), dpi=150, clear_plot=False):

  components_df = components_df.copy()
  
  image = Image.open(input_image_filepath)

  plt.figure(figsize=figsize, dpi=dpi)
  ax = plt.imshow(image, cmap=cm.gray)

  for _, row in components_df.iterrows():

    bbox, ocr_text, confidence, right_side_center = row['bounding_box'], row['text'], row['confidence'], row.get('bbox_right_side_center',None)
    
    # bounding box
    if all([isinstance(x,list) and len(x)==2 for x in bbox]):
      vertices = bbox
    else:
      vertices = [(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)]
    
    polygon_patch = mpb_polygon(vertices, closed=True, fill=False, linewidth=0.6, color='b', alpha=confidence)
    ax.axes.add_patch(polygon_patch)
    
    # text
    plt.text(vertices[1][0], vertices[1][1], ocr_text.rstrip('$'), fontsize=fontsize, color='r', va="top")

    if right_side_center is not None:
      # right side center dot
      plt.text(right_side_center[0], right_side_center[1], '.', fontsize=8, color='#66FF66', ha='left', va="baseline")

  if output_image_filepath != '':
    plt.savefig(output_image_filepath, bbox_inches='tight', pad_inches=0)

  if clear_plot: ## usefulness to be tested, created to release memory during batch plotting
    # Clear the current axes
    plt.cla() 
    # Clear the current figure
    plt.clf() 
    # Closes all the windows
    plt.close('all')   
    del ax
    gc.collect()


def save_dict_to_json(dic, filepath):
  with open(filepath, 'w') as f:
    json.dump(dic, f)

def read_dict_from_json(filepath):
  with open(filepath, 'r') as f:
    dic = json.load(f)
  return dic

############# QUICK ACCESS OCR #############

def ms_ocr(img_path, mark_image = True, show_numeric = False, fontsize = 10, figsize = (20,20), dpi = 150, clear_plot=False, wait_interval = 10):

  raw_ocr_result_filepath = img_path.split('.')[0] + '_raw_ocr_result.txt'
  if not os.path.exists(raw_ocr_result_filepath):
    result = get_ms_ocr_result(img_path, wait_interval = wait_interval)
    save_dict_to_json(result, raw_ocr_result_filepath)
  else:
    print('Raw OCR result found.')
    result = None

  ocr_result_table_filepath = img_path.split('.')[0] + '_ocr_result_table.csv'
  if not os.path.exists(ocr_result_table_filepath):
    if result is None:
      result = read_dict_from_json(raw_ocr_result_filepath)
    comp_df = parse_ms_ocr_result(result)
    comp_df.to_csv(ocr_result_table_filepath, index=False)
  else:
    print('OCR result table found.')

  if mark_image:
    comp_df = pd.read_csv(ocr_result_table_filepath)
    comp_df['bounding_box'] = comp_df['bounding_box'].apply(ast.literal_eval)
    if not show_numeric:
      comp_df = comp_df[~(comp_df['text'].str.isnumeric())]
    ocr_result_marked_img_path = img_path.split('.')[0] + '_ocr_result_marked_img.' + img_path.split('.')[1]
    mark_ms_ocr_result(img_path, comp_df, output_image_filepath=ocr_result_marked_img_path, fontsize=fontsize, figsize=figsize, dpi=dpi, clear_plot=clear_plot)

############# UTILS FOR BASCI PREPROCESSING BEFORE OCR #############

def make_greyscale_img(img_path):
  out_img_path = img_path.replace('.', '_grey.')
  if not os.path.exists(out_img_path):
    grey_img = cv2.imread(img_path, 0)
    im = Image.fromarray(grey_img)
    im.save(out_img_path)
    return out_img_path
  else:
    print('Greyscale version of the image already exists.')
    return out_img_path


def resize_img(img_path, target_size):
  img = cv2.imread(img_path)
  img_shape = img.shape
  if img_shape[0] != img_shape[1]:
    raise '[Error] Input image not a square shape.'
  orig_size = img_shape[0]
  out_img_path = img_path.replace('.', '_' + str(orig_size) + '_to_' + str(target_size) + '.')
  out_img_path = re.sub(r'(000)(_|\.)', r'k\2', out_img_path)
  if not os.path.exists(out_img_path):
    resized_img = cv2.resize(img, (target_size, target_size))
    if len(img_shape) == 3:
      resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(resized_img)
    im.save(out_img_path)
    return out_img_path
  else:
    print('Resized version of the image already exists.')
    return out_img_path

def rotate_180_degree(img_filepath):
  img = cv2.imread(img_filepath)[:,:,:3]
  rotated_img = cv2.flip(cv2.flip(img, 0), 1)
  rotated_img_path = '/'.join(img_filepath.split('/')[:-1])+'/'+img_filepath.split('/')[-1].split('.')[0]+'__rotated.png'
  cv2.imwrite(rotated_img_path, rotated_img)
  return rotated_img_path

def invert_binary(img):
  return cv2.bitwise_not(img)
def adaptive_threshold(img, size = 9, C = 9):
  return cv2.adaptiveThreshold(img.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, size, C)
def otsu_threshold(img, verbose=False):
  threshold_value, binarized_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  if verbose:
    print('The threshold value found is', threshold_value)
  return binarized_img

#==================================================================================================#










#=============================== COLOR AND COLOR CODE CONVERSION ==================================#

def rgb_to_grey(img):
  return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def bgr_to_grey(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def bgr_to_rgb(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def rgb_to_bgr(img):
  return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
def grey_to_bgr(img):
  return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
def grey_to_rgb(img):
  return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
def hsv2rgb(h,s=1.0,v=1.0):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))
def hsv2bgr(h,s=1.0,v=1.0):
  r,g,b = hsv2rgb(h,s,v)
  return (b,g,r)

def rgb_code_to_lab_code(rgb_tuple):
  pixel = np.zeros((1,1,3),dtype=np.uint8)
  pixel[0,0,:] = rgb_tuple if isinstance(rgb_tuple,tuple) else tuple(rgb_tuple)
  return tuple([int(v) for v in list(cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0][0])])
def lab_code_to_rgb_code(lab_tuple):
  pixel = np.zeros((1,1,3),dtype=np.uint8)
  pixel[0,0,:] = lab_tuple if isinstance(lab_tuple,tuple) else tuple(lab_tuple)
  return tuple([int(v) for v in list(cv2.cvtColor(pixel, cv2.COLOR_LAB2RGB)[0][0])])
 
def rgb_code_to_hsv_code(rgb_tuple):
  pixel = np.zeros((1,1,3),dtype=np.uint8)
  pixel[0,0,:] = rgb_tuple
  return tuple([int(v) for v in list(cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0][0])])

def hsv_code_to_rgb_code(hsv_tuple):
  pixel = np.zeros((1,1,3),dtype=np.uint8)
  pixel[0,0,:] = hsv_tuple
  return tuple([int(v) for v in list(cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0][0])])

def hex_code_to_rgb_code(hex_code):
    # reference: https://stackoverflow.com/a/29643643
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def hex_code_to_hsv_code(hex_tuple):
  return hex_code_to_rgb_code(rgb_code_to_hsv_code(hex_tuple))


# def check_within_range(code, lower_range, upper_range):
#     a,b,c = code
#     a_low, b_low, c_low = lower_range
#     a_high, b_high, c_high = upper_range
#     return a_low <= a and a <= a_high and b_low <= b and b <= b_high and c_low <= c and c <= c_high

# for key in exemplar_hsv_code_to_color_name_mapping.keys():
#     lower_range, upper_range = create_range_around_hsv_code(key, radius= (3,9,9) if exemplar_hsv_code_to_color_name_mapping[key]=='brown' else (10,10,10))
#     if lower_range[0]>upper_range[0]:
#         if check_within_range( hsv_code, lower_range, (180,upper_range[1],upper_range[2]) ) or check_within_range( hsv_code, (0,lower_range[1],lower_range[2]), upper_range ) :
#             return exemplar_hsv_code_to_color_name_mapping[key]
#     else:
#         if check_within_range( hsv_code, lower_range, upper_range ):
#             return exemplar_hsv_code_to_color_name_mapping[key]

#==================================================================================================#










#===================================== GET BBOX FEATURES ==========================================#

###### UTILS FOR GET_BBOX_FEATURES ######
def euc_dist(pt1, pt2, rounding = 1):
  return np.round(np.linalg.norm(pt2-pt1), rounding)
  
def get_bbox_features(bbox):

  widths = euc_dist(bbox[0], bbox[1]), euc_dist(bbox[2], bbox[3])
  width = np.mean(widths)
  diff_in_widths = np.abs(np.diff(widths))
  width_diff_prop = np.round((diff_in_widths/width)[0], 2)

  heights = [ dist_from_point_to_line(bbox[2], bbox[3], bbox[0]),  dist_from_point_to_line(bbox[2], bbox[3], bbox[1]),  dist_from_point_to_line(bbox[0], bbox[1], bbox[2]),  dist_from_point_to_line(bbox[0], bbox[1], bbox[3]) ] 
  height = np.mean(heights)
  diff_in_heights = np.abs(np.diff(heights))
  height_diff_prop = np.round((diff_in_heights/height)[0], 2)

  left_side_center = np.mean((bbox[0], bbox[3]), axis=0)
  right_side_center = np.mean((bbox[1], bbox[2]), axis=0)
  reading_direction_vector = right_side_center - left_side_center
  reading_direction = get_vector_direction(reading_direction_vector)

  center = bbox.mean(axis=0)

  return width, width_diff_prop, height, height_diff_prop, reading_direction, center, left_side_center, right_side_center

def add_bbox_features_to_table(map_ocr_results_table, ocr_entry_id_start = 1, id_column = 'map_id'):

  # Calculate bounding box features and add them as columns
  map_ocr_results_table[['bbox_'+col for col in 'width, width_diff_prop, height, height_diff_prop, reading_direction, center, left_side_center, right_side_center'.split(', ')]] = pd.DataFrame(map_ocr_results_table['bounding_box'].apply(get_bbox_features).tolist(), index = map_ocr_results_table.index)
  # Convert numpy ndarray column to nested list format for better I/O from/to CSV
  map_ocr_results_table['bounding_box'] = map_ocr_results_table['bounding_box'].apply(lambda x: x.tolist())
  map_ocr_results_table['bbox_center'] = map_ocr_results_table['bbox_center'].apply(lambda x: x.tolist())
  map_ocr_results_table['bbox_left_side_center'] = map_ocr_results_table['bbox_left_side_center'].apply(lambda x: x.tolist())
  map_ocr_results_table['bbox_right_side_center'] = map_ocr_results_table['bbox_right_side_center'].apply(lambda x: x.tolist())

  # Add a unique id for each ocr entry (each row in the table), this is unique within the original full image/map.
  map_ocr_results_table['ocr_entry_id'] = range(ocr_entry_id_start, ocr_entry_id_start + len(map_ocr_results_table))

  output_columns = [id_column, 'ocr_entry_id', 'text', 'confidence', 'bounding_box', 'bbox_width', 'bbox_width_diff_prop', 'bbox_height', 'bbox_height_diff_prop', 'bbox_reading_direction',  'bbox_center', 'bbox_left_side_center', 'bbox_right_side_center']  
  return map_ocr_results_table[output_columns]

def cv2_contourize(np_bbox): 
  return np.array(np_bbox).reshape((-1,1,2)).astype(np.int32)

#==================================================================================================#


#======================================== CUT / CROP / RE-COMBINE IMAGE ========================================#

############# UTILS FOR CUT / CROP IMAGE #############

def get_area_size_from_geo_point_list(geo_point_list):
  """Input: geo_point_list in format of [(lon, lat), ...]
  Output: size of area in m^2
  # Reference: https://stackoverflow.com/questions/68118907/shapely-pyproj-find-area-in-m2-of-a-polygon-created-from-latitude-and-longi
  """
  polygon = shapely_polygon(geo_point_list)
  geod = Geod(ellps="WGS84")
  try:
    poly_area, poly_perimeter = geod.geometry_area_perimeter(polygon)
    return int(poly_area)
  except:
    return -1

def calculate_area_per_pixel(raw_image_filepath):
  if raw_image_filepath.endswith('.tif'):
    dataset = rasterio.open(raw_image_filepath)
    transform_matrix = dataset.transform
  elif raw_image_filepath.endswith('.png'):
    dataset = cv2.imread(raw_image_filepath)
    transform_matrix = create_transform_matrix(read_geotransform_parameters(raw_image_filepath.replace('.png','.png.aux.xml')))

  h, w = dataset.shape[0], dataset.shape[1]
  image_pixel_area_size = w*h
  corner_geo_point_list = [tuple([float(x) for x in np.round(transform_matrix * point,6)]) for point in [(0,0), (0,h), (w,h), (w,0), (0,0)]]

  image_geo_area_size = get_area_size_from_geo_point_list(corner_geo_point_list)

  area_per_pixel = image_geo_area_size/image_pixel_area_size
  area_per_pixel = round(area_per_pixel,6)
  return area_per_pixel

def calculate_area_per_pixel_list(raw_image_filepaths, verbose = False):
  area_per_pixel_list = []
  for raw_image_index in tqdm(range(0, len(raw_image_filepaths))):
    raw_image_filepath = raw_image_filepaths[raw_image_index]
    if verbose:
      print('------------------- raw image index '+str(raw_image_index)+' -------------------')
      print(raw_image_filepath.split('/')[-1])
    area_per_pixel = calculate_area_per_pixel(raw_image_filepath)
    if verbose:
      print('area_per_pixel:', area_per_pixel)
    area_per_pixel_list.append(area_per_pixel)
  return area_per_pixel_list

def read_geotransform_parameters(aux_xml_filepath):
  with open(aux_xml_filepath, 'r') as f:
    raw = f.read()
  geotransform_parameters = [ast.literal_eval(v.strip()) for v in re.findall('<GeoTransform>(.*?)</GeoTransform>',raw)[0].split(',')]
  return geotransform_parameters

def create_transform_matrix(para):
  transform_matrix = Affine(para[1], para[2], para[0], para[4], para[5] , para[3])
  return transform_matrix

######################################################

def cut_tiff_into_pngs(path, window_side_length, window_stride = None, output_directory_path = None, skip_if_directory_exists = False):

  if window_stride is None:
    window_stride = window_side_length - window_side_length//5

  dataset = rasterio.open(path)
  dataset_name = dataset.name.split('/')[-1].split('.')[0]
  if output_directory_path is None:
    output_directory_path = '/'.join(path.split('/')[:-1]) +'/'+ dataset_name+'__wsl_'+str(window_side_length)+'_ws_'+str(window_stride)
  else:
    output_directory_path = output_directory_path.rstrip('/')

  if not os.path.exists(output_directory_path):
    os.mkdir(output_directory_path)
  else:
    if skip_if_directory_exists:
      return []
    else:
      raise ValueError("[Error] Output directory already exists, please check and resolve.")

  area_per_pixel = calculate_area_per_pixel(path)
  
  dataset_meta = dict(dataset.meta)
  dataset_meta['area_per_pixel'] = area_per_pixel
  print(dataset_meta)

  dataset_meta_string = str(dataset_meta)
  dataset_meta_string = dataset_meta_string.replace('\n','')
  dataset_meta_string = re.sub(r'CRS\.from_epsg\((\d+)\)',r"'epsg:\1'",dataset_meta_string)
  dataset_meta_string = re.sub(r'Affine\((.*?)\)',r'[\1]',dataset_meta_string)
  
  with open(output_directory_path +'/'+ 'metadata.txt', 'w') as f:
    f.write(dataset_meta_string)

  dataset_band_count = dataset.count
  dataset_width = dataset.width
  dataset_height = dataset.height
  window_width_indices = range(dataset_width//window_stride+1)
  window_height_indices = range(dataset_height//window_stride+1)
  
  offset_pair_list = []
  for window_width_index in window_width_indices:
    col_off = window_width_index * window_stride
    col_off = col_off if col_off + window_side_length <= dataset_width else dataset_width - window_side_length
    for window_height_index in window_height_indices:
      row_off = window_height_index * window_stride
      row_off = row_off if row_off + window_side_length <= dataset_height else dataset_height - window_side_length
      offset_pair_list.append((col_off, row_off))

  cropped_image_filepath_list = []
  for col_off, row_off in tqdm(offset_pair_list):
    bands = []
    for band_index in range(1,dataset_band_count+1):
      bands.append( dataset.read(band_index, window = Window(col_off, row_off, window_side_length, window_side_length)) )
    window_img = np.stack(bands, axis=2)
    
    # check to ensure image is not completely empty
    if window_img[:,:,0].sum() != 0:
      im = Image.fromarray(window_img)
      output_path = output_directory_path +'/'+ dataset_name+'_xoff'+str(col_off)+'_yoff'+str(row_off)+'_wsl_'+str(window_side_length)+'.png'
      im.save(output_path)

      # check if the border of the image is transparent on all 4 sides, meaning the actual raster data is fully contained
      if window_img[0,:,0].sum() + window_img[:,0,0].sum() + window_img[-1,:,0].sum() + window_img[:,-1,0].sum() == 0:
        for p in cropped_image_filepath_list:
          os.remove(p)
        cropped_image_filepath_list = [output_path]
        break # if so, break out out loop and return the current crop only

      cropped_image_filepath_list.append(output_path)
      
  cropped_image_filepath_list = sorted(set(cropped_image_filepath_list))
  return cropped_image_filepath_list

######################################################

def cut_image_into_pngs(path, window_side_length, window_stride = None, output_directory_path = None, skip_if_directory_exists = False, georeferenced = False):

  dataset_name = path.split('/')[-1].split('.')[0]

  if window_stride is None:
    window_stride = window_side_length - window_side_length//5

  im_bgr = cv2.imread(path)
  im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

  if output_directory_path is None:
    output_directory_path = '/'.join(path.split('/')[:-1]) +'/'+ dataset_name+'__wsl_'+str(window_side_length)+'_ws_'+str(window_stride)
  else:
    output_directory_path = output_directory_path.rstrip('/')

  if not os.path.exists(output_directory_path):
    os.mkdir(output_directory_path)
  else:
    if skip_if_directory_exists:
      return []
    else:
      raise ValueError("[Error] Output directory already exists, please check and resolve.")

  dataset_height, dataset_width, dataset_band_count = im_rgb.shape

  file_extension = path.split('.')[-1]
  aux_xml_filepath = path.replace('.'+file_extension,'.'+file_extension+'.aux.xml')
  if georeferenced and os.path.exists(aux_xml_filepath) and aux_xml_filepath != path:
    affine_info_string = repr(create_transform_matrix(read_geotransform_parameters(aux_xml_filepath)))
    area_per_pixel = calculate_area_per_pixel(path)
    dataset_meta_string = "{'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': "+str(dataset_width)+", 'height': "+str(dataset_height)+", 'count': 4, 'crs': CRS.from_epsg(4326), 'transform': "+affine_info_string+", 'area_per_pixel':"+str(area_per_pixel)+"}"
    print(dataset_meta_string)
    dataset_meta_string = str(dataset_meta)
    dataset_meta_string = dataset_meta_string.replace('\n','')
    dataset_meta_string = re.sub(r'CRS\.from_epsg\((\d+)\)',r"'epsg:\1'",dataset_meta_string)
    dataset_meta_string = re.sub(r'Affine\((.*?)\)',r'[\1]',dataset_meta_string)
    with open(output_directory_path +'/'+ 'metadata.txt', 'w') as f:
      f.write(dataset_meta_string)
  
  window_width_indices = range(dataset_width//window_stride+1)
  window_height_indices = range(dataset_height//window_stride+1)

  offset_pair_list = []
  for window_width_index in window_width_indices:
    col_off = window_width_index * window_stride
    col_off = col_off if col_off + window_side_length <= dataset_width else dataset_width - window_side_length
    for window_height_index in window_height_indices:
      row_off = window_height_index * window_stride
      row_off = row_off if row_off + window_side_length <= dataset_height else dataset_height - window_side_length
      offset_pair_list.append((col_off, row_off))

  cropped_image_filepath_list = []

  for col_off, row_off in tqdm(offset_pair_list):
    window_img = im_rgb[row_off:row_off+window_side_length, col_off:col_off+window_side_length, :]
      
    # check to ensure image is not completely empty
    if window_img[:,:,0].sum() != 0:
      im = Image.fromarray(window_img)
      output_path = output_directory_path +'/'+ dataset_name+'_xoff'+str(col_off)+'_yoff'+str(row_off)+'_wsl_'+str(window_side_length)+'.png'
      im.save(output_path)

      # check if the border of the image is transparent on all 4 sides, meaning the actual raster data is fully contained
      if window_img[0,:,0].sum() + window_img[:,0,0].sum() + window_img[-1,:,0].sum() + window_img[:,-1,0].sum() == 0:
        for p in cropped_image_filepath_list:
          os.remove(p)
        cropped_image_filepath_list = [output_path]
        break # if so, break out out loop and return the current crop only

      cropped_image_filepath_list.append(output_path)

  cropped_image_filepath_list = sorted(set(cropped_image_filepath_list))
  return cropped_image_filepath_list  

######################################################

def combine_relative_tables(relative_table_filepath_list, rotated = False):

  map_ocr_results_table = pd.DataFrame()
  for ocr_result_table_path in relative_table_filepath_list:
    ### For each crop of map ###
    table = pd.read_csv(ocr_result_table_path)
    # Take data in the filename and put them into the table
    table_filename = ocr_result_table_path.split('/')[-1]
    map_id = int(table_filename.split('_')[0])
    wsl = int(re.findall('_wsl_(\d+)_',table_filename)[0])
    table['map_id'] = map_id
    table['x_offset'] = int(re.findall('_xoff(\d+)_',table_filename)[0])
    table['y_offset'] = int(re.findall('_yoff(\d+)_',table_filename)[0])
    table['wsl'] = wsl
    map_ocr_results_table = map_ocr_results_table.append(table, ignore_index=True)

  # Rename "bounding box" column in the raw table to be "relative bounding box"
  map_ocr_results_table = map_ocr_results_table.rename(columns={'bounding_box':'relative_bounding_box'})

  # Evaluate relative bounding box, string -> list
  map_ocr_results_table['relative_bounding_box'] = map_ocr_results_table['relative_bounding_box'].apply(ast.literal_eval)

  if rotated:
    map_ocr_results_table['relative_bounding_box'] =   map_ocr_results_table['relative_bounding_box'].apply(lambda li: [[wsl - li[0], wsl - li[1]], [wsl - li[2], wsl - li[3]], [wsl - li[4], wsl - li[5]], [wsl - li[6], wsl - li[7]]])

  # Add x and y offsets so that "relative bounding box" become "bounding box" in the original full image/map
  map_ocr_results_table['bounding_box'] = map_ocr_results_table[['relative_bounding_box','x_offset','y_offset']].apply(lambda row: np.array(row['relative_bounding_box']).reshape(-1,2) + np.array((row['x_offset'], row['y_offset'])), axis=1)

  return map_ocr_results_table

#==================================================================================================#





#======================================== COLOR ANALYZER ==========================================#


def check_hsv_for_criteria(hsv_code, h_range, s_range, v_range):
  h, s, v = hsv_code
  return (h_range[0] <= h)&(h <= h_range[1])   &   (s_range[0] <= s)&(s <= s_range[1])   &   (v_range[0] <= v)&(v <= v_range[1])

def check_ocr_crop_background_color(img, bbox_contour, h_range, s_range, v_range):
  ocr_crop = mask_with_contours(img, [bbox_contour])
  primary_color, secondary_color = analyze_color(ocr_crop, n_cluster=2, plot_bar=False, return_hsv=True)
  return check_hsv_for_criteria(primary_color, h_range, s_range, v_range)

#==================================================================================================#






#======================================== MPL IMAGE SAVING ========================================#

def imsave(img, filename):
  plt.imsave(filename, img, cmap=cm.gray)

def get_w_h_ratio(img):
  return img.shape[1]/img.shape[0]
  
def imshow(img, width = None, height = None, dpi = 90, title = None, no_axis = False):
    
    w_h_ratio = img.shape[1]/img.shape[0]

    if width is not None:
        plt.figure(figsize=(width,round(width/w_h_ratio,1)), dpi=dpi)
    elif height is not None:
        plt.figure(figsize=(round(height*w_h_ratio,1),height), dpi=dpi)
    else:
        if w_h_ratio < 0.8:
            width = 3
            plt.figure(figsize=(width,round(width/w_h_ratio,1)), dpi=dpi)
        else:
            height = 3
            plt.figure(figsize=(round(height*w_h_ratio,1),height), dpi=dpi)
    
    plt.grid(False)
    if no_axis:
        plt.axis('off')

    if len(img.shape)==2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(bgr_to_rgb(img))

    if title is not None:
        plt.title(title)

def save_graph(filename = '', dpi = 150, padding = 0.3, transparent = False, add_title = False, folder = None):

  orig_filename = filename

  if not os.path.exists('saved_graphs'):
    os.mkdir('saved_graphs')

  if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png'):
    filename = filename+'.png'

  if filename == '':
    image_paths = [p.split('/')[-1].split('.')[0] for p in glob('./saved_graphs/*')]
    available_indices = [int(p) for p in image_paths if p.isnumeric()]
    if len(available_indices) == 0:
      next_index = 1
    else:
      next_index = max(available_indices)+1
    filename = str(next_index).zfill(2)+'.png'


  if add_title:
    if orig_filename!='':
      plt.suptitle(orig_filename)

  if folder == None:
    plt.savefig('./saved_graphs/'+filename, dpi=dpi, bbox_inches='tight', transparent=transparent, pad_inches=padding)
  else:
    plt.savefig(folder+filename, dpi=dpi, bbox_inches='tight', transparent=transparent, pad_inches=padding)
  print('Graph "'+filename+'" saved.')

#==================================================================================================#








#========================= HIGH LEVEL GEOMETRY BASED FEATURE EXTRACTOR ============================#

###### FIND CONTOURS ######

def create_hierarchy_df(hierarchy):
  # Reference: https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
  hierarchy = hierarchy[0]
  hierarchy_df = pd.DataFrame(hierarchy, columns=['next','prev','child','parent'])
  import networkx as nx
  G = nx.DiGraph()
  for index, row in hierarchy_df.query('parent != -1').iterrows():
    G.add_edge(row['parent'], index)
  root_nodes = hierarchy_df.query('parent == -1 & child != -1').index.tolist()
  node_depth_mapping = {}
  for root_node in root_nodes:
    node_depth_mapping.update( nx.shortest_path_length(G, root_node) )
  hierarchy_df['level'] = hierarchy_df.index.map(node_depth_mapping).fillna(0)
  hierarchy_df['level'] = hierarchy_df['level'].apply(int)
  return hierarchy_df

def get_contour_area_size(cnt):
  try:
    return cv2.contourArea(cnt)
  except TypeError as e:
    return cv2.contourArea(np.float32(cnt))

def find_contours(img, min_area_size = 1000, max_area_size = None, top_k = None, color_mode = 'rainow', border_width = 2, show = True, only_exterior = False, only_lowest_k = None, verbose = True):

  img = img.copy()

  opencv_version = int(cv2.__version__.split('.')[0])

  if only_exterior:

    output = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if opencv_version == 4:
      contours, hierarchy = output
    elif opencv_version == 3:
      _, contours, hierarchy = output
  elif only_lowest_k != None:
    output = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if opencv_version == 4:
      contours, hierarchy = output
    elif opencv_version == 3:
      _, contours, hierarchy = output
    hierarchy_df = create_hierarchy_df(hierarchy)
    contours_indices = hierarchy_df.loc[hierarchy_df['level'].isin(  sorted(hierarchy_df['level'].unique())[:only_lowest_k]  )].index.tolist()
    contours = np.array(contours, dtype=object)[contours_indices]
  else:
    output = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if opencv_version == 4:
      contours, hierarchy = output
    elif opencv_version == 3:
      _, contours, hierarchy = output

  contours = [cnt for cnt in contours if get_contour_area_size(cnt)>=min_area_size]
  if max_area_size != None:
    contours = [cnt for cnt in contours if get_contour_area_size(cnt)<=max_area_size]
  contours = sorted(contours, key=lambda cnt: get_contour_area_size(cnt), reverse = True)
  if verbose:
    print(len(contours),'contours found.')

  if top_k != None:
    print('Showing the',top_k,'largest contours.')
    contours = contours[:top_k]

  if show:

    if color_mode == 'red':

      # Draw contours in red
      colored_img = grey_to_bgr(img)
      for cnt in contours:
        colored_img = cv2.drawContours(colored_img, [cnt], 0, (255,0,0), border_width)
      imshow(colored_img)

    elif color_mode == 'rainbow':
      
      # Draw contours in rainbow colors
      n_colors = 8
      color_range = range(1,n_colors*10+1,n_colors)
      colors = [hsv2rgb(num/100) for num in color_range]

      colored_img = grey_to_bgr(img)

      for i in range(len(contours)):
          cnt = contours[i]
          color = colors[i%len(color_range)]
          colored_img = cv2.drawContours(colored_img , [cnt] , -1, color , border_width)

      imshow(colored_img)
  
  return contours

###### SMOOTH/SIMPLIFY CONTOURS ######
def approximate_contours(contours = [], precision_level = 0.01, border_width = 2, show = False, img = None, epsilon_cap = 3):
  approx_contours = []
  for cnt in contours:
    hull = cv2.convexHull(cnt)
    epsilon = precision_level*cv2.arcLength(hull,True)
    epsilon = min(epsilon, epsilon_cap)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    approx_contours.append(approx)
  approx_contours = [cnt for cnt in approx_contours if len(cnt)>2]
  contour_count_change = len(contours) - len(approx_contours)
  if contour_count_change>0:
    print(str(contour_count_change) + ' contours are dropped because they have less than 3 edges.')
  if show:
    colored_img = grey_to_bgr(img)
    for approx in approx_contours:
      colored_img = cv2.drawContours(colored_img, [approx], 0, (255,0,0), border_width)
    imshow(colored_img)
  return approx_contours

###### SMOOTH/SIMPLIFY CONTOURS (EXPERIMENTAL) ######
def get_movement_direction(x_diff, y_diff):
  return np.arctan2(x_diff,y_diff)/np.pi*180
def get_movement_distance(x_diff, y_diff):
  return np.linalg.norm((x_diff, y_diff))
def angle_diff(sourceA,targetA):
  a = targetA - sourceA
  a = (a + 180) % 360 - 180
  return a
def get_contour_info_df(cnt):
  movements = np.diff(cnt,axis=0)
  directions = [get_movement_direction(*mov[0]) for mov in movements]
  distances = [get_movement_distance(*mov[0]) for mov in movements]
  cnt_info = pd.DataFrame(zip([(i-1,i) for i in range(1,len(movements))],[mov[0] for mov in movements],distances,directions), columns=['point_index','movement','distance','direction'])
  cnt_info['prev_direction'] = cnt_info.direction.shift(1)
  cnt_info['direction_change'] = cnt_info.apply(lambda row: angle_diff(row['prev_direction'],row['direction']), axis=1)
  cnt_info = cnt_info = cnt_info.drop('prev_direction',axis=1)
  return cnt_info
def keep_segments_longer_than(cnt, min_segment_len):
  cnt_info = get_contour_info_df(cnt)
  preserved_point_indices = flatten_list( cnt_info.loc[cnt_info['distance']>=min_segment_len,'point_index'].tolist() )
  preserved_cnt = cnt[preserved_point_indices]
  return preserved_cnt
def get_length_of_segments(cnt, order = 'original'):
  cnt_with_head_at_tail = np.array(list(cnt)+list(cnt[:1]))
  length_of_segments = list(np.round(np.sqrt(np.sum(np.square((cnt_with_head_at_tail[:-1] - cnt_with_head_at_tail[1:])), axis=2)).flatten(),1))
  if order.startswith('orig'):
    return length_of_segments
  elif order.startswith('asc'):
    return sorted(length_of_segments)
  elif order.startswith('des'):
    return sorted(reversed(length_of_segments))
def stop_at_abrupt_change(contours, sudden_change_ratio = 10):
  output_contours = []
  prev_cnt_size = 0
  for cnt in contours:
    cnt_size = cv2.contourArea(cnt)
    if cnt_size < 1e4 or prev_cnt_size/cnt_size > sudden_change_ratio:
      break
    output_contours.append(cnt)
    prev_cnt_size = cnt_size
  return output_contours

###### VISUALIZE CONTOURS ######
def draw_many_contours(img, contours, text_content_list=None, dpi=None, border_width=2, n_colors = 10, border_color = None, font_scale = 1, is_bgr = True, save_not_show = False):
  
  color_range = range(1,n_colors*10+1,n_colors)
  colors = [hsv2bgr(num/100) if is_bgr else hsv2rgb(num/100) for num in color_range]

  if len(img.shape)==2:
    colored_img = grey_to_bgr(img)
  elif len(img.shape)==3:
    if is_bgr:
      colored_img = img.copy()
    else:
      colored_img = rgb_to_bgr(img)

  if text_content_list=='':
    text_content_list = ['']*len(contours)
  elif text_content_list is None:
    text_content_list = range(len(contours))
  else:
    pass

  for i in range(len(contours)):
    cnt = contours[i]
    if border_color is None:
      color = colors[i%n_colors]
    else:
      color = border_color
    colored_img = cv2.drawContours(colored_img, [cnt], 0, color, border_width)


    if font_scale > 0:
      
      M = cv2.moments(cnt)
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
      text_position = (cx, cy)
      
      text_content = str(text_content_list[i])
      font_family = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = font_scale
      color = color
      thickness = 2
      line_type = cv2.LINE_AA

      colored_img = cv2.putText(colored_img, text_content , text_position, font_family, font_scale, color, thickness, line_type)
  
  if save_not_show:

    return colored_img

  else:

    if dpi != None:
      imshow(colored_img, dpi = dpi)
    else:
      imshow(colored_img)


###### CONTOUR I/O WITH CSV ######
def stringify_contour(contour):
  return '|'.join([','.join(map(str,list(point[0]))) for point in contour])
def recover_contour_from_string(contour_string):
  list_of_points = [list(point.split(',')) for point in contour_string.split('|')]
  return np.array(list_of_points, dtype=np.int32)

###### FIND LINES ######

def auto_edgify(img, verbose = False):
  # Reference: https://stackoverflow.com/a/42037449
  if len(img.shape) == 3:
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  elif len(img.shape) == 2:
    grey = img.copy()
  else:
    raise
  v = np.median(grey.flatten()[grey.flatten() != 0])  
  lower_thresh = int(max(0, (1.0 - 0.33) * v))
  upper_thresh = int(min(255, (1.0 + 0.33) * v))
  if verbose:
    print('lower_thresh:', lower_thresh, '   upper_thresh:', upper_thresh)
  edges = cv2.Canny(grey, lower_thresh, upper_thresh)
  return edges

def get_cc_from_df(df):
  G = nx.Graph()
  col_1, col_2 = df.columns
  for _, row in df[[col_1, col_2]].iterrows():
    G.add_edge(row[col_1], row[col_2])
  groups = [list(group) for group in list(nx.connected_components(G))]
  return groups

def get_projected_point_on_ab_line(a, b, pt):
  pt_x, pt_y = pt
  b_prime = pt_y + 1/a * pt_x
  ppt_x = (b_prime - b)/(a + 1/a)
  ppt_y = a * ppt_x + b
  ppt = (round(ppt_x), round(ppt_y))
  return ppt

def get_projected_point_on_p1p2_line(p1, p2, p3, must_on_segment = False):
  ## Reference: https://stackoverflow.com/a/61343727
  l2 = np.sum((p1-p2)**2)
  if l2 == 0:
    raise '[Error] p1 and p2 are the same points'
  t = np.sum((p3 - p1) * (p2 - p1)) / l2 # on line extention connecting p1 and p2 is okay
  if must_on_segment:
    if t > 1 or t < 0:
      print('p3 does not project onto p1-p2 line segment')
    t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2)) # on line segment between p1 and p2 or closest point of the line segment
  projection = p1 + t * (p2 - p1)
  return projection

def find_lines(img, rho = 1, theta = np.pi / 180, threshold = 25, min_line_length = 100, max_line_gap = 20, return_what = 'overlay_layer'):
  # rho              # distance resolution in pixels of the Hough grid
  # theta            # angular resolution in radians of the Hough grid
  # threshold        # minimum number of votes (intersections in Hough grid cell)
  # min_line_length  # minimum number of pixels making up a line
  # max_line_gap     # maximum gap in pixels between connectable line segments

  # `lines` contain endpoints of detected line segments
  lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

  if len(lines) == 0:
    return None

  if return_what == 'overlay_layer':
    # draw lines on a RGB layer
    lines_layer_rgb = np.zeros((*img.shape, 3), dtype = np.uint8)
    for line in lines:
      for x1,y1,x2,y2 in line:
        cv2.line(lines_layer_rgb,(x1,y1),(x2,y2),(255,0,0),1)
    overlayed = cv2.addWeighted(grey_to_rgb(img), 0.8, lines_layer_rgb, 1, 0)
    return overlayed

  if return_what == 'lines_layer':
    # draw lines on a binary layer
    lines_layer = np.zeros(img.shape, dtype = np.uint8)
    for line in lines:
      for x1,y1,x2,y2 in line:
        cv2.line(lines_layer,(x1,y1),(x2,y2),255,1)
    return lines_layer
  
  if return_what == 'lines':
    return lines


#==================================================================================================#






#=========================== HIGH LEVEL COLOR BASED FEATURE EXTRACTOR =============================#

###### UTILS ######
def calculate_contour_average_brightness(input_img, cnt):
    x_min, y_min = cnt.min(axis=0)[0].astype(int)
    x_max, y_max = cnt.max(axis=0)[0].astype(int)
    local_cnt = cnt - np.array([[x_min, y_min]])
    local_img = input_img[y_min:y_max,x_min:x_max]
    cnt_img = mask_with_contours(local_img, [local_cnt])
    average_brightness = round( cnt_img.mean() * ((x_max - x_min)*(y_max - y_min)) / get_contour_area_size(local_cnt), 1)
    return average_brightness


def random_sample_position_of_certain_value(ndarray, value, format = 'xy'):
  yx = np.unravel_index( np.random.choice(np.where(ndarray.flatten() == value)[0]), ndarray.shape )
  if format == 'xy':
    return tuple(reversed(yx))
  elif format == 'yx':
    return yx

def get_vicinity(img,pixel_pos,radius): # pixel_pos in (x,y) format
  x, y = pixel_pos
  vicinity = img[max(0,y-radius):min(y+radius+1,img.shape[0]), max(0,x-radius):min(x+radius+1,img.shape[1])].copy()
  return vicinity


#==================================================================================================#






#================================== UTILS FOR GDRIVE FOLDER =======================================#
 
def move_shallow_folder(from_folder, to_folder, copy = False, exclude_regex = None, wait_interval = 0.1):

  from_folder = from_folder.rstrip('/')
  to_folder = to_folder.rstrip('/')

  if glob(to_folder+'/*'):
    decision = input('[Warning] to_folder is not empty, do you want to "overwrite" or "append" or "stop"?')
    if decision == 'overwrite':
      for p in glob(to_folder+'/*'):
        os.remove(p)
    elif decision == 'stop':
      return
    elif decision != 'append':
      raise ValueError('[Error] Invalid decision code.')
  
  from_filepaths = glob(from_folder+'/*')
  if exclude_regex is not None:
    from_filepaths = [p for p in from_filepaths if not re.findall(exclude_regex, p)]
  if not os.path.exists(to_folder):
    print('Creating to_folder at:', to_folder)
    os.mkdir(to_folder)

  for from_p in tqdm(from_filepaths):
    to_p = to_folder + '/' + from_p[::-1].split('/', maxsplit = 1)[0][::-1]
    if copy:
      shutil.copy(from_p, to_p)
    else:
      shutil.move(from_p, to_p)
    time.sleep(wait_interval)

#==================================================================================================#








#======================================= SELF FUZZY CLUSTER =======================================#

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







#=================================== DEDUPLICATE OCR RESULT =======================================#

def get_dbscan_labels(data, field, radius = 0.5):
    clusterer = DBSCAN(eps=radius, algorithm='auto', metric='euclidean', min_samples=2)
    coorindates_array = np.array(data[field].tolist())
    if len(coorindates_array.shape)==1:
        coorindates_array = coorindates_array.reshape(-1, 1)
    clusterer.fit(coorindates_array)
    clabels = clusterer.labels_
    return clabels

def detect_duplicates(df, minimum_text_area_side_length = 20, minimum_area_thres = None, dbscan_radius = None, fuzzy_scorer=fuzz.partial_ratio, fuzzy_score_cutoff=80, intersection_cover_smaller_shape_by = 0.8, no_numeric = False):

  ## If not explicitly specified, initialize thresholds based on minimum_text_area_side_length (measured in pixels)
  if minimum_area_thres is None:
    minimum_area_thres = int(minimum_text_area_side_length**2)
  if dbscan_radius is None: # scale by 5, an experience based choice which can be fine-tuned
    dbscan_radius = int(2 * minimum_text_area_side_length)

  ## Placeholder for outputs
  ocr_entry_ids_to_drop = []
  backup_ids_mapping = {} # for data provenance and backtracking errors

  ## -------------------------------------------------------------------------------------------------

  ## "basic quality check" within "the whole image"
    
  # criteria is 'cleaned text is not empty & cleaned text is not numeric & bounding box area size > thres'
  # some entries with low ocr confidence are correct, so not using ocr confidence as criteria
  poor_quality_filter = (df['cleaned_text'].apply(len)==0) | (df['bbox_area']<minimum_area_thres)
  if no_numeric:
    poor_quality_filter = poor_quality_filter | (df['cleaned_text'].str.isnumeric())
  poor_quality_ocr_entry_ids = df.loc[poor_quality_filter, 'ocr_entry_id'].tolist()
  ocr_entry_ids_to_drop += poor_quality_ocr_entry_ids
  backup_ids_mapping[-1] = poor_quality_ocr_entry_ids
  df = df[~df['ocr_entry_id'].isin(ocr_entry_ids_to_drop)].copy()

  ## -------------------------------------------------------------------------------------------------

  ## For de-duplication purpose, cluster the ocr bounding box by their centers' locations on the image

  ## "intersection based de-duplications" within "clusters"

  df['dbscan_cluster_id'] = get_dbscan_labels(df, 'bbox_center', radius = dbscan_radius)

  cluster_id_list = get_non_single_elements(df, 'dbscan_cluster_id')
  if -1 in cluster_id_list:
    cluster_id_list.remove(-1)

  for cluster_id in tqdm(cluster_id_list):

    cluster_df = df[df['dbscan_cluster_id'] == cluster_id].copy()

    cluster_df['fuzzy_matched_text'] = cluster_df['text'].map(self_fuzzy_cluster(cluster_df, 'text', scorer = fuzzy_scorer, score_cutoff = fuzzy_score_cutoff))

    repeated_terms = get_non_single_elements(cluster_df, 'fuzzy_matched_text')

    for term in repeated_terms:
      
      same_term_group = cluster_df[cluster_df['fuzzy_matched_text']==term].copy()
      same_term_group['shapely_polygon'] = same_term_group['bounding_box'].apply(lambda x: shapely_polygon(x))
      same_term_group['shapely_polygon_area_size'] = same_term_group['shapely_polygon'].apply(lambda x: x.area)
      same_term_group = same_term_group.sort_values('shapely_polygon_area_size', ascending=False).reset_index(drop=True)

      for i in range(len(same_term_group)-1):
        # print('i:',i)
        larger_shape_ocr_entry_id = same_term_group['ocr_entry_id'][i]
        if larger_shape_ocr_entry_id in ocr_entry_ids_to_drop:
          # print('skip large')
          continue
        larger_shape = same_term_group['shapely_polygon'][i]

        for j in range(i+1, len(same_term_group)):
          # print('j:',j)
          smaller_shape_ocr_entry_id = same_term_group['ocr_entry_id'][j]
          if smaller_shape_ocr_entry_id in ocr_entry_ids_to_drop:
            # print('skip small')
            continue
          smaller_shape = same_term_group['shapely_polygon'][j]

          smaller_shape_area_size = same_term_group['shapely_polygon_area_size'][j]
          
          ## try except to catch the error where one or both of the shapes are invalid
          try:
            intersection_area_size = larger_shape.intersection(smaller_shape).area
          except:
            intersection_area_size = None

          if intersection_area_size == None or intersection_area_size/smaller_shape_area_size > intersection_cover_smaller_shape_by:  
            # If intersection between larger and smaller shapes cover the majority of the smaller shape, 
            # then we can say that the smaller shape is contained in the larger shape, thus likely a duplicate
            ocr_entry_ids_to_drop.append(smaller_shape_ocr_entry_id)
            backup_ids_mapping[larger_shape_ocr_entry_id] = backup_ids_mapping.get(larger_shape_ocr_entry_id, [])+[smaller_shape_ocr_entry_id]

  df = df[~df['ocr_entry_id'].isin(ocr_entry_ids_to_drop)].copy()

  ## -------------------------------------------------------------------------------------------------

  return ocr_entry_ids_to_drop, backup_ids_mapping

#==================================================================================================#








#===================================== WRITE SHAPEFILE ============================================#

def get_dtype_as_string(x):
  return 'int' if isinstance(x,int) else 'float' if isinstance(x,float) else 'str'

def create_shapefile_from_df(filepath, dataframe, properties_columns, geometry_column, geometry_type = 'Polygon', crs = 'EPSG:4326'):

  ##############################
  # Copy dataframe & reset index
  dataframe = dataframe.copy().reset_index(drop=True)

  ##############################
  # Build schema based on inputs
  sample_value_dict = dataframe[:1].T.to_dict()[0]
  schema = {'properties': [( col,  get_dtype_as_string(sample_value_dict[col])  ) for col in properties_columns], 
  'geometry': geometry_type}

  ##############################
  # Open a fiona object
  shp_file = fiona.open(filepath, mode = 'w', driver = 'ESRI Shapefile', schema = schema, crs = crs)
  # Create records
  records = []
  for _, row in dataframe.iterrows():
    records.append({
        'properties': {col: row[col] for col in properties_columns} ,
        'geometry' : {'type': geometry_type, 'coordinates': [ row[geometry_column] ]} , # remember to keep the outer square brackets
    })
  # Write records
  shp_file.writerecords(records)
  # Close fiona object
  shp_file.close()

#==================================================================================================#




#========================================= Geospatial =============================================#

# def latlon_to_xy(latlon_pair, affine_transform_object):
#   transform_string = repr(affine_transform_object)
#   transform_string = re.findall(r'\((.*?)\)',transform_string.replace('\n',''))[0]
#   num_string_list = re.split(r'\,\s+',transform_string)
#   transform_matrix = np.array([eval(x) for x in num_string_list]+[0,0,1]).reshape((3,3))
#   product_vector = np.array([latlon_pair[1], latlon_pair[0], 1])
#   xy = np.linalg.solve(transform_matrix, product_vector)[:2]
#   return xy

def raster_geocode(point, affine_matrix, reverse = False, rounding = 6):
    if reverse:
        affine_matrix = ~affine_matrix
        return tuple(np.round(affine_matrix * point,0).astype(int))
    else:
        return np.round(affine_matrix * point, rounding)

#==================================================================================================#







#======================================= Image Matching ===========================================#

def get_matching_point(input_point, anchor_points):
  input_ref_pts = np.array(anchor_points[:4], dtype=np.float32)
  output_ref_pts = np.array(anchor_points[4:], dtype=np.float32)
  M = cv2.getPerspectiveTransform(input_ref_pts, output_ref_pts)
  output_point = tuple(np.round(cv2.perspectiveTransform(np.array([[input_point]], dtype=np.float32), M),0).astype(int)[0][0])
  return output_point

def get_angle_diff(angle1, angle2):
  return min(abs(angle1-angle2),360-abs(angle1-angle2))
def get_average_angle(arr):
  arr = np.array(arr)
  return np.round(np.degrees(np.arctan2(np.mean(np.sin(np.radians(arr))),  np.mean(np.cos(np.radians(arr))))),1)
def rotate_coordinate(p, origin=(0, 0), degrees=0):
  # Reference: https://stackoverflow.com/a/58781388
  angle = np.deg2rad(degrees)
  R = np.array([[np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]])
  o = np.atleast_2d(origin)
  p = np.atleast_2d(p)
  return np.squeeze((R @ (p.T-o.T) + o.T).T).astype(np.int32)

#==================================================================================================#





clear_output()
print('\nImage data mining (IDM) module is ready. Enjoy exploring!\n')

#========================================= References =============================================#

# https://stackoverflow.com/questions/16705721/opencv-floodfill-with-mask
# https://blog.csdn.net/qq_37385726/article/details/82313004
# https://stackoverflow.com/questions/60197665/opencv-how-to-use-floodfill-with-rgb-image
# https://rasterio.readthedocs.io/en/latest/topics/georeferencing.html
# https://hatarilabs.com/ih-en/how-to-create-a-pointlinepolygon-shapefile-with-python-and-fiona-tutorial

#==================================================================================================#


def explode_geometry(gdf, geom_colum = 'geometry', id_column = None, drop_duplicates = False):
    gdf = gdf.explode(geom_colum, index_parts = False).reset_index(drop=True)
    if drop_duplicates:
        if id_column is None:
            raise '[Error] Please specify id_column parameter to use drop duplicates functionality.'
        gdf[geom_colum+'__area_size'] = gdf[geom_colum].area
        gdf = gdf.sort_values(geom_colum+'__area_size', ascending=False)
        gdf = gdf.drop_duplicates(subset=[id_column], keep='first')
        gdf = gdf.drop([geom_colum+'__area_size'], axis=1)
    return gdf


def add_coordinates_column(data, geometry_column = 'geometry', new_column = None):
    if new_column is None:
        new_column = geometry_column+'__coordinates'
    data[new_column] = data[geometry_column].apply(lambda x: np.round(np.array(x.exterior.coords.xy).T,6))
    return data

def add_reverse_geocode_column(data, coordinates_column, affine_transform_column = 'affine_transform', new_column = None):
    if new_column is None:
        new_column = coordinates_column+'__on_canvas'
    data[new_column] = data[[coordinates_column, affine_transform_column]].apply(lambda row: [raster_geocode(pt, row[affine_transform_column], reverse=True) for pt in row[coordinates_column]], axis=1)
    return data

def draw_poly(input_img, points, close = False, color = (255,0,0), thickness = 5, fill=False):    
    if fill:
        output_img = cv2.fillPoly(input_img, [np.array(points)], color=color)
    else:
        output_img = cv2.polylines(input_img, [np.array(points)], isClosed=close, color=color, thickness=thickness)
    return output_img


def get_today_as_string():
    return str(datetime.now().date()).replace('-','')

def get_xy_range_without_black_border(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    col_positive_positions = np.where( img.mean(axis=1) > 0 )[0]
    row_positive_positions = np.where( img.mean(axis=0) > 0 )[0]
    y_start, y_end = col_positive_positions.min(), col_positive_positions.max()
    x_start, x_end = row_positive_positions.min(), row_positive_positions.max()

    return (x_start, x_end, y_start, y_end)

def crop_and_downscale(img, downscale_ratio, x_start = None, x_end = None, y_start = None, y_end = None):

  x_start = 0 if x_start is None else x_start
  y_start = 0 if y_start is None else y_start
  x_end = img.shape[1] if x_end is None else x_end
  y_end = img.shape[0] if y_end is None else y_end

  x_start_r = int(np.ceil(x_start/downscale_ratio)*downscale_ratio)
  y_start_r = int(np.ceil(y_start/downscale_ratio)*downscale_ratio)
  x_end_r = int(np.floor(x_end/downscale_ratio)*downscale_ratio)
  y_end_r = int(np.floor(y_end/downscale_ratio)*downscale_ratio)

  cropped = img[y_start_r:y_end_r, x_start_r:x_end_r,:]
  target_size = (cropped.shape[1]//downscale_ratio, cropped.shape[0]//downscale_ratio) 
  resized = cv2.resize(cropped, target_size, interpolation = cv2.INTER_AREA) # INTER_AREA: "resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results" according to OpenCV doc

  return resized, x_start_r, x_end_r, y_start_r, y_end_r

def distances_to_point(data, field, point):
  return data[field].apply(lambda x: np.linalg.norm(np.array(x)-np.array(point)))

def get_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))

def move_point(pt, direction, step=1):
    x,y = pt
    return (x-step,y) if direction == 'left' else (x+step,y) if direction == 'right' else (x,y-step) if direction == 'up' else (x,y+step) if direction == 'down' else (x,y)

def search_seed_point_in_one_direction(seed_pixel_pool, contour, potential_seed, move_direction, move_step = 1, vicinity_radius = 2):
  
    # evaluate the distance from seed point to the contour that we want to capture by floodfilling from the seed, 
    # and whether the vicinity of the seed contains dark color (color of shape border) that will influence a floodfill operation
    dist_to_contour = cv2.pointPolygonTest(contour, potential_seed, measureDist=True)
    seed_vicinity_contain_dark_color = invert_binary(get_vicinity(seed_pixel_pool, potential_seed, radius = vicinity_radius)).sum() > 0

    # initialize prev_dist_to_contour with a large value
    prev_dist_to_contour = max(seed_pixel_pool.shape)+1

    # check if seed point is outside the contour or if the vicinity of the seed contains dark color that influences floodfill
    while dist_to_contour <= 0 or seed_vicinity_contain_dark_color:

        potential_seed = move_point(potential_seed, direction = move_direction, step = move_step)

        # re-evaluate with the new seed point position
        dist_to_contour = cv2.pointPolygonTest(contour, potential_seed, measureDist=True)
        seed_vicinity_contain_dark_color = invert_binary(get_vicinity(seed_pixel_pool, potential_seed, radius = vicinity_radius)).sum() > 0

        # check if seed point is outside the contour and moving in the wrong direction (i.e. movinig away from the contour)
        # if so, return None to indicate this is a bad direction to search for seed
        if dist_to_contour <= 0 and abs(dist_to_contour) > abs(prev_dist_to_contour):
            return None

        # save current dist as prev for comparison in the next iteration
        prev_dist_to_contour = dist_to_contour

    # exiting while looping meaning seed point meet criteria
    return potential_seed

def extract_map_id(p):
    return int(p.split('/')[-1].split('_')[0])

def get_contour_centroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)


def get_representative_point(cnt):
    # Reference: https://stackoverflow.com/a/65409262
    poly = shapely_polygon(cnt.squeeze())
    cx = int(round(poly.representative_point().x))
    cy = int(round(poly.representative_point().y))
    return cx, cy

def point_list_to_contour(li):
    return np.array([[pt] for pt in li], dtype=np.int32)

def contour_to_point_list(cnt):
    return [tuple(pt[0]) for pt in cnt]

def get_min_area_rect_cnt(cnt):
    return np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))

def get_min_area_rect_stats(cnt):
    min_area_rect_center, min_area_rect_w_h, min_area_rect_angle=cv2.minAreaRect(cnt)
    _w, _h = min_area_rect_w_h
    min_area_rect_aspect_ratio = round(max( _w/_h, _h/_w ),2)
    return min_area_rect_center,min_area_rect_aspect_ratio,min_area_rect_angle


def draw_text(img, text, pos, font = cv2.FONT_HERSHEY_SIMPLEX, size = 1, color = 0, thickness = 2, align = 'center', line_type = cv2.LINE_AA):
    # opencv 4.1.2
    if align == 'center':
        text_width, text_height = cv2.getTextSize(text, fontFace = font, fontScale = size, thickness = thickness)[0]
        x_offset = int(round(text_width/2))
        y_offset = int(round(text_height/2))
        pos = (pos[0] - x_offset, pos[1] - y_offset)
    img = cv2.putText(img, text, pos, fontFace = font, fontScale = size, thickness = thickness, lineType = line_type, color = color)
    return img





def get_seed_pixel_pool(input_img):
    im = rgb_to_grey(input_img)
    im = adaptive_threshold(im, size=101, C = 25)
    im = invert_binary(im)
    im = cv2.dilate(im, get_kernel(3), iterations = 2)
    im = invert_binary(im)
    return im

def flood_fill(img, seed_pixel, color_mode = 'rgb', color_variations = (5,5,5), neighbor = 8, fill_value = (0,0,0), return_mask = True):

    flood_img = img.copy()

    if fill_value != (0,0,0):
        black_pixels = np.where((flood_img[:, :, 0] == 0) & (flood_img[:, :, 1] == 0) & (flood_img[:, :, 2] == 0))
        flood_img[black_pixels] = fill_value

    h, w = flood_img.shape[:2]

    seed_pixel_list = seed_pixel if isinstance( seed_pixel, list ) else [seed_pixel]

    mask_list = []
    for seed_pixel in seed_pixel_list:
        num, flood_img, mask, rect = cv2.floodFill(flood_img, np.zeros((h+2,w+2),np.uint8), seed_pixel, fill_value, color_variations, color_variations, neighbor)
        mask_list.append(mask)

    combined_mask = mask_list.pop(0)
    while mask_list:
        combined_mask = cv2.bitwise_or(combined_mask, mask_list.pop())
    combined_mask = combined_mask[1:-1,1:-1]
    combined_mask = combined_mask*255

    if return_mask:
        return combined_mask
    else:
        return flood_img


def create_range_around_hsv_code(hsv_code, radius = (3,10,10)):
  lower_bound, upper_bound = [], []
  for i in range(len(hsv_code)):
    lower_value = hsv_code[i]-radius[i]
    upper_value = hsv_code[i]+radius[i]
    if i == 0:
      lower_value = lower_value % 180
      upper_value = upper_value % 180
    else:
      lower_value = np.clip(lower_value, 0, 255)
      upper_value = np.clip(upper_value, 0, 255)
    lower_bound.append(lower_value)
    upper_bound.append(upper_value)
  return tuple(lower_bound), tuple(upper_bound)

def find_area_of_hsv_color(img, hsv_code, radius, alpha = 0.5, dpi = 150, overlay = True, show = True, return_mask = False):

  img = img.copy()
  img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  lower_bound, upper_bound = create_range_around_hsv_code(hsv_code = hsv_code, radius = radius)

  if lower_bound[0]>upper_bound[0]: # for color hue like red that are on the edge of hue range
    lower_mask = cv2.inRange(img_hsv, (0,lower_bound[1],lower_bound[2]), (lower_bound[0],upper_bound[1],upper_bound[2]))
    upper_mask = cv2.inRange(img_hsv, (upper_bound[0],lower_bound[1],lower_bound[2]), (180,upper_bound[1],upper_bound[2]))
    mask = cv2.bitwise_or(lower_mask, upper_mask)
  else:  # for other color not on the edge
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)

  cropped_hsv = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
  cropped = cv2.cvtColor(cropped_hsv, cv2.COLOR_HSV2BGR)
  if show:
    if overlay:
      overlayed = cv2.addWeighted(cropped, alpha, img, 1-alpha, 0.0)
      imshow(overlayed, dpi = dpi)
    else:
      imshow(cropped, dpi = dpi)
  if return_mask:
    return mask


def get_min_area_rect_stats(cnt):
    min_area_rect_center, min_area_rect_w_h, min_area_rect_angle=cv2.minAreaRect(cnt)
    _w, _h = min_area_rect_w_h
    min_area_rect_aspect_ratio = round(max( _w/_h, _h/_w ),2)
    return min_area_rect_center,min_area_rect_aspect_ratio,min_area_rect_angle

def rgb_to_hsv(img):
  return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def fast_analyze_color(img, n_clusters, sample_size = 100000, return_classifier = False):

    # assuming 3-channel RGB
    img = img.reshape(-1,3)

    # ignore pure black pixles (background)
    img = img[~np.all(img == 0, axis=1)]

    # random downsampling if pixel count is greater than specified sample size
    pixel_count = img.shape[0]
    if pixel_count > sample_size:
        img = img[np.random.choice(pixel_count, sample_size, replace=False), :]

    # run kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img)

    # calculate color codes
    color_codes = [tuple(c) for c in np.round(kmeans.cluster_centers_).astype(np.int32)]

    # count color proportions
    color_pixel_counts = tuple(np.bincount(kmeans.labels_))

    # re-order the color info in order of ascending proportion
    centroid_colors_info = sorted(zip(color_pixel_counts, color_codes), reverse=True)

    if return_classifier:
      return kmeans, centroid_colors_info

    return centroid_colors_info

def show_colors(colors):
    display_width = 100
    display_height = 20
    pixel_counts = np.array([tup[0] for tup in colors])
    color_proportions = np.int0(pixel_counts/pixel_counts.sum()*display_width)
    color_row = []
    for i in range(len(colors)):
        color_row += [colors[i][1][::-1]]*color_proportions[i]
    color_img = np.array([color_row]*display_height)
    imshow(color_img,no_axis=True,dpi=50)

def plot_palette(color_info, mode='rgb', width = 1000, height = None, gap_size = None, dpi = 120, font_size = 6, jiggle = False):
    
    # These parameter affects visual style only and can be exposed to user later
    
    if gap_size is None:
        gap_size = max(1, int(width * 0.005))

    if height is None:
        height = max(50, int(width * 0.2))
    
    palette = np.zeros((height, width, 3), np.uint8)
    
    cluster_centers = [hsv_code_to_rgb_code(color_code) if mode=='hsv' else color_code for count, color_code in color_info]
    
    # Count how many pixels belong to each color cluster, let this decides the color's relative width in the palette
    label_counts = np.array([count for count, color_code in color_info])
    cluster_proportion = label_counts/label_counts.sum()
    cluster_width_list = list(cluster_proportion * width)
    cluster_width_list = [int(x) for x in saferound(cluster_width_list, places=0)]

    # Coloring the palette canvas based on color and width
    endpoints = list(np.cumsum(cluster_width_list))
    startpoints = [0]+endpoints[:-1]
    for cluster_index in range(len(cluster_centers)):
        palette[:, startpoints[cluster_index]:startpoints[cluster_index]+gap_size, :] = (255,255,255) # draw a white gap
        palette[:, startpoints[cluster_index]+gap_size:endpoints[cluster_index], :] = cluster_centers[cluster_index] # draw the color

    # Displaying the palette
    plt.figure(dpi = dpi)
    plt.imshow(palette)

    # Marking the cluster index
    for cluster_index in range(len(cluster_centers)):
        x_pos = (startpoints[cluster_index]+gap_size + endpoints[cluster_index])//2
        y_pos = height//2
        if jiggle:
            y_pos = y_pos + (height//5) * ((-1)**cluster_index)
        plt.text(x = x_pos, y = y_pos, s = str(cluster_index), fontsize = font_size, ha = 'center', va = 'center')
    plt.axis('off')
    plt.show()


# color_classifier, centroid_colors_info = fast_analyze_color( primary_color_temp_img_hsv, n_clusters=15, sample_size = 10000000, return_classifier = True )
# plot_palette(centroid_colors_info, mode='hsv', dpi=150, jiggle=True)

# color_name_to_color_code_mapping = {'background':[0,1,2,3,5,8,11],
# 'red':[4,6,9],
# 'yellow':[7,14],
# 'green':[10,12],
# 'blue':[13]}

# color_code_to_color_name_mapping = {}
# for k,v in color_name_to_color_code_mapping.items():
#     color_code_to_color_name_mapping.update( dict(zip(v, [k]*len(v))) ) 

# color_cluster_labels = color_classifier.predict(   primary_color_temp_img_hsv[0,:,:]   )

# footprint_gdf['primary_color_name'] = list(map(color_code_to_color_name_mapping.get, color_cluster_labels))

# footprint_gdf['primary_color_name'].value_counts(normalize=True).round(2)

# # background    0.60
# # red           0.15
# # green         0.14
# # yellow        0.11
# # blue          0.01

def round_point(pt):
    if pt[0] is None or pt[1] is None:
        return np.nan
    return tuple(np.round(pt,0).astype(int))

def round_value(v):
    if v is None or np.isnan(v):
        return np.nan
    return int(round(v))

def camel_to_snake(x):
    if x.isupper():
        return x.lower()
    output = ''
    for l in list(x):
        output += '_'+l.lower() if l.isupper() else l
    output = output.strip('_')
    return output


#######################################################################################################################


def dist_from_point_to_line(line_endpoint_a, line_endpoint_b, point):
    if len(line_endpoint_a.shape)>1:
        return np.abs(np.cross(line_endpoint_b - line_endpoint_a, line_endpoint_a - point)) / np.linalg.norm(line_endpoint_b - line_endpoint_a, axis=1)
    else:
        return np.abs(np.cross(line_endpoint_b - line_endpoint_a, line_endpoint_a - point)) / np.linalg.norm(line_endpoint_b - line_endpoint_a)

def get_vector_direction(vector, rounding = 1):
  """np_arctan2_in_degree (-180 to 180 reference angle is the positive direction of x axis in cartesian space)""" 
  x, y = vector
  return np.round(np.arctan2(y, x) * 180 / np.pi, rounding)

def get_angle_between_vectors(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    angle_in_degrees = np.degrees(angle)
    return angle_in_degrees

def add_bbox_feature_columns(df):

    np.seterr(invalid='ignore')

    df['bounding_box'] = df['bounding_box'].apply(lambda bbox: np.array(bbox, dtype=np.int32).reshape((-1,2)))

    top_left     = np.array( df['bounding_box'].apply(lambda li: li[0]).tolist() )
    top_right    = np.array( df['bounding_box'].apply(lambda li: li[1]).tolist() )
    bottom_right = np.array( df['bounding_box'].apply(lambda li: li[2]).tolist() )
    bottom_left  = np.array( df['bounding_box'].apply(lambda li: li[3]).tolist() )

    width_arr = ( np.linalg.norm(top_right - top_left, axis=1) + np.linalg.norm(bottom_right - bottom_left, axis=1) )/2

    height_by_different_measures = [ dist_from_point_to_line(bottom_left, bottom_right, top_left),  
                                    dist_from_point_to_line(bottom_left, bottom_right, top_right),  
                                    dist_from_point_to_line(top_left, top_right, bottom_left),  
                                    dist_from_point_to_line(top_left, top_right, bottom_right) ] 

    height_arr = np.nanmean(height_by_different_measures, axis=0)

    bbox_center_arr = (top_left + top_right + bottom_right + bottom_left)/4
    left_side_center_arr  = (top_left + bottom_left)/2
    right_side_center_arr = (top_right + bottom_right)/2

    reading_direction_vector = right_side_center_arr - left_side_center_arr
    reading_direction_arr = np.arctan2(reading_direction_vector[:,1], reading_direction_vector[:,0]) / np.pi * 180

    del top_left, top_right, bottom_right, bottom_left

    df['bbox_width'] = width_arr
    df['bbox_height'] = height_arr
    df['bbox_center'] = bbox_center_arr.tolist()
    df['bbox_left_side_center'] = left_side_center_arr.tolist()
    df['bbox_right_side_center'] = right_side_center_arr.tolist()
    df['bbox_reading_direction'] = reading_direction_arr

    df['text_'] = df['text'].apply(unidecode)
    df.loc[df['text_'].apply(len) == df['text'].apply(len),'text'] = df.loc[df['text_'].apply(len) == df['text'].apply(len),'text_']
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\x00-\x7F]','',x)) # r'[^A-Za-z0-9\.\,\-\=\&\']'
    df = df.drop('text_',axis=1)

    np.seterr(invalid='warn')

    return df
