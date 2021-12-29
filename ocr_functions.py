from IPython.display import clear_output

import pandas as pd
import numpy as np
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt
plt.rcParams["font.serif"] = "cmr10"

import os
from glob import glob
import re
from collections import Counter
import ast

import time
from PIL import Image

os.system('pip install Rasterio')
import rasterio
from rasterio.windows import Window

os.system('pip install tqdm')
from tqdm import tqdm

os.system('pip install pyproj')

os.system('pip install --upgrade azure-cognitiveservices-vision-computervision')
clear_output()
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

computervision_client = ComputerVisionClient(input('\nEndpoint?\n'), CognitiveServicesCredentials(input('\nKey?\n')))
clear_output()


####################################

def flatten_list(l):
  return [item for sublist in l for item in sublist]

def get_ms_ocr_result(read_image_path, wait_interval=10):

  # Open the image
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

def mark_ms_ocr_result(image_file_path, components_df, filename='', fontsize=10,  figsize=(20,20), dpi=150):

  image = Image.open(image_file_path)

  plt.figure(figsize=figsize, dpi=dpi)
  ax = plt.imshow(image, cmap=cm.gray)

  polygons = []
  for _, row in components_df.iterrows():
    polygons.append((row['bounding_box'], row['text']))

  for bbox, ocr_text in polygons:
    vertices = [(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)]
    patch = Polygon(vertices, closed=True, fill=False, linewidth=1, color='b')
    ax.axes.add_patch(patch)
    plt.text(vertices[1][0], vertices[1][1], ocr_text, fontsize=fontsize, color='r', va="top")

  if filename != '':
    plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=0)


######################################################################################

import cv2


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

######################################################################################


import json


def save_dict_to_json(dic, filepath):
  with open(filepath, 'w') as f:
    json.dump(dic, f)


def read_dict_from_json(filepath):
  with open(filepath, 'r') as f:
    dic = json.load(f)
  return dic

def ms_ocr(img_path, mark_image = True, show_numeric = False, fontsize = 10, figsize = (20,20), dpi = 150):

  raw_ocr_result_filepath = img_path.split('.')[0] + '_raw_ocr_result.txt'
  if not os.path.exists(raw_ocr_result_filepath):
    result = get_ms_ocr_result(img_path)
    save_dict_to_json(result, raw_ocr_result_filepath)
  else:
    print('Raw OCR result found.')
    result = read_dict_from_json(raw_ocr_result_filepath)

  ocr_result_table_filepath = img_path.split('.')[0] + '_ocr_result_table.csv'
  if not os.path.exists(ocr_result_table_filepath):
    comp_df = parse_ms_ocr_result(result)
    comp_df.to_csv(ocr_result_table_filepath, index=False)
  else:
    print('OCR result table found.')
    comp_df = pd.read_csv(ocr_result_table_filepath)
    comp_df['bounding_box'] = comp_df['bounding_box'].apply(ast.literal_eval)

  if mark_image:
    if not show_numeric:
      comp_df = comp_df[~(comp_df['text'].str.isnumeric())]
    ocr_result_marked_img_path = img_path.split('.')[0] + '_ocr_result_marked_img.' + img_path.split('.')[1]
    mark_ms_ocr_result(img_path, comp_df, filename=ocr_result_marked_img_path, fontsize=fontsize, figsize=figsize, dpi=dpi)

######################################################################################

def cut_tiff_into_pngs(path, window_side_length, window_stride = None, output_directory_path = None):

  cutted_image_filepath_list = []

  if window_stride is None:
    window_stride = window_side_length//2

  dataset = rasterio.open(path)
  dataset_name = dataset.name.split('/')[-1].split('.')[0]
  if output_directory_path is None:
    output_directory_path = '/'.join(path.split('/')[:-1]) +'/'+ dataset_name+'__wsl_'+str(window_side_length)+'_ws_'+str(window_stride)
  else:
    output_directory_path = output_directory_path.rstrip('/')

  if not os.path.exists(output_directory_path):
    os.mkdir(output_directory_path)
  else:
    raise ValueError("[Error] Output directory already exists, please check and resolve.")

  print(dataset.meta)
  with open(output_directory_path +'/'+ 'metadata.txt', 'w') as f:
    f.write(str(dataset.meta))

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

  for col_off, row_off in tqdm(offset_pair_list):
    bands = []
    for band_index in range(1,dataset_band_count+1):
      bands.append( dataset.read(band_index, window = Window(col_off, row_off, window_side_length, window_side_length)) )
    window_img = np.stack(bands, axis=2)
    if window_img[:,:,0].sum() != 0:
      im = Image.fromarray(window_img)
      output_path = output_directory_path +'/'+ dataset_name+'_xoff'+str(col_off)+'_yoff'+str(row_off)+'_wsl_'+str(window_side_length)+'.png'
      im.save(output_path)
      cutted_image_filepath_list.append(output_path)

  cutted_image_filepath_list = sorted(set(cutted_image_filepath_list))
  return cutted_image_filepath_list

######################################################################################

def get_corners_of_geotiff(dataset):
  h, w = dataset.shape
  transform_matrix = dataset.transform
  geo_point_list = [tuple([float(x) for x in np.round(transform_matrix * point,6)]) for point in [(0,0), (0,h), (w,h), (w,0), (0,0)]]
  return geo_point_list


import pyproj
from pyproj import Geod
from shapely.geometry import Polygon

def get_area_size_from_geo_point_list(geo_point_list):
  """Input: geo_point_list in format of [(lon, lat), ...]
  Output: size of area in m^2
  # Reference: https://stackoverflow.com/questions/68118907/shapely-pyproj-find-area-in-m2-of-a-polygon-created-from-latitude-and-longi
  """
  polygon = Polygon(geo_point_list)
  geod = Geod(ellps="WGS84")
  poly_area, poly_perimeter = geod.geometry_area_perimeter(polygon)
  return int(poly_area)

def calculate_area_per_pixel(raw_image_filepath):
  dataset = rasterio.open(raw_image_filepath)
  image_pixel_area_size = dataset.shape[0]*dataset.shape[1]
  image_geo_area_size = get_area_size_from_geo_point_list(get_corners_of_geotiff(dataset))
  area_per_pixel = image_geo_area_size/image_pixel_area_size
  area_per_pixel = round(area_per_pixel,6)
  return area_per_pixel

def calculate_area_per_pixel_list(orig_tif_filepaths, verbose = False):
  area_per_pixel_list = []
  for raw_image_index in tqdm(range(0, len(orig_tif_filepaths))):
    raw_image_filepath = orig_tif_filepaths[raw_image_index]
    if verbose:
      print('------------------- raw image index '+str(raw_image_index)+' -------------------')
      print(raw_image_filepath.split('/')[-1])
    area_per_pixel = calculate_area_per_pixel(raw_image_filepath)
    if verbose:
      print('area_per_pixel:', area_per_pixel)
    area_per_pixel_list.append(area_per_pixel)
  return area_per_pixel_list

def reject_outliers(data, m = 2.):
  # Reference: https://stackoverflow.com/a/16562028
  data = np.array(data)
  d = np.abs(data - np.median(data))
  mdev = np.median(d)
  s = d/mdev if mdev else 0.
  return data[s<m]

def get_outlier_bounds(data, m = 2.):
  data = np.array(data)
  data_without_outlier = reject_outliers(data, m = m)
  return data_without_outlier.min(), data_without_outlier.max()
  