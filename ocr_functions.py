from IPython.display import clear_output

import pandas as pd
import numpy as np
from matplotlib import cm
from matplotlib.patches import Polygon as mpb_polygon
from matplotlib import pyplot as plt
plt.rcParams["font.serif"] = "cmr10"

import os
from glob import glob
import re
from collections import Counter
import ast

import time

from PIL import Image
import cv2

os.system('pip install Rasterio')
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

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
    patch = mpb_polygon(vertices, closed=True, fill=False, linewidth=1, color='b')
    ax.axes.add_patch(patch)
    plt.text(vertices[1][0], vertices[1][1], ocr_text, fontsize=fontsize, color='r', va="top")

  if filename != '':
    plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=0)


######################################################################################

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

  area_per_pixel = calculate_area_per_pixel(raw_image_filepath)
  
  dataset_meta = dict(dataset.meta)
  dataset_meta['area_per_pixel'] = area_per_pixel
  print(dataset_meta)
  with open(output_directory_path +'/'+ 'metadata.txt', 'w') as f:
    f.write(str(dataset_meta))

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

######################################################################################

def get_geo_points_of_the_corners_of_image(dataset):
  
  return geo_point_list


import pyproj
from pyproj import Geod
from shapely.geometry import Polygon as shapely_polygon

def get_area_size_from_geo_point_list(geo_point_list):
  """Input: geo_point_list in format of [(lon, lat), ...]
  Output: size of area in m^2
  # Reference: https://stackoverflow.com/questions/68118907/shapely-pyproj-find-area-in-m2-of-a-polygon-created-from-latitude-and-longi
  """
  polygon = shapely_polygon(geo_point_list)
  geod = Geod(ellps="WGS84")
  poly_area, poly_perimeter = geod.geometry_area_perimeter(polygon)
  return int(poly_area)

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
  
######################################################################################


def read_geotransform_parameters(aux_xml_filepath):
  with open(aux_xml_filepath, 'r') as f:
    raw = f.read()
  geotransform_parameters = [ast.literal_eval(v.strip()) for v in re.findall('<GeoTransform>(.*?)</GeoTransform>',raw)[0].split(',')]
  return geotransform_parameters

def create_transform_matrix(para):
  transform_matrix = Affine(para[1], para[2], para[0], para[4], para[5] , para[3])
  return transform_matrix


def cut_png_into_pngs(path, window_side_length, window_stride = None, output_directory_path = None, skip_if_directory_exists = False):

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

  aux_xml_filepath = path.replace('.png','.png.aux.xml')
  affine_info_string = repr(create_transform_matrix(read_geotransform_parameters(aux_xml_filepath)))


  area_per_pixel = calculate_area_per_pixel(path)

  dataset_meta_string = "{'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': "+str(dataset_width)+", 'height': "+str(dataset_height)+", 'count': 4, 'crs': CRS.from_epsg(4326), 'transform': "+affine_info_string+", 'area_per_pixel':"+str(area_per_pixel)+"}"
  print(dataset_meta_string)
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
