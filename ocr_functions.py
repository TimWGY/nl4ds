from IPython.display import clear_output

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import os
from glob import glob
import re
from collections import Counter
import ast

import time
from PIL import Image
from matplotlib.patches import Polygon

os.system('pip install --upgrade azure-cognitiveservices-vision-computervision')
clear_output()
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

computervision_client = ComputerVisionClient(input('\nEndpoint?\n'), CognitiveServicesCredentials(input('\nKey?\n')))
clear_output()


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

def mark_ms_ocr_result(image_file_path, components_df, fontsize=10, filename=''):

  image = Image.open(image_file_path)

  plt.figure(figsize=(20, 20), dpi=150)
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

def ms_ocr(img_path):

  raw_ocr_result_filepath = img_path.split('.')[0] + '_raw_ocr_result.txt'
  if not os.path.exists(raw_ocr_result_filepath):
    result = get_ms_ocr_result(img_path)
    save_dict_to_json(result, raw_ocr_result_filepath)

  ocr_result_table_filepath = img_path.split('.')[0] + '_ocr_result_table.csv'
  if not os.path.exists(ocr_result_table_filepath):
    comp_df = parse_ms_ocr_result(result)
    comp_df.to_csv(ocr_result_table_filepath, index=False)
  else:
    print('OCR result table found.')
    comp_df = pd.read_csv(ocr_result_table_filepath)
    comp_df['bounding_box'] = comp_df['bounding_box'].apply(ast.literal_eval)

  ocr_result_marked_img_path = img_path.split('.')[0] + '_ocr_result_marked_img.' + img_path.split('.')[1]
  mark_ms_ocr_result(img_path, comp_df, fontsize=10, filename=ocr_result_marked_img_path)

######################################################################################
