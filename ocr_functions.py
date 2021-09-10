from IPython.display import clear_output

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import os
from glob import glob
import re
from collections import Counter

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

  plt.figure(figsize=(20, 20), dpi=300)
  ax = plt.imshow(image, cmap=cm.gray)

  polygons = []
  for _, row in components_df.iterrows():
    polygons.append((row['bounding_box'], row['text']))

  for bbox, ocr_text in polygons:
    vertices = [(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)]
    patch = Polygon(vertices, closed=True, fill=False, linewidth=1, color='b')
    ax.axes.add_patch(patch)
    plt.text(vertices[1][0], vertices[1][1], ocr_text, fontsize=fontsize, color='r', va="top")

  plt.show()

  if filename != '':
    plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=0)
