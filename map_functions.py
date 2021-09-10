from IPython.display import clear_output

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.cm as cm

import os
from glob import glob

import re
from collections import Counter

import cv2
from google.colab.patches import cv2_imshow

from PIL import Image

os.system('pip install rasterio')
os.system('pip install iteround')

import rasterio
from rasterio.plot import reshape_as_image

from iteround import saferound

import colorsys

import plotly.graph_objs as go
import seaborn as sns

import warnings
warnings.filterwarnings('error')

def hsv2rgb(h,s=1.0,v=1.0):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def imsave(img, filename):
  plt.imsave(filename, img, cmap=cm.gray)

def get_w_h_ratio(img):
  return img.shape[1]/img.shape[0]
  
def imshow(img, width = 9, dpi = 90):
    
  w_h_ratio = get_w_h_ratio(img)
  plt.figure(figsize=(width,round(width*w_h_ratio,1)), dpi=dpi)
  plt.grid(False)
  if len(img.shape)==2:
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
  else:
    plt.imshow(img)

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

def param_search(img, func, param_1_range, param_2_range = None, param_layout = None):

# param_search(sample, cv2.medianBlur, range(5, 60, 10))
# param_search(sample, cv2.bilateralFilter, range(19, 70, 10), range(29, 140, 20), 122)

  if param_2_range == None:
    param_2_range = [None]
  param_1_choice_count = len(param_1_range)
  param_2_choice_count = len(param_2_range)
  

  print('Generating',param_1_choice_count,'by',param_2_choice_count,'grid...')

  if param_layout == None:
    if param_2_range[0] == None:
      param_layout = 1
    else:
      param_layout = 12
  param_layout = ','.join(['param_'+num for num in list(str(param_layout))])

  plt.figure(figsize=(30,30),dpi=150)

  ax_counter = 1
  for param_1 in param_1_range:
    for param_2 in param_2_range:

      ax = plt.subplot(param_1_choice_count, param_2_choice_count, ax_counter)
      ax_counter += 1
      
      params = eval(param_layout)
      if ',' in param_layout:
        processed_img = func(img,*params)
      else:
        processed_img = func(img,params)

      if len(processed_img.shape)==2:
        ax.imshow(processed_img, cmap='gray', vmin=0, vmax=255)
      else:
        ax.imshow(processed_img)

      subplot_title = str(param_1) if param_2 == None else str(param_1)+','+str(param_2)
      ax.title.set_text(subplot_title)
  
  func_name = str(func).split()[-1].strip('>')
  plt.suptitle('function: '+func_name+', param_layout: '+param_layout, x = 0.5, horizontalalignment = 'center', y=0.9)

  save_graph()  

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

def find_contours(img, min_area_size = 1000, max_area_size = None, top_k = None, color_mode = 'rainbow', border_width = 2, show = True, only_exterior = False):

  img = img.copy()

  if only_exterior:
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  else:
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  contours = [cnt for cnt in contours if cv2.contourArea(cnt)>=min_area_size]
  if max_area_size != None:
    contours = [cnt for cnt in contours if cv2.contourArea(cnt)<=max_area_size]

  contours = sorted(contours, key=cv2.contourArea, reverse = True)
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

def otsu_threshold(img):
  threshold_value, binarized_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  print('The threshold value found is', threshold_value)
  return binarized_img

def find_lines(img, canny_lower_thresh = 50, canny_upper_thresh = 200, rho = 1, theta = np.pi / 180, threshold = 25, min_line_length = 100, max_line_gap = 20, show = True, return_lines = False, return_layer = True):
  # rho              # distance resolution in pixels of the Hough grid
  # theta            # angular resolution in radians of the Hough grid
  # threshold        # minimum number of votes (intersections in Hough grid cell)
  # min_line_length  # minimum number of pixels making up a line
  # max_line_gap     # maximum gap in pixels between connectable line segments

  img = img.copy()

  edges = cv2.Canny(img,canny_lower_thresh,canny_upper_thresh)

  # `lines` contain endpoints of detected line segments
  lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

  if show:
    # draw lines on a RGB layer
    lines_layer_rgb = np.zeros((*img.shape, 3), dtype = np.uint8)
    for line in lines:
      for x1,y1,x2,y2 in line:
        cv2.line(lines_layer_rgb,(x1,y1),(x2,y2),(255,0,0),1)
    overlayed = cv2.addWeighted(grey_to_rgb(img), 0.8, lines_layer_rgb, 1, 0)
    imshow(overlayed)

  if return_lines:
    return lines

  if return_layer:
    # draw lines on a binary layer
    lines_layer = np.zeros(img.shape, dtype = np.uint8)
    for line in lines:
      for x1,y1,x2,y2 in line:
        cv2.line(lines_layer,(x1,y1),(x2,y2),255,1)
    return lines_layer

def invert_binary(img):
  return cv2.bitwise_not(img)

def approximate_contours(contours = [], precision_level = 0.01, border_width = 2, show = False, img = None):

  approx_contours = []
  for cnt in contours:
    hull = cv2.convexHull(cnt)
    epsilon = precision_level*cv2.arcLength(hull,True)
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

def draw_many_contours(img, contours, dpi=None, border_width=2, n_colors = 8, font_scale = 1, is_bgr = True):
  
  color_range = range(1,n_colors*10+1,n_colors)
  colors = [hsv2rgb(num/100) for num in color_range]

  if len(img.shape)==2:
    colored_img = grey_to_bgr(img)
  elif len(img.shape)==3:
    if is_bgr:
      colored_img = img.copy()
    else:
      colored_img = rgb_to_bgr(img)

  for i in range(len(contours)):
    cnt = contours[i]
    color = colors[i%n_colors]
    colored_img = cv2.drawContours(colored_img, [cnt], 0, color, border_width)

    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    text_position = (cx, cy)
    
    text_content = str(i)
    font_family = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_scale
    color = color
    thickness = 2
    line_type = cv2.LINE_AA

    colored_img = cv2.putText(colored_img, text_content , text_position, font_family, font_scale, color, thickness, line_type)
  
  if dpi != None:
    imshow(colored_img, dpi = dpi)
  else:
    imshow(colored_img)

def keep_segments_longer_than(cnt, min_segment_len):
  cnt_info = get_contour_info_df(cnt)
  preserved_point_indices = flatten_list( cnt_info.loc[cnt_info['distance']>=min_segment_len,'point_index'].tolist() )
  preserved_cnt = cnt[preserved_point_indices]
  return preserved_cnt

def rgb_code_to_hsv_code(rgb_tuple):
  pixel = np.zeros((1,1,3),dtype=np.uint8)
  pixel[0,0,:] = rgb_tuple
  return tuple([int(v) for v in list(cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0][0])])

def limit_within(value, vmin, vmax):
  return max(min(value, vmax), vmin)

def create_range_around(hsv_code, radius = (3,10,10)):
  lower_bound, upper_bound = [], []
  for i in range(len(hsv_code)):

    lower_value = hsv_code[i]-radius[i]
    upper_value = hsv_code[i]+radius[i]

    if i == 0:
      lower_value = lower_value % 180
      upper_value = upper_value % 180
    else:
      lower_value = limit_within(lower_value, 0, 255)
      upper_value = limit_within(upper_value, 0, 255)

    lower_bound.append(lower_value)
    upper_bound.append(upper_value)

  return tuple(lower_bound), tuple(upper_bound)

def find_area_of_color(img, hsv_cde, radius, alpha = 0.5, dpi = 150, overlay = True, show = True, return_mask = False):
  img = img.copy()
  img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  lower_bound, upper_bound = create_range_around(hsv_code = hsv_cde, radius = radius)

  if lower_bound[0]>upper_bound[0]: # for color hue like red that are on the edge of hue range
    lower_mask = cv2.inRange(img_hsv, (0,lower_bound[1],lower_bound[2]), (lower_bound[0],upper_bound[1],upper_bound[2]))
    upper_mask = cv2.inRange(img_hsv, (upper_bound[0],lower_bound[1],lower_bound[2]), (180,upper_bound[1],upper_bound[2]))
    mask = cv2.bitwise_or(lower_mask, upper_mask)
  else:  # for other color not on the edge
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound )

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

clear_output()
print('OpenCV Version:',cv2.__version__)
print('\nEnvironment ready, happy exploring!\n\n')


from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq

def analyze_color(input_image, transparency_threshold = 50, plot_3d = False, plot_bar = True, n_cluster = None, max_cluster = 10, ignore_pure_black = True, use_sample = True, return_colors = True):

  # Copy to prevent modification (useful but mechanism needs clarification)
  input_image = input_image.copy()

  # Check input shape
  assert(len(input_image.shape) == 3)
  assert(input_image.shape[-1] in {3,4})

  # Turn color info of pixels into dataframe, filter by transparency if RGBA image is passed
  if input_image.shape[-1] == 4:
    color_df = pd.DataFrame(input_image.reshape(-1,4), columns=list('rgba'))
    # Get the rgb info of pixels in the non-transparent part of the image
    color_df = color_df[color_df['a']>=transparency_threshold]
  if input_image.shape[-1] == 3:
    color_df = pd.DataFrame(input_image.reshape(-1,3), columns=list('rgb'))

  if ignore_pure_black:
    color_df = color_df[~((color_df['r']==0)&(color_df['g']==0)&(color_df['b']==0))]

  # Handle large pixel color_df
  if not use_sample and len(color_df)>1e5:
      sample_or_not = (input('Large image detected, would you like to sample the pixels in this image? (Y/N) ')).lower()[0] == 'y'
      if sample_or_not:
        print('Sampled 100,000 pixels from the image, note that you can also resize the image before passing it to this function.')
        color_df = color_df.sample(n = int(1e5), random_state = 0)
      else:
        print('Not sampling performed, but note that rendering 3D plot for the pixels may crash your session and K-means clustering will be slow.')

  # Get std for reverse-transform the kmeans results to a meaningful rgb palette
  r_std, g_std, b_std = color_df[list('rgb')].std()
  reverse_whiten_array = np.array((r_std, g_std, b_std))

  # Normalize observations on a per feature basis, forcing features to have unit variance
  # Doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.whiten.html
  for color in list('rgb'):
    color_df['scaled_'+color] = whiten(color_df[color])


  ## 3D scatter plot showing color groups
  if plot_3d:
    trace = go.Scatter3d(x=color_df['r'], y=color_df['g'], z=color_df['b'], mode='markers',
                        marker=dict(color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(color_df['r'].values, color_df['g'].values, color_df['b'].values)],
                                    size=1, opacity=1))
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

  ## Use K-means to identify main colors
  cluster_centers_list = []
  avg_distortion_list = []

  if n_cluster != None:
    n_cluster_range = [n_cluster-1] # note minus 1 to get exactly n
  else:
    n_cluster_range = range(max_cluster+1)

  if plot_bar:
    # Initialize plt graph
    f, ax = plt.subplots(len(n_cluster_range), 1, figsize=(10,10))

  for n in n_cluster_range:

    ###### Train clusters ######

    cluster_centers, avg_distortion = kmeans(color_df[['scaled_r', 'scaled_g', 'scaled_b']], n + 1)

    ###### Assign labels ######

    labels, distortions = vq( color_df[['scaled_r', 'scaled_g', 'scaled_b']] , cluster_centers)

    color_df['label'] = labels
    color_df['distortion'] = distortions

    ###### Build palette ######

    # These parameter affects visual style only and can be exposed to user later
    height = 200
    width = 1000
    gap_size = 5
    palette = np.zeros((height, width, 3), np.uint8)

    # Count how many pixels falls under which category, let this decides the color's relative width in the palette
    cluster_proportion = color_df['label'].value_counts().sort_index()/len(color_df)
    cluster_width_list = (cluster_proportion * width).to_list()
    cluster_width_list = [int(x) for x in saferound(cluster_width_list, places=0)]

    # Reorder clusters and widths according to the proportion, largest to smallest
    reordered_cluster_df = pd.DataFrame(zip(cluster_centers, cluster_width_list),columns=['cluster','width']).sort_values('width',ascending=False)
    cluster_centers = reordered_cluster_df['cluster'].tolist()
    cluster_width_list = reordered_cluster_df['width'].tolist()

    # Storing information
    cluster_centers_list.append(cluster_centers)
    avg_distortion_list.append(avg_distortion)

    if plot_bar:
      # Coloring the palette canvas based on color and width
      endpoints = list(np.cumsum(cluster_width_list))
      startpoints = [0]+endpoints[:-1]
      for cluster_index in range(len(cluster_centers)):
        # Notice here we apply the reverse_whiten_array to get meaningful RGB colors
        palette[:, startpoints[cluster_index]+gap_size:endpoints[cluster_index], :] = cluster_centers[cluster_index] * reverse_whiten_array
        palette[:, startpoints[cluster_index]:startpoints[cluster_index]+gap_size, :] = (255,255,255)

      # Displaying the palette when performing K-means with parameter n
      if n_cluster != None:
        ax.imshow(palette)
        ax.axis('off')
      else:
        ax[n].imshow(palette)
        ax[n].axis('off')

  if plot_bar:
    ### Show the entire palette
    f.tight_layout()
    plt.show()
    ### Show the elbow plot for choosing best n_cluster parameter for K-means
    fig = plt.figure()
    plt.scatter(x = n_cluster_range, y = avg_distortion_list)
    fig.suptitle('Elbow Plot for K-means')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Distortion')
    print()

  if return_colors:
    if n_cluster != None:
      return (cluster_centers_list[0]*reverse_whiten_array).astype(np.uint8)
    else:
      return [(cluster_centers*reverse_whiten_array).astype(np.uint8) for cluster_centers in cluster_centers_list]

def recover_contour_from_string(contour_string):
  list_of_points = [list(point.split(',')) for point in contour_string.split('|')]
  return np.array(list_of_points, dtype=np.int32)

def flood_fill(img, seed_pixel, return_mask = False, fill_value = (0,0,0), color_variation = 5, neighbor = 4):

  if isinstance( seed_pixel, list ):
    seed_pixel_list = seed_pixel
  else:
    seed_pixel_list = [seed_pixel]

  flood_img = img.copy()

  if fill_value != (0,0,0):
    black_pixels = np.where((flood_img[:, :, 0] == 0) & (flood_img[:, :, 1] == 0) & (flood_img[:, :, 2] == 0))
    flood_img[black_pixels] = fill_value

  h, w = flood_img.shape[:2]

  mask_list = []
  for seed_pixel in seed_pixel_list:
    num, flood_img, mask, rect = cv2.floodFill(flood_img, np.zeros((h+2,w+2),np.uint8), seed_pixel, fill_value, tuple([color_variation]*3), tuple([color_variation]*3), neighbor)
    mask_list.append(mask)

  combined_mask = mask_list.pop(0)
  while mask_list:
    combined_mask = cv2.bitwise_or(combined_mask, mask_list.pop())
  combined_mask = combined_mask[1:-1,1:-1]
  combined_mask = combined_mask*255

  if return_mask:
    return flood_img, combined_mask
  return flood_img

def adaptive_threshold(img, size = 9, C = 18):
  return cv2.adaptiveThreshold(img.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, size, C)

def mask_with_contours(img, contours):
  img = img.copy()
  mask_color = 255 if len(img.shape)==2 else (255,255,255) if len(img.shape)==3 else 255
  contours_mask = cv2.drawContours(np.zeros(img.shape, dtype=np.uint8), contours, -1, mask_color, -1)
  masked_img = cv2.bitwise_and(img, contours_mask)
  return masked_img

def get_vicinity(img,pixel_pos,radius): # pixel_pos in (x,y) format
  x, y = pixel_pos
  vicinity = img[y-radius:y+radius+1, x-radius:x+radius+1].copy()
  return vicinity

def random_sample_position_of_certain_value(ndarray, value, format = 'xy'):
  yx = np.unravel_index( np.random.choice(np.where(ndarray.flatten() == value)[0]), ndarray.shape )
  if format == 'xy':
    return tuple(reversed(yx))
  elif format == 'yx':
    return yx

def rgb_code_to_lab_code(rgb_tuple):
  pixel = np.zeros((1,1,3),dtype=np.uint8)
  pixel[0,0,:] = rgb_tuple if isinstance(rgb_tuple,tuple) else tuple(rgb_tuple)
  return tuple([int(v) for v in list(cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0][0])])

def lab_code_to_rgb_code(lab_tuple):
  pixel = np.zeros((1,1,3),dtype=np.uint8)
  pixel[0,0,:] = lab_tuple if isinstance(lab_tuple,tuple) else tuple(lab_tuple)
  return tuple([int(v) for v in list(cv2.cvtColor(pixel, cv2.COLOR_LAB2RGB)[0][0])])

def stringify_contour(contour):
  return '|'.join([','.join(map(str,list(point[0]))) for point in contour])


def get_length_of_segments(cnt, order = 'original'):
  cnt_with_head_at_tail = np.array(list(cnt)+list(cnt[:1]))
  length_of_segments = list(np.round(np.sqrt(np.sum(np.square((cnt_with_head_at_tail[:-1] - cnt_with_head_at_tail[1:])), axis=2)).flatten(),1))
  if order.startswith('orig'):
    return length_of_segments
  elif order.startswith('asc'):
    return sorted(length_of_segments)
  elif order.startswith('des'):
    return sorted(reversed(length_of_segments))
