from IPython.display import clear_output

# ------------------------------------Import Libraries----------------------------------------

# pip install folium
import folium

def show_map(data):

  orig_data_length = len(data)
  data = df.dropna(subset=['coordinates'])
  print(orig_data_length - len(data), 'records are dropped due to lack of cooridnates data.')

  # aerial
  # tile_style = 'https://maps.nyc.gov/xyz/1.0.0/photo/1924/{z}/{x}/{y}.png8'
  # old map
  # tile_style = 'https://{s}.tile.thunderforest.com/pioneer/{z}/{x}/{y}.png'
  # black and white
  tile_style = 'Stamen Toner'

  map_center = np.array(data['coordinates'].tolist()).mean(axis=0)
  m = folium.Map(location=map_center, tiles=tile_style, attr='... contributors', control_scale=True)

  place_type_to_color_mapping = {'Home': 'green', 'Restaurant': 'orangered', 'Non-restaurant Business': 'blue'}

  for _, row in data.iterrows():
    folium.Circle(row['coordinates'], radius=1, color=place_type_to_color_mapping.get(row['place_type'], 'grey'), tooltip=f"Name: {row['Name FULL']}<br>Address: {row['Address']}<br>Type: {row['place_type']}<br>HBCR: {row['HBCR']}<br>Year: {int(row['YR'])}<br>FID: {int(row['FID'])}").add_to(m)

  return m
