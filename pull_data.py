import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import folium  # To visualize on a map
import matplotlib.pyplot as plt
import geodesk
import geopandas as gpd
import pandas as pd
import googlemaps
import osmnx as ox
import networkx as nx
from pathlib import Path
import getpass
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.plot
from rasterio import features
from math import sqrt

from pyproj import Transformer
from scipy.spatial import cKDTree
from numba import njit

# Load the pixels_dict from the pickle file
pickle_file = "pixels_dict.pkl"

with open(pickle_file, 'rb') as file:
    pixels_dict = pickle.load(file)

# Convert pixels_dict to arrays for efficient querying
pixels_points = np.array(list(pixels_dict.keys()))  # Array of [lat, lon]
pixels_values = np.array(list(pixels_dict.values()))  # Array of NO2 values

# Build a KDTree for efficient nearest neighbor search
pixels_tree = cKDTree(pixels_points)

# Function to get NO2 value from a point using KDTree
def get_value_from_point(lat, lon):
    distance, index = pixels_tree.query([lat, lon])
    return pixels_values[index]

# Build the road network graph using OSMnx
place_name = "Krakow, Poland"
G = ox.graph_from_place(place_name, network_type='bike')
H = ox.graph_from_place(place_name, network_type='bike')
# Extract node data into arrays
node_ids = []
latitudes = []
longitudes = []

for node_id, data in G.nodes(data=True):
    node_ids.append(node_id)
    latitudes.append(data['y'])
    longitudes.append(data['x'])

node_ids = np.array(node_ids)
latitudes = np.array(latitudes)
longitudes = np.array(longitudes)

# Create a mapping from node IDs to indices for quick lookup
node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

# Extract edge data into arrays
edge_u = []
edge_v = []
edge_lengths = []

for u, v, data in G.edges(data=True):
    edge_u.append(u)
    edge_v.append(v)
    edge_lengths.append(data['length'])

edge_u = np.array(edge_u)
edge_v = np.array(edge_v)
edge_lengths = np.array(edge_lengths)

# Map edge node IDs to indices
edge_u_idx = np.array([node_id_to_idx[u] for u in edge_u])
edge_v_idx = np.array([node_id_to_idx[v] for v in edge_v])

# Query the KDTree to get NO2 values for each node
node_positions = np.column_stack((latitudes, longitudes))
_, indices = pixels_tree.query(node_positions)
node_no2_values = pixels_values[indices]

# Use Numba to speed up the edge length adjustments
@njit
def adjust_edge_lengths(edge_u_idx, edge_v_idx, edge_lengths, node_no2_values):
    edge_lengths_adjusted = np.empty_like(edge_lengths)
    for i in range(edge_lengths.shape[0]):
        u_idx = edge_u_idx[i]
        v_idx = edge_v_idx[i]
        # Average NO2 value from both nodes
        no2_value = (node_no2_values[u_idx] + node_no2_values[v_idx]) / 2.0
        edge_lengths_adjusted[i] = edge_lengths[i] * (1 + no2_value)
    return edge_lengths_adjusted

# Adjust the edge lengths
edge_lengths_adjusted = adjust_edge_lengths(edge_u_idx, edge_v_idx, edge_lengths, node_no2_values)

# Update the graph with the new edge lengths
for i in range(len(edge_u)):
    u = edge_u[i]
    v = edge_v[i]
    G[u][v][0]['length'] = edge_lengths_adjusted[i]  # Use [0] if G is a MultiDiGraph


# Compute the sum of lengths of all edges in graph G
total_length_G = sum(data['length'] for _, _, data in G.edges(data=True) if 'length' in data)

# Compute the sum of lengths of all edges in graph H
total_length_H = sum(data['length'] for _, _, data in H.edges(data=True) if 'length' in data)

# Subtract the total lengths
length_difference = total_length_G - total_length_H

# Print the results
print(f"Total length of all edges in G: {total_length_G}")
print(f"Total length of all edges in H: {total_length_H}")
print(f"Difference in total lengths (G - H): {length_difference}")


# import requests
# import json
# import pandas as pd
# from bs4 import BeautifulSoup
# import folium  # To visualize on a map
# import matplotlib.pyplot as plt
# import geodesk
# import geopandas as gpd
# import pandas as pd
# import googlemaps
# import osmnx as ox
# import networkx as nx
# from pathlib import Path
# import getpass
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import rasterio
# import rasterio.plot
# from rasterio import features
# from math import sqrt

# from pyproj import Transformer
 

# # Specify the file name where the dictionary was saved
# pickle_file = "pixels_dict.pkl"

# # Open the file in read-binary mode and load the dictionary
# with open(pickle_file, 'rb') as file:
#     pixels_dict = pickle.load(file)

# def get_value_from_point(lat, lon):
#     # If the exact point exists, return it
#     if (lat, lon) in pixels_dict:
#         return pixels_dict[(lat, lon)]
    
#     # Calculate the nearest neighbor in the dictionary
#     nearest_point = None
#     min_distance = float('inf')

#     for point in pixels_dict:
#         point_lat, point_lon = point
#         distance = sqrt((lat - point_lat)**2 + (lon - point_lon)**2)
        
#         if distance < min_distance:
#             min_distance = distance
#             nearest_point = point

#     # Return the value from the nearest point
#     return pixels_dict[nearest_point]

# place_name = "Krakow, Poland"
# # If using place name
# G = ox.graph_from_place(place_name, network_type='drive')

# # If using coordinates
# # G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
# # Simple plot
# # ox.plot_graph(G)

# # For an interactive plot (requires folium)
# # m = ox.plot_graph_folium(G, popup_attribute='name', weight=2, color='blue')
# # m.save('map.html')
# # Get basic stats
# stats = ox.basic_stats(G)
# print(stats)

# # Find the shortest path between two nodes
# orig_node = list(G.nodes())[0]
# dest_node = list(G.nodes())[-1]
# route = nx.shortest_path(G, orig_node, dest_node, weight='length')

# # Plot the route
# # fig, ax = ox.plot_graph_route(G, route)
# # Save the graph as a GraphML file
# # ox.save_graphml(G, filepath='road_network.graphml')

# # Load the graph from the GraphML file
# # G = ox.load_graphml('road_network.graphml')

# node_no2_values = {}
# count = 0 
# for node, data in G.nodes(data=True):
#     # Extract the latitude and longitude from the node attributes
#     lat = data['y']
#     lon = data['x']

#     edges = G.edges(node, data=True)

#     # Get the NO2 value for this point
#     no2_value = get_value_from_point(lat, lon)
#     for u, v, data in edges:
#         data['length'] *= (1+no2_value)

# print(count)





###############


# # Helper function to get data from a URL
# def get_html(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         return response.text
#     except requests.RequestException as e:
#         print(f"Failed to retrieve data from {url}: {e}")
#         return None

# # Example: Getting data from Google Maps Traffic
# def get_google_maps_traffic(api_key, location):
#     try:
#         endpoint = f"https://maps.googleapis.com/maps/api/directions/json"
#         params = {
#             'origin': location['origin'],
#             'destination': location['destination'],
#             'key': api_key,
#             'departure_time': 'now',  # To get real-time traffic
#         }
#         response = requests.get(endpoint, params=params)
#         if response.status_code == 200:
#             data = response.json()
#             print("Google Maps traffic data successfully retrieved.")
#             return data
#         else:
#             print(f"Error: {response.status_code} when trying to retrieve Google Maps data.")
#     except requests.RequestException as e:
#         print(f"Failed to retrieve Google Maps Traffic data: {e}")
#         return None

# # Example: Scraping Accident Data
# def get_accident_data():
#     url = "https://obserwatoriumbrd.pl/mapa-wypadkow/"
#     html_content = get_html(url)
#     if html_content:
#         soup = BeautifulSoup(html_content, 'html.parser')
#         # Again, need to find out the structure of the data
#         print("Successfully fetched Accident data.")



# # Example: Data Integration and Processing
# def integrate_data():
#     # Placeholder to demonstrate how data might be processed
#     traffic_data = get_google_maps_traffic(api_key='AIzaSyB6HhxmD-kuyqcEJNifwNn9txouTnYWvPQ', location={'origin': 'Krakow', 'destination': 'Zakopane'})
#     accident_data = get_accident_data()
#     # More integration logic and combining datasets
#     print("Data integrated successfully.")

# # Visualization Example
# def visualize_data():
#     # Example using folium to visualize traffic incidents on a map
#     m = folium.Map(location=[50.0647, 19.9450], zoom_start=10)
#     # Add markers, polylines, etc. to represent traffic or accident data.
#     folium.Marker([50.0647, 19.9450], popup="Sample Traffic Incident").add_to(m)
#     m.save("traffic_map.html")
#     print("Map visualization saved as traffic_map.html")

# # Main function to call all other functions
# def main():
#     print("Fetching ViaMichelin Traffic Data:")
    
#     print("\nFetching Accident Data:")
#     print(get_accident_data())
    
#     print("\nFetching Strava Heatmap Data:")
    
#     print("\nIntegrating All Retrieved Data:")
#     print(integrate_data())
    
#     print("\nVisualizing Data on Map:")
#     print(visualize_data())

# if __name__ == "__main__":
#     main()
