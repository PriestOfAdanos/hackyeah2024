import osmnx as ox
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rasterio import features
from math import sqrt

from pyproj import Transformer
from scipy.spatial import cKDTree
from numba import njit
import requests
import math

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="kkkkkk")



import csv
import time
from geopy.geocoders import Nominatim

def geocode_csv_addresses(csv_filename):
    """
    Reads a CSV file line by line, selects the first element of each line,
    uses Nominatim to get its longitude and latitude, and writes the results
    to an output CSV file.

    :param csv_filename: str, path to the input CSV file.
    :param output_filename: str, path to the output CSV file where results will be saved.
    """
    # Initialize the Nominatim geocoder with a user agent
    geolocator = Nominatim(user_agent="my_geocoder")
    output_dict = {}
    # Open the input and output CSV files
    with open(csv_filename, 'r', newline='', encoding='utf-8') as csvfile_in:
        reader = csv.reader(csvfile_in)

        # Process each row in the input CSV file
        for row in reader:
            address = row[0]  # Select the first element of the line
            try:
                # Use Nominatim to geocode the address
                location = geolocator.geocode(f"Krakow, {address}")
                if location:
                    latitude = location.latitude
                    longitude = location.longitude
                    output_dict[(latitude, longitude)] = row[1]
            except Exception as e:
                print(f"Error geocoding {address}: {e}")

            # Append the latitude and longitude to the row

            # Sleep to respect Nominatim's usage policy
            time.sleep(1)
    return output_dict



def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two points on the Earth.
    
    Parameters:
        lat1, lon1: Latitude and longitude of point 1 in decimal degrees
        lat2, lon2: Latitude and longitude of point 2 in decimal degrees
    
    Returns:
        Distance in kilometers between point 1 and point 2
    """
    
    R = 6371.0  # Radius of the Earth in kilometers
    
    # Convert decimal degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance

def find_nearest_value(target, points_dict):
    """
    Find the value in 'points_dict' whose key (latitude, longitude) is nearest to the 'target' point.
    
    Parameters:
        target: Tuple of (latitude, longitude) for the target point
        points_dict: Dictionary with keys as (latitude, longitude) tuples and values as associated data
    
    Returns:
        The value corresponding to the nearest key, or None if the dictionary is empty
    """
    if not points_dict:
        return None  # Return None if the dictionary is empty
    
    min_distance = 100
    nearest_value = 0
    target_lat, target_lon = target
    
    for idx, (point, value) in enumerate(points_dict.items()):
        point_lat, point_lon = point
        distance = haversine_distance(target_lat, target_lon, point_lat, point_lon)
        # Debug: Print each distance calculated
        # print(f"Distance to key {idx} ({point_lat}, {point_lon}): {distance:.2f} km")
        
        if distance < min_distance:
            # min_distance = distance
            nearest_value = int(value) * ((min_distance - distance)/min_distance)
    
    return nearest_value


geocode_dict = geocode_csv_addresses('worst_edges.csv')

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
        edge_lengths_adjusted[i] = edge_lengths[i] * (1 + (no2_value*10))
    return edge_lengths_adjusted

# Adjust the edge lengths
edge_lengths_adjusted = adjust_edge_lengths(edge_u_idx, edge_v_idx, edge_lengths, node_no2_values)

# Update the graph with the new edge lengths
for i in range(len(edge_u)):
    u = edge_u[i]
    v = edge_v[i]
    G[u][v][0]['length'] = edge_lengths_adjusted[i]  # Use [0] if G is a MultiDiGraph

# Specify the filename to save the graph
graph_filename = "graph_G.pkl"

sum=0
for u, v, key, data in G.edges(data=True, keys=True):
    # Get the coordinates of the start node
    u_lat = G.nodes[u]['y']
    u_lon = G.nodes[u]['x']

    # Get the coordinates of the end node
    v_lat = G.nodes[v]['y']
    v_lon = G.nodes[v]['x']

    # You can use either the start or end point, or an average coordinate
    lat = (u_lat + v_lat) / 2
    lon = (u_lon + v_lon) / 2

    # Find the nearest value using the coordinates
    nearest_value = find_nearest_value((lat, lon), geocode_dict)

    # Update the length attribute with the value from find_nearest_value
    if 'length' in data:
        sum+= nearest_value
        data['length'] *= (1+nearest_value)
    else:
        data['length'] = nearest_value
print(sum)

# Open the file in write-binary mode and pickle the graph
with open(graph_filename, 'wb') as file:
    pickle.dump(G, file)
