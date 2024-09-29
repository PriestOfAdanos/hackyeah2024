from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import osmnx as ox
import networkx as nx
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from geopy.geocoders import Nominatim
import csv
import time
from typing import Dict
import os

# Create FastAPI instance
app = FastAPI()

# Initialize the Nominatim geocoder
geolocator = Nominatim(user_agent="my_geocoder")

@app.get("/geocode/")
def geocode_csv_addresses() -> Dict:
    """
    Endpoint to geocode addresses from a CSV file on the local filesystem.
    The CSV file is expected to have addresses in the first column.
    
    :return: JSON response containing latitude and longitude for addresses.
    """
    filename = 'worst_edges.csv'
    # Verify if the file exists
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        output_dict = {}
        # Open and read the CSV file
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile_in:
            reader = csv.reader(csvfile_in)

            # Process each row in the CSV
            for row in reader:
                address = row[0]  # Select the first element of the line
                try:
                    # Use Nominatim to geocode the address
                    location = geolocator.geocode(f"Krakow, {address}")
                    if location:
                        latitude = location.latitude
                        longitude = location.longitude
                        # Store the result in the dictionary with a string key
                        key = f"{latitude},{longitude}"
                        output_dict[key] = row[1] if len(row) > 1 else None
                except Exception as e:
                    print(f"Error geocoding {address}: {e}")

                # Sleep to respect Nominatim's usage policy
                time.sleep(1)

        return JSONResponse(content=output_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {e}")


# Load the pickled graph
graph_filename = "graph_G.pkl"
try:
    with open(graph_filename, 'rb') as file:
        G = pickle.load(file)
    print(f"Graph G has been successfully loaded from {graph_filename}.")
except FileNotFoundError:
    print(f"Error: The file {graph_filename} was not found.")
    G = None

# Define a request body model using Pydantic
class Location(BaseModel):
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float

# Function to find the nearest node on the graph
def get_nearest_node(graph, lat, lon):
    return ox.distance.nearest_nodes(graph, lon, lat)
# Define the list of allowed origins
origins = [
    "http://localhost:3000",  # Allow your frontend running on localhost
    "https://yourdomain.com",  # Allow your production domain
    "*",  # Allow all origins (not recommended for production)
]

# Add CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Origins that are allowed to make requests
    allow_credentials=True,  # Allow cookies to be sent cross-origin
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the FastAPI endpoint
@app.post("/get_path")
async def get_path(location: Location):
    if G is None:
        raise HTTPException(status_code=500, detail="Graph data is not loaded.")

    try:
        # Get the closest nodes for origin and destination
        orig_node = get_nearest_node(G, location.origin_lat, location.origin_lon)
        dest_node = get_nearest_node(G, location.dest_lat, location.dest_lon)
        
        # Find the shortest path using the length as the weight
        route = nx.shortest_path(G, orig_node, dest_node, weight='length')

        # Extract the route nodes' coordinates
        route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
        
        return {
            "origin": {"lat": location.origin_lat, "lon": location.origin_lon},
            "destination": {"lat": location.dest_lat, "lon": location.dest_lon},
            "path": route_coords
        }

    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="No path found between the given points.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run using: uvicorn filename:app --host 0.0.0.0 --port 8000
