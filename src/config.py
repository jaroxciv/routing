# src/config.py

from pathlib import Path

# Data directories
DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")

# File paths
DELIVERIES_PATH = DATA_DIR / "deliveries_utm44.gpkg"
ROADS_PATH = DATA_DIR / "hyd_roads.gpkg"
HUBS_PATH = DATA_DIR / "ApexHub_Deliveries.xlsx"
GRAPHML_PATH = DATA_DIR / "hyd_road_graph.graphml"
EDGES_PICKLE_PATH = DATA_DIR / "edges_exp_3857.pkl"
NODES_PICKLE_PATH = DATA_DIR / "nodes_exp_3857.pkl"

# City/buffer settings
CITY_NAME = "Hyderabad, Telangana, India"
BUFFER_METERS = 10_000

WGS84_CRS = 4326
PROJ_CRS = 3857          # For spatial ops (buffering, clustering in meters)
PLOTTING_CRS = 3857      # Web Mercator for basemap compatibility

# Clustering settings
CLUSTER_METHOD = "haversine"  # or "kmeans", "dbscan", etc.
PINCODES = ["500084", "500032", "500081", "500001", "500033"]
N_CLUSTERS = len(PINCODES)    # Only used if method is kmeans, etc.
CLUSTER_PARAMS = {"eps": 0.01, "min_samples": 5}

SKIP_CLUSTERS = ["500001"]  # List of cluster labels/pincodes to ignore
