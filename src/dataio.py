# src/dataio.py

import osmnx as ox
import pandas as pd
import geopandas as gpd
import pyogrio
from .config import *
from .logging_utils import get_logger

logger = get_logger("DataIO")

def read_gpkg_auto(path, layer=None, **kwargs):
    """
    Reads a GeoPackage, auto-selecting the layer if not provided.
    Returns a GeoDataFrame.
    """
    layers = pyogrio.list_layers(str(path))
    if layer is None:
        if len(layers) == 1:
            layer = layers[0]
            logger.info(f"Auto-selected only layer: {layer}")
        else:
            raise ValueError(
                f"GeoPackage has multiple layers: {layers}. "
                "Specify 'layer' argument explicitly."
            )
    elif layer not in layers:
        raise ValueError(f"Layer '{layer}' not found in {layers}")
    return gpd.read_file(path, layer=layer, engine="pyogrio", **kwargs)

def load_deliveries(path=DELIVERIES_PATH, layer=None, crs=PROJ_CRS):
    """Load deliveries GeoDataFrame with CRS conversion. Auto-selects layer if needed."""
    gdf = read_gpkg_auto(path, layer=layer, use_arrow=True)
    logger.info(f"Loaded {len(gdf)} deliveries.")
    return gdf.to_crs(crs)

def load_city_boundary(crs=PROJ_CRS):
    """Load the city administrative boundary as a GeoDataFrame."""
    gdf = ox.geocode_to_gdf(CITY_NAME)
    return gdf.to_crs(crs)

def load_roads(crs=PROJ_CRS):
    """Load road GeoDataFrame."""
    return gpd.read_file(ROADS_PATH, engine="pyogrio", use_arrow=True).to_crs(crs)

def load_hub(crs=PROJ_CRS):
    """Load hub location as GeoDataFrame and lat/lon."""
    df_dc = pd.read_excel(HUBS_PATH, sheet_name='Distribution Center Location')
    hub_lat, hub_lon = map(float, df_dc.loc[0, 'Hub Coordinates'].split(','))
    hub_gdf = gpd.GeoDataFrame(
        {'geometry': [gpd.points_from_xy([hub_lon], [hub_lat])[0]]}, crs='EPSG:4326'
    ).to_crs(crs)
    logger.info(f"Loaded hub at {hub_lat}, {hub_lon} in CRS EPSG:{crs}")
    return hub_gdf, hub_lat, hub_lon

# ---- ROAD GRAPH LOADING/SAVING ----
def get_buffered_city_polygon(admin_gdf, buffer_meters=BUFFER_METERS):
    city_3857 = admin_gdf.to_crs(PROJ_CRS)
    buffered_3857 = city_3857.geometry.buffer(buffer_meters)
    gdf_3857 = gpd.GeoDataFrame(geometry=buffered_3857, crs=f"EPSG:{PROJ_CRS}")
    gdf_wgs = gdf_3857.to_crs(epsg=4326)
    return gdf_wgs.geometry.iloc[0]  # Return shapely polygon

def save_graph(G, path=GRAPHML_PATH):
    ox.save_graphml(G, path)

def load_graph(path=GRAPHML_PATH):
    return ox.load_graphml(path)

def get_or_create_osmnx_graph(admin_gdf, force_rebuild=False):
    """
    Returns: G, nodes_gdf, edges_gdf (all in EPSG:4326)
    Buffers admin, checks cache, builds/saves if needed.
    """
    if not GRAPHML_PATH.exists() or force_rebuild:
        logger.info("Building and saving new OSMnx graph...")
        polygon = get_buffered_city_polygon(admin_gdf)
        G = ox.graph_from_polygon(polygon, network_type='drive')
        save_graph(G, GRAPHML_PATH)
    else:
        logger.info("Loading cached OSMnx graph...")
        G = load_graph(GRAPHML_PATH)
    nodes, edges = ox.graph_to_gdfs(G)
    logger.info(f"Loaded graph with {len(nodes)} nodes and {len(edges)} edges")
    return G, nodes, edges
