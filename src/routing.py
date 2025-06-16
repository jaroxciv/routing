import time
import numpy as np
import networkx as nx
import osmnx as ox
from src.logging_utils import get_logger

logger = get_logger("Routing")

def precompute_distance_matrix(G, node_list):
    """Compute all-pairs shortest path lengths between given nodes."""
    logger.info(f"Precomputing distance matrix for {len(node_list)} nodes.")
    node_idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)
    dist_matrix = np.full((num_nodes, num_nodes), np.inf)
    for i, n1 in enumerate(node_list):
        lengths = nx.single_source_dijkstra_path_length(G, n1, weight='length')
        for n2, l in lengths.items():
            if n2 in node_idx:
                j = node_idx[n2]
                dist_matrix[i, j] = l
    logger.info("Distance matrix computed.")
    return dist_matrix, node_idx

def fast_nearest_neighbor_route(
        deliveries_gdf, 
        hub_gdf, 
        G,
        algorithm, 
        cluster_label=None, 
        debug=False
        ):
    """
    MUCH FASTER: Precompute all-pair shortest paths and run NN.
    """
    logger.info(
        f"Routing cluster {cluster_label} with {len(deliveries_gdf)} deliveries..."
        )

    deliv_wgs = deliveries_gdf.to_crs(epsg=4326)
    hub_wgs = hub_gdf.to_crs(epsg=4326)
    hub_point = hub_wgs.geometry.iloc[0]
    hub_node = ox.nearest_nodes(G, hub_point.x, hub_point.y)
    deliv_nodes = list(ox.nearest_nodes(G, deliv_wgs.geometry.x, deliv_wgs.geometry.y))
    node_list = [hub_node] + deliv_nodes

    # Precompute distance matrix
    t0 = time.time()
    dist_matrix, node_idx = precompute_distance_matrix(G, node_list)
    t1 = time.time()
    logger.info(f"Distance matrix ready (in {t1 - t0:.2f}s). Starting route calculation.")

    current_idx = 0  # start at hub
    unvisited = set(range(1, len(node_list)))  # all deliveries
    route_idxs = [current_idx]
    total_length = 0.0

    t2 = time.time()
    while unvisited:
        # Find closest unvisited
        dists = {j: dist_matrix[current_idx, j] for j in unvisited}
        next_idx = min(dists, key=dists.get)
        total_length += dists[next_idx]
        route_idxs.append(next_idx)
        current_idx = next_idx
        unvisited.remove(next_idx)
    t3 = time.time()

    # Return to hub
    total_length += dist_matrix[current_idx, 0]
    route_idxs.append(0)
    t4 = time.time()
    route_nodes = [node_list[i] for i in route_idxs]
    node_xy = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route_nodes]

    logger.info(
        f"Cluster {cluster_label} route complete. Total distance: {total_length/1000:.2f} km. "
        f"Time: {t4-t0:.2f}s (precompute: {t1-t0:.2f}s, {algorithm} solve: {t3-t2:.2f}s)"
    )

    return {
        'assigned_pincode': cluster_label,
        'algorithm': algorithm,
        'ordered_node_ids': route_nodes,
        'ordered_latlon': node_xy,
        'total_distance_km': total_length / 1000,
        'run_time_seconds': t4 - t0,
        'notes': ''
    }
