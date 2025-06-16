from src.dataio import (
    load_city_boundary, load_deliveries, load_hub, get_or_create_osmnx_graph
)
from src.routing import fast_nearest_neighbor_route
from src.config import OUTPUTS_DIR, PROJ_CRS, SKIP_CLUSTERS

import pandas as pd
import os
import pickle

if __name__ == "__main__":
    # 1. Load data
    hyd_admin = load_city_boundary(crs=PROJ_CRS)
    deliveries = load_deliveries(
        path=OUTPUTS_DIR / "deliveries_clustered.gpkg",
        layer="deliveries_clustered"
        )
    print(f"deliveries clusters: {deliveries['assigned_cluster'].unique()}")
    hub_gdf, hub_lat, hub_lon = load_hub()
    
    # 2. Load/create expanded road network graph
    G, nodes, edges = get_or_create_osmnx_graph(hyd_admin, force_rebuild=False)

    # 3. Iterate over clusters and optimize routes
    cluster_col = "assigned_cluster"  # Or "assigned_pincode"
    results = []
    for cluster_label in deliveries[cluster_col].unique():
        if cluster_label in SKIP_CLUSTERS:
            continue
        cluster_df = deliveries[deliveries[cluster_col] == cluster_label]
        if len(cluster_df) < 1:
            continue
        try:
            result = fast_nearest_neighbor_route(
                cluster_df, hub_gdf, G, "dijkstra", cluster_label=cluster_label, debug=False
            )
            results.append(result)
        except Exception as e:
            # Logging will be handled inside the module if needed
            continue

    # 4. Save results as pickle
    results_path = os.path.join(OUTPUTS_DIR, "routing_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    # 5. Save summary table as CSV
    summary_df = pd.DataFrame([
        {
            "cluster": r["assigned_pincode"],
            "algorithm": r["algorithm"],
            "distance_km": r["total_distance_km"],
            "runtime_s": r["run_time_seconds"],
            "notes": r["notes"],
        }
        for r in results
    ])
    summary_csv = os.path.join(OUTPUTS_DIR, "routing_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
