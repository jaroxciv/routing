from src.dataio import load_deliveries, load_city_boundary
from src.utils import filter_deliveries_within_boundary
from src.config import CLUSTER_METHOD, PINCODES, OUTPUTS_DIR, PROJ_CRS
from src.clustering import (
    geocode_pincodes,
    make_pincode_centroid_gdf,
    assign_deliveries_to_clusters
)
import os

if __name__ == "__main__":
    # Load city boundary and deliveries
    hyd_admin = load_city_boundary(crs=PROJ_CRS)
    deliveries = load_deliveries(layer="deliveries_utm44")

    # Filter deliveries within the city boundary
    deliveries_in_city, outliers = filter_deliveries_within_boundary(deliveries, hyd_admin)

    # Geocode pincodes and build centroids GeoDataFrame
    pincode_coords = geocode_pincodes(PINCODES)
    centroids_gdf = make_pincode_centroid_gdf(pincode_coords, crs=PROJ_CRS)

    # Cluster using your selected method
    deliveries_clustered = assign_deliveries_to_clusters(
        deliveries_in_city,
        centroids=centroids_gdf,
        method=CLUSTER_METHOD
    )

    # Save the clustered deliveries to outputs
    output_path = os.path.join(OUTPUTS_DIR, "deliveries_clustered.gpkg")
    deliveries_clustered.to_file(output_path, driver="GPKG")
