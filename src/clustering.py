import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Dict, Tuple, Optional
from .logging_utils import get_logger

logger = get_logger("Clustering")

def geocode_pincodes(pincodes: List[str], user_agent="my_geocoder") -> Dict[str, Tuple[float, float]]:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    logger.info(f"Geocoding pincodes: {pincodes}")
    geolocator = Nominatim(user_agent=user_agent)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    coords = {}
    for pin in pincodes:
        location = geocode(f"{pin}, India")
        if location:
            logger.info(f"Geocoded {pin}: {location.latitude}, {location.longitude}")
            coords[pin] = (location.longitude, location.latitude)
        else:
            logger.warning(f"Failed to geocode pincode {pin}")
    return coords

def make_pincode_centroid_gdf(pincode_coords: Dict[str, Tuple[float, float]], crs: str) -> gpd.GeoDataFrame:
    logger.info(f"Building centroid GeoDataFrame for pincodes: {list(pincode_coords.keys())}")
    df = pd.DataFrame([
        {"pincode": pin, "lon": lon, "lat": lat}
        for pin, (lon, lat) in pincode_coords.items()
    ])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])],
        crs=crs
    )
    return gdf

def haversine_matrix(deliv_lonlat: np.ndarray, centroid_lonlat: np.ndarray) -> np.ndarray:
    logger.info(f"Calculating haversine distance matrix: {deliv_lonlat.shape[0]} deliveries x {centroid_lonlat.shape[0]} centroids")
    R = 6371  # Earth radius in km
    lon1, lat1 = np.radians(deliv_lonlat[:, 0]), np.radians(deliv_lonlat[:, 1])
    lon2, lat2 = np.radians(centroid_lonlat[:, 0]), np.radians(centroid_lonlat[:, 1])
    dlon = lon1[:, None] - lon2[None, :]
    dlat = lat1[:, None] - lat2[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lat1)[:, None] * np.cos(lat2)[None, :] * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def assign_deliveries_to_clusters(
    deliveries_gdf: gpd.GeoDataFrame,
    centroids: Optional[gpd.GeoDataFrame] = None,
    n_clusters: Optional[int] = None,
    method: str = "haversine",
    delivery_crs: str = "EPSG:4326",
    random_state: int = 42,
    debug: bool = False
) -> gpd.GeoDataFrame:
    logger.info(f"Assigning deliveries to clusters using method: {method}")
    # Ensure correct CRS
    if deliveries_gdf.crs.to_string() != delivery_crs:
        logger.info(f"Reprojecting deliveries to {delivery_crs}")
        deliveries_gdf = deliveries_gdf.to_crs(delivery_crs)

    deliv_coords = np.array([(pt.x, pt.y) for pt in deliveries_gdf.geometry])

    if method == "haversine":
        if centroids is None:
            logger.error("Centroids required for 'haversine' method.")
            raise ValueError("centroids required for 'haversine' method")
        centroids = centroids.to_crs(delivery_crs)
        centroid_coords = centroids[['lon', 'lat']].to_numpy()

        dist_matrix = haversine_matrix(deliv_coords, centroid_coords)
        nearest_idx = np.argmin(dist_matrix, axis=1)
        deliveries_gdf = deliveries_gdf.copy()
        centroids = centroids.reset_index(drop=True)
        deliveries_gdf['assigned_cluster'] = centroids.iloc[nearest_idx]['pincode'].values
        num_clusters = len(deliveries_gdf['assigned_cluster'].unique())
        logger.info(f"Assigned {num_clusters} clusters (by nearest centroid)")

        if debug:
            logger.debug(f"Nearest centroid indices (first 10): {nearest_idx[:10]}")
            logger.debug(f"Assigned cluster counts: {dict(pd.Series(nearest_idx).value_counts())}")

    elif method == "kmeans":
        from sklearn.cluster import KMeans
        if n_clusters is None:
            logger.error("n_clusters required for 'kmeans'.")
            raise ValueError("n_clusters required for 'kmeans'")
        logger.info(f"Clustering with KMeans (n_clusters={n_clusters})")
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
            cluster_labels = kmeans.fit_predict(deliv_coords)
            deliveries_gdf = deliveries_gdf.copy()
            deliveries_gdf['assigned_cluster'] = cluster_labels
            num_clusters = len(np.unique(cluster_labels))
            logger.info(f"KMeans produced {num_clusters} clusters.")
            if debug:
                label_counts = dict(pd.Series(cluster_labels).value_counts())
                logger.debug(f"KMeans assigned counts per cluster: {label_counts}")
                logger.debug(f"KMeans cluster labels (first 10): {cluster_labels[:10]}")
        except Exception as e:
            logger.error(f"KMeans clustering failed: {e}")
            raise

    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        logger.info(f"Clustering with DBSCAN (eps=0.01, min_samples=5)")
        db = DBSCAN(eps=0.01, min_samples=5).fit(deliv_coords)
        deliveries_gdf = deliveries_gdf.copy()
        deliveries_gdf['assigned_cluster'] = db.labels_
        num_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        logger.info(f"DBSCAN produced {num_clusters} clusters (excluding noise).")

    else:
        logger.error(f"Unknown clustering method: {method}")
        raise ValueError(f"Unknown clustering method: {method}")

    logger.info("Clustering assignment complete.")
    return deliveries_gdf

