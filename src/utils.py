# src/utils.py
import geopandas as gpd
from .logging_utils import get_logger

logger = get_logger("Utils")

def filter_deliveries_within_boundary(deliveries_gdf, boundary_gdf):
    """
    Filter deliveries to those within a given city boundary polygon.
    Args:
        deliveries_gdf: GeoDataFrame of deliveries
        boundary_gdf: GeoDataFrame with single-row polygon
    Returns:
        (in_city, outliers) as separate GeoDataFrames
    """
    city_polygon = boundary_gdf.unary_union  # works for single or multi
    in_city = deliveries_gdf[deliveries_gdf.within(city_polygon)].copy()
    outliers = deliveries_gdf[~deliveries_gdf.within(city_polygon)].copy()
    logger.info(f"Filtered deliveries: {len(in_city)} inside, {len(outliers)} outliers")
    return in_city, outliers
