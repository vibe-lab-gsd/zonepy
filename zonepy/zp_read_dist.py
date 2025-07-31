import geopandas as gpd
import os
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, MultiLineString
from zonepy.zp_get_crs import zp_get_crs

def zp_read_dist(path, trans_crs=None, index_col="zoning_id"):
    """
    Reads a district GeoJSON (or any geospatial file), drops rows with missing or invalid geometry,
    reprojects to a chosen CRS (either user-provided or automatically determined via get_crs),
    and assigns a zoning_id.

    Parameters:
    -----------
    path : str
        File path to the GeoJSON or shapefile.
    trans_crs : str or None, default None
        If not None, this is the target CRS (e.g., "EPSG:3081") to reproject districts.
        If None, the function calls get_crs(path) to determine the best State Plane EPSG.
    index_col : str, default "zoning_id"
        Name of the new index column for zoning ID.

    Returns:
    --------
    GeoDataFrame
        Cleaned and reprojected district layer with zoning_id column.
    """

    # 1. Drop rows whose geometry is missing, non‐standard type, or empty
    dist_gdf = gpd.read_file(path)
    valid_types = (Polygon, LineString, Point, MultiPolygon, MultiLineString)
    dist_gdf = dist_gdf[dist_gdf.geometry.apply(lambda geom: isinstance(geom, valid_types) and not geom.is_empty)]

    # 2. Determine target CRS: if trans_crs provided, use it; else call get_crs()
    if trans_crs is not None:
        target_crs = trans_crs
    else:
        auto_epsg = zp_get_crs(path, large_area=False)
        target_crs = f"EPSG:{auto_epsg}"

    # 3. Reproject to the target CRS
    dist_gdf = dist_gdf.to_crs(target_crs)

    # 4. Reset index and assign zoning_id
    # dist_gdf = dist_gdf.reset_index(drop=True)
    dist_gdf[index_col] = dist_gdf.index.astype(str)

    return dist_gdf