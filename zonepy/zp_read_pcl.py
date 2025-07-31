import geopandas as gpd
import os
from zonepy import zp_find_district_idx

def zp_read_pcl(path, dist):
    """
    Reads a parcel GeoJSON (or any geospatial file), reprojects it to a specified CRS 
    (either user-provided or automatically determined via get_crs), then computes and merges 
    district indices.

    Parameters:
    -----------
    path : str
        File path to the GeoJSON or shapefile.
    dist : GeoDataFrame
        District layer used by find_district_idx().
    trans_crs : str or None, default None
        If not None, this is the target CRS (e.g., "EPSG:3081") to reproject parcels.
        If None, the function calls get_crs(path) to determine the best State Plane EPSG.

    Returns:
    --------
    GeoDataFrame
        Reprojected parcel layer with an added zoning_id column from find_district_idx().
    """

    # 1. Determine target CRS: if trans_crs is provided, use it; otherwise call get_crs()
    target_crs = dist.crs

    # 2. Reproject to the target CRS
    parcel_gdf = gpd.read_file(path)
    parcel_gdf = parcel_gdf.to_crs(target_crs)

    # 3. Compute district index and merge results
    find_district_idx_results = zp_find_district_idx(parcel_gdf, dist)
    parcel_gdf = parcel_gdf.merge(find_district_idx_results, left_index=True, right_index=True, how='left')

    # 4. Rename or drop extra columns (if find_district_idx returns parcel_id_x / parcel_id_y)
    if 'parcel_id_x' in parcel_gdf.columns and 'parcel_id_y' in parcel_gdf.columns:
        parcel_gdf = parcel_gdf.rename(columns={'parcel_id_x': 'parcel_id'})
        parcel_gdf = parcel_gdf.drop(columns=['parcel_id_y'])

    return parcel_gdf
