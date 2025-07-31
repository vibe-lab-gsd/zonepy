import geopandas as gpd

def zp_get_parcel_geo(parcels_gdf):
    """
    Isolate dimensional rows from tidyparcel data.

    Parameters:
    ----------
    parcels_gdf : geopandas.GeoDataFrame
        Loaded GeoDataFrame object.

    Returns:
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing only parcel_id, side, and geometry for each parcel.
    """
    # 1. Filter out rows where side is "unknown" or "centroid"
    parcels_geo = parcels_gdf.loc[
        (parcels_gdf["side"] != "unknown") &
        (parcels_gdf["side"] != "centroid")
    ].copy()  # .copy() is optional here unless you plan to mutate parcels_geo later

    # 2. Keep only the required columns and reset index
    result = parcels_geo[["parcel_id", "side", "geometry"]].reset_index(drop=True)

    return result