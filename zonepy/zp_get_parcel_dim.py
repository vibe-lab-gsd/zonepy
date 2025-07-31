def zp_get_parcel_dim(parcels_gdf):
    """
    Isolate dimensional rows from tidyparcel data.

    Parameters:
    ----------
    parcels_gdf : geopandas.GeoDataFrame
        Loaded GeoDataFrame object.

    Returns:
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing only parcel_id, dimensions (width, depth, area), lot type (corner/regular), confidence (yes/no), and zoning_id for each parcel.
    """
    # 1. Find all parcel_ids with "exterior side" to mark as corner lots
    corner_ids = set(
        parcels_gdf.loc[
            parcels_gdf["side"] == "exterior side",
            "parcel_id"
        ].unique()
    )

    # 2. Extract all parcel_ids where side is neither "unknown" nor "centroid" to mark as confident
    confident_ids = set(
        parcels_gdf.loc[
            (parcels_gdf["side"] != "unknown") &
            (parcels_gdf["side"] != "centroid"),
            "parcel_id"
        ].unique()
    )

    # 3. Select only "centroid" rows, one per parcel_id
    parcels_dim = parcels_gdf.loc[
        parcels_gdf["side"] == "centroid"
    ].copy()  # Copy to avoid chained assignment warning

    # 4. Add lot_type column: "corner" if parcel_id in corner_ids, else "regular"
    parcels_dim["lot_type"] = parcels_dim["parcel_id"].apply(
        lambda pid: "corner" if pid in corner_ids else "regular"
    )

    # 5. Add conf column: "yes" if parcel_id in confident_ids, else "no"
    parcels_dim["conf"] = parcels_dim["parcel_id"].apply(
        lambda pid: "yes" if pid in confident_ids else "no"
    )

    # 6. Keep only required columns (including geometry)
    result = parcels_dim[
        ["parcel_id", "lot_width", "lot_depth", "lot_area", "lot_type", "conf", "side", "geometry", "zoning_id"]
    ].reset_index(drop=True)

    return result