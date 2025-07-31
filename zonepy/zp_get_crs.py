import geopandas as gpd
import os

def zp_get_crs(geom_data, large_area=False):
    """
    Finds EPSG code for State Plane CRS.
    get_crs() uses state plane geometry to find an appropriate NAD83 CRS for the input geometry.
    
    Parameters:
    -----------
    geom_data : GeoDataFrame or str
        Either a GeoDataFrame or a file path to a GeoJSON file.
        
    large_area : bool
        If True, consider the full geometry when determining the best state plane (for datasets that
        may cross multiple state planes). If False, only the first feature will be used.
    
    Returns:
    --------
    int
        The appropriate EPSG code (as an integer).
    
    Raises:
    -------
    ValueError
        If geom_data is neither a valid GeoDataFrame nor an existing file path, or
        if no intersection is found with any state plane zone.
    
    Notes:
    ------
    - Assumes `state_planes_crs` is a pre-loaded GeoDataFrame containing:
        * A 'geometry' column defining each State Plane zone (NAD83).
        * An 'EPSG_NAD83' column holding the corresponding EPSG code for each zone.
    """
    
    # Determine input type
    if isinstance(geom_data, gpd.GeoDataFrame):
        geom = geom_data.copy()
    elif isinstance(geom_data, str):
        if os.path.exists(geom_data):
            geom = gpd.read_file(geom_data)
        else:
            raise ValueError(
                "Input must be an existing file path or a GeoDataFrame."
            )
    else:
        raise ValueError(
            "Input must be an existing file path or a GeoDataFrame."
        )
    
    # If not a large area, use only the first geometry
    if not large_area:
        geom = geom.loc[~geom.geometry.is_empty].iloc[[0]].copy()
    
    # Ensure valid geometries (use buffer(0) to fix potential invalidity)
    geom["geometry"] = geom["geometry"].buffer(0)
    
    # Load the state plane CRS definitions from a raw GitHub URL
    state_planes_crs = gpd.read_file(
        "https://raw.githubusercontent.com/KamrynMansfield/tidyzoning/main/inst/extdata/sp_crs.geojson"
    )
    # Reproject state_planes_crs to match the input geometry's CRS
    state_planes_crs = state_planes_crs.to_crs(geom.crs)
    
    # Perform spatial join to find intersecting state plane zones
    joined = gpd.sjoin(geom, state_planes_crs, how="inner", predicate="intersects")
    
    if joined.empty:
        raise ValueError(
            "No intersection found with any State Plane zone."
        )
    
    # Count how many times each state plane index appears
    counts = joined["index_right"].value_counts()
    sp_idx = counts.idxmax()
    
    # Retrieve the EPSG code from state_planes_crs using the index
    crs_code = int(state_planes_crs.loc[sp_idx, "EPSG_NAD83"])
    
    return crs_code