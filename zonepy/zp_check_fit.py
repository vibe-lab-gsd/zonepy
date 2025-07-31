import numpy as np
import pandas as pd
import shapely.geometry
import shapely.affinity
import rasterio
import rasterio.features
import rasterio.transform
import math
import numba
from tqdm import tqdm

# numba really does help here, 32us -> ~1us.
@numba.jit(nopython=True)
def fits(mask, width, depth):
    '''
    This function checks whether a rectangle of width x depth can fit inside the rasterized polygon described by
    mask.
    '''
    for x in range(mask.shape[0]):  # the number of rows = the depth of the mask
        for y in range(mask.shape[1]):  # the number of columns = the width of the mask
            if mask[x, y]:  # choose the raster point = True
                if ((x + width) <= mask.shape[0] and (y + depth) <= mask.shape[1]
                        and np.all(mask[x:x + width, y:y + depth])):
                    return True
                if ((x + depth) <= mask.shape[0] and (y + width) <= mask.shape[1]
                        and np.all(mask[x:x + depth, y:y + width])):
                    return True
    return False

def rect_fit(geom, dims):
    '''
    This function tests whether a rectangle of `dims` fits within the `geom`, by rasterization.
    '''
    if geom is None or geom.is_empty:
        return np.array([False for d in dims])

    w, s, e, n = rasterio.features.bounds(geom)
    # Force width/depth to exact meters
    w = math.floor(w)
    s = math.floor(s)
    n = math.ceil(n)
    e = math.ceil(e)
    width = int(round(e - w))
    depth = int(round(n - s))
    xform = rasterio.transform.from_bounds(w, s, e, n, width, depth)
    mask = rasterio.features.geometry_mask([geom], (depth, width), xform, invert=True)
    return [fits(mask, w, d) for w, d in dims]

def rot_fit(geom, dims, rotations_deg=np.arange(0, 90, 15)):
    '''
    Check if it fits for all possible rotations, 0-90 degrees. Only need to rotate through 90 degrees
    because fit() checks for fit both horizontally and vertically, and because rectangles are symmetrical.
    '''
    dims = np.array(dims)
    out = np.array([False for dim in dims])

    for rot in rotations_deg:
        if rot == 0:
            rot_geom = geom
        else:
            rot_geom = shapely.affinity.rotate(geom, rot, use_radians=False)

        out[~out] |= rect_fit(rot_geom, dims[~out])

        if np.sum(~out) == 0:
            break

    return out

def zp_check_fit(buildable_area, vars):
    '''
    Function to check if buildings fit within parcels using both strict and relaxable buildable areas.

    For each parcel, the function first tests if the building (given by its dimensions in tidybuilding)
    fits within the strict buildable area ('buildable_geometry_strict'). If it does, the result is True.
    If not, it then tests the relaxable buildable area ('buildable_geometry_relaxable'):
      - If the building fits in the relaxable area, allowed is set to "MAYBE"
      - Otherwise, allowed is False.

    Parameters:
        tidyparcel_gdf (GeoDataFrame): GeoDataFrame containing parcel geometries with columns 
                                       'buildable_geometry_strict' and 'buildable_geometry_relaxable'.
        tidybuilding (DataFrame): DataFrame containing building dimensions (with columns 'width', 'depth').

    Returns:
        DataFrame: Results with columns ['parcel_id', 'allowed'].
                 'allowed' is True if the building fits strictly, "MAYBE" if it only fits relaxably,
                 and False if it does not fit.
    '''
    results = []

    for _, parcel in buildable_area.iterrows():
        # Get strict and relaxable geometries from the parcel
        strict_geom = parcel['buildable_geometry_strict']
        relaxable_geom = parcel['buildable_geometry_relaxable']

        # Check if the geometries are valid
        if (isinstance(strict_geom, str) and strict_geom == "error") or (isinstance(relaxable_geom, str) and relaxable_geom == "error"):
            results.append([parcel['parcel_id'], "MAYBE"])
            continue
        
        # Skip parcels with no valid geometry in both cases.
        if ((strict_geom is None or strict_geom.is_empty) or (relaxable_geom is None or relaxable_geom.is_empty)):
            results.append([parcel['parcel_id'], "MAYBE"])
            continue
        
        allowed_result = False  # default allowed is False
        
        # Loop through each building dimension (assumes tidybuilding has one or more rows)
        for _, bldg in vars.iterrows():
            dims = [(bldg['bldg_width'] * 0.3048, bldg['bldg_depth'] * 0.3048)] # Convert feet to meters
            
            # First, check the strict buildable area
            if strict_geom is not None and not strict_geom.is_empty:
                strict_fit = rot_fit(strict_geom, dims)[0]
            else:
                strict_fit = False
            
            if strict_fit:
                allowed_result = True
                # If any building fits strictly, we consider the parcel allowed.
                break
            
            # If not strictly allowed, check the relaxable buildable area
            if relaxable_geom is not None and not relaxable_geom.is_empty:
                relaxable_fit = rot_fit(relaxable_geom, dims)[0]
            else:
                relaxable_fit = False
            
            if relaxable_fit:
                allowed_result = "MAYBE"
                # Optionally, you could break here if one "MAYBE" is sufficient.
                break
        
        results.append([parcel['parcel_id'], allowed_result])

    result_df = pd.DataFrame(results, columns=['parcel_id', 'allowed'])
    return result_df