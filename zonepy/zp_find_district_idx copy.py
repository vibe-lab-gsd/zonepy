import numpy as np
import geopandas as gpd  
import pandas as pd     

def zp_find_district_idx(tidyparcel, tidyzoning):
    """
    Optimized version: Find the indices of the districts in `tidyzoning` that contain the centroids of the `tidyparcel`.

    Parameters:
    tidyparcel (GeoDataFrame): A GeoDataFrame representing the parcel.
                               Must include rows with 'centroid' in the 'side' column.
    tidyzoning (GeoDataFrame): A GeoDataFrame representing zoning districts with geometries.

    Returns:
                    Prop_ID and Pacel_id are from Tidyparcel
                    tidyzoning_index are from Tidyzoning
                    (Prop_ID, parcel_id, tidyzoning_index) if a match is found, or (Prop_ID, parcel_id, None) if no match is found.
    How to use:
    find_district_idx_results = find_district_idx(tidyparcel, tidyzoning)
    """
    # Filter rows with centroids
    centroid_rows = tidyparcel[tidyparcel['side'] == 'centroid']
    if centroid_rows.empty:
        print("No centroids found in tidyparcel.")
        return []

    # Perform spatial join to find matches
    joined = gpd.sjoin(centroid_rows, tidyzoning, how='left', predicate='within')

    # Create the DataFrame directly with required columns
    df_pairs = pd.DataFrame({
        "parcel_id": joined["parcel_id"],
        "zoning_id": joined["zoning_id"]
    })

    # 4. Define a helper to collapse multiple matches into list,
    #    single matches to scalar, and no matches to NaN
    def collapse_zoning_ids(ids):
        clean = [v for v in ids if pd.notnull(v)]
        if len(clean) > 1:
            return clean
        elif len(clean) == 1:
            return clean[0]
        else:
            return np.nan

    # 5. Aggregate zoning_id by parcel_id
    agg_df = (
        df_pairs
        .groupby('parcel_id', as_index=False)['zoning_id']
        .agg(collapse_zoning_ids)
    )

    # 6. Map the aggregated results back onto the original centroids,
    #    preserving the original index
    zoning_map = dict(zip(agg_df['parcel_id'], agg_df['zoning_id']))
    result = centroid_rows[['parcel_id']].copy()
    result['zoning_id'] = result['parcel_id'].map(zoning_map)

    return result