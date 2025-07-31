import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union, polygonize
from shapely.validation import make_valid

def zp_get_buildable_area(parcel_with_setbacks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Equivalent to the R function zr_get_buildable_area(). Always returns a GeoDataFrame with columns:
      - parcel_id
      - buildable_geometry_relaxable
      - buildable_geometry_strict

    Assumes the input is in a projected CRS with units in meters, and that the GeoDataFrame contains:
      - 'parcel_id' (unique identifier for the parcel)
      - 'geometry' (LineString edges of the parcel)
      - 'setback' (None, a single number in feet, or a list of numbers in feet)
    """
    # 1. Merge all edge geometries and polygonize to get the full parcel polygon
    combined_lines = unary_union(parcel_with_setbacks.geometry)
    polygons = list(polygonize(combined_lines))
    parcel_polygon = unary_union(polygons)

    # 2. Extract per-edge min and max setback values (in feet):
    #    - None remains None
    #    - Single value remains that value
    #    - List yields its min and max
    def get_min(sb):
        if sb is None:
            return None
        return min(sb) if isinstance(sb, (list, tuple)) else sb

    def get_max(sb):
        if sb is None:
            return None
        return max(sb) if isinstance(sb, (list, tuple)) else sb

    df = parcel_with_setbacks.copy()
    df['min_setback'] = df['setback'].apply(get_min)
    df['max_setback'] = df['setback'].apply(get_max)

    # 3. If all setbacks are None/NA, return the parcel polygon as both relaxed and strict buildable area
    if df['min_setback'].isna().all() and df['max_setback'].isna().all():
        parcel_id = df['parcel_id'].iat[0]
        return gpd.GeoDataFrame(
            [{
                'parcel_id': parcel_id,
                'buildable_geometry_relaxable': parcel_polygon,
                'buildable_geometry_strict': parcel_polygon
            }],
            geometry='buildable_geometry_strict',
            crs=df.crs
        )

    # 4. Replace NA with 0.1 ft, then convert feet to meters
    df['min_setback'] = df['min_setback'].fillna(0.1) * 0.3048
    df['max_setback'] = df['max_setback'].fillna(0.1) * 0.3048

    # 5. Determine if every edge uses the same setback (single-case)
    single_case = (df['min_setback'] == df['max_setback']).all()

    # Helper: when result is a MultiPolygon, pick the polygon with the most vertices
    def pick_by_vertices(geom):
        if geom.geom_type != 'MultiPolygon':
            return geom
        parts = list(geom.geoms)
        return max(parts, key=lambda g: len(g.exterior.coords))

    # 6. Single-case: relaxed == strict
    if single_case:
        buffers = [row.geometry.buffer(row['min_setback']) for _, row in df.iterrows()]
        union_buffer = unary_union(buffers)
        buildable = make_valid(parcel_polygon).difference(make_valid(union_buffer))
        main_polygon = pick_by_vertices(buildable)
        parcel_id = df['parcel_id'].iat[0]
        return gpd.GeoDataFrame(
            [{
                'parcel_id': parcel_id,
                'buildable_geometry_relaxable': main_polygon,
                'buildable_geometry_strict': main_polygon
            }],
            geometry='buildable_geometry_strict',
            crs=df.crs
        )

    # 7. Multi-case: compute relaxed (min) and strict (max) separately
    # Relaxed (using min_setback)
    buffers_min = [row.geometry.buffer(row['min_setback']) for _, row in df.iterrows()]
    union_min = unary_union(buffers_min)
    area_relaxed = make_valid(parcel_polygon).difference(make_valid(union_min))
    area_relaxed = pick_by_vertices(area_relaxed)

    # Strict (using max_setback)
    buffers_max = [row.geometry.buffer(row['max_setback']) for _, row in df.iterrows()]
    union_max = unary_union(buffers_max)
    area_strict = make_valid(parcel_polygon).difference(make_valid(union_max))
    area_strict = pick_by_vertices(area_strict)

    # 8. Return a one-row GeoDataFrame with both buildable areas
    parcel_id = df['parcel_id'].iat[0]
    return gpd.GeoDataFrame(
        [{
            'parcel_id': parcel_id,
            'buildable_geometry_relaxable': area_relaxed,
            'buildable_geometry_strict': area_strict
        }],
        geometry='buildable_geometry_strict',
        crs=df.crs
    )
