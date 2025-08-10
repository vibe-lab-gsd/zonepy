import warnings
import pandas as pd
import geopandas as gpd
from itertools import zip_longest

def zp_add_setbacks(parcel_gdf: gpd.GeoDataFrame,
                    district_data,
                    zoning_req):
    """
    Fully equivalent to the R version zr_add_setbacks(), and supports
    scalar/vector cases for the three rules: setback_dist_boundary,
    setback_side_sum, setback_front_sum.
    """
    # 1. If zoning_req is a string, just add a column and return
    if isinstance(zoning_req, str):
        out = parcel_gdf.copy()
        out['setback'] = None
        return out

    # 2. Take only one row of district geometry
    if isinstance(district_data, gpd.GeoDataFrame):
        district_row = district_data.iloc[0]
    else:
        district_row = district_data

    # 3. Basic setback
    name_key = {
        'front': 'setback_front',
        'interior side': 'setback_side_int',
        'exterior side': 'setback_side_ext',
        'rear': 'setback_rear'
    }
    pg = parcel_gdf.copy()

    setbacks = []
    missing = False
    for _, row in pg.iterrows():
        side = row['side']
        key = name_key.get(side)
        if key is None:
            setbacks.append(None)
            missing = True
        else:
            match = zoning_req[zoning_req['constraint_name']==key]
            setbacks.append(match.iloc[0]['min_value'] if not match.empty else None)
    if missing:
        warnings.warn("No side label. Setbacks not considered.")
    pg['setback'] = pd.Series(setbacks, index=pg.index, dtype=object)

    # 4. Collect extra rules
    extra = {}
    for rule in ('setback_dist_boundary','setback_side_sum','setback_front_sum'):
        if rule in zoning_req['constraint_name'].values:
            extra[rule] = zoning_req.loc[
                zoning_req['constraint_name']==rule, 'min_value'
            ].iloc[0]

    # Helper: mimic R's pmax + vector recycling + convergence
    def pmax_like(a, b):
        """
        a, b can be scalar or vector (list/tuple/ndarray).
        Do element-wise max, recycle the shorter one, and
        if all elements are equal, return a scalar.
        """
        # Standardize to list
        la = list(a) if isinstance(a, (list,tuple,pd.Series)) else [a]
        lb = list(b) if isinstance(b, (list,tuple,pd.Series)) else [b]
        # Use zip_longest to recycle the shorter one
        fill_a = la[-1] if len(la)>=len(lb) else None
        fill_b = lb[-1] if len(lb)>=len(la) else None
        paired = zip_longest(la, lb,
                             fillvalue=fill_a if len(la)>=len(lb) else fill_b)
        result = [max(x,y) for x,y in paired]
        return result[0] if all(r==result[0] for r in result) else result

    # 5. setback_dist_boundary
    if 'setback_dist_boundary' in extra:
        db = extra['setback_dist_boundary']
        # Create 5m buffer zone
        buf = district_row.geometry.boundary.buffer(5)
        pg['on_boundary'] = pg.geometry.within(buf)

        # Only apply pmax to rows where on_boundary=True
        mask = pg['on_boundary']
        pg.loc[mask, 'setback'] = pg.loc[mask, 'setback'].apply(
            lambda sb: None if sb is None else pmax_like(db, sb)
        )

    # 6. setback_side_sum
    if 'setback_side_sum' in extra:
        ss = extra['setback_side_sum']
        # Find a pair of interior / exterior
        int_idx = pg.index[pg['side']=='interior side']
        ext_idx = pg.index[pg['side']=='exterior side']
        if int_idx.any() and ext_idx.any():
            i0, e0 = int_idx[0], ext_idx[0]
            v_int = pg.at[i0,'setback'] or 0
            v_ext = pg.at[e0,'setback'] or 0
            # R: diff = ss - (v_int+v_ext), keep only >0 part
            diff = ss - (v_int + v_ext)
            inc  = diff if isinstance(diff,(int,float)) and diff>0 else \
                   ([d for d in diff if d>0] if isinstance(diff,(list,tuple)) else 0)
            # Add inc to the second (interior side)
            new_int = pmax_like(v_int, (v_int if isinstance(v_int,(list,tuple)) else [v_int]) + 
                                       (inc if isinstance(inc,(list,tuple)) else [inc]))
            pg.at[i0,'setback'] = new_int
        # else:
        #     warnings.warn("setback_side_sum cannot be calculated due to lack of parcel edges")

    # 7. setback_front_sum
    if 'setback_front_sum' in extra:
        fs = extra['setback_front_sum']
        f_idx = pg.index[pg['side']=='front']
        r_idx = pg.index[pg['side']=='rear']
        if f_idx.any() and r_idx.any():
            f0, r0 = f_idx[0], r_idx[0]
            v_f = pg.at[f0,'setback'] or 0
            v_r = pg.at[r0,'setback'] or 0
            diff = fs - (v_f + v_r)
            inc  = diff if isinstance(diff,(int,float)) and diff>0 else \
                   ([d for d in diff if d>0] if isinstance(diff,(list,tuple)) else 0)
            new_r = pmax_like(v_r, (v_r if isinstance(v_r,(list,tuple)) else [v_r]) +
                                     (inc if isinstance(inc,(list,tuple)) else [inc]))
            pg.at[r0,'setback'] = new_r
        # else:
        #     warnings.warn("setback_front_sum cannot be calculated due to missing front or rear edge")

    return pg