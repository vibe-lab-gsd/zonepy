import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

def zp_add_setbacks(parcel_gdf: gpd.GeoDataFrame,
                    district_data,
                    zoning_req):
    """
    Python equivalent of zr_add_setbacks():
    - Basic setbacks (front/int/ext/rear)
    - setback_dist_boundary (boundary rule, pmax)
    - setback_side_sum (sum of side setbacks)
    - setback_front_sum (sum of front/rear setbacks)
    Supports scalar/vector (list/tuple/ndarray) inputs and uses numpy
    for element-wise operations and broadcasting.
    """

    # ---------- helpers ----------
    def to_vec(x):
        """Convert scalar / list / tuple / ndarray / None / NaN into a 1D float ndarray (can be empty)."""
        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            vals = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
            arr = np.asarray(vals, dtype=float)
        else:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                arr = np.array([], dtype=float)
            else:
                arr = np.array([x], dtype=float)
        return arr

    def pmax_vec(a, b):
        """Equivalent to R's pmax: element-wise maximum, supports broadcasting."""
        A = to_vec(a)
        B = to_vec(b)
        if A.size == 0 and B.size == 0:
            return np.array([], dtype=float)
        if A.size == 0:
            A = np.zeros_like(B)
        if B.size == 0:
            B = np.zeros_like(A)
        return np.maximum(A, B)

    def add_vec(a, b):
        """Element-wise addition with broadcasting."""
        A = to_vec(a)
        B = to_vec(b)
        if A.size == 0 and B.size == 0:
            return np.array([], dtype=float)
        if A.size == 0:
            A = np.zeros_like(B)
        if B.size == 0:
            B = np.zeros_like(A)
        return A + B

    def sub_vec(a, b):
        """Element-wise subtraction with broadcasting."""
        A = to_vec(a)
        B = to_vec(b)
        if A.size == 0 and B.size == 0:
            return np.array([], dtype=float)
        if A.size == 0:
            A = np.zeros_like(B)
        if B.size == 0:
            B = np.zeros_like(A)
        return A - B

    def clamp_neg_to_zero(x):
        """Replace negative values with zero (element-wise)."""
        X = to_vec(x)
        if X.size == 0:
            return X
        return np.where(X < 0, 0.0, X)

    def maybe_collapse_scalar(arr):
        """
        Collapse to scalar if length is 1,
        or if length is 2 and both elements are equal;
        otherwise, return as ndarray.
        """
        if not isinstance(arr, np.ndarray):
            arr = to_vec(arr)
        if arr.size == 1:
            return float(arr[0])
        if arr.size == 2 and np.isfinite(arr).all() and np.isclose(arr[0], arr[1]):
            return float(arr[0])
        return arr

    # ---------- 1) If zoning_req is a string: return immediately ----------
    if isinstance(zoning_req, str):
        out = parcel_gdf.copy()
        out['setback'] = None
        return out

    # ---------- 2) Extract district row ----------
    if isinstance(district_data, gpd.GeoDataFrame):
        district_row = district_data.iloc[0]
    else:
        district_row = district_data

    # ---------- 3) Basic setbacks ----------
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
            match = zoning_req[zoning_req['constraint_name'] == key]
            setbacks.append(match.iloc[0]['min_value'] if not match.empty else None)
    if missing:
        warnings.warn("No side label. Setbacks not considered.")
    pg['setback'] = pd.Series(setbacks, index=pg.index, dtype=object)

    # ---------- 4) Collect extra rules ----------
    extras = {}
    for rule in ('setback_dist_boundary', 'setback_side_sum', 'setback_front_sum'):
        m = zoning_req.loc[zoning_req['constraint_name'] == rule, 'min_value']
        if not m.empty:
            extras[rule] = m.iloc[0]

    # ---------- 5) setback_dist_boundary ----------
    if 'setback_dist_boundary' in extras:
        dist_boundary = extras['setback_dist_boundary']
        # Equivalent to: st_cast -> boundary -> buffer(5)
        buf = district_row.geometry.boundary.buffer(5)
        pg['on_boundary'] = pg.geometry.within(buf)

        mask = pg['on_boundary'] == True
        if mask.any():
            def _apply_pmax(sb):
                if sb is None:
                    return None
                return maybe_collapse_scalar(pmax_vec(dist_boundary, sb))
            pg.loc[mask, 'setback'] = pg.loc[mask, 'setback'].apply(_apply_pmax)

    # ---------- 6) setback_side_sum ----------
    if 'setback_side_sum' in extras:
        side_sum = extras['setback_side_sum']
        int_idx = pg.index[pg['side'] == 'interior side']
        ext_idx = pg.index[pg['side'] == 'exterior side']
        if len(int_idx) > 0 and len(ext_idx) > 0:
            # In R, exterior side is prioritized as side_1
            side_1_idx = ext_idx[0]
            side_2_idx = int_idx[0]

            v_ext = pg.at[side_1_idx, 'setback']  # side_1
            v_int = pg.at[side_2_idx, 'setback']  # side_2

            summed_sides_check = sub_vec(side_sum, add_vec(v_ext, v_int))
            side_setback_increase = clamp_neg_to_zero(summed_sides_check)
            new_int = add_vec(v_int, side_setback_increase)

            pg.at[side_2_idx, 'setback'] = maybe_collapse_scalar(new_int)
        else:
            warnings.warn("setback_side_sum cannot be calculated due to lack of parcel side edges")

    # ---------- 7) setback_front_sum ----------
    if 'setback_front_sum' in extras:
        front_sum = extras['setback_front_sum']
        f_idx = pg.index[pg['side'] == 'front']
        r_idx = pg.index[pg['side'] == 'rear']
        if len(f_idx) > 0 and len(r_idx) > 0:
            front_idx = f_idx[0]
            rear_idx  = r_idx[0]

            v_front = pg.at[front_idx, 'setback']
            v_rear  = pg.at[rear_idx,  'setback']

            summed_front_check   = sub_vec(front_sum, add_vec(v_front, v_rear))
            rear_setback_increase = clamp_neg_to_zero(summed_front_check)
            new_rear = add_vec(v_rear, rear_setback_increase)

            pg.at[rear_idx, 'setback'] = maybe_collapse_scalar(new_rear)
        else:
            warnings.warn("setback_front_sum cannot be calculated due to missing front or rear edge")

    return pg