import pandas as pd
import numpy as np

def zp_check_constraints(vars,
                         zoning_req,
                         checks=None):
    """
    Python equivalent of zr_check_constraints()

    Parameters
    ----------
    vars : pandas.DataFrame (single row) or dict-like
        Result from zr_get_variables(), must contain at least the columns corresponding to checks.
    zoning_req : pandas.DataFrame or str
        Result from zr_get_zoning_req(); if it is a string (R returns "No zoning..."), return the same string directly.
        Must contain columns:
          constraint_name, min_value, max_value,
          min_val_note, max_val_note,
          min_val_error, max_val_error
    checks : list[str] or None
        List of constraint names to check; if None, use the default list from R.

    Returns
    -------
    pandas.DataFrame or str
        Columns: constraint_name, allowed, (optional) warning
    """

    # --- 0. Default checks list (same as R default) ---
    if checks is None:
        checks = [
            "far", 
            "fl_area", 
            "fl_area_first", 
            "fl_area_top", 
            "footprint",
            "height", 
            "height_eave", 
            "lot_cov_bldg", 
            "lot_size",
            "parking_enclosed", 
            "stories",
            "unit_0bed", 
            "unit_1bed", 
            "unit_2bed", 
            "unit_3bed", 
            "unit_4bed",
            "unit_density",
            "unit_pct_0bed", 
            "unit_pct_1bed", 
            "unit_pct_2bed",
            "unit_pct_3bed", 
            "unit_pct_4bed",
            "total_units", 
            "unit_size_avg"
        ]

    # --- 1. If zoning_req is a string, all checks are allowed by default (allowed=True) ---
    if isinstance(zoning_req, str):
        return pd.DataFrame({
            "constraint_name": checks,
            "allowed": [True] * len(checks)
        })

    # --- 2. Keep only constraints in checks ---
    filtered_req = zoning_req[zoning_req['constraint_name'].isin(checks)].copy()

    # --- 3. Get values from vars ---
    vars_row = vars.iloc[0].to_dict()

    # --- 4. Helper functions ---
    def _ensure_list(x, fill_if_na=None):
        """
        In R, min_value / max_value columns are often list-columns.
        Here, always convert to list.
        fill_if_na gives the default value (R: 0 or 100000).
        """
        if pd.isna(x):
            return [fill_if_na]
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        return [x]

    def _decide_allowed_single(value, min_req_val, max_req_val):
        """ Directly check single value lower/upper bounds """
        return (value >= min_req_val) and (value <= max_req_val)

    def _check_two_values(value, req_list, note_either, is_min=True):
        """
        Branch for multiple values in R (e.g., min_requirement is a list of 2 values).
        - value: actual value
        - req_list: list of length ≥2
        - note_either: whether "either"
        - is_min: True for lower bound check, False for upper bound check
        Returns True / False / "MAYBE"
        """

        r1 = min(req_list)
        r2 = max(req_list)

        if is_min:
            # min_check_1 <- min(min_requirement) <= value
            # min_check_2 <- max(min_requirement) <= value
            c1 = (r1 <= value)
            c2 = (r2 <= value)
        else:
            # max_check_1 <- min(max_requirement) >= value
            # max_check_2 <- max(max_requirement) >= value
            c1 = (r1 >= value)
            c2 = (r2 >= value)

        if note_either is True:  # either mode
            if not c1 and not c2:
                return False
            else:
                return True
        else:
            if c1 and c2:
                return True
            elif (not c1) and (not c2):
                return False
            else:
                return "MAYBE"

    # --- 5. Main loop ---
    warnings_col = []  # Record warnings (R uses list-column, here just string/None)

    allowed_list = []
    for i, row in filtered_req.iterrows():
        constraint = row['constraint_name']
        value = vars_row.get(constraint, None)

        # If vars does not have this variable
        if value is None:
            filtered_req.at[i, 'warning'] = "Variable not found"
            allowed_list.append(True)
            continue

        # Get min/max requirements
        min_req_raw = row['min_value']
        max_req_raw = row['max_value']

        # In R: is.na -> 0 or 100000
        min_list = _ensure_list(min_req_raw, fill_if_na=0)
        max_list = _ensure_list(max_req_raw, fill_if_na=100000)

        # Get note (R's *_val_note) to check if either
        min_note = row.get('min_val_note', None)
        max_note = row.get('max_val_note', None)
        min_either = (min_note == "either")
        max_either = (max_note == "either")

        # Single value case (R code's if (length(min_requirement) & length(max_requirement) == 1))
        # That R line is a bit odd, here use a clearer check: both sides length 1
        if len(min_list) == 1 and len(max_list) == 1:
            is_allowed = _decide_allowed_single(value, min_list[0], max_list[0])
            allowed_list.append(is_allowed)
            continue

        # Multiple values case
        # Lower bound check
        min_check = _check_two_values(value, min_list, min_either, is_min=True)
        # Upper bound check
        max_check = _check_two_values(value, max_list, max_either, is_min=False)

        # Combine min/max
        if (min_check is False) or (max_check is False):
            allowed_list.append(False)
            continue
        elif (min_check is True) and (max_check is True):
            allowed_list.append(True)
            continue
        else:
            # MAYBE, also combine error explanations (R code)
            explanation = []
            min_err = row.get('min_val_error', None)
            max_err = row.get('max_val_error', None)
            if pd.notna(min_err):
                explanation.append(min_err)
            if pd.notna(max_err):
                explanation.append(max_err)
            if explanation:
                filtered_req.at[i, 'warning'] = "; ".join(map(str, explanation))
            allowed_list.append("MAYBE")

    filtered_req['allowed'] = allowed_list

    # Final columns to return
    if 'warning' in filtered_req.columns and filtered_req['warning'].notna().any():
        out_df = filtered_req[['constraint_name', 'allowed', 'warning']].copy()
    else:
        out_df = filtered_req[['constraint_name', 'allowed']].copy()

    return out_df