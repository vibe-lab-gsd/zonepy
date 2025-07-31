import json
import pandas as pd
import numpy as np
from zonepy import zp_get_variables  # make sure this exists in your package

def zp_get_zoning_req(district_data, bldg_data=None, parcel_data=None, zoning_data=None, vars=None):
    """
    Python translation of `zr_get_zoning_req()`.

    Returns
    -------
    pandas.DataFrame or str
        DataFrame with columns:
          - constraint_name
          - min_value
          - max_value
          - min_val_error
          - max_val_error
        or a string "No zoning requirements recorded for this district" when applicable.
    """

    # ---------------------- Step 1: Read constraints ----------------------
    # district_data can be dict, Series or single-row DataFrame
    if isinstance(district_data, dict):
        constraints_raw = district_data.get('constraints')
    else:  # Series or DataFrame
        constraints_raw = (district_data.iloc[0]['constraints']
                           if isinstance(district_data, pd.DataFrame)
                           else district_data['constraints'])

    # No constraints recorded -> return the same message as R
    if constraints_raw is None or (isinstance(constraints_raw, float) and pd.isna(constraints_raw)):
        return "No zoning requirements recorded for this district"

    # If the constraints field is JSON text, parse it
    if isinstance(constraints_raw, str):
        try:
            listed_constraints = json.loads(constraints_raw)
        except Exception:
            return "No zoning requirements recorded for this district"
    else:
        listed_constraints = constraints_raw

    # Remove the special key "unit_size" (handled by other functions)
    if 'unit_size' in listed_constraints:
        listed_constraints = {k: v for k, v in listed_constraints.items() if k != 'unit_size'}

    if not listed_constraints:
        return "No zoning requirements recorded for this district"

    constraints = list(listed_constraints.keys())

    # ---------------------- Step 2: Prepare variables ----------------------
    # If not supplied, compute vars using zp_get_variables
    if vars is None:
        vars = zp_get_variables(bldg_data, parcel_data, district_data, zoning_data)

    # Convert vars (DataFrame or dict-like) to a simple dict for eval context
    if isinstance(vars, pd.DataFrame):
        vars_dict = vars.iloc[0].to_dict()
    else:
        vars_dict = dict(vars)

    # ---------------------- Helper functions ----------------------
    def _to_list(x):
        """Ensure x is a list. Strings and None become [x] / [] respectively."""
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        return list(x)

    def _eval_expressions(expressions):
        """
        Evaluate a list of expression strings against vars_dict.
        Returns (values_list, note_or_None).
        """
        vals = []
        note = None
        for expr in expressions:
            try:
                vals.append(eval(expr, vars_dict))
            except Exception:
                note = "Unable to evaluate expression: incorrect format or missing variables"
        return vals, note

    def _finalize(values, mm_flag):
        """
        Replicate the R logic for choosing a final value:
        - If only one value -> return it
        - If multiple and 'min_max' == 'min' or 'max' -> pick the corresponding
        - Otherwise return (min, max). If they are equal, return the single value.
        """
        if len(values) == 1:
            return values[0]
        if mm_flag == 'min':
            return min(values)
        if mm_flag == 'max':
            return max(values)
        mn, mx = min(values), max(values)
        return mn if mn == mx else (mn, mx)

    def _process_val_list(val_list):
        """
        Core selection logic matching the R code:
        - Try to find an item where all conditions evaluate to TRUE (true_id)
        - Otherwise collect items with no FALSE (only TRUE/MAYBE) as maybe_ids
        - If neither, return (None, "No constraint conditions met")
        - Evaluate expressions in selected items and finalize
        Returns (value, note)
        """
        if not val_list:
            return None, None

        # If only one item, R just takes it directly (no condition check)
        if len(val_list) == 1:
            item = val_list[0]
            exprs = _to_list(item.get('expression'))
            vals, note = _eval_expressions(exprs)
            return _finalize(vals, item.get('min_max')), note

        true_id = None
        maybe_ids = []
        note = None

        for idx, item in enumerate(val_list):
            conds = _to_list(item.get('condition'))
            if not conds:
                # No condition -> treat as MAYBE
                maybe_ids.append(idx)
                continue

            results = []
            for cond in conds:
                try:
                    results.append(eval(cond, vars_dict))
                except Exception:
                    results.append("MAYBE")

            # Use == instead of "is" to catch numpy.bool_ etc.
            if all(r == True for r in results):
                true_id = idx
                break
            elif any(r == False for r in results):
                continue
            else:
                maybe_ids.append(idx)

        selected_ids = [true_id] if true_id is not None else maybe_ids
        if not selected_ids:
            return None, "No constraint conditions met"

        values = []
        for idx in selected_ids:
            exprs = _to_list(val_list[idx].get('expression'))
            v, n = _eval_expressions(exprs)
            if n and not note:
                note = n
            values.extend(v)

        if not values:
            return None, note

        mm = val_list[selected_ids[0]].get('min_max')
        return _finalize(values, mm), note

    def _round_num(v):
        """Round numeric (or tuple of numerics) to 4 decimals, else return as-is."""
        if isinstance(v, (int, float, np.number)):
            return round(float(v), 4)
        if isinstance(v, tuple):
            return tuple(round(float(x), 4) for x in v)
        return v

    # ---------------------- Step 3: Loop through constraints ----------------------
    min_values, max_values = [], []
    min_notes, max_notes = [], []

    for cname in constraints:
        cdef = listed_constraints[cname]

        min_val, min_note = _process_val_list(cdef.get('min_val', []))
        max_val, max_note = _process_val_list(cdef.get('max_val', []))

        min_values.append(_round_num(min_val))
        max_values.append(_round_num(max_val))
        min_notes.append(min_note)
        max_notes.append(max_note)

    # ---------------------- Step 4: Build final DataFrame ----------------------
    df = pd.DataFrame({
        'constraint_name': constraints,
        'min_value': min_values,
        'max_value': max_values,
        'min_val_error': min_notes,
        'max_val_error': max_notes,
    })

    return df