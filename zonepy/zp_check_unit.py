import json
import pandas as pd
from zonepy import zp_get_unit_info

def zp_check_unit(district_data, bldg_data, vars):
    """
    Python equivalent of zr_check_unit()

    Parameters
    ----------
    district_data : pandas.Series or dict-like (one row of zoning df)
        Must contain a 'constraints' field (JSON string or dict)
    bldg_data : dict or path
    vars : pandas.DataFrame (one row) from zp_get_variables()

    Returns
    -------
    True / False / "MAYBE"
    """
    # 1. No constraints → True
    constraints_dict = district_data.iloc[0]['constraints']
    if constraints_dict is None or pd.isna(constraints_dict):
        return True

    if "unit_size" not in constraints_dict:
        # If there's no unit_size constraint at all, treat as allowed
        return True
    constraint_list = constraints_dict["unit_size"]

    # 3. Get unit info DF and group by bedrooms
    unit_info_df = zp_get_unit_info(bldg_data)
    if 'bedrooms' not in unit_info_df.columns:
        # Mirror R behavior: assume 2 if missing
        print("Warning: No bedroom qty recorded in the tidybuilding. Results may be innacurate.")
        unit_info_df['bedrooms'] = 2

    grouped = (
        unit_info_df.groupby('bedrooms', as_index=False)
        .agg(min=('fl_area', 'min'), max=('fl_area', 'max'))
    )

    # 4. Export vars into a dict for eval
    vars_dict = vars.iloc[0].to_dict()

    # Helpers --------------------------------------------------
    def _to_list(x):
        """Ensure x is a list; string -> [string], None -> []"""
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        return list(x)

    def _eval_expressions(expressions):
        vals = []
        note = None
        for expr in expressions:
            try:
                vals.append(eval(expr, vars_dict))
            except Exception:
                note = "Unable to evaluate expression: incorrect format or missing variables"
        return vals, note

    def _process_val_list(val_list):
        """
        Mimic R logic for picking an item and evaluating expressions.
        val_list: list of dicts, each with 'condition' and 'expression' (and maybe 'min_max').

        Returns: (value, note)
            value: number / tuple / None
            note:  string / None
        """
        if not val_list:
            return None, None

        # single item → take directly
        if len(val_list) == 1:
            item = val_list[0]
            values, note = _eval_expressions(_to_list(item.get('expression')))
            return _finalize_values(values, item.get('min_max')), note

        true_id = None
        maybe_ids = []

        for idx, item in enumerate(val_list):
            conds = _to_list(item.get('condition'))  # <-- 改动1：统一成列表
            if not conds:
                maybe_ids.append(idx)
                continue

            results = []
            for cond in conds:
                try:
                    results.append(eval(cond, vars_dict))
                except Exception:
                    results.append("MAYBE")

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

        note = None
        values = []
        for idx in selected_ids:
            exprs = _to_list(val_list[idx].get('expression', []))  # <-- 改动2：统一成列表
            v, n = _eval_expressions(exprs)
            if n and not note:
                note = n
            values.extend(v)

        if not values:
            return None, note

        first_item = val_list[selected_ids[0]]
        return _finalize_values(values, first_item.get('min_max')), note

    def _finalize_values(values, mm_flag):
        """
        R logic to produce final scalar/tuple:
        - single value → just return it
        - multiple values → if min_max == 'min' or 'max' pick the corresponding
          else return (min, max); if they are same, return that single value
        """
        if len(values) == 1:
            return values[0]
        if mm_flag == 'min':
            return min(values)
        if mm_flag == 'max':
            return max(values)
        mn, mx = min(values), max(values)
        return mn if mn == mx else (mn, mx)

    def _round_if_num(v):
        if isinstance(v, (int, float)):
            return round(v, 4)
        if isinstance(v, tuple):
            return tuple(round(x, 4) for x in v)
        return v
    
    # Main Loop----------------------------------------------------------

    # Prepare columns
    grouped['min_val'] = None
    grouped['max_val'] = None
    grouped['min_ok'] = None
    grouped['max_ok'] = None
    grouped['min_val_note'] = None
    grouped['max_val_note'] = None
    grouped['permitted'] = None

    # Loop each bedroom type row
    for i in range(len(grouped)):
        bedrooms = grouped.loc[i, 'bedrooms']

        # Insert 'bedrooms' into eval namespace (R uses assign)
        vars_dict['bedrooms'] = bedrooms

        # For each of min_val / max_val
        min_val, min_note = _process_val_list(constraint_list.get('min_val', []))
        max_val, max_note = _process_val_list(constraint_list.get('max_val', []))

        # Store numeric with rounding like R (4 decimals)
        min_val = _round_if_num(min_val)
        max_val = _round_if_num(max_val)

        grouped.at[i, 'min_val'] = min_val
        grouped.at[i, 'max_val'] = max_val
        # You can also store notes somewhere if you want (R saved them in *_notes but didn't return)
        grouped.at[i, 'min_val_note'] = min_note
        grouped.at[i, 'max_val_note'] = max_note

        # --- Compare actual vs required ---
        actual_min = grouped.loc[i, 'min']
        actual_max = grouped.loc[i, 'max']

        # Check min requirement
        min_check = _compare_side(actual_min, actual_max, min_val, op='>=')
        # Check max requirement
        max_check = _compare_side(actual_min, actual_max, max_val, op='<=')

        # Determine permitted for this bedroom type
        if min_check is False or max_check is False:
            permitted = False
        elif min_check is True and max_check is True:
            permitted = True
        else:
            permitted = "MAYBE"

        grouped.at[i, 'min_ok'] = min_check
        grouped.at[i, 'max_ok'] = max_check
        grouped.at[i, 'permitted'] = permitted

    # Final overall decision
    perms = grouped['permitted'].unique()
    if False in perms:
        return False
    elif "MAYBE" in perms:
        return "MAYBE"
    else:
        return True
        
def _compare_side(actual_min, actual_max, req, op='>='):
    if req is None:
        return True

    def cmp(a, b, oper):
        return (a >= b) if oper == '>=' else (a <= b)

    # transfer list into tuple
    if isinstance(req, list):
        req = tuple(req)
        
    # the turple range value
    if isinstance(req, tuple) and len(req) == 2:
        r1, r2 = req

        # four bool
        c11 = cmp(actual_min, r1, op)
        c12 = cmp(actual_min, r2, op)
        c21 = cmp(actual_max, r1, op)
        c22 = cmp(actual_max, r2, op)

        sum_min = int(c11) + int(c12)
        sum_max = int(c21) + int(c22)

        if sum_min == 0 or sum_max == 0:
            return False
        elif sum_min == 2 or sum_max == 2:
            return True
        else:
            return "MAYBE"

    # single value
    else:
        ok_min = cmp(actual_min, req, op)
        ok_max = cmp(actual_max, req, op)
        if ok_min and ok_max:
            return True
        elif (not ok_min) or (not ok_max):
            return False
        else:
            return "MAYBE"