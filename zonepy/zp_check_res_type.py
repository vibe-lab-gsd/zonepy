import pandas as pd
import numpy as np

def zp_check_res_type(vars, district_data):
    """
    Is the building allowed based on use_type?

    Compares the building use type with the permitted use types in the zoning
    code and returns True or False. If the zoning file does not specify any
    res_types_allowed (i.e. it's None or NaN or an empty list), defaults to True.
    """
    res_type = vars.iloc[0]["res_type"]
    allowed = district_data.iloc[0].get("res_types_allowed", None)

    if allowed is None:
        return True
    if isinstance(allowed, float) and np.isnan(allowed):
        return True
    if isinstance(allowed, (list, tuple)) and len(allowed) == 0:
        return True

    return res_type in allowed
