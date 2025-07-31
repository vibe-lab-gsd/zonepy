import json
from typing import Union, Dict, Any
import pandas as pd

def zp_get_unit_info(bldg_data: Union[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame with unit info from an OZFS building JSON.

    Parameters
    ----------
    bldg_data : str or dict
        Either a file path to a JSON file containing OZFS building attributes,
        or a dict already parsed from such JSON.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['fl_area', 'bedrooms', 'qty'], one row per unit.
    """
    # Load JSON from file or use provided dict
    if isinstance(bldg_data, str):
        try:
            with open(bldg_data, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(
                "bldg_data must be a file path to an OZFS *.bldg JSON file or a dict"
            ) from e
    elif isinstance(bldg_data, dict):
        data = bldg_data
    else:
        raise TypeError("Improper input: bldg_data must be str or dict")

    # Validate required sections
    for section in ('bldg_info', 'unit_info', 'level_info'):
        if section not in data:
            raise KeyError(
                f"Improper format: JSON must contain '{section}' section"
            )

    # Extract unit fields
    fl_areas = []
    bedrooms = []
    qtys = []
    for unit in data['unit_info']:
        fl_areas.append(unit.get('fl_area'))
        bedrooms.append(unit.get('bedrooms'))
        qtys.append(unit.get('qty'))

    # Build DataFrame
    df = pd.DataFrame({
        'fl_area': fl_areas,
        'bedrooms': bedrooms,
        'qty': qtys
    })

    return df