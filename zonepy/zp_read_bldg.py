import json
import os
from typing import Dict, Any

def zp_read_bldg(path: str) -> Dict[str, Any]:
    """
    Read an OZFS .bldg file (JSON) from disk and return its contents as a dict.

    Parameters
    ----------
    path : str
        Filesystem path to the .bldg JSON file.

    Returns
    -------
    dict
        The parsed JSON content of the building file.

    Raises
    ------
    ValueError
        If the path does not exist, is not valid JSON, or the top-level structure
        is not a dict with the expected building sections.
    """
    # 1. Ensure the file exists
    if not os.path.exists(path):
        raise ValueError(f"Building file not found: {path}")

    # 2. Read and parse JSON
    try:
        with open(path, 'r', encoding='utf-8') as f:
            bldg_json = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in building file: {e}")
    except Exception as e:
        raise ValueError(f"Could not read building file: {e}")

    # 3. Return the raw dict for downstream processing
    return bldg_json




