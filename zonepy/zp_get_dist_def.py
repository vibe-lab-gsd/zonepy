import json
import os
from typing import Dict, Any

def zp_get_dist_def(path: str) -> Dict[str, Any]:
    """
    Read a zoning GeoJSON (or .zoning) file and return its 'definitions' as a dict.
    """
    # 1. Check file exists
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")

    # 2. Load JSON
    try:
        with open(path, 'r', encoding='utf-8') as f:
            zoning_json = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")

    # 3. Extract definitions
    try:
        defs = zoning_json['definitions']
    except KeyError:
        raise ValueError("No 'definitions' key found in the zoning JSON")

    # 4. Validate its type
    if not isinstance(defs, dict):
        raise ValueError(f"'definitions' is not a dict (got {type(defs).__name__})")

    return defs