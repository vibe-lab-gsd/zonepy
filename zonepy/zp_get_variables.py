import json
import pandas as pd

def zp_get_variables(bldg_data, parcel_data, district_data, zoning_data):

    # Step 1: Load zoning data
    if isinstance(zoning_data, str):
        try:
            with open(zoning_data, 'r') as f:
                zoning_json = json.load(f)
            zoning_defs = zoning_json['definitions']
        except:
            raise ValueError("The zoning_data file path does not seem to be in json format")
    elif isinstance(zoning_data, dict):
        zoning_defs = zoning_data
    else:
        raise ValueError("Improper input: zoning_data")

    # Step 2: Load building data
    if isinstance(bldg_data, str):
        try:
            with open(bldg_data, 'r') as f:
                bldg_json = json.load(f)
        except:
            raise ValueError("bldg_data must be a file path to an OZFS *.bldg file or a list created from said file")
    elif isinstance(bldg_data, dict):
        bldg_json = bldg_data
    else:
        raise ValueError("Improper input: bldg_data")

    if 'bldg_info' not in bldg_json or 'unit_info' not in bldg_json or 'level_info' not in bldg_json:
        raise ValueError("Improper format: json must contain bldg_info, unit_info, and level_info sections")

    # Step 3: Create DataFrame for unit info and level info
    unit_info_df = pd.DataFrame(bldg_json['unit_info'])
    level_info_df = pd.DataFrame(bldg_json['level_info'])

    # Step 4: Extract and calculate building variables
    bldg_depth = bldg_json['bldg_info']['depth']
    bldg_width = bldg_json['bldg_info']['width']
    footprint = bldg_depth * bldg_width  
    dist_abbr = district_data['dist_abbr'].iloc[0]
    fl_area = level_info_df['gross_fl_area'].sum()
    fl_area_first = level_info_df[level_info_df['level'] == 1]['gross_fl_area'].sum() if len(level_info_df[level_info_df['level'] == 1]) == 1 else 0
    fl_area_top = level_info_df[level_info_df['level'] == level_info_df['level'].max()]['gross_fl_area'].sum() if level_info_df['level'].max() > 1 else 0
    floors = level_info_df['level'].max()
    height_deck = bldg_json['bldg_info'].get('height_deck', bldg_json['bldg_info']['height_top'])
    height_eave = bldg_json['bldg_info'].get('height_eave', bldg_json['bldg_info']['height_top'])
    height_plate = bldg_json['bldg_info']['height_plate']
    height_top = bldg_json['bldg_info']['height_top']
    height_tower = bldg_json['bldg_info'].get('height_tower', 0)
    lot_area = parcel_data['lot_area'].iloc[0]
    lot_depth = parcel_data['lot_depth'].iloc[0]
    lot_width = parcel_data['lot_width'].iloc[0]
    lot_type = parcel_data['lot_type'].iloc[0]
    max_unit_size = unit_info_df['fl_area'].max()
    min_unit_size = unit_info_df['fl_area'].min()
    n_ground_entry = unit_info_df[unit_info_df['entry_level'] == 1]['qty'].sum()
    n_outside_entry = unit_info_df[unit_info_df['outside_entry'] == True]['qty'].sum()
    parking_enclosed = bldg_json['bldg_info'].get('parking', 0)
    roof_type = bldg_json['bldg_info'].get('roof_type', 'flat')
    sep_platting = bldg_json['bldg_info'].get('sep_platting', False)
    unit_separation = bldg_json['bldg_info'].get('unit_separation', 'open_area')
    total_bedrooms = (unit_info_df['bedrooms'] * unit_info_df['qty']).sum()
    total_units = unit_info_df['qty'].sum()
    units_0bed = unit_info_df[unit_info_df['bedrooms'] == 0]['qty'].sum()
    units_1bed = unit_info_df[unit_info_df['bedrooms'] == 1]['qty'].sum()
    units_2bed = unit_info_df[unit_info_df['bedrooms'] == 2]['qty'].sum()
    units_3bed = unit_info_df[unit_info_df['bedrooms'] == 3]['qty'].sum()
    units_4bed = unit_info_df[unit_info_df['bedrooms'] > 3]['qty'].sum()
    unit_pct_0bed = units_0bed / total_units
    unit_pct_1bed = units_1bed / total_units 
    unit_pct_2bed = units_2bed / total_units 
    unit_pct_3bed = units_3bed / total_units 
    unit_pct_4bed = units_4bed / total_units
    unit_size_avg = float(unit_info_df['fl_area'].mean())
    lot_cov_bldg = (footprint / (lot_area * 43560)) * 100
    unit_density = total_units / lot_area
    far = fl_area / (lot_area * 43560)

    # Step 5: Construct the resulting DataFrame
    vars_df = pd.DataFrame({
        'bldg_depth': [bldg_depth],
        'bldg_width': [bldg_width],
        'dist_abbr': [dist_abbr],
        'fl_area': [fl_area],
        'fl_area_first': [fl_area_first],
        'fl_area_top': [fl_area_top],
        'floors': [floors],
        'height_deck': [height_deck],
        'height_eave': [height_eave],
        'height_plate': [height_plate],
        'height_top': [height_top],
        'height_tower': [height_tower],
        'lot_area': [lot_area],
        'lot_depth': [lot_depth],
        'lot_width': [lot_width],
        'lot_type': [lot_type],
        'max_unit_size': [max_unit_size],
        'min_unit_size': [min_unit_size],
        'n_ground_entry': [n_ground_entry],
        'n_outside_entry': [n_outside_entry],
        'parking_enclosed': [parking_enclosed],
        'roof_type': [roof_type],
        'sep_platting': [sep_platting],
        'unit_separation': [unit_separation],
        'total_bedrooms': [total_bedrooms],
        'total_units': [total_units],
        'units_0bed': [units_0bed],
        'units_1bed': [units_1bed],
        'units_2bed': [units_2bed],
        'units_3bed': [units_3bed],
        'units_4bed': [units_4bed],
        'unit_pct_0bed': [unit_pct_0bed],
        'unit_pct_1bed': [unit_pct_1bed],
        'unit_pct_2bed': [unit_pct_2bed],
        'unit_pct_3bed': [unit_pct_3bed],
        'unit_pct_4bed': [unit_pct_4bed],
        'unit_size_avg': [unit_size_avg],
        'lot_cov_bldg': [lot_cov_bldg],
        'unit_density': [unit_density], 
        'far': [far]
    })

    # Step 6: Dynamically compute each zoning variable from zoning_defs and add to vars_df
    for var_name, var_list in zoning_defs.items():
        matched = False
        for condition in var_list:
            cond_raw = condition['condition']
            # If the condition is a list of multiple sub-conditions, join them with '&';
            # otherwise treat the single string as the full condition expression
            if isinstance(cond_raw, list):
                cond_str = " & ".join(cond_raw)
            else:
                cond_str = cond_raw

            try:
                # Evaluate the boolean expression
                if eval(cond_str):
                    # If the condition is True, evaluate the corresponding expression
                    # to determine the variable’s value
                    value = eval(condition['expression'])
                    matched = True
                    break  # Stop checking further var_list for this variable
            except Exception:
                # If parsing or evaluation fails, skip this condition and continue
                continue

        if not matched:
            # If no condition matched for this variable, raise an error
            raise ValueError(f"No conditions met for variable {var_name}")

        # Assign the computed value into vars_df under the column named var_name
        vars_df[var_name] = value

    return vars_df