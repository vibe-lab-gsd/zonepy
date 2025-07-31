import time
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from shapely.geometry import box
from shapely.geometry import mapping
import os
import json
import glob

# input my libraries
from zonepy import zp_get_crs
from zonepy import zp_find_district_idx
from zonepy import zp_read_dist
from zonepy import zp_read_pcl
from zonepy import zp_read_bldg
from zonepy import zp_get_dist_def
from zonepy import zp_get_parcel_dim
from zonepy import zp_get_parcel_geo
from zonepy import zp_get_unit_info
from zonepy import zp_get_variables
from zonepy import zp_get_zoning_req
from zonepy import zp_check_unit
from zonepy import zp_check_res_type
from zonepy import zp_check_constraints
from zonepy import zp_add_setbacks
from zonepy import zp_get_buildable_area
from zonepy import zp_check_fit
from zonepy import possible_checks


def zp_run_zoning_checks(
    bldg_file, 
    parcel_files, 
    zoning_files,
    detailed_check=False,
    print_checkpoints=True,
    checks=None,
    save_to=None
):
    """
    Python equivalent of zr_run_zoning_checks():
    Returns, for each parcel_id, whether building is allowed and the reason,
    with geometry set to the parcel centroid.

    Parameters are identical to the R version:
      bldg_file:   path to the .bldg file
      parcel_files: path to a .parcel file or a directory of such files
      zoning_files: path to a .zoning file or a directory of such files
      detailed_check: whether to retain all intermediate FALSE/MAYBE records
      print_checkpoints: whether to print timing information for each step
      checks:       list of checks to perform (default is possible_checks)
      save_to:      if not None, path to save the output GeoJSON
    """
    
    start_time = time.time()

    # ————————— 0. Parameter validation ————————— #
    # 1. Validate checks input parameter
    if checks is None:
        checks = possible_checks.copy()
    bad = [c for c in checks if c not in possible_checks]
    if bad:
        warnings.warn(f"Unknown constraints: {bad}")

    # 2. Determine initial constraints to run (excluding built-in res_type/unit_size/bldg_fit/overlay)
    initial_checks = [
        c for c in checks
        if c not in ("res_type", "unit_size", "bldg_fit", "overlay")
    ]

    # 3. Process zoning_files: support single file or directory
    if isinstance(zoning_files, str):
        if os.path.isdir(zoning_files):
            zoning_files = glob.glob(os.path.join(zoning_files, "*.zoning"))
        else:
            zoning_files = [zoning_files]

    # 4. Process parcel_files: same as above
    if isinstance(parcel_files, str):
        if os.path.isdir(parcel_files):
            parcel_files = glob.glob(os.path.join(parcel_files, "*.parcel"))
        else:
            parcel_files = [parcel_files]

    # ————————— 1. Data preparation ————————— #
    # 1. Read building JSON
    bldg_data = zp_read_bldg(bldg_file)

    # 2. Read zoning layers and corresponding JSON
    zoning_gdfs = []
    zoning_jsons = []

    first = True
    target_crs = None
    for muni_id, zf in enumerate(zoning_files):
        gz = zp_read_dist(zf)
        if first:
            target_crs = gz.crs
            first = False
        else:
            gz = gz.to_crs(target_crs)
        with open(zf, "r") as f:
            js = json.load(f)
        gz["muni_name"] = js.get("muni_name")  
        gz["muni_id"] = muni_id
        zoning_gdfs.append(gz)
        zoning_jsons.append(js)
    zoning_all = pd.concat(zoning_gdfs, ignore_index=True)
    
    # 3. Create unique zoning_id by combining muni_id and original zoning_id
    zoning_all['zoning_id'] = (zoning_all['muni_id'].astype(int).astype(str) + '_' + zoning_all['zoning_id'].astype(int).astype(str))
    
    # 4. Separate into overlays, planned_dev districts, and base zones
    overlays = zoning_all[zoning_all["overlay"]   == True]
    pd_districts = zoning_all[zoning_all["planned_dev"]== True]
    base_zones  = zoning_all[(zoning_all["overlay"]   == False) & (zoning_all["planned_dev"]== False)]

    # 5. Read parcels and assign base zoning IDs
    parcel_list = [ zp_read_pcl(p, base_zones) for p in parcel_files ]
    parcels_sf = pd.concat(parcel_list, ignore_index=True)
    # 6. Generate parcel_geo with 'side' labels
    parcel_geo  = zp_get_parcel_geo(parcels_sf)
    # 7. Generate parcel_dims with centroid and dimensions, rename zoning_id to muni_base_id
    parcel_dims = zp_get_parcel_dim(parcels_sf)
    parcel_dims = parcel_dims.rename(columns={"zoning_id":"muni_base_id"})
    
    # 8. Add muni_pd_id and muni_overlay_id
    pd_idx = zp_find_district_idx(parcel_dims, pd_districts).rename(columns={"zoning_id":"muni_pd_id"})
    ov_idx = zp_find_district_idx(parcel_dims, overlays).rename(columns={"zoning_id":"muni_overlay_id"})
    parcel_dims = parcel_dims.merge(pd_idx[["parcel_id","muni_pd_id"]], on='parcel_id', how="left")
    parcel_dims = parcel_dims.merge(ov_idx[["parcel_id","muni_overlay_id"]], on='parcel_id', how="left")

    # 9. Add dist_abbr and muni_name for all zone IDs
    dist_abbr_map = zoning_all.set_index("zoning_id")["dist_abbr"].to_dict()
    muni_name_map = zoning_all.set_index("zoning_id")["muni_name"].to_dict()
    def collect_all(ids, mapping):
        """给一个 list of zoning_id，去 mapping 里拿值并去重。"""
        vals = []
        for z_id in ids:
            if pd.notna(z_id):
                v = mapping.get(z_id)
                if isinstance(v, (list, tuple)):
                    vals.extend(v)
                elif v is not None:
                    vals.append(v)
        seen = set()
        uniq = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    parcel_dims["all_zone_ids"] = parcel_dims[[
        "muni_base_id","muni_pd_id","muni_overlay_id"
    ]].values.tolist()
    parcel_dims["dist_abbr"] = parcel_dims["all_zone_ids"].apply(lambda ids: collect_all(ids, dist_abbr_map))
    parcel_dims["muni_name"] = parcel_dims["all_zone_ids"].apply(lambda ids: collect_all(ids, muni_name_map))
    
    # 10. Initialize false_reasons and maybe_reasons columns
    parcel_dims["false_reasons"] = None
    parcel_dims["maybe_reasons"] = None
    false_dfs = []
    maybe_dfs = []
    
    if print_checkpoints:
        elapsed = time.time() - start_time
        print(f"___data_prep___ {elapsed:.1f}s\n")

    # ————————— 2. Planned Development check ————————— #
    t0 = time.time()
    if not pd_districts.empty:
        # Identify parcels in planned development districts via muni_pd_id
        pd_parcels = pd_idx["parcel_id"][pd_idx["muni_pd_id"].notna()]
        # Mark these parcels as FALSE and record the reason
        mask_pd = parcel_dims["parcel_id"].isin(pd_parcels)
        parcel_dims.loc[mask_pd, "false_reasons"] = (
            parcel_dims.loc[mask_pd, "false_reasons"]
            .fillna("") .apply(lambda s: "PD_dist" if s=="" else s+",PD_dist")
        )
        # Add intermediate variable check_pd for detailed_check logic
        parcel_dims["check_pd"] = ~mask_pd
        if not detailed_check:
            df0 = parcel_dims[ mask_pd ]
            false_dfs.append(df0)
            parcel_dims = parcel_dims[~mask_pd]
        if print_checkpoints:
            print(f"___planned_dev_check___ {time.time()-t0:.1f}s, kept {parcel_dims.shape[0]} parcels\n")

    # ————————— 3. District checks ————————— #
    t1 = time.time()
    # 1. Warn about parcels crossing multiple base zones
    mask_dist_cross = parcel_dims["muni_base_id"].apply(lambda x: isinstance(x, (list, tuple)) and len(x) > 1)
    crossing = parcel_dims.loc[mask_dist_cross, "parcel_id"]
    if not crossing.empty:
        # Append 'cross_base_district' to maybe_reasons
        parcel_dims.loc[mask_dist_cross, "maybe_reasons"] = (
            parcel_dims.loc[mask_dist_cross, "maybe_reasons"]
            .fillna("")  # None -> ""
            .apply(lambda s: "cross_base_district" if s == "" else s + ",cross_base_district")
        )
        warnings.warn(f"{len(crossing)}/{len(parcel_dims)} parcels acrossed by zoning")
        # Record intermediate flag for detailed_check logic
        parcel_dims["check_cross_base"] = ~mask_dist_cross
    # 2. Warn about parcels not covered by any district
    mask_dist_cover = (parcel_dims["muni_base_id"].isna() & parcel_dims["muni_pd_id"].isna() & parcel_dims["muni_overlay_id"].isna())
    missing = parcel_dims.loc[mask_dist_cover, "parcel_id"]
    if not missing.empty:
        # Append 'no_district' to maybe_reasons
        parcel_dims.loc[mask_dist_cover, "maybe_reasons"] = (
            parcel_dims.loc[mask_dist_cover, "maybe_reasons"]
            .fillna("")  # None -> ""
            .apply(lambda s: "no_district" if s == "" else s + ",no_district")
        )
        warnings.warn(f"{len(missing)}/{len(parcel_dims)} parcels not covered by zoning")
        # Add intermediate variable check_dist_cover for detailed_check logic
        parcel_dims["check_dist_cover"] = ~mask_dist_cover

    # Handle cross-base and no-district cases
    df_cross = parcel_dims.loc[mask_dist_cross].copy()
    mask_dist = mask_dist_cross | mask_dist_cover
    df_dist = parcel_dims.loc[mask_dist]
    maybe_dfs.append(df_dist)
    parcel_dims = parcel_dims.loc[~mask_dist]
    if print_checkpoints:
        print(f"___cross_no_dist_check___ {time.time()-t1:.1f}s, kept {parcel_dims.shape[0]} parcels\n")
            
    # ————————— 4. Variables & requirements ————————— #
    t2 = time.time()
    vars_map = {}
    req_map  = {}
    no_setback = set()
    for idx in parcel_dims.index:
        # Use double brackets to extract a DataFrame instead of a Series
        parcel_data = parcel_dims.loc[[idx]]   
        pid = parcel_data.at[idx, "parcel_id"]
        # Find the corresponding district subset DataFrame
        ids = (parcel_data.at[idx, "muni_base_id"]
            if isinstance(parcel_data.at[idx, "muni_base_id"], (list, tuple))
            else [parcel_data.at[idx, "muni_base_id"]])
        dist_df = zoning_all[zoning_all["zoning_id"].isin(ids)]
        # Skip if no district found
        if dist_df.empty:
            continue
        # Extract muni JSON definitions from the first row of dist_df
        muni = int(dist_df.iloc[0]["muni_id"])
        munijson = zoning_jsons[muni]["definitions"]
        # Call zp_get_variables with correct types
        v = zp_get_variables(bldg_data, parcel_data, dist_df, munijson)
        z = zp_get_zoning_req(dist_df, bldg_data=None, parcel_data=None, zoning_data=None, vars=v)
        vars_map[pid] = v
        req_map[pid]  = z
        # Record parcels with no setbacks
        if isinstance(z, str):
            no_setback.add(pid)
        else:
            sb = z[z["constraint_name"].str.contains("setback")]["min_value"]
            if sb.isnull().all() or sb.apply(lambda x: sum(x if isinstance(x,(list,tuple)) else [x])).sum()==0:
                no_setback.add(pid)
    if print_checkpoints:
        print(f"___get_zoning_req___ {time.time()-t2:.1f}s\n")

    # ————————— 5. Initial checks (res_type, constraints, unit_size) ————————— #
    t3 = time.time()
    init_results = {}
    for idx in parcel_dims.index:
        parcel_data = parcel_dims.loc[[idx]]
        pid = parcel_data.at[idx, "parcel_id"]
        ids = (parcel_data.at[idx, "muni_base_id"]
            if isinstance(parcel_data.at[idx, "muni_base_id"], (list, tuple))
            else [parcel_data.at[idx, "muni_base_id"]])
        dist_df = zoning_all[zoning_all["zoning_id"].isin(ids)]
        if dist_df.empty:
            continue
        v   = vars_map[pid]
        z   = req_map[pid]

        # 1) Residential type check
        if "res_type" in checks:
            rt_ok = zp_check_res_type(v, dist_df)
            df_rt = pd.DataFrame({"res_type":[rt_ok]})
        else:
            df_rt = pd.DataFrame()

        # 2) Other quantitative constraints
        try:
            df_cons = zp_check_constraints(v, z, checks=initial_checks)
        except:
            df_cons = pd.DataFrame()
        if not df_cons.empty:
            df_c = df_cons.set_index("constraint_name")["allowed"].to_frame().T
        else:
            df_c = pd.DataFrame()

        # 3) Unit size check
        if "unit_size" in checks:
            u_ok = zp_check_unit(dist_df, bldg_data, v)
            df_u = pd.DataFrame({"unit_size":[u_ok]})
        else:
            df_u = pd.DataFrame()

        # Combine results into one DataFrame
        merged = pd.concat([df_rt, df_c.reset_index(drop=True), df_u], axis=1)
        merged.index = [idx]
        init_results[idx] = merged

        # Record reasons for FALSE/MAYBE
        vals = merged.iloc[0].astype(str)
        fals = vals[vals=="False"].index.tolist()
        mays = vals[vals=="MAYBE"].index.tolist()
        if fals:
            s = parcel_dims.at[idx,"false_reasons"] or ""
            parcel_dims.at[idx,"false_reasons"] = (",".join(fals) if s=="" else s + "," + ",".join(fals))
        if mays:
            s = parcel_dims.at[idx,"maybe_reasons"] or ""
            parcel_dims.at[idx,"maybe_reasons"] = (",".join(mays)if s=="" else s + "," + ",".join(mays))

        # If FALSE and not detailed_check, remove parcel to false_dfs
        if ("False" in vals.values) and (not detailed_check):
            false_dfs.append(parcel_dims.loc[[idx]])
            parcel_dims.drop(idx, inplace=True)

    # Concatenate all individual results into one DataFrame, preserving original idx as index
    init_df = pd.concat(init_results, axis=0)
    # init_df now has a two-level index; drop the first level (original idx)
    init_df.index = init_df.index.droplevel(0)
    # Treat NA as True (matching R behavior)
    init_df = init_df.fillna(True)
    # Join init_df back to parcel_dims aligned by index
    parcel_dims = parcel_dims.join(init_df)
    if print_checkpoints:
        print(f"___initial_checks___ {time.time()-t3:.1f}s\n")

    # ————————— 6. Side label check ————————— #
    if "bldg_fit" in checks and not parcel_dims.empty:
        t4 = time.time()
        known = set(parcel_geo.loc[parcel_geo["side"]!="unknown","parcel_id"])
        keep = known.union(no_setback)
        # Mark parcel_side_lbl flag
        parcel_dims["parcel_side_lbl"] = parcel_dims["parcel_id"].isin(keep)
        # Parcels without side label get MAYBE reason
        mask = ~parcel_dims["parcel_side_lbl"]
        parcel_dims.loc[mask, "maybe_reasons"] = (
            parcel_dims.loc[mask, "maybe_reasons"]
            .fillna("") .apply(lambda s: "side_lbl" if s=="" else s+",side_lbl")
        )
        # If not detailed_check, remove them to false_dfs
        if not detailed_check:
            false_dfs.append(parcel_dims[mask])
            parcel_dims = parcel_dims[~mask]
        if print_checkpoints:
            print(f"___side_label_check___ {time.time()-t4:.1f}s\n")

    # ————————— 7. Building fit check ————————— #
    parcel_dims["bldg_fit"] = None
    parcel_dims["bldg_fit"] = parcel_dims["bldg_fit"].astype(object)

    if "bldg_fit" in checks and not parcel_dims.empty:
        t5 = time.time()
        for idx in parcel_dims.index:
            if not parcel_dims.at[idx, "parcel_side_lbl"]:
                continue
            pid = parcel_dims.at[idx, "parcel_id"]
            sides = parcel_geo[parcel_geo["parcel_id"] == pid]
            if sides.empty:
                continue
            base_ids = parcel_dims.at[idx, "muni_base_id"]
            if not isinstance(base_ids, (list, tuple)):
                base_ids = [base_ids]
            dist_df = zoning_all[zoning_all["zoning_id"].isin(base_ids)]
            if dist_df.empty:
                continue
            v   = vars_map[pid]
            z   = req_map[pid]
            pg_sb = zp_add_setbacks(sides, dist_df, z)
            bz    = zp_get_buildable_area(pg_sb)
            fitdf = zp_check_fit(bz, v)
            ok = fitdf.loc[0,"allowed"]
            parcel_dims.at[idx,"bldg_fit"] = ok
            if ok=="MAYBE":
                s = parcel_dims.at[idx,"maybe_reasons"] or ""
                parcel_dims.at[idx,"maybe_reasons"] = "bldg_fit" if s=="" else s+",bldg_fit"
            if ok==False:
                s = parcel_dims.at[idx,"false_reasons"] or ""
                parcel_dims.at[idx,"false_reasons"] = "bldg_fit" if s=="" else s+",bldg_fit"
        if not detailed_check:
            mask = parcel_dims["bldg_fit"]==False
            false_dfs.append(parcel_dims[mask])
            parcel_dims = parcel_dims[~mask]
        if print_checkpoints:
            print(f"___bldg_fit___ {time.time()-t5:.1f}s\n")

    # ————————— 8. Overlay check ————————— #
    if "overlay" in checks and not overlays.empty:
        t6 = time.time()
        o_parcels = set(overlays["parcel_id"])
        mask = parcel_dims["parcel_id"].isin(o_parcels)
        parcel_dims.loc[mask, "maybe_reasons"] = (
            parcel_dims.loc[mask, "maybe_reasons"]
            .fillna("") .apply(lambda s: "overlay" if s=="" else s+",overlay")
        )
        if print_checkpoints:
            print(f"___overlay_check___ {time.time()-t6:.1f}s\n")

    # ————————— 9. Merge & output ————————— #
    # Concatenate all parcels removed to false_dfs plus those still in parcel_dims
    final = pd.concat(false_dfs + [parcel_dims], ignore_index=True)
    
    # Compute allowed & reason columns
    def summarize(r):
        fr = r.get("false_reasons") or ""
        mr = r.get("maybe_reasons") or ""
        if fr: return (False, fr)
        if mr: return ("MAYBE", mr)
        return (True, "Building allowed")

    final[["allowed","reason"]] = pd.DataFrame(
        final.apply(summarize, axis=1).tolist(),
        index=final.index
    )

    # Select output columns based on detailed_check
    if not detailed_check:
        out = final[[
            "parcel_id","muni_name","dist_abbr","allowed","reason","geometry"
        ]].copy()
    else:
        drop_cols = [
            "false_reasons","maybe_reasons",
            "lot_width","lot_depth","lot_area","lot_type",
            "zoning_id","pd_id","overlay_id"
        ]
        keep = [c for c in final.columns if c not in drop_cols]
        out = final[keep].copy()

    # Handle duplicate parcel_id cases (overlapping zones)
    dups = out["parcel_id"][out["parcel_id"].duplicated()].unique()
    if len(dups)>0:
        merged = []
        for pid in dups:
            sub = out[out["parcel_id"]==pid]
            vals = sub["allowed"].tolist()
            if all(v==True for v in vals):
                v = True
            elif all(v==False for v in vals):
                v = False
            else:
                v = "MAYBE"
            reasons = " ---||--- ".join(sub["reason"].tolist())
            row = sub.iloc[0].copy()
            row["allowed"] = v
            row["reason"]  = reasons
            merged.append(row)
        out = out[~out["parcel_id"].isin(dups)].append(merged, ignore_index=True)

    # Save to GeoJSON if requested
    if save_to:
        dirname = os.path.dirname(save_to) or "."
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        out.to_file(save_to, driver="GeoJSON")
        if print_checkpoints:
            print(f"output saved to {save_to}")

    if print_checkpoints:
        total = time.time() - start_time
        ct_true = (out["allowed"]==True).sum()
        ct_may  = (out["allowed"]=="MAYBE").sum()
        print(f"_____summary_____")
        print(f"total runtime: {total:.1f}s")
        print(f"{ct_true}/{len(out)} parcels allow the building; {ct_may}/{len(out)} maybe allow")

    return gpd.GeoDataFrame(out, geometry="geometry", crs=parcels_sf.crs)
