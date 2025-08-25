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
    overlays     = zoning_all[zoning_all["overlay"] == True]
    overlays_list = overlays['zoning_id'].tolist()
    pd_districts = zoning_all[zoning_all["planned_dev"] == True]
    pd_list = pd_districts['zoning_id'].tolist()
    pd_overlay = zoning_all[(zoning_all["overlay"] == True) & (zoning_all["planned_dev"]== True)]
    pd_overlay_list = pd_overlay['zoning_id'].tolist()
    base_zones   = zoning_all[zoning_all["overlay"] == False]
    base_list = base_zones['zoning_id'].tolist()

    # 5. Read parcels and assign base zoning IDs
    parcel_list = [ zp_read_pcl(p, zoning_all) for p in parcel_files ]
    parcels_sf = pd.concat(parcel_list, ignore_index=True)
    # 6. Generate parcel_geo with 'side' labels
    parcel_geo  = zp_get_parcel_geo(parcels_sf)
    # 7. Generate parcel_dims with centroid and dimensions, rename zoning_id to muni_base_id
    parcel_dims = zp_get_parcel_dim(parcels_sf)
    # 8. Add parcel attribute
    parcel_dims['is_overlay'] = parcel_dims['zoning_id'].isin(overlays_list)
    parcel_dims['is_pd'] = parcel_dims['zoning_id'].isin(pd_list)
    parcel_dims['is_pd_overlay'] = parcel_dims['zoning_id'].isin(pd_overlay_list)
    parcel_dims['is_base'] = parcel_dims['zoning_id'].isin(base_list)
    parcel_dims = parcel_dims.drop_duplicates(keep='first').reset_index(drop=True)
    
    # 9. Add dist_abbr and muni_name for all zone IDs
    dist_abbr_map = zoning_all.set_index("zoning_id")["dist_abbr"].to_dict()
    muni_name_map = zoning_all.set_index("zoning_id")["muni_name"].to_dict()

    parcel_dims["dist_abbr"] = parcel_dims["zoning_id"].map(dist_abbr_map)
    parcel_dims["muni_name"] = parcel_dims["zoning_id"].map(muni_name_map)
    
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
        # Two masks: PD with overlay first, then standard PD
        mask_pd_overlay = (parcel_dims["is_pd_overlay"] == True)
        mask_pd_dist    = (parcel_dims["is_pd"] == True) & ~mask_pd_overlay
        # Write FALSE reason
        def _append_reason(df, col, mask, tag):
            cur = df.loc[mask, col].fillna("")
            df.loc[mask, col] = np.where(cur == "", tag, cur + "," + tag)
        _append_reason(parcel_dims, "false_reasons", mask_pd_overlay, "PD_overlay")
        _append_reason(parcel_dims, "false_reasons", mask_pd_dist,    "PD_dist")
        # Whether it passes the PD check (R: check_pd)
        parcel_dims["check_pd"] = ~(mask_pd_overlay | mask_pd_dist)
        # If not detailed_check: drop parcels that are FALSE in the PD stage
        if not detailed_check:
            pd_false = parcel_dims.loc[~parcel_dims["check_pd"]]
            false_dfs.append(pd_false)
            parcel_dims = parcel_dims.loc[parcel_dims["check_pd"]].copy()
    else:
        parcel_dims["check_pd"] = True
    if print_checkpoints:
        print(f"___planned_dev_check___ {time.time()-t0:.1f}s, kept {parcel_dims.shape[0]} parcels\n")
        
    # ————————— 3. District checks ————————— #
    t1 = time.time()
    # 1. Warn about parcels not covered by any district
    mask_dist_cover = parcel_dims["zoning_id"].isna()
    missing = parcel_dims.loc[mask_dist_cover, "parcel_id"]
    if not missing.empty:
        # Append 'no_district' to maybe_reasons
        parcel_dims.loc[mask_dist_cover, "maybe_reasons"] = (
            parcel_dims.loc[mask_dist_cover, "maybe_reasons"]
            .fillna("")  # None -> ""
            .apply(lambda s: "no_district" if s == "" else s + ",no_district")
        )
        # Add intermediate variable check_dist_cover for detailed_check logic
        parcel_dims["check_dist_cover"] = ~mask_dist_cover
        # Handle cross-base and no-district cases
        if not detailed_check:
            df_dist = parcel_dims.loc[mask_dist_cover]
            maybe_dfs.append(df_dist)
            parcel_dims = parcel_dims.loc[~mask_dist_cover]
    else:
        parcel_dims["check_dist_cover"] = True
    if print_checkpoints:
        print(f"___no_dist_check___ {time.time()-t1:.1f}s, kept {parcel_dims.shape[0]} parcels\n")
            
    # ————————— 4. Variables & requirements ————————— #
    t2 = time.time()
    vars_map = {}
    req_map  = {}
    no_setback = set()

    def safe_sum(x):
        """
        Safely sum a value that may be None, NaN, or a list/tuple containing None/NaN.
        - None or NaN are treated as 0.
        - Lists/tuples are filtered of None/NaN before summing.
        """
        # Treat None or NaN as 0
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        # If list or tuple, filter out None/NaN and sum
        if isinstance(x, (list, tuple)):
            return sum(i for i in x if i is not None and not (isinstance(i, float) and np.isnan(i)))
        # Otherwise, return the numeric value directly
        return x

    for idx in parcel_dims.index:
        # Extract a one-row DataFrame (keeps column types) instead of a Series
        parcel_data = parcel_dims.loc[[idx]]
        pid = parcel_data.at[idx, "parcel_id"]
        # Build list of zoning IDs (may be a single ID or a list)
        ids = (parcel_data.at[idx, "zoning_id"]
            if isinstance(parcel_data.at[idx, "zoning_id"], (list, tuple))
            else [parcel_data.at[idx, "zoning_id"]])
        # Subset zoning_all by those IDs
        dist_df = zoning_all[zoning_all["zoning_id"].isin(ids)]
        # Skip if there isn’t exactly one matching district
        if dist_df.shape[0] != 1:
            continue
        # Load that municipality’s JSON definitions
        muni_id = int(dist_df.iloc[0]["muni_id"])
        munijson = zoning_jsons[muni_id]["definitions"]
        # Compute variables and zoning requirements
        v = zp_get_variables(bldg_data, parcel_data, dist_df, munijson)
        z = zp_get_zoning_req(dist_df, bldg_data=None, parcel_data=None, zoning_data=None, vars=v)
        vars_map[pid] = v
        req_map[pid]  = z
        # Identify parcels with no setback requirements
        if isinstance(z, str):
            # If z is an error string, treat as no setbacks
            no_setback.add(pid)
        else:
            # Filter constraints containing “setback”
            sb = z[z["constraint_name"].str.contains("setback")]["min_value"]
            # If all are NaN or their safe sum is zero, mark as no setback
            if sb.isnull().all() or sb.apply(safe_sum).sum() == 0:
                no_setback.add(pid)
    if print_checkpoints:
        print(f"___get_zoning_req___ {time.time()-t2:.1f}s\n")

    # ————————— 5. Initial checks (res_type, constraints, unit_size) ————————— #
    t3 = time.time()
    init_results = {}
    for idx in parcel_dims.index:
        parcel_data = parcel_dims.loc[[idx]]
        pid = parcel_data.at[idx, "parcel_id"]
        ids = (parcel_data.at[idx, "zoning_id"]
            if isinstance(parcel_data.at[idx, "zoning_id"], (list, tuple))
            else [parcel_data.at[idx, "zoning_id"]])
        dist_df = zoning_all[zoning_all["zoning_id"].isin(ids)]
        # Skip if there isn’t exactly one matching district
        if dist_df.shape[0] != 1:
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
        known = set(parcel_geo["parcel_id"].unique())
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
            base_ids = parcel_dims.at[idx, "zoning_id"]
            if not isinstance(base_ids, (list, tuple)):
                base_ids = [base_ids]
            dist_df = zoning_all[zoning_all["zoning_id"].isin(base_ids)]
            if dist_df.shape[0] != 1:
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
        mask_overlay = parcel_dims["is_overlay"] == True
        cur = parcel_dims.loc[mask_overlay, "maybe_reasons"].fillna("")
        parcel_dims.loc[mask_overlay, "maybe_reasons"] = np.where(
            cur == "", "overlay", cur + ",overlay"
        )
        parcel_dims["check_overlay"] = np.where(mask_overlay, "MAYBE", True)

    if print_checkpoints:
        print(f"___overlay_check___ {time.time()-t6:.1f}s\n")

    # ————————— 9. Merge & output ————————— #
    # Concatenate all parcels removed to false_dfs plus those still in parcel_dims
    final = pd.concat(false_dfs + maybe_dfs + [parcel_dims], ignore_index=True)
    # Immediately deduplicate after summarize() to avoid identical rows entering subsequent steps
    final["_geom_wkb"] = final.geometry.apply(lambda g: g.wkb if g is not None else None)
    final = final.drop_duplicates(
        subset=[c for c in ["parcel_id", "zoning_id", "false_reasons","maybe_reasons","_geom_wkb"] if c in final.columns],
        keep="first"
    )
    final = final.drop(columns=["_geom_wkb"])
    final = final.reset_index(drop=True)
    
    # Compute allowed & reason columns
    def summarize(r):
        fr = r.get("false_reasons")
        mr = r.get("maybe_reasons")
        if fr: return (False, fr)
        if mr: return ("MAYBE", mr)
        return (True, "Building allowed")

    final[["allowed","reason"]] = pd.DataFrame(
        final.apply(summarize, axis=1).tolist(),
        index=final.index
    )
    final["is_duplicate"] = final.duplicated(subset=["parcel_id"], keep=False)
            
    # def _harmonize_with_R(final_df, *, detailed=False):
    #     """
    #     与R严格对齐的后处理（仅操作 final 表，不改前面判定）：
    #     1) 若组内出现 PD_overlay -> 整组：一行（或多行？见实现），allowed=False, reason='PD_overlay'
    #     2) 否则若组内出现 PD（非 overlay）-> 整组：allowed=False, reason='PD_dist'
    #     3) 否则若同一 parcel_id 有多个 base 覆盖 -> 保留多行（不折叠），并 is_duplicate=True
    #     4) 否则：
    #     - overlay（非PD） -> allowed='MAYBE'，reason 添加 'overlay'
    #     - 结合组内 allowed：False > MAYBE > True
    #     - 折叠为一行，代表行仅用于 muni_name/dist_abbr/geometry 取值
    #     detailed=False：输出精简列；True：尽量保留更多列（从代表行取）
    #     """
    #     import pandas as pd
    #     import geopandas as gpd

    #     if final_df.empty:
    #         return gpd.GeoDataFrame(final_df, geometry="geometry", crs=getattr(final_df, "crs", None))

    #     # 工具函数
    #     def _normalize_reasons(series):
    #         s = series.fillna("").astype(str)
    #         # 统一逗号空格
    #         s = s.str.replace(r"\s*,\s*", ", ", regex=True)
    #         return s

    #     def _agg_allowed(vals):
    #         vals = [str(v) for v in vals]
    #         if any(v.upper() == "FALSE" or v is False for v in vals):
    #             return False
    #         if any(v == "MAYBE" for v in vals):
    #             return "MAYBE"
    #         return True

    #     def _join_reasons(reasons):
    #         parts = []
    #         for r in reasons:
    #             if not r:
    #                 continue
    #             for p in str(r).split(","):
    #                 p = p.strip()
    #                 if p:
    #                     parts.append(p)
    #         # 去重，维持顺序
    #         uniq = list(dict.fromkeys(parts))
    #         # 保留 R 优先级语义（但这里只在 overlay 场景用；PD_* 已在前面处理掉）
    #         return ", ".join(uniq) if uniq else "Building allowed"

    #     def _row_rank(row):
    #         # 代表行优先级（只用于拿 labels/geometry；不影响 allowed/reason 聚合）
    #         # base(0) > PD_overlay(1) > PD_dist(2) > overlay(3) > other(4)
    #         if row.get("is_base", False):
    #             return 0
    #         if row.get("is_pd_overlay", False):
    #             return 1
    #         if row.get("is_pd", False):
    #             return 2
    #         if row.get("is_overlay", False):
    #             return 3
    #         return 4

    #     out_blocks = []
    #     slim_cols = [c for c in ["parcel_id","muni_name","dist_abbr","allowed","reason","geometry","is_duplicate"] if c in final_df.columns]

    #     # 规范一次 reason
    #     if "reason" in final_df.columns:
    #         final_df = final_df.copy()
    #         final_df["reason"] = _normalize_reasons(final_df["reason"])

    #     for pid, g in final_df.groupby("parcel_id", dropna=False):
    #         g = g.copy()

    #         # 1) PD_overlay 整组否决
    #         if "is_pd_overlay" in g.columns and g["is_pd_overlay"].any():
    #             # 保持一行输出更清晰；代表行取 is_pd_overlay=True 的第一行，否则按 rank
    #             g["_rank"] = g.apply(_row_rank, axis=1)
    #             gg = g[g.get("is_pd_overlay", False) == True] if "is_pd_overlay" in g.columns else g
    #             top = gg.iloc[0] if not gg.empty else g.sort_values("_rank").iloc[0]
    #             rec = {
    #                 "parcel_id": pid,
    #                 "muni_name": top.get("muni_name"),
    #                 "dist_abbr": top.get("dist_abbr"),
    #                 "allowed": False,
    #                 "reason": "PD_overlay",
    #                 "geometry": top.get("geometry"),
    #                 "is_duplicate": (len(g) > 1)
    #             }
    #             if detailed:
    #                 # 尽量保留更多列（从代表行取），再覆盖关键列
    #                 full = top.to_dict()
    #                 full.update(rec)
    #                 out_blocks.append(full)
    #             else:
    #                 out_blocks.append(rec)
    #             continue

    #         # 2) 仅 PD（非 overlay）整组否决
    #         if "is_pd" in g.columns and g["is_pd"].any():
    #             # 但剔除 overlay 的 PD 已在上一步处理；这里只要存在 is_pd=True 的行即可
    #             g["_rank"] = g.apply(_row_rank, axis=1)
    #             gg = g[g.get("is_pd", False) == True]
    #             top = gg.iloc[0] if not gg.empty else g.sort_values("_rank").iloc[0]
    #             rec = {
    #                 "parcel_id": pid,
    #                 "muni_name": top.get("muni_name"),
    #                 "dist_abbr": top.get("dist_abbr"),
    #                 "allowed": False,
    #                 "reason": "PD_dist",
    #                 "geometry": top.get("geometry"),
    #                 "is_duplicate": (len(g) > 1)
    #             }
    #             if detailed:
    #                 full = top.to_dict()
    #                 full.update(rec)
    #                 out_blocks.append(full)
    #             else:
    #                 out_blocks.append(rec)
    #             continue

    #         # 3) 多 base -> 保留多行（不折叠），并 is_duplicate=True
    #         is_base = g.get("is_base", pd.Series(False, index=g.index))
    #         base_rows = g[is_base] if "is_base" in g.columns else g.iloc[0:0]
    #         multi_base = False
    #         if not base_rows.empty:
    #             if "zoning_id" in base_rows.columns:
    #                 multi_base = base_rows["zoning_id"].nunique(dropna=True) > 1
    #             elif "dist_abbr" in base_rows.columns:
    #                 multi_base = base_rows["dist_abbr"].nunique(dropna=True) > 1

    #         if multi_base:
    #             gg = g.copy()
    #             gg["is_duplicate"] = True
    #             # Overlay（非PD）行若存在，可把它们的 allowed 调整为 MAYBE 并在 reason 加 overlay（不改动原判定也行）
    #             # 但与R一致：R是在 overlay 阶段把通过所有检查的剩余行标注 MAYBE；这里保持已有 allowed/ reason，仅统一 is_duplicate。
    #             if detailed:
    #                 out_blocks.append(gg)
    #             else:
    #                 out_blocks.append(gg[slim_cols] if slim_cols else gg)
    #             continue

    #         # 4) 单一 base / 无 base：按 overlay 与 allowed 聚合后折叠一行
    #         # 组层面的 allowed 聚合
    #         grp_allowed = _agg_allowed(g["allowed"]) if "allowed" in g.columns else True

    #         # overlay（非PD）→ MAYBE + reason 加 'overlay'
    #         # 注意：若本就有 False，这里的 MAYBE 不覆盖 False
    #         if "is_overlay" in g.columns and g["is_overlay"].any():
    #             if grp_allowed is True:
    #                 grp_allowed = "MAYBE"
    #             # 合成/追加 overlay 到 reason（只在最终为 TRUE/MAYBE 时有意义）
    #             if "reason" in g.columns:
    #                 all_r = g["reason"].tolist() + ["overlay"]
    #                 grp_reason = _join_reasons(all_r)
    #             else:
    #                 grp_reason = "overlay"
    #         else:
    #             # 没有 overlay：按已有 reason 聚合
    #             grp_reason = _join_reasons(g["reason"].tolist()) if "reason" in g.columns else (
    #                 "Building allowed" if grp_allowed is True else ("MAYBE" if grp_allowed == "MAYBE" else "")
    #             )

    #         # 代表行用于拿 labels/geometry 或 detailed 时补其它列
    #         g["_rank"] = g.apply(_row_rank, axis=1)
    #         top = g.sort_values("_rank").iloc[0]

    #         base_rec = {
    #             "parcel_id": pid,
    #             "muni_name": top.get("muni_name"),
    #             "dist_abbr": top.get("dist_abbr"),
    #             "allowed": grp_allowed,
    #             "reason": grp_reason if grp_reason else ("Building allowed" if grp_allowed is True else ""),
    #             "geometry": top.get("geometry"),
    #             "is_duplicate": (len(g) > 1)
    #         }

    #         if detailed:
    #             full = top.to_dict()
    #             full.update(base_rec)
    #             out_blocks.append(full)
    #         else:
    #             out_blocks.append(base_rec)

    #     # 组装输出
    #     if any(isinstance(b, (pd.DataFrame, gpd.GeoDataFrame)) for b in out_blocks):
    #         out = pd.concat([b if isinstance(b, (pd.DataFrame, gpd.GeoDataFrame)) else pd.DataFrame([b]) for b in out_blocks],
    #                         ignore_index=True)
    #     else:
    #         out = pd.DataFrame(out_blocks)

    #     out = gpd.GeoDataFrame(out, geometry="geometry" if "geometry" in out.columns else None, crs=getattr(final_df, "crs", None))

    #     # 非详细：统一列序
    #     if not detailed:
    #         col_order = [c for c in ["parcel_id","muni_name","dist_abbr","allowed","reason","geometry","is_duplicate"] if c in out.columns]
    #         out = out[col_order]
    #     else:
    #         # 详细：确保 reason 逗号空格规范
    #         if "reason" in out.columns:
    #             out["reason"] = _normalize_reasons(out["reason"])

    #     return out

    # # —— 与 R 完全对齐的后处理（利用 is_* 标志列）——
    # if detailed_check:
    #     # detailed：先保留你想要的列，再做R对齐的组裁决
    #     drop_cols = ["false_reasons","maybe_reasons","lot_width","lot_depth","lot_area","lot_type"]
    #     keep = [c for c in final.columns if c not in drop_cols]
    #     interim = final[keep].copy()
    #     out = _harmonize_with_R(interim, detailed=True)
    # else:
    #     # 非详细：R默认展示口径（精简列），但严格遵循PD/overlay/多base规则
    #     cols_for = [c for c in ["parcel_id","muni_name","dist_abbr","allowed","reason","geometry","is_duplicate","zoning_id",
    #                             "is_overlay","is_pd","is_pd_overlay","is_base"]
    #                 if c in final.columns]
    #     interim = final[cols_for].copy()
    #     out = _harmonize_with_R(interim, detailed=False)

    # Select output columns based on detailed_check
    if not detailed_check:
        out = final[[
            "parcel_id","muni_name","dist_abbr","allowed","reason","geometry","is_duplicate","zoning_id"
        ]].copy()
    else:
        drop_cols = [
            "false_reasons","maybe_reasons",
            "lot_width","lot_depth","lot_area","lot_type",
        ]
        keep = [c for c in final.columns if c not in drop_cols]
        out = final[keep].copy()

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

    return gpd.GeoDataFrame(out, geometry="geometry", crs=parcels_sf.crs).to_crs(epsg=4326)

