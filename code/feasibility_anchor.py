# -*- coding: utf-8 -*-
"""Feasibility audit for a real Ka-Ku observational anchor (P5).

Question: can the eta_diff thresholds of the manuscript be placed on one side
of the boundary with observations that exist on this machine?

The script does four things and writes out/feasibility_anchor.{json,csv}:

1. inventories the local holdings by frequency band (Ku / laser / optical /
   lidar / model) and checks for any Ka-band (SARAL/AltiKa, CRISTAL) file;
2. measures the CryoSat-2 (Ku) coverage actually available (98 monthly CCI
   L3C files, 2010-10 to 2024-04): valid 25 km cell-months and the latitude
   distribution of the thick-ice state used in the paper;
3. measures the only physically possible local pairing, Ku x laser, against
   ICESat-2 ATL20 V005 (19 monthly files, 2018-10 to 2020-04): overlap months,
   co-located valid cell-months on the two 25 km grids;
4. converts that sample size into the slope standard error SE(eta) for three
   differential-accuracy levels, and compares 2.8016*SE with the four
   thresholds quoted in the paper.

Data are read-only; nothing is downloaded. Run from code/:

    python feasibility_anchor.py
"""
import glob
import json
import os
import re

import netCDF4
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
OUT = os.path.join(ROOT, "out")
def _root(flag, env, default):
    """Data root: command-line flag, then environment, then a documented default.

    The holdings audited in the paper are read-only archives on the authors'
    machine and are not redistributed with this package.  The archived output
    of this script for those holdings is out/feasibility_anchor.json.
    """
    import argparse
    import sys
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return os.environ.get(env, default)


DATA = _root("--data-root", "P5_DATA_ROOT",
             os.path.join(os.path.expanduser("~"), "seaice_holdings"))
ATL20 = _root("--atl20", "P5_ATL20_ROOT",
              os.path.join(DATA, "icesat2", "atl20"))

# ---------------------------------------------------------------- inventory
SENSOR_DIRS = {
    "CCI CryoSat-2 (Ku)": os.path.join(DATA, "cci"),
    "CCI Envisat (Ku)": os.path.join(DATA, "cci_sensors", "envisat"),
    "CCI ERS-2 (Ku)": os.path.join(DATA, "cci_sensors", "ers2"),
    "CCI Sentinel-3A (Ku)": os.path.join(DATA, "cci_sensors", "sentinel3a"),
    "CCI Sentinel-3B (Ku)": os.path.join(DATA, "cci_sensors", "sentinel3b"),
    "CS2SMOS v206 (Ku+SMOS)": os.path.join(DATA, "cs2smos"),
    "PIOMAS (model)": os.path.join(DATA, "piomas"),
    "ICESat-2 ATL20 (laser)": ATL20,
}


def inventory():
    rows = []
    ka_hits = []
    for name, d in SENSOR_DIRS.items():
        n = len(glob.glob(os.path.join(d, "**", "*.nc"), recursive=True)) + \
            len(glob.glob(os.path.join(d, "**", "*.h5"), recursive=True))
        rows.append({"dataset": name, "path": d, "n_files": n})
    for base in (DATA, ATL20):
        for f in glob.glob(os.path.join(base, "**", "*"), recursive=True):
            if re.search(r"altika|saral|cristal|SARAL|ALTIKA|CRISTAL", os.path.basename(f)):
                ka_hits.append(f)
    return rows, ka_hits


# ------------------------------------------------------- CCI Ku coverage
def cci_stats():
    files = sorted(glob.glob(os.path.join(DATA, "cci", "*.nc")))
    months = [re.search(r"(\d{6})-fv", os.path.basename(f)).group(1) for f in files]
    per_month, lat_s, th_s = [], [], []
    for f in files:
        d = netCDF4.Dataset(f)
        th = np.ma.filled(np.ma.array(d.variables["sea_ice_thickness"][0], dtype=float), np.nan)
        fb = np.ma.filled(np.ma.array(d.variables["radar_freeboard"][0], dtype=float), np.nan)
        lat = np.array(d.variables["lat"][:], dtype=float)
        d.close()
        ok = np.isfinite(th) & (th > 0) & np.isfinite(fb)
        per_month.append(int(ok.sum()))
        lat_s.append(lat[ok])
        th_s.append(th[ok])
    lat = np.concatenate(lat_s)
    th = np.concatenate(th_s)
    sel = (th >= 3.5) & (th <= 4.3)
    bands = {}
    for lo, hi in ((60, 70), (70, 75), (75, 80), (80, 82), (82, 88)):
        b = (lat >= lo) & (lat < hi)
        bands["%d-%d" % (lo, hi)] = {
            "n": int(b.sum()),
            "mean_h_m": float(np.mean(th[b])) if b.sum() else None,
            "p90_h_m": float(np.percentile(th[b], 90)) if b.sum() else None,
        }
    return {
        "n_months": len(files),
        "first_month": months[0],
        "last_month": months[-1],
        "valid_cell_months": int(np.sum(per_month)),
        "valid_cells_mean_per_month": float(np.mean(per_month)),
        "valid_cells_min_per_month": int(np.min(per_month)),
        "valid_cells_max_per_month": int(np.max(per_month)),
        "thick_cells_3p5_4p3_m": int(sel.sum()),
        "thick_cells_poleward_81p45_frac": float(np.mean(lat[sel] > 81.45)),
        "ice_area_poleward_81p45_frac": float(np.mean(lat > 81.45)),
        "lat_bands": bands,
    }, files, months


# ------------------------------------------------- Ku x laser co-location
def atl20_stats(files, cci_files, cci_months):
    atl = sorted(glob.glob(os.path.join(ATL20, "*", "*", "*.h5")))
    amonths = [os.path.basename(os.path.dirname(os.path.dirname(f)))
               + os.path.basename(os.path.dirname(f)) for f in atl]
    if not atl:
        return {"available": False}
    import h5py
    h = h5py.File(atl[0], "r")
    glat = np.array(h["grid_lat"][:], dtype=float).ravel()
    glon = np.array(h["grid_lon"][:], dtype=float).ravel()
    h.close()

    def unit(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)

    tree = cKDTree(unit(glat, glon))
    d0 = netCDF4.Dataset(cci_files[0])
    clat = np.array(d0.variables["lat"][:], dtype=float).ravel()
    clon = np.array(d0.variables["lon"][:], dtype=float).ravel()
    d0.close()
    _, idx = tree.query(unit(clat, clon), k=1)
    idx = idx.reshape(432, 432)

    overlap = sorted(set(cci_months) & set(amonths))
    coloc, per_month = 0, {}
    for m in overlap:
        fc = cci_files[cci_months.index(m)]
        fa = atl[amonths.index(m)]
        d = netCDF4.Dataset(fc)
        th = np.ma.filled(np.ma.array(d.variables["sea_ice_thickness"][0], dtype=float), np.nan)
        fb = np.ma.filled(np.ma.array(d.variables["radar_freeboard"][0], dtype=float), np.nan)
        d.close()
        import h5py
        with h5py.File(fa, "r") as hh:
            afb = np.array(hh["monthly/mean_fb"][:], dtype=float)
            nsg = np.array(hh["monthly/n_segs"][:], dtype=float)
            lsum = np.array(hh["monthly/length_sum"][:], dtype=float)
        ok_a = (afb.ravel() < 1e20) & (nsg.ravel() != 2147483647) & (nsg.ravel() > 0)
        okc = np.isfinite(th) & (th > 0) & np.isfinite(fb)
        ok = okc & ok_a.ravel()[idx]
        n = int(ok.sum())
        coloc += n
        per_month[m] = n
    # Kish effective samples per 25 km cell, rho_1 = 0.3, n_nominal = 10
    neff_per_cell = 10.0 * (1.0 - 0.3) / (1.0 + 0.3)
    return {
        "available": True,
        "n_months": len(atl),
        "first_month": amonths[0],
        "last_month": amonths[-1],
        "overlap_months": overlap,
        "n_overlap_months": len(overlap),
        "colocated_cell_months": coloc,
        "colocated_cells_per_month": float(np.mean(list(per_month.values()))),
        "n_eff_per_cell_rho03": neff_per_cell,
        "n_eff_total": float(coloc * neff_per_cell),
        "per_month": per_month,
    }


# ------------------------------------------------------- slope arithmetic
def eta_se(n_eff, sigma_diff_m, delta_rho="30"):
    return (sigma_diff_m / (float(delta_rho) * np.sqrt(n_eff))) * 1000.0  # mm/(kg m-3)


def _archived_thresholds():
    """The 10 %/50 % edges, read from out/p5_window.csv (never hardcoded).

    An earlier version of this script carried its own copy of the numbers,
    which is how a superseded threshold survived a revision.  The edges now
    come from the same file the manuscript quotes.
    """
    import csv
    path = os.path.join(OUT, "p5_window.csv")
    if not os.path.exists(path):
        raise SystemExit("run p5_numbers.py first: %s missing" % path)
    rows = {(r["state"], r["config"]): r for r in csv.DictReader(open(path))}
    out = {}
    for state in ("low", "high"):
        r = rows[(state, "ku_ka_p50")]
        out["10pct_" + state] = float(r["rho_i_10_lo"])
        out["50pct_" + state] = float(r["rho_i_50_lo"])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    inv, ka_hits = inventory()
    cci, cci_files, cci_months = cci_stats()
    atl = atl20_stats(cci_files, cci_files, cci_months)

    thresholds = _archived_thresholds()
    regimes = []
    for sd in (0.06, 0.15, 0.40):
        se = eta_se(atl["n_eff_total"], sd) if atl.get("available") else float("nan")
        regimes.append({"sigma_diff_m": sd, "SE_eta_mm_per_kgm3": se,
                        "MDE_2p8016sigma_mm_per_kgm3": 2.8016 * se,
                        "threshold_over_MDE_low10pct": (thresholds["10pct_low"] / (2.8016 * se)) if se else None,
                        "threshold_over_MDE_high50pct": (thresholds["50pct_high"] / (2.8016 * se)) if se else None})

    verdict = {
        "ka_band_files_on_machine": len(ka_hits),
        "ka_pair_possible": len(ka_hits) > 0,
        "only_local_pair": "CryoSat-2 Ku x ICESat-2 ATL20 laser (not Ka)",
        "altika_inclination_deg": 98.55,
        "altika_max_lat_deg": 81.45,
        "thick_state_cells_inaccessible_to_altika_frac":
            cci["thick_cells_poleward_81p45_frac"],
        "density_truth_available": False,
        "verdict": ("no empirical Ka-Ku anchor is constructible from local holdings; "
                    "route = GMD methods for assessment of models"),
    }

    payload = {"inventory": inv, "ka_hits": ka_hits, "cci": cci,
               "atl20": atl, "thresholds_mm_per_kgm3": thresholds,
               "eta_resolution": regimes, "verdict": verdict}
    with open(os.path.join(OUT, "feasibility_anchor.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)

    rows = [("ka_band_files_on_machine", len(ka_hits), "files"),
            ("cci_months", cci["n_months"], "months"),
            ("cci_valid_cell_months", cci["valid_cell_months"], "cell-months"),
            ("atl20_months", atl["n_months"], "months"),
            ("overlap_months", atl["n_overlap_months"], "months"),
            ("colocated_cell_months", atl["colocated_cell_months"], "cell-months"),
            ("n_eff_total_rho03", round(atl["n_eff_total"], 1), "samples"),
            ("thick_cells_poleward_81p45_frac", round(cci["thick_cells_poleward_81p45_frac"], 4), "fraction")]
    for r in regimes:
        rows.append(("SE_eta_sigma_diff_%gm" % r["sigma_diff_m"], round(r["SE_eta_mm_per_kgm3"], 5),
                     "mm/(kg m-3)"))
    with open(os.path.join(OUT, "feasibility_anchor.csv"), "w", encoding="utf-8") as f:
        f.write("quantity,value,unit\n")
        for q, v, u in rows:
            f.write("%s,%s,%s\n" % (q, v, u))

    print(json.dumps({"verdict": verdict, "eta_resolution": regimes,
                      "colocated": atl.get("colocated_cell_months")}, indent=1))


if __name__ == "__main__":
    main()
