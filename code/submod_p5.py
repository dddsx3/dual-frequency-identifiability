# -*- coding: utf-8 -*-
"""GLS 口径下的子模性重检（对应 n10 的同一批 3–4 区小规模实例）。

目标：  I(S) = 1ᵀ R_S⁻¹ 1        （GLS 信息量，正确口径）
对照：  n_eff(S) = n²/1ᵀR_S1     （Kish 设计效应，旧口径，保留作对照）

判据（A ⊆ B 均为可行集，x ∉ B）：
    单调   Δ_A(x) ≥ 0
    子模   Δ_A(x) ≥ Δ_B(x)   （边际递减）
    超模   Δ_A(x) ≤ Δ_B(x)   （边际递增）

输出（data/）：gls_submod.csv / gls_witnesses.csv / gls_three_sortie.csv / gls_greedy_gap.csv
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

import gls_core as gc

N9 = Path(__file__).resolve().parent
DATA = N9.parent / "out"
DATA.mkdir(exist_ok=True)

ANCHOR_D = gc.ANCHOR_D
ANCHOR_R = gc.ANCHOR_R
_PCHIP = PchipInterpolator(ANCHOR_D, ANCHOR_R)
_EXP_L = -50.0 / np.log(ANCHOR_R[2] / ANCHOR_R[1])
_EXP_S = ANCHOR_R[1] * np.exp(25.0 / _EXP_L)

GEOMETRIES = [
    ("GC1_3reg_12.5in_50btw", [0.0, 50.0, 100.0], [0.0, 12.5, 25.0]),
    ("GC2_3reg_25in_75btw", [0.0, 75.0, 150.0], [0.0, 25.0, 50.0]),
    ("GC3_4reg_10in_25btw", [0.0, 25.0, 50.0, 75.0], [0.0, 10.0, 20.0]),
    ("GC4_4reg_12.5in_50btw", [0.0, 50.0, 100.0, 150.0], [0.0, 12.5, 25.0]),
    ("GC5_3reg_5in_25btw", [0.0, 25.0, 50.0], [0.0, 5.0, 10.0]),
]


def rho(d, kind):
    d = abs(float(d))
    if d == 0.0:
        return 1.0
    if kind == "empirical":
        return float(np.interp(d, ANCHOR_D, ANCHOR_R, left=1.0, right=0.0))
    if kind == "pchip":
        return float(_PCHIP(min(d, 150.0))) if d < 150.0 else 0.0
    if kind == "expnugget":
        return float(_EXP_S * np.exp(-d / _EXP_L))
    if kind == "exp326":
        return float(np.exp(-d / 326.0))      # 不截断：截断会破坏正定性
    raise ValueError(kind)


def build_geometry(centers, offsets, kind):
    elems = [(r, s, c + o) for r, c in enumerate(centers) for s, o in enumerate(offsets)]
    pos = np.array([e[2] for e in elems])
    m = len(elems)
    R = np.array([[rho(pos[i] - pos[j], kind) for j in range(m)] for i in range(m)])
    return elems, pos, R + 1e-10 * np.eye(m)


def feasible_masks(m, regions, domain):
    cap = {"all": 10 ** 9, "two_per_region": 2, "one_per_region": 1}[domain]
    feas = np.zeros(1 << m, bool)
    for mask in range(1 << m):
        ok = True
        for r in set(regions):
            cnt = sum(1 for i in range(m) if regions[i] == r and (mask >> i) & 1)
            if cnt > cap:
                ok = False
                break
        feas[mask] = ok
    return feas


def values_all(R, m, feas):
    I = np.full(1 << m, np.nan)
    NE = np.full(1 << m, np.nan)
    I[0] = 0.0
    for mask in range(1, 1 << m):
        if not feas[mask]:
            continue
        S = [i for i in range(m) if (mask >> i) & 1]
        sub = R[np.ix_(S, S)]
        I[mask] = gc.gls_info(sub)
        NE[mask] = gc.kish_neff(sub)
    return I, NE


def superset_pairs(feas, m):
    """所有 (A,B) 对，A ⊆ B、B 可行、且 A 可行。"""
    pa, pb = [], []
    idx = np.flatnonzero(feas)
    for a in idx:
        for b in idx:
            if a & b == a:
                pa.append(a)
                pb.append(b)
    return np.array(pa, int), np.array(pb, int)


def analyse(R, regions, domain):
    m = len(R)
    feas = feasible_masks(m, regions, domain)
    I, NE = values_all(R, m, feas)
    D = np.full((1 << m, m), np.nan)
    for x in range(m):
        tgt = np.arange(1 << m) | (1 << x)
        D[:, x] = I[tgt] - I
    pa, pb = superset_pairs(feas, m)

    sub_bad = sup_bad = 0
    worst_sub, worst_sup = (-np.inf, None), (-np.inf, None)
    for x in range(m):
        notx = ~((pb >> x) & 1)          # x ∉ B
        da, db = D[pa, x], D[pb, x]
        ok = notx & np.isfinite(da) & np.isfinite(db) & (pa != pb)
        if not ok.any():
            continue
        da, db = da[ok], db[ok]
        # da >= db ：边际递减（子模行为）；da < db ：子模违例
        gap_sub = db - da
        gap_sup = da - db                # >0 ⇒ 边际递增（超模行为）
        n_sub = int(np.sum(gap_sub > 1e-10))
        n_sup = int(np.sum(gap_sup > 1e-10))
        sub_bad += n_sub
        sup_bad += n_sup
        if n_sub:
            i = int(np.argmax(gap_sub))
            if gap_sub[i] > worst_sub[0]:
                k = np.flatnonzero(ok)[i]
                worst_sub = (gap_sub[i], (pa[k], pb[k], x))
        if n_sup:
            j = int(np.argmax(gap_sup))
            if gap_sup[j] > worst_sup[0]:
                k = np.flatnonzero(ok)[j]
                worst_sup = (gap_sup[j], (pa[k], pb[k], x))

    mono = int(np.sum(np.nanmin(np.where(np.isnan(D), np.inf, D), axis=1) < -1e-12))
    verdict = ("modular" if sub_bad == 0 and sup_bad == 0
               else "submodular" if sub_bad == 0
               else "supermodular" if sup_bad == 0
               else "neither")
    return dict(mono=mono, sub=sub_bad, sup=sup_bad, verdict=verdict,
                worst_sub=worst_sub, worst_sup=worst_sup, I=I, NE=NE, feas=feas)


def greedy_I(R, m, feas, k, I_all):
    chosen = []
    for _ in range(k):
        best, bk = -np.inf, None
        for x in range(m):
            if x in chosen:
                continue
            tgt = sum(1 << i for i in chosen + [x])
            if not feas[tgt]:
                continue
            v = I_all[tgt]
            if v > best:
                best, bk = v, x
        if bk is None:
            break
        chosen.append(bk)
    return chosen, (I_all[sum(1 << i for i in chosen)] if chosen else 0.0)


def best_I(feas, k, I_all):
    cand = [mask for mask in range(1, len(I_all))
            if feas[mask] and bin(mask).count("1") <= k]
    if not cand:
        return None, 0.0
    vals = np.array([I_all[c] for c in cand])
    i = int(np.argmax(vals))
    return cand[i], float(vals[i])


def describe(mask, elems):
    return "{" + ",".join(f"{elems[i][0]}.{elems[i][1]}"
                          for i in range(len(elems)) if (mask >> i) & 1) + "}"


def main():
    geo = {n: (c, o) for n, c, o in GEOMETRIES}
    cases = [(k, n) for k in ("empirical", "pchip", "expnugget", "exp326")
             for n, _, _ in GEOMETRIES]
    rows, wrows, grows = [], [], []
    for kind, name in cases:
        centers, offsets = geo[name]
        elems, pos, R = build_geometry(centers, offsets, kind)
        regions = [e[0] for e in elems]
        m = len(elems)
        for domain in ("all", "two_per_region", "one_per_region"):
            res = analyse(R, regions, domain)
            rows.append(dict(
                geometry=name, kernel=kind, domain=domain,
                n_elements=m, n_feasible=int(res["feas"].sum()),
                mono_violations=res["mono"],
                submod_violations=res["sub"], supermod_violations=res["sup"],
                verdict=res["verdict"],
                max_sub_gap=("" if res["worst_sub"][1] is None
                             else f"{res['worst_sub'][0]:.6e}"),
                max_sup_gap=("" if res["worst_sup"][1] is None
                             else f"{res['worst_sup'][0]:.6e}"),
            ))
            for key, typ in (("worst_sub", "submodularity"), ("worst_sup", "supermodularity")):
                val, hit = res[key]
                if hit is None or val <= 1e-10:
                    continue
                ma, mb, x = hit
                wrows.append(dict(
                    geometry=name, kernel=kind, domain=domain, type=typ,
                    A=describe(ma, elems), B=describe(mb, elems), x=describe(1 << x, elems),
                    I_A=f"{res['I'][ma]:.6f}", I_B=f"{res['I'][mb]:.6f}",
                    I_Ax=f"{res['I'][ma | (1 << x)]:.6f}", I_Bx=f"{res['I'][mb | (1 << x)]:.6f}",
                    delta_A=f"{res['I'][ma | (1 << x)] - res['I'][ma]:.6f}",
                    delta_B=f"{res['I'][mb | (1 << x)] - res['I'][mb]:.6f}",
                    gap=f"{val:.6e}",
                    positions=f"A:{list(pos[[i for i in range(m) if (ma >> i) & 1]])} "
                              f"B:{list(pos[[i for i in range(m) if (mb >> i) & 1]])} x:{pos[x]}"))

        feas = feasible_masks(m, regions, "all")
        I_all, _ = values_all(R, m, feas)
        for k in range(1, m + 1):
            bm, vbest = best_I(feas, k, I_all)
            ch, vg = greedy_I(R, m, feas, k, I_all)
            grows.append(dict(geometry=name, kernel=kind, k=k, m=m, greedy_I=vg, optimal_I=vbest,
                              ratio=vg / vbest if vbest > 0 else np.nan,
                              mde_penalty=(np.sqrt(vbest / vg) if vg > 0 else np.nan),
                              greedy_set=describe(sum(1 << i for i in ch), elems),
                              optimal_set=describe(bm, elems) if bm is not None else ""))

    pd.DataFrame(rows).to_csv(DATA / "gls_submod.csv", index=False, float_format="%.6g")
    pd.DataFrame(wrows).to_csv(DATA / "gls_witnesses.csv", index=False, float_format="%.6g")
    pd.DataFrame(grows).to_csv(DATA / "gls_greedy_gap.csv", index=False, float_format="%.6g")

    # ---- 三架次最小反例：同一实例上 I 单调、n_eff 不单调 ----
    _, pos3, R3 = build_geometry([0.0], [0.0, 12.5, 25.0], "empirical")
    e3 = [(0, 0, 0.0), (0, 1, 12.5), (0, 2, 25.0)]
    t3 = []
    for mask in range(8):
        S = [i for i in range(3) if (mask >> i) & 1]
        n = len(S)
        sub = R3[np.ix_(S, S)] if n else None
        t3.append(dict(subset=describe(mask, e3), positions=";".join(str(pos3[i]) for i in S),
                       n=n,
                       sum_rho=float(sub.sum()) if n else 0.0,
                       n_eff=gc.kish_neff(sub) if n else 0.0,
                       I_GLS=gc.gls_info(sub) if n else 0.0,
                       Var_mean=(1.0 / gc.gls_info(sub)) if n else 1.0))
    pd.DataFrame(t3).to_csv(DATA / "gls_three_sortie.csv", index=False, float_format="%.6f")

    # ---- Fig. 4(a) witness: identical information, falling Kish n_eff ----
    # The third sortie at 12.5 km is redundant in the GLS metric (its marginal
    # Schur gain is zero to machine precision) while the design effect still
    # counts it, so n_eff falls.  This table is the archive behind Fig. 4(a).
    S2 = [0, 2]
    R2 = R3[np.ix_(S2, S2)]
    r_x = R3[np.ix_([1], S2)].ravel()
    wit = [("I_GLS_two_sorties", gc.gls_info(R2)),
           ("I_GLS_three_sorties", gc.gls_info(R3)),
           ("n_eff_two_sorties", gc.kish_neff(R2)),
           ("n_eff_three_sorties", gc.kish_neff(R3)),
           ("marginal_gain_delta", gc.delta_x(R2, r_x, float(R3[1, 1])))]
    pd.DataFrame([dict(metric=k, value="%.10g" % v) for k, v in wit]).to_csv(
        DATA / "p5_witness.csv", index=False)

    d = pd.DataFrame(rows)
    print(d.groupby(["kernel", "verdict"]).size().to_string())
    print()
    print(d[d.kernel == "empirical"][["geometry", "domain", "mono_violations",
                                      "submod_violations", "supermod_violations", "verdict"]]
          .to_string(index=False))
    g = pd.DataFrame(grows)
    print()
    print(g.groupby(["kernel", "geometry"])["ratio"].agg(["min", "mean"]).to_string())
    print("\nworst greedy/optimal ratio = %.4f" % g.ratio.min())
    print("\nthree-sortie:\n", pd.DataFrame(t3).to_string(index=False))


if __name__ == "__main__":
    main()
