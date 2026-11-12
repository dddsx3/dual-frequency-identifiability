# -*- coding: utf-8 -*-
"""Regenerate every number, table and archived artefact of the manuscript.

Run from code/:   python p5_numbers.py [--check]

Writes to ../out:
  p5_state.csv            calibrated states (Sec. 2.1)
  p5_metrics.csv          Table 1: posterior sigma, delta-sigma, MDE, alpha_null, a
  p5_identity.csv         Table 1 check: sigma^2 = 1/(1'C_-rho^-1 1) + a, 28 cells
  p5_density_split.csv    a split into ice- and snow-density linkages
  p5_response.csv         Table 2 and Fig. 3: r(eta) for both linkages, both states
  p5_window.csv           Table 2: the 10 %/50 % windows, plateaux, refraction ceiling
  p5_mission.csv          Table 3: required |eta_i| against thickness
  p5_width_floor.csv      Sec. 9: two-component noise, three-term width bound
  p5_checks.csv           every assertion, with pass/fail

The model lives in info_geometry.py; this file only formats.  Nothing here reads
an external data file, so the run is self-contained and deterministic.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

import info_geometry as ig

OUT = ig.OUT
DIRS = ("density", "snow", "penetration", "representativeness")

# declared per-footprint differential height noise of a same-platform pair
SIG_INSTR = 0.020      # m, Ka band (ASR-class altimeter)
SIG_REP_CORR = 0.027   # m, representativeness, common to both channels
DN_DRHO_S = 8.23e-4    # refractive index per kg m^-3 at rho_s = 320 (Tiuri)

CHECKS = []


def check(name, value, expected, tol, note=""):
    ok = (value == value) and abs(value - expected) <= tol
    CHECKS.append(dict(check=name, value=value, expected=expected, tol=tol,
                       pass_flag=bool(ok), note=note))
    return ok


def write(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (("%.10g" % v) if isinstance(v, float) else v)
                        for k, v in r.items()})


def fnum(x, nd=10):
    return ("%%.%dg" % nd) % x if isinstance(x, float) else x


# ------------------------------------------------------------------ states
def state_rows():
    rows = []
    for nm, st in ig.STATES.items():
        rows.append(dict(state=nm, h=st["h"], hs=st["hs"], sigma_hs=st["sig_hs"],
                         sigma_f_freeboard=st["sig_f"], kappa=ig.KAPPA,
                         deff=ig.DEFF, sigma_rho_i=ig.SIG_RHO_I,
                         sigma_rho_s=ig.SIG_RHO_S,
                         a2_d2=st["d2"]["a2"], b2_d2=st["d2"]["b2"],
                         c2_d2=st["d2"]["c2"], d2_d2=st["d2"]["d2"]))
    write("p5_state.csv", rows)
    return rows


# ------------------------------------------------------------------ Table 1
def metrics_rows():
    rows = []
    for nm in ("low", "high"):
        st = ig.STATES[nm]
        for cid, label, chans, parent in ig.CONFIGS:
            sp, C, R, G, B, rep = ig.sigma_post(st, chans, regime="operational")
            an = ig.alpha_null(st, chans, "operational")
            a = ig.added_var(st, chans, "operational", "density")
            if parent is None:
                d_pct = dmde = float("nan")
            else:
                sp_p, *_ = ig.sigma_post(st, ig.CONF[parent][2], regime="operational")
                d_pct = 100.0 * (sp_p - sp) / sp_p
                dmde = ig.MDE_K * (sp_p - sp)
            rows.append(dict(state=nm, config=cid, label=label, n=len(chans),
                             p=ig.CONF[cid][2][0].get("p") if len(chans) > 1 else
                             (0.0 if cid == "ku_ku" else ""),
                             sigma_m=sp, delta_sigma_pct=d_pct, delta_MDE_m=dmde,
                             alpha_null=an, a_density_m2=a))
    write("p5_metrics.csv", rows)
    return rows


# ------------------------------------------------------------------ identity
def identity_rows():
    """sigma^2 = 1/(1' C_-rho^-1 1) + a, on every configuration and regime."""
    rows = []
    worst = 0.0
    for nm, st in ig.STATES.items():
        for regime in ig.REGIMES:
            for cid, label, chans, parent in ig.CONFIGS:
                sig2 = ig.sigma_var(st, chans, regime)
                without = ig.sigma_var(st, chans, regime, dirs=("hs", "eps", "rep"))
                a = ig.added_var(st, chans, regime, "density")
                resid = sig2 - (without + a)
                worst = max(worst, abs(resid))
                rows.append(dict(state=nm, regime=regime, config=cid,
                                 sigma2_m2=sig2, inv_I_minus_rho_m2=without,
                                 a_m2=a, residual_m2=resid,
                                 pass_flag=abs(resid) < 1e-12))
    check("identity_sigma2_equals_invI_plus_a_all_28_cells", worst, 0.0, 1e-12,
          "max |residual| over 7 configurations x 2 states x 2 regimes")
    write("p5_identity.csv", rows)
    return rows, worst


def split_rows():
    rows = []
    for nm in ("low", "high"):
        st = ig.STATES[nm]
        for cid in ("cci", "cci_atl20", "ku_ka_p50", "ku_ka_p50_atl20"):
            sp = ig.density_split(st, ig.CONF[cid][2], "operational")
            rows.append(dict(state=nm, config=cid, a_total_m2=sp["total"],
                             a_closed_form_m2=sp["closed_form"],
                             a_rho_i_m2=sp["rho_i"], a_rho_s_m2=sp["rho_s"],
                             share_rho_i=sp["rho_i"] / sp["closed_form"],
                             share_rho_s=sp["rho_s"] / sp["closed_form"],
                             marginal_rho_i_m2=sp["marginal_rho_i"],
                             marginal_rho_s_m2=sp["marginal_rho_s"]))
    write("p5_density_split.csv", rows)
    for r in rows:
        if r["config"] == "ku_ka_p50":
            check("density_split_sums_to_total_%s" % r["state"],
                  r["a_rho_i_m2"] + r["a_rho_s_m2"], r["a_total_m2"], 1e-9)
    return rows


# ------------------------------------------------------------------ response
def response_rows():
    rows = []
    for nm in ("low", "high"):
        st = ig.STATES[nm]
        for cid in ("ku_ka_p50", "ku_ka_p50_atl20"):
            chans = ig.CONF[cid][2]
            for e in np.arange(-3.0, 12.0001, 0.05):
                rows.append(dict(state=nm, config=cid, eta_mm_per_kgm3=float(e),
                                 r_rho_i_pct=ig.response(st, chans, "operational", float(e),
                                                         variable="rho_i"),
                                 r_rho_s_pct=ig.response(st, chans, "operational", float(e),
                                                         variable="rho_s")))
    write("p5_response.csv", rows)
    return rows


def window_rows():
    rows = []
    for nm in ("low", "high"):
        st = ig.STATES[nm]
        sp = ig.density_split(st, ig.CONF["ku_ka_p50"][2], "operational")
        ceiling_i = 100.0 * sp["rho_i"] / sp["closed_form"]   # best r on the rho_i branch
        ceiling_s = 100.0 * sp["rho_s"] / sp["closed_form"]   # best r on the rho_s branch
        eta_max = st["hs"] * DN_DRHO_S * 1000.0
        for cid in ("ku_ka_p50", "ku_ka_p50_atl20"):
            chans = ig.CONF[cid][2]
            grid = np.arange(-3.0, 40.0001, 0.05)
            rgrid = np.array([ig.response(st, chans, "operational", float(e))
                              for e in grid])
            row = dict(state=nm, config=cid, a0_m2=sp["total"],
                       plateau_rho_i_pct=ceiling_i, plateau_rho_s_pct=ceiling_s,
                       eta_opt_mm_per_kgm3=float(grid[int(np.argmax(rgrid))]),
                       r_max_pct=float(rgrid.max()),
                       eta_max_refraction_mm_per_kgm3=eta_max)
            for target in (10.0, 50.0):
                lo_i, hi_i = ig.response_window(st, chans, "operational", target,
                                                variable="rho_i", lo=-1.0, hi=40.0)
                lo_s, hi_s = ig.response_window(st, chans, "operational", target,
                                                variable="rho_s", lo=-1.0, hi=40.0)
                row["rho_i_%g_lo" % target] = lo_i
                row["rho_i_%g_hi" % target] = hi_i
                row["rho_s_%g_reachable" % target] = bool(hi_s == hi_s)
            rows.append(row)
    write("p5_window.csv", rows)

    # the 10 % window must exist on the rho_i branch and be empty on the rho_s one
    for r in rows:
        if r["config"] == "ku_ka_p50":
            check("rho_i_10pct_window_exists_%s" % r["state"],
                  1.0 if r["rho_i_10_hi"] == r["rho_i_10_hi"] else 0.0, 1.0, 0.0)
            check("rho_s_10pct_unreachable_at_any_magnitude_%s" % r["state"],
                  0.0 if r["rho_s_10_reachable"] else 1.0, 1.0, 0.0)
            check("rho_s_ceiling_below_10pct_%s" % r["state"],
                  1.0 if r["plateau_rho_s_pct"] < 10.0 else 0.0, 1.0, 0.0)
    return rows


# ------------------------------------------------------------------ Appendix A
def appendix_rows():
    """Numbers of Appendix A: Woodbury vs direct inversion, and the ceiling.

    Three claims of Appendix A are asserted here, because each of them is a
    derivation that could silently stop being true if the model were edited:
      A.4  the two-by-two Woodbury form reproduces the direct inversion
      A.5  max_eta r(eta) equals the ice-density share exactly
      A.5  the maximiser is the GLS-orthogonality point, not h/rho_w
    """
    rows, worst_wb, worst_opt = [], 0.0, 0.0
    grid = np.arange(-1.0, 40.0001, 0.05)      # response search range
    coarse = np.arange(-1.0, 40.0001, 0.50)    # Woodbury cross-check range
    for nm in ("low", "high"):
        st = ig.STATES[nm]
        sp = ig.density_split(st, ig.CONF["ku_ka_p50"][2], "operational")
        share = 100.0 * sp["rho_i"] / sp["closed_form"]
        for cid in ("ku_ka_p50", "ku_ka_p50_atl20"):
            chans = ig.CONF[cid][2]
            # A.4: Woodbury against direct inversion
            wb = 0.0
            for e in coarse:
                ev = ig._eta_vector(len(chans), 1, float(e))
                wb = max(wb, abs(ig.added_var_woodbury(st, chans, "operational", eta_vec=ev)
                                 - ig.added_var(st, chans, "operational", "density", eta_vec=ev)))
            r = np.array([ig.response(st, chans, "operational", float(e)) for e in grid])
            e_grid = float(grid[int(np.argmax(r))])
            e_closed = ig.eta_opt(st, chans, "operational")
            rows.append(dict(state=nm, config=cid, share_rho_i_pct=share,
                             r_max_pct=float(r.max()), eta_opt_grid=e_grid,
                             eta_opt_closed_mm=e_closed,
                             eta_zero_cross_mm=st["h"] / ig.RHO_W * 1e3,
                             r_at_zero_cross_pct=ig.response(
                                 st, chans, "operational", st["h"] / ig.RHO_W * 1e3),
                             woodbury_vs_direct_max_abs_m2=wb))
            worst_wb = max(worst_wb, wb)
            worst_opt = max(worst_opt, abs(e_closed - e_grid))
            check("woodbury_matches_direct_%s_%s" % (nm, cid), wb, 0.0, 1e-12)
            check("ceiling_equals_ice_share_%s_%s" % (nm, cid),
                  float(r.max()), share, 1e-3)
            check("eta_opt_matches_grid_argmax_%s_%s" % (nm, cid),
                  e_closed, e_grid, 0.05)
    write("p5_appendix.csv", rows)
    return rows, worst_wb, worst_opt


# ------------------------------------------------------------------ Fig. 2
def direction_rows():
    """Marginal cost of each nuisance group, relative to the Ku-only cell.

    The quantity plotted in Fig. 2 is (1 - a_d(config)/a_d(cci)), i.e. the
    reduction of a direction's induced variance that a configuration buys.  It
    is zero for the density group by construction and stays zero for every
    configuration: that is the content of Eq. (identity), shown as a table.
    """
    rows = []
    for nm in ("low", "high"):
        st = ig.STATES[nm]
        base = {g: ig.added_var(st, ig.CONF["cci"][2], "operational", g)
                for g in ("density", "snow", "penetration")}
        for cid, label, chans, parent in ig.CONFIGS:
            for g in ("density", "snow", "penetration", "representativeness"):
                a = ig.added_var(st, chans, "operational", g)
                b = base.get(g, 0.0)
                rows.append(dict(config=cid, state=nm, regime="operational",
                                 direction=g, added_var_m2=a,
                                 baseline_cci_added_var_m2=b,
                                 reduction_pct=(0.0 if b == 0 else 100.0 * (1.0 - a / b))))
    write("p5_directions.csv", rows)
    for r in rows:
        if r["direction"] == "density":
            check("density_marginal_is_configuration_independent_%s_%s"
                  % (r["state"], r["config"]), r["reduction_pct"], 0.0, 1e-9)
    return rows



# ------------------------------------------------------------------ thresholds
def threshold_rows():
    """One row per tier: the window edges and the ceiling, for direct citation.

    The edges also appear inside p5_window.csv, but a reader looking for "the
    threshold" should not have to reconstruct it from a wide table.  This file is
    the four numbers the paper quotes plus the ceiling that bounds them.
    """
    rows = []
    for nm in ("low", "high"):
        st = ig.STATES[nm]
        for cid in ("ku_ka_p50", "ku_ka_p50_atl20"):
            chans = ig.CONF[cid][2]
            sp = ig.density_split(st, chans, "operational")
            for tier in (10.0, 50.0):
                lo, hi = ig.response_window(st, chans, "operational", tier,
                                            variable="rho_i")
                rows.append(dict(state=nm, config=cid, tier_pct=tier,
                                 eta_lower_mm_per_kgm3=lo, eta_upper_mm_per_kgm3=hi,
                                 ceiling_pct=100.0 * sp["rho_i"] / sp["closed_form"],
                                 eta_at_ceiling_mm_per_kgm3=ig.eta_opt(st, chans, "operational")))
    write("p5_thresholds.csv", rows)
    return rows


# ------------------------------------------------------------------ Table 3
def swept_state(h):
    """A state at thickness h: snow depth, its variability and the operational
    freeboard budget interpolated linearly in h between the two calibrated
    states; same density priors."""
    lo, hi = ig.STATES["low"], ig.STATES["high"]
    w = (h - lo["h"]) / (hi["h"] - lo["h"])
    hs = lo["hs"] + w * (hi["hs"] - lo["hs"])
    return dict(name="h=%.2f" % h, h=h, hs=hs,
                sig_hs=lo["sig_hs"] + w * (hi["sig_hs"] - lo["sig_hs"]),
                sig_f=lo["sig_f"] + w * (hi["sig_f"] - lo["sig_f"]), d2={})


def mission_rows():
    rows = []
    chans = ig.CONF["ku_ka_p50"][2]
    for h in (1.0, 1.5, 2.0, 3.0, 4.0):
        st = swept_state(h)
        eta_max = st["hs"] * DN_DRHO_S * 1000.0
        row = dict(h_m=h, hs_m=st["hs"], eta_max_refraction_mm_per_kgm3=eta_max)
        for target in (10.0, 50.0):
            lo, hi = ig.response_window(st, chans, "operational", target,
                                        variable="rho_i", lo=-1.0, hi=60.0)
            edge = min(abs(lo), abs(hi)) if hi == hi else float("nan")
            row["eta_%g_abs" % target] = edge
            row["disp_%g_mm_per_30" % target] = edge * 30.0
        lo, hi = ig.response_window(st, chans, "ideal", 10.0,
                                    variable="rho_i", lo=-1.0, hi=60.0)
        row["eta_10_abs_ideal"] = min(abs(lo), abs(hi)) if hi == hi else float("nan")
        rows.append(row)
    write("p5_mission.csv", rows)
    # monotonicity: the requirement must grow with thickness
    req = [r["eta_10_abs"] for r in rows]
    check("requirement_monotone_in_thickness", float(np.all(np.diff(req) > 0)), 1.0, 0.0,
          "10 % requirement at h = 1.0, 1.5, 2.0, 3.0, 4.0 m")
    return rows


# ------------------------------------------------------------------ width floor
def width_rows():
    kappa, lam, beta, beta_m = ig.KAPPA, ig.G_HS_FT, ig.BETA, ig.BETA_M
    dkappa = ig.RHO_W / ig.DEFF ** 2
    dlam = dkappa * (1.0 - beta)
    obs_coef = 2.0 * (kappa + math.sqrt(2.0) * lam / (1.0 + beta_m))
    snow_term = 2.0 * lam * 0.020                       # best-case 2 cm snow prior
    dens_term = 2.0 * 2.0 * abs(dkappa * 0.30 - dlam * 0.12)   # 2 kg m^-3 prior
    rows = []
    for n_eff, rho1 in ((20.0, None), (10.0, None), (5.4, 0.3), (3.3, 0.5), (1.8, 0.7)):
        delta_f = math.sqrt(SIG_INSTR ** 2 / n_eff + SIG_REP_CORR ** 2)
        obs = obs_coef * delta_f
        rows.append(dict(n_nominal=(20.0 if n_eff == 20.0 else 10.0), n_eff=n_eff,
                         rho1=("" if rho1 is None else rho1),
                         delta_f_m=delta_f, obs_coef=obs_coef, obs_term_m=obs,
                         snow_term_m=snow_term, density_term_m=dens_term,
                         width_m=obs + snow_term + dens_term,
                         obs_share=obs / (obs + snow_term + dens_term)))
    write("p5_width_floor.csv", rows)
    by = {r["n_eff"]: r for r in rows}
    # a doubled channel count must barely move the observation term
    check("doubling_channels_moves_observation_term_by_less_than_2pct",
          100.0 * (by[20.0]["obs_term_m"] / by[10.0]["obs_term_m"] - 1.0), 0.0, 2.0)
    # the observation term must be dominated by the correlated part, not by 1/sqrt(n)
    spread = abs(by[1.8]["width_m"] - by[10.0]["width_m"])
    check("width_floor_insensitive_to_n_eff", spread / by[10.0]["width_m"], 0.0, 0.10,
          "relative change of w between n_eff = 10 and n_eff = 1.8")
    check("width_floor_at_n_eff_10", by[10.0]["width_m"], 1.178, 0.005)
    check("observation_term_at_n_eff_10", by[10.0]["obs_term_m"], 0.871, 0.005)
    return rows


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any assertion fails")
    args = ap.parse_args(argv)

    state_rows()
    m = metrics_rows()
    ident, worst = identity_rows()
    split_rows()
    response_rows()
    direction_rows()
    w = window_rows()
    threshold_rows()
    appendix_rows()
    mission_rows()
    width_rows()

    def _a(state, regime):
        return [r for r in ident if r["state"] == state and r["regime"] == regime
                and r["config"] == "cci"][0]["a_m2"]

    check("a_low_matches_closed_form", _a("low", "operational"), 0.1646353438, 1e-9)
    check("a_high_matches_closed_form", _a("high", "operational"), 1.0218061740, 1e-9,
          "operational, Ku-only, high-latitude state")

    write("p5_checks.csv", CHECKS)
    json.dump(dict(identity_max_residual=worst,
                   windows=w, checks=[{k: str(v) for k, v in c.items()} for c in CHECKS]),
              open(os.path.join(OUT, "p5_numbers.json"), "w"), indent=1)

    print("sigma (operational, low):")
    for r in m:
        if r["state"] == "low":
            print("  %-18s n=%d sigma=%.4f  %.1f%%  anull=%.4f  a=%.6f"
                  % (r["config"], r["n"], r["sigma_m"],
                     r["delta_sigma_pct"] if r["delta_sigma_pct"] == r["delta_sigma_pct"] else 0.0,
                     r["alpha_null"], r["a_density_m2"]))
    print("\nidentity: max |residual| = %.2e m^2" % worst)
    print("\nwindows (eta_i in mm per kg m^-3, positive eta_rhoi is helpful):")
    for r in w:
        print("  %-5s %-18s 10%%[%.3f,%.3f] 50%%[%.3f,%.3f] plateau rho_i=%.1f%% rho_s=%.1f%%"
              % (r["state"], r["config"], r["rho_i_10_lo"], r["rho_i_10_hi"],
                 r["rho_i_50_lo"], r["rho_i_50_hi"],
                 r["plateau_rho_i_pct"], r["plateau_rho_s_pct"]))
    print("\nwidth floor:")
    for r in open(os.path.join(OUT, "p5_width_floor.csv"), encoding="utf-8").read().splitlines()[1:]:
        print("  " + r)
    bad = [c for c in CHECKS if not c["pass_flag"]]
    print("\nchecks: %d/%d pass" % (len(CHECKS) - len(bad), len(CHECKS)))
    for c in bad:
        print("  FAIL %s  got %s expected %s" % (c["check"], c["value"], c["expected"]))
    if args.check and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
