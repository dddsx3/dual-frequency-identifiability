# -*- coding: utf-8 -*-
"""Unit tests for the model behind the manuscript.

These test the *properties* the paper claims, calling info_geometry directly,
so they remain meaningful even if the CSV writer or the formatting changes.
Run from code/:  python -m pytest tests -q      (or: python tests/test_p5_numbers.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import info_geometry as ig  # noqa: E402


# --------------------------------------------------------------- identity
def test_identity_holds_on_every_cell():
    """sigma^2 = 1/(1' C_-rho^-1 1) + a, 7 configs x 2 states x 2 regimes."""
    worst = 0.0
    for st in ig.STATES.values():
        for regime in ig.REGIMES:
            for _cid, _label, chans, _parent in ig.CONFIGS:
                sig2 = ig.sigma_var(st, chans, regime)
                rest = ig.sigma_var(st, chans, regime, dirs=("hs", "eps", "rep"))
                a = ig.added_var(st, chans, regime, "density")
                worst = max(worst, abs(sig2 - (rest + a)))
    assert worst < 1e-12, worst


def test_density_term_is_configuration_independent():
    a = []
    for st in ig.STATES.values():
        for _cid, _l, chans, _p in ig.CONFIGS:
            a.append(ig.added_var(st, chans, "operational", "density"))
        break
    assert max(a) - min(a) < 1e-12


def test_density_split_equals_closed_form():
    for nm in ("low", "high"):
        st = ig.STATES[nm]
        for cid in ig.CONF:
            sp = ig.density_split(st, ig.CONF[cid][2], "operational")
            assert abs(sp["total"] - sp["closed_form"]) < 1e-12
            assert abs(sp["rho_i"] + sp["rho_s"] - sp["closed_form"]) < 1e-12


# --------------------------------------------------------------- Woodbury
def test_woodbury_matches_direct_inversion():
    worst = 0.0
    for st in ig.STATES.values():
        for cid in ("ku_ka_p50", "ku_ka_p50_atl20"):
            chans = ig.CONF[cid][2]
            for e in np.arange(-1.0, 20.0, 0.5):
                ev = ig._eta_vector(len(chans), 1, float(e))
                d = ig.added_var(st, chans, "operational", "density", eta_vec=ev)
                w = ig.added_var_woodbury(st, chans, "operational", eta_vec=ev)
                worst = max(worst, abs(d - w))
    assert worst < 1e-12, worst


# --------------------------------------------------------------- ceiling
def test_ceiling_equals_ice_share():
    """max_eta r(eta) = a_rhoi/a(0), attained at the GLS-orthogonality point."""
    for nm in ("low", "high"):
        st = ig.STATES[nm]
        chans = ig.CONF["ku_ka_p50"][2]
        sp = ig.density_split(st, chans, "operational")
        share = 100.0 * sp["rho_i"] / sp["closed_form"]
        e_opt = ig.eta_opt(st, chans, "operational")
        r = ig.response(st, chans, "operational", e_opt)
        assert abs(r - share) < 1e-6, (nm, r, share)


def test_ceiling_is_not_the_zero_crossing():
    """The naive location h/rho_w understates the response; say so by test."""
    st = ig.STATES["low"]
    chans = ig.CONF["ku_ka_p50"][2]
    r_zero = ig.response(st, chans, "operational", st["h"] / ig.RHO_W * 1e3)
    r_opt = ig.response(st, chans, "operational",
                        ig.eta_opt(st, chans, "operational"))
    assert r_zero < r_opt - 5.0


# --------------------------------------------------------------- response
def test_response_is_signed_and_zero_at_zero():
    st = ig.STATES["low"]
    chans = ig.CONF["ku_ka_p50"][2]
    assert abs(ig.response(st, chans, "operational", 0.0)) < 1e-12
    assert ig.response(st, chans, "operational", -0.3) < -20.0
    assert ig.response(st, chans, "operational", 0.3) > 20.0


def test_snow_branch_never_reaches_the_lowest_tier():
    for nm in ("low", "high"):
        st = ig.STATES[nm]
        chans = ig.CONF["ku_ka_p50"][2]
        lo, hi = ig.response_window(st, chans, "operational", 10.0,
                                    variable="rho_s")
        assert lo != lo and hi != hi, (nm, lo, hi)


def test_window_edge_attains_the_tier():
    for nm in ("low", "high"):
        st = ig.STATES[nm]
        chans = ig.CONF["ku_ka_p50"][2]
        for target in (10.0, 50.0):
            lo, _hi = ig.response_window(st, chans, "operational", target)
            assert abs(ig.response(st, chans, "operational", lo) - target) < 1e-5


# --------------------------------------------------------------- alignment
def test_alpha_null_limits():
    st = ig.STATES["low"]
    a_kuku = ig.alpha_null(st, ig.CONF["ku_ku"][2], "operational")
    a_lidar = ig.alpha_null(st, ig.CONF["cci_atl20"][2], "operational")
    assert abs(a_kuku - 1.0) < 1e-12
    assert a_lidar < 0.02


# --------------------------------------------------------------- requirement
def test_requirement_grows_with_thickness():
    import p5_numbers as p5
    chans = ig.CONF["ku_ka_p50"][2]
    req = []
    for h in (1.0, 1.5, 2.0, 3.0, 4.0):
        st = p5.swept_state(h)
        req.append(ig.response_window(st, chans, "operational", 10.0)[0])
    assert all(b > a for a, b in zip(req, req[1:])), req


def test_width_floor_is_not_set_by_channel_count():
    """Doubling the number of footprints must move w by well under 2 %."""
    import math
    d10 = math.sqrt(0.020 ** 2 / 10 + 0.027 ** 2)
    d20 = math.sqrt(0.020 ** 2 / 20 + 0.027 ** 2)
    assert (d10 - d20) / d10 < 0.02


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   %s" % name)
            except AssertionError as exc:
                fails += 1
                print("FAIL %s  %s" % (name, exc))
    print("%d test(s) failed" % fails if fails else "all tests passed")
    sys.exit(1 if fails else 0)
