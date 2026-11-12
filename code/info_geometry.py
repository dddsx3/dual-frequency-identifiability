"""Information geometry of the multi-channel thickness retrieval chain.

Forward model  y = 1 h + G x + n,  x ~ N(0,B), n ~ N(0,R), in
thickness-equivalent units z = kappa y.

Sign convention for the density column (derived in Appendix A of the paper).
The GLS model needs the sensitivity of the *forward* map, the prediction of the
observation from the state, taken at fixed thickness:

    y_k = h / kappa(rho_i) + hs (p_k - beta(rho_s))
    dy_k/drho_i |_h = -h / rho_w
    dz_k/drho_i |_h = kappa * (-h/rho_w) = -h / Delta rho      (negative)

The opposite sign, +h/Delta rho, is obtained by differentiating the *inverse*
map h = kappa y at fixed measurement, which is a different object and is not the
one the likelihood uses.  Because the two are easy to confuse, the paper states
the partial derivative explicitly; the sign fixes which sign of the differential
sensitivity eta is the helpful one (a band that is *less* density-coupled than
Ku, i.e. eta > 0 with this convention).

The density term is defined as a *marginal* removal from the complete model,
    a_d = sigma^2(C_full) - sigma^2(C_full without group d),
which is order-free (no arbitrary base covariance) and, at eta = 0, equal to
the closed form (h/Delta rho)^2 sigma_i^2 + (hs/Delta rho)^2 sigma_s^2.

Constants follow the archived states of the growth decomposition
(kappa = 8.7778, Delta rho = 117, sigma_rhoi = 30, sigma_rhos = 60,
eps_sd = 0.2887).
"""
import csv
import json
import math
import os

import numpy as np

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "out"))
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- constants
RHO_W = 1027.0
RHO_I = 910.0
RHO_S = 320.0
DEFF = RHO_W - RHO_I              # 117
KAPPA = RHO_W / DEFF              # 8.7778
BETA = RHO_S / RHO_W              # 0.31159
BETA_M = 0.233606                # product-internal (sif-rfb)/hs, n1
N0 = 1.3784
SIG_RHO_I = 30.0
SIG_RHO_S = 60.0
SIG_EPS = 0.2887                 # eps_sd in d2
G_EPS_UNIT = KAPPA * (N0 - 1.0)   # 3.3215 per metre of snow
MDE_K = 1.96 + 0.8416            # 2.8016

# sign of the forward-map derivative, dz/drho_i = C_RHO_I_SIGN * h / Delta rho
C_RHO_I_SIGN = -1.0

# observation-level snow coefficients, thickness units (dz/dhs)
G_HS_RFB = -KAPPA * (BETA + BETA_M)   # -4.7859  Ku radar freeboard
G_HS_FT = KAPPA * (1.0 - BETA)        # +6.0428  laser total freeboard
G_HS_SIF = -KAPPA * BETA              # -2.7357  ice freeboard (control)
SNOW_SENS_D2 = KAPPA * (N0 - 1.0) + RHO_S / DEFF   # 6.0565, d2 declared chain

ALL_DIRS = ("rho_i", "rho_s", "hs", "eps", "rep")

# ---------------------------------------------------------------- states
# calibrated from d2 growth_decomposition.csv (mean-square components)
D2_COMPONENTS = {
    "low": dict(a2=0.32944232140979196, b2=0.45625089046154615,
                c2=0.164635343805564, d2=0.03627614035857013),
    "high": dict(a2=0.7349670583917967, b2=0.8829692700690734,
                 c2=1.0218061739932371, d2=0.11677461543965005),
}
D2_SIG2_GROWTH = 1.362162307048337


def calibrate_state(name):
    c = D2_COMPONENTS[name]
    hs = math.sqrt(c["d2"]) / (SIG_EPS * G_EPS_UNIT)
    h = math.sqrt(max(c["c2"] - (hs / DEFF * SIG_RHO_S) ** 2, 0.0)) * DEFF / SIG_RHO_I
    sig_hs = math.sqrt(c["b2"]) / SNOW_SENS_D2
    sig_f = math.sqrt(c["a2"]) / KAPPA
    return dict(name=name, h=h, hs=hs, sig_hs=sig_hs, sig_f=sig_f, d2=c)


STATES = {k: calibrate_state(k) for k in ("low", "high")}

# ---------------------------------------------------------------- channels
def channel(kind, p=0.0):
    if kind == "rfb":
        return dict(kind="rfb", label="Ku rfb", g_hs=G_HS_RFB, g_eps=1.0)
    if kind == "ft":
        return dict(kind="ft", label="Laser f_t", g_hs=G_HS_FT, g_eps=0.0)
    if kind == "sif":
        return dict(kind="sif", label="Ku sif", g_hs=G_HS_SIF, g_eps=1.0)
    if kind == "ka":
        # Ka scattering surface sits p snow-depth units above the Ku one.
        # p=0 identical to Ku; p=1+beta_m = snow top (laser-equivalent).
        return dict(kind="ka", label="Ka rfb p=%.2f" % p,
                    g_hs=G_HS_RFB + KAPPA * p, g_eps=1.0, p=p)
    raise ValueError(kind)


CONFIGS = [
    ("cci", "CCI only (Ku rfb)", [channel("rfb")], None),
    ("cci_atl20", "CCI + ATL20", [channel("rfb"), channel("ft")], "cci"),
    ("ku_ku", "CCI + 2nd Ku (control)", [channel("rfb"), channel("rfb")], "cci"),
    ("ku_ka_p25", "Ku + Ka (p=0.25)", [channel("rfb"), channel("ka", 0.25)], "cci"),
    ("ku_ka_p50", "Ku + Ka (p=0.50)", [channel("rfb"), channel("ka", 0.50)], "cci"),
    ("ku_ka_p75", "Ku + Ka (p=0.75)", [channel("rfb"), channel("ka", 0.75)], "cci"),
    ("ku_ka_p50_atl20", "Ku + Ka(p=.5) + lidar",
     [channel("rfb"), channel("ka", 0.50), channel("ft")], "ku_ka_p50"),
]
CONF = {c[0]: c for c in CONFIGS}
REGIMES = ("ideal", "operational")
SIG_DECLARED = {"rfb": 0.0120, "ft": 0.0200, "ka": 0.0200, "sif": 0.0120}
SIG_REP = 0.027   # m, ATL20 representativeness lower bound (d2)


def noise(state, chans, regime):
    if regime == "ideal":
        sig = np.array([SIG_DECLARED[c["kind"]] for c in chans]) * KAPPA
        rep = 0.0
    else:
        f = state["sig_f"]
        rep = SIG_REP
        sig = np.array([math.sqrt(max(f * f - rep * rep, 0.0)) for _ in chans]) * KAPPA
    return np.diag(sig ** 2), rep


# ---------------------------------------------------------------- design
def design(state, chans, regime, dirs=ALL_DIRS, eta_vec=None, eta_s_vec=None):
    """R_meas, G (n x p), B (p x p), rep (m)."""
    n = len(chans)
    R, rep = noise(state, chans, regime)
    cols, diag = [], []
    if "rho_i" in dirs:
        # dz/drho_i = -h / Delta rho, common to every channel (Appendix A)
        g = np.full(n, C_RHO_I_SIGN * state["h"] / DEFF)
        if eta_vec is not None:                      # eta in freeboard m /(kg/m3)
            g = g + KAPPA * np.asarray(eta_vec, float)
        cols.append(g); diag.append(SIG_RHO_I ** 2)
    if "rho_s" in dirs:
        g = np.full(n, -state["hs"] / DEFF)
        if eta_s_vec is not None:                    # snow-linked differential
            g = g + KAPPA * np.asarray(eta_s_vec, float)
        cols.append(g); diag.append(SIG_RHO_S ** 2)
    if "hs" in dirs:
        cols.append(np.array([c["g_hs"] for c in chans])); diag.append(state["sig_hs"] ** 2)
    if "eps" in dirs:
        cols.append(np.array([c["g_eps"] * G_EPS_UNIT * state["hs"] for c in chans]))
        diag.append(SIG_EPS ** 2)
    if "rep" in dirs and rep > 0:
        cols.append(np.ones(n)); diag.append((KAPPA * rep) ** 2)
    if cols:
        G = np.column_stack(cols); B = np.diag(diag)
    else:
        G = np.zeros((n, 0)); B = np.zeros((0, 0))
    return R, G, B, rep


def sigma_post(state, chans, regime, eta_vec=None):
    n = len(chans)
    R, G, B, rep = design(state, chans, regime, ALL_DIRS, eta_vec)
    C = R + (G @ B @ G.T if G.shape[1] else 0.0)
    s = np.ones(n)
    return 1.0 / math.sqrt(float(s @ np.linalg.inv(C) @ s)), C, R, G, B, rep


def sigma_var(state, chans, regime, dirs=ALL_DIRS, eta_vec=None, eta_s_vec=None):
    """Model variance 1/(1' C^-1 1) for the nuisance groups listed in ``dirs``."""
    n = len(chans)
    R, G, B, _ = design(state, chans, regime, dirs, eta_vec, eta_s_vec)
    C = R + (G @ B @ G.T if G.shape[1] else 0.0)
    s = np.ones(n)
    return 1.0 / float(s @ np.linalg.inv(C) @ s)


GROUP_DIRS = {"density": ("rho_i", "rho_s"), "rho_i": ("rho_i",),
              "rho_s": ("rho_s",), "snow": ("hs",),
              "penetration": ("eps",), "representativeness": ("rep",)}


def added_var(state, chans, regime, group, eta_vec=None, eta_s_vec=None):
    """Marginal cost of one nuisance group: sigma^2(full) - sigma^2(full - group).

    Order-free and base-free: no arbitrary reference covariance is chosen, so
    the value cannot be moved by deciding which other nuisances to include
    first.  At eta = 0 this equals the closed form
    (h/Delta rho)^2 sigma_i^2 + (hs/Delta rho)^2 sigma_s^2.
    """
    drop = GROUP_DIRS[group]
    rest = tuple(d for d in ALL_DIRS if d not in drop)
    return (sigma_var(state, chans, regime, ALL_DIRS, eta_vec, eta_s_vec)
            - sigma_var(state, chans, regime, rest, eta_vec, eta_s_vec))


def density_split(state, chans, regime, eta_vec=None, eta_s_vec=None):
    """Split the density term into its ice and snow linkage parts.

    At eta = 0 the two linkages are collinear (both columns are proportional to
    one), so the split is exact and additive:
        a = (h/Delta rho)^2 sigma_i^2 + (hs/Delta rho)^2 sigma_s^2.
    Individual-parameter removals are *not* additive in that case (they overlap
    on the shared direction), so the additive split is reported here and the
    removals are reported separately as diagnostics.
    """
    a_i = (state["h"] / DEFF) ** 2 * SIG_RHO_I ** 2
    a_s = (state["hs"] / DEFF) ** 2 * SIG_RHO_S ** 2
    full = sigma_var(state, chans, regime, ALL_DIRS, eta_vec, eta_s_vec)
    return dict(total=added_var(state, chans, regime, "density", eta_vec, eta_s_vec),
                closed_form=a_i + a_s,
                rho_i=a_i,
                rho_s=a_s,
                marginal_rho_i=full - sigma_var(
                    state, chans, regime,
                    tuple(d for d in ALL_DIRS if d != "rho_i"), eta_vec, eta_s_vec),
                marginal_rho_s=full - sigma_var(
                    state, chans, regime,
                    tuple(d for d in ALL_DIRS if d != "rho_s"), eta_vec, eta_s_vec))


def _eta_vector(n, position, eta_mm):
    ev = np.zeros(n)
    if position is not None and eta_mm:
        ev[position] = eta_mm * 1e-3
    return ev


# ---------------------------------------------------------------- Appendix A
def density_blocks(state, chans, regime, eta_vec=None, eta_s_vec=None):
    """Split C into a dense part A and the two density columns (Appendix A.4).

    A       covariance of noise, snow depth, penetration and representativeness
    Gd      (n x 2) density sensitivity columns [g_rhoi(eta), g_rhos(eta_s)]
    Bd      (2 x 2) diagonal prior covariance of the two densities
    IA      1' A^-1 1, so that 1/IA is sigma^2 of the model without density

    This is exactly the decomposition C = A + Gd Bd Gd' with A free of any
    density direction, i.e. C_{-rho} of Eq. (identity) extended to eta != 0.
    """
    n = len(chans)
    rest = tuple(d for d in ALL_DIRS if d not in ("rho_i", "rho_s"))
    R, G, B, _ = design(state, chans, regime, rest)
    A = R + (G @ B @ G.T if G.shape[1] else 0.0)
    one = np.ones(n)
    IA = float(one @ np.linalg.solve(A, one))
    g_i = np.full(n, C_RHO_I_SIGN * state["h"] / DEFF)
    if eta_vec is not None:
        g_i = g_i + KAPPA * np.asarray(eta_vec, float)
    g_s = np.full(n, -state["hs"] / DEFF)
    if eta_s_vec is not None:
        g_s = g_s + KAPPA * np.asarray(eta_s_vec, float)
    return A, np.column_stack([g_i, g_s]), np.diag([SIG_RHO_I ** 2, SIG_RHO_S ** 2]), IA, one


def sigma_var_woodbury(state, chans, regime, eta_vec=None, eta_s_vec=None):
    """1/(1' C^-1 1) by a two-by-two Woodbury step (Appendix A.4).

        1' C^-1 1 = I_A - mu' M^-1 mu,
        mu = Gd' A^-1 1,   M = Bd^-1 + Gd' A^-1 Gd.

    Equals the direct inversion of ``sigma_var`` to machine precision; it is the
    path the response function uses, because it needs one dense solve plus a
    2 x 2 inverse instead of a fresh inversion at every eta.
    """
    A, Gd, Bd, IA, one = density_blocks(state, chans, regime, eta_vec, eta_s_vec)
    Ainv1 = np.linalg.solve(A, one)
    mu = Gd.T @ Ainv1
    M = np.linalg.inv(Bd) + Gd.T @ np.linalg.solve(A, Gd)
    I = IA - float(mu @ np.linalg.solve(M, mu))
    return 1.0 / I


def added_var_woodbury(state, chans, regime, eta_vec=None, eta_s_vec=None):
    """Marginal density term a(eta) = sigma^2(C) - 1/I_A, via Woodbury."""
    _, _, _, IA, _ = density_blocks(state, chans, regime, eta_vec, eta_s_vec)
    return (sigma_var_woodbury(state, chans, regime, eta_vec, eta_s_vec)
            - 1.0 / IA)


def eta_opt(state, chans, regime, channel=1):
    """Differential that maximises the response, in mm (kg m^-3)^-1.

    Appendix A.5: since a(eta) >= a_rhos with equality iff
    1' C_{-rhoi}^-1 g_rhoi(eta) = 0, the response is maximised where the
    ice-density column is GLS-orthogonal to the signal,

        eta_opt = -(h/Delta rho) * (1' C_{-rhoi}^-1 1)
                  / (kappa * [C_{-rhoi}^-1 1]_channel),

    with C_{-rhoi} = A + sigma_rhos^2 g_rhos g_rhos' the covariance that still
    contains the snow linkage.  This is *not* h/rho_w, the point where the
    second band's net ice-density sensitivity passes through zero: the
    orthogonality condition is taken in the noise metric, so the nuisance
    structure shifts the optimum.
    """
    n = len(chans)
    A, Gd, Bd, IA, one = density_blocks(state, chans, regime)
    g_s = Gd[:, 1]
    C_mi = A + SIG_RHO_S ** 2 * np.outer(g_s, g_s)
    Ci1 = np.linalg.solve(C_mi, one)
    e = np.zeros(n)
    e[channel] = 1.0
    c = C_RHO_I_SIGN * state["h"] / DEFF
    return -(c * float(one @ Ci1)) / (KAPPA * float(e @ Ci1)) * 1e3


def response(state, chans, regime, eta_mm, channel=1, variable="rho_i"):
    """r(eta) = 1 - a(eta)/a(0), density term from the marginal definition.

    ``eta_mm`` is a differential scattering-surface sensitivity in
    mm (kg m^-3)^-1 carried by channel ``channel``; all other channels carry
    eta = 0.  ``variable`` says which density linkage the differential
    responds to: "rho_i" (the buoyancy linkage, c = -h/Delta rho) or "rho_s"
    (the snow linkage, c_s = -hs/Delta rho).
    """
    a0 = added_var_woodbury(state, chans, regime)
    if variable == "rho_i":
        a = added_var_woodbury(state, chans, regime,
                               eta_vec=_eta_vector(len(chans), channel, eta_mm))
    elif variable == "rho_s":
        a = added_var_woodbury(state, chans, regime,
                               eta_s_vec=_eta_vector(len(chans), channel, eta_mm))
    else:
        raise ValueError(variable)
    return 100.0 * (1.0 - a / a0)


def _bisect_crossing(a, b, f, n=60):
    """Root of f in [a, b], given f(a) and f(b) on opposite sides of zero."""
    fa = f(a)
    for _ in range(n):
        m = 0.5 * (a + b)
        fm = f(m)
        if (fm >= 0.0) == (fa >= 0.0):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def response_window(state, chans, regime, target=10.0, variable="rho_i",
                    lo=-1.0, hi=40.0, step=0.05):
    """Interval of eta (mm per kg m^-3) on which r(eta) >= target.

    The response is not monotone: it rises to a plateau where one density
    linkage is annihilated and then falls again as the band overshoots, so the
    set {r >= target} is an interval, not a half-line.  A coarse scan locates
    the two crossings and bisection refines each of them; no monotonicity is
    assumed beyond a single crossing on each side, which is checked by the
    scan itself (the reported interval contains every sampled point that
    reaches the target).  An empty interval is returned as (nan, nan): the
    target is then unreachable for that linkage at any magnitude.
    """
    es = np.arange(lo, hi + 1e-12, step)
    f = lambda e: response(state, chans, regime, float(e), variable=variable) - target
    rs = np.array([f(e) for e in es])
    ok = rs >= 0.0
    if not ok.any():
        return float("nan"), float("nan")
    i = int(np.argmax(ok))
    lo_e = es[i] if i == 0 else _bisect_crossing(es[i - 1], es[i], f)
    j = len(es) - 1 - int(np.argmax(ok[::-1]))
    hi_e = es[j] if j == len(es) - 1 else _bisect_crossing(es[j + 1], es[j], f)
    return float(lo_e), float(hi_e)


def alpha_decomposition(state, chans, regime):
    """Raw and posterior sigma plus the fractional information gain alpha."""
    n = len(chans)
    R, _, _, _ = design(state, chans, regime, dirs=())
    sigma0 = 1.0 / math.sqrt(float(np.ones(n) @ np.linalg.inv(R) @ np.ones(n)))
    sig_post = math.sqrt(sigma_var(state, chans, regime, ALL_DIRS))
    alpha = 1.0 - sigma0 ** 2 / sig_post ** 2
    return dict(sigma_raw=sigma0, sigma_post=sig_post, alpha=alpha)


def alpha_null(state, chans, regime):
    """cos^2_{R^-1}(s, g_hs): fraction of signal killed by nulling snow."""
    n = len(chans)
    R, _, _, _ = design(state, chans, regime, dirs=())
    W = np.linalg.inv(R)
    s = np.ones(n)
    g = np.array([c["g_hs"] for c in chans])
    return float((s @ W @ g) ** 2 / ((s @ W @ s) * (g @ W @ g)))


def nuisance_vector(state, ch, eta=0.0):
    return np.array([-state["h"] / DEFF + KAPPA * eta,
                     -state["hs"] / DEFF,
                     ch["g_hs"],
                     ch["g_eps"] * G_EPS_UNIT * state["hs"],
                     1.0])


def marginal_gain(state, parent_id, child_id, regime="operational"):
    """Closed-form marginal information gain d^2/Q of the added last channel."""
    pc, cc = CONF[parent_id][2], CONF[child_id][2]
    R0, G0, B0, _ = design(state, pc, regime, ALL_DIRS)
    n0 = len(pc)
    s0 = np.ones(n0)
    C0 = R0 + G0 @ B0 @ G0.T
    C0inv = np.linalg.inv(C0)
    g_new = nuisance_vector(state, cc[-1])
    if G0.shape[1] < 5:
        g_new = g_new[:G0.shape[1]]
    Rn, _ = noise(state, [cc[-1]], regime)
    r_new = float(Rn[0, 0])
    d = 1.0 - float(s0 @ C0inv @ G0 @ B0 @ g_new)
    Q = r_new + float(g_new @ B0 @ g_new) \
        - float(g_new @ B0 @ G0.T @ C0inv @ G0 @ B0 @ g_new)
    dI = d * d / Q
    R1, G1, B1, _ = design(state, cc, regime, ALL_DIRS)
    I0 = float(s0 @ C0inv @ s0)
    I1 = float(np.ones(n0 + 1) @ np.linalg.inv(R1 + G1 @ B1 @ G1.T) @ np.ones(n0 + 1))
    return dict(dI_closed=dI, dI_numeric=I1 - I0, d=d, Q=Q,
                sigma_parent=1.0 / math.sqrt(I0), sigma_child=1.0 / math.sqrt(I1))


if __name__ == "__main__":
    print("info_geometry is a library; run p5_numbers.py to regenerate the tables.")
