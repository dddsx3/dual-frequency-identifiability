# -*- coding: utf-8 -*-
"""Build the four manuscript figures from the CSVs written by p5_numbers.py
and submod_p5.py.  Vector PDF plus 300-dpi PNG previews are written to ../figs.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "out"))
FIG = os.path.abspath(os.path.join(HERE, os.pardir, "figs"))
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "font.family": "DejaVu Sans", "pdf.fonttype": 42,
})

C_LOW, C_HIGH = "#0072B2", "#D55E00"
C_GREY = "#7F7F7F"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


SHORT = {
    "cci": "Ku only\n(CCI)",
    "cci_atl20": "Ku + lidar\n(CCI+ATL20)",
    "ku_ku": "Ku + Ku\n(control)",
    "ku_ka_p25": "Ku + Ka\n$p$=0.25",
    "ku_ka_p50": "Ku + Ka\n$p$=0.50",
    "ku_ka_p75": "Ku + Ka\n$p$=0.75",
    "ku_ka_p50_atl20": "Ku+Ka+lidar",
}
ORDER = ["cci", "cci_atl20", "ku_ku", "ku_ka_p25", "ku_ka_p50", "ku_ka_p75", "ku_ka_p50_atl20"]


def fig1():
    m = pd.read_csv(os.path.join(OUT, "p5_metrics.csv"))
    m = m[m.state == "low"]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ys = np.arange(len(ORDER))[::-1]
    vals = [float(m[m.config == k].sigma_m.iloc[0]) for k in ORDER]
    cols = [C_GREY] + [C_LOW] * 6
    ax.barh(ys, vals, color=cols, edgecolor="k", linewidth=0.5, height=0.72)
    ax.set_yticks(ys)
    ax.set_yticklabels([SHORT[k] for k in ORDER], fontsize=7)
    ax.set_xlabel("posterior $\\sigma(h)$  (m), low-latitude state")
    ax.set_xlim(0, 1.0)
    for y, k, v in zip(ys, ORDER, vals):
        row = m[m.config == k].iloc[0]
        lab = "%.2f" % v
        if k != "cci" and row.delta_sigma_pct == row.delta_sigma_pct:
            lab += "  ($-%.1f\\%%$)" % row.delta_sigma_pct
        ax.text(v + 0.015, y, lab, va="center", fontsize=6.6)
    ax.grid(axis="x", color="0.85", lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig1_sigma_post")


def fig2():
    d = pd.read_csv(os.path.join(OUT, "p5_directions.csv"))
    d = d[(d.regime == "operational") & (d.config != "cci")]
    dirs = ["density", "snow", "penetration", "representativeness"]
    labels = ["density\n$\\rho_i,\\rho_s$", "snow depth", "penetration", "representativeness"]
    sel = ["cci_atl20", "ku_ku", "ku_ka_p25", "ku_ka_p50", "ku_ka_p50_atl20"]
    names = ["Ku+lidar", "Ku+Ku", "Ku+Ka $p$=0.25", "Ku+Ka $p$=0.50", "Ku+Ka+lidar"]
    cols = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    w = 0.16
    for i, (k, nm, col) in enumerate(zip(sel, names, cols)):
        vals = []
        for g in dirs:
            r = d[(d.config == k) & (d.state == "low") & (d.direction == g)]
            vals.append(float(r.reduction_pct.iloc[0]) if len(r) else 0.0)
        ax.bar(np.arange(len(dirs)) + (i - 2) * w, vals, width=w, label=nm, color=col,
               edgecolor="k", linewidth=0.35)
    ax.set_xticks(range(len(dirs)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("reduction of direction-induced\nvariance (%)")
    ax.set_ylim(-3, 105)
    ax.text(0, 3.0, "0 % in\nall five", ha="center", va="bottom", fontsize=6.3,
            color="0.35")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(axis="y", color="0.85", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False,
              columnspacing=0.8, handlelength=1.0, handletextpad=0.4)
    save(fig, "fig2_directions")


def fig3():
    c = pd.read_csv(os.path.join(OUT, "p5_response.csv"))
    c = c[c.config == "ku_ka_p50"]
    win = pd.read_csv(os.path.join(OUT, "p5_window.csv"))
    win = win[win.config == "ku_ka_p50"]
    app = pd.read_csv(os.path.join(OUT, "p5_appendix.csv"))
    app = app[app.config == "ku_ka_p50"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.1),
                                  gridspec_kw={"width_ratios": [2.0, 1.0]})
    style = {"low": ("-", "o", C_LOW), "high": ("--", "s", C_HIGH)}

    # (a) the ice-density branch over the design range
    for nm, (ls, mk, col) in style.items():
        s = c[c.state == nm].sort_values("eta_mm_per_kgm3")
        ax.plot(s.eta_mm_per_kgm3, s.r_rho_i_pct, color=col, ls=ls, lw=1.4,
                label="%s latitude" % nm)
        w = win[win.state == nm].iloc[0]
        a = app[app.state == nm].iloc[0]
        ax.plot([w.rho_i_10_lo], [10.0], mk, ms=4.6, color=col, zorder=6,
                mfc=col if nm == "low" else "white", mec="k", mew=0.3)
        ax.plot([w.rho_i_50_lo], [50.0], mk, ms=4.6, color=col, zorder=6, mfc="white",
                mec="k", mew=0.3)
        ax.plot([a.eta_opt_closed_mm], [a.r_max_pct], "*", ms=9, color=col,
                mec="k", mew=0.4, zorder=7)
        ax.annotate("%.3f" % w.rho_i_50_lo, (w.rho_i_50_lo, 50.0),
                    textcoords="offset points", xytext=(-15, -12), fontsize=6.4,
                    color=col, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
        ax.annotate("%.1f %%" % a.r_max_pct, (a.eta_opt_closed_mm, a.r_max_pct),
                    textcoords="offset points", xytext=(-8, 7), fontsize=6.4,
                    color=col, ha="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
        ax.plot([w.eta_max_refraction_mm_per_kgm3] * 2, [-50, -45], color=col, lw=1.0)
        ax.annotate("$\\eta_{\\max}$", (w.eta_max_refraction_mm_per_kgm3, -44),
                    textcoords="offset points", xytext=(0, 1), fontsize=6.0,
                    color=col, rotation=90, ha="center", va="bottom")
    for y in (10, 50):
        ax.axhline(y, color="0.5", lw=0.6, ls=(0, (4, 2)))
        ax.text(9.7, y + 1.8, "%d %%" % y, fontsize=6.2, color="0.35", ha="right")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlim(-2.5, 10)
    ax.set_ylim(-50, 108)
    ax.set_xlabel("$\\eta_{\\rho_i}$  (mm (kg m$^{-3}$)$^{-1}$)")
    ax.set_ylabel("change of the density term  $r$  (%)")
    ax.grid(color="0.93", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.4, loc="lower right")
    ax.set_title("(a) ice-linked differential", fontsize=8)

    # (b) the snow-density branch on its own scale: it never reaches 10 %
    for nm, (ls, mk, col) in style.items():
        s = c[(c.state == nm) & (c.eta_mm_per_kgm3 >= 0) & (c.eta_mm_per_kgm3 <= 0.9)]
        ax2.plot(s.eta_mm_per_kgm3, s.r_rho_s_pct, color=col, ls=ls, lw=1.3)
        sh = float(win[win.state == nm].plateau_rho_s_pct.iloc[0])
        ax2.axhline(sh, color=col, lw=0.7, ls=(0, (1, 2)))
        ax2.text(0.88, sh + 0.35, "ceiling %.1f %%" % sh, fontsize=6.0, color=col,
                 ha="right")
    ax2.axhline(10.0, color="0.5", lw=0.6, ls=(0, (4, 2)))
    ax2.text(0.88, 10.35, "10 % tier", fontsize=6.0, color="0.35", ha="right")
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_xlim(0, 0.9)
    ax2.set_ylim(-2, 12)
    ax2.set_xlabel("$\\eta_{\\rho_s}$  (mm (kg m$^{-3}$)$^{-1}$)")
    ax2.set_ylabel("change of the density term  $r$  (%)")
    ax2.grid(color="0.93", lw=0.5)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) snow-linked differential", fontsize=8)
    fig.tight_layout()
    save(fig, "fig3_eta_threshold")


def fig4():
    w = pd.read_csv(os.path.join(OUT, "p5_witness.csv")).set_index("metric")["value"]
    sm = pd.read_csv(os.path.join(OUT, "gls_submod.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), gridspec_kw={"width_ratios": [1, 1.3]})

    ax = axes[0]
    ne = [float(w["n_eff_two_sorties"]), float(w["n_eff_three_sorties"])]
    ax.bar([0, 1], ne, color=[C_LOW, C_HIGH], edgecolor="k", linewidth=0.5, width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2 sorties\n(0, 25 km)", "3 sorties\n(+12.5 km)"], fontsize=7)
    ax.set_ylabel("effective sample size  $n_{\\rm eff}$")
    ax.set_ylim(0, 2.05)
    for x, v in zip([0, 1], ne):
        ax.text(x, v + 0.025, "%.3f" % v, ha="center", fontsize=7)
    ax.text(0.5, 2.0, "$I=%.6f$ in both\n$\\Delta=%.1f\\times10^{-20}$"
            % (float(w["I_GLS_two_sorties"]), float(w["marginal_gain_delta"]) / 1e-20),
            ha="center", va="top", fontsize=6.3)
    ax.set_title("(a) redundant middle sortie", fontsize=8)
    ax.grid(axis="y", color="0.88", lw=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    n_cfg = sm.groupby("kernel").size()
    n_cfg = n_cfg.reindex(["empirical", "pchip", "exp326", "expnugget"])
    xs = np.arange(len(n_cfg))
    props = ["mono_violations", "submod_violations", "supermod_violations"]
    names = ["monotone (0/60)", "submodular (0/60)", "supermodular (60/60)"]
    cols = ["#009E73", C_LOW, C_HIGH]
    ymax = float(n_cfg.max())
    for i, (p, nm, col) in enumerate(zip(props, names, cols)):
        counts = []
        for k in n_cfg.index:
            v = sm[sm.kernel == k][p]
            counts.append(int((v > 0).sum()))
        ax.bar(xs + (i - 1) * 0.24, counts, width=0.22, label=nm, color=col,
               edgecolor="k", linewidth=0.35)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xticks(xs)
    ax.set_xticklabels(n_cfg.index, rotation=12, fontsize=7)
    ax.set_ylabel("configurations per kernel (of %d)" % ymax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              fontsize=6.5, handlelength=1.0, columnspacing=0.8)
    ax.set_title("(b) 15 spatio-temporal designs $\\times$ 4 kernels", fontsize=8)
    ax.grid(axis="y", color="0.88", lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig4_gls_submod")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("figures written to", FIG)
