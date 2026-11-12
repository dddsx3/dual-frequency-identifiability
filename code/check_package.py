# -*- coding: utf-8 -*-
"""Single entry point for the reproduction package of this manuscript.

Run from code/:   python check_package.py [--eta 0.300] [--skip-model]

Checks, in order
  1. every model assertion of p5_numbers.py (identity, split, ceiling,
     Woodbury, monotonicity, width floor) -- 42 assertions;
  2. every headline number of the manuscript against the archived CSV that
     produced it, so that a number cannot drift between text and table;
  3. citation coverage: every \\cite key resolves, every bib entry is used,
     every entry carries a DOI (or is flagged);
  4. every \\includegraphics target exists;
  5. the protocol of Section 2.3 applied to a supplied eta, printed as a tier.

Exit status is non-zero if any check fails, which is what makes the package
usable as a gate before submission.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
OUT = os.path.join(ROOT, "out")
TEX = os.path.join(ROOT, "manuscript.tex")
BIB = os.path.join(ROOT, "refs.bib")

FAIL = []


def note(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(rows, **where):
    for r in rows:
        if all(str(r[k]) == str(v) for k, v in where.items()):
            return r
    raise KeyError(where)


def f(x):
    return float(x)


# --------------------------------------------------------------------- 1
def run_model_checks():
    print("[1] model assertions (p5_numbers.py --check)")
    r = subprocess.run([sys.executable, "p5_numbers.py", "--check"], cwd=HERE,
                       capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if l.startswith("checks:")]
    ok = r.returncode == 0
    note(ok, "p5_numbers.py --check -> %s" % (tail[0] if tail else "no summary"))
    for l in r.stdout.splitlines():
        if l.strip().startswith("FAIL"):
            print("       " + l.strip())
    return ok


# --------------------------------------------------------------------- 2
def manuscript_numbers():
    """Headline numbers, each tied to the CSV cell it comes from.

    The check is a string match on the manuscript, so a number can only pass
    if the exact printed form is present.  Formatting here must therefore match
    the text: this is deliberate, because a mismatch is exactly the failure
    mode (text and table disagreeing) that the check exists to catch.
    """
    ident = load("p5_identity.csv")
    split = load("p5_density_split.csv")
    win = load("p5_window.csv")
    app = load("p5_appendix.csv")
    met = load("p5_metrics.csv")
    width = load("p5_width_floor.csv")
    mission = load("p5_mission.csv")

    a_low = f(num(ident, state="low", regime="operational", config="cci")["a_m2"])
    a_high = f(num(ident, state="high", regime="operational", config="cci")["a_m2"])
    sp_low = num(split, state="low", config="ku_ka_p50")
    sp_high = num(split, state="high", config="ku_ka_p50")
    w_low = num(win, state="low", config="ku_ka_p50")
    w_high = num(win, state="high", config="ku_ka_p50")
    ap_low = num(app, state="low", config="ku_ka_p50")
    ap_high = num(app, state="high", config="ku_ka_p50")
    m_low = num(met, state="low", config="cci")
    m_atl = num(met, state="low", config="cci_atl20")
    m_kuku = num(met, state="low", config="ku_ku")
    wd = num(width, n_eff="10")

    return [
        ("a, low state", "%.4f" % a_low),
        ("a, high state", "%.4f" % a_high),
        ("ice share, low", "%.1f" % (100 * f(sp_low["share_rho_i"]))),
        ("snow share, low", "%.1f" % (100 * f(sp_low["share_rho_s"]))),
        ("ice share, high", "%.1f" % (100 * f(sp_high["share_rho_i"]))),
        ("snow share, high", "%.1f" % (100 * f(sp_high["share_rho_s"]))),
        ("10 % edge, low", "%.3f" % f(w_low["rho_i_10_lo"])),
        ("10 % edge, high", "%.3f" % f(w_high["rho_i_10_lo"])),
        ("50 % edge, low", "%.3f" % f(w_low["rho_i_50_lo"])),
        ("50 % edge, high", "%.3f" % f(w_high["rho_i_50_lo"])),
        ("ceiling, low", "%.1f" % f(ap_low["r_max_pct"])),
        ("ceiling, high", "%.1f" % f(ap_high["r_max_pct"])),
        ("ceiling location, low", "%.3f" % f(ap_low["eta_opt_closed_mm"])),
        ("ceiling location, high", "%.3f" % f(ap_high["eta_opt_closed_mm"])),
        ("refraction ceiling, low",
         "%.3f" % f(w_low["eta_max_refraction_mm_per_kgm3"])),
        ("refraction ceiling, high",
         "%.3f" % f(w_high["eta_max_refraction_mm_per_kgm3"])),
        ("sigma, Ku only, low", "%.4f" % f(m_low["sigma_m"])),
        ("sigma, Ku+lidar, low", "%.4f" % f(m_atl["sigma_m"])),
        ("sigma, Ku+Ku, low", "%.4f" % f(m_kuku["sigma_m"])),
        ("width floor, n_eff=10", "%.2f" % f(wd["width_m"])),
        ("observation term, n_eff=10", "%.2f" % f(wd["obs_term_m"])),
        ("requirement at h=1.0", "%.3f" % f(num(mission, h_m="1")["eta_10_abs"])),
        ("requirement at h=4.0", "%.3f" % f(num(mission, h_m="4")["eta_10_abs"])),
    ]


def check_numbers(tex):
    print("[2] manuscript numbers against the archived CSVs")
    ok = True
    for label, literal in manuscript_numbers():
        good = literal in tex
        note(good, "%-30s %s" % (label, literal))
        ok &= good
    return ok


# --------------------------------------------------------------------- 3
def check_citations(tex, bib):
    print("[3] citation coverage")
    cited = set()
    for m in re.finditer(r"\\cite[pt]?\*?\{([^}]*)\}", tex):
        cited.update(k.strip() for k in m.group(1).split(","))
    keys = re.findall(r"@\w+\{([^,]+),", bib)
    doi = len(re.findall(r"doi\s*=\s*\{", bib))
    orphan_cite = sorted(cited - set(keys))
    unused = sorted(set(keys) - cited)
    note(not orphan_cite, "every \\cite key resolves (%d keys, %d entries)"
         % (len(cited), len(keys)))
    if orphan_cite:
        print("       missing from bib: %s" % orphan_cite)
    note(not unused, "every bib entry is cited")
    if unused:
        print("       never cited: %s" % unused)
    note(doi == len(keys), "every entry carries a DOI (%d/%d)" % (doi, len(keys)))
    return not orphan_cite and not unused


# --------------------------------------------------------------------- 4
def check_figures(tex):
    print("[4] figure targets")
    ok = True
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        exists = os.path.exists(os.path.join(ROOT, name))
        note(exists, name)
        ok &= exists
    return ok


# --------------------------------------------------------------------- 5
def worked_assessment(eta_mm):
    import info_geometry as ig
    chans = ig.CONF["ku_ka_p50"][2]
    out = {}
    for nm in ("low", "high"):
        st = ig.STATES[nm]
        red = ig.response(st, chans, "operational", eta_mm, variable="rho_i")
        tier = "useful" if red >= 50 else ("boundary" if red >= 10 else "not useful")
        out[nm] = dict(reduction=red, tier=tier,
                       edge10=ig.response_window(st, chans, "operational", 10.0)[0])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=0.300,
                    help="differential to assess (mm per kg m^-3), default 0.300")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args(argv)

    have_tex = os.path.exists(TEX)
    tex = open(TEX, encoding="utf-8").read() if have_tex else ""
    bib = open(BIB, encoding="utf-8").read() if os.path.exists(BIB) else ""

    if not args.skip_model:
        run_model_checks()
    elif not all(os.path.exists(os.path.join(OUT, n)) for n in
                 ("p5_metrics.csv", "p5_window.csv", "p5_appendix.csv")):
        print("out/ is incomplete: run p5_numbers.py first")
        return 2
    if have_tex:
        check_numbers(tex)
        check_citations(tex, bib)
        check_figures(tex)
    else:
        print("[2-4] manuscript.tex not present: this is the artefact repository, so the")
        print("      number-by-number match and the citation audit are skipped here.")
        print("      Run this script from the paper package to exercise them.")

    print("[5] protocol applied to eta = %.3f mm (kg m^-3)^-1, ku_ka_p50"
          % args.eta)
    for nm, r in worked_assessment(args.eta).items():
        print("  %-4s reduction=%6.2f %%  tier=%-10s  10 %% edge=%.3f"
              % (nm, r["reduction"], r["tier"], r["edge10"]))

    print()
    if FAIL:
        print("%d check(s) failed" % len(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
