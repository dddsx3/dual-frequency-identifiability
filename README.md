# Dual-frequency identifiability of sea-ice density

A second radar band is proposed to add density information to sea-ice thickness
retrieval. Whether it adds *identifiability* is a system property, and it can be
settled before launch. This repository computes the answer: every channel is
converted through the same buoyancy factor, so the density direction is
common-mode, and the density term adds to the posterior variance as

    sigma^2 = 1/(1' C_-rho^-1 1) + a

with `a` split exactly between an ice-density part (93.7 / 96.7 %) and a
snow-density part (6.3 / 3.3 %). A differential scattering-surface sensitivity
moves `a` only through a signed, linkage-specific response whose ceiling is that
share.

## What is here

    code/info_geometry.py   states, channels, noise, information, the Woodbury
                            form, the response function, the ceiling
    code/p5_numbers.py      every table and number, with 42 assertions
    code/make_figures.py    the four figures
    code/submod_p5.py       monotonicity and submodularity over 60 designs
    code/check_package.py   gate: assertions, number-by-number match against the
                            manuscript, citation coverage, worked example
    code/tests/             12 unit tests of the claimed properties
    out/                    every table as CSV, including p5_thresholds.csv
    figures/

## Run it

    pip install numpy scipy pandas matplotlib
    python code/p5_numbers.py --check     # 42 assertions, exits non-zero on failure
    python code/tests/test_p5_numbers.py  # 12 tests
    python code/check_package.py          # the gate, plus a worked assessment

`check_package.py --eta 0.300` applies the paper's own protocol to a supplied
differential and prints the tier it lands in.

## The two numbers to look at first

    out/p5_thresholds.csv     the 10 % and 50 % edges (0.119 / 0.299 mm per kg m^-3)
                              and the ceiling (93.7 / 96.7 %) with the eta where it
                              is attained (2.194 / 5.806)
    out/p5_appendix.csv       the ceiling against a grid search, and the Woodbury
                              residual against direct inversion (1.8e-15)

## Licence and citation

MIT (see LICENSE). Derived tables CC BY 4.0. Cite the article and this repository.
