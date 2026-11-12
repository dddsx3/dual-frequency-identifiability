# -*- coding: utf-8 -*-
"""n9 GLS 口径：把目标从 Kish 设计效应 n_eff 换成 GLS 信息量 I(S)=1ᵀR_S⁻¹1。

口径对照（必须在报告中标注用哪个）：
  - 等权/Kish 口径： n_eff = n²/1ᵀR1,      Var(mean)=σ²/n_eff,  MDE=2.802σ/√n_eff
  - GLS 口径：       I = 1ᵀR⁻¹1,            Var(mean)=σ²/I,      MDE=2.802σ/√I
两者在 R 为复合对称（全部非对角元相等）时等价：I = n/(1+(n−1)ρ) = n_eff；
一般相关结构下不等价，GLS 是正确的 BLUE 方差，等权口径是设计效应近似。

Schur 补增益（本文推导，见 01_formulation.md §2）：
  Δ_x(S) = I(S∪x) − I(S) = (1ᵀR_S⁻¹ r_x − 1)² / (r_xx − r_xᵀ R_S⁻¹ r_x)  ≥ 0
所以 I(S) 对任意正定 R **单调不减**（n_eff 不具此性质，见 n10 三架次反例）。
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# 实测相关锚点（n10 一致口径）：rho(0)=1, rho(25)=0.269, rho(75)=0.080, rho>=150=0
ANCHOR_D = np.array([0.0, 25.0, 75.0, 150.0])
ANCHOR_R = np.array([1.0, 0.269, 0.080, 0.0])


def rho_empirical(d):
    """分段线性插值；d>=150 km 归零。d 单位 km。"""
    d = np.abs(np.asarray(d, float))
    return np.interp(d, ANCHOR_D, ANCHOR_R, left=1.0, right=0.0)


def rho_exp(d, L, rho_c=0.0):
    """rho_c + (1-rho_c) exp(-d/L)。"""
    d = np.abs(np.asarray(d, float))
    return rho_c + (1.0 - rho_c) * np.exp(-d / L)


def corr_matrix(positions_km, kernel="empirical", L=326.0, rho_c=0.0, jitter=1e-10):
    """一维位置（km）上的相关矩阵。"""
    p = np.asarray(positions_km, float)
    D = np.abs(p[:, None] - p[None, :])
    if kernel == "empirical":
        R = rho_empirical(D)
    else:
        R = rho_exp(D, L, rho_c)
    R = R + jitter * np.eye(len(p))
    return R


def gls_info(R):
    """I(S) = 1ᵀR⁻¹1。"""
    n = R.shape[0]
    if n == 0:
        return 0.0
    try:
        c = cho_factor(R, lower=True, check_finite=False)
        x = cho_solve(c, np.ones(n), check_finite=False)
        return float(np.sum(x))
    except Exception:
        return float(np.sum(np.linalg.solve(R, np.ones(n))))


def kish_neff(R):
    """n_eff = n²/1ᵀR1（等权口径，保留作对照）。"""
    return float(R.shape[0] ** 2 / R.sum())


def delta_x(R_S, r_x, r_xx):
    """Schur 补边际增益 (1ᵀR_S⁻¹r_x − 1)²/(r_xx − r_xᵀR_S⁻¹r_x)。"""
    n = R_S.shape[0]
    if n == 0:
        return float((0 - 1) ** 2 / r_xx)
    c = cho_factor(R_S, lower=True, check_finite=False)
    h = float(np.sum(cho_solve(c, r_x, check_finite=False)))
    v = float(r_x @ cho_solve(c, r_x, check_finite=False))
    s = r_xx - v
    return float((h - 1.0) ** 2 / max(s, 1e-12))


def gls_band_var(R, band_idx, W, sigma, n_camp=1):
    """带权目标 Σ_b W_b μ_b 的 GLS 方差：Wᵀ(AᵀR⁻¹A)⁻¹W / n_camp。

    A 为单位阵的按带分块指示矩阵；AᵀR⁻¹A 为 nb×nb Gram 矩阵。
    sigma 为逐带标准差向量；内部按相关阵计算，最后乘 σ 尺度。
    """
    nb = len(W)
    n = R.shape[0]
    A = np.zeros((n, nb))
    for i, b in enumerate(band_idx):
        if b >= 0:
            A[i, b] = 1.0
    c = cho_factor(R, lower=True, check_finite=False)
    X = cho_solve(c, A, check_finite=False)             # R⁻¹A
    G = A.T @ X                                         # AᵀR⁻¹A
    covered = np.diag(G) > 0
    # 未覆盖带：按 1 个有效样本罚
    Gc = G[np.ix_(covered, covered)]
    if Gc.size == 0:
        return float(np.sum((W * sigma) ** 2)), np.zeros(nb)
    wv = (W * sigma)[covered]
    try:
        Gi = np.linalg.inv(Gc + 1e-12 * np.eye(Gc.shape[0]))
    except Exception:
        Gi = np.linalg.pinv(Gc)
    var_cov = float(wv @ Gi @ wv)
    pen = float(np.sum((W * sigma)[~covered] ** 2))
    return var_cov / n_camp + pen, G
