"""Model B: two modified-Lorentzian components with smooth Re_tau dependence.

    f*phi_pp^+ = g_in(f+) + g_out(f^o; Re_tau)

    g = A * 2^r * (f/fb)^p_low * (1 + (f/fb)^q)^(-r),   r = (p_low - p_high)/q

The inner component is Re_tau-invariant (f+ break 0.1). The outer component has
break f^o = 2/3 (T^o = 3/2), p_low = 3, and A, p_high, q given by logistic
functions of log10(Re_tau) (eq. 3.14 of the paper).
"""
from __future__ import annotations

import warnings

import numpy as np

# Table 2 (full precision, from the fitted parameter file)
A_INNER, FB_INNER, PLOW_INNER, PHIGH_INNER, Q_INNER = 1.6, 0.1, 1.0, -6.0, 3.5
FB_OUTER, PLOW_OUTER = 2.0 / 3.0, 3.0
DEFAULT_PARAMS = (
    2.38825114, 3.5481589,   # a_A, b_A
    -0.22465544, 3.57890138,  # a_p, b_p
    -1.09031052, 3.77538926,  # a_q, b_q
)


def _lorentzian(f, A, fb, p_low, p_high, q):
    r = (p_low - p_high) / q
    x = f / fb
    return A * 2.0**r * x**p_low * (1.0 + x**q) ** (-r)


def outer_parameters(re_tau, params=DEFAULT_PARAMS):
    """Return ``(A_out, p_high_out, q_out)`` at ``re_tau`` (eq. 3.14a-c)."""
    aA, bA, ap, bp, aq, bq = params
    X = np.log10(re_tau)
    A = 1.0 / (1.0 + np.exp(aA * (-X + bA)))
    p_high = -2.0 + 1.5 / (1.0 + np.exp(ap * (-X + bp)))
    q = 0.2 + 0.6 / (1.0 + np.exp(aq * (-X + bq)))
    return A, p_high, q


def model_b_components(f_plus, f_outer, re_tau: float, params=DEFAULT_PARAMS):
    """Return ``(g_inner, g_outer)`` for inner frequency ``f+ = f nu/u_tau^2`` and
    outer frequency ``f^o = f delta/U_e`` (arrays of equal shape)."""
    f_plus = np.asarray(f_plus, dtype=float)
    f_outer = np.asarray(f_outer, dtype=float)
    g_in = _lorentzian(f_plus, A_INNER, FB_INNER, PLOW_INNER, PHIGH_INNER, Q_INNER)
    A, ph, q = outer_parameters(re_tau, params)
    g_out = _lorentzian(f_outer, A, FB_OUTER, PLOW_OUTER, ph, q)
    return g_in, g_out


def model_b(f_plus, re_tau: float, utau_over_uouter: float, params=DEFAULT_PARAMS) -> np.ndarray:
    """Pre-multiplied spectrum ``f phi_pp / tau_w^2`` from model B.

    ``f^o = f+ * Re_tau * (u_tau/U_e)`` links the inner and outer frequencies.
    """
    if not 4794 <= re_tau <= 47015:
        warnings.warn(
            f"Re_tau={re_tau:g} is outside the range 4794-47015 the outer component was calibrated on; "
            "model B is an extrapolation here (smooth, but unvalidated)",
            stacklevel=2,
        )
    f_plus = np.asarray(f_plus, dtype=float)
    g_in, g_out = model_b_components(f_plus, f_plus * re_tau * utau_over_uouter, re_tau, params)
    return g_in + g_out
