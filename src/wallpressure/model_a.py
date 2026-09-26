"""Model A: two log-normal components in the pre-multiplied spectrum.

    f*phi_pp^+ = r_v * (g1(T+) + g2(T+))

with g_i = A_i * exp(-s_i * (log10 T - log10 T_i)^2). ``T+`` is the
inner-scaled period ``1/f+`` with ``f+ = f nu / u_tau^2``. The outer period is
tied to the inner one through ``T^o = T+ * U_e+/delta+``, i.e. the outer peak
sits at ``T+ = T^o_bar * delta+ * u_tau / U_e``.

Constants are Table 1 of Massey, Smits & McKeon (2026), as implemented in the
code that generated the paper's figures. Note that ``s_i`` here is the
coefficient multiplying the squared log10 distance (the paper's sigma_2 column
is the equivalent 10**(1/sqrt(s_2)); see docs/MODEL.md).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

# Re_tau ranges of the calibration data (paper, fig. 1)
CALIBRATED_RANGE = {"boundary_layer": (500, 11064), "pipe": (4794, 47015), "channel": (180, 5186)}
FLOWS = ("boundary_layer", "pipe", "channel")


def cf_approx(re_tau):
    """Approximate ``Cf/2 = (u_tau/U_e)^2`` for a zero-pressure-gradient BL."""
    return (np.log(re_tau) / 0.384 + 4.127) ** -2


@dataclass(frozen=True)
class _Constants:
    A1: Callable[[float], float]
    s1: float
    T1_plus: float
    A2: Callable[[float], float]
    s2: float
    To_bar: float
    r1: float
    r2: float


_CONSTANTS = {
    "boundary_layer": _Constants(lambda R: 2.2, 3.9, 20, lambda R: 1.4 * (np.log10(R) - 2.2), 1.2, 0.82, 0.5, 7),
    "pipe": _Constants(lambda R: 2.9 * (1 - 1000 / R), 4.3, 20, lambda R: 0.91 * (np.log10(R) - 2.2), 1.0, 0.18, 0.5, 7),
    "channel": _Constants(lambda R: 2.1 * (1 - 100 / R), 4.4, 12, lambda R: 0.9 * (np.log10(R) - 2.2), 1.0, 0.6, 0.5, 3),
}


def model_a_components(
    T_plus, re_tau: float, flow: str, utau_over_uouter: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(g1, g2, r_v)``; the modelled spectrum is ``r_v * (g1 + g2)``.

    Parameters
    ----------
    T_plus : array_like
        Inner-scaled period(s), ``T+ = u_tau^2 / (f nu)``.
    re_tau : float
        Friction Reynolds number delta+.
    flow : {"boundary_layer", "pipe", "channel"}
    utau_over_uouter : float, optional
        ``u_tau / U_e`` (BL free stream, or pipe/channel centreline). Required for
        pipe and channel; for a BL it defaults to ``sqrt(cf_approx(re_tau))``.
    """
    if flow not in _CONSTANTS:
        raise ValueError(f"flow must be one of {FLOWS}, got {flow!r}")
    if utau_over_uouter is None:
        if flow != "boundary_layer":
            raise ValueError("utau_over_uouter (= u_tau/U_centreline) is required for pipe and channel")
        utau_over_uouter = np.sqrt(cf_approx(re_tau))
    lo, hi = CALIBRATED_RANGE[flow]
    if not lo <= re_tau <= hi:
        warnings.warn(
            f"Re_tau={re_tau:g} is outside the calibrated range {lo}-{hi} for {flow} flow; "
            "model A is an extrapolation here"
            + (" (the pipe inner amplitude A1 is negative below Re_tau=1000)" if flow == "pipe" and re_tau < 1000 else ""),
            stacklevel=2,
        )
    c = _CONSTANTS[flow]
    T = np.asarray(T_plus, dtype=float)
    # smooth step, written in a form that cannot overflow
    rv = 0.5 * (1.0 + np.tanh(0.5 * c.r1 * (T - c.r2)))
    To_plus = c.To_bar * re_tau * utau_over_uouter
    g1 = c.A1(re_tau) * np.exp(-c.s1 * (np.log10(T) - np.log10(c.T1_plus)) ** 2)
    g2 = c.A2(re_tau) * np.exp(-c.s2 * (np.log10(T) - np.log10(To_plus)) ** 2)
    return g1, g2, rv


def model_a(T_plus, re_tau: float, flow: str, utau_over_uouter: Optional[float] = None) -> np.ndarray:
    """Pre-multiplied spectrum ``f phi_pp / tau_w^2`` from model A."""
    g1, g2, rv = model_a_components(T_plus, re_tau, flow, utau_over_uouter)
    return rv * (g1 + g2)
