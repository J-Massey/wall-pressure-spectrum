"""High-level, dimensional interface."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_a import model_a, model_a_components
from .model_b import model_b


@dataclass
class Spectrum:
    f: np.ndarray            # frequency [Hz]
    premultiplied: np.ndarray  # f * phi_pp / tau_w^2  (dimensionless)
    phi_pp: np.ndarray       # one-sided spectrum [Pa^2/Hz] (units of tau_w^2/Hz if rho=1)
    f_plus: np.ndarray
    T_plus: np.ndarray
    re_tau: float
    tau_w: float


def predict_spectrum(
    f,
    *,
    model: str = "A",
    flow: str = "boundary_layer",
    u_tau: float,
    nu: float,
    delta: float,
    u_outer: float | None = None,
    rho: float = 1.0,
) -> Spectrum:
    """Predict the wall-pressure spectrum at frequencies ``f`` [Hz].

    Parameters
    ----------
    model : "A" (log-normal) or "B" (modified Lorentzian).
    flow : "boundary_layer", "pipe" or "channel".
    u_tau, nu : friction velocity [m/s] and kinematic viscosity [m^2/s].
    delta : BL thickness (99 %), pipe radius or channel half-height [m].
    u_outer : free-stream (BL) or centreline (pipe/channel) velocity [m/s].
        Optional only for a BL with model A, where a Cf correlation is used.
    rho : density [kg/m^3]; with rho=1 the spectrum is in units of (u_tau^2)^2/Hz.
    """
    f = np.asarray(f, dtype=float)
    if np.any(f <= 0):
        raise ValueError("frequencies must be positive")
    re_tau = u_tau * delta / nu
    f_plus = f * nu / u_tau**2
    ratio = None if u_outer is None else u_tau / u_outer
    m = model.upper()
    if m == "A":
        pre = model_a(1.0 / f_plus, re_tau, flow, ratio)
    elif m == "B":
        if ratio is None:
            raise ValueError("model B needs u_outer")
        pre = model_b(f_plus, re_tau, ratio)
    else:
        raise ValueError("model must be 'A' or 'B'")
    tau_w = rho * u_tau**2
    return Spectrum(f, pre, pre * tau_w**2 / f, f_plus, 1.0 / f_plus, re_tau, tau_w)


def variance_plus(spec_f_plus, premultiplied) -> float:
    """``<p^2>/tau_w^2`` = integral of (f phi/tau_w^2) d ln f+ (trapezoid, log-spaced grid)."""
    x = np.log(np.asarray(spec_f_plus))
    y = np.asarray(premultiplied)
    i = np.argsort(x)
    return float(np.sum(0.5 * (y[i][1:] + y[i][:-1]) * np.diff(x[i])))


def variance_lee_moser(re_tau):
    """Channel-flow variance trend ``<p^2>/tau_w^2 = 2.24 ln(Re_tau) - 9.18`` (Lee & Moser 2015)."""
    return 2.24 * np.log(re_tau) - 9.18
