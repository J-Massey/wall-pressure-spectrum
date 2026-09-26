from __future__ import annotations

import os
from typing import Iterable

from matplotlib import pyplot as plt
import numpy as np
from icecream import ic


class EvalPipeModel:
    def __init__(
        self,
        re_tau: float,
        f: np.ndarray,
        params: Iterable[float] | None = None,
        f_inner: np.ndarray | None = None,
        f_outer: np.ndarray | None = None,
    ) -> None:
        self.re_tau = re_tau
        self.f = f
        if f_inner is None:
            self.convert_f_inner()
        if f_outer is None:
            self.convert_f_outer()
        elif f_inner is not None and f_outer is not None:
            self.f_inner = f_inner
            self.f_outer = f_outer
        else:
            print("No f provided")
        if params is None:
            self.params = self._load_model_params()
        else:
            self.params = params
        self.a1, self.a2 = self.params[:2]
        self.ph1, self.ph2 = self.params[2:4]
        self.q1, self.q2 = self.params[4:6]


    def convert_f_inner(self) -> None:
        R = 0.4505
        nu = 1.5e-5
        U_tau_m = self.re_tau * nu / R
        self.f_inner = self.f * nu / U_tau_m**2

    def convert_f_outer(self) -> None:
        U_cl = self.U_Re()
        R = 0.4505
        self.f_outer = self.f * R / U_cl

    def _load_model_params(self) -> np.ndarray:
        """
        Load model parameters from a .npy file.
        """
        params = np.load("data/model_params.npy")
        return params
    
    def U_Re(self) -> float:
        """
        Relationship between Re_tau and U_cl for Dacome et al. (2025) pipe flow.
        """
        a, b, c = -0.0163797 ,  1.20780467, -3.64204499
        U = 10**(a*(np.log10(self.re_tau)**2)
                + b*np.log10(self.re_tau)
                + c)
        return U

    def inner_model(
        self,
        A: float = 1.6,
        f0: float = 0.1,
        p_low: float = 1,
        p_high: float = -6,
        q: float = 3.5,
    ) -> np.ndarray:
        r = (p_low - p_high) / q
        # ic(r)
        # Normalize so that W(f0)=2^(-r), hence:
        phi_in = (
            A
            * (2**r)
            * ((self.f_inner / f0) ** p_low * (1 + (self.f_inner / f0) ** q) ** (-r))
        )
        return phi_in

    def outer_model(self, f0: float = 2 / 3, p_low: float = 3) -> np.ndarray:
        r = (p_low - self.p_high_func()) / self.q_func()
        # ic(r)
        return (
            self.A_func()
            * (2**r)
            * ((self.f_outer / f0) ** p_low * (1 + (self.f_outer / f0) ** self.q_func()) ** (-r))
        )

    def A_func(self, L: float = 1) -> float:
        """
        Compute outer model parameter A as a function of Reynolds number (Re).
        Using a logarithmic dependency: A(Re) = A_const + A_slope * log(Re).
        """
        X  = np.log10(self.re_tau)
        return L / (1 + np.exp(self.a1*(-X + self.a2)))

    def p_high_func(self, L: float = 1.5) -> float:
        """
        Compute outer model parameter p_high as a linear fit
        decaying to L at the limit of Re_\\tau \\to \\infty.
        """
        X = np.log10(self.re_tau)
        # ph1 = -2.9
        # ph2 = 0.5
        return -2 + L / (1 + np.exp(self.ph1*(-X + self.ph2)))

    def q_func(self, L: float = 0.6) -> float:
        """
        Compute outer model parameter q as a function of Reynolds number (Re).
        Using a logarithmic dependency: q(Re) = q_const + q_slope * log(Re).
        """
        X = np.log10(self.re_tau)
        lam = abs(self.q1 / self.q2)
        # q1 = -0.16
        # q2 = 0.2
        q = L + ((self.q1 - L) + self.q2*X) / (1 + (X/lam)**2)
        return 0.2 + L / (1 + np.exp(self.q1*(-X + self.q2)))


if __name__ == "__main__":
    f = np.linspace(0.1, 10, 100)
    pipe_model = EvalPipeModel(7000, f=f)
    ic("tested")
    os.system("python src/plotting.py")
