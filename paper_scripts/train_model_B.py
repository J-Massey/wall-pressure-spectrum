from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit, minimize, least_squares

from icecream import ic

import h5py
from src.inference_model_B import EvalPipeModel


class DacomePipe:
    metadata = {
        4794: {"U_tau": 0.162, "tau_w": 0.032, "l_star": 94.0, "U_CL": 3.837},
        7148: {"U_tau": 0.242, "tau_w": 0.070, "l_star": 63.0, "U_CL": 5.833},
        14004: {"U_tau": 0.473, "tau_w": 0.269, "l_star": 32.2, "U_CL": 12.11},
        22877: {"U_tau": 0.773, "tau_w": 0.718, "l_star": 19.7, "U_CL": 20.71},
        31614: {"U_tau": 1.068, "tau_w": 1.368, "l_star": 14.3, "U_CL": 29.50},
        38271: {"U_tau": 1.293, "tau_w": 2.008, "l_star": 11.8, "U_CL": 34.13},
        47015: {"U_tau": 1.588, "tau_w": 3.001, "l_star": 9.58, "U_CL": 44.60},
    }
    filenames = {
        4794: "data/dacome25/4794.csv",
        7148: "data/dacome25/7148.csv",
        14004: "data/dacome25/14004.csv",
        22877: "data/dacome25/22877.csv",
        31614: "data/dacome25/31614.csv",
        38271: "data/dacome25/38271.csv",
        47015: "data/dacome25/47015.csv",
    }
    def __init__(self):
        self.re_tau = np.array([4794, 7148, 14004, 22877, 31614, 38271, 47015])
        self.datasets = {}
        for Re in self.re_tau:
            fn = f"data/dacome25/{int(Re)}_clean.h5"
            with h5py.File(fn, "r") as hf:
                self.datasets[int(Re)] = {
                    'f_outer': hf["f_outer"][:],
                    'f_inner': hf["f_inner"][:],
                    'pp_wall_plus': hf["pp_wall_plus"][:],
                }

    def inner_model(
        self,
        f: np.ndarray,
        A: float = 1.6,
        f0: float = 0.1,
        p_low: float = 1,
        p_high: float = -6,
        q: float = 3.5,
    ) -> np.ndarray:
        """
         φ_in(ω) = A · [W(ω)/W(ω0)], where
        W(ω) = (ω/ω0)^p_low · [1 + (ω/ω0)^q]^(-r)
        with the high-frequency condition:
        p_low - q·r = p_high  →  r = (p_low - p_high)/q.
        This guarantees that φ_in(ω0) = A.
        Conditions:
        1. Low-frequency behavior:  W(ω) ~ (ω/ω0)^p_low.
        2. High-frequency behavior: W(ω) ~ (ω/ω0)^(p_low - q*r) = (ω/ω0)^p_high.
        3. Sharpness of the transition: q

        These yield:
        r = -log2(A_match)
        q*r = p_low - p_high  →  q = (p_low - p_high) / r

        Note: The slope at ω0 becomes
            (d lnW/d lnω)|ω0 = p_low - (q*r)/2 = (p_low + p_high)/2,
        so the effective transition slope is fixed by p_low and p_high.
        """
        r = (p_low - p_high) / q
        # Normalize so that W(f0)=2^(-r), hence:
        phi_in = (
            A
            * (2**r)
            * ((f / f0) ** p_low * (1 + (f / f0) ** q) ** (-r))
        )
        return phi_in

    def outer_model(
        self,
        f: np.ndarray,
        A: float,
        p_high: float,
        q: float,
        f0: float = 2 / 3,
        p_low: float = 3,
    ) -> np.ndarray:
        """
        Outer model for wall-pressure fluctuations:
        φ_out(ω) = A · (2^r) · (ω/ω0)^p_low · [1 + (ω/ω0)^q]^(-r)
        with:
        r = (p_low - p_high) / q

        Parameters:
        f  : frequency array
        A      : amplitude scaling (ensures φ_out(ω0) = A)
        p_high : high-frequency exponent (so that asymptotically φ_out ~ ω^p_high)
        q      : controls transition sharpness
        f0 : fixed frequency scale (default = 1)
        p_low  : fixed low-frequency exponent (default = 6/5)

        Returns:
        φ_out evaluated at f.
        """
        r = (p_low - p_high) / q
        return (
            A
            * (2**r)
            * ((f / f0) ** p_low * (1 + (f / f0) ** q) ** (-r))
        )

    def fit_outer_model(
        self,
        f: np.ndarray,
        data: np.ndarray,
        initial_guess: tuple[float, float, float] = (1.0, -0.7, 0.5),
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Fit the outer_model to the provided data via nonlinear least squares.

        Parameters:
        f         : 1D numpy array of frequency values (independent variable)
        data          : 1D numpy array of measured φ_out values (dependent variable)
        f0        : fixed frequency scale
        p_low         : fixed low-frequency exponent
        initial_guess : tuple (A, p_high, q) for initial parameter guess

        Returns:
        popt : optimal parameters [A, p_high, q]
        pcov : covariance matrix of the fit
        mse  : mean squared error of the fit
        """
        # Define a lambda to fix f0 and p_low during fitting
        model_func = lambda om, A, p_high, q: self.outer_model(om, A, p_high, q)

        # Perform the curve fitting
        popt, pcov = curve_fit(model_func, f, data, p0=initial_guess)

        # Compute mean squared error
        fitted = self.outer_model(f, *popt)
        mse = np.mean((data - fitted) ** 2)

        return popt, pcov, mse
    
    def A_func(
        self,
        Re: float | np.ndarray,
        a1: float,
        a2: float,
        L: float = 1,
    ) -> np.ndarray:
        X = np.log10(Re)
        # a1 = 3.2
        # a2 = 1.4
        return L / (1 + np.exp(a1*(-X + a2)))

    def p_high_func(
        self,
        Re: float | np.ndarray,
        ph1: float,
        ph2: float,
        L: float = 1.5,
    ) -> np.ndarray:
        """
        Compute outer model parameter p_high as a sigmoid function of Reynolds number (Re).
        p_high(Re) transitions smoothly between two asymptotes (ph1 to ph2).
        """
        X = np.log10(Re)
        # ph1 = -2.9
        # ph2 = 0.5
        # lam = abs((ph1 - L) / ph2)
        return -2+(L) / (1 + np.exp(ph1*(-X + ph2)))

    def q_func(
        self,
        Re: float | np.ndarray,
        q1: float,
        q2: float,
        L: float = 0.6,
    ) -> np.ndarray:
        """
        Compute outer model parameter q as a function of Reynolds number (Re).
        Using a logarithmic dependency: q(Re) = q_const + q_slope * log(Re).
        """
        X = np.log10(Re)
        # lam = abs(q1 / q2)
        # q1 = -0.16
        # q2 = 0.2
        # q = L + ((q1 - L) + q2*X) / (1 + (X/lam)**2)
        return 0.2+L / (1 + np.exp(q1*(-X + q2)))


    def total_loss(self, params: np.ndarray) -> float:
        """Calculate the total loss (sum of squared errors) for the outer model across all datasets.
        This uses the Re-dependent A, p_high, q functions and ignores any inner model contributions."""
        # Unpack parameter vector (A: [coeff, exp], p_high: [const, slope], q: [const, slope])
        a1, a2, ph1, ph2, q1, q2 = params

        total_error = 0.0
        # Loop over each dataset (each with its own Re, f_outer, measured spectrum)
        for Re in self.re_tau[:]:
            f_outer = self.datasets[Re]['f_outer']
            f_inner = self.datasets[Re]['f_inner']
            pp_wall_plus = self.datasets[Re]['pp_wall_plus']
            data = pp_wall_plus * f_inner
            A_val = self.A_func(Re, a1, a2)
            p_high_val = self.p_high_func(Re, ph1, ph2)
            q_val = self.q_func(Re, q1, q2)
            # Predict the outer model spectrum (ignoring inner model)
            predicted = self.outer_model(
                f_outer, A_val, p_high_val, q_val
            )
            # Accumulate squared error for this dataset
            im = self.inner_model(f_inner)
            model = predicted + im
            diff = model - data
            total_error += np.sum(diff ** 2)

        Re_taus = np.logspace(2, 6, 256) # change to 1 for more exact variance at low_re
        f_m = np.logspace(-2, 9, 256)
        for Re in Re_taus:
            model = EvalPipeModel(Re, f_m, params=params)
            data = model.outer_model() + model.inner_model()

            i = np.argsort(model.f_inner)
            var = np.trapezoid(data[i]/model.f_inner[i], model.f_inner[i])
            theoretical = -9.18 + 2.24 * np.log(Re)
            total_error += (var - theoretical) ** 2/100

        return total_error
    

    def rms_form(self, Re: float | np.ndarray, params: np.ndarray) -> np.ndarray:
        return np.sqrt(params[0] + params[1] * np.log(Re / 333))

    def fit_theoretical_rms(self) -> np.ndarray:
        Re_list = []
        rms_list = []

        # 1. compute rms from HDF5 files at self.re_tau
        for key in self.re_tau:
            with h5py.File(f"data/dacome25/{key}_clean.h5", "r") as hf:
                f_inner = hf["f_inner"][:]
                pp_wall_plus = hf["pp_wall_plus"][:]
            i_sort = np.argsort(f_inner)
            var = np.trapezoid(pp_wall_plus[i_sort], f_inner[i_sort])
            rms = np.sqrt(var)
            Re_list.append(key)
            rms_list.append(rms)
        Re_array = np.array(Re_list, dtype=float)
        rms_array = np.array(rms_list, dtype=float)

        # 2. load published RMS data from CSV files
        for fn in ["P17","YU22"]:
            re_vals, rms_vals = np.genfromtxt(f"data/RMS/{fn}.csv", delimiter=",", unpack=True)
            Re_array = np.concatenate([Re_array, re_vals.flatten()])
            rms_array = np.concatenate([rms_array, rms_vals.flatten()])

        # 4. perform least‐squares fit
        initial_guess = np.array([6.5, 2], dtype=float)

        def residuals(params):
            return self.rms_form(Re_array, params) - rms_array

        result = least_squares(residuals, initial_guess)
        return result.x

    def fit_model(self) -> np.ndarray:
        """
        Fit the outer model parameters (A, p_high, q) across all datasets by minimizing the total loss.
        Only outer model parameters are fitted (inner model is ignored).
        """
        # 4.0644588 , 3.56929862, 0.97915764, 4.18458685, 0.172006282.44443864
        a_guess = np.array([4.0644588, 3.56929862], dtype=float)
        ph_guess = np.array([0.97915764, 4.18458685], dtype=float)
        q_guess = np.array([0.17200628, 2.44443864], dtype=float)
        initial_guess = np.concatenate((a_guess, ph_guess, q_guess))
        # return initial_guess
        result = minimize(self.total_loss, initial_guess, method='Nelder-Mead')
        ic(self.total_loss(result.x))
        ic(self.rms_loss(result.x))
        return result.x

    def save_params(self) -> None:
        np.save("data/model_params.npy", self.fit_model())
        
if __name__ == "__main__":
    dacome = DacomePipe()
    ic(dacome.fit_theoretical_rms())
    dacome.save_params()
    os.system("python src/plotting.py")
