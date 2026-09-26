from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import UnivariateSpline
import pandas as pd

from icecream import ic

import h5py

class DacomePipeClean:
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

    @staticmethod
    def read_data(filename: os.PathLike | str) -> pd.DataFrame:
        data = pd.read_csv(filename)
        data = data.sort_values(by=data.columns[0])
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
        raw_data = pd.DataFrame({data.columns[0]: x, f"{data.columns[1]}": y})
        return raw_data

    @staticmethod
    def clean_data_spline(x: np.ndarray, y: np.ndarray, smoothing_factor: float = 3) -> np.ndarray:
        x_interp = np.logspace(np.log10(x[0]), np.log10(x[-1]), 100)
        y_interp = np.interp(x_interp, x, y)

        spline = UnivariateSpline(x_interp, y_interp, s=smoothing_factor)
        y_smooth = spline(x_interp)
        clean_data = np.array([x_interp, y_smooth]).T
        return clean_data

    def lambdaplus_to_f(self, lambxplus: np.ndarray, U_tau: float) -> np.ndarray:
        Uc_plus = 10
        f_inner = Uc_plus / lambxplus
        nu = 1.5 * 1e-5
        f = f_inner * U_tau**2 / nu
        return f

    def clean_data(self) -> None:
        for key in self.metadata.keys():
            U_tau = self.metadata[key]["U_tau"]
            tau_w = self.metadata[key]["tau_w"]
            U_cl = self.metadata[key]["U_CL"]
            nu = 1.5 * 1e-5
            R = 0.4505  # Pipe radius in meters

            filename = self.filenames[key]
            raw_data = self.read_data(filename)

            lambda_xplus_raw = np.array(raw_data.iloc[:, 0].values)
            phipp_plus_raw = np.array(raw_data.iloc[:, 1].values)

            # Clean the data
            clean_data = self.clean_data_spline(
                lambda_xplus_raw, phipp_plus_raw, smoothing_factor=1e-2
            )
            lambda_xplus_raw = clean_data[:, 0]
            phipp_plus_raw = clean_data[:, 1]

            f = self.lambdaplus_to_f(
                lambda_xplus_raw, U_tau
            )
            f_inner =  10 / lambda_xplus_raw
            f_outer = f * R / U_cl
            kx_plus = 2 * np.pi / lambda_xplus_raw
            kx = kx_plus * U_tau / nu

            pp_wall_plus = phipp_plus_raw / f_inner

            rho = tau_w / U_tau**2
            pp_wall = pp_wall_plus * (rho**2 * U_tau**2 * nu)

            clean = np.array([f_inner, f, pp_wall_plus, pp_wall]).T
            with h5py.File(
                f"data/dacome25/{key}_clean.h5", "w"
            ) as hf:
                hf.create_dataset("f", data=clean[:, 1])
                hf.create_dataset("f_inner", data=clean[:, 0])
                hf.create_dataset("f_outer", data=f_outer)
                hf.create_dataset("pp_wall_plus", data=clean[:, 2])
                hf.create_dataset("pp_wall", data=clean[:, 3])
    
    def outer_inner_correlation_grad(self) -> float:
        """
        Find the gradient of the correlated outer-scales
        """
        x = []
        y = []
        for Re in self.re_tau:
            with h5py.File(f"data/dacome25/{Re}_clean.h5", "r") as hf:
                f_outer = hf["f_outer"][:]
                f_inner = hf["f_inner"][:]
                pp_wall_plus = hf["pp_wall_plus"][:]
                data = pp_wall_plus * f_inner
            mask = np.logical_and(f_outer >= 1, f_outer <= 4)
            x.append(np.log10(f_outer[mask]))
            y.append(data[mask])
            # x.append(np.log10(f_inner[f_inner>=0.6]))
            # y.append(data[f_inner>=0.6])
        x = np.concatenate(x)
        y = np.concatenate(y)
        # Fit a linear model to the data
        slope, intercept = np.polyfit(x, y, 1)
        return slope
    
if __name__ == "__main__":
    dacome = DacomePipeClean()
    dacome.clean_data()
    dacome.outer_inner_correlation_grad()