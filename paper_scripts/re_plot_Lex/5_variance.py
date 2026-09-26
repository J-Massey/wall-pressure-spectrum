from turtle import color
import numpy as np
import h5py
import itertools
from typing import Tuple
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from icecream import ic
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

from src.train_model_B import DacomePipe
from src.model_A import bl_model, pipe_model, channel_model

plt.style.use(["science", "grid", ])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")
sns.set_palette("colorblind")

def cf_approx(Re_tau: float) -> float:
    return (np.log(Re_tau) / 0.384 + 4.127) ** -2

def Re_t(t_w, Cf, nu_u_tau, Ue, delta):
    u_tau = np.sqrt(Cf * Ue**2 / 2)  # (m s⁻¹)
    nu = nu_u_tau * u_tau
    Re_tau = u_tau * delta / nu
    return Re_tau

def goody_pressure_spectrum(Sh, Rt):
    c1, c2 = 0.5, 3
    c3 = 1.1 * Rt**-0.57
    Phi = c2 * Sh**2 / ((Sh**0.75 + c1) ** 3.7 + (c3 * Sh) ** 7)
    return Phi

def bl_var_eitel():
    Re_eitel = [500, 1000, 1500, 2000]
    fns_eitel = [
        f"eitel{re}.csv" for re in Re_eitel
    ]
    p_var = np.empty_like(Re_eitel, dtype=float)
    p_var_model = np.empty_like(Re_eitel, dtype=float)
    p_var_goody = np.empty_like(Re_eitel, dtype=float)
    for idf, fn in enumerate(fns_eitel):
        T_plus, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        p_var[idf] = np.trapezoid(np.abs(fphip * T_plus), 1/T_plus)
        g1, g2, rv = bl_model(T_plus, Re_eitel[idf], cf_approx(Re_eitel[idf]))
        fphi_model = rv * (g1 + g2)
        p_var_model[idf] = np.trapezoid(np.abs(fphi_model * T_plus), 1/T_plus)

        cf = cf_approx(Re_eitel[idf])
        Sh = 2 * np.pi * Re_eitel[idf] * np.sqrt(cf) / T_plus
        Rt = Re_eitel[idf] * np.sqrt(cf)
        Phi = goody_pressure_spectrum(Sh, Rt)
        p_var_goody[idf] = np.trapezoid(np.abs(Sh * Phi * T_plus), 1/T_plus)

    return Re_eitel, p_var, p_var_model, p_var_goody

def bl_var_fritsch():
    Re_Fritsch = [4021, 6654, 11064]
    cf_Fritsch = [2.578/1000, 0.00254, 0.00233]
    fns_fritsch = [
        f"fritsch{re}.csv" for re in Re_Fritsch
    ]
    p_var = np.empty_like(Re_Fritsch, dtype=float)
    p_var_model = np.empty_like(Re_Fritsch, dtype=float)
    p_var_goody = np.empty_like(Re_Fritsch, dtype=float)
    for idf, fn in enumerate(fns_fritsch):
        T_plus, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        p_var[idf] = np.trapezoid(np.abs(fphip * T_plus), 1/T_plus)

        g1, g2, rv = bl_model(T_plus, Re_Fritsch[idf], cf_Fritsch[idf]/2)
        fphi_model = rv * (g1 + g2)
        p_var_model[idf] = np.trapezoid(np.abs(fphi_model * T_plus), 1/T_plus)

        Sh = 2 * np.pi * Re_Fritsch[idf] * np.sqrt(cf_Fritsch[idf]/2) / T_plus
        Rt = Re_Fritsch[idf] * np.sqrt(cf_Fritsch[idf]/2)
        Phi = goody_pressure_spectrum(Sh, Rt)
        p_var_goody[idf] = np.trapezoid(np.abs(Sh * Phi * T_plus), 1/T_plus)
    return Re_Fritsch, p_var, p_var_model, p_var_goody

def pipe_var(model: DacomePipe):
    Re_pipe = [Re for Re in model.metadata.keys()]
    p_var = np.empty_like(Re_pipe, dtype=float)
    p_var_model = np.empty_like(Re_pipe, dtype=float)
    for idp, Re in enumerate(Re_pipe):
        with h5py.File(f"data/dacome25/{Re}_clean.h5", "r") as hf:
            f = hf["f"][:].astype(float)  # dimensional frequency
            f_inner = hf["f_inner"][:].astype(float)  # inner-scaled frequency
            f_outer = hf["f_outer"][:].astype(float)  # outer-scaled frequency
            pp_wall_plus = hf["pp_wall_plus"][:].astype(float)
        data = pp_wall_plus * f_inner  # type: ignore
        p_var[idp] = np.trapezoid(np.abs(data / f_inner), f_inner)

        g1, g2, rv = pipe_model(1/f_inner, Re, model.metadata[Re]["U_tau"], model.metadata[Re]["U_CL"])
        fphi_model = rv * (g1 + g2)
        p_var_model[idp] = np.trapezoid(np.abs(fphi_model / f_inner), f_inner)

    return Re_pipe, p_var, p_var_model

def channel_var():
    files = [
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R0180_pp.h5',
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R0550_pp.h5',
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R1000_pp.h5',
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R2000_pp.h5',
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R5200_pp.h5',
    ]
    Re_tau = np.empty_like(files, dtype=float)
    p_var = np.empty(len(files), dtype=float)
    p_var_model = np.empty(len(files), dtype=float)
    for id_re in range(len(files)):
        with h5py.File(files[id_re], "r") as f:
            kx = f["/kx"][1:]      # (Nkx,)
            y  = f["/y_loc"][:]     # (Ny,)
            utau = float(f["/u_tau"][()])
            nu   = float(f["/nu"][()])
            Re_tau[id_re]  = float(f["/Re_tau"][()])
            E_pp_wall = f["/E_pp_wall"][:, 1:]      # (Nkx, Nkz)


        tau_w = utau**2
        kx_plus = kx /Re_tau[id_re]  # (Nkx,)
        Tp = 2 * np.pi * utau / (0.8 * kx_plus)
        E_pp_plus = E_pp_wall.sum(axis=0)/(1.5*utau**4)  # (Nkx,) 1.5 is Lz/2pi
        fphi = kx * E_pp_plus
        p_var[id_re] = np.trapezoid(np.abs(fphi * Tp), 1/Tp)

        g1, g2, rv = channel_model(Tp, Re_tau[id_re], utau, 1)
        fphi_model = rv * (g1 + g2)
        p_var_model[id_re] = np.trapezoid(np.abs(fphi_model * Tp), 1/Tp)

    return Re_tau, p_var, p_var_model

def model_B_var(model: DacomePipe):
    re_range = (1e3, 5e5)
    Re_taus = np.logspace(np.log10(re_range[0]), np.log10(re_range[1]), 50)
    nu = 1.5e-5
    U_taus = Re_taus * nu
    # ax.set_yscale("log")
    a1, a2, ph1, ph2, q1, q2 = np.load("data/model_params.npy")
    f = np.logspace(-4, 10, 1000)  # from 0.1 to 100
    # Re_taus = np.array([i for i in model.metadata.keys()])
    U_taus = Re_taus * nu
    model_var = []
    for re_tau, U_tau in zip(Re_taus, U_taus):
        f_inner = f * nu / (U_tau**2)
        U = 1 * U_tau * (1 / 0.41 * np.log(re_tau) + 5.2)  # outer velocity scale
        f_outer = f / U

        a = model.A_func(re_tau, a1, a2)
        p = model.p_high_func(re_tau, ph1, ph2)
        q = model.q_func(re_tau, q1, q2)
        inner_p = model.inner_model(f_inner)
        outer_p = model.outer_model(f_outer, a, p, q)
        data = inner_p + outer_p
        # ax.plot(f_inner, data, lw=0.5, label=f"$Re_\\tau={re_tau:.0f}$", ls='--')

        i = np.argsort(f)
        var = np.trapezoid(data[i] / f_inner[i], f_inner[i])
        model_var.append(var)  # /np.sqrt(10)
    return Re_taus, np.array(model_var)


def plot_5_variance(model):
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5), tight_layout=True, dpi=600)
    # Set labels and scales for the top row axes.
    ax.set_xlabel(r"$\delta^+$")
    ax.set_ylabel(r"$\langle p^{+2} \rangle$")
    ax.set_xscale("log")

    ax.set_ylim(0, 24)
    ax.set_xlim(100, 5e5)
    labels = []
    lines = []

    Re_eitel, p_var, p_var_model, p_var_goody = bl_var_eitel()
    ax.plot(Re_eitel, np.abs(p_var), ls='', marker="^", color='blue')
    # ax.plot(Re_eitel, np.abs(p_var_model), ls='', marker="^", color='grey', markerfacecolor='none')
    ax.plot(Re_eitel, np.abs(p_var_goody), ls='', marker="^", color='black', markerfacecolor='red')

    Re_Fritsch, p_var, p_var_model, p_var_goody = bl_var_fritsch()
    ax.plot(Re_Fritsch, np.abs(p_var), ls='', marker="^", color='blue')
    # ax.plot(Re_Fritsch, np.abs(p_var_model), ls='', marker="^", color='grey', markerfacecolor='none')
    ax.plot(Re_Fritsch[:1], np.abs(p_var_goody[:1]), ls='', marker="^", color='black', markerfacecolor='red', label=r"BL--Goody Model")

    labels.append(r"BL")
    lines.append(Line2D([], [], marker="^", color="blue", linestyle="", label=labels[0]))
    labels.append(r"BL--Model A")
    lines.append(Line2D([], [], marker="^", color="grey", linestyle="", markerfacecolor='none', label=labels[1]))
    # labels.append(r"BL Goody")
    # lines.append(Line2D([], [], marker="x", color="black", linestyle="", label=labels[2]))
    leg1 = ax.legend(                   # multi‑column legend
           loc='upper left',
           frameon=True,
           fontsize=7,)
    ax.add_artist(leg1)

    Re_pipe, p_var, p_var_model = pipe_var(model)
    ax.plot(Re_pipe, np.abs(p_var), ls='', marker="o", color='green')
    # ax.plot(Re_pipe, np.abs(p_var_model), ls='', marker="o", color='grey', markerfacecolor='none')

    labels.append(r"Pipe")
    lines.append(Line2D([], [], marker="o", color="green", linestyle="", label=labels[3-1]))
    labels.append(r"Pipe--Model A")
    lines.append(Line2D([], [], marker="o", color="grey", linestyle="", markerfacecolor='none', label=labels[4-1]))

    Re_tau, p_var, p_var_model = channel_var()
    ax.plot(Re_tau, np.abs(p_var), ls='', marker="s", color='orange')
    # ax.plot(Re_tau, np.abs(p_var_model), ls='', marker="s", color='grey', markerfacecolor='none')

    labels.append(r"Channel")
    lines.append(Line2D([], [], marker="s", color="orange", linestyle="", label=labels[5-1]))
    labels.append(r"Channel--Model A")
    lines.append(Line2D([], [], marker="s", color="grey", linestyle="", markerfacecolor='none', label=labels[6-1]))


    fig.legend(lines, labels,
           ncol=3,                       # multi‑column legend
           loc='lower center',
           bbox_to_anchor=(0.55, 0.9),   # centred just above the axes
           frameon=False,
           fontsize=7,
           handlelength=1.5,
           handletextpad=0.5)
    # schlatter and Orlu 
    Re_samples = np.logspace(2, np.log10(5e5), 64)
    p_var_schlatter = 2.42 * np.log(Re_samples) - 8.96
    line_schlatter, = ax.plot(Re_samples, p_var_schlatter, color='black', linestyle='-', label=r"Schlatter and Örlü (2010)")
    # Lee & Moser
    p_var_lee_moser = 2.24 * np.log(Re_samples) - 9.18
    line_lee_moser, = ax.plot(Re_samples, p_var_lee_moser, color='black', linestyle='--', label=r"Lee and Moser (2015)")
    
    Re_B, model_var = model_B_var(model)
    # line_model_B, = ax.plot(Re_B, model_var, color='red', linestyle='-.', label=r"Prediction: Pipe Model B")
    # Only show these three in leg2
    # leg2 = ax.legend([line_schlatter, line_lee_moser, line_model_B],
    #                  [r"Schlatter and Örlü (2010)", r"Lee and Moser (2015)", r"Prediction: Pipe Model B"],
    #                  loc='lower right', fontsize=7)
    leg2 = ax.legend([line_schlatter, line_lee_moser],
                     [r"Schlatter and Örlü (2010)", r"Lee and Moser (2015)"],
                     loc='lower right', fontsize=7)

    # fig.tight_layout(rect=[0, 0, 0.8, 1.1])   # shrink plot area to leave top margin

    # plt.savefig(f"figures/APS/1_variance.pdf")
    plt.savefig(f"figures/APS/2_variance.pdf")
    # plt.savefig(f"figures/APS/3_variance.pdf")
    # plt.savefig(f"figures/lex_reformat/5_variance.pdf")
    # plt.savefig(f"figures/lex_reformat/5_variance.png")
    # plt.savefig(f"figures/fig_submission/figure6.pdf")
    # plt.savefig(f"figures/fig_submission/figure6.png")
    # plt.savefig(f"figures/fig_submission/figure6.eps")
    plt.close()

def plot_APS_variance_cartoon(model):
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), tight_layout=True, dpi=600)
    # Set labels and scales for the top row axes.
    ax.set_xlabel(r"$\delta^+T^+$")
    ax.set_ylabel(r"$\langle p^{+2} \rangle$")
    ax.set_xscale("log")

    ax.set_ylim(0, 24)
    ax.set_xlim(100, 5e5)
    labels = []
    lines = []

    Re_eitel, p_var, p_var_model, p_var_goody = bl_var_eitel()
    ax.plot(Re_eitel, np.abs(p_var), ls='', marker="^", color='blue')
    ax.plot(Re_eitel, np.abs(p_var_model), ls='', marker="^", color='grey', markerfacecolor='none')
    ax.plot(Re_eitel, np.abs(p_var_goody), ls='', marker="^", color='black', markerfacecolor='red')

    Re_Fritsch, p_var, p_var_model, p_var_goody = bl_var_fritsch()
    ax.plot(Re_Fritsch, np.abs(p_var), ls='', marker="^", color='blue')
    ax.plot(Re_Fritsch, np.abs(p_var_model), ls='', marker="^", color='grey', markerfacecolor='none')
    ax.plot(Re_Fritsch[:1], np.abs(p_var_goody[:1]), ls='', marker="^", color='black', markerfacecolor='red', label=r"BL--Goody Model")

    labels.append(r"BL")
    lines.append(Line2D([], [], marker="^", color="blue", linestyle="", label=labels[0]))
    labels.append(r"BL--Model A")
    lines.append(Line2D([], [], marker="^", color="grey", linestyle="", markerfacecolor='none', label=labels[1]))
    # labels.append(r"BL Goody")
    # lines.append(Line2D([], [], marker="x", color="black", linestyle="", label=labels[2]))
    leg1 = ax.legend(                   # multi‑column legend
           loc='upper left',
           frameon=True,
           fontsize=7,)
    ax.add_artist(leg1)

    Re_pipe, p_var, p_var_model = pipe_var(model)
    ax.plot(Re_pipe, np.abs(p_var), ls='', marker="o", color='green')
    ax.plot(Re_pipe, np.abs(p_var_model), ls='', marker="o", color='grey', markerfacecolor='none')

    labels.append(r"Pipe")
    lines.append(Line2D([], [], marker="o", color="green", linestyle="", label=labels[3-1]))
    labels.append(r"Pipe--Model A")
    lines.append(Line2D([], [], marker="o", color="grey", linestyle="", markerfacecolor='none', label=labels[4-1]))

    Re_tau, p_var, p_var_model = channel_var()
    ax.plot(Re_tau, np.abs(p_var), ls='', marker="s", color='orange')
    ax.plot(Re_tau, np.abs(p_var_model), ls='', marker="s", color='grey', markerfacecolor='none')

    labels.append(r"Channel")
    lines.append(Line2D([], [], marker="s", color="orange", linestyle="", label=labels[5-1]))
    labels.append(r"Channel--Model A")
    lines.append(Line2D([], [], marker="s", color="grey", linestyle="", markerfacecolor='none', label=labels[6-1]))


    fig.legend(lines, labels,
           ncol=3,                       # multi‑column legend
           loc='lower center',
           bbox_to_anchor=(0.55, 0.9),   # centred just above the axes
           frameon=False,
           fontsize=7,
           handlelength=1.5,
           handletextpad=0.5)
    # schlatter and Orlu 
    Re_samples = np.logspace(2, np.log10(5e5), 64)
    p_var_schlatter = 2.42 * np.log(Re_samples) - 8.96
    line_schlatter, = ax.plot(Re_samples, p_var_schlatter, color='black', linestyle='-', label=r"Schlatter and Örlü (2010)")
    # Lee & Moser
    p_var_lee_moser = 2.24 * np.log(Re_samples) - 9.18
    line_lee_moser, = ax.plot(Re_samples, p_var_lee_moser, color='black', linestyle='--', label=r"Lee and Moser (2015)")
    
    Re_B, model_var = model_B_var(model)
    line_model_B, = ax.plot(Re_B, model_var, color='red', linestyle='-.', label=r"Prediction: Pipe Model B")
    # Only show these three in leg2
    leg2 = ax.legend([line_schlatter, line_lee_moser, line_model_B],
                     [r"Schlatter and Örlü (2010)", r"Lee and Moser (2015)", r"Prediction: Pipe Model B"],
                     loc='lower right', fontsize=7)
    # leg2 = ax.legend([line_schlatter, line_lee_moser],
    #                  [r"Schlatter and Örlü (2010)", r"Lee and Moser (2015)"],
    #                  loc='lower right', fontsize=7)

    # fig.tight_layout(rect=[0, 0, 0.8, 1.1])   # shrink plot area to leave top margin

    # plt.savefig(f"figures/APS/1_variance.pdf")
    plt.savefig(f"figures/APS/cartoon_variance.pdf")
    # plt.savefig(f"figures/APS/3_variance.pdf")
    # plt.savefig(f"figures/lex_reformat/5_variance.pdf")
    # plt.savefig(f"figures/lex_reformat/5_variance.png")
    # plt.savefig(f"figures/fig_submission/figure6.pdf")
    # plt.savefig(f"figures/fig_submission/figure6.png")
    # plt.savefig(f"figures/fig_submission/figure6.eps")
    plt.close()


if __name__ == "__main__":
    model = DacomePipe()
    # plot_5_variance(model)
    plot_APS_variance_cartoon(model)

