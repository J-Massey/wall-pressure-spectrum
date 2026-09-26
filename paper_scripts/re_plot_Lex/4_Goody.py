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

plt.style.use(["science", "grid", ])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")
sns.set_palette("colorblind")

def cf_approx(Re_tau: float) -> float:
    return (np.log(Re_tau) / 0.384 + 4.127) ** -2

def bl_model(Tplus, Re_tau: float, cf_2: float) -> np.ndarray:
    r1 = 0.5
    r2 = 7
    rv = np.exp(r1 * Tplus)/(np.exp(r1*r2) + np.exp(r1 * Tplus)) # correct
    rv = np.nan_to_num(rv, nan=1)  # replace NaNs with 0
    A1 = 2.2
    sig1 = 2.9
    sig2 = 1.2
    mean_Tplus = 20
    mean_To = 0.8
    mean_To_plus = mean_To * Re_tau * np.sqrt(cf_2)
    A2 = 1.4 * (np.log10(Re_tau) - 2.2)
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus))**2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus))**2)
    # ic(g1[0], Tplus[0], rv[0])
    return g1, g2, rv

# def goody_model():

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

def plot_4_Gbl():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=False, sharex=True, tight_layout=True, dpi=600)
    # Set labels and scales for the top row axes.
    axl.set_xlabel(r"$T^+$")
    axl.set_ylabel(r"${f\phi_{pp}}^+$")
    axl.set_xscale("log")

    axr.set_xlabel(r"$T^+$")
    axr.set_xscale("log")
    labels = [r"Goody (2004)"]
    lines = [
        Line2D([], [], color="black", linestyle=":", label=labels[0]),
    ]
    axl.legend(
        handles=lines,
        loc="upper right",
        fontsize=9,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
    )

    Re_eitel = [500, 1000, 1500, 2000]
    colors = sns.color_palette("Set1", len(Re_eitel))
    table_ax = fig.add_axes([0.125, 0.97, 0.8, 0.06], frameon=False)
    table_ax.axis("off")

    fns_eitel = [
        f"eitel{re}.csv" for re in Re_eitel
    ]

    axl.set_xlim(2e0, 5e4)
    axr.set_xlim(2e0, 5e4)
    axr.set_ylim(0, 4)
    axl.set_ylim(0, 8)

    cell_colours = []
    cell_text = []
    for idf, fn in enumerate(fns_eitel):
        T_plus, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        axl.plot(T_plus, fphip, color=colors[idf])
        axr.plot(T_plus, fphip, color=colors[idf])

        cf = cf_approx(Re_eitel[idf])
        Sh = 2 * np.pi * Re_eitel[idf] * np.sqrt(cf) / T_plus
        Rt = Re_eitel[idf] * np.sqrt(cf)
        Phi = goody_pressure_spectrum(Sh, Rt)
        axl.plot(T_plus, Sh * Phi, color=colors[idf], linestyle=":", lw=0.7)

        norm_fact = 1 / (Sh * Phi).max()
        axr.plot(T_plus, norm_fact * (Sh * Phi)*3.36, color=colors[idf], linestyle=":", lw=0.7)
        
        cell_colours.append(colors[idf])
        cell_text.append(rf"E14 ${Re_eitel[idf]}$")

    Re_Fritsch = [4021, 6654, 11064]
    cf_Fritsch = [2.578/1000, 0.00254, 0.00233]
    fns_fritsch = [
        f"fritsch{re}.csv" for re in Re_Fritsch
    ]
    f_colors = sns.color_palette("Greys", len(Re_Fritsch)+1)[1:]
    for idf, fn in enumerate(fns_fritsch):
        T_plus, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        axl.plot(T_plus, fphip, color=f_colors[idf])
        axr.plot(T_plus, fphip, color=f_colors[idf])
        Sh = 2 * np.pi * Re_Fritsch[idf] * np.sqrt(cf_Fritsch[idf]/2) / T_plus
        Rt = Re_Fritsch[idf] * np.sqrt(cf_Fritsch[idf]/2)
        Phi = goody_pressure_spectrum(Sh, Rt)
        axl.plot(T_plus, Sh * Phi, color=f_colors[idf], linestyle=":", lw=0.7)
        norm_fact = 1 / (Sh * Phi).max()
        axr.plot(T_plus, norm_fact * (Sh * Phi)*3.36, color=f_colors[idf], linestyle=":", lw=0.7)
        
        cell_text.append(rf"F22 ${Re_Fritsch[idf]}$")
        cell_colours.append(f_colors[idf])

    tbl = table_ax.table(
        cellText=[cell_text], cellColours=[cell_colours], cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for cell in tbl.get_celld().values():
        cell.set_linewidth(1)
    plt.savefig(f"figures/lex_reformat/4_bl_Gmodel.pdf")
    plt.savefig(f"figures/lex_reformat/4_bl_Gmodel.png")
    plt.savefig(f"figures/fig_submission/figure1.pdf")
    plt.savefig(f"figures/fig_submission/figure1.png")
    plt.savefig(f"figures/fig_submission/figure1.eps")
    # plt.savefig(f"comms/jfm_v1/figures/figure1.pdf")
    # plt.savefig(f"comms/jfm_v1/figures/figure1.png")
    # plt.savefig(f"comms/jfm_v1/figures/figure1.eps")

    plt.close()


def plot_G_intro():
    fig, axr = plt.subplots(1, 1, figsize=(3, 2.5), sharey=False, sharex=True, tight_layout=True, dpi=600)
    # Set labels and scales for the top row axes.
    axr.set_xlabel(r"$T^+$")
    axr.set_ylabel(r"${f\phi_{pp}}^+$")
    axr.set_xscale("log")

    axr.set_xlabel(r"$T^+= u_\tau^2 / (f \nu)$")
    axr.set_xscale("log")
    labels = [r"Goody (2004)"]
    lines = [
        Line2D([], [], color="black", linestyle=":", label=labels[0]),
    ]
    axr.legend(
        handles=lines,
        loc="upper right",
        fontsize=9,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
    )

    Re_eitel = [500, 1000]#, 1500, 2000]
    colors = sns.color_palette("Set1", len(Re_eitel))
    table_ax = fig.add_axes([0.125, 0.97, 0.8, 0.06], frameon=False)
    table_ax.axis("off")

    fns_eitel = [
        f"eitel{re}.csv" for re in Re_eitel
    ]

    axr.set_xlim(1e0, 1e4)
    axr.set_ylim(0, 5)
    # axl.set_ylim(0, 4)

    cell_colours = []
    cell_text = []
    for idf, fn in enumerate(fns_eitel):
        T_plus, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        # axl.plot(T_plus, fphip, color=colors[idf])
        axr.plot(T_plus, fphip, color=colors[idf])

        cf = cf_approx(Re_eitel[idf])
        Sh = 2 * np.pi * Re_eitel[idf] * np.sqrt(cf) / T_plus
        Rt = Re_eitel[idf] * np.sqrt(cf)
        Phi = goody_pressure_spectrum(Sh, Rt)
        axr.plot(T_plus, Sh * Phi, color=colors[idf], linestyle=":", lw=0.7)
        
        cell_colours.append(colors[idf])
        cell_text.append(rf"E14 ${Re_eitel[idf]:,.0f}$")

    tbl = table_ax.table(
        cellText=[cell_text], cellColours=[cell_colours], cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for cell in tbl.get_celld().values():
        cell.set_linewidth(1)
    plt.savefig(f"figures/APS/1_Goody.pdf")

    plt.close()

def plot_G_wrong():
    fig, axr = plt.subplots(1, 1, figsize=(4.5, 2.5), sharey=False, sharex=True, tight_layout=True, dpi=600)
    # Set labels and scales for the top row axes.
    axr.set_xlabel(r"$T^+= u_\tau^2 / (f \nu)$")
    axr.set_ylabel(r"${f\phi_{pp}}^+$")
    axr.set_xscale("log")

    axr.set_xscale("log")
    labels = [r"Goody (2004)"]
    lines = [
        Line2D([], [], color="black", linestyle=":", label=labels[0]),
    ]
    axr.legend(
        handles=lines,
        loc="upper right",
        fontsize=9,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
    )

    Re_eitel = [500, 1000, 1500, 2000]
    colors = sns.color_palette("Set1", len(Re_eitel))
    table_ax = fig.add_axes([0.125, 0.97, 0.8, 0.06], frameon=False)
    table_ax.axis("off")

    fns_eitel = [
        f"eitel{re}.csv" for re in Re_eitel
    ]

    axr.set_xlim(1e0, 1e4)
    axr.set_ylim(0, 7)
    # axl.set_ylim(0, 4)

    cell_colours = []
    cell_text = []
    for idf, fn in enumerate(fns_eitel):
        T_plus, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        # axl.plot(T_plus, fphip, color=colors[idf])
        axr.plot(T_plus, fphip, color=colors[idf])

        cf = cf_approx(Re_eitel[idf])
        Sh = 2 * np.pi * Re_eitel[idf] * np.sqrt(cf) / T_plus
        Rt = Re_eitel[idf] * np.sqrt(cf)
        Phi = goody_pressure_spectrum(Sh, Rt)
        axr.plot(T_plus, Sh * Phi, color=colors[idf], linestyle=":", lw=0.7)
        
        cell_colours.append(colors[idf])
        cell_text.append(rf"E14 ${Re_eitel[idf]:,.0f}$")

    Re_Fritsch = [4021, 6654, 11064]
    cf_Fritsch = [2.578/1000, 0.00254, 0.00233]
    fns_fritsch = [
        f"fritsch{re}.csv" for re in Re_Fritsch
    ]
    f_colors = sns.color_palette("Greys", len(Re_Fritsch)+1)[1:]
    for idf, fn in enumerate(fns_fritsch):
        T_plus, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        # axl.plot(T_plus, fphip, color=f_colors[idf])
        axr.plot(T_plus, fphip, color=f_colors[idf])
        Sh = 2 * np.pi * Re_Fritsch[idf] * np.sqrt(cf_Fritsch[idf]/2) / T_plus
        Rt = Re_Fritsch[idf] * np.sqrt(cf_Fritsch[idf]/2)
        Phi = goody_pressure_spectrum(Sh, Rt)
        axr.plot(T_plus, Sh * Phi, color=f_colors[idf], linestyle=":", lw=0.7)
        norm_fact = 1 / (Sh * Phi).max()
        # axr.plot(T_plus, norm_fact * (Sh * Phi)*3.36, color=f_colors[idf], linestyle=":", lw=0.7)
        
        cell_text.append(rf"F22 ${Re_Fritsch[idf]:,.0f}$")
        cell_colours.append(f_colors[idf])

    tbl = table_ax.table(
        cellText=[cell_text], cellColours=[cell_colours], cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for cell in tbl.get_celld().values():
        cell.set_linewidth(1)
    plt.savefig(f"figures/APS/2vi_Goody.pdf")

    plt.close()


if __name__ == "__main__":
    # plot_G_intro()
    # plot_G_wrong()
    plot_4_Gbl()

