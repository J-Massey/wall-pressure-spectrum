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
from src.train_model_B import DacomePipe

plt.style.use(["science", "grid", ])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")
sns.set_palette("colorblind")

def plot_1_bl():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.4, 2.), sharey=True, tight_layout=True, dpi=600)
    # Set labels and scales for the top row axes.
    # axl.set_xlabel(r"$T^+$")
    axl.set_ylabel(r"${f\phi_{pp}}^+$")
    axl.set_xscale("log")

    # axr.set_xlabel(r"$T^{\circ}$")
    axr.set_xscale("log")

    # axl.set_xlabel(r"$T^+ = u_\tau^2 / (f \nu)$")
    # axr.set_xlabel(r"$T^{\circ} = U_e / (f \delta)$")

    Re_eitel = [500, 1000, 1500, 2000]
    colors = sns.color_palette("Set1", len(Re_eitel))
    table_ax = fig.add_axes([0.125, 0.97, 0.8, 0.06], frameon=False)
    table_ax.axis("off")

    fns = [
        f"eitel{re}.csv" for re in Re_eitel
    ]

    axl.set_xlim(2e0, 5e4)
    axr.set_xlim(2e-3, 1e2)
    # axl.set_xticklabels([])
    # axr.set_xticklabels([])
    axl.set_ylim(0, 4)

    for idf, fn in enumerate(fns):
        T, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        axl.plot(T, fphip, color=colors[idf])
        axr.plot(1/To*(2*np.pi), fphip, color=colors[idf])
    cell_colours = colors
    cell_text = [rf"E14 ${key}$" for key in Re_eitel]

    Re_Fritsch = [4021, 6654, 11064]
    fns = [
        f"fritsch{re}.csv" for re in Re_Fritsch
    ]
    f_colors = sns.color_palette("Greys", len(Re_Fritsch)+1)[1:]
    for idf, fn in enumerate(fns):
        T, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        axl.plot(T, fphip, color=f_colors[idf])
        axr.plot(To, fphip, color=f_colors[idf])
        cell_text.append(rf"F22 ${Re_Fritsch[idf]:,.0f}$")
        cell_colours.append(f_colors[idf])
    ic(cell_colours)
    ic(cell_text)


    tbl = table_ax.table(
        cellText=[cell_text], cellColours=[cell_colours], cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for cell in tbl.get_celld().values():
        cell.set_linewidth(1)
    # plt.savefig(f"figures/APS/bl_data.pdf")
    # plt.savefig(f"figures/lex_reformat/1_bl.pdf")
    # plt.savefig(f"figures/lex_reformat/1_bl.png")
    # plt.savefig(f"figures/lex_reformat/1_bl.eps")
    plt.savefig(f"figures/fig_submission/figure2a.pdf")
    plt.savefig(f"figures/fig_submission/figure2a.png")
    plt.savefig(f"figures/fig_submission/figure2a.eps")
    # plt.savefig(f"comms/jfm_v1/figures/figure2a.pdf")
    # plt.savefig(f"comms/jfm_v1/figures/figure2a.png")
    # plt.savefig(f"comms/jfm_v1/figures/figure2a.eps")

    plt.close()

def plot_1_pipe(model):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.4, 2.), sharey=True, tight_layout=True, dpi=600)
    # Set labels and scales for the top row axes.
    # axl.set_xlabel(r"$T^+$")
    axl.set_ylabel(r"${f\phi_{pp}}^+$")
    axl.set_xscale("log")

    # axr.set_xlabel(r"$T^{\circ}$")
    axr.set_xscale("log")

    keys = model.metadata.keys()
    palette = sns.color_palette("Set2", len(keys))

    # Loop over keys to plot your data.

    for key in model.metadata.keys():
        with h5py.File(f"data/dacome25/{key}_clean.h5", "r") as hf:
            f = hf["f"][:].astype(float)  # dimensional frequency
            f_inner = hf["f_inner"][:].astype(float)  # inner-scaled frequency
            f_outer = hf["f_outer"][:].astype(float)  # outer-scaled frequency
            pp_wall_plus = hf["pp_wall_plus"][:].astype(float)
        data = pp_wall_plus * f_inner  # type: ignore

        # Plot on the top left axis (using inner-scaled frequency).
        axl.plot(
            1/f_inner,
            data,
            label=r"$\mathit{Re}=" + f"{key}$",
            color=palette[
                list(model.metadata.keys()).index(key)
            ],
        )
        # Plot on the top right axis (using outer-scaled frequency).
        axr.plot(
            1/f_outer,
            data,
            color=palette[
                list(model.metadata.keys()).index(key)
            ],
        )
    axl.set_ylim(0, 4)
    axl.set_xlim(2e0, 5e4)
    axr.set_xlim(2e-3, 1e2)
    axl.set_xticklabels([])
    axr.set_xticklabels([])

    table_ax = fig.add_axes([0.125, 0.97, 0.8, 0.06], frameon=False)
    table_ax.axis("off")
    cell_text = [[rf"D25 ${key:,.0f}$" for key in keys]]
    cell_colours = [palette]
    tbl = table_ax.table(
        cellText=cell_text, cellColours=cell_colours, cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for cell in tbl.get_celld().values():
        cell.set_linewidth(1)
    plt.savefig(f"figures/lex_reformat/1_pipe.pdf")
    plt.savefig(f"figures/fig_submission/figure2b.pdf")
    plt.savefig(f"figures/fig_submission/figure2b.png")
    plt.savefig(f"figures/fig_submission/figure2b.eps")
    plt.close()

def plot_1_channel():
    files = [
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R0180_pp.h5',
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R0550_pp.h5',
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R1000_pp.h5',
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R2000_pp.h5',
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R5200_pp.h5',
    ]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True, tight_layout=True, dpi=600)

    cell_text = []
    palette = sns.color_palette("Set3", len(files)+2)[2:]
    for id_re in range(len(files)):
        with h5py.File(files[id_re], "r") as f:
            kx = f["/kx"][:]      # (Nkx,)
            y  = f["/y_loc"][:]     # (Ny,)
            utau = float(f["/u_tau"][()])
            nu   = float(f["/nu"][()])
            Re_tau  = float(f["/Re_tau"][()])
            E_pp_wall = f["/E_pp_wall"][:]      # (Nkx, Nkz)
        
        cell_text.append(rf"LM15 ${Re_tau:,.0f}$")

        tau_w = utau**2
        kx_plus = kx /Re_tau  # (Nkx,)
        Tp = 2 * np.pi * utau / (0.8 * kx_plus)
        E_pp_plus = E_pp_wall.sum(axis=0)/(1.5*utau**4)  # (Nkx,) 1.5 is Lz/2pi
        ic((kx * E_pp_plus).max())
        To = Tp / (Re_tau * utau)

        axl.plot(Tp, kx * E_pp_plus, color=palette[id_re])
        axr.plot(To, kx * E_pp_plus, color=palette[id_re])
    axl.set_xscale("log")
    axr.set_xscale("log")
    axl.set_xlabel(r"$T^+ = u_\tau^2 / (f \nu)$")
    axr.set_xlabel(r"$T^{\circ} = U_e / (f \delta)$")
    axl.set_ylabel(r"${f\phi_{pp}}^+$")


    axl.set_xlim(2e0, 5e4)
    axr.set_xlim(2e-3, 1e2)
    axl.set_ylim(0, 4)

    table_ax = fig.add_axes([0.125, 0.97, 0.8, 0.06], frameon=False)
    table_ax.axis("off")
    cell_colours = [palette]
    tbl = table_ax.table(
        cellText=[cell_text], cellColours=cell_colours, cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for cell in tbl.get_celld().values():
        cell.set_linewidth(1)

    fig.savefig(f"figures/lex_reformat/1_channel.pdf")
    fig.savefig(f"figures/lex_reformat/1_channel.png")
    fig.savefig(f"figures/fig_submission/figure2c.pdf")
    fig.savefig(f"figures/fig_submission/figure2c.png")
    fig.savefig(f"figures/fig_submission/figure2c.eps")

if __name__ == "__main__":
    plot_1_bl()
    model = DacomePipe()
    plot_1_pipe(model)
    plot_1_channel()