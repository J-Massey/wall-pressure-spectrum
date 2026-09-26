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

from src.model_A import bl_model, pipe_model, channel_model, cf_approx


plt.style.use(["science", "grid", ])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")
sns.set_palette("colorblind")

def plot_3_bl():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.5, 2.), sharey=True, sharex=True, tight_layout=True, dpi=600)
    # Set labels and scales for the top row axes.
    # axl.set_xlabel(r"$T^+$")
    axl.set_ylabel(r"$f\phi_{pp}^+$")
    axl.set_xscale("log")

    # axr.set_xlabel(r"$T^+$")
    axr.set_xscale("log")
    labels = [r"$g_1$", r"$g_2$", "$g_1 + g_2$"]
    lines = [
        Line2D([], [], color="black", linestyle="--", label=labels[2]),
        Line2D([], [], color="black", linestyle=":", label=labels[0]),
        Line2D([], [], color="black", linestyle="-.", label=labels[1]),
    ]
    axr.legend(
        handles=lines,
        loc="upper right",
        fontsize=8,
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

    axl.set_xlim(1e0, 5e4)
    axl.set_xticklabels([])
    axl.set_ylim(0, 4)

    cell_colours = []
    cell_text = []
    for idf, fn in enumerate(fns_eitel):
        T_plus, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        axl.plot(T_plus, fphip, color=colors[idf])
        g1, g2, rv = bl_model(T_plus, Re_eitel[idf], cf_approx(Re_eitel[idf]))
        axl.plot(T_plus, rv * (g1 + g2), linestyle="--", color=colors[idf], lw=0.7)
        cell_colours.append(colors[idf])
        cell_text.append(rf"E14 ${Re_eitel[idf]}$")

    Re_Fritsch = [4021, 6654, 11064]
    cf_Fritsch = [2.578/1000, 0.00254, 0.00233]
    fns_fritsch = [
        f"fritsch{re}.csv" for re in Re_Fritsch
    ]
    f_colors = sns.color_palette("Greys", len(Re_Fritsch))
    for idf, fn in enumerate(fns_fritsch):
        T_plus, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        axl.plot(T_plus, fphip, color=f_colors[idf])
        g1, g2, rv = bl_model(T_plus, Re_Fritsch[idf], cf_Fritsch[idf]/2)
        axl.plot(T_plus, rv * (g1 + g2), linestyle="--", color=f_colors[idf], lw=0.7)
        cell_text.append(rf"F22 ${Re_Fritsch[idf]}$")
        cell_colours.append(f_colors[idf])

    T_plus, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fns_eitel[1]}", delimiter=" ", skip_header=1).T
    axr.plot(T_plus, fphip, color=colors[1])
    g1, g2, rv = bl_model(T_plus, Re_eitel[1], cf_approx(Re_eitel[1]))
    axr.plot(T_plus, rv * (g1), linestyle=":", color=colors[1], lw=0.7)
    axr.plot(T_plus, rv * (g2), linestyle="-.", color=colors[1], lw=0.7)
    axr.plot(T_plus, rv * (g1 + g2), linestyle="--", color=colors[1], lw=0.7)

    T_plus, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fns_fritsch[-1]}", delimiter=" ", skip_header=1).T
    axr.plot(T_plus, fphip, color=f_colors[-1])
    g1, g2, rv = bl_model(T_plus, Re_Fritsch[-1], cf_Fritsch[-1]/2)
    axr.plot(T_plus, rv * (g1), linestyle=":", color=f_colors[-1], lw=0.7)
    axr.plot(T_plus, rv * (g2), linestyle="-.", color=f_colors[-1], lw=0.7)
    axr.plot(T_plus, rv * (g1 + g2), linestyle="--", color=f_colors[-1], lw=0.7)



    tbl = table_ax.table(
        cellText=[cell_text], cellColours=[cell_colours], cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for cell in tbl.get_celld().values():
        cell.set_linewidth(1)
    plt.savefig(f"figures/lex_reformat/3_bl_model.pdf")
    plt.savefig(f"figures/lex_reformat/3_bl_model.png")
    plt.savefig(f"comms/jfm_v1/figures/figure4a.pdf")
    plt.savefig(f"comms/jfm_v1/figures/figure4a.png")
    plt.savefig(f"comms/jfm_v1/figures/figure4a.eps")
    plt.close()

def plot_3_pipe(model):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.5, 2.), sharey=True, sharex=True, tight_layout=True, dpi=600)
    # Set labels and scales for the top row axes.
    # axl.set_xlabel(r"$T^+$")
    axl.set_ylabel(r"$f\phi_{pp}^+$")
    axl.set_xscale("log")

    # axr.set_xlabel(r"$T^{\circ}$")
    axr.set_xscale("log")

    keys = model.metadata.keys()
    colours = sns.color_palette("Set2", len(keys))

    Re_pipe = [Re for Re in model.metadata.keys()]
    axl.set_xlim(1e0, 5e4)


    # labels = [r"$r_v g_1$", r"$r_v g_2$", "$r_v(g_1 + g_2)$"]
    # lines = [
    #     Line2D([], [], color="black", linestyle=":", label=labels[0]),
    #     Line2D([], [], color="black", linestyle="-.", label=labels[1]),
    #     Line2D([], [], color="black", linestyle="--", label=labels[2]),
    # ]
    # axr.legend(
    #     handles=lines,
    #     loc="upper right",
    #     fontsize=8,
    #     frameon=False,
    #     handlelength=1.5,
    #     handletextpad=0.5,
    # )

    for idp, Re in enumerate(Re_pipe):
        with h5py.File(f"data/dacome25/{Re}_clean.h5", "r") as hf:
            f = hf["f"][:].astype(float)  # dimensional frequency
            f_inner = hf["f_inner"][:].astype(float)  # inner-scaled frequency
            f_outer = hf["f_outer"][:].astype(float)  # outer-scaled frequency
            pp_wall_plus = hf["pp_wall_plus"][:].astype(float)
        data = pp_wall_plus * f_inner  # type: ignore

        # Plot on the top left axis (using inner-scaled frequency).
        axl.plot(
            1/f_inner,
            data,
            label=r"$\mathit{Re}=" + f"{Re}$",
            color=colours[idp],
        )
        g1, g2, rv = pipe_model(1/f_inner, Re, model.metadata[Re]["U_tau"], model.metadata[Re]["U_CL"])
        axl.plot(1/f_inner, rv * (g1 + g2), linestyle="--", color=colours[idp], lw=0.7)

        if idp == 0 or idp == len(Re_pipe) - 1:
            axr.plot(
                1/f_inner,
                data,
                label=r"$\mathit{Re}=" + f"{Re}$",
                color=colours[idp],
            )
            axr.plot(1/f_inner, rv * (g1 + g2), linestyle="--", color=colours[idp], lw=0.7)
            axr.plot(1/f_inner, rv * (g1), linestyle=":", color=colours[idp], lw=0.7)
            axr.plot(1/f_inner, rv * (g2), linestyle="-.", color=colours[idp], lw=0.7)

    axl.set_ylim(0, 4)
    axl.set_xlim(1e0, 5e4)
    axl.set_xticklabels([])

    table_ax = fig.add_axes([0.125, 0.97, 0.8, 0.06], frameon=False)
    table_ax.axis("off")
    cell_text = [[rf"D25 ${key}$" for key in keys]]
    cell_colours = [colours]
    tbl = table_ax.table(
        cellText=cell_text, cellColours=cell_colours, cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for cell in tbl.get_celld().values():
        cell.set_linewidth(1)
    plt.savefig(f"figures/lex_reformat/3_pipe_model.pdf")
    plt.savefig(f"figures/lex_reformat/3_pipe_model.png")
    plt.savefig(f"figures/fig_submission/figure4b.pdf")
    plt.savefig(f"figures/fig_submission/figure4b.png")
    plt.savefig(f"figures/fig_submission/figure4b.eps")
    plt.close()

def plot_3_channel():
    files = [
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R0180_pp.h5',
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R0550_pp.h5',
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R1000_pp.h5',
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R2000_pp.h5',
    '/home/masseyj/Workspace/SAPPHiRe/ReScalingPoisson/data/R5200_pp.h5',
    ]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=True, sharex=True, tight_layout=True, dpi=600)

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
        
        cell_text.append(rf"LM15 ${Re_tau:.0f}$")

        tau_w = utau**2
        kx_plus = kx /Re_tau  # (Nkx,)
        Tp = 2 * np.pi * utau / (0.8 * kx_plus)
        E_pp_plus = E_pp_wall.sum(axis=0)/(1.5*utau**4)  # (Nkx,) 1.5 is Lz/2pi
        
        axl.plot(Tp, kx * E_pp_plus, color=palette[id_re])
        g1, g2, rv = channel_model(Tp, Re_tau, utau, 1)
        axl.plot(Tp, rv * (g1 + g2), linestyle="--", color=palette[id_re])
        if id_re == 0 or id_re == len(files) - 1:
            axr.plot(Tp, kx * E_pp_plus, color=palette[id_re])
            axr.plot(Tp, rv * (g1 + g2), linestyle="--", color=palette[id_re], lw=0.7)
            axr.plot(Tp, rv * (g1), linestyle=":", color=palette[id_re], lw=0.7)
            axr.plot(Tp, rv * (g2), linestyle="-.", color=palette[id_re], lw=0.7)


    axl.set_xscale("log")
    axl.set_xlabel(r"$T^+$")
    axr.set_xlabel(r"$T^+$")
    axl.set_ylabel(r"$f\,\phi_{pp}^+$")

    axl.set_xlim(1e0, 5e4)
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

    fig.savefig(f"figures/lex_reformat/3_channel_model.pdf")
    fig.savefig(f"figures/lex_reformat/3_channel_model.png")
    fig.savefig(f"figures/fig_submission/figure4c.pdf")
    fig.savefig(f"figures/fig_submission/figure4c.png")
    fig.savefig(f"figures/fig_submission/figure4c.eps")
    plt.close()

if __name__ == "__main__":
    # plot_3_bl()
    # model = DacomePipe()
    # plot_3_pipe(model)
    plot_3_channel()

