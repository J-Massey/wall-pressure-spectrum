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

def plot_3_bl():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.5, 2.), sharey=True, sharex=True, tight_layout=True, dpi=600)
    # Set labels and scales for the top row axes.
    # axl.set_xlabel(r"$T^+$")
    axl.set_ylabel(r"${f\phi_{pp}}^+$")
    axl.set_xscale("log")

    # axr.set_xlabel(r"$T^+$")
    axr.set_xscale("log")
    labels = [r"$r_v g_1$", r"$r_v g_2$", "$r_v(g_1 + g_2)$"]
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
    plt.close()

def plot_7_pipe(model):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.5, 2.), tight_layout=True, dpi=600)
    # Set labels and scales for the top row axes.
    axl.set_xlabel(r"$T^+$")
    axr.set_xlabel(r"$\delta^+$")
    axl.set_ylabel(r"${f\phi_{pp}}^+$")
    axr.set_ylabel(r"$\mathrm{MSE}$")
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

    a1, a2, ph1, ph2, q1, q2 = np.load("data/model_params.npy")

    for idp, Re in enumerate(Re_pipe):
        with h5py.File(f"data/dacome25/{Re}_clean.h5", "r") as hf:
            f = hf["f"][:].astype(float)  # dimensional frequency
            f_inner = hf["f_inner"][:].astype(float)  # inner-scaled freque ncy
            f_outer = hf["f_outer"][:].astype(float)  # outer-scaled frequency
            pp_wall_plus = hf["pp_wall_plus"][:].astype(float)
        data = pp_wall_plus * f_inner  # type: ignore

        # Plot on the top left axis (using inner-scaled frequency).
        # axl.plot(
        #     1/f_inner,
        #     data,
        #     label=r"$\mathit{Re}=" + f"{Re}$",
        #     color=colours[idp],
        # )
        g1, g2, rv = pipe_model(1/f_inner, Re, model.metadata[Re]["U_tau"], model.metadata[Re]["U_CL"])
        # axl.plot(1/f_inner, rv * (g1 + g2), linestyle="--", color=colours[idp], lw=0.7)
        a = model.A_func(Re, a1, a2)
        p = model.p_high_func(Re, ph1, ph2)
        q = model.q_func(Re, q1, q2)
        inner_p = model.inner_model(f_inner)
        outer_p = model.outer_model(f_outer, a, p, q)

        diff_A = np.abs(data - rv * (g1 + g2))
        mse_A = np.mean(diff_A**2)
        diff_B = np.abs(data - (inner_p + outer_p))
        mse_B = np.mean(diff_B**2)
        axr.plot(Re, mse_A, marker="o", color='grey', markerfacecolor='none')
        axr.plot(Re, mse_B, marker="o", color='black', markerfacecolor='none')
        # axl.plot(f, outer_p, color="green", linestyle="--", lw=0.5)
        # axl.plot(f, inner_p, color="purple", linestyle="--", lw=0.5)
        # axl.plot(1/f_inner, inner_p+outer_p, color="black", linestyle="--", lw=0.75)

        if idp == 0 or idp == len(Re_pipe) - 1:
            axl.plot(
                1/f_inner,
                data,
                color=colours[idp],
            )
            axl.plot(1/f_inner, rv * (g1 + g2), linestyle="--", color=colours[idp], lw=0.7)
            # axl.plot(1/f_inner, rv * (g1), linestyle=":", color=colours[idp], lw=0.7)
            # axl.plot(1/f_inner, rv * (g2), linestyle="-.", color=colours[idp], lw=0.7)
            # axl.plot(1/f_inner, outer_p, color="green", linestyle="--", lw=0.5)
            # axl.plot(1/f_inner, inner_p, color="purple", linestyle="--", lw=0.5)
            axl.plot(1/f_inner, inner_p+outer_p, color=colours[idp], linestyle="-.", lw=0.75)

    l_labels = [rf"D25 ${Re_pipe[0]}$", rf"D25 ${Re_pipe[-1]}$", "Data", "Model A", "Model B"]
    lines = [
        Line2D([], [], color=colours[0], linestyle=":", label=l_labels[0]),
        Line2D([], [], color=colours[-1], linestyle="-.", label=l_labels[1]),
        Line2D([], [], color="black", linestyle="-", label=l_labels[2]),
        Line2D([], [], color="black", linestyle="--", label=l_labels[3]),
        Line2D([], [], color="black", linestyle="-.", label=l_labels[4]),
    ]
    axl.legend(
        handles=lines,
        loc="upper right",
        fontsize=7,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
    )
    axl.set_ylim(0, 4)
    axl.set_xlim(1e0, 5e4)

    r_labels = ["Pipe A", "Pipe B"]
    r_lines = [
        Line2D([], [], color="grey", marker="o", markerfacecolor='none', label=r_labels[0]),
        Line2D([], [], color="black", marker="o", markerfacecolor='none', label=r_labels[1]),
    ]
    axr.legend(
        handles=r_lines,
        loc="upper right",
        fontsize=7,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
    )

    # table_ax = fig.add_axes([0.125, 0.97, 0.8, 0.06], frameon=False)
    # table_ax.axis("off")
    # cell_text = [[rf"D25 ${key}$" for key in keys]]
    # cell_colours = [colours]
    # tbl = table_ax.table(
    #     cellText=cell_text, cellColours=cell_colours, cellLoc="center", loc="center"
    # )
    # tbl.auto_set_font_size(False)
    # tbl.set_fontsize(7)
    # for cell in tbl.get_celld().values():
    #     cell.set_linewidth(1)
    plt.savefig(f"figures/lex_reformat/7_pipe_model_comp.pdf")
    plt.savefig(f"figures/lex_reformat/7_pipe_model_comp.png")
    plt.savefig(f"figures/fig_submission/figure7.pdf")
    plt.savefig(f"figures/fig_submission/figure7.png")
    plt.savefig(f"figures/fig_submission/figure7.eps")
    plt.close()

def graphical_abs(model):
    fig, (axl) = plt.subplots(figsize=(3.2, 2), dpi=300)
    # Set labels and scales for the top row axes.
    axl.set_xlabel(r"$T^+$")
    # axr.set_xlabel(r"$\mathit{Re}_\tau$")
    axl.set_ylabel(r"${f\phi_{pp}}^+$")
    # axr.set_ylabel(r"$\mathrm{MSE}$")
    axl.set_xscale("log")

    # axr.set_xlabel(r"$T^{\circ}$")
    # axr.set_xscale("log")

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

    a1, a2, ph1, ph2, q1, q2 = np.load("data/model_params.npy")

    for idp, Re in enumerate(Re_pipe):
        with h5py.File(f"data/dacome25/{Re}_clean.h5", "r") as hf:
            f = hf["f"][:].astype(float)  # dimensional frequency
            f_inner = hf["f_inner"][:].astype(float)  # inner-scaled freque ncy
            f_outer = hf["f_outer"][:].astype(float)  # outer-scaled frequency
            pp_wall_plus = hf["pp_wall_plus"][:].astype(float)
        data = pp_wall_plus * f_inner  # type: ignore

        # Plot on the top left axis (using inner-scaled frequency).
        # axl.plot(
        #     1/f_inner,
        #     data,
        #     label=r"$\mathit{Re}=" + f"{Re}$",
        #     color=colours[idp],
        # )
        g1, g2, rv = pipe_model(1/f_inner, Re, model.metadata[Re]["U_tau"], model.metadata[Re]["U_CL"])
        # axl.plot(1/f_inner, rv * (g1 + g2), linestyle="--", color=colours[idp], lw=0.7)
        a = model.A_func(Re, a1, a2)
        p = model.p_high_func(Re, ph1, ph2)
        q = model.q_func(Re, q1, q2)
        inner_p = model.inner_model(f_inner)
        outer_p = model.outer_model(f_outer, a, p, q)

        diff_A = np.abs(data - rv * (g1 + g2))
        mse_A = np.mean(diff_A**2)
        diff_B = np.abs(data - (inner_p + outer_p))
        mse_B = np.mean(diff_B**2)
        # axr.plot(Re, mse_A, marker="o", color='grey', markerfacecolor='none')
        # axr.plot(Re, mse_B, marker="o", color='black', markerfacecolor='none')
        # axl.plot(f, outer_p, color="green", linestyle="--", lw=0.5)
        # axl.plot(f, inner_p, color="purple", linestyle="--", lw=0.5)
        # axl.plot(1/f_inner, inner_p+outer_p, color="black", linestyle="--", lw=0.75)

        if idp == 0 or idp == len(Re_pipe) - 1:
            axl.plot(
                1/f_inner,
                data,
                color=colours[idp],
            )
            axl.plot(1/f_inner, rv * (g1 + g2), linestyle="--", color=colours[idp], lw=0.7)
            # axl.plot(1/f_inner, rv * (g1), linestyle=":", color=colours[idp], lw=0.7)
            # axl.plot(1/f_inner, rv * (g2), linestyle="-.", color=colours[idp], lw=0.7)
            # axl.plot(1/f_inner, outer_p, color="green", linestyle="--", lw=0.5)
            # axl.plot(1/f_inner, inner_p, color="purple", linestyle="--", lw=0.5)
            axl.plot(1/f_inner, inner_p+outer_p, color=colours[idp], linestyle="-.", lw=0.75)

    l_labels = [rf"Pipe: $\delta^+={Re_pipe[0]:,.0f}$", rf"Pipe: $\delta^+={Re_pipe[-1]:,.0f}$", r"Dacome \textit{et al.} (2025)", "Model A", "Model B"]
    lines = [
        Line2D([], [], color=colours[0], linestyle="-", label=l_labels[0]),
        Line2D([], [], color=colours[-1], linestyle="-", label=l_labels[1]),
        Line2D([], [], color="black", linestyle="-", label=l_labels[2]),
        Line2D([], [], color="black", linestyle="--", label=l_labels[3]),
        Line2D([], [], color="black", linestyle="-.", label=l_labels[4]),
    ]
    axl.legend(
        handles=lines,
        loc="upper right",
        fontsize=6,
        frameon=True,
        handlelength=1.5,
        handletextpad=0.5,
    )
    axl.set_ylim(0, 4)
    axl.set_xlim(1e0, 5e4)
    # axl.axis('off')

    r_labels = ["Pipe A", "Pipe B"]
    r_lines = [
        Line2D([], [], color="grey", marker="o", markerfacecolor='none', label=r_labels[0]),
        Line2D([], [], color="black", marker="o", markerfacecolor='none', label=r_labels[1]),
    ]
    # axl.legend(
    #     handles=r_lines,
    #     loc="upper right",
    #     fontsize=7,
    #     frameon=True,
    #     handlelength=1.5,
    #     handletextpad=0.5,
    # )

    # table_ax = fig.add_axes([0.125, 0.97, 0.8, 0.06], frameon=False)
    # table_ax.axis("off")
    # cell_text = [[rf"D25 ${key}$" for key in keys]]
    # cell_colours = [colours]
    # tbl = table_ax.table(
    #     cellText=cell_text, cellColours=cell_colours, cellLoc="center", loc="center"
    # )
    # tbl.auto_set_font_size(False)
    # tbl.set_fontsize(7)
    # for cell in tbl.get_celld().values():
    #     cell.set_linewidth(1)
    # plt.savefig(f"figures/lex_reformat/7_pipe_model_comp.pdf")
    # plt.savefig(f"figures/lex_reformat/7_pipe_model_comp.png")
    # plt.savefig(f"figures/fig_submission/graphical_abstract.jpeg")
    plt.savefig(f"figures/APS/graphical_abstract.pdf", dpi=600)
    plt.close()



if __name__ == "__main__":
    model = DacomePipe()
    # plot_7_pipe(model)
    graphical_abs(model)

