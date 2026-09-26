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

def plot_1_bl(model):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.4, 2.5), sharey=True, tight_layout=True, dpi=600)
    # Set labels and scales for the top row axes.
    # axl.set_xlabel(r"$T^+$")
    axl.set_ylabel(r"${f\phi_{pp}}^+$")
    axl.set_xscale("log")

    # axr.set_xlabel(r"$T^{\circ}$")
    axr.set_xscale("log")

    axl.set_xlabel(r"$T^+$")
    axr.set_xlabel(r"$T^+$")

    llims_fritsch = [3.762, 13.92, 12.749]
    ulims_fritsch = [3762, 13920, 127490]
    lheights = [0.1, 0.3, 0.5]
    table_ax = fig.add_axes([0.07, 0.97, 0.905, 0.06], frameon=False)
    table_ax.axis("off")


    axl.set_xlim(2, 5e4)
    axr.set_xlim(2, 5e4)
    # axl.set_xticklabels([])
    # axr.set_xticklabels([])
    axl.set_ylim(0, 4)

    # for idf, fn in enumerate(fns):
    #     T, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
    #     axl.plot(T, fphip, color=colors[idf])
    #     axr.plot(1/To*(2*np.pi), fphip, color=colors[idf])
    cell_colours = []
    cell_text = []

    Re_Fritsch = [4021, 6654, 11064]
    fns = [
        f"fritsch{re}.csv" for re in Re_Fritsch
    ]
    f_colors = sns.color_palette("Greys", len(Re_Fritsch)+1)[1:]
    for idf, fn in enumerate(fns):
        axl.hlines(
            y=lheights[idf],
            xmin=llims_fritsch[idf],
            xmax=ulims_fritsch[idf],
            color=f_colors[idf],
            linestyles="--"
        )
        T, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        axl.plot(T, fphip, color=f_colors[idf])
        # axr.plot(To, fphip, color=f_colors[idf])
        cell_text.append(rf"F22 ${Re_Fritsch[idf]}$")
        cell_colours.append(f_colors[idf])
    ic(cell_colours)
    ic(cell_text)

    llim_dacome = [0.0432, 0.0964, 0.369, 0.984, 1.878, 2.75, 4.15]
    ulim_dacome = [173, 385, 1474, 3934, 7510, 11000, 16580]
    lheights = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]


    keys = model.metadata.keys()
    palette = sns.color_palette("Set2", len(keys))

    for key in keys:
        with h5py.File(f"data/dacome25/{key}_clean.h5", "r") as hf:
            f = hf["f"][:].astype(float)  # dimensional frequency
            f_inner = hf["f_inner"][:].astype(float)  # inner-scaled frequency
            f_outer = hf["f_outer"][:].astype(float)  # outer-scaled frequency
            pp_wall_plus = hf["pp_wall_plus"][:].astype(float)
        data = pp_wall_plus * f_inner  # type: ignore
        axr.hlines(
            y=lheights[list(keys).index(key)],
            xmin=llim_dacome[list(keys).index(key)],
            xmax=ulim_dacome[list(keys).index(key)],
            color=palette[list(keys).index(key)],
            linestyles="--"
        )

        # Plot on the top left axis (using inner-scaled frequency).
        axr.plot(
            1/f_inner,
            data,
            label=r"$\mathit{Re}=" + f"{key}$",
            color=palette[
                list(keys).index(key)
            ],
        )
        cell_text.append(rf"D25 ${key}$")
        cell_colours.append(palette[list(keys).index(key)])



    tbl = table_ax.table(
        cellText=[cell_text], cellColours=[cell_colours], cellLoc="center", loc="center"
    )
    # tbl.auto_set_font_size(False)
    # tbl.set_fontsize(6)
    for cell in tbl.get_celld().values():
        cell.set_linewidth(1)
    # plt.savefig(f"figures/lex_reformat/1_bl.pdf")
    # plt.savefig(f"figures/lex_reformat/1_bl.png")
    # plt.savefig(f"figures/lex_reformat/1_bl.eps")
    plt.savefig(f"figures/fig_submission/figure9.pdf")
    plt.savefig(f"figures/fig_submission/figure9.png")
    plt.savefig(f"figures/fig_submission/figure9.eps")
    # plt.savefig(f"comms/jfm_v1/figures/figure2a.pdf")
    # plt.savefig(f"comms/jfm_v1/figures/figure2a.png")
    # plt.savefig(f"comms/jfm_v1/figures/figure2a.eps")

    plt.close()


if __name__ == "__main__":
    model = DacomePipe()
    plot_1_bl(model)
    # plot_1_pipe(model)