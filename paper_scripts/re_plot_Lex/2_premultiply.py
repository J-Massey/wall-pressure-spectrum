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

plt.style.use(["science", "grid", ])
plt.rcParams["font.size"] = "10.5"
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")
sns.set_palette("colorblind")

def plot_2_a():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.5, 2.2), tight_layout=True, dpi=600)
    axl.set_xlabel(r"$f$ (Hz)")
    axr.set_xlabel(r"$T^+$")
    axl.set_ylabel(r"$\phi_{pp}(s)^+$")
    axr.set_ylabel(r"${f\phi_{pp}}^+$")
    axl.set_xscale("log")
    axl.set_yscale("log")
    axr.set_xscale("log")

    # axl.set_xlim(1e2, 1e4)
    axr.set_xlim(2, 5e4)

    Re_Fritsch = [4021, 6654, 11064]
    cf_Fritsch = [2.578/1000, 0.00254, 0.00233]
    fns_fritsch = [
        f"fritsch{re}.csv" for re in Re_Fritsch
    ]
    fns_f_fritsch = [
        f"f_fritsch{re}.csv" for re in Re_Fritsch
    ]

    table_ax = fig.add_axes([0.15, 0.97, 0.8, 0.06], frameon=False)
    table_ax.axis("off")
    cell_text = []
    cell_colours = []

    f_colors = sns.color_palette("Greys", len(Re_Fritsch)+1)[1:]
    for idf, fn in enumerate(fns_fritsch):
        T_plus, To, fphip = np.genfromtxt(f"data/lex_data/fig1/bl/{fn}", delimiter=" ", skip_header=1).T
        f = np.genfromtxt(f"data/lex_data/fig1/bl/{fns_f_fritsch[idf]}", delimiter=" ", skip_header=0).T
        # f = Re_Fritsch[idf] * np.sqrt(cf_Fritsch[idf]/2) / T_plus
        if idf == 0:
            axl.plot(f[1:], fphip/f[1:], color=f_colors[idf])
        else:
            axl.plot(f, fphip/f, color=f_colors[idf])
        axr.plot(T_plus, fphip, color=f_colors[idf])
        cell_text.append(rf"F22 ${Re_Fritsch[idf]}$")
        cell_colours.append(f_colors[idf])

    # T_plus, To, f, phi_pp = np.genfromtxt(f"data/lex_data/fig2/sltest.csv", delimiter=" ", skip_header=1).T
    # axl.plot(f, phi_pp, color="red", label=r"$\phi_{pp}$")
    # axr.plot(T_plus, f * phi_pp, color="red")
    # cell_text.append(rf"K08 $10^6$")
    # cell_colours.append("red")
    tbl = table_ax.table(
        cellText=[cell_text], cellColours=[cell_colours], cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for cell in tbl.get_celld().values():
        cell.set_linewidth(1)

    f_samples = np.logspace(2.7, 4, 64)
    axl.plot(f_samples, 6 * f_samples**-1, color="black", linestyle="--", lw=0.7)
    axl.annotate(
        r"$\phi_{pp}^+ \sim f^{-1}$",
        xy=(f_samples[0], 6 * f_samples[0]**-1),
        xytext=(f_samples[0] * 3, 1.7 * f_samples[0]**-1),
        rotation=-35,
        color="black",
        fontsize=8,
        ha="left",
        va="center",
    )
    plt.savefig(f"figures/lex_reformat/2_pre_mult.pdf")
    plt.savefig(f"figures/lex_reformat/2_pre_mult.png")

    plt.savefig(f"figures/fig_submission/figure3.pdf")
    plt.savefig(f"figures/fig_submission/figure3.png")
    plt.savefig(f"figures/fig_submission/figure3.eps")
    plt.savefig(f"comms/jfm_v1/figures/figure3.pdf")
    plt.savefig(f"comms/jfm_v1/figures/figure3.png")
    plt.savefig(f"comms/jfm_v1/figures/figure3.eps")

    plt.close()


if __name__ == "__main__":
    plot_2_a()