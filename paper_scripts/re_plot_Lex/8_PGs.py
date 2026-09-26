from turtle import color
import numpy as np
import h5py
import itertools
from typing import Tuple
from scipy.interpolate import make_interp_spline
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

def plot_8a():
    fns = ['cp0.csv', 'cp4.csv', 'cp10.csv']
    re_labels = [4021, 6654, 11064]
    aoa_labs = [0, -4, -10]
    fig, ax = plt.subplots(1, figsize=(4, 2.5), sharey=True, sharex=True, tight_layout=True, dpi=600)
    for fn in fns:
        x, y = np.genfromtxt(f"data/lex_data/fig8/{fn}", delimiter=" ", skip_header=1).T
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        ax.scatter(x, y, s=10)
        x_fit = np.linspace(x.min(), x.max(), 200)
        spline = make_interp_spline(x, y, k=min(3, len(x) - 1))
        y_fit = spline(x_fit)
        ax.plot(x_fit, y_fit, label=fr"${aoa_labs[fns.index(fn)]}^{{\circ}}$")
        # ax.annotate()

    ax.set_xlabel(r"$x$[m]")
    ax.set_ylabel(r"$C_p$")

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

    ax.set_xlim(0, 6)
    ax.set_ylim(-0.6, 0.2)

    ax.legend(title=r"$\alpha$", loc="lower left", frameon=True, fancybox=True, edgecolor="black", fontsize=8)
    plt.savefig(f"figures/fig_submission/figure8a.pdf")
    plt.savefig(f"figures/fig_submission/figure8a.eps")

plot_8a()

def plot_8b():
    fns = ['spec0.csv', 'spec10.csv']
    aoa_labs = [0, -10]
    fns2 = ['spec0h.csv', 'spec2h.csv', 'spec4h.csv', 'spec10h.csv']
    aoa_labs2 = [0, -2, -4, -10]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True, sharex=True, tight_layout=True, dpi=600)

    for fn in fns:
        x, y = np.genfromtxt(f"data/lex_data/fig8/{fn}", delimiter=" ", skip_header=1).T
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        axl.plot(x, y, label=fr"${aoa_labs[fns.index(fn)]}^{{\circ}}$")
    for fn in fns2:
        x, y = np.genfromtxt(f"data/lex_data/fig8/{fn}", delimiter=" ", skip_header=1).T
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        axr.plot(x, y, label=fr"${aoa_labs2[fns2.index(fn)]}^{{\circ}}$")

    axl.set_xlabel(r"$T^+$")
    axr.set_xlabel(r"$T^+$")
    axl.set_ylabel(r"$f{\phi_{pp}}^+$")

    axl.set_xlim(2e0, 5e4)
    axl.set_ylim(0, 4)

    axl.set_xscale("log")

    axl.annotate(r"$\delta^+=6650$", xy=(2e1, .5), xytext=(2e1, .5), textcoords="data", color="k")
    axr.annotate(r"$\delta^+\approx 11000$", xy=(2e1, .5), xytext=(2e1, .5), textcoords="data", color="k")

    axl.legend(title=r"$\alpha$", loc="upper right", frameon=True, fancybox=True, edgecolor="black", fontsize=8)
    axr.legend(title=r"$\alpha$", loc="upper right", frameon=True, fancybox=True, edgecolor="black", fontsize=8)
    plt.savefig(f"figures/fig_submission/figure8b.pdf")
    plt.savefig(f"figures/fig_submission/figure8b.eps")

plot_8b()
