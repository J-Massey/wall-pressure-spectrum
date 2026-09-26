"""Reproduce the style of the paper's model curves (no data required)."""
import matplotlib.pyplot as plt
import numpy as np
import wallpressure as wp

fig, ax = plt.subplots(1, 3, figsize=(11, 3), sharey=True, constrained_layout=True)
T = np.logspace(0, 5, 500)
res_bl = [500, 1000, 4000, 11000]
res_pipe = [4794, 14004, 31614, 47015]

for re in res_bl:  # boundary layer, model A
    ax[0].semilogx(T, wp.model_a(T, re, "boundary_layer"), label=rf"$\delta^+={re}$")
ax[0].set_title("BL, model A")

for re in res_pipe:  # pipe, model A (U_cl+ ~ 25)
    ax[1].semilogx(T, wp.model_a(T, re, "pipe", 1 / 25), label=rf"$\delta^+={re}$")
ax[1].set_title("Pipe, model A ($U_{CL}^+=25$)")

for re in res_pipe:  # pipe, model B
    ax[2].semilogx(T, wp.model_b(1 / T, re, 1 / 25), label=rf"$\delta^+={re}$")
ax[2].set_title("Pipe, model B ($U_{CL}^+=25$)")

for a in ax:
    a.set_xlabel(r"$T^+$")
    a.grid(alpha=0.3)
ax[0].set_ylabel(r"$f\phi_{pp}/\tau_w^2$")
ax[0].legend(frameon=False, fontsize=8)
fig.savefig("docs/spectra.png", dpi=200)
