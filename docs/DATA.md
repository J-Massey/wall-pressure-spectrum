# Data sources

The datasets used to calibrate and validate the models are **not** redistributed here.
Obtain them from the original sources:

| Flow | Dataset | Range of Re_tau | Source |
|---|---|---|---|
| Boundary layer (LES) | Eitel-Amor, Örlü & Schlatter (2014), *Int. J. Heat Fluid Flow* 47, 57–69 | 500–2000 | Contact authors / paper |
| Boundary layer (experiment) | Fritsch et al. (2020, 2022): *Surface Pressure Spectra Beneath High Reynolds Number Smooth and Rough Wall Boundary Layers in Pressure Gradients* | 4021–11 064 | Virginia Tech data repository, <https://doi.org/10.7294/20457189> |
| Pipe (experiment, CICLoPE) | Dacome, Lazzarini, Talamelli, Bellani & Baars (2025), *J. Fluid Mech.* 1013, A48 | 4794–47 015 | Digitised from the paper's figures (see below) |
| Channel (DNS) | Lee & Moser (2015), *J. Fluid Mech.* 774, 395–415 | 180–5186 | <https://turbulence.oden.utexas.edu/channel2015/content/README_2015.html> |

Variance/RMS collation (Fig. 6) additionally draws on published wall-pressure r.m.s. trends
(Schlatter & Örlü 2010; Panton, Lee & Moser 2017; Yu et al. 2022 and others cited in the paper).

**Pipe data.** The pipe spectra were reconstructed from figures in Dacome et al. (2025)
rather than taken from a public data file. `paper_scripts/clean_dacome.py` shows how the digitised
curves were smoothed and converted (lambda_x+ → f with U_c+ = 10, nu = 1.5e-5 m²/s).
The CICLoPE pipe radius is R = 0.4505 m; the tabulated flow conditions are in
`paper_scripts/train_model_B.py` (`DacomePipe.metadata`).

Channel pre-multiplied spectra were formed from the LM15 2-D pressure spectra by summing over
k_z and converting k_x to a period with a fixed U_c+ = 10 (see `paper_scripts/re_plot_Lex/3_model.py`).
