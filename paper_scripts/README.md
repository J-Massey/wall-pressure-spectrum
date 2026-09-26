# Archived paper scripts

These are the original, unedited scripts used to fit the models and generate the paper's figures.
They are kept for provenance and reproducibility, **not** as a supported API — use the `wallpressure`
package instead. They assume the data listed in `../docs/DATA.md` laid out under `data/`, run from the
repo root as `python -m paper_scripts.re_plot_Lex.<n>_<name>` after fixing imports (they were written
as `src.*`), need LaTeX for figure text, and contain some absolute paths from the original machine.

* `model_A.py` – original model A (the package version is regression-tested against it)
* `inference_model_B.py`, `train_model_B.py` – model B and its Nelder–Mead fit (`fit_model`)
* `clean_dacome.py` – smoothing/conversion of digitised pipe spectra
* `re_plot_Lex/` – figure scripts
