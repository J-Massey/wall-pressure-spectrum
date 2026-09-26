import os
fns = [
    "1_data.py",
    "2_premultiply.py",
    "3_model.py",
    "4_Goody.py",
    "5_variance.py",
    "7_model_comparison.py",
    ]

for fn in fns:
    os.system(f"python -m src.re_plot_Lex.{fn}")