import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
torch.set_num_threads(1)
from optimizers.bo import fit_model

rows = []
for tag, pat in [
    ("D2", "results/raw/exp1/exp1_quad_bo_ard_D2_k2_aligned_s{}"),
    ("D20k2", "results/raw/exp2/exp2_quad_bo_ard_D20_k2_aligned_s{}"),
    ("D20k20", "results/raw/exp1/exp1_quad_bo_ard_D20_k20_aligned_s{}"),
]:
    for s in range(5):
        h = pd.read_csv(pat.format(s) + ".csv")
        X = h.filter(like="x_").to_numpy()
        y = h["f"].to_numpy()
        U = torch.tensor((X + 1) / 2)
        Y = torch.tensor(y).unsqueeze(-1)
        m, _ = fit_model(U, Y, "ard")
        noise_std_units = float(m.likelihood.noise.item())
        sdY = y.std()
        rows.append(
            {
                "cond": tag,
                "seed": s,
                "regret": y.min(),
                "noise_var_standardised": noise_std_units,
                "noise_sd_f_units": np.sqrt(noise_std_units) * sdY,
                "sd_Y": sdY,
                "frac_last100_below_1e3": np.mean(y[-100:] < 1e-3),
            }
        )
d = pd.DataFrame(rows)
print(d.groupby("cond").median(numeric_only=True).drop(columns="seed").to_string())
