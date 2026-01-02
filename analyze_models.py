
from pathlib import Path
import pandas as pd

IN_PATH = Path("output_models/series_cv_results.csv")
OUT_PATH = Path("output_models/best_models_by_series.csv")

df = pd.read_csv(IN_PATH)

df.columns = [c.strip().lower() for c in df.columns]

metrics = [m for m in ["rmse", "mae", "r2"] if m in df.columns]
for m in metrics:
    df[m] = pd.to_numeric(df[m], errors="coerce")

def best_row(g: pd.DataFrame) -> pd.Series:
    out = {"series": g["series"].iloc[0]}

    if "rmse" in g.columns:
        r = g.loc[g["rmse"].idxmin()]
        out["best_model_rmse"] = r["model"]
        out["rmse_best"] = r["rmse"]

    if "mae" in g.columns:
        r = g.loc[g["mae"].idxmin()]
        out["best_model_mae"] = r["model"]
        out["mae_best"] = r["mae"]

    if "r2" in g.columns:
        r = g.loc[g["r2"].idxmax()]
        out["best_model_r2"] = r["model"]
        out["r2_best"] = r["r2"]

    return pd.Series(out)

best = df.groupby("series", as_index=False).apply(best_row)

if "rmse_best" in best.columns:
    best = best.sort_values("rmse_best")

print(best.to_string(index=False))