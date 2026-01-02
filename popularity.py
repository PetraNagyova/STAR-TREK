import numpy as np
import pandas as pd

episodes = pd.read_csv("project_tables_csv/episode_clean.csv")

episodes["avg_rating"] = pd.to_numeric(episodes["avg_rating"], errors="coerce")

# z-score v ramci serialu
group_cols = ["series"]

mean_g = episodes.groupby(group_cols)["avg_rating"].transform("mean")
std_g = episodes.groupby(group_cols)["avg_rating"].transform("std")

episodes["popularity"] = (episodes["avg_rating"] - mean_g) / std_g

# std 0 alebo NA
episodes.loc[std_g.isna() | (std_g == 0), "popularity"] = np.nan

episodes.to_csv("project_tables_csv/episode_popularity.csv", index=False)

print(episodes["popularity"].describe().to_string())