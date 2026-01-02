import pandas as pd

# normalizacia id
SERIES_INDEX_MAP = {
    "TOS": "001", "TAS": "002", "TNG": "003", "DS9": "004",
    "VOY": "005", "ENT": "006", "DSC": "007", "STT": "008",
    "PIC": "009", "LD": "010", "PRO": "011", "SNW": "012",
}

def make_proj_ep_id(series: str, season: int, order_ep: int) -> str | None:
    idx = SERIES_INDEX_MAP.get(series)
    if idx is None or pd.isna(season) or pd.isna(order_ep):
        return None
    return f"{idx}{int(season):02d}{int(order_ep):02d}"

episode = pd.read_csv("project_tables_csv/episode.csv", dtype=str)

episode["season"] = pd.to_numeric(episode["season"], errors="coerce").astype("Int64")
episode["order_ep"] = pd.to_numeric(episode["order_ep"], errors="coerce").astype("Int64")

mask = episode["proj_ep_id"].isna()
episode.loc[mask, "proj_ep_id"] = episode.loc[mask].apply(
    lambda r: make_proj_ep_id(r["series"], r["season"], r["order_ep"]), axis=1
)

dup = episode[episode["proj_ep_id"].notna() & episode["proj_ep_id"].duplicated(keep=False)]
if len(dup) > 0:
    print("WARNING: duplicitné proj_ep_id:")
    print(dup[["proj_ep_id", "ep_id", "series", "season", "order_ep"]].sort_values("proj_ep_id").to_string(index=False))

episode.to_csv("project_tables_csv/episode_clean.csv", index=False, encoding="utf-8")

# kontrola cast_clean
cast = pd.read_csv("project_tables_csv/cast_clean.csv", dtype=str)

cast["billing_order"] = pd.to_numeric(cast["billing_order"], errors="coerce").astype("Int64")
cast["is_main_cast"] = pd.to_numeric(cast["is_main_cast"], errors="coerce").fillna(0).astype(int)

if "character_name_norm" not in cast.columns:
    raise ValueError("chyba character_name_norm")

cast = cast.sort_values(["proj_ep_id", "nconst", "billing_order"], ascending=[True, True, True])

cast_clean = (
    cast.groupby(["proj_ep_id", "nconst", "character_name_norm"], dropna=False, as_index=False)
        .agg(
            billing_order=("billing_order", "min"),
            is_main_cast=("is_main_cast", "max"),
            name=("name", "first"),
        )
)

cast_clean.to_csv("project_tables_csv/cast_clean.csv", index=False, encoding="utf-8")

# chybajuce data
ep = pd.read_csv("project_tables_csv/episode_clean.csv", dtype=str)

ep["has_air_date"] = ep["air_date"].notna()
ep["has_rating"] = ep["avg_rating"].notna() & ep["num_votes"].notna()
ep["is_unmatched_startrekdb"] = ep["air_date"].isna()

ep["num_votes"] = pd.to_numeric(ep["num_votes"], errors="coerce")
ep["avg_rating"] = pd.to_numeric(ep["avg_rating"], errors="coerce")

needs_patch = ep[ep["air_date"].isna() | ep["avg_rating"].isna() | ep["num_votes"].isna()].copy()
needs_patch = needs_patch.sort_values(["series", "season", "order_ep"])

print("Epizody s missing (air_date alebo rating):", len(needs_patch))
print(needs_patch[["proj_ep_id", "ep_id", "series", "season", "order_ep", "ep_title", "air_date", "avg_rating", "num_votes"]]
      .head(50).to_string(index=False))