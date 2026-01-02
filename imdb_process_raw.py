import pandas as pd
from pathlib import Path

DATA_DIR = Path("imdb_raw")
OUT_DIR = Path("imdb_processed")
OUT_DIR.mkdir(exist_ok=True)

NA = r"\N"

EXACT_TITLES = {
    "Star Trek: The Next Generation",
    "Star Trek: Deep Space Nine",
    "Star Trek: Voyager",
    "Star Trek: Enterprise",
    "Star Trek: Discovery",
    "Star Trek: Short Treks",
    "Star Trek: Picard",
    "Star Trek: Lower Decks",
    "Star Trek: Prodigy",
    "Star Trek: Strange New Worlds",
    "Star Trek: The Animated Series",
}

# pre TOS
STAR_TREK_YEARS = {"1966"}

def load_12_series_tconsts() -> pd.DataFrame:
    basics_path = DATA_DIR / "title.basics.tsv.gz"
    usecols = ["tconst", "titleType", "primaryTitle", "originalTitle", "startYear", "endYear"]

    parts = []
    for chunk in pd.read_csv(
        basics_path, sep="\t", compression="gzip",
        usecols=usecols, dtype=str, na_values=NA, keep_default_na=False,
        chunksize=1_000_000
    ):
        mask_exact = chunk["primaryTitle"].isin(EXACT_TITLES) | chunk["originalTitle"].isin(EXACT_TITLES)
        mask_star_trek = (
            (chunk["primaryTitle"] == "Star Trek") &
            (chunk["startYear"].isin(STAR_TREK_YEARS)) &
            (chunk["titleType"] == "tvSeries")
        )
        sub = chunk[mask_exact | mask_star_trek]
        sub = sub[sub["titleType"].isin(["tvSeries"])]
        if not sub.empty:
            parts.append(sub)

    series = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["tconst"])
        .sort_values(["primaryTitle", "startYear"])
    )
    return series

def load_episodes_for_series(series_tconsts: set[str]) -> pd.DataFrame:
    episode_path = DATA_DIR / "title.episode.tsv.gz"
    usecols = ["tconst", "parentTconst", "seasonNumber", "episodeNumber"]

    parts = []
    for chunk in pd.read_csv(
        episode_path, sep="\t", compression="gzip",
        usecols=usecols, dtype=str, na_values=NA, keep_default_na=False,
        chunksize=1_000_000
    ):
        sub = chunk[chunk["parentTconst"].isin(series_tconsts)]
        if not sub.empty:
            parts.append(sub)

    episodes = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["tconst"])
    return episodes

def filter_by_tconst(path: Path, tconst_set: set[str], usecols: list[str], chunksize: int) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(
        path, sep="\t", compression="gzip",
        usecols=usecols, dtype=str, na_values=NA, keep_default_na=False,
        chunksize=chunksize
    ):
        sub = chunk[chunk["tconst"].isin(tconst_set)]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True)

def filter_names_by_nconst(nconst_set: set[str]) -> pd.DataFrame:
    names_path = DATA_DIR / "name.basics.tsv.gz"
    usecols = ["nconst", "primaryName", "birthYear", "deathYear", "primaryProfession", "knownForTitles"]

    parts = []
    for chunk in pd.read_csv(
        names_path, sep="\t", compression="gzip",
        usecols=usecols, dtype=str, na_values=NA, keep_default_na=False,
        chunksize=1_000_000
    ):
        sub = chunk[chunk["nconst"].isin(nconst_set)]
        if not sub.empty:
            parts.append(sub)

    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["nconst"])

series_df = load_12_series_tconsts()
series_tconsts = set(series_df["tconst"])

episodes_df = load_episodes_for_series(series_tconsts)
episode_tconsts = set(episodes_df["tconst"])

ratings_df = filter_by_tconst(
    DATA_DIR / "title.ratings.tsv.gz",
    episode_tconsts,
    usecols=["tconst", "averageRating", "numVotes"],
    chunksize=2_000_000
)

principals_df = filter_by_tconst(
    DATA_DIR / "title.principals.tsv.gz",
    episode_tconsts,
    usecols=["tconst", "ordering", "nconst", "category", "job", "characters"],
    chunksize=2_000_000
)

principals_cast_df = principals_df[principals_df["category"].isin(["actor", "actress", "director", "self"])].copy()

nconsts = set(principals_cast_df["nconst"].dropna().unique())
names_df = filter_names_by_nconst(nconsts)

series_df.to_csv(OUT_DIR / "series.tsv", sep="\t", index=False, encoding="utf-8")
episodes_df.to_csv(OUT_DIR / "episodes.tsv", sep="\t", index=False, encoding="utf-8")
ratings_df.to_csv(OUT_DIR / "ratings.tsv", sep="\t", index=False, encoding="utf-8")
principals_cast_df.to_csv(OUT_DIR / "principals_cast.tsv", sep="\t", index=False, encoding="utf-8")
names_df.to_csv(OUT_DIR / "names.tsv", sep="\t", index=False, encoding="utf-8")

print("OK")
print("Series:", len(series_df))
print("Episodes:", len(episodes_df))
print("Ratings rows:", len(ratings_df))
print("Principals (cast) rows:", len(principals_cast_df))
print("Names:", len(names_df))
