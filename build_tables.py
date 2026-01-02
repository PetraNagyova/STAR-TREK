from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

IMDB_DIR = Path("imdb_processed")
ST_DIR = Path("startrek_processed")
OUT_DB = Path("project.db")
OUT_CSV_DIR = Path("project_tables_csv")

MAIN_CAST_CUTOFF = 10

SERIES_ABBR_MAP = {
    "Star Trek": "TOS",
    "Star Trek: The Original Series": "TOS",
    "Star Trek: The Animated Series": "TAS",
    "Star Trek: The Next Generation": "TNG",
    "Star Trek: Deep Space Nine": "DS9",
    "Star Trek: Voyager": "VOY",
    "Star Trek: Enterprise": "ENT",
    "Star Trek: Discovery": "DSC",
    "Star Trek: Short Treks": "STT",
    "Star Trek Short Treks": "STT",
    "Star Trek: Picard": "PIC",
    "Star Trek: Lower Decks": "LD",
    "Star Trek: Prodigy": "PRO",
    "Star Trek: Strange New Worlds": "SNW",
}

SERIES_INDEX_MAP = {
    "TOS": "001",
    "TAS": "002",
    "TNG": "003",
    "DS9": "004",
    "VOY": "005",
    "ENT": "006",
    "DSC": "007",
    "STT": "008",
    "PIC": "009",
    "LD":  "010",
    "PRO": "011",
    "SNW": "012",
}

def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, na_values=[r"\N"])


def safe_int(x):
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def make_proj_ep_id(series_abbr: str, season: int, order_ep: int) -> str | None:
    idx = SERIES_INDEX_MAP.get(series_abbr)
    if idx is None or season is None or order_ep is None:
        return None
    return f"{idx}{season:02d}{order_ep:02d}"

def parse_first_character(characters_field: str | None) -> str | None:
    if characters_field is None or pd.isna(characters_field):
        return None
    s = str(characters_field).strip()
    if not s or s == r"\N":
        return None
    try:
        val = json.loads(s)
        if isinstance(val, list) and len(val) > 0:
            return str(val[0])
    except Exception:
        pass
    return None

st_eps = pd.read_csv(ST_DIR / "episodes.csv", dtype=str)
st_eps["season"] = st_eps["season"].map(safe_int)
st_eps["order_ep"] = st_eps["episode_number"].map(safe_int)
st_eps["series_abbr"] = st_eps["series_title"].map(SERIES_ABBR_MAP)

imdb_series = read_tsv(IMDB_DIR / "series.tsv")

series_key = "tconst" if "tconst" in imdb_series.columns else imdb_series.columns[0]
series_title_col = "primaryTitle" if "primaryTitle" in imdb_series.columns else (
    "series_title" if "series_title" in imdb_series.columns else None
)
if series_title_col is None:
    raise ValueError("V series.tsv nie je (primaryTitle/series_title).")

imdb_series = imdb_series[[series_key, series_title_col]].rename(
    columns={series_key: "parentTconst", series_title_col: "series_title"}
)

imdb_eps = read_tsv(IMDB_DIR / "episodes.tsv")
rename_map = {}
for c in imdb_eps.columns:
    if c.lower() == "tconst":
        rename_map[c] = "ep_id"
    if c.lower() == "parenttconst":
        rename_map[c] = "parentTconst"
    if c.lower() == "seasonnumber":
        rename_map[c] = "season"
    if c.lower() == "episodenumber":
        rename_map[c] = "order_ep"
imdb_eps = imdb_eps.rename(columns=rename_map)

needed = {"ep_id", "parentTconst", "season", "order_ep"}
missing = needed - set(imdb_eps.columns)
if missing:
    raise ValueError(f"V episodes.tsv chýbajú stĺpce: {missing}")

imdb_eps["season"] = imdb_eps["season"].map(safe_int)
imdb_eps["order_ep"] = imdb_eps["order_ep"].map(safe_int)

imdb_ratings = read_tsv(IMDB_DIR / "ratings.tsv")
imdb_ratings = imdb_ratings.rename(columns={
    "tconst": "ep_id",
    "averageRating": "avg_rating",
    "numVotes": "num_votes",
})
if "ep_id" not in imdb_ratings.columns:
    imdb_ratings = imdb_ratings.rename(columns={imdb_ratings.columns[0]: "ep_id"})

imdb_names = read_tsv(IMDB_DIR / "names.tsv")
imdb_names = imdb_names.rename(columns={
    "nconst": "nconst",
    "primaryName": "name",
})
if "name" not in imdb_names.columns:
    possible = [c for c in imdb_names.columns if "name" in c.lower()]
    if not possible:
        raise ValueError("V names.tsv neviem nájsť stĺpec s menom osoby (primaryName).")
    imdb_names = imdb_names.rename(columns={possible[0]: "name"})
imdb_names = imdb_names[["nconst", "name"]].drop_duplicates()

principals = read_tsv(IMDB_DIR / "principals_cast.tsv")
principals = principals.rename(columns={
    "tconst": "ep_id",
})

if "ordering" in principals.columns:
    principals["billing_order"] = principals["ordering"].map(safe_int)
else:
    principals["billing_order"] = None


# episode
imdb_eps = imdb_eps.merge(imdb_series, on="parentTconst", how="left")
imdb_eps["series_abbr"] = imdb_eps["series_title"].map(SERIES_ABBR_MAP)

# join so startrek-db
episode = imdb_eps.merge(
    st_eps[["series_abbr", "season", "order_ep", "episode_title", "airdate"]],
    on=["series_abbr", "season", "order_ep"],
    how="left",
)

# ratings
episode = episode.merge(
    imdb_ratings[["ep_id", "avg_rating", "num_votes"]],
    on="ep_id",
    how="left",
)

# proj_ep_id
episode["proj_ep_id"] = episode.apply(
    lambda r: make_proj_ep_id(r["series_abbr"], r["season"], r["order_ep"])
    if pd.notna(r["series_abbr"]) and pd.notna(r["season"]) and pd.notna(r["order_ep"])
    else None,
    axis=1,
)

episode_table = episode.rename(columns={
    "series_abbr": "series",
    "episode_title": "ep_title",
    "airdate": "air_date",
})[[
    "proj_ep_id",
    "ep_id",
    "series",
    "ep_title",
    "season",
    "order_ep",
    "air_date",
    "avg_rating",
    "num_votes",
]].copy()

episode_table["popularity"] = None

# director
directors = principals[principals["category"].eq("director")].copy()
directors = directors.merge(imdb_names, on="nconst", how="left")
directors = directors.merge(
    episode_table[["proj_ep_id", "ep_id"]],
    on="ep_id",
    how="inner",
)

director_table = directors[["proj_ep_id", "ep_id", "name"]].dropna(subset=["proj_ep_id", "ep_id"])

# cast
cast = principals[principals["category"].isin(["actor", "actress", "self"])].copy()
cast["character_name"] = cast["characters"].apply(parse_first_character) if "characters" in cast.columns else None
cast = cast.merge(imdb_names, on="nconst", how="left")
cast = cast.merge(
    episode_table[["proj_ep_id", "ep_id"]],
    on="ep_id",
    how="inner",
)

cast["is_main_cast"] = cast["billing_order"].apply(lambda x: int(x is not None and x <= MAIN_CAST_CUTOFF))

cast_table = cast[[
    "proj_ep_id",
    "nconst",
    "name",
    "character_name",
    "billing_order",
    "is_main_cast",
]].copy()


OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

episode_table.to_csv(OUT_CSV_DIR / "episode.csv", index=False, encoding="utf-8")
director_table.to_csv(OUT_CSV_DIR / "director.csv", index=False, encoding="utf-8")
cast_table.to_csv(OUT_CSV_DIR / "cast.csv", index=False, encoding="utf-8")

conn = sqlite3.connect(OUT_DB)
episode_table.to_sql("episode", conn, if_exists="replace", index=False)
director_table.to_sql("director", conn, if_exists="replace", index=False)
cast_table.to_sql("cast", conn, if_exists="replace", index=False)

# indexy
conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_ep_id ON episode(ep_id);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_proj ON episode(proj_ep_id);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_dir_proj ON director(proj_ep_id);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_cast_proj ON cast(proj_ep_id);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_cast_nconst ON cast(nconst);")
conn.commit()
conn.close()

print(f"SQLite: {OUT_DB.resolve()}")
print(f"CSV: {OUT_CSV_DIR.resolve()}")

matched = episode_table["air_date"].notna().sum()
total = len(episode_table)
print(f"Prepojené air_date: {matched}/{total} ({matched/total:.1%})")

# manualne doplnenie dat
import pandas as pd

df = pd.read_csv(OUT_CSV_DIR / "episode.csv", dtype=str)

patches = pd.DataFrame([
    {
        "proj_ep_id": "0040402",
        "ep_id": "tt0708645",
        "ep_title": "The Visitor",
        "air_date": "1995-10-09",
    },
    {
        "proj_ep_id": "0060102",
        "ep_id": "tt25396064",
        "ep_title": "Broken Bow, Part 2",
        "air_date": "2001-09-26",
    },
    {
        "proj_ep_id": "0110102",
        "ep_id": "tt14080498",
        "ep_title": "Lost & Found, Part 2",
        "air_date": "2021-10-28",
    },
    {
        "proj_ep_id": "0030102",
        "ep_id": "tt0708810",
        "ep_title": "The Naked Now",
        "air_date": "1987-10-03",
    },
    {
        "proj_ep_id": "0050102",
        "ep_id": "tt0708943",
        "ep_title": "Parallax",
        "air_date": "1995-01-23",
    },
{
        "proj_ep_id": "0050516",
        "ep_id": "tt0708980",
        "ep_title": "The Disease",
        "air_date": "1999-02-24",
    },
{
        "proj_ep_id": "0050710",
        "ep_id": "tt0708970",
        "ep_title": "Shattered",
        "air_date": "2001-01-17",
    },
{
        "proj_ep_id": "0010100",
        "ep_id": "tt0059753",
        "ep_title": "The Cage",
        "air_date": "1988-10-04",
    }
])

df = df.merge(patches, on="ep_id", how="left", suffixes=("", "_fix"))
for col in ["proj_ep_id", "ep_title", "air_date"]:
    df[col] = df[col].fillna(df[f"{col}_fix"])
    df = df.drop(columns=[f"{col}_fix"])

df.to_csv(OUT_CSV_DIR / "episode.csv", index=False, encoding="utf-8")

# vypis chybajucich hodnot
df = pd.read_csv(
    OUT_CSV_DIR / "episode.csv",
    dtype={
        "proj_ep_id": "string",
        "ep_id": "string",
        "series": "string",
        "ep_title": "string",
        "air_date": "string",
        "avg_rating": "string",
        "num_votes": "string",
        "popularity": "string",
    },
)

missing = df[
    df["proj_ep_id"].isna()
    | df["ep_title"].isna()
    | df["air_date"].isna()
    | df["avg_rating"].isna()
    | df["num_votes"].isna()
].copy()

missing = missing.sort_values(["series", "season", "order_ep"])

print("Rows with missing values:", len(missing))
print(
    missing[
        ["proj_ep_id", "ep_id", "series", "season", "order_ep", "ep_title", "air_date", "avg_rating", "num_votes"]
    ].to_string(index=False)
)