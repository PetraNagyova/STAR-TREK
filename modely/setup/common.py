from __future__ import annotations

from dataclasses import dataclass
import re
import sqlite3
import numpy as np
import pandas as pd

from .config import TABLE_DIR, DB_PATH

@dataclass
class Bundle:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    meta: pd.DataFrame


def canon_proj_ep_id(x) -> str:
    s = str(x).strip()
    if s.isdigit():
        return str(int(s))
    return s


def load_episode_df() -> pd.DataFrame:
    for name in ["episode_popularity.csv", "episode_clean.csv", "episode.csv"]:
        p = TABLE_DIR / name
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM episode", conn)
        conn.close()

    df = df.copy()
    df["proj_ep_id"] = df["proj_ep_id"].map(canon_proj_ep_id)
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    if "order_ep" in df.columns:
        df["order_ep"] = pd.to_numeric(df["order_ep"], errors="coerce")

    if "year" not in df.columns:
        if "air_date" in df.columns:
            dt = pd.to_datetime(df["air_date"], errors="coerce")
            df["year"] = dt.dt.year
        else:
            df["year"] = np.nan

    return df


def load_cast_df() -> pd.DataFrame:
    for name in ["cast_clean.csv", "cast.csv"]:
        p = TABLE_DIR / name
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM cast", conn)
        conn.close()

    df = df.copy()
    df["proj_ep_id"] = df["proj_ep_id"].map(canon_proj_ep_id)
    if "character_name_norm" not in df.columns:
        src = "character_name_norm" if "character_name_norm" in df.columns else "character_name"
        df["character_name_norm"] = df[src].astype(str)

    if "is_main_cast" in df.columns:
        df["is_main_cast"] = pd.to_numeric(df["is_main_cast"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_main_cast"] = 0

    return df


def load_director_df() -> pd.DataFrame:
    p = TABLE_DIR / "director.csv"
    if p.exists():
        df = pd.read_csv(p)
    else:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM director", conn)
        conn.close()

    df = df.copy()
    df["proj_ep_id"] = df["proj_ep_id"].map(canon_proj_ep_id)
    if "name" not in df.columns:
        if "director_name" in df.columns:
            df["name"] = df["director_name"].astype(str)
        else:
            df["name"] = df.iloc[:, -1].astype(str)
    return df


def load_relevant_characters_df() -> pd.DataFrame:
    for name in ["relevant_characters.csv", "filtered_out_characters.csv"]:
        p = TABLE_DIR / name
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("relevant_characters.csv / filtered_out_characters.csv")

    df = df.copy()
    if "character_name_norm" not in df.columns:
        if "character_name" in df.columns:
            df["character_name_norm"] = df["character_name"].astype(str)
        else:
            raise ValueError("relevant_characters: chýba character_name_norm (alebo character_name).")
    return df


_REMOVE_PREFIXES = [
    r"Captain", r"Capt\.", r"Commander", r"Cmdr\.",
    r"Lieutenant Commander", r"Lieutenant", r"Lt\.", r"Lt",
    r"Doctor", r"Dr\.", r"Dr",
    r"Ensign", r"Ens\.",
    r"Mr\.", r"Mrs\.", r"Ms\."
]

def normalize_cast_character_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("’", "'").replace("“", "'").replace("”", "'")
    changed = True
    while changed:
        old = s
        for p in _REMOVE_PREFIXES:
            s = re.sub(rf"^(?:{p})\s+", "", s, flags=re.IGNORECASE)
        changed = (s != old)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_alias_map_by_overlap(
    cast_norm: pd.DataFrame,
    series: str,
    min_eps: int = 5,
    overlap_thr: float = 0.90,
) -> dict[str, str]:

    sub = cast_norm[cast_norm["series"] == series].copy()
    if sub.empty:
        return {}

    groups = {}
    for name, g in sub.groupby("character_name_norm"):
        eps = set(g["proj_ep_id"].astype(str).tolist())
        groups[str(name)] = eps

    names = list(groups.keys())
    alias = {}

    for a in names:
        eps_a = groups[a]
        if len(eps_a) < min_eps:
            continue
        a_tok = a.strip()

        for b in names:
            if a == b:
                continue
            eps_b = groups[b]
            if len(eps_b) < min_eps:
                continue

            if re.search(rf"(?:^|\s){re.escape(a_tok)}$", b) is None:
                continue

            inter = len(eps_a & eps_b)
            denom = min(len(eps_a), len(eps_b))
            if denom == 0:
                continue
            overlap = inter / denom
            if overlap >= overlap_thr:
                if len(b) > len(a):
                    alias[a] = b

    def resolve(x):
        seen = set()
        while x in alias and x not in seen:
            seen.add(x)
            x = alias[x]
        return x

    alias = {k: resolve(v) for k, v in alias.items()}
    return alias


def build_meta_X_y_groups(
    use_meta: bool = True,
    use_characters: bool = False,
    use_director: bool = False,
    main_cast_only: bool = True,
    series_filter: str | None = None,
    group_col: str = "series",
    merge_aliases: bool = True,
) -> Bundle:
    ep = load_episode_df()

    if series_filter is not None:
        ep = ep[ep["series"].astype(str) == str(series_filter)].copy()

    if "popularity" not in ep.columns:
        raise ValueError("episode_df nemá stĺpec popularity.")
    y = ep["popularity"].astype(float)
    ok = y.notna()
    ep = ep.loc[ok].copy()
    y = y.loc[ok].copy()

    ep = ep.reset_index(drop=True)
    y = y.reset_index(drop=True)

    ep["proj_ep_id"] = ep["proj_ep_id"].map(canon_proj_ep_id)
    meta = ep[["proj_ep_id", "series", "season", "order_ep", "year"]].copy().reset_index(drop=True)

    if group_col not in ep.columns:
        raise ValueError(f"group_col='{group_col}' nie je v episode_df stĺpcoch.")

    if group_col == "season" and series_filter is None:
        groups = (ep["series"].astype(str) + "_S" + ep["season"].astype(str)).reset_index(drop=True)
    else:
        groups = ep[group_col].copy().reset_index(drop=True)

    # X: meta
    X_parts = []
    if use_meta:
        X_meta = ep[["season", "order_ep", "year"]].copy()
        X_meta = X_meta.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_parts.append(X_meta)

    # X: characters
    if use_characters:
        cast = load_cast_df()

        if "series" not in cast.columns:
            cast = cast.merge(ep[["proj_ep_id", "series", "season"]], on="proj_ep_id", how="left")

        cast["character_name_norm"] = cast["character_name_norm"].map(normalize_cast_character_name)

        if series_filter is not None:
            cast = cast[cast["series"].astype(str) == str(series_filter)].copy()

        if main_cast_only:
            cast = cast[cast["is_main_cast"] == 1].copy()

        rel = load_relevant_characters_df()
        if "series" in rel.columns and series_filter is not None:
            rel = rel[rel["series"].astype(str) == str(series_filter)].copy()

        alias = {}
        if merge_aliases and series_filter is not None:
            alias = build_alias_map_by_overlap(cast_norm=cast, series=str(series_filter))

        if alias:
            rel["character_name_norm"] = rel["character_name_norm"].map(lambda x: alias.get(str(x), str(x)))
            cast["character_name_norm"] = cast["character_name_norm"].map(lambda x: alias.get(str(x), str(x)))

        vocab = set(rel["character_name_norm"].astype(str).tolist())
        cast = cast[cast["character_name_norm"].isin(vocab)].copy()

        if cast.empty:
            X_char = pd.DataFrame(index=ep.index)
        else:
            tab = (
                pd.crosstab(
                    cast["proj_ep_id"].astype(str),
                    cast["character_name_norm"].astype(str)
                )
                .clip(upper=1)
                .astype(float)
            )
            tab = tab.reindex(ep["proj_ep_id"].astype(str)).fillna(0.0)
            X_char = tab.reset_index(drop=True)

        X_parts.append(X_char)

    # X: director
    if use_director:
        d = load_director_df()
        if series_filter is not None and "proj_ep_id" in d.columns:
            d = d.merge(ep[["proj_ep_id", "series"]], on="proj_ep_id", how="left")
            d = d[d["series"].astype(str) == str(series_filter)].copy()

        if d.empty:
            X_dir = pd.DataFrame(index=ep.index)
        else:
            d = d.copy()
            d["name"] = d["name"].astype(str).fillna("")
            tab = pd.get_dummies(d.set_index("proj_ep_id")["name"], prefix="director")
            tab = tab.groupby(level=0).max()  # ak by bolo viac režisérov
            tab = tab.reindex(ep["proj_ep_id"]).fillna(0.0).astype(float)
            X_dir = tab.reset_index(drop=True)

        X_parts.append(X_dir)

    if X_parts:
        X = pd.concat(X_parts, axis=1)
    else:
        X = pd.DataFrame(index=ep.index)

    if X.shape[1] == 0:
        X = pd.DataFrame({"__bias0": np.zeros(len(ep), dtype=float)})

    return Bundle(X=X.reset_index(drop=True), y=y.reset_index(drop=True), groups=groups.reset_index(drop=True), meta=meta)