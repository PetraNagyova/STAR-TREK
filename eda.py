from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPISODES_PATH = "project_tables_csv/episode_popularity.csv"
RELEVANT_CHAR_PATH = "project_tables_csv/relevant_characters.csv"
DIRECTOR_PATH = "project_tables_csv/director.csv"

COL_PROJ_EP_ID = "proj_ep_id"
COL_SERIES = "series"
COL_SEASON = "season"
COL_ORDER = "order_ep"
COL_AIRDATE = "air_date"
COL_AVG_RATING = "avg_rating"
COL_NUM_VOTES = "num_votes"
COL_POP = "popularity"

COL_DIRECTOR_NAME = "name"

COL_CHAR_NAME = "character_name_norm"
COL_CHAR_N_EP = "n_episodes"


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def print_df(title: str, df: pd.DataFrame, max_rows: int = 30) -> None:
    print("\n")
    print(title)
    print()
    if df.empty:
        print("(prázdne)")
        return
    if len(df) <= max_rows:
        print(df.to_string(index=False))
    else:
        print(df.head(max_rows).to_string(index=False))


def plot_hist(series: pd.Series, title: str, xlabel: str, bins: int = 40) -> None:
    x = series.dropna().values
    plt.figure()
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("počet")
    plt.tight_layout()
    plt.show()


def plot_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, xlog: bool = False) -> None:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    plt.figure()
    plt.scatter(tmp["x"].values, tmp["y"].values, s=10)
    if xlog:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_box_by_group(df: pd.DataFrame, group_col: str, value_col: str, title: str) -> None:
    groups = []
    labels = []
    for g, part in df.groupby(group_col):
        vals = part[value_col].dropna().values
        if len(vals) >= 10:
            groups.append(vals)
            labels.append(str(g))

    if not groups:
        return

    plt.figure(figsize=(10, 5))
    plt.boxplot(groups, tick_labels=labels, showfliers=False)
    plt.title(title)
    plt.xlabel(group_col)
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.show()


def plot_top_barh(names: pd.Series, values: pd.Series, title: str, xlabel: str) -> None:
    names = names.astype(str)
    values = pd.to_numeric(values, errors="coerce")

    plt.figure(figsize=(10, max(4, 0.28 * len(names))))
    y = np.arange(len(names))
    plt.barh(y, values.values)
    plt.yticks(y, names.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()

def plot_popularity_overview(ep: pd.DataFrame) -> None:
    plot_hist(ep[COL_POP], "Rozdelenie popularity (z-score) – všetky epizódy", "popularity", bins=40)

    if COL_SERIES in ep.columns:
        plot_box_by_group(ep, COL_SERIES, COL_POP, "Popularita podľa seriálu (bez extrémov)")

    if COL_AIRDATE in ep.columns and ep[COL_AIRDATE].notna().any():
        tmp = ep[[COL_AIRDATE, COL_POP]].dropna().sort_values(COL_AIRDATE)
        plt.figure(figsize=(10, 4))
        plt.plot(tmp[COL_AIRDATE].values, tmp[COL_POP].values, marker="o", linestyle="None", markersize=2)
        plt.title("Popularita v čase (podľa dátumu vysielania)")
        plt.xlabel("air_date")
        plt.ylabel("popularity")
        plt.tight_layout()
        plt.show()

    if COL_ORDER in ep.columns and ep[COL_ORDER].notna().any():
        tmp = ep[[COL_ORDER, COL_POP]].dropna().sort_values(COL_ORDER)
        plt.figure(figsize=(10, 4))
        plt.plot(tmp[COL_ORDER].values, tmp[COL_POP].values, marker="o", linestyle="None", markersize=2)
        plt.title("Popularita podľa poradia epizódy (order_ep)")
        plt.xlabel("order_ep")
        plt.ylabel("popularity")
        plt.tight_layout()
        plt.show()


def votes_diagnostic(ep: pd.DataFrame) -> None:
    if COL_NUM_VOTES not in ep.columns or not ep[COL_NUM_VOTES].notna().any():
        print("\n(num_votes chýba alebo je prázdny) – preskakujem votes diagnostiku.")
        return

    cols = [COL_PROJ_EP_ID, COL_NUM_VOTES, COL_POP]
    for c in [COL_SERIES, COL_SEASON, COL_ORDER]:
        if c in ep.columns:
            cols.append(c)

    tmp = ep[cols].copy()
    tmp[COL_NUM_VOTES] = pd.to_numeric(tmp[COL_NUM_VOTES], errors="coerce")
    tmp[COL_POP] = pd.to_numeric(tmp[COL_POP], errors="coerce")
    tmp = tmp.dropna(subset=[COL_NUM_VOTES, COL_POP, COL_PROJ_EP_ID])

    plot_scatter(
        tmp[COL_NUM_VOTES],
        tmp[COL_POP],
        "Popularita vs počet hlasov (log x)",
        "num_votes (log scale)",
        "popularity",
        xlog=True,
    )

    q_low = float(tmp[COL_NUM_VOTES].quantile(0.05))
    extreme = tmp[(tmp[COL_NUM_VOTES] <= q_low) & (tmp[COL_POP].abs() >= 2.0)].copy()

    if "ep_title" in ep.columns and not extreme.empty:
        extreme = extreme.merge(
            ep[[COL_PROJ_EP_ID, "ep_title"]].drop_duplicates(),
            on=COL_PROJ_EP_ID,
            how="left",
        )

    print_df(
        f"Nízky num_votes + extrémna popularita (num_votes <= 5% kvantil = {q_low:.0f}, |pop|>=2)",
        extreme.sort_values([COL_NUM_VOTES, COL_POP]),
        max_rows=200,
    )


def character_frequency_eda(rel_chars: pd.DataFrame, top_n: int = 30) -> None:
    need = [COL_SERIES, COL_CHAR_NAME, COL_CHAR_N_EP]
    missing = [c for c in need if c not in rel_chars.columns]
    if missing:
        print("character_frequency_eda: chýbajú stĺpce:", missing)
        return

    tmp = rel_chars[[COL_SERIES, COL_CHAR_NAME, COL_CHAR_N_EP]].copy()
    tmp[COL_CHAR_N_EP] = pd.to_numeric(tmp[COL_CHAR_N_EP], errors="coerce")
    tmp = tmp.dropna(subset=[COL_CHAR_N_EP, COL_CHAR_NAME, COL_SERIES])

    desc = tmp[COL_CHAR_N_EP].describe().reset_index()
    desc.columns = ["stat", "value"]
    print_df("Relevantné postavy (per-series counts): describe(n_episodes)", desc)

    plot_hist(
        tmp[COL_CHAR_N_EP],
        "Rozdelenie frekvencie postáv (n_episodes) – per series",
        "n_episodes",
        bins=30,
    )

    top_per_series = (
        tmp.sort_values([COL_SERIES, COL_CHAR_N_EP], ascending=[True, False])
           .groupby(COL_SERIES, as_index=False, group_keys=False)
           .head(top_n)
           .reset_index(drop=True)
    )

    print("\n")
    print(f"Top {top_n} postáv PER SERIES (podľa n_episodes)")
    for s, g in top_per_series.groupby(COL_SERIES):
        g2 = g[[COL_SERIES, COL_CHAR_NAME, COL_CHAR_N_EP]].copy()
        print(f"\n[{s}] (zobrazených {len(g2)}):")
        print(g2.to_string(index=False))

    for s, g in tmp.groupby(COL_SERIES):
        gtop = g.sort_values(COL_CHAR_N_EP, ascending=False).head(min(15, top_n))
        if gtop.empty:
            continue
        plot_top_barh(
            gtop[COL_CHAR_NAME],
            gtop[COL_CHAR_N_EP],
            f"Top {min(15, top_n)} postáv v seriáli {s}",
            "n_episodes",
        )

    franchise = (
        tmp.groupby(COL_CHAR_NAME, as_index=False)[COL_CHAR_N_EP].sum()
           .rename(columns={COL_CHAR_N_EP: "n_episodes_total"})
           .sort_values("n_episodes_total", ascending=False)
           .reset_index(drop=True)
    )

    top_franchise = franchise.head(top_n)
    print_df(f"Top {top_n} postáv FRANCHISE (súčet n_episodes cez series)", top_franchise, max_rows=top_n)

    plot_top_barh(
        top_franchise[COL_CHAR_NAME],
        top_franchise["n_episodes_total"],
        f"Top {top_n} postáv FRANCHISE (súčet cez series)",
        "n_episodes_total",
    )


def director_eda(ep: pd.DataFrame, director: pd.DataFrame) -> None:
    if COL_PROJ_EP_ID not in director.columns or COL_DIRECTOR_NAME not in director.columns:
        print("\n(director tabuľka nemá očakávané stĺpce)")
        return
    if COL_PROJ_EP_ID not in ep.columns:
        print("\n(episodes tabuľka nemá proj_ep_id)")
        return

    d = director[[COL_PROJ_EP_ID, COL_DIRECTOR_NAME]].dropna().drop_duplicates()
    merged = ep.merge(d, on=COL_PROJ_EP_ID, how="left")

    counts = (
        merged.groupby(COL_DIRECTOR_NAME)[COL_PROJ_EP_ID]
        .nunique()
        .rename("n_episodes")
        .reset_index()
        .sort_values("n_episodes", ascending=False)
    )
    print_df("Režiséri: počet epizód (top)", counts, max_rows=50)

    top20 = counts.head(20).reset_index(drop=True)
    plot_top_barh(top20[COL_DIRECTOR_NAME], top20["n_episodes"], "Top 20 režisérov podľa počtu epizód", "počet epizód")

    stats = (
        merged.dropna(subset=[COL_DIRECTOR_NAME, COL_POP])
        .groupby(COL_DIRECTOR_NAME)[COL_POP]
        .agg(n="count", mean="mean", std="std", min="min", median="median", max="max")
        .reset_index()
        .sort_values(["n", "mean"], ascending=[False, False])
    )
    print_df("Režiséri: popularita – zoradené podľa n a mean", stats, max_rows=80)

    stats2 = stats[stats["n"] >= 5].copy()
    if not stats2.empty:
        best = stats2.sort_values("mean", ascending=False).head(10)
        worst = stats2.sort_values("mean", ascending=True).head(10)
        both = pd.concat([worst, best], axis=0).reset_index(drop=True)

        plot_top_barh(
            both[COL_DIRECTOR_NAME],
            both["mean"],
            "Priemerná popularita podľa režiséra – bottom 10 + top 10",
            "mean popularity",
        )


def main() -> None:
    ep = pd.read_csv(EPISODES_PATH)
    rel_chars = pd.read_csv(RELEVANT_CHAR_PATH)
    director = pd.read_csv(DIRECTOR_PATH)

    for c in [COL_POP, COL_NUM_VOTES, COL_SEASON, COL_ORDER]:
        if c in ep.columns:
            ep[c] = pd.to_numeric(ep[c], errors="coerce")
    if COL_AIRDATE in ep.columns:
        ep[COL_AIRDATE] = safe_to_datetime(ep[COL_AIRDATE])

    plot_popularity_overview(ep)
    votes_diagnostic(ep)

    character_frequency_eda(rel_chars)
    director_eda(ep, director)


if __name__ == "__main__":
    main()
