import pandas as pd

MIN_EPISODES = 3
CAST_PATH = "project_tables_csv/cast_clean.csv"
EPISODE_PATH = "project_tables_csv/episode_clean.csv"
OUT_RELEVANT = "project_tables_csv/relevant_characters.csv"
OUT_FILTERED = "project_tables_csv/filtered_out_characters.csv"


def main() -> None:
    cast = pd.read_csv(CAST_PATH, dtype=str)
    ep = pd.read_csv(EPISODE_PATH, dtype=str)
    cast = cast.merge(ep[["proj_ep_id", "series"]], on="proj_ep_id", how="left")
    cast = cast[cast["character_name_norm"].notna()].copy()

    stats = (
        cast.groupby(["series", "character_name_norm"], as_index=False)
            .agg(
                n_episodes=("proj_ep_id", "nunique"),
                n_rows=("proj_ep_id", "size"),
            )
            .sort_values(["n_episodes", "n_rows"], ascending=False)
    )

    relevant = stats[stats["n_episodes"] >= MIN_EPISODES].copy()
    filtered_out = stats[stats["n_episodes"] < MIN_EPISODES].copy()

    relevant.to_csv(OUT_RELEVANT, index=False, encoding="utf-8")
    filtered_out.to_csv(OUT_FILTERED, index=False, encoding="utf-8")

    print(f"MIN_EPISODES = {MIN_EPISODES}")
    print(f"Relevant characters: {len(relevant)} -> {OUT_RELEVANT}")
    print(f"Filtered out (<{MIN_EPISODES} episodes): {len(filtered_out)} -> {OUT_FILTERED}")

    print("\nTOP 30 relevant characters:")
    print(relevant.tail(30).to_string(index=False))

    print(f"\nFiltered out characters:")
    print(filtered_out.head(100).to_string(index=False))


if __name__ == "__main__":
    main()
