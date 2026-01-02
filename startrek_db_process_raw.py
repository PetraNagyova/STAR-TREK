import sqlite3
from pathlib import Path

import pandas as pd


EXCLUDE_SERIES = {
    "Star Trek Continues",
    "Star Trek: very Short Treks",
}


def export_startrek_minimal(
    db_path: str | Path,
    out_dir: str | Path = "startrek_processed",
) -> None:
    db_path = Path(db_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    # 1) Seriály (bez vybraných dvoch)
    series_df = pd.read_sql_query(
        """
        SELECT
            series_id,
            title AS series_title,
            begin AS series_begin,
            end   AS series_end
        FROM series
        WHERE title NOT IN (?, ?)
        ORDER BY series_title
        """,
        conn,
        params=tuple(EXCLUDE_SERIES),
    )

    # 2) Epizódy (join na series + rovnaký filter)
    episodes_df = pd.read_sql_query(
        """
        SELECT
            e.episode_id,
            e.series_id,
            s.title AS series_title,
            e.season,
            e.episode_number,
            e.title AS episode_title,
            COALESCE(e.airdate, e.remastered_airdate) AS airdate,
            e.airdate AS original_airdate,
            e.remastered_airdate,
            e.production_code,
            e.stardate,
            e.vignette
        FROM episode e
        JOIN series s ON s.series_id = e.series_id
        WHERE s.title NOT IN (?, ?)
        ORDER BY s.title, e.season, e.episode_number, e.episode_id
        """,
        conn,
        params=tuple(EXCLUDE_SERIES),
    )

    conn.close()

    series_df.to_csv(out_dir / "series.csv", index=False, encoding="utf-8")
    episodes_df.to_csv(out_dir / "episodes.csv", index=False, encoding="utf-8")

    print(f"OK: {len(series_df)} seriálov -> {out_dir/'series.csv'}")
    print(f"OK: {len(episodes_df)} epizód  -> {out_dir/'episodes.csv'}")


if __name__ == "__main__":
    export_startrek_minimal("startrek_db/startrek.db", out_dir="startrek_processed")
