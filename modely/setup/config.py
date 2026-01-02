from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

TABLE_DIR = ROOT_DIR / "project_tables_csv"

DB_PATH = ROOT_DIR / "project.db"

COL_PROJ_EP_ID = "proj_ep_id"
COL_SERIES = "series"
COL_SEASON = "season"
COL_ORDER_EP = "order_ep"
COL_YEAR = "year"
COL_TARGET = "popularity"
