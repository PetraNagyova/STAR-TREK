import re
import pandas as pd

JUNK_PATTERNS = [
    r"\(uncredited\)",
    r"\(credit only\)",
    r"\(archive footage\)",
    r"\(voice\)",
    r"\(as .*?\)",
    r"\(.*?uncredited.*?\)",
    r"\(.*?credit.*?\)",
]

CHARACTER_ALIASES = {
    "Kirk": "James T. Kirk",
    "James Kirk": "James T. Kirk",
    "James T Kirk": "James T. Kirk",
    "James T. Kirk": "James T. Kirk",
    "Jim Kirk": "James T. Kirk",
    "Sam Kirk": "George Samuel Kirk",
    "George Samuel 'Sam' Kirk": "George Samuel Kirk",
}

def strip_credit_junk(s: str) -> str:
    out = s
    for pat in JUNK_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return out

def normalize_character_name(raw: str | None, actor_name: str | None = None) -> str | None:
    if raw is None or pd.isna(raw):
        return None

    s = str(raw).strip()
    s = re.sub(r"[,;:]+$", "", s).strip()
    s = re.sub(
        r"^(capt\.?|captain|lt\.?|lieutenant|lt\.?\s*cmdr\.?|lieutenant\s*commander|cmdr\.?|commander|ens\.?|ensign)\s+",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()

    # dvakrat lebo niekedy bol 'Captain Commander'
    s = re.sub(
        r"^(capt\.?|captain|lt\.?|lieutenant|lt\.?\s*cmdr\.?|lieutenant\s*commander|cmdr\.?|commander|ens\.?|ensign)\s+",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()

    s = re.sub(
        r"^(\s*\(?\s*jg\s*\)?\s*)\s+",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()

    s = re.sub(r"^(mr|dr)\.\s+", "", s, flags=re.IGNORECASE).strip()

    s = re.sub(r"^Doctor\s+(?!The\b)", "", s, flags=re.IGNORECASE).strip()

    if not s or s == r"\N":
        return None

    s = strip_credit_junk(s)
    s = s.replace('"', "").replace("“", "").replace("”", "").strip()

    s = re.sub(r"\s+", " ", s).strip()

    if s.lower() in {"self", "himself", "herself"}:
        return None

    # (ak chceš zachovať všetky, viď funkcia split_characters nižšie)
    s = re.split(r"\s*/\s*|\s+&\s+|,\s*", s)[0].strip()

    s = CHARACTER_ALIASES.get(s, s)

    return s or None

def split_characters(raw: str | None) -> list[str] | None:
    if raw is None or pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s or s == r"\N":
        return None
    s = strip_credit_junk(s)
    s = s.replace('"', "").strip()
    s = re.sub(r"\s+", " ", s).strip()

    parts = re.split(r"\s*/\s*|\s+&\s+|;\s*", s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or None

cast = pd.read_csv("project_tables_csv/cast.csv", dtype=str)

cast["billing_order"] = pd.to_numeric(cast["billing_order"], errors="coerce").astype("Int64")
cast["is_main_cast"] = pd.to_numeric(cast["is_main_cast"], errors="coerce").fillna(0).astype(int)

cast["character_name_raw"] = cast["character_name"]

cast["character_name_norm"] = cast.apply(
    lambda r: normalize_character_name(r["character_name_raw"], r["name"]),
    axis=1,
)

changed = cast[
    cast["character_name_raw"].notna()
    & cast["character_name_norm"].notna()
    & (cast["character_name_raw"].astype(str).str.strip() != cast["character_name_norm"].astype(str).str.strip())
].copy()

summary = (
    changed.groupby(["character_name_raw", "character_name_norm"], as_index=False)
           .size()
           .sort_values("size", ascending=False)
)

print("Počet zmien:", len(summary))
print(summary.head(50).to_string(index=False))  # top 50 zmien

cast = cast.sort_values(
    ["proj_ep_id", "nconst", "billing_order"],
    ascending=[True, True, True],
)

cast_dedup = (
    cast.groupby(["proj_ep_id", "nconst", "character_name_norm"], dropna=False, as_index=False)
        .agg(
            name=("name", "first"),
            billing_order=("billing_order", "min"),
            is_main_cast=("is_main_cast", "max"),
            character_name_raw=("character_name_raw", "first"),
        )
)

cast_dedup.to_csv("project_tables_csv/cast_clean.csv", index=False, encoding="utf-8")
print("cast_clean.csv rows:", len(cast_dedup))