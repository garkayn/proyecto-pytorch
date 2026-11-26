"""Carga y prepara el dataset del análisis original de hits de Spotify."""

import hashlib
from io import BytesIO
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd


DATASET = "paradisejoy/top-hits-spotify-from-20002019/versions/4"
POPULARITY_THRESHOLD = 8.9
SPLIT_YEARS = {"train": (1998, 2009), "validation": (2010, 2014), "test": (2015, 2019)}
FEATURES = (
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
)
REQUIRED_COLUMNS = {"artist", "song", "genre", "year", "duration_ms", "popularity", *FEATURES}


def find_csv(csv: Path | None) -> Path:
    if csv is not None:
        return csv

    import kagglehub

    dataset_path = Path(kagglehub.dataset_download(DATASET))
    matches = [
        path for path in sorted(dataset_path.rglob("*.csv"))
        if REQUIRED_COLUMNS.issubset(pd.read_csv(path, nrows=0).columns)
    ]
    if len(matches) != 1:
        raise ValueError(f"Se esperaba un solo CSV compatible en {dataset_path}. Encontrados: {len(matches)}.")
    return matches[0]


def song_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_splits(csv: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str | int]]:
    raw = csv.read_bytes()
    songs = pd.read_csv(BytesIO(raw))
    missing_columns = REQUIRED_COLUMNS.difference(songs.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas en el CSV: {', '.join(sorted(missing_columns))}.")

    songs = songs.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    source_rows = len(songs)
    numeric_columns = ["year", "duration_ms", "popularity", *FEATURES]
    songs[numeric_columns] = (
        songs[numeric_columns].apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    genre = songs["genre"].astype("string").fillna("").str.strip()
    for column in ("artist", "song"):
        songs[column] = songs[column].astype("string").fillna("").str.strip()

    songs = songs.loc[
        songs["year"].between(SPLIT_YEARS["train"][0], SPLIT_YEARS["test"][1])
        & songs["year"].mod(1).eq(0)
        & songs["duration_ms"].gt(0)
        & songs["popularity"].ge(POPULARITY_THRESHOLD)
        & genre.ne("") & genre.ne("set()")
        & songs["artist"].ne("") & songs["song"].ne("")
    ].copy()
    eligible_rows = len(songs)
    songs["artist_key"] = songs["artist"].map(song_key)
    songs["song_key"] = songs["song"].map(song_key)
    songs = songs.sort_values("year", kind="stable").drop_duplicates(["artist_key", "song_key"])
    songs["year"] = songs["year"].astype(int)
    songs["duration_s"] = songs["duration_ms"] / 1_000

    splits = {
        name: songs.loc[songs["year"].between(first, last)].reset_index(drop=True)
        for name, (first, last) in SPLIT_YEARS.items()
    }
    for name, frame in splits.items():
        if frame.empty:
            first, last = SPLIT_YEARS[name]
            raise ValueError(f"La partición {name} ({first}–{last}) quedó vacía después de la limpieza.")
    return splits, {
        "file": csv.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_rows": source_rows,
        "eligible_rows": eligible_rows,
        "removed_duplicates": eligible_rows - len(songs),
    }
