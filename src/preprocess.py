"""
Stage 1 of the pipeline: RAW DATA -> PROCESSED DATA

Reads the two TMDB CSVs from data/raw/, cleans them, builds the combined
'tags' column (genres + cast + director + keywords + overview, stemmed),
and writes a single clean MoviesDF.pickle to data/processed/.

Run directly:
    python src/preprocess.py
"""

import pandas as pd
from pathlib import Path

import config
from text_utils import parse_name_list, parse_top_cast, parse_director, build_tag_list, finalize_tags


def load_raw():
    movies_csv = Path(config.MOVIES_CSV)
    credits_csv = Path(config.CREDITS_CSV)
    if not movies_csv.exists() or not credits_csv.exists():
        raise FileNotFoundError(
            f"Expected raw CSVs at:\n  {movies_csv}\n  {credits_csv}\n"
            "Place the TMDB 'movies' and 'credits' CSVs in data/raw/ before running preprocess.py."
        )
    movies = pd.read_csv(movies_csv)
    credits = pd.read_csv(credits_csv)
    return movies, credits


def preprocess(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(movies, credits, on="title")
    df = merged[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]].copy()

    # drop rows missing an overview, drop exact duplicates
    df = df.dropna(subset=["overview"]).copy()
    df = df.drop_duplicates().copy()

    df["genres"] = df["genres"].apply(parse_name_list)
    df["keywords"] = df["keywords"].apply(parse_name_list)
    df["cast"] = df["cast"].apply(lambda s: parse_top_cast(s, limit=3))
    df["crew"] = df["crew"].apply(parse_director)

    df["tags"] = df.apply(
        lambda row: build_tag_list(row["overview"], row["genres"], row["keywords"], row["cast"], row["crew"]),
        axis=1,
    )
    df["tags"] = df["tags"].apply(finalize_tags)

    processed = df[["movie_id", "title", "tags"]].reset_index(drop=True)

    # Precompute lookup helpers once here so every downstream stage
    # (train.py, recommend.py, add_movie.py) can just load and use them.
    processed["title_norm"] = processed["title"].astype(str).str.strip().str.lower()
    processed["tag_tokens"] = processed["tags"].str.split().apply(set)

    return processed


def run():
    Path(config.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    movies, credits = load_raw()
    processed = preprocess(movies, credits)
    processed.to_pickle(Path(config.PROCESSED_PICKLE))
    print(f"Saved {len(processed)} processed movies to {config.PROCESSED_PICKLE}")
    return processed


if __name__ == "__main__":
    run()
