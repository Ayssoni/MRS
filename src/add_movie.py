"""
Stage 4 of the pipeline: ADD A NEW MOVIE

Adds a movie that isn't in the dataset yet, using the EXACT SAME tag-building
steps as preprocess.py (via text_utils), then retrains the model so the new
movie's words enter the vocabulary and its similarity to every other movie
is computed correctly.

This fixes the old bug where a newly added movie's tags were passed through
`cv.transform()` on the *old* vocabulary and never refit — meaning its own
unique keywords/cast/director never influenced any similarity score.

Run directly for an interactive prompt:
    python src/add_movie.py

Or call add_movie(...) directly from other code.
"""

import pandas as pd

import config
from text_utils import build_tag_list, finalize_tags
from train import load_processed, run as retrain


def add_movie(movie_id: int, title: str, overview: str, genres, keywords, cast, crew):
    """
    genres, keywords, cast: lists of plain strings, e.g. ["Action", "Adventure"]
    crew: list containing the director's name, e.g. ["Christopher Nolan"]
    """
    processed = load_processed()

    if (processed["title_norm"] == title.strip().lower()).any():
        print(f"'{title}' already exists in the dataset — skipping add.")
        return processed

    tag_list = build_tag_list(overview, genres, keywords, cast, crew)
    tags = finalize_tags(tag_list)

    new_row = pd.DataFrame([{
        "movie_id": movie_id,
        "title": title,
        "tags": tags,
        "title_norm": title.strip().lower(),
        "tag_tokens": set(tags.split()),
    }])

    processed = pd.concat([processed, new_row], ignore_index=True)
    processed.to_pickle(config.PROCESSED_PICKLE)
    print(f"Added '{title}' to {config.PROCESSED_PICKLE}. Retraining model...")

    retrain()  # refits vectorizer + similarity on the full, updated dataset
    return processed


def _interactive():
    title = input("Movie title: ").strip()
    movie_id = int(input("Movie ID: ").strip())
    overview = input("Overview (a sentence or two): ").strip()
    genres = [g.strip() for g in input("Genres (comma separated): ").split(",") if g.strip()]
    keywords = [k.strip() for k in input("Keywords (comma separated): ").split(",") if k.strip()]
    cast = [c.strip() for c in input("Cast (comma separated, main 3): ").split(",") if c.strip()]
    director = [input("Director: ").strip()]

    add_movie(movie_id, title, overview, genres, keywords, cast, director)


if __name__ == "__main__":
    _interactive()
