"""
Stage 3 of the pipeline: RECOMMEND

Loads data/processed/MoviesDF.pickle + models/model.pickle and serves
recommendations from a single free-text query that can be a movie title,
a genre, a cast member, or a director.

Run directly:
    python src/recommend.py "kgf"
    python src/recommend.py "christopher nolan"
"""

import sys
import difflib
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import config
from text_utils import normalize_query_tokens
from train import load_processed


def load_model() -> dict:
    model_pickle = Path(config.MODEL_PICKLE)
    if not model_pickle.exists():
        raise FileNotFoundError(
            f"{model_pickle} not found. Run train.py first."
        )
    return pd.read_pickle(model_pickle)


class MovieRecommender:
    def __init__(self):
        self.df = load_processed()

        # backward-compatible with older MoviesDF.pickle files that don't
        # yet have these precomputed helper columns
        if "title_norm" not in self.df.columns:
            self.df["title_norm"] = self.df["title"].astype(str).str.strip().str.lower()
        if "tag_tokens" not in self.df.columns:
            self.df["tag_tokens"] = self.df["tags"].str.split().apply(set)

        model = load_model()
        self.vectors = model["vectors"]  # sparse matrix, shape (n_movies, n_vocab)
        self.vocab = model["vocab"]
        self.all_tokens = set().union(*self.df["tag_tokens"])

    def _recommend_from_index(self, idx, top_n):
        distances = cosine_similarity(self.vectors[idx], self.vectors).flatten()
        ranked = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:top_n + 1]
        return [self.df.iloc[i]["title"] for i, _ in ranked]

    def _movies_for_tag(self, token, top_n):
        matches = self.df[self.df["tag_tokens"].apply(lambda s: token in s)]
        if matches.empty:
            return []
        if token in self.vocab:
            col = self.vocab.index(token)
            scores = self.vectors[matches.index, col].toarray().flatten()
            order = scores.argsort()[::-1][:top_n]
            return matches.iloc[order]["title"].tolist()
        return matches["title"].head(top_n).tolist()

    def search(self, query: str, top_n: int = config.TOP_N_DEFAULT):
        """Return (mode, matched_value, [recommended titles])."""
        if not query or not query.strip():
            return "not found", query, []

        qn = query.strip().lower()

        # 1. exact title match (case/space-insensitive)
        exact = self.df[self.df["title_norm"] == qn]
        if not exact.empty:
            idx = exact.index[0]
            return "title", self.df.loc[idx, "title"], self._recommend_from_index(idx, top_n)

        # 2. exact tag token match (genre / cast / director / keyword)
        tokens = normalize_query_tokens(query)
        for token in tokens:
            if token in self.all_tokens:
                return "genre/cast/director", token, self._movies_for_tag(token, top_n)

        # 3. fuzzy title match (typo tolerance)
        close_titles = difflib.get_close_matches(
            qn, self.df["title_norm"].tolist(), n=1, cutoff=config.TITLE_FUZZY_CUTOFF
        )
        if close_titles:
            idx = self.df[self.df["title_norm"] == close_titles[0]].index[0]
            return "title (fuzzy)", self.df.loc[idx, "title"], self._recommend_from_index(idx, top_n)

        # 4. fuzzy tag token
        for token in tokens:
            close_tokens = difflib.get_close_matches(
                token, self.all_tokens, n=1, cutoff=config.TAG_FUZZY_CUTOFF
            )
            if close_tokens:
                return (
                    "genre/cast/director (fuzzy)",
                    close_tokens[0],
                    self._movies_for_tag(close_tokens[0], top_n),
                )

        return "not found", query, []


_engine_singleton = None


def get_engine() -> MovieRecommender:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = MovieRecommender()
    return _engine_singleton


def get_recommendation(query: str, top_n: int = config.TOP_N_DEFAULT):
    engine = get_engine()
    mode, matched, results = engine.search(query, top_n)

    if mode == "not found":
        msg = f"Error: '{query}' not found as a title, genre, cast member, or director."
        print(msg)
        return msg

    label = {
        "title": f"movie '{matched}'",
        "title (fuzzy)": f"movie '{matched}' (closest match to your input)",
        "genre/cast/director": f"'{matched}'",
        "genre/cast/director (fuzzy)": f"'{matched}' (closest match to your input)",
    }[mode]

    print(f"\nTop {len(results)} movies for {label}:")
    for title in results:
        print(f"- {title}")
    return results


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "kgf"
    get_recommendation(query)
