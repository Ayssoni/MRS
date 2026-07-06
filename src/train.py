"""
Stage 2 of the pipeline: PROCESSED DATA -> TRAINED MODEL

Loads MoviesDF.pickle from data/processed/, fits the CountVectorizer,
computes the cosine-similarity matrix, and saves both to models/model.pickle.

This is also called by add_movie.py after a new movie is appended, so the
vectorizer's vocabulary and the similarity matrix are always refit on the
FULL, current dataset (this is what fixes the old bug where a newly added
movie's unique words were silently dropped because only `.transform()`,
never `.fit_transform()`, was used after adding a movie).

Run directly:
    python src/train.py
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import config


def load_processed() -> pd.DataFrame:
    processed_pickle = Path(config.PROCESSED_PICKLE)
    if not processed_pickle.exists():
        raise FileNotFoundError(
            f"{processed_pickle} not found. Run preprocess.py first."
        )
    return pd.read_pickle(processed_pickle)


def train(processed: pd.DataFrame) -> dict:
    cv = CountVectorizer(max_features=config.MAX_FEATURES, stop_words=config.STOP_WORDS)
    # keep the sparse matrix as-is (no .toarray()) — a dense 4807x4807
    # similarity matrix would be ~185MB; computing similarity rows on
    # demand in recommend.py keeps the saved model small and just as fast
    # for lookups.
    vectors = cv.fit_transform(processed["tags"])
    vocab = list(cv.get_feature_names_out())

    return {
        "vectorizer": cv,
        "vectors": vectors,
        "vocab": vocab,
    }


def run():
    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    processed = load_processed()
    model = train(processed)
    with open(Path(config.MODEL_PICKLE), "wb") as f:
        pickle.dump(model, f)
    print(f"Trained on {len(processed)} movies. Vocabulary size: {len(model['vocab'])}")
    print(f"Saved model artifacts to {config.MODEL_PICKLE}")
    return model


if __name__ == "__main__":
    run()
