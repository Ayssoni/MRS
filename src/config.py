"""
Central configuration for the movie recommender pipeline.
All other modules import paths from here so nothing is hardcoded
in multiple places.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root (MRS/)

# --- raw data (your CSVs live in MRS/Dataset/) ---
RAW_DIR = BASE_DIR / "data" / "raw"
MOVIES_CSV = RAW_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = RAW_DIR / "tmdb_5000_credits.csv"

# --- processed data (output of preprocess.py, input to train.py/recommend.py) ---
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_PICKLE = PROCESSED_DIR / "MoviesDF.pickle"

# --- trained model artifacts (output of train.py, input to recommend.py) ---
MODELS_DIR = BASE_DIR / "models"
MODEL_PICKLE = MODELS_DIR / "model.pickle"  # holds vectorizer + vectors + similarity

# --- vectorizer settings (kept in one place so preprocess/train/add all agree) ---
MAX_FEATURES = 5000
STOP_WORDS = "english"

# --- search settings ---
TOP_N_DEFAULT = 5
TITLE_FUZZY_CUTOFF = 0.75
TAG_FUZZY_CUTOFF = 0.80