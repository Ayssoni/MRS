# Movie Recommender — ML Pipeline

## Structure
```
movie_recommender/
├── main.py                 # CLI entry point for every stage
├── data/
│   ├── raw/                 # put tmdb_5000_movies.csv + tmdb_5000_credits.csv here
│   └── processed/
│       └── MoviesDF.pickle  # output of preprocess.py (your existing data is already here)
├── models/
│   └── model.pickle         # output of train.py: vectorizer + vectors + similarity matrix
└── src/
    ├── config.py             # all file paths / settings, in one place
    ├── text_utils.py          # shared tag-building logic (used by preprocess.py AND add_movie.py)
    ├── preprocess.py           # raw CSVs -> data/processed/MoviesDF.pickle
    ├── train.py                # data/processed -> models/model.pickle
    ├── recommend.py            # loads processed data + model, serves search()
    └── add_movie.py             # adds a new movie, then retrains
```

## Usage
```bash
# 1. (only if you want to rebuild from scratch) drop the two TMDB CSVs into data/raw/, then:
python main.py preprocess
python main.py train

# 2. search by title, genre, cast, or director — all through one command
python main.py recommend "kgf"
python main.py recommend "christopher nolan"
python main.py recommend "science fiction"

# 3. add a new movie (updates data + retrains the model automatically)
python main.py add
```

Your existing `MoviesDF.pickle` is already in `data/processed/`, and the model
has already been trained once (`models/model.pickle` exists) — so `recommend`
works immediately without needing the raw CSVs at all.

## Why it's split this way
- **preprocess.py** and **add_movie.py** share `text_utils.py`, so a movie added
  later goes through *identical* tag-building steps as the original CSV data —
  this is what was silently broken in the old single-file script.
- **train.py** is a separate stage so `add_movie.py` can call it to fully
  **refit** the vectorizer on the updated dataset (not just `.transform()` with
  a stale vocabulary), which is what lets a new movie's own genre/cast/keywords
  actually influence its similarity to other movies.
- **recommend.py** only loads already-trained artifacts — it never re-fits
  anything, so lookups are fast regardless of dataset size.
