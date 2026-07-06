"""
Shared text-processing helpers.

Both preprocess.py (bulk CSV -> tags) and add_movie.py (single new movie ->
tags) import from here. Keeping this logic in ONE place is what guarantees
a newly-added movie's tags are built exactly the same way as every movie
that came from the original CSVs — that mismatch was the root cause of
poor recommendations for manually-added movies in the old script.
"""

import ast
import re

from nltk.stem.porter import PorterStemmer

_ps = PorterStemmer()


def parse_name_list(json_like_string: str):
    """Turn a TMDB-style JSON string like [{"id":1,"name":"Action"}, ...]
    into a plain list of names: ["Action", ...]."""
    names = []
    for item in ast.literal_eval(json_like_string):
        names.append(item["name"])
    return names


def parse_top_cast(json_like_string: str, limit: int = 3):
    """Same as parse_name_list but only keeps the first `limit` entries
    (main cast members)."""
    names = []
    for i, item in enumerate(ast.literal_eval(json_like_string)):
        if i >= limit:
            break
        names.append(item["name"])
    return names


def parse_director(json_like_string: str):
    """Pull the Director's name out of a TMDB-style crew JSON string."""
    for item in ast.literal_eval(json_like_string):
        if item.get("job") == "Director":
            return [item["name"]]
    return []


def strip_internal_spaces(words):
    """['Science Fiction'] -> ['ScienceFiction'] so multi-word terms become
    a single token (keeps 'Science Fiction' distinct from separate
    'Science' and 'Fiction' tokens)."""
    return [w.replace(" ", "") for w in words]


def build_tag_list(overview: str, genres, keywords, cast, crew):
    """Combine all fields into one flat list of words, exactly matching
    the original pipeline's field order: genres, cast, crew, keywords,
    overview."""
    overview_tokens = strip_internal_spaces(overview.split())
    genres = strip_internal_spaces(genres)
    keywords = strip_internal_spaces(keywords)
    cast = strip_internal_spaces(cast)
    crew = strip_internal_spaces(crew)
    return genres + cast + crew + keywords + overview_tokens


def finalize_tags(tag_list) -> str:
    """[List of words] -> single lowercase, stemmed, space-joined string,
    ready for CountVectorizer."""
    joined = " ".join(tag_list).lower()
    return " ".join(_ps.stem(word) for word in joined.split())


def normalize_query_tokens(query: str):
    """Turn free-text user input (a genre, cast name, or director name)
    into candidate tag tokens for lookup. Handles case, extra whitespace,
    and word order (so 'Sam Worthington' and 'Worthington Sam' both
    resolve to the same tag token)."""
    words = re.findall(r"[a-z0-9]+", query.strip().lower())
    variants = {"".join(words), "".join(reversed(words))}
    return [_ps.stem(v) for v in variants if v]
