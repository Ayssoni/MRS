"""
Edit MOVIE_NAME below and run this file — no command-line arguments needed.

    python search.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recommend import get_recommendation

# ---------------------------------------------------------
# PUT YOUR MOVIE NAME, GENRE, CAST MEMBER, OR DIRECTOR HERE
MOVIE_NAME = "lovelyrunner"
# ---------------------------------------------------------

get_recommendation(MOVIE_NAME)
