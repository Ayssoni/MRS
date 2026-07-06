"""
Pipeline orchestrator — run any stage from the project root.

    python main.py preprocess          # raw CSVs -> data/processed/MoviesDF.pickle
    python main.py train               # processed data -> models/model.pickle
    python main.py recommend "kgf"     # search by title / genre / cast / director
    python main.py add                 # interactive prompt to add a new movie
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import preprocess
import train
import recommend
import add_movie


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "preprocess":
        preprocess.run()
    elif command == "train":
        train.run()
    elif command == "recommend":
        query = sys.argv[2] if len(sys.argv) > 2 else input("Search (title/genre/cast/director): ")
        recommend.get_recommendation(query)
    elif command == "add":
        add_movie._interactive()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
