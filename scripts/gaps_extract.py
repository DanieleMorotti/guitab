"""Extract the GAPS annotation zip with case-safe file names (see guitab/gaps.py).

    curl -L -o data/gaps/gaps_v1_no_audio.zip "https://zenodo.org/records/13962272/files/gaps_v1_no_audio.zip?download=1"
    (cd data/gaps && unzip -o gaps_v1_no_audio.zip "gaps_v1/*.csv")   # only the csv files are safe to unzip
    python scripts/gaps_extract.py                                   # -> data/gaps/safe/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.gaps import SAFE, extract  # noqa: E402

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print(f"extracted {extract()} files to {SAFE}")
