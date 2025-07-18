import argparse
import os
import src.mcc_computer as mcc
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--root_dir",
        help="Absolute path to the rood directory of source files",
        required=True,
    )

    args = parser.parse_args()

    return (args.root_dir)

def main() -> None:
    (root_dir) = get_args()

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                mcc.compute_mcc(Path(os.path.join(root, file)))

if __name__ == "__main__":
    main()