"""Load the Iris dataset and split it into train/test sets.

Usage:
    python src/data_prep.py --output-dir data/processed
"""
import argparse
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def prepare_data(output_dir: str, test_size: float = 0.2, random_state: int = 42) -> None:
    os.makedirs(output_dir, exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "label"})

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Wrote {len(train_df)} training rows to {train_path}")
    print(f"Wrote {len(test_df)} test rows to {test_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    prepare_data(args.output_dir, args.test_size, args.random_state)


if __name__ == "__main__":
    main()
