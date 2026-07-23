"""Train a RandomForestClassifier on the prepared Iris training data.

Usage:
    python src/train.py --data-dir data/processed --model-dir outputs/model
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train(data_dir: str, model_dir: str, n_estimators: int = 100, max_depth: int = 5) -> str:
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    clf.fit(X_train, y_train)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(clf, model_path)

    print(f"Model trained and saved to {model_path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    train(args.data_dir, args.model_dir, args.n_estimators, args.max_depth)


if __name__ == "__main__":
    main()
