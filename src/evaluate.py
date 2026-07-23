"""Evaluate the trained model on the held-out test set and enforce a quality gate.

Exits with a non-zero status if accuracy is below --min-accuracy, so this can be used
as a hard gate in a CI/CD pipeline before a model is registered or deployed.

Usage:
    python src/evaluate.py --model-dir outputs/model --data-dir data/processed
"""
import argparse
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def evaluate(model_dir: str, data_dir: str, min_accuracy: float = 0.85) -> dict:
    model_path = os.path.join(model_dir, "model.pkl")
    clf = joblib.load(model_path)

    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    metrics = {"accuracy": accuracy, "f1_macro": f1, "min_accuracy_threshold": min_accuracy}

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics: {metrics}")

    if accuracy < min_accuracy:
        print(
            f"FAILED quality gate: accuracy {accuracy:.4f} < threshold {min_accuracy}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"PASSED quality gate: accuracy {accuracy:.4f} >= threshold {min_accuracy}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    args = parser.parse_args()

    evaluate(args.model_dir, args.data_dir, args.min_accuracy)


if __name__ == "__main__":
    main()
