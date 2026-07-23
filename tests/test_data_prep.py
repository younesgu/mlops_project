import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_prep import prepare_data  # noqa: E402


def test_prepare_data_creates_expected_splits(tmp_path):
    output_dir = str(tmp_path)
    prepare_data(output_dir, test_size=0.2, random_state=42)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    with open(train_path) as f:
        train_lines = f.readlines()
    with open(test_path) as f:
        test_lines = f.readlines()

    # 150 rows total (+1 header each), 80/20 split
    assert len(train_lines) - 1 == 120
    assert len(test_lines) - 1 == 30


def test_prepare_data_has_expected_columns(tmp_path):
    output_dir = str(tmp_path)
    prepare_data(output_dir)

    with open(os.path.join(output_dir, "train.csv")) as f:
        header = f.readline().strip()

    assert "label" in header
