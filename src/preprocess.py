import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Assumed ODIR multi-label columns; adjust to match your CSV header.
CLASS_NAMES: List[str] = ["N", "D", "G", "C", "A", "H", "M", "O"]


def load_odir_dataframe(csv_path: str) -> pd.DataFrame:
    """Load the ODIR CSV file and validate label columns."""
    df = pd.read_csv(csv_path)
    missing = [c for c in CLASS_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns in CSV: {missing}")
    return df


def build_image_paths(
    df: pd.DataFrame,
    image_root: str,
    image_column: str = "Image",
) -> List[str]:
    """Build absolute image paths from a dataframe."""
    if image_column not in df.columns:
        raise ValueError(f"CSV must contain image column '{image_column}'")
    return [os.path.join(image_root, str(fname)) for fname in df[image_column]]


def build_label_matrix(df: pd.DataFrame, class_names: List[str] = None) -> np.ndarray:
    """Extract multi-label targets as a float32 NumPy array."""
    if class_names is None:
        class_names = CLASS_NAMES
    labels = df[class_names].astype("float32").values
    return labels


def get_paths_and_labels(
    csv_path: str,
    image_root: str,
    image_column: str = "Image",
    class_names: List[str] = None,
) -> Tuple[List[str], np.ndarray]:
    """Convenience helper to get image paths and label matrix."""
    if class_names is None:
        class_names = CLASS_NAMES
    df = load_odir_dataframe(csv_path)
    image_paths = build_image_paths(df, image_root, image_column=image_column)
    labels = build_label_matrix(df, class_names=class_names)
    return image_paths, labels


def split_dataframe(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into train/val/test sets.

    This performs a simple random split; multi-label targets are preserved as
    columns and are not altered.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    df_train, df_temp = train_test_split(
        df, test_size=(1.0 - train_frac), random_state=random_state, shuffle=True
    )
    relative_val_frac = val_frac / (val_frac + test_frac)
    df_val, df_test = train_test_split(
        df_temp, test_size=(1.0 - relative_val_frac), random_state=random_state, shuffle=True
    )
    return df_train, df_val, df_test


def save_splits_to_csv(
    df: pd.DataFrame,
    output_dir: str,
    train_name: str = "train.csv",
    val_name: str = "val.csv",
    test_name: str = "test.csv",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[str, str, str]:
    """
    Split a dataframe into train/val/test and save them as CSV files.

    Returns the three created file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_train, df_val, df_test = split_dataframe(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
    )

    train_path = os.path.join(output_dir, train_name)
    val_path = os.path.join(output_dir, val_name)
    test_path = os.path.join(output_dir, test_name)

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def main():
    """
    Simple CLI for splitting a single ODIR CSV into train/val/test CSVs.

    Example:
        python -m src.preprocess \\
            --input_csv data/processed/odir_all.csv \\
            --output_dir data/processed \\
            --train_frac 0.7 --val_frac 0.15 --test_frac 0.15
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Split ODIR CSV into train/val/test preserving multi-label targets."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = load_odir_dataframe(args.input_csv)
    train_path, val_path, test_path = save_splits_to_csv(
        df,
        output_dir=args.output_dir,
        train_name="train.csv",
        val_name="val.csv",
        test_name="test.csv",
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.random_state,
    )

    print("Saved splits:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")


__all__ = [
    "CLASS_NAMES",
    "load_odir_dataframe",
    "build_image_paths",
    "build_label_matrix",
    "get_paths_and_labels",
    "split_dataframe",
    "save_splits_to_csv",
]


if __name__ == "__main__":
    main()