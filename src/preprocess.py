import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CLASS_NAMES, DEFAULT_IMAGE_COLUMN, DEFAULT_RANDOM_STATE
from .utils import resolve_image_column


def load_odir_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [column for column in CLASS_NAMES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns in CSV: {missing}")
    return df


def build_image_paths(
    df: pd.DataFrame,
    image_root: str,
    image_column: str = DEFAULT_IMAGE_COLUMN,
) -> List[str]:
    resolved_image_column = resolve_image_column(df.columns, preferred=image_column)
    filenames = [os.path.basename(str(value)) for value in df[resolved_image_column]]
    return [os.path.join(image_root, filename) for filename in filenames]


def build_label_matrix(df: pd.DataFrame, class_names: List[str] = None) -> np.ndarray:
    if class_names is None:
        class_names = list(CLASS_NAMES)
    return df[class_names].astype("float32").values


def get_paths_and_labels(
    csv_path: str,
    image_root: str,
    image_column: str = DEFAULT_IMAGE_COLUMN,
    class_names: List[str] = None,
) -> Tuple[List[str], np.ndarray]:
    if class_names is None:
        class_names = list(CLASS_NAMES)

    dataframe = load_odir_dataframe(csv_path)
    image_paths = build_image_paths(dataframe, image_root=image_root, image_column=image_column)
    labels = build_label_matrix(dataframe, class_names=class_names)
    return image_paths, labels


def split_dataframe(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    df_train, df_temp = train_test_split(
        df, test_size=(1.0 - train_frac), random_state=random_state, shuffle=True
    )
    relative_val_frac = val_frac / (val_frac + test_frac)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - relative_val_frac),
        random_state=random_state,
        shuffle=True,
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
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[str, str, str]:
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Split the ODIR dataframe into train/val/test CSV files."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    dataframe = load_odir_dataframe(args.input_csv)
    train_path, val_path, test_path = save_splits_to_csv(
        dataframe,
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


if __name__ == "__main__":
    main()
